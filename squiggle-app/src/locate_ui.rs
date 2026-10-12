//! Pick and Locate windows, and alignment across phenomenologies.
//!
//! Picking runs STA/LTA + AIC on the selected tracks' plotted series and lists
//! one onset per track for review. Locating grid-searches every included pick
//! with a sensor position and a phase speed. Both run off the UI thread and
//! commit undoable commands; manual picks and locked offsets are never
//! replaced automatically.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::dsp::{PickSettings, pick};
use squiggle_core::locate::{Fit, LocateSettings, Observation, locate, surface_distance, travel_time};
use squiggle_core::sensor::Provenance;
use squiggle_core::{
    Alignment, Change, Id, IqComponent, Location, Origin, PhenomenologyId, Pick, Project, SampleKind, Timestamp,
};

use crate::process_ui::config_hash;
use crate::strips::colormap;
use crate::tracks::{Track, TrackView};
use crate::workspace::Workspace;

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
const GOOD_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 210, 110);
const PICK_STAGE: &str = "pick.sta-lta-aic";
const LOCATE_STAGE: &str = "locate.grid";
const MAP_SIZE: f32 = 320.0;

/// Starting STA/LTA windows per built-in phenomenology; others scale with rate.
pub fn default_pick_settings(phenomenology: &str, rate: f64) -> PickSettings {
    let (sta_s, lta_s) = match phenomenology {
        "seismic" => (0.3, 5.0),
        "infrasound" => (0.2, 3.0),
        "acoustic" => (0.01, 0.5),
        "emp" | "rf" => (1e-5, 1e-3),
        _ => (20.0 / rate, 400.0 / rate),
    };
    PickSettings { sta_s, lta_s, ..PickSettings::default() }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Choice {
    /// Earliest trigger: the first arrival.
    #[default]
    First,
    Strongest,
}

/// One trace to pick, copied out on the UI thread.
pub struct PickInput {
    pub values: Vec<f32>,
    pub rate: f64,
    /// Recorded time of `values[0]`.
    pub start: Timestamp,
    pub settings: PickSettings,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Picked {
    /// On the trace's own clock.
    pub time: Timestamp,
    pub ratio: f64,
    pub uncertainty_s: f64,
}

pub fn run_pick(input: &PickInput, choice: Choice) -> Result<Option<Picked>, String> {
    let onsets = pick(&input.values, input.rate, &input.settings)?;
    let chosen = match choice {
        Choice::First => onsets.first(),
        Choice::Strongest => onsets.iter().max_by(|a, b| a.ratio.total_cmp(&b.ratio)),
    };
    Ok(chosen.map(|o| Picked {
        time: input.start.add_seconds(o.index as f64 / input.rate),
        ratio: o.ratio,
        uncertainty_s: o.uncertainty_s,
    }))
}

/// The automatic pick for `trace` at `picked` in `event`, as a change.
pub fn pick_change(event: Option<Id>, trace: Id, phase: &str, picked: &Picked, stage: &str, config_hash: &str) -> Change {
    let pick = Pick {
        id: Id::generate(),
        trace,
        time: picked.time,
        phase: phase.to_string(),
        uncertainty_s: picked.uncertainty_s,
        provenance: Provenance::Auto { stage: stage.to_string(), config_hash: config_hash.to_string() },
        event,
    };
    Change::SetPick { pick, revert: false }
}

/// A pick as a locator input, times in seconds after `reference`.
pub fn observation(project: &Project, pick: &Pick, reference: Timestamp) -> Result<Observation, &'static str> {
    let trace = project.trace(pick.trace).ok_or("no trace")?;
    let sensor = project.sensor(trace.sensor).ok_or("no sensor")?;
    let location = sensor.location.ok_or("sensor has no location")?;
    let phenomenology = project.trace_phenomenology(trace).ok_or("unassigned")?;
    let speed_m_s = phenomenology.propagation.speed(&pick.phase).ok_or("no speed for this phase")?;
    let time = project.pick_time(pick).ok_or("no trace")?.seconds_since(reference);
    Ok(Observation { location, speed_m_s, time, sigma_s: pick.uncertainty_s })
}

/// The traces alignment edits may touch: `event`'s members, or every trace.
fn scope(project: &Project, event: Option<Id>) -> Vec<&squiggle_core::Trace> {
    match event.and_then(|e| project.event(e)) {
        Some(e) => e.members.iter().filter_map(|m| project.trace(m.trace)).collect(),
        None => project.traces().iter().collect(),
    }
}

/// Shifts that place each trace's first-phase arrival from `origin` at the
/// origin time, in `event`. Locked traces are left out and counted.
pub fn moveout_changes(project: &Project, event: Option<Id>, origin: &Origin) -> (Vec<Change>, usize) {
    let mut locked = 0;
    let mut changes = Vec::new();
    for trace in scope(project, event) {
        let location = project.sensor(trace.sensor).and_then(|s| s.location);
        let speed = project.trace_phenomenology(trace).and_then(|p| p.propagation.speed(""));
        let (Some(location), Some(speed)) = (location, speed) else {
            continue;
        };
        if project.is_locked(event, trace.id) {
            locked += 1;
            continue;
        }
        let offset_s = -travel_time(origin.source(), &location, speed);
        let provenance = Provenance::Auto { stage: "align.moveout".to_string(), config_hash: origin_hash(origin) };
        let alignment = Alignment { offset_s, provenance, locked: false };
        changes.push(Change::SetAlignment { trace: trace.id, alignment: Some(alignment), revert: false, event });
    }
    (changes, locked)
}

/// Shifts that put each trace's pick on the reference trace's pick of the
/// same phase: the reference's earliest phase when the trace has it, else the
/// trace's earliest (onto the reference's earliest if the reference lacks that
/// phase). Picks, offsets and members are those of `event`. Locked traces are
/// left out and counted.
pub fn pick_alignment_changes(project: &Project, event: Option<Id>, reference: Id, traces: &[Id]) -> (Vec<Change>, usize) {
    // Where a pick is drawn: corrected time plus its trace's offset.
    let drawn = |pick: &Pick| Some(project.pick_time(pick)?.add_seconds(project.offset(event, pick.trace)));
    let in_scope: HashSet<Id> = scope(project, event).iter().map(|t| t.id).collect();
    let earliest = |trace: Id| {
        project
            .event_picks(event, trace)
            .filter_map(|p| Some((drawn(p)?, p)))
            .min_by(|a, b| a.0.seconds_since(b.0).total_cmp(&0.0))
    };
    let of_phase = |trace: Id, phase: &str| project.event_picks(event, trace).find(|p| p.phase == phase).and_then(drawn);
    let Some((anchor_at, anchor)) = earliest(reference) else {
        return (Vec::new(), 0);
    };
    let mut locked = 0;
    let mut changes = Vec::new();
    for &id in traces.iter().filter(|&&id| id != reference && in_scope.contains(&id)) {
        if project.trace(id).is_none() {
            continue;
        }
        let pair = match of_phase(id, &anchor.phase) {
            Some(at) => Some((at, anchor_at)),
            None => earliest(id).map(|(at, p)| (at, of_phase(reference, &p.phase).unwrap_or(anchor_at))),
        };
        let Some((at, target)) = pair else {
            continue;
        };
        if project.is_locked(event, id) {
            locked += 1;
            continue;
        }
        let offset_s = project.offset(event, id) + target.seconds_since(at);
        let provenance = Provenance::Auto { stage: "align.picks".to_string(), config_hash: config_hash(&reference.to_string()) };
        let alignment = Alignment { offset_s, provenance, locked: false };
        changes.push(Change::SetAlignment { trace: id, alignment: Some(alignment), revert: false, event });
    }
    (changes, locked)
}

fn origin_hash(origin: &Origin) -> String {
    config_hash(&format!("{} {} {} {}", origin.latitude, origin.longitude, origin.depth_m, origin.time))
}

struct PickRow {
    trace: Id,
    code: String,
    phase: String,
    result: PickOutcome,
    /// The pick this would replace: its time and whether it is manual.
    existing: Option<(Timestamp, bool)>,
    apply: bool,
}

type PickOutcome = Result<Option<Picked>, String>;

struct PickJob {
    rows: Vec<(Id, String, String)>,
    config_hash: String,
    event: Option<Id>,
}

struct LocateJob {
    /// Observation order.
    picks: Vec<Id>,
    reference: Timestamp,
    config_hash: String,
    event: Option<Id>,
}

struct FitView {
    fit: Fit,
    reference: Timestamp,
    /// Observation order.
    picks: Vec<Id>,
    config_hash: String,
    event: Option<Id>,
}

#[derive(Resource)]
pub struct LocateWindows {
    pub pick_open: bool,
    pub locate_open: bool,
    settings: HashMap<PhenomenologyId, PickSettings>,
    choice: Choice,
    pick_task: Option<(PickJob, Task<Vec<PickOutcome>>)>,
    pick_rows: Option<(Vec<PickRow>, String, Option<Id>)>,
    max_depth_km: f64,
    /// Zero: from the sensor spread.
    extent_km: f64,
    excluded: HashSet<Id>,
    locate_task: Option<(LocateJob, Task<Option<Fit>>)>,
    fit: Option<FitView>,
}

impl Default for LocateWindows {
    fn default() -> Self {
        Self {
            pick_open: false,
            locate_open: false,
            settings: HashMap::new(),
            choice: Choice::default(),
            pick_task: None,
            pick_rows: None,
            max_depth_km: 20.0,
            extent_km: 0.0,
            excluded: HashSet::new(),
            locate_task: None,
            fit: None,
        }
    }
}

pub fn locate_menu(ui: &mut egui::Ui, windows: &mut LocateWindows) {
    if ui.button("Pick…").on_hover_text("Automatic onsets on the selected tracks (STA/LTA + AIC)").clicked() {
        windows.pick_open = true;
        ui.close();
    }
    if ui.button("Locate…").on_hover_text("Source position and origin time from picks").clicked() {
        windows.locate_open = true;
        ui.close();
    }
}

pub fn windows(
    mut contexts: EguiContexts,
    mut workspace: ResMut<Workspace>,
    view: Res<TrackView>,
    tracks: Query<&Track>,
    mut windows: ResMut<LocateWindows>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    poll(windows, &workspace);
    pick_window(&ctx, windows, &mut workspace, &view, &tracks);
    locate_window(&ctx, windows, &mut workspace, &view, &tracks);
    Ok(())
}

fn poll(windows: &mut LocateWindows, workspace: &Workspace) {
    if let Some((_, task)) = &mut windows.pick_task
        && let Some(results) = check_ready(task)
    {
        let (job, _) = windows.pick_task.take().expect("checked above");
        let project = &workspace.project;
        let rows = job
            .rows
            .into_iter()
            .zip(results)
            .map(|((trace, code, phase), result)| {
                let existing = project
                    .event_picks(job.event, trace)
                    .find(|p| p.phase == phase)
                    .map(|p| (p.time, p.provenance.is_manual()));
                let apply = matches!(result, Ok(Some(_))) && !existing.is_some_and(|e| e.1);
                PickRow { trace, code, phase, result, existing, apply }
            })
            .collect();
        windows.pick_rows = Some((rows, job.config_hash, job.event));
    }
    if let Some((_, task)) = &mut windows.locate_task
        && let Some(fit) = check_ready(task)
    {
        let (job, _) = windows.locate_task.take().expect("checked above");
        windows.fit = fit.map(|fit| FitView { fit, reference: job.reference, picks: job.picks, config_hash: job.config_hash, event: job.event });
    }
}

fn selected<'a>(tracks: &'a Query<&Track>) -> Vec<&'a Track> {
    let mut selected: Vec<&Track> = tracks.iter().filter(|t| t.selected).collect();
    selected.sort_by_key(|t| t.row.map_or(i64::MAX, |(top, _)| top as i64));
    selected
}

/// Seconds from the project reference, signed, to the microsecond.
fn relative(project: &Project, at: Timestamp) -> String {
    match project.reference {
        Some(reference) => format!("{:+.6} s", at.seconds_since(reference)),
        None => at.to_string(),
    }
}

/// Copies the chosen tracks' series out (within `event`'s window, members
/// only) and picks them off the UI thread. Tracks without settings for their
/// phenomenology are skipped.
fn start_pick(windows: &mut LocateWindows, project: &Project, chosen: &[&Track], event: Option<Id>) {
    let window = event.and_then(|e| project.event(e));
    let mut inputs = Vec::new();
    let mut rows = Vec::new();
    for track in chosen {
        let (Some(id), Some(series), Some(trace)) = (&track.phenomenology_id, &track.series, project.trace(track.trace)) else {
            continue;
        };
        if !track.member {
            continue;
        }
        let Some(&settings) = windows.settings.get(id) else {
            continue;
        };
        let phase = project.phenomenology(id).and_then(|p| p.propagation.first_phase()).unwrap_or("").to_string();
        let values = match track.kind {
            SampleKind::Real => series.values.clone(),
            SampleKind::Complex => series.samples.series(IqComponent::Magnitude),
        };
        let (lo, hi) = match window {
            Some(e) => {
                let corrected = project.corrected_start(trace);
                let index = |t: Timestamp| (t.seconds_since(corrected) * track.rate).clamp(0.0, values.len() as f64) as usize;
                (index(e.start), index(e.end))
            }
            None => (0, values.len()),
        };
        let start = trace.start.add_seconds(lo as f64 / track.rate);
        inputs.push(PickInput { values: values.range(lo, hi).collect(), rate: track.rate, start, settings });
        rows.push((track.trace, track.code.clone(), phase));
    }
    let key = format!("{:?} {:?}", windows.choice, inputs.iter().map(|i| i.settings).collect::<Vec<_>>());
    let choice = windows.choice;
    let task = AsyncComputeTaskPool::get().spawn(async move { inputs.iter().map(|i| run_pick(i, choice)).collect() });
    windows.pick_rows = None;
    windows.pick_task = Some((PickJob { rows, config_hash: config_hash(&key), event }, task));
}

fn start_locate(windows: &mut LocateWindows, usable: &[(Id, Observation)], reference: Timestamp, event: Option<Id>) {
    let settings = LocateSettings {
        max_depth_m: windows.max_depth_km * 1e3,
        half_extent_m: (windows.extent_km > 0.0).then_some(windows.extent_km * 1e3),
        ..LocateSettings::default()
    };
    let key = format!("{} {} {:?}", settings.max_depth_m, windows.extent_km, usable.iter().map(|u| u.0).collect::<Vec<_>>());
    let (ids, observations): (Vec<Id>, Vec<Observation>) = usable.iter().copied().unzip();
    let task = AsyncComputeTaskPool::get().spawn(async move { locate(&observations, &settings) });
    windows.locate_task = Some((LocateJob { picks: ids, reference, config_hash: config_hash(&key), event }, task));
}

fn pick_window(ctx: &egui::Context, windows: &mut LocateWindows, workspace: &mut Workspace, view: &TrackView, tracks: &Query<&Track>) {
    if !windows.pick_open {
        return;
    }
    let chosen = selected(tracks);
    let mut open = true;
    egui::Window::new("Pick").open(&mut open).default_width(560.0).show(ctx, |ui| {
        ui.weak("One onset per selected track: an STA/LTA trigger refined by AIC, on the plotted series. Automatic picks take the first phase.");
        if view.event.is_some() {
            ui.weak("Within the event under review: its members and window only.");
        }
        if view.show_raw {
            ui.colored_label(WARN_COLOR, "⚠ Raw display is on: picks use unprocessed samples.");
        }
        let project = &workspace.project;
        // Settings per phenomenology present.
        let mut present: Vec<(PhenomenologyId, String, f64)> = Vec::new();
        for track in &chosen {
            if let Some(id) = &track.phenomenology_id {
                match present.iter_mut().find(|p| &p.0 == id) {
                    Some(p) => p.2 = p.2.min(track.rate),
                    None => present.push((id.clone(), track.phenomenology.clone().unwrap_or_default(), track.rate)),
                }
            }
        }
        if present.is_empty() {
            ui.label("Select assigned tracks to pick (click their rows).");
        } else {
            egui::Grid::new("pick_settings").striped(true).num_columns(6).show(ui, |ui| {
                for heading in ["", "STA", "LTA", "On", "Off", "Recursive"] {
                    ui.strong(heading);
                }
                ui.end_row();
                for (id, name, rate) in &present {
                    let s = windows.settings.entry(id.clone()).or_insert_with(|| default_pick_settings(id.as_str(), *rate));
                    ui.label(name);
                    let (sta_speed, lta_speed) = (s.sta_s * 0.01, s.lta_s * 0.01);
                    ui.add(egui::DragValue::new(&mut s.sta_s).range(1e-12..=1e6).speed(sta_speed).suffix(" s"));
                    ui.add(egui::DragValue::new(&mut s.lta_s).range(1e-12..=1e6).speed(lta_speed).suffix(" s"));
                    ui.add(egui::DragValue::new(&mut s.on).range(1.0..=1e6).speed(0.1));
                    ui.add(egui::DragValue::new(&mut s.off).range(0.1..=1e6).speed(0.1));
                    ui.checkbox(&mut s.recursive, "");
                    ui.end_row();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Take");
                ui.radio_value(&mut windows.choice, Choice::First, "first trigger");
                ui.radio_value(&mut windows.choice, Choice::Strongest, "strongest trigger");
            });
        }

        let busy = windows.pick_task.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(!present.is_empty() && !busy, egui::Button::new("Pick")).clicked() {
                start_pick(windows, project, &chosen, view.event);
            }
            if busy {
                ui.spinner();
                ui.label("picking…");
            }
        });

        let mut applied = false;
        if let Some((rows, hash, event)) = &mut windows.pick_rows {
            ui.separator();
            egui::Grid::new("pick_results").striped(true).num_columns(6).show(ui, |ui| {
                for heading in ["", "Trace", "Phase", "Onset", "STA/LTA", "σ"] {
                    ui.strong(heading);
                }
                ui.end_row();
                for row in rows.iter_mut() {
                    let manual = row.existing.is_some_and(|e| e.1);
                    let enabled = matches!(row.result, Ok(Some(_))) && !manual;
                    ui.add_enabled(enabled, egui::Checkbox::without_text(&mut row.apply));
                    ui.monospace(&row.code);
                    ui.label(if row.phase.is_empty() { "onset" } else { &row.phase });
                    match &row.result {
                        Ok(Some(p)) => {
                            ui.monospace(relative(project, p.time));
                            ui.monospace(format!("{:.1}", p.ratio));
                            ui.monospace(format!("{:.2e} s", p.uncertainty_s));
                        }
                        Ok(None) => {
                            ui.colored_label(WARN_COLOR, "no trigger");
                            ui.label("");
                            ui.label("");
                        }
                        Err(error) => {
                            ui.colored_label(WARN_COLOR, error);
                            ui.label("");
                            ui.label("");
                        }
                    }
                    ui.end_row();
                    if manual {
                        ui.label("");
                        ui.colored_label(WARN_COLOR, "🔒 manual pick kept");
                        ui.end_row();
                    }
                }
            });
            let changes: Vec<Change> = rows
                .iter()
                .filter(|r| r.apply)
                .filter_map(|r| Some(pick_change(*event, r.trace, &r.phase, r.result.as_ref().ok()?.as_ref()?, PICK_STAGE, hash)))
                .collect();
            ui.horizontal(|ui| {
                let n = changes.len();
                if ui.add_enabled(n > 0, egui::Button::new(format!("Apply {n} pick{}", if n == 1 { "" } else { "s" }))).clicked()
                    && workspace.execute_or_report(format!("Pick {n} track{}", if n == 1 { "" } else { "s" }), changes)
                {
                    applied = true;
                }
                ui.weak("Replaces automatic picks of the same phase.");
            });
        }
        if applied {
            windows.pick_rows = None;
        }
    });
    windows.pick_open = open;
}

fn locate_window(ctx: &egui::Context, windows: &mut LocateWindows, workspace: &mut Workspace, view: &TrackView, tracks: &Query<&Track>) {
    if !windows.locate_open {
        return;
    }
    let mut open = true;
    let mut changes: Option<(String, Vec<Change>)> = None;
    egui::Window::new("Locate").open(&mut open).default_width(620.0).show(ctx, |ui| {
        let project = &workspace.project;
        let event = view.event;
        let scoped: Vec<&Pick> = project.picks().iter().filter(|p| p.event == event).collect();
        let reference = project.reference.or_else(|| scoped.iter().filter_map(|p| project.pick_time(p)).reduce(Timestamp::min));
        if event.is_some() {
            ui.weak("Picks, origin and alignment of the event under review.");
        }
        // A solution belongs to the event it was computed for.
        if windows.fit.as_ref().is_some_and(|f| f.event != event) {
            windows.fit = None;
        }
        let fit_residual = |id: Id| {
            let view = windows.fit.as_ref()?;
            let i = view.picks.iter().position(|p| *p == id)?;
            Some(view.fit.residuals[i])
        };

        if scoped.is_empty() {
            ui.label("No picks yet. Use Process ▸ Pick…, or hover a row and press P (first phase) or S.");
        }
        let mut usable = Vec::new();
        egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
            egui::Grid::new("locate_picks").striped(true).num_columns(6).show(ui, |ui| {
                for heading in ["", "Sensor", "Phase", "Arrival", "σ", "Residual"] {
                    ui.strong(heading);
                }
                ui.end_row();
                let mut picks = scoped.clone();
                picks.sort_by(|a, b| {
                    let t = |p: &Pick| project.pick_time(p).map_or(f64::INFINITY, |t| t.seconds_since(reference.unwrap_or(t)));
                    t(a).total_cmp(&t(b))
                });
                for pick in picks {
                    let code = project.trace(pick.trace).and_then(|t| project.sensor(t.sensor)).map_or("?", |s| s.code.as_str());
                    let observation = reference.ok_or("no reference").and_then(|r| observation(project, pick, r));
                    let mut include = !windows.excluded.contains(&pick.id);
                    if ui.add_enabled(observation.is_ok(), egui::Checkbox::without_text(&mut include)).changed() {
                        if include {
                            windows.excluded.remove(&pick.id);
                        } else {
                            windows.excluded.insert(pick.id);
                        }
                    }
                    ui.monospace(code);
                    let manual = if pick.provenance.is_manual() { " ✋" } else { "" };
                    ui.label(format!("{}{manual}", pick.phase_label()));
                    ui.monospace(project.pick_time(pick).map_or("?".into(), |t| relative(project, t)));
                    ui.monospace(format!("{:.2e} s", pick.uncertainty_s));
                    match (&observation, fit_residual(pick.id)) {
                        (Err(why), _) => ui.colored_label(WARN_COLOR, *why),
                        (Ok(_), Some(r)) => ui.monospace(format!("{r:+.2e} s")),
                        (Ok(_), None) => ui.label(""),
                    };
                    ui.end_row();
                    if let Ok(o) = observation
                        && include
                    {
                        usable.push((pick.id, o));
                    }
                }
            });
        });

        egui::Grid::new("locate_settings").num_columns(2).show(ui, |ui| {
            ui.label("Depth");
            ui.add(egui::DragValue::new(&mut windows.max_depth_km).range(0.0..=700.0).speed(0.5).prefix("0 to ").suffix(" km"))
                .on_hover_text("Zero fixes the source at the surface");
            ui.end_row();
            ui.label("Grid");
            ui.add(egui::DragValue::new(&mut windows.extent_km).range(0.0..=5000.0).speed(1.0).prefix("± ").suffix(" km"))
                .on_hover_text("Half-width of the first grid; zero sizes it from the sensor spread");
            ui.end_row();
        });
        let busy = windows.locate_task.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(usable.len() >= 2 && !busy && reference.is_some(), egui::Button::new("Locate")).clicked()
                && let Some(reference) = reference
            {
                start_locate(windows, &usable, reference, event);
            }
            if busy {
                ui.spinner();
                ui.label("searching…");
            } else if usable.len() < 4 {
                ui.colored_label(WARN_COLOR, format!("{} usable pick(s); four or more constrain position, depth and time", usable.len()));
            }
        });

        if let Some(view) = &windows.fit {
            ui.separator();
            let fit = &view.fit;
            let origin_time = view.reference.add_seconds(fit.origin);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    egui::Grid::new("fit").num_columns(2).show(ui, |ui| {
                        ui.label("Latitude");
                        ui.monospace(format!("{:.5}°", fit.source.latitude));
                        ui.end_row();
                        ui.label("Longitude");
                        ui.monospace(format!("{:.5}°", fit.source.longitude));
                        ui.end_row();
                        ui.label("Depth");
                        ui.monospace(format!("{:.2} km", fit.source.depth_m / 1e3));
                        ui.end_row();
                        ui.label("Origin");
                        ui.monospace(relative(project, origin_time));
                        ui.end_row();
                        ui.label("RMS");
                        ui.monospace(format!("{:.2e} s", fit.rms_s));
                        ui.end_row();
                        ui.label("χ²/dof");
                        let reduced = fit.chi2 / fit.dof as f64;
                        let color = if reduced <= 4.0 { GOOD_COLOR } else { WARN_COLOR };
                        ui.colored_label(color, format!("{reduced:.2}")).on_hover_text("Near 1 when residuals match the pick uncertainties");
                        ui.end_row();
                        if let Some((source, time)) = workspace.truth_origin {
                            ui.label("vs truth");
                            let error = surface_distance(fit.source.latitude, fit.source.longitude, source.latitude, source.longitude);
                            ui.monospace(format!(
                                "{:.0} m off, depth {:+.0} m, origin {:+.2e} s",
                                error,
                                fit.source.depth_m - source.depth_m,
                                origin_time.seconds_since(time)
                            ));
                            ui.end_row();
                        }
                    });
                    ui.add_space(8.0);
                    if ui.button("Save origin").on_hover_text("Store this solution; tracks show predicted arrivals from it").clicked() {
                        let origin = Origin {
                            latitude: fit.source.latitude,
                            longitude: fit.source.longitude,
                            depth_m: fit.source.depth_m,
                            time: origin_time,
                            rms_s: fit.rms_s,
                            provenance: Provenance::Auto { stage: LOCATE_STAGE.to_string(), config_hash: view.config_hash.clone() },
                        };
                        changes = Some(("Save origin".to_string(), vec![Change::SetOrigin { origin: Some(origin), event }]));
                    }
                });
                map(ui, project, view, workspace.truth_origin.map(|t| t.0));
            });
        }

        ui.separator();
        ui.strong("Align across phenomenologies");
        ui.horizontal(|ui| {
            match project.origin_of(event) {
                Some(origin) => {
                    let (moveout, locked) = moveout_changes(project, event, origin);
                    let n = moveout.len();
                    let response = ui
                        .add_enabled(n > 0, egui::Button::new(format!("Remove moveout ({n})")))
                        .on_hover_text("Shift every placed trace by minus its first-phase travel time, so arrivals line up at the origin time");
                    if response.clicked() {
                        changes = Some((format!("Remove moveout from {n} traces"), moveout));
                    }
                    if locked > 0 {
                        ui.colored_label(WARN_COLOR, format!("🔒 {locked} locked kept"));
                    }
                    if ui.button("Clear origin").clicked() {
                        changes = Some(("Clear origin".to_string(), vec![Change::SetOrigin { origin: None, event }]));
                    }
                }
                None => {
                    ui.weak("Save an origin to remove moveout.");
                }
            }
        });
        ui.horizontal(|ui| {
            let reference = view.reference.filter(|id| project.event_picks(event, *id).next().is_some());
            let chosen: Vec<Id> = selected(tracks).iter().map(|t| t.trace).collect();
            let (aligned, locked) = reference.map_or((Vec::new(), 0), |r| pick_alignment_changes(project, event, r, &chosen));
            let n = aligned.len();
            if ui
                .add_enabled(n > 0, egui::Button::new(format!("Align picks to ★ reference ({n})")))
                .on_hover_text("Shift selected tracks so their pick sits on the reference trace's pick of the same phase")
                .clicked()
            {
                changes = Some((format!("Align {n} tracks on picks"), aligned));
            }
            if reference.is_none() {
                ui.weak("Needs a ★ reference track with a pick.");
            } else if locked > 0 {
                ui.colored_label(WARN_COLOR, format!("🔒 {locked} locked kept"));
            }
        });
    });
    if let Some((label, changes)) = changes {
        workspace.execute_or_report(label, changes);
    }
    windows.locate_open = open;
}

fn color32(color: squiggle_core::Rgba) -> egui::Color32 {
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgb(byte(color.red), byte(color.green), byte(color.blue))
}

/// The first grid's misfit, sensors, solution and truth on a local plane.
fn map(ui: &mut egui::Ui, project: &Project, view: &FitView, truth: Option<squiggle_core::locate::Source>) {
    let (response, painter) = ui.allocate_painter(egui::vec2(MAP_SIZE, MAP_SIZE), egui::Sense::hover());
    let rect = response.rect;
    let grid = &view.fit.map;
    let (lat0, lon0) = grid.centre;
    let scale = (MAP_SIZE as f64) / (2.0 * grid.half_extent_m);
    let to_screen = |lat: f64, lon: f64| {
        let north = (lat - lat0).to_radians() * squiggle_core::locate::EARTH_RADIUS_M;
        let east = (lon - lon0).to_radians() * squiggle_core::locate::EARTH_RADIUS_M * lat0.to_radians().cos();
        rect.center() + egui::vec2((east * scale) as f32, (-north * scale) as f32)
    };
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(20));

    let n = grid.nodes;
    let min = grid.chi2.iter().copied().fold(f64::INFINITY, f64::min);
    let max = grid.chi2.iter().copied().filter(|v| v.is_finite()).fold(min, f64::max);
    let span = ((max - min) + 1.0).ln().max(1e-9);
    let cell = MAP_SIZE / (n - 1) as f32;
    for north in 0..n {
        for east in 0..n {
            let (lat, lon) = grid.node(east, north);
            let t = 1.0 - (((grid.chi2[north * n + east] - min) + 1.0).ln() / span) as f32;
            let [r, g, b, _] = colormap(t * 0.85);
            let centre = to_screen(lat, lon);
            painter.rect_filled(egui::Rect::from_center_size(centre, egui::vec2(cell + 0.5, cell + 0.5)), 0.0, egui::Color32::from_rgb(r, g, b));
        }
    }

    let placed: HashSet<Id> = view.picks.iter().filter_map(|p| project.pick(*p)).filter_map(|p| project.trace(p.trace)).map(|t| t.sensor).collect();
    for sensor in project.sensors() {
        let Some(Location { latitude, longitude, .. }) = sensor.location else {
            continue;
        };
        let color = sensor
            .assignment
            .as_ref()
            .and_then(|a| project.phenomenology(&a.phenomenology))
            .map_or(egui::Color32::GRAY, |p| color32(p.color));
        let at = to_screen(latitude, longitude);
        let size = 6.0;
        let triangle = vec![at + egui::vec2(0.0, -size), at + egui::vec2(size * 0.87, size * 0.5), at + egui::vec2(-size * 0.87, size * 0.5)];
        let stroke = if placed.contains(&sensor.id) { egui::Stroke::new(1.0, egui::Color32::WHITE) } else { egui::Stroke::NONE };
        painter.add(egui::Shape::convex_polygon(triangle, color, stroke));
    }
    if let Some(truth) = truth {
        let at = to_screen(truth.latitude, truth.longitude);
        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 255, 120));
        painter.line_segment([at + egui::vec2(-6.0, -6.0), at + egui::vec2(6.0, 6.0)], stroke);
        painter.line_segment([at + egui::vec2(-6.0, 6.0), at + egui::vec2(6.0, -6.0)], stroke);
    }
    let source = view.fit.source;
    let at = to_screen(source.latitude, source.longitude);
    painter.circle(at, 5.0, egui::Color32::WHITE, egui::Stroke::new(1.5, egui::Color32::BLACK));
    let caption = format!("± {:.1} km · ○ solution{}", grid.half_extent_m / 1e3, if truth.is_some() { " · ✕ truth" } else { "" });
    painter.text(rect.left_bottom() + egui::vec2(4.0, -4.0), egui::Align2::LEFT_BOTTOM, caption, egui::FontId::monospace(10.0), egui::Color32::WHITE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;
    use crate::workspace::{SampleState, SyntheticEvent};
    use squiggle_core::Analyst;

    /// M4 self-check: pick every trace of the synthetic EMP, seismic and
    /// infrasound event with the default settings, then locate.
    #[test]
    fn the_located_event_is_found_within_tolerance() {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Located).unwrap();
        let project = workspace.project.clone();
        let mut changes = Vec::new();
        for trace in project.traces() {
            let Some(SampleState::Loaded(samples)) = workspace.samples.get(&trace.id) else { panic!("loaded") };
            let phenomenology = project.trace_phenomenology(trace).unwrap();
            let input = PickInput {
                values: samples.series(IqComponent::Magnitude).iter().collect(),
                rate: trace.sample_rate,
                start: trace.start,
                settings: default_pick_settings(phenomenology.id.as_str(), trace.sample_rate),
            };
            let picked = run_pick(&input, Choice::First).unwrap().expect("a trigger");
            let phase = phenomenology.propagation.first_phase().unwrap_or("");
            changes.push(pick_change(None, trace.id, phase, &picked, PICK_STAGE, "test"));

            // Each pick lands within a sample and a half of the true first arrival.
            let code = &project.sensor(trace.sensor).unwrap().code;
            let station = synthetic::STATIONS.iter().find(|s| &s.code() == code).unwrap();
            let truth = synthetic::ORIGIN.parse::<Timestamp>().unwrap().add_seconds(station.arrivals()[0].1);
            let error = picked.time.seconds_since(truth);
            assert!(error.abs() <= 1.5 / trace.sample_rate, "{code}: {error:+e} s");
        }
        workspace.execute("pick", changes).unwrap();

        let project = &workspace.project;
        let reference = project.reference.unwrap();
        let observations: Vec<Observation> = project.picks().iter().map(|p| observation(project, p, reference).unwrap()).collect();
        assert_eq!(observations.len(), synthetic::STATIONS.len());
        let fit = locate(&observations, &LocateSettings::default()).unwrap();
        let (source, time) = workspace.truth_origin.unwrap();
        let horizontal = surface_distance(fit.source.latitude, fit.source.longitude, source.latitude, source.longitude);
        let origin = reference.add_seconds(fit.origin).seconds_since(time);
        assert!(horizontal < 100.0, "{horizontal} m");
        assert!((fit.source.depth_m - source.depth_m).abs() < 250.0, "{:?}", fit.source);
        // EMP picks anchor the origin to microseconds.
        assert!(origin.abs() < 1e-5, "{origin:+e} s");
    }

    #[test]
    fn moveout_lines_arrivals_up_at_the_origin() {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Located).unwrap();
        let (source, time) = workspace.truth_origin.unwrap();
        let origin = Origin {
            latitude: source.latitude,
            longitude: source.longitude,
            depth_m: source.depth_m,
            time,
            rms_s: 0.0,
            provenance: Provenance::Auto { stage: "test".into(), config_hash: "x".into() },
        };
        let (changes, locked) = moveout_changes(&workspace.project, None, &origin);
        assert_eq!((changes.len(), locked), (synthetic::STATIONS.len(), 0));
        workspace.execute("moveout", changes).unwrap();
        let project = &workspace.project;
        for trace in project.traces() {
            let code = &project.sensor(trace.sensor).unwrap().code;
            let station = synthetic::STATIONS.iter().find(|s| &s.code() == code).unwrap();
            let arrival = synthetic::ORIGIN.parse::<Timestamp>().unwrap().add_seconds(station.arrivals()[0].1);
            // Where the arrival is drawn after the shift.
            let drawn = arrival.add_seconds(trace.offset());
            assert!(drawn.seconds_since(time).abs() < 1e-4, "{code}");
        }
    }

    #[test]
    fn pick_alignment_matches_phases() {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Located).unwrap();
        let traces: Vec<Id> = workspace.project.traces().iter().map(|t| t.id).collect();
        let manual = |trace: Id, phase: &str, at: f64| Change::SetPick {
            pick: Pick {
                id: Id::generate(),
                trace,
                time: synthetic::ORIGIN.parse::<Timestamp>().unwrap().add_seconds(at),
                phase: phase.to_string(),
                uncertainty_s: 0.01,
                provenance: Provenance::Manual { analyst: Analyst("ana".into()), at: Timestamp::now() },
                event: None,
            },
            revert: false,
        };
        // Seismic traces 3 and 4: P at 12 s and 13 s.
        workspace.execute("picks", vec![manual(traces[3], "P", 12.0), manual(traces[4], "S", 20.0), manual(traces[4], "P", 13.0)]).unwrap();
        let (changes, _) = pick_alignment_changes(&workspace.project, None, traces[3], &traces);
        assert_eq!(changes.len(), 1);
        workspace.execute("align", changes).unwrap();
        assert!((workspace.project.trace(traces[4]).unwrap().offset() + 1.0).abs() < 1e-9);
    }
}
