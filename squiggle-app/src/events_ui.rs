//! Events: detection and association, the catalogue, and event edits.
//!
//! Detection streams each track's plotted series through its phenomenology's
//! detector in chunks, as a live feed would, refines each trigger to an onset
//! by AIC, adds matched-filter onsets from the template library (sensors with
//! templates), and associates onsets across traces into candidate events, located
//! when enough members have positions. Applying replaces automatic events;
//! reviewed and rejected ones are kept. Any analyst edit to an automatic
//! event marks it reviewed.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::associate::{AssociateSettings, Candidate, associate, Arrival};
use squiggle_core::dsp::{DetectorSpec, MatchedSpec, Trigger, detect, matched, refine};
use squiggle_core::template::TemplateTrace;
use squiggle_core::locate::surface_distance;
use squiggle_core::sensor::Provenance;
use squiggle_core::{
    Analyst, Change, Event, EventStatus, Id, IqComponent, Location, Member, Origin, PhenomenologyId, Pick, Project,
    SampleKind, Timestamp,
};

use crate::classify_ui::{accept_change, label_change, label_picker, review_order};
use crate::locate_ui::{Picked, default_pick_settings, pick_change};
use crate::plot::Plot;
use crate::process_ui::config_hash;
use crate::tracks::{Track, TrackView};
use crate::workspace::Workspace;
use crate::theme;

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
const GOOD_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 210, 110);
const REJECT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 80, 80);
/// Samples per detector call: tracks stream through detectors as a feed would.
pub const CHUNK: usize = 4096;
const DETECT_STAGE: &str = "detect";

/// Classic STA/LTA with the picker's windows for the phenomenology.
pub fn default_detector(phenomenology: &str, rate: f64) -> DetectorSpec {
    let p = default_pick_settings(phenomenology, rate);
    DetectorSpec::StaLta { sta_s: p.sta_s, lta_s: p.lta_s, on: p.on, off: p.off, recursive: false }
}

/// One trace to detect on, copied out on the UI thread.
pub struct DetectInput {
    pub trace: Id,
    pub sensor: Id,
    pub values: Vec<f32>,
    pub rate: f64,
    /// Clock-corrected time of `values[0]`, seconds after the reference.
    pub start: f64,
    /// Recorded time of `values[0]`.
    pub recorded: Timestamp,
    pub spec: DetectorSpec,
    /// First phase: what detections are picked as.
    pub phase: String,
    pub speed_m_s: Option<f64>,
    pub location: Option<Location>,
    /// Spread of the phenomenology's phases at the association radius.
    pub coda_s: f64,
    /// Library templates of this sensor, for the matched filter.
    pub templates: Vec<TemplateTrace>,
    pub matched: MatchedSpec,
}

impl DetectInput {
    /// From a trace's samples and the project. `None` while unassigned.
    pub fn new(project: &Project, trace: &squiggle_core::Trace, values: Vec<f32>, spec: DetectorSpec, radius_m: f64) -> Option<Self> {
        let phenomenology = project.trace_phenomenology(trace)?;
        let reference = project.reference.unwrap_or(trace.start);
        let phases = phenomenology.propagation.phases();
        let coda_s = match (phases.first(), phases.last()) {
            (Some(first), Some(last)) if phases.len() > 1 => radius_m * (1.0 / last.1 - 1.0 / first.1),
            _ => 0.0,
        };
        Some(Self {
            trace: trace.id,
            sensor: trace.sensor,
            values,
            rate: trace.sample_rate,
            start: project.corrected_start(trace).seconds_since(reference),
            recorded: trace.start,
            spec,
            phase: phenomenology.propagation.first_phase().unwrap_or("").to_string(),
            speed_m_s: phenomenology.propagation.speed(""),
            location: project.sensor(trace.sensor).and_then(|s| s.location),
            coda_s: coda_s.max(0.0),
            templates: Vec::new(),
            matched: MatchedSpec::default(),
        })
    }

    /// Adds a matched filter on the library's templates of this sensor.
    pub fn with_templates(mut self, project: &Project, spec: MatchedSpec) -> Self {
        self.templates = project.templates().iter().flat_map(|t| &t.traces).filter(|t| t.sensor == self.sensor).cloned().collect();
        self.matched = spec;
        self
    }
}

/// Matched-filter onsets this close to a detector's are the same arrival;
/// detector onsets within the template span after one are its later phases.
const SAME_ONSET_S: f64 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Index into the inputs.
    pub input: usize,
    pub onset: Picked,
    /// Onset and trigger end, corrected, seconds after the reference.
    pub at: f64,
    pub end: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DetectOutcome {
    pub detections: Vec<Detection>,
    /// Arrivals index `detections`.
    pub candidates: Vec<Candidate>,
    pub errors: Vec<(Id, String)>,
}

/// Detects on every input in chunks of `chunk` samples, refines onsets and associates them.
pub fn run_detection(inputs: &[DetectInput], settings: &AssociateSettings, chunk: usize) -> DetectOutcome {
    let mut detections = Vec::new();
    let mut errors = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let mut own = Vec::new();
        match detect(&input.spec, input.rate, &input.values, chunk) {
            Ok(triggers) => {
                let mut floor = 0;
                for trigger in triggers {
                    own.extend(input.onset(i, &trigger, floor));
                    floor = trigger.off;
                }
            }
            Err(error) => errors.push((input.trace, error)),
        }
        if !input.templates.is_empty() {
            let templates: Vec<&TemplateTrace> = input.templates.iter().collect();
            match matched(&templates, input.rate, &input.matched, &input.values, chunk) {
                Ok(triggers) => {
                    let matched: Vec<Detection> = triggers.iter().map(|t| input.matched_onset(i, t)).collect();
                    own = merge_matched(own, &matched, input.matched.length_s);
                }
                Err(error) => errors.push((input.trace, error)),
            }
        }
        detections.extend(own);
    }
    let refs: Vec<&DetectInput> = inputs.iter().collect();
    let candidates = associate(&arrivals(&refs, &detections), settings);
    DetectOutcome { detections, candidates, errors }
}

impl DetectInput {
    /// Samples before a trigger's `on` that onset refinement may reach.
    pub fn lookback(&self) -> usize {
        4 * self.spec.window(self.rate) + 8
    }

    /// Samples after a trigger's `on` that onset refinement reads.
    pub fn lookahead(&self) -> usize {
        self.spec.window(self.rate) + 8
    }

    /// A detector trigger refined to an onset by AIC over `values`, not before `floor`.
    pub fn onset(&self, input: usize, trigger: &Trigger, floor: usize) -> Option<Detection> {
        let onset = refine(&self.values, self.rate, trigger, self.spec.window(self.rate), floor)?;
        let offset = onset.index as f64 / self.rate;
        Some(Detection {
            input,
            onset: Picked { time: self.recorded.add_seconds(offset), ratio: onset.ratio, uncertainty_s: onset.uncertainty_s },
            at: self.start + offset,
            end: self.start + trigger.off as f64 / self.rate,
        })
    }

    /// A matched-filter trigger: its onset is the best alignment's.
    pub fn matched_onset(&self, input: usize, trigger: &Trigger) -> Detection {
        let top = self.templates.iter().map(|t| t.rate).fold(0.0, f64::max);
        let step = (self.rate / top).floor().max(1.0) / self.rate;
        let offset = trigger.on as f64 / self.rate;
        Detection {
            input,
            onset: Picked { time: self.recorded.add_seconds(offset), ratio: trigger.peak, uncertainty_s: step },
            at: self.start + offset,
            end: self.start + trigger.off as f64 / self.rate,
        }
    }
}

/// One trace's detector onsets with its matched-filter onsets, in order: a
/// matched onset near a detector's is dropped; detector onsets within `span`
/// after a kept one are its later phases and are dropped.
pub fn merge_matched(mut own: Vec<Detection>, matched: &[Detection], span: f64) -> Vec<Detection> {
    for m in matched {
        if own.iter().any(|d| (d.at - m.at).abs() < SAME_ONSET_S) {
            continue;
        }
        own.retain(|d| !(d.at > m.at && d.at < m.at + span));
        own.push(m.clone());
    }
    own
}

/// Association inputs for `detections`.
pub fn arrivals(inputs: &[&DetectInput], detections: &[Detection]) -> Vec<Arrival> {
    detections
        .iter()
        .map(|d| {
            let input = inputs[d.input];
            Arrival {
                trace: input.trace,
                sensor: input.sensor,
                time: d.at,
                speed_m_s: input.speed_m_s,
                location: input.location,
                sigma_s: d.onset.uncertainty_s,
                coda_s: input.coda_s,
            }
        })
        .collect()
}

/// What applying a detection needs of each input.
#[derive(Clone, Debug)]
pub struct DetectMeta {
    pub trace: Id,
    pub phase: String,
    pub rate: f64,
}

impl From<&DetectInput> for DetectMeta {
    fn from(input: &DetectInput) -> Self {
        Self { trace: input.trace, phase: input.phase.clone(), rate: input.rate }
    }
}

/// Events and picks for each candidate; with `replace`, automatic events go
/// first. A candidate whose arrivals are mostly on members of a kept event,
/// within its window, is that event again and is skipped.
pub fn event_changes(project: &Project, meta: &[DetectMeta], outcome: &DetectOutcome, reference: Timestamp, replace: bool, hash: &str) -> Vec<Change> {
    let mut changes: Vec<Change> = if replace {
        project.events().iter().filter(|e| Project::replaceable(e)).map(|e| Change::RemoveEvent { event: e.id }).collect()
    } else {
        Vec::new()
    };
    let kept: Vec<&Event> = project.events().iter().filter(|e| !replace || !Project::replaceable(e)).collect();
    let auto = |stage: &str| Provenance::Auto { stage: stage.to_string(), config_hash: hash.to_string() };
    for candidate in &outcome.candidates {
        let members: Vec<&Detection> = candidate.arrivals.iter().map(|&i| &outcome.detections[i]).collect();
        let again = kept.iter().any(|e| {
            let (from, to) = (e.start.seconds_since(reference), e.end.seconds_since(reference));
            let inside = members.iter().filter(|d| e.member(meta[d.input].trace).is_some() && (from..=to).contains(&d.at)).count();
            2 * inside > members.len()
        });
        if again {
            continue;
        }
        let first = members.iter().map(|d| d.at).fold(f64::INFINITY, f64::min);
        let last = candidate
            .arrivals
            .iter()
            .chain(&candidate.absorbed)
            .map(|&i| outcome.detections[i].end)
            .fold(f64::NEG_INFINITY, f64::max);
        let slowest = members.iter().map(|d| meta[d.input].rate).fold(f64::INFINITY, f64::min);
        let pad = (0.25 * (last - first)).max(20.0 / slowest);
        let id = Id::generate();
        let origin = candidate.fit.as_ref().map(|fit| Origin {
            latitude: fit.source.latitude,
            longitude: fit.source.longitude,
            depth_m: fit.source.depth_m,
            time: reference.add_seconds(fit.origin),
            rms_s: fit.rms_s,
            provenance: auto("associate.locate"),
        });
        let mut event_members: Vec<Member> = Vec::new();
        for d in &members {
            if event_members.iter().all(|m| m.trace != meta[d.input].trace) {
                event_members.push(Member::new(meta[d.input].trace));
            }
        }
        changes.push(Change::AddEvent(Event {
            id,
            start: reference.add_seconds(first - pad),
            end: reference.add_seconds(last + pad),
            members: event_members,
            origin,
            status: EventStatus::Auto,
            provenance: auto(DETECT_STAGE),
            label: None,
            predictions: Vec::new(),
        }));
        for d in &members {
            let m = &meta[d.input];
            changes.push(pick_change(Some(id), m.trace, &m.phase, &d.onset, DETECT_STAGE, hash));
        }
    }
    changes
}

fn manual(analyst: &Analyst) -> Provenance {
    Provenance::Manual { analyst: analyst.clone(), at: Timestamp::now() }
}

/// An analyst edit: an automatic event becomes reviewed.
fn edited(mut event: Event) -> Event {
    if event.status == EventStatus::Auto {
        event.status = EventStatus::Reviewed;
    }
    event
}

/// A reviewed event made by the analyst over `start..end`.
pub fn new_event(analyst: &Analyst, start: Timestamp, end: Timestamp, traces: &[Id]) -> Event {
    Event {
        id: Id::generate(),
        start,
        end,
        members: traces.iter().map(|&t| Member::new(t)).collect(),
        origin: None,
        status: EventStatus::Reviewed,
        provenance: manual(analyst),
        label: None,
        predictions: Vec::new(),
    }
}

/// Splits `id` at `at` into two reviewed events with the same members; picks
/// at or after `at` move to the second. Both lose their origin.
pub fn split_changes(project: &Project, id: Id, at: Timestamp) -> Option<Vec<Change>> {
    let event = project.event(id)?;
    if at.seconds_since(event.start) <= 0.0 || event.end.seconds_since(at) <= 0.0 {
        return None;
    }
    let left = edited(Event { end: at, origin: None, ..event.clone() });
    let right = Event { id: Id::generate(), start: at, end: event.end, ..left.clone() };
    let mut changes = vec![Change::SetEvent(left), Change::AddEvent(right.clone())];
    for pick in project.picks().iter().filter(|p| p.event == Some(id)) {
        if project.pick_time(pick).is_some_and(|t| t.seconds_since(at) >= 0.0) {
            changes.push(Change::SetPick { pick: Pick { event: Some(right.id), ..pick.clone() }, revert: true });
        }
    }
    Some(changes)
}

/// Merges `ids` into the earliest: the union of windows and members, picks
/// moved unless that slot is taken. The merged event is reviewed, unlocated.
pub fn merge_changes(project: &Project, ids: &[Id]) -> Option<Vec<Change>> {
    let mut events: Vec<&Event> = ids.iter().filter_map(|&id| project.event(id)).collect();
    if events.len() < 2 {
        return None;
    }
    events.sort_by(|a, b| a.start.seconds_since(b.start).total_cmp(&0.0));
    let mut merged = edited(Event { origin: None, ..events[0].clone() });
    for other in &events[1..] {
        if other.start.seconds_since(merged.start) < 0.0 {
            merged.start = other.start;
        }
        if other.end.seconds_since(merged.end) > 0.0 {
            merged.end = other.end;
        }
        for member in &other.members {
            if merged.member(member.trace).is_none() {
                merged.members.push(member.clone());
            }
        }
    }
    let mut taken: Vec<(Id, String)> =
        project.picks().iter().filter(|p| p.event == Some(merged.id)).map(|p| (p.trace, p.phase.clone())).collect();
    let mut changes = vec![Change::SetEvent(merged.clone())];
    for other in &events[1..] {
        for pick in project.picks().iter().filter(|p| p.event == Some(other.id)) {
            let slot = (pick.trace, pick.phase.clone());
            if !taken.contains(&slot) {
                taken.push(slot);
                changes.push(Change::SetPick { pick: Pick { event: Some(merged.id), ..pick.clone() }, revert: true });
            }
        }
    }
    changes.extend(events[1..].iter().map(|e| Change::RemoveEvent { event: e.id }));
    Some(changes)
}

pub fn window_change(project: &Project, id: Id, start: Timestamp, end: Timestamp) -> Option<Change> {
    let event = project.event(id)?;
    Some(Change::SetEvent(edited(Event { start, end, ..event.clone() })))
}

pub fn member_change(project: &Project, id: Id, trace: Id, add: bool) -> Option<Change> {
    let mut event = project.event(id)?.clone();
    if add {
        if event.member(trace).is_some() {
            return None;
        }
        event.members.push(Member::new(trace));
    } else {
        let before = event.members.len();
        event.members.retain(|m| m.trace != trace);
        if event.members.len() == before {
            return None;
        }
    }
    Some(Change::SetEvent(edited(event)))
}

pub fn status_change(project: &Project, id: Id, status: EventStatus) -> Option<Change> {
    let event = project.event(id)?;
    (event.status != status).then(|| Change::SetEvent(Event { status, ..event.clone() }))
}

/// Starts reviewing `id`: frames its window and selects its members.
pub fn open_event(view: &mut TrackView, tracks: &mut Query<&mut Track>, project: &Project, id: Id, width: f64) {
    let (Some(event), Some(reference)) = (project.event(id), project.reference) else {
        return;
    };
    view.event = Some(id);
    let (from, to) = (event.start.seconds_since(reference), event.end.seconds_since(reference));
    let pad = 0.1 * (to - from);
    view.time.fit(from - pad, to + pad, width);
    for mut track in tracks.iter_mut() {
        track.selected = event.member(track.trace).is_some();
    }
}

struct DetectJob {
    meta: Vec<DetectMeta>,
    reference: Timestamp,
    hash: String,
    replace: bool,
}

#[derive(Resource)]
pub struct EventWindows {
    pub detect_open: bool,
    pub catalogue_open: bool,
    specs: HashMap<PhenomenologyId, DetectorSpec>,
    associate: AssociateSettings,
    /// Matched filter on library templates, where a sensor has any.
    matched: bool,
    matched_spec: MatchedSpec,
    replace: bool,
    selected_only: bool,
    task: Option<(DetectJob, Task<DetectOutcome>)>,
    result: Option<(DetectJob, DetectOutcome)>,
    filter: Option<EventStatus>,
    /// Catalogue and `]`/`[` follow the review queue: automatic events, most uncertain first.
    pub queue: bool,
    merge: Vec<Id>,
}

impl Default for EventWindows {
    fn default() -> Self {
        Self {
            detect_open: false,
            catalogue_open: false,
            specs: HashMap::new(),
            associate: AssociateSettings::default(),
            matched: true,
            matched_spec: MatchedSpec::default(),
            replace: true,
            selected_only: false,
            task: None,
            result: None,
            filter: None,
            queue: false,
            merge: Vec::new(),
        }
    }
}

pub fn events_menu(ui: &mut egui::Ui, windows: &mut EventWindows, view: &mut TrackView) {
    if ui.button("Detect…").on_hover_text("Find events: detectors on every track, then association across sensors").clicked() {
        windows.detect_open = true;
        ui.close();
    }
    if ui.button("Catalogue…").on_hover_text("Review, accept, reject, merge events").clicked() {
        windows.catalogue_open = true;
        ui.close();
    }
    ui.separator();
    if ui.add_enabled(view.event.is_some(), egui::Button::new("Close review")).clicked() {
        view.event = None;
        ui.close();
    }
    ui.weak("⌘/Ctrl+drag on the tracks makes an event");
}

pub fn windows(
    mut contexts: EguiContexts,
    plot: Res<Plot>,
    mut workspace: ResMut<Workspace>,
    mut view: ResMut<TrackView>,
    mut tracks: Query<&mut Track>,
    mut windows: ResMut<EventWindows>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    let width = plot.view.plot_width();
    if let Some((_, task)) = &mut windows.task
        && let Some(outcome) = check_ready(task)
    {
        let (job, _) = windows.task.take().expect("checked above");
        windows.result = Some((job, outcome));
    }
    detect_window(&ctx, windows, &mut workspace, &tracks);
    catalogue_window(&ctx, windows, &mut workspace, &mut view, &mut tracks, width);
    review_keys(&ctx, windows.queue, &mut workspace, &mut view, &mut tracks, width);
    Ok(())
}

/// While reviewing: `]` / `[` next and previous event (in queue order when
/// queued), `A` accept, `R` reject, `1`–`9` label, `Esc` close.
fn review_keys(ctx: &egui::Context, queue: bool, workspace: &mut Workspace, view: &mut TrackView, tracks: &mut Query<&mut Track>, width: f64) {
    let Some(current) = view.event else {
        return;
    };
    if ctx.egui_wants_keyboard_input() {
        return;
    }
    const DIGITS: [egui::Key; 9] = [
        egui::Key::Num1,
        egui::Key::Num2,
        egui::Key::Num3,
        egui::Key::Num4,
        egui::Key::Num5,
        egui::Key::Num6,
        egui::Key::Num7,
        egui::Key::Num8,
        egui::Key::Num9,
    ];
    let (next, previous, accept, reject, close, digit) = ctx.input(|i| {
        let plain = !i.modifiers.command;
        (
            i.key_pressed(egui::Key::CloseBracket),
            i.key_pressed(egui::Key::OpenBracket),
            plain && i.key_pressed(egui::Key::A),
            plain && i.key_pressed(egui::Key::R),
            i.key_pressed(egui::Key::Escape),
            DIGITS.iter().position(|k| plain && i.key_pressed(*k)),
        )
    });
    let project = &workspace.project;
    if next || previous {
        let order = review_order(project, queue);
        let target = match order.iter().position(|id| *id == current) {
            Some(i) => order.get(if next { i + 1 } else { i.wrapping_sub(1) }),
            // Left the queue (accepted or rejected): continue from its head.
            None => order.first(),
        };
        if let Some(&id) = target {
            open_event(view, tracks, project, id, width);
        }
    }
    if accept && let Some(change) = accept_change(&workspace.project, current) {
        workspace.execute_or_report("Accept event", vec![change]);
    }
    if reject && let Some(change) = status_change(&workspace.project, current, EventStatus::Rejected) {
        workspace.execute_or_report("Reject event", vec![change]);
    }
    if let Some(n) = digit
        && let Some(label) = workspace.project.taxonomy().ordered().get(n).map(|l| l.id.clone())
        && let Some(change) = label_change(&workspace.project, current, Some(label))
    {
        workspace.execute_or_report("Label event", vec![change]);
    }
    if close {
        view.event = None;
    }
}

fn spec_editor(ui: &mut egui::Ui, spec: &mut DetectorSpec, rate: f64) {
    let kinds = [
        DetectorSpec::StaLta { sta_s: 0.3, lta_s: 5.0, on: 4.0, off: 1.5, recursive: false },
        DetectorSpec::StaLta { sta_s: 0.3, lta_s: 5.0, on: 4.0, off: 1.5, recursive: true },
        DetectorSpec::Energy { window_s: 20.0 / rate, on: 0.1, off: 0.05 },
        DetectorSpec::Kurtosis { window_s: 200.0 / rate, on: 3.0, off: 1.0 },
    ];
    let current = spec.label();
    egui::ComboBox::from_id_salt(ui.next_auto_id()).selected_text(current).show_ui(ui, |ui| {
        for kind in kinds {
            if ui.selectable_label(kind.label() == current, kind.label()).clicked() {
                // Keep windows when switching between STA/LTA forms.
                *spec = match (*spec, kind) {
                    (DetectorSpec::StaLta { sta_s, lta_s, on, off, .. }, DetectorSpec::StaLta { recursive, .. }) => {
                        DetectorSpec::StaLta { sta_s, lta_s, on, off, recursive }
                    }
                    _ => kind,
                };
            }
        }
    });
    let seconds = |ui: &mut egui::Ui, value: &mut f64, label: &str| {
        let speed = *value * 0.01;
        ui.add(egui::DragValue::new(value).range(1e-12..=1e6).speed(speed).prefix(label).suffix(" s"));
    };
    let ratio = |ui: &mut egui::Ui, value: &mut f64, label: &str| {
        ui.add(egui::DragValue::new(value).range(0.0..=1e9).speed(0.05).prefix(label));
    };
    match spec {
        DetectorSpec::StaLta { sta_s, lta_s, on, off, .. } => {
            seconds(ui, sta_s, "STA ");
            seconds(ui, lta_s, "LTA ");
            ratio(ui, on, "on ");
            ratio(ui, off, "off ");
        }
        DetectorSpec::Energy { window_s, on, off } => {
            seconds(ui, window_s, "window ");
            ratio(ui, on, "on RMS ");
            ratio(ui, off, "off RMS ");
        }
        DetectorSpec::Kurtosis { window_s, on, off } => {
            seconds(ui, window_s, "window ");
            ratio(ui, on, "on ");
            ratio(ui, off, "off ");
        }
    }
}

fn detect_window(ctx: &egui::Context, windows: &mut EventWindows, workspace: &mut Workspace, tracks: &Query<&mut Track>) {
    if !windows.detect_open {
        return;
    }
    let mut open = true;
    let mut apply: Option<(String, Vec<Change>)> = None;
    theme::tool_window(ctx, "Detect Events").open(&mut open).default_width(640.0).show(ctx, |ui| {
        ui.weak("Each track's plotted series streams through its phenomenology's detector; onsets are refined by AIC, then associated across sensors.");
        ui.checkbox(&mut windows.selected_only, "Selected tracks only");
        let chosen: Vec<&Track> =
            tracks.iter().filter(|t| t.phenomenology_id.is_some() && t.series.is_some() && (!windows.selected_only || t.selected)).collect();
        let mut present: Vec<(PhenomenologyId, String, f64)> = Vec::new();
        for track in &chosen {
            let id = track.phenomenology_id.clone().expect("filtered");
            match present.iter_mut().find(|p| p.0 == id) {
                Some(p) => p.2 = p.2.min(track.rate),
                None => present.push((id, track.phenomenology.clone().unwrap_or_default(), track.rate)),
            }
        }
        egui::Grid::new("detectors").num_columns(2).show(ui, |ui| {
            for (id, name, rate) in &present {
                ui.label(name);
                let spec = windows.specs.entry(id.clone()).or_insert_with(|| default_detector(id.as_str(), *rate));
                ui.horizontal(|ui| spec_editor(ui, spec, *rate));
                ui.end_row();
            }
        });
        if present.is_empty() {
            ui.label("No assigned, loaded tracks.");
        }
        let templated = workspace.project.templates().iter().flat_map(|t| &t.traces).map(|t| t.sensor).collect::<std::collections::HashSet<Id>>();
        let with = chosen.iter().filter(|t| workspace.project.trace(t.trace).is_some_and(|t| templated.contains(&t.sensor))).count();
        ui.horizontal(|ui| {
            ui.add_enabled(with > 0, egui::Checkbox::new(&mut windows.matched, "Matched filter"))
                .on_hover_text("Also correlate each track with the library templates of its sensor (Events ▸ Templates); finds repeats below the detector's threshold");
            let m = &mut windows.matched_spec;
            ui.add_enabled_ui(with > 0 && windows.matched, |ui| {
                ui.add(egui::DragValue::new(&mut m.length_s).range(0.5..=600.0).speed(0.1).prefix("first ").suffix(" s"));
                ui.add(egui::DragValue::new(&mut m.on).range(0.05..=1.0).speed(0.01).prefix("on CC "));
                ui.add(egui::DragValue::new(&mut m.off).range(0.0..=1.0).speed(0.01).prefix("off "));
            });
            ui.weak(format!("{with} of {} tracks have templates", chosen.len()));
        });
        ui.separator();
        let a = &mut windows.associate;
        let mut radius_km = a.radius_m / 1e3;
        let mut depth_km = a.max_depth_m / 1e3;
        egui::Grid::new("association").num_columns(2).show(ui, |ui| {
            ui.label("Radius");
            ui.add(egui::DragValue::new(&mut radius_km).range(0.001..=20_000.0).speed(1.0).suffix(" km"))
                .on_hover_text("Farthest a source may be from a sensor that records it; sets each phenomenology's coincidence window");
            ui.end_row();
            ui.label("Unknown propagation");
            ui.add(egui::DragValue::new(&mut a.max_window_s).range(0.0..=1e6).speed(0.1).prefix("± ").suffix(" s"));
            ui.end_row();
            ui.label("Minimum sensors");
            ui.add(egui::DragValue::new(&mut a.min_sensors).range(1..=1000));
            ui.end_row();
            ui.label("Outliers beyond");
            ui.add(egui::DragValue::new(&mut a.outlier_sigma).range(1.0..=1e6).speed(0.5).suffix(" σ"))
                .on_hover_text("Located events drop arrivals whose residual exceeds this many pick deviations");
            ui.end_row();
            ui.label("Depth");
            ui.add(egui::DragValue::new(&mut depth_km).range(0.0..=700.0).speed(0.5).prefix("0 to ").suffix(" km"));
            ui.end_row();
        });
        a.radius_m = radius_km * 1e3;
        a.max_depth_m = depth_km * 1e3;
        ui.checkbox(&mut windows.replace, "Replace automatic events").on_hover_text("Reviewed and rejected events are always kept");

        let project = &workspace.project;
        let busy = windows.task.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(!chosen.is_empty() && !busy && project.reference.is_some(), theme::primary("Detect")).clicked() {
                let mut inputs = Vec::new();
                for track in &chosen {
                    let (Some(trace), Some(series), Some(id)) = (project.trace(track.trace), &track.series, &track.phenomenology_id) else {
                        continue;
                    };
                    let values = match track.kind {
                        SampleKind::Real => series.values.iter().collect(),
                        SampleKind::Complex => series.samples.series(IqComponent::Magnitude).iter().collect(),
                    };
                    let spec = windows.specs.get(id).copied().unwrap_or_else(|| default_detector(id.as_str(), track.rate));
                    let input = DetectInput::new(project, trace, values, spec, windows.associate.radius_m);
                    inputs.extend(input.map(|i| if windows.matched { i.with_templates(project, windows.matched_spec) } else { i }));
                }
                let settings = windows.associate;
                let templates: Vec<Id> =
                    if inputs.iter().any(|i| !i.templates.is_empty()) { project.templates().iter().map(|t| t.id).collect() } else { Vec::new() };
                let key = format!("{settings:?} {:?} {templates:?} {:?}", inputs.iter().map(|i| i.spec).collect::<Vec<_>>(), windows.matched_spec);
                let job = DetectJob {
                    meta: inputs.iter().map(DetectMeta::from).collect(),
                    reference: project.reference.expect("checked"),
                    hash: config_hash(&key),
                    replace: windows.replace,
                };
                let task = AsyncComputeTaskPool::get().spawn(async move { run_detection(&inputs, &settings, CHUNK) });
                windows.result = None;
                windows.task = Some((job, task));
            }
            if busy {
                ui.spinner();
                ui.label("detecting…");
            }
        });

        if let Some((job, outcome)) = &windows.result {
            ui.separator();
            for (trace, error) in &outcome.errors {
                let code = project.trace(*trace).and_then(|t| project.sensor(t.sensor)).map_or("?", |s| s.code.as_str());
                ui.colored_label(WARN_COLOR, format!("{code}: {error}"));
            }
            let located = outcome.candidates.iter().filter(|c| c.fit.is_some()).count();
            let associated: usize = outcome.candidates.iter().map(|c| c.arrivals.len() + c.absorbed.len()).sum();
            ui.label(format!(
                "{} triggers → {} events ({located} located); {} triggers not associated.",
                outcome.detections.len(),
                outcome.candidates.len(),
                outcome.detections.len() - associated,
            ));
            egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                egui::Grid::new("candidates").striped(true).num_columns(4).show(ui, |ui| {
                    for heading in ["First arrival", "Members", "Origin", "RMS"] {
                        ui.strong(heading);
                    }
                    ui.end_row();
                    for c in &outcome.candidates {
                        let first = outcome.detections[c.arrivals[0]].at;
                        ui.monospace(format!("{first:+.3} s"));
                        ui.label(c.arrivals.len().to_string());
                        match &c.fit {
                            Some(fit) => {
                                ui.monospace(format!("{:.4}°, {:.4}°, {:.1} km, {:+.3} s", fit.source.latitude, fit.source.longitude, fit.source.depth_m / 1e3, fit.origin));
                                ui.monospace(format!("{:.2e} s", fit.rms_s));
                            }
                            None => {
                                ui.weak("not located");
                                ui.label("");
                            }
                        }
                        ui.end_row();
                    }
                });
            });
            let replaced = if job.replace { project.events().iter().filter(|e| Project::replaceable(e)).count() } else { 0 };
            ui.horizontal(|ui| {
                let n = outcome.candidates.len();
                let label = format!("Apply {n} event{}", if n == 1 { "" } else { "s" });
                if ui.add_enabled(n > 0 || replaced > 0, theme::primary(label)).clicked() {
                    let changes = event_changes(project, &job.meta, outcome, job.reference, job.replace, &job.hash);
                    apply = Some((format!("Detect {n} events"), changes));
                }
                if replaced > 0 {
                    ui.weak(format!("replaces {replaced} automatic event{}", if replaced == 1 { "" } else { "s" }));
                }
            });
        }
    });
    if let Some((label, changes)) = apply
        && workspace.execute_or_report(label, changes)
    {
        windows.result = None;
    }
    windows.detect_open = open;
}

fn status_color(status: EventStatus) -> egui::Color32 {
    match status {
        EventStatus::Auto => egui::Color32::from_gray(200),
        EventStatus::Reviewed => GOOD_COLOR,
        EventStatus::Rejected => REJECT_COLOR,
    }
}

fn catalogue_window(
    ctx: &egui::Context,
    windows: &mut EventWindows,
    workspace: &mut Workspace,
    view: &mut TrackView,
    tracks: &mut Query<&mut Track>,
    width: f64,
) {
    if !windows.catalogue_open {
        return;
    }
    let mut open = true;
    let mut changes: Option<(String, Vec<Change>)> = None;
    let mut review: Option<Id> = None;
    theme::tool_window(ctx, "Event Catalogue").open(&mut open).default_width(760.0).show(ctx, |ui| {
        let project = &workspace.project;
        windows.merge.retain(|id| project.event(*id).is_some());
        ui.horizontal(|ui| {
            ui.label("Show");
            egui::ComboBox::from_id_salt("status_filter")
                .selected_text(windows.filter.map_or("All", |s| s.label()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut windows.filter, None, "All");
                    for status in EventStatus::ALL {
                        ui.selectable_value(&mut windows.filter, Some(status), status.label());
                    }
                });
            ui.toggle_value(&mut windows.queue, "Queue").on_hover_text("Automatic events, most uncertain first; ] / [ follow it");
            let counts = EventStatus::ALL.map(|s| project.events().iter().filter(|e| e.status == s).count());
            ui.weak(format!("{} automatic · {} reviewed · {} rejected", counts[0], counts[1], counts[2]));
        });
        let events: Vec<&Event> = review_order(project, windows.queue)
            .into_iter()
            .filter_map(|id| project.event(id))
            .filter(|e| windows.queue || windows.filter.is_none_or(|s| e.status == s))
            .collect();
        if events.is_empty() {
            ui.label("No events. Use Events ▸ Detect…, or ⌘/Ctrl+drag across the tracks.");
        }
        let truth = &workspace.truth_events;
        egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
            egui::Grid::new("catalogue").striped(true).num_columns(if truth.is_empty() { 10 } else { 11 }).show(ui, |ui| {
                for heading in ["", "Time", "Duration", "Members", "Phenomenologies", "Origin", "Predicted", "Label", "Status", ""] {
                    ui.strong(heading);
                }
                if !truth.is_empty() {
                    ui.strong("vs truth");
                }
                ui.end_row();
                for event in events {
                    let mut merge = windows.merge.contains(&event.id);
                    if ui.checkbox(&mut merge, "").on_hover_text("Mark for merging").changed() {
                        if merge {
                            windows.merge.push(event.id);
                        } else {
                            windows.merge.retain(|id| *id != event.id);
                        }
                    }
                    let reviewing = view.event == Some(event.id);
                    let time = match project.reference {
                        Some(r) => format!("{:+.3} s", event.time().seconds_since(r)),
                        None => event.time().to_string(),
                    };
                    let time = if reviewing { egui::RichText::new(format!("▶ {time}")).strong() } else { egui::RichText::new(time) };
                    if ui.add(egui::Label::new(time.monospace()).sense(egui::Sense::click())).on_hover_text("Review").clicked() {
                        review = Some(event.id);
                    }
                    ui.monospace(format!("{:.3} s", event.duration()));
                    ui.label(event.members.len().to_string());
                    ui.horizontal(|ui| {
                        let mut seen: Vec<&PhenomenologyId> = Vec::new();
                        for member in &event.members {
                            let Some(p) = project.trace(member.trace).and_then(|t| project.trace_phenomenology(t)) else {
                                continue;
                            };
                            if !seen.contains(&&p.id) {
                                seen.push(&p.id);
                                let c = p.color;
                                let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
                                ui.colored_label(egui::Color32::from_rgb(byte(c.red), byte(c.green), byte(c.blue)), &p.name);
                            }
                        }
                    });
                    match &event.origin {
                        Some(o) => ui.monospace(format!("{:.4}°, {:.4}°, {:.1} km", o.latitude, o.longitude, o.depth_m / 1e3)),
                        None => ui.weak("—"),
                    };
                    match project.consensus(event).first() {
                        Some((label, p)) => {
                            let name = project.taxonomy().get(label).map_or(label.as_str(), |l| l.name.as_str());
                            ui.label(format!("{name} {:.0}%", p * 100.0))
                                .on_hover_text(format!("uncertainty {:.2} over {} model(s)", project.uncertainty(event), event.predictions.len()))
                        }
                        None => ui.weak("—"),
                    };
                    if let Some(label) = label_picker(ui, ("catalogue_label", event.id), project, event.label.as_ref())
                        && let Some(change) = label_change(project, event.id, label)
                    {
                        changes = Some(("Label event".to_string(), vec![change]));
                    }
                    ui.colored_label(status_color(event.status), event.status.label());
                    ui.horizontal(|ui| {
                        if ui.small_button("Review").clicked() {
                            review = Some(event.id);
                        }
                        let accept = accept_change(project, event.id);
                        if ui.add_enabled(accept.is_some(), egui::Button::new("✔").small()).on_hover_text("Accept, taking the predicted label when unlabelled (A)").clicked()
                            && let Some(change) = accept
                        {
                            changes = Some(("Accept event".to_string(), vec![change]));
                        }
                        if theme::danger(ui, event.status != EventStatus::Rejected, egui::Button::new("✘").small()).on_hover_text("Reject as noise (R)").clicked()
                            && let Some(change) = status_change(project, event.id, EventStatus::Rejected)
                        {
                            changes = Some(("Reject event".to_string(), vec![change]));
                        }
                        if ui.small_button("🗑").on_hover_text("Delete the event and its picks").clicked() {
                            changes = Some(("Delete event".to_string(), vec![Change::RemoveEvent { event: event.id }]));
                        }
                    });
                    if !truth.is_empty() {
                        match nearest_truth(event, truth) {
                            Some((distance, dt, i)) => {
                                let class = workspace.truth_labels.get(i).map_or(String::new(), |l| format!(", {l}"));
                                ui.monospace(format!("{:.0} m, {dt:+.3} s{class}", distance))
                            }
                            None => ui.weak("—"),
                        };
                    }
                    ui.end_row();
                }
            });
        });
        ui.separator();
        ui.horizontal(|ui| {
            let n = windows.merge.len();
            if ui.add_enabled(n >= 2, egui::Button::new(format!("Merge {n}"))).clicked()
                && let Some(merged) = merge_changes(project, &windows.merge)
            {
                changes = Some((format!("Merge {n} events"), merged));
            }
            if ui.add_enabled(view.event.is_some(), egui::Button::new("Close review")).clicked() {
                view.event = None;
            }
            ui.weak("While reviewing: ] / [ next and previous, A accept, R reject, 1–9 label, Esc close.");
        });
    });
    if let Some((label, changes)) = changes
        && workspace.execute_or_report(label, changes)
    {
        windows.merge.clear();
    }
    if let Some(id) = review {
        open_event(view, tracks, &workspace.project, id, width);
    }
    windows.catalogue_open = open;
}

/// Distance and origin-time error to the nearest known event, and its index, for located events.
fn nearest_truth(event: &Event, truth: &[(squiggle_core::locate::Source, Timestamp)]) -> Option<(f64, f64, usize)> {
    let origin = event.origin.as_ref()?;
    truth
        .iter()
        .enumerate()
        .map(|(i, (s, t))| (surface_distance(origin.latitude, origin.longitude, s.latitude, s.longitude), origin.time.seconds_since(*t), i))
        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;
    use crate::workspace::{SampleState, SyntheticEvent};

    fn inputs(workspace: &Workspace) -> Vec<DetectInput> {
        let project = &workspace.project;
        project
            .traces()
            .iter()
            .filter_map(|trace| {
                let Some(SampleState::Loaded(samples)) = workspace.samples.get(&trace.id) else { return None };
                let values = samples.series(IqComponent::Magnitude).iter().collect();
                let id = project.trace_phenomenology(trace)?.id.clone();
                DetectInput::new(project, trace, values, default_detector(id.as_str(), trace.sample_rate), AssociateSettings::default().radius_m)
            })
            .collect()
    }

    /// M5 self-check: the swarm's detectable events are found, nothing else,
    /// and chunked and whole runs agree exactly.
    #[test]
    fn the_swarm_is_detected_at_known_recall_and_precision() {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Swarm).unwrap();
        let inputs = inputs(&workspace);
        let settings = AssociateSettings::default();
        let outcome = run_detection(&inputs, &settings, CHUNK);
        for chunk in [1, 997, usize::MAX] {
            assert_eq!(run_detection(&inputs, &settings, chunk), outcome, "chunk {chunk}");
        }

        let meta: Vec<DetectMeta> = inputs.iter().map(DetectMeta::from).collect();
        let reference = workspace.project.reference.unwrap();
        let changes = event_changes(&workspace.project, &meta, &outcome, reference, true, "test");
        workspace.execute("detect", changes).unwrap();
        let events = workspace.project.events();
        let truth = &workspace.truth_events;
        let mut found = vec![false; truth.len()];
        let mut false_events = 0;
        for event in events {
            match nearest_truth(event, truth) {
                Some((distance, dt, _)) if distance < 200.0 && dt.abs() < 0.05 => {
                    let i = truth.iter().position(|(_, t)| event.origin.as_ref().unwrap().time.seconds_since(*t).abs() < 0.05).unwrap();
                    found[i] = true;
                    assert_eq!(event.members.len(), 8);
                }
                _ => false_events += 1,
            }
        }
        let recall = found.iter().filter(|f| **f).count();
        assert_eq!((recall, false_events), (synthetic::SWARM_DETECTABLE, 0), "{events:#?}");
        assert!(!found[3], "the weak event should be missed");
        assert_eq!(workspace.project.picks().len(), 8 * synthetic::SWARM_DETECTABLE);

        // A re-run replaces automatic events, keeps reviewed ones and does not detect them again.
        let kept = events[0].id;
        let accept = status_change(&workspace.project, kept, EventStatus::Reviewed).unwrap();
        workspace.execute("accept", vec![accept]).unwrap();
        let changes = event_changes(&workspace.project, &meta, &outcome, reference, true, "test");
        workspace.execute("again", changes).unwrap();
        assert_eq!(workspace.project.events().len(), synthetic::SWARM_DETECTABLE);
        assert!(workspace.project.event(kept).is_some());
        let changes = event_changes(&workspace.project, &meta, &outcome, reference, false, "test");
        assert!(changes.is_empty(), "adding finds nothing new");
    }

    fn event_with_picks() -> (Workspace, Id, Vec<Id>) {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Swarm).unwrap();
        let reference = workspace.project.reference.unwrap();
        let traces: Vec<Id> = workspace.project.traces().iter().map(|t| t.id).collect();
        let event = new_event(&workspace.analyst, reference.add_seconds(10.0), reference.add_seconds(40.0), &traces[..3]);
        let id = event.id;
        let pick = |trace: Id, at: f64| Change::SetPick {
            pick: Pick {
                id: Id::generate(),
                trace,
                time: reference.add_seconds(at),
                phase: "P".into(),
                uncertainty_s: 0.01,
                provenance: Provenance::Manual { analyst: Analyst("ana".into()), at: Timestamp::now() },
                event: Some(id),
            },
            revert: false,
        };
        workspace.execute("make", vec![Change::AddEvent(event), pick(traces[0], 14.0), pick(traces[1], 34.0)]).unwrap();
        (workspace, id, traces)
    }

    #[test]
    fn splitting_moves_later_picks_and_undoes() {
        let (mut workspace, id, traces) = event_with_picks();
        let at = workspace.project.reference.unwrap().add_seconds(25.0);
        let changes = split_changes(&workspace.project, id, at).unwrap();
        workspace.execute("split", changes).unwrap();
        let project = &workspace.project;
        assert_eq!(project.events().len(), 2);
        let right = project.events().iter().find(|e| e.id != id).unwrap();
        assert_eq!(project.event(id).unwrap().end, at);
        assert_eq!(project.event_picks(Some(right.id), traces[1]).count(), 1);
        assert_eq!(project.event_picks(Some(id), traces[0]).count(), 1);
        workspace.undo();
        assert_eq!(workspace.project.events().len(), 1);
        assert_eq!(workspace.project.event_picks(Some(id), traces[1]).count(), 1);
    }

    #[test]
    fn merging_unites_windows_members_and_picks() {
        let (mut workspace, id, traces) = event_with_picks();
        let reference = workspace.project.reference.unwrap();
        let other = new_event(&workspace.analyst, reference.add_seconds(50.0), reference.add_seconds(60.0), &traces[2..5]);
        let other_id = other.id;
        workspace.execute("other", vec![Change::AddEvent(other)]).unwrap();
        let changes = merge_changes(&workspace.project, &[other_id, id]).unwrap();
        workspace.execute("merge", changes).unwrap();
        let project = &workspace.project;
        let merged = project.event(id).unwrap();
        assert_eq!(project.events().len(), 1);
        assert_eq!(merged.members.len(), 5);
        assert_eq!(merged.end, reference.add_seconds(60.0));
        assert_eq!(project.picks().len(), 2);
    }

    #[test]
    fn edits_mark_automatic_events_reviewed() {
        let (mut workspace, id, traces) = event_with_picks();
        let auto = Event { status: EventStatus::Auto, ..workspace.project.event(id).unwrap().clone() };
        workspace.execute("auto", vec![Change::SetEvent(auto)]).unwrap();
        let change = member_change(&workspace.project, id, traces[5], true).unwrap();
        workspace.execute("member", vec![change]).unwrap();
        let event = workspace.project.event(id).unwrap();
        assert_eq!((event.members.len(), event.status), (4, EventStatus::Reviewed));
        assert!(member_change(&workspace.project, id, traces[5], true).is_none());
    }
}
