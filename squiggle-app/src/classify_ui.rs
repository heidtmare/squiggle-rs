//! Classification: event features, the rules classifier, labels and the
//! taxonomy, the review queue, and training-dataset export.
//!
//! Features use each track's plotted (processed) series, as detection does.
//! Classifying records one prediction per model on each event and never
//! changes its status or label. Labelling, or accepting a prediction, marks
//! the event reviewed.

use std::collections::HashMap;
use std::path::PathBuf;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::classify::{RULES_MODEL, RuleSettings, Verdict, rules};
use squiggle_core::features::{EVENT_FEATURES, EventData, FEATURE_SET, FeatureSettings, Features, MEMBER_FEATURES, OriginData, TraceData, features};
use squiggle_core::taxonomy::NOISE;
use squiggle_core::{Change, Event, EventStatus, Id, IqComponent, Label, LabelId, Project, SampleKind, Timestamp, Trace};
use squiggle_io::dataset::{FORMAT, Manifest, Table, write_dataset};

use crate::process_ui::config_hash;
use crate::tracks::{Track, TrackView};
use crate::workspace::Workspace;

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
const EXPLOSIVE_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 140, 70);
const QUAKE_COLOR: egui::Color32 = egui::Color32::from_rgb(110, 170, 240);

/// Every assigned trace with values, as features need it.
pub fn trace_data(project: &Project, values: impl Fn(&Trace) -> Option<Vec<f32>>) -> Vec<TraceData> {
    let Some(reference) = project.reference else {
        return Vec::new();
    };
    project
        .traces()
        .iter()
        .filter_map(|trace| {
            let phenomenology = project.trace_phenomenology(trace)?;
            Some(TraceData {
                trace: trace.id,
                sensor: trace.sensor,
                phenomenology: phenomenology.id.clone(),
                propagation: phenomenology.propagation.clone(),
                values: values(trace)?,
                rate: trace.sample_rate,
                start: project.corrected_start(trace).seconds_since(reference),
                location: project.sensor(trace.sensor).and_then(|s| s.location),
            })
        })
        .collect()
}

/// Plotted values of every loaded track.
pub(crate) fn track_values(tracks: &Query<&Track>) -> HashMap<Id, Vec<f32>> {
    tracks
        .iter()
        .filter_map(|track| {
            let series = track.series.as_ref()?;
            let values = match track.kind {
                SampleKind::Real => series.values.iter().collect(),
                SampleKind::Complex => series.samples.series(IqComponent::Magnitude).iter().collect(),
            };
            Some((track.trace, values))
        })
        .collect()
}

/// `event` over `traces`; members without data are left out.
pub fn event_data(project: &Project, event: &Event, traces: &[TraceData]) -> Option<EventData> {
    let reference = project.reference?;
    let index = |id: Id| traces.iter().position(|t| t.trace == id);
    Some(EventData {
        start: event.start.seconds_since(reference),
        end: event.end.seconds_since(reference),
        origin: event.origin.as_ref().map(|o| OriginData { source: o.source(), time: o.time.seconds_since(reference), rms_s: o.rms_s }),
        members: event.members.iter().filter_map(|m| index(m.trace)).collect(),
        picks: project
            .picks()
            .iter()
            .filter(|p| p.event == Some(event.id))
            .filter_map(|p| Some((index(p.trace)?, p.phase.clone(), project.pick_time(p)?.seconds_since(reference))))
            .collect(),
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClassifySettings {
    pub features: FeatureSettings,
    pub rules: RuleSettings,
}

impl ClassifySettings {
    pub fn hash(&self) -> String {
        config_hash(&format!("{FEATURE_SET} {RULES_MODEL} {self:?}"))
    }
}

#[derive(Clone, Debug)]
pub struct Classified {
    pub event: Id,
    pub data: EventData,
    pub features: Features,
    pub verdict: Verdict,
}

pub fn classify_one(traces: &[TraceData], event: Id, data: EventData, settings: &ClassifySettings) -> Classified {
    let features = features(traces, &data, &settings.features);
    let verdict = rules(&features.event, &settings.rules);
    Classified { event, data, features, verdict }
}

pub fn classify(traces: &[TraceData], events: Vec<(Id, EventData)>, settings: &ClassifySettings) -> Vec<Classified> {
    events.into_iter().map(|(id, data)| classify_one(traces, id, data, settings)).collect()
}

/// Every event in `project` over `traces`.
pub fn all_events(project: &Project, traces: &[TraceData]) -> Vec<(Id, EventData)> {
    project.events().iter().filter_map(|e| Some((e.id, event_data(project, e, traces)?))).collect()
}

/// Records each result's rules prediction, replacing that model's previous one.
pub fn prediction_changes(project: &Project, results: &[Classified], hash: &str) -> Vec<Change> {
    results
        .iter()
        .filter_map(|r| {
            let event = project.event(r.event)?;
            let predictions = event.with_prediction(r.verdict.prediction(hash, project.taxonomy().version));
            (predictions != event.predictions).then_some(Change::SetPredictions { event: r.event, predictions })
        })
        .collect()
}

/// Labels `id` (or clears its label) and marks it reviewed.
pub fn label_change(project: &Project, id: Id, label: Option<LabelId>) -> Option<Change> {
    let event = project.event(id)?;
    (event.label != label || event.status != EventStatus::Reviewed)
        .then(|| Change::SetEvent(Event { label, status: EventStatus::Reviewed, ..event.clone() }))
}

/// Accepts `id`: reviewed, and labelled with the consensus prediction when unlabelled.
pub fn accept_change(project: &Project, id: Id) -> Option<Change> {
    let event = project.event(id)?;
    let label = event
        .label
        .clone()
        .or_else(|| event.consensus().first().map(|l| l.0.clone()).filter(|l| project.taxonomy().get(l).is_some()));
    label_change(project, id, label)
}

/// Review order: by time, or in queue mode the automatic events, most uncertain first.
pub fn review_order(project: &Project, queue: bool) -> Vec<Id> {
    let mut events: Vec<&Event> = project.events().iter().filter(|e| !queue || e.status == EventStatus::Auto).collect();
    events.sort_by(|a, b| {
        let by_time = a.time().seconds_since(b.time()).total_cmp(&0.0);
        if queue { b.uncertainty().total_cmp(&a.uncertainty()).then(by_time) } else { by_time }
    });
    events.into_iter().map(|e| e.id).collect()
}

/// The class an event trains as: noise when rejected, else its analyst label.
pub fn training_class(event: &Event) -> Option<String> {
    match event.status {
        EventStatus::Rejected => Some(NOISE.to_string()),
        _ => event.label.as_ref().map(|l| l.0.clone()),
    }
}

/// An exported event: its result, the event and its training class.
type Row<'a> = (&'a Classified, &'a Event, Option<String>);

/// Events and members tables and the manifest for `results`.
pub fn dataset(project: &Project, traces: &[TraceData], results: &[Classified], include_unlabelled: bool, analyst: &str) -> (Table, Table, Manifest) {
    let taxonomy = project.taxonomy();
    let rows: Vec<Row> = results
        .iter()
        .filter_map(|r| {
            let event = project.event(r.event)?;
            let class = training_class(event);
            (class.is_some() || include_unlabelled).then_some((r, event, class))
        })
        .collect();

    let text = |f: &dyn Fn(&Row) -> Option<String>| rows.iter().map(f).collect::<Vec<_>>();
    let mut events = Table::default();
    events.text("event", text(&|r| Some(r.1.id.to_string())));
    events.text("time", text(&|r| Some(r.1.time().to_string())));
    events.number("time_unix", rows.iter().map(|r| r.1.time().unix_millis() as f64 / 1e3).collect());
    events.text("status", text(&|r| Some(r.1.status.label().to_string())));
    events.text("class", text(&|r| r.2.clone()));
    events.text("class_path", text(&|r| {
        let label = r.1.label.as_ref().filter(|_| r.1.status != EventStatus::Rejected)?;
        let mut path: Vec<&str> = taxonomy.lineage(label).map(LabelId::as_str).collect();
        path.reverse();
        Some(path.join("/"))
    }));
    events.text("source", text(&|r| Some(match &r.1.provenance {
        squiggle_core::sensor::Provenance::Auto { stage, config_hash } => format!("{stage}:{config_hash}"),
        squiggle_core::sensor::Provenance::Manual { analyst, .. } => format!("manual:{}", analyst.0),
    })));
    events.text("predicted", text(&|r| r.0.verdict.labels.first().map(|l| l.0.0.clone())));
    events.number("predicted_p", rows.iter().map(|r| r.0.verdict.labels.first().map_or(f64::NAN, |l| f64::from(l.1))).collect());
    events.number("latitude", rows.iter().map(|r| r.1.origin.as_ref().map_or(f64::NAN, |o| o.latitude)).collect());
    events.number("longitude", rows.iter().map(|r| r.1.origin.as_ref().map_or(f64::NAN, |o| o.longitude)).collect());
    for (i, name) in EVENT_FEATURES.iter().enumerate() {
        events.number(name, rows.iter().map(|r| r.0.features.event[i]).collect());
    }

    let member_rows: Vec<(&Classified, &Event, usize)> =
        rows.iter().flat_map(|r| (0..r.0.data.members.len()).map(move |m| (r.0, r.1, m))).collect();
    let trace = |r: &(&Classified, &Event, usize)| &traces[r.0.data.members[r.2]];
    let mut members = Table::default();
    members.text("event", member_rows.iter().map(|r| Some(r.1.id.to_string())).collect());
    members.text("trace", member_rows.iter().map(|r| Some(trace(r).trace.to_string())).collect());
    members.text("sensor", member_rows.iter().map(|r| project.sensor(trace(r).sensor).map(|s| s.code.clone())).collect());
    members.text("phenomenology", member_rows.iter().map(|r| Some(trace(r).phenomenology.0.clone())).collect());
    members.number("rate", member_rows.iter().map(|r| trace(r).rate).collect());
    members.number(
        "onset_s",
        member_rows
            .iter()
            .map(|r| {
                let index = r.0.data.members[r.2];
                let zero = r.0.data.origin.map_or(r.0.data.start, |o| o.time);
                r.0.data.picks.iter().filter(|p| p.0 == index).map(|p| p.2 - zero).fold(f64::NAN, f64::min)
            })
            .collect(),
    );
    for (i, name) in MEMBER_FEATURES.iter().enumerate() {
        members.number(name, member_rows.iter().map(|r| r.0.features.members[r.2][i]).collect());
    }

    let mut classes: Vec<(Option<String>, usize)> = Vec::new();
    for (_, _, class) in &rows {
        match classes.iter_mut().find(|c| &c.0 == class) {
            Some(c) => c.1 += 1,
            None => classes.push((class.clone(), 1)),
        }
    }
    let manifest = Manifest {
        format: FORMAT,
        project: project.name.clone(),
        created: Timestamp::now().to_string(),
        analyst: analyst.to_string(),
        feature_set: FEATURE_SET.to_string(),
        taxonomy_version: taxonomy.version,
        labels: taxonomy.labels.clone(),
        noise_class: NOISE,
        event_features: EVENT_FEATURES.iter().map(|s| s.to_string()).collect(),
        member_features: MEMBER_FEATURES.iter().map(|s| s.to_string()).collect(),
        classes,
        events: 0,
        members: 0,
        hashes: Vec::new(),
    };
    (events, members, manifest)
}

struct ExportJob {
    project: Project,
    traces: Vec<TraceData>,
    settings: ClassifySettings,
    include_unlabelled: bool,
    analyst: String,
}

#[derive(Resource, Default)]
pub struct ClassifyWindows {
    pub classify_open: bool,
    pub features_open: bool,
    pub labels_open: bool,
    pub export_open: bool,
    settings: ClassifySettings,
    task: Option<(String, Task<Vec<Classified>>)>,
    /// Event, revision and settings behind the features panel.
    panel: Option<(Id, u64, ClassifySettings, Classified)>,
    /// Sensor of each trace the panel's result indexes.
    panel_sensors: Vec<Id>,
    new_id: String,
    new_name: String,
    new_parent: Option<LabelId>,
    include_unlabelled: bool,
    folder: Option<Task<Option<PathBuf>>>,
    export: Option<Task<Result<(PathBuf, Manifest), String>>>,
    exported: Option<Result<(PathBuf, Manifest), String>>,
}

pub fn classify_menu(ui: &mut egui::Ui, windows: &mut ClassifyWindows) {
    for (label, tip, open) in [
        ("Classify…", "Features and the rules classifier on every event", &mut windows.classify_open),
        ("Features…", "Features, discriminants and predictions of the event under review", &mut windows.features_open),
        ("Labels…", "The taxonomy: add subtypes, rename, remove", &mut windows.labels_open),
        ("Export dataset…", "Labelled events and rejected noise as Parquet, for offline training", &mut windows.export_open),
    ] {
        if ui.button(label).on_hover_text(tip).clicked() {
            *open = true;
            ui.close();
        }
    }
}

pub fn windows(
    mut contexts: EguiContexts,
    mut workspace: ResMut<Workspace>,
    view: Res<TrackView>,
    tracks: Query<&Track>,
    mut windows: ResMut<ClassifyWindows>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    if let Some((_, task)) = &mut windows.task
        && let Some(results) = check_ready(task)
    {
        let (hash, _) = windows.task.take().expect("checked above");
        let changes = prediction_changes(&workspace.project, &results, &hash);
        if !changes.is_empty() {
            workspace.execute_or_report(format!("Classify {} events", changes.len()), changes);
        }
    }
    poll_export(windows, &workspace, &tracks);
    classify_window(&ctx, windows, &workspace, &tracks);
    features_window(&ctx, windows, &mut workspace, &view, &tracks);
    labels_window(&ctx, windows, &mut workspace);
    export_window(&ctx, windows, &workspace);
    Ok(())
}

fn drag(ui: &mut egui::Ui, label: &str, value: &mut f64, speed: f64, suffix: &str) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).speed(speed).suffix(suffix));
    ui.end_row();
}

fn classify_window(ctx: &egui::Context, windows: &mut ClassifyWindows, workspace: &Workspace, tracks: &Query<&Track>) {
    if !windows.classify_open {
        return;
    }
    let mut open = true;
    egui::Window::new("Classify Events").open(&mut open).default_width(420.0).show(ctx, |ui| {
        ui.weak("Features from each member's plotted series, then the rules classifier. Predictions are recorded per event and never change its status or label.");
        let s = &mut windows.settings;
        egui::CollapsingHeader::new("Rules").default_open(true).show(ui, |ui| {
            egui::Grid::new("rules").num_columns(2).show(ui, |ui| {
                let r = &mut s.rules;
                drag(ui, "log₁₀ P/S above", &mut r.p_s_log, 0.01, "");
                drag(ui, "  scale", &mut r.p_s_scale, 0.01, "");
                drag(ui, "log₁₀ high/low above", &mut r.hf_lf_log, 0.01, "");
                drag(ui, "  scale", &mut r.hf_lf_scale, 0.01, "");
                let mut km = r.shallow_m / 1e3;
                drag(ui, "Shallower than", &mut km, 0.05, " km");
                r.shallow_m = km * 1e3;
                drag(ui, "First motion above", &mut r.first_motion, 0.01, "");
                drag(ui, "Coupled at", &mut r.coupled_db, 0.5, " dB");
            });
        });
        egui::CollapsingHeader::new("Features").show(ui, |ui| {
            egui::Grid::new("feature_settings").num_columns(2).show(ui, |ui| {
                let f = &mut s.features;
                let b = &mut f.seismic_bands;
                drag(ui, "Seismic low band from", &mut b.low.0, 0.1, " Hz");
                drag(ui, "  to", &mut b.low.1, 0.1, " Hz");
                drag(ui, "Seismic high band from", &mut b.high.0, 0.1, " Hz");
                drag(ui, "  to", &mut b.high.1, 0.1, " Hz");
                drag(ui, "P and S windows up to", &mut f.phase_window_s, 0.05, " s");
                let mut km = f.coupling_tolerance_m / 1e3;
                drag(ui, "Coupling search ±", &mut km, 0.1, " km");
                f.coupling_tolerance_m = km * 1e3;
            });
            if ui.small_button("Defaults").clicked() {
                *s = ClassifySettings::default();
            }
        });
        let project = &workspace.project;
        let n = project.events().len();
        ui.horizontal(|ui| {
            let busy = windows.task.is_some();
            if ui.add_enabled(n > 0 && !busy, egui::Button::new(format!("Classify {n} events"))).clicked() {
                let values = track_values(tracks);
                let traces = trace_data(project, |t| values.get(&t.id).cloned());
                let events = all_events(project, &traces);
                let settings = windows.settings;
                let task = AsyncComputeTaskPool::get().spawn(async move { classify(&traces, events, &settings) });
                windows.task = Some((settings.hash(), task));
            }
            if busy {
                ui.spinner();
            }
        });
        let mut counts: Vec<(String, usize)> = Vec::new();
        for event in project.events() {
            if let Some(prediction) = event.prediction(RULES_MODEL)
                && let Some((label, _)) = prediction.top()
            {
                match counts.iter_mut().find(|c| c.0 == label.0) {
                    Some(c) => c.1 += 1,
                    None => counts.push((label.0.clone(), 1)),
                }
            }
        }
        if !counts.is_empty() {
            let summary: Vec<String> = counts.iter().map(|(l, n)| format!("{n} {l}")).collect();
            ui.label(format!("Rules predict: {}.", summary.join(", ")));
        }
    });
    windows.classify_open = open;
}

fn format_value(value: f64) -> String {
    if value.is_nan() {
        "—".to_string()
    } else if value != 0.0 && (value.abs() >= 1e5 || value.abs() < 1e-3) {
        format!("{value:.3e}")
    } else {
        format!("{value:.3}")
    }
}

fn label_color(label: &str) -> egui::Color32 {
    match label {
        squiggle_core::taxonomy::EXPLOSION => EXPLOSIVE_COLOR,
        squiggle_core::taxonomy::EARTHQUAKE => QUAKE_COLOR,
        _ => egui::Color32::from_gray(190),
    }
}

/// Probabilities as labelled bars.
fn probability_bars(ui: &mut egui::Ui, project: &Project, labels: &[(LabelId, f32)]) {
    for (label, p) in labels {
        ui.horizontal(|ui| {
            let name = project.taxonomy().get(label).map_or(label.0.clone(), |l| l.name.clone());
            ui.add_sized([90.0, 16.0], egui::Label::new(name));
            ui.add(egui::ProgressBar::new(*p).desired_width(160.0).fill(label_color(&label.0).gamma_multiply(0.8)).text(format!("{:.0}%", p * 100.0)));
        });
    }
}

/// The label picker shared by the panel and the catalogue: `Some` when changed.
pub fn label_picker(ui: &mut egui::Ui, salt: impl std::hash::Hash + std::fmt::Debug, project: &Project, current: Option<&LabelId>) -> Option<Option<LabelId>> {
    let taxonomy = project.taxonomy();
    let text = current.map_or("—".to_string(), |l| taxonomy.path(l));
    let mut chosen = None;
    egui::ComboBox::from_id_salt(salt).selected_text(text).show_ui(ui, |ui| {
        if ui.selectable_label(current.is_none(), "—").clicked() {
            chosen = Some(None);
        }
        for (i, label) in taxonomy.ordered().into_iter().enumerate() {
            let depth = taxonomy.lineage(&label.id).count() - 1;
            let key = if i < 9 { format!("{}  ", i + 1) } else { "   ".to_string() };
            let text = format!("{key}{}{}", "   ".repeat(depth), label.name);
            if ui.selectable_label(current == Some(&label.id), text).clicked() {
                chosen = Some(Some(label.id.clone()));
            }
        }
    });
    chosen
}

fn features_window(ctx: &egui::Context, windows: &mut ClassifyWindows, workspace: &mut Workspace, view: &TrackView, tracks: &Query<&Track>) {
    if !windows.features_open {
        return;
    }
    let Some(id) = view.event.filter(|id| workspace.project.event(*id).is_some()) else {
        let mut open = true;
        egui::Window::new("Event Features").open(&mut open).show(ctx, |ui| {
            ui.label("Review an event (Events ▸ Catalogue) to see its features.");
        });
        windows.features_open = open;
        return;
    };
    let revision = workspace.revision();
    let stale = !matches!(&windows.panel, Some((e, r, s, _)) if *e == id && *r == revision && *s == windows.settings);
    if stale {
        let values = track_values(tracks);
        let project = &workspace.project;
        let event = project.event(id).expect("checked above");
        let member_traces: Vec<Id> = event.members.iter().map(|m| m.trace).collect();
        // Members, plus every air and EM trace for coupled arrivals.
        let traces = trace_data(project, |t| {
            let wanted = member_traces.contains(&t.id) || project.trace_phenomenology(t).is_some_and(|p| matches!(p.propagation, squiggle_core::Propagation::Constant { .. }));
            if wanted { values.get(&t.id).cloned() } else { None }
        });
        windows.panel = event_data(project, event, &traces).map(|data| (id, revision, windows.settings, classify_one(&traces, id, data, &windows.settings)));
        windows.panel_sensors = traces.iter().map(|t| t.sensor).collect();
    }
    let mut open = true;
    let mut change: Option<(String, Change)> = None;
    egui::Window::new("Event Features").open(&mut open).default_width(560.0).show(ctx, |ui| {
        let project = &workspace.project;
        let event = project.event(id).expect("checked above");
        let Some((.., result)) = &windows.panel else {
            ui.label("No reference time: features need one.");
            return;
        };
        ui.horizontal(|ui| {
            ui.strong(match project.reference {
                Some(r) => format!("Event at {:+.3} s", event.time().seconds_since(r)),
                None => format!("Event at {}", event.time()),
            });
            ui.label(event.status.label());
            ui.separator();
            ui.label("Label");
            if let Some(label) = label_picker(ui, ("panel_label", id), project, event.label.as_ref())
                && let Some(c) = label_change(project, id, label)
            {
                change = Some(("Label event".to_string(), c));
            }
            if ui.button("Accept").on_hover_text("Reviewed; takes the predicted label when unlabelled (A)").clicked()
                && let Some(c) = accept_change(project, id)
            {
                change = Some(("Accept event".to_string(), c));
            }
        });
        ui.separator();
        ui.columns(2, |columns| {
            let ui = &mut columns[0];
            ui.strong("Rules (computed now)");
            probability_bars(ui, project, &result.verdict.labels);
            for prediction in &event.predictions {
                ui.add_space(4.0);
                let stale = prediction.taxonomy_version != project.taxonomy().version;
                ui.horizontal(|ui| {
                    ui.strong(format!("Recorded: {}", prediction.model));
                    if stale {
                        ui.colored_label(WARN_COLOR, "⚠").on_hover_text("Predicted under an older taxonomy version");
                    }
                });
                probability_bars(ui, project, &prediction.labels);
            }
            let ui = &mut columns[1];
            ui.strong("Evidence (log-odds toward explosion)");
            if result.verdict.evidence.is_empty() {
                ui.weak("No discriminant available: needs seismic P and S, a location, or air/EM sensors.");
            }
            egui::Grid::new("evidence").num_columns(3).striped(true).show(ui, |ui| {
                for e in &result.verdict.evidence {
                    ui.label(e.reason).on_hover_text(e.feature);
                    ui.monospace(format_value(e.value));
                    let color = if e.weight >= 0.0 { EXPLOSIVE_COLOR } else { QUAKE_COLOR };
                    ui.colored_label(color, format!("{:+.2}", e.weight));
                    ui.end_row();
                }
            });
        });
        ui.separator();
        egui::CollapsingHeader::new("Event features").default_open(true).show(ui, |ui| {
            egui::Grid::new("event_features").num_columns(4).striped(true).show(ui, |ui| {
                for (i, chunk) in EVENT_FEATURES.chunks(2).enumerate() {
                    for (j, name) in chunk.iter().enumerate() {
                        ui.label(*name);
                        ui.monospace(format_value(result.features.event[2 * i + j]));
                    }
                    ui.end_row();
                }
            });
        });
        egui::CollapsingHeader::new("Member features").show(ui, |ui| {
            egui::ScrollArea::horizontal().show(ui, |ui| {
                egui::Grid::new("member_features").num_columns(MEMBER_FEATURES.len() + 1).striped(true).show(ui, |ui| {
                    ui.strong("Sensor");
                    for name in MEMBER_FEATURES {
                        ui.strong(name);
                    }
                    ui.end_row();
                    for (m, values) in result.features.members.iter().enumerate() {
                        let code = windows
                            .panel_sensors
                            .get(result.data.members[m])
                            .and_then(|sensor| project.sensor(*sensor))
                            .map_or("?", |s| s.code.as_str());
                        ui.label(code);
                        for v in values {
                            ui.monospace(format_value(*v));
                        }
                        ui.end_row();
                    }
                });
            });
        });
        ui.weak(FEATURE_SET);
    });
    if let Some((label, c)) = change {
        workspace.execute_or_report(label, vec![c]);
    }
    windows.features_open = open;
}

fn labels_window(ctx: &egui::Context, windows: &mut ClassifyWindows, workspace: &mut Workspace) {
    if !windows.labels_open {
        return;
    }
    let mut open = true;
    let mut change: Option<(String, Change)> = None;
    egui::Window::new("Labels").open(&mut open).default_width(420.0).show(ctx, |ui| {
        let project = &workspace.project;
        let taxonomy = project.taxonomy();
        ui.weak(format!("Taxonomy version {}. Keys 1–9 label the reviewed event in this order. Rejected events export as \"{NOISE}\".", taxonomy.version));
        egui::Grid::new("labels").num_columns(4).striped(true).show(ui, |ui| {
            for (i, label) in taxonomy.ordered().into_iter().enumerate() {
                ui.monospace(if i < 9 { (i + 1).to_string() } else { String::new() });
                let depth = taxonomy.lineage(&label.id).count() - 1;
                let mut name = label.name.clone();
                ui.horizontal(|ui| {
                    ui.add_space(12.0 * depth as f32);
                    let response = ui.add(egui::TextEdit::singleline(&mut name).desired_width(140.0));
                    if response.lost_focus() && !name.trim().is_empty() && name != label.name {
                        change = Some(("Rename label".to_string(), Change::DefineLabel(Label { name: name.trim().to_string(), ..label.clone() })));
                    }
                });
                ui.monospace(label.id.as_str());
                let used = project.events().iter().filter(|e| e.label.as_ref() == Some(&label.id)).count();
                let parent = taxonomy.labels.iter().any(|l| l.parent.as_ref() == Some(&label.id));
                let tip = if used > 0 { format!("{used} events carry it") } else if parent { "Has subtypes".to_string() } else { "Remove".to_string() };
                if ui.add_enabled(used == 0 && !parent, egui::Button::new("🗑").small()).on_hover_text(tip).clicked() {
                    change = Some(("Remove label".to_string(), Change::RemoveLabel { label: label.id.clone() }));
                }
                ui.end_row();
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut windows.new_name).hint_text("Name").desired_width(110.0));
            if windows.new_id.is_empty() || windows.new_id == slug(&windows.new_name[..windows.new_name.len().saturating_sub(1)]) {
                windows.new_id = slug(&windows.new_name);
            }
            ui.add(egui::TextEdit::singleline(&mut windows.new_id).hint_text("id").desired_width(90.0));
            let parent = windows.new_parent.as_ref().map_or("top level".to_string(), |p| taxonomy.path(p));
            egui::ComboBox::from_id_salt("new_parent").selected_text(parent).show_ui(ui, |ui| {
                ui.selectable_value(&mut windows.new_parent, None, "top level");
                for label in taxonomy.ordered() {
                    ui.selectable_value(&mut windows.new_parent, Some(label.id.clone()), taxonomy.path(&label.id));
                }
            });
        });
        let id = LabelId::new(windows.new_id.trim());
        let candidate = Label { id: id.clone(), name: windows.new_name.trim().to_string(), parent: windows.new_parent.clone(), expects: Vec::new() };
        let problem = if candidate.name.is_empty() {
            Some("needs a name".to_string())
        } else if taxonomy.get(&id).is_some() {
            Some(format!("{id} exists"))
        } else {
            taxonomy.check(&candidate).err()
        };
        ui.horizontal(|ui| {
            if ui.add_enabled(problem.is_none(), egui::Button::new("Add label")).clicked() {
                change = Some(("Add label".to_string(), Change::DefineLabel(candidate)));
                windows.new_id.clear();
                windows.new_name.clear();
            }
            if let Some(problem) = problem.filter(|_| !windows.new_name.is_empty()) {
                ui.colored_label(WARN_COLOR, problem);
            }
        });
    });
    if let Some((label, c)) = change {
        workspace.execute_or_report(label, vec![c]);
    }
    windows.labels_open = open;
}

/// Lowercase id from a name: `Quarry blast` → `quarry_blast`.
fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}

fn export_window(ctx: &egui::Context, windows: &mut ClassifyWindows, workspace: &Workspace) {
    if !windows.export_open {
        return;
    }
    let mut open = true;
    egui::Window::new("Export Dataset").open(&mut open).default_width(420.0).show(ctx, |ui| {
        let project = &workspace.project;
        let labelled = project.events().iter().filter(|e| training_class(e).is_some()).count();
        let rejected = project.events().iter().filter(|e| e.status == EventStatus::Rejected).count();
        ui.label(format!(
            "{labelled} events with a class ({rejected} rejected, exported as \"{NOISE}\"), of {}.",
            project.events().len()
        ));
        ui.weak("Writes events.parquet, members.parquet and dataset.json into a folder: features, classes, provenance and the taxonomy version.");
        ui.checkbox(&mut windows.include_unlabelled, "Include unlabelled events (class left empty)");
        let busy = windows.folder.is_some() || windows.export.is_some();
        ui.horizontal(|ui| {
            let n = if windows.include_unlabelled { project.events().len() } else { labelled };
            if ui.add_enabled(n > 0 && !busy, egui::Button::new(format!("Export {n} events…"))).clicked() {
                windows.exported = None;
                windows.folder = Some(IoTaskPool::get().spawn(async {
                    let folder = rfd::AsyncFileDialog::new().set_title("Export dataset to folder").pick_folder().await?;
                    Some(folder.path().to_path_buf())
                }));
            }
            if busy {
                ui.spinner();
            }
        });
        match &windows.exported {
            Some(Ok((path, manifest))) => {
                let classes: Vec<String> =
                    manifest.classes.iter().map(|(c, n)| format!("{n} {}", c.as_deref().unwrap_or("unlabelled"))).collect();
                ui.label(format!("Wrote {} events and {} members to {}: {}.", manifest.events, manifest.members, path.display(), classes.join(", ")));
            }
            Some(Err(error)) => {
                ui.colored_label(WARN_COLOR, error);
            }
            None => {}
        }
    });
    windows.export_open = open;
}

fn poll_export(windows: &mut ClassifyWindows, workspace: &Workspace, tracks: &Query<&Track>) {
    if let Some(task) = &mut windows.folder
        && let Some(folder) = check_ready(task)
    {
        windows.folder = None;
        if let Some(folder) = folder {
            let values = track_values(tracks);
            let job = ExportJob {
                project: workspace.project.clone(),
                traces: trace_data(&workspace.project, |t| values.get(&t.id).cloned()),
                settings: windows.settings,
                include_unlabelled: windows.include_unlabelled,
                analyst: workspace.analyst.0.clone(),
            };
            windows.export = Some(AsyncComputeTaskPool::get().spawn(async move {
                let events: Vec<(Id, EventData)> = all_events(&job.project, &job.traces)
                    .into_iter()
                    .filter(|(id, _)| job.include_unlabelled || job.project.event(*id).and_then(training_class).is_some())
                    .collect();
                let results = classify(&job.traces, events, &job.settings);
                let (events, members, manifest) = dataset(&job.project, &job.traces, &results, job.include_unlabelled, &job.analyst);
                write_dataset(&folder, &events, &members, manifest).map(|m| (folder, m))
            }));
        }
    }
    if let Some(task) = &mut windows.export
        && let Some(outcome) = check_ready(task)
    {
        windows.export = None;
        windows.exported = Some(outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events_ui::{CHUNK, DetectInput, DetectMeta, default_detector, event_changes, run_detection};
    use crate::synthetic;
    use crate::workspace::{SampleState, SyntheticEvent};
    use squiggle_core::associate::AssociateSettings;
    use squiggle_core::taxonomy::{EARTHQUAKE, EXPLOSION};
    use squiggle_core::{Analyst, Change};

    fn loaded(workspace: &Workspace, trace: &Trace) -> Option<Vec<f32>> {
        match workspace.samples.get(&trace.id) {
            Some(SampleState::Loaded(samples)) => Some(samples.series(IqComponent::Magnitude).iter().collect()),
            _ => None,
        }
    }

    /// Detects and locates the mixed scene's events.
    fn detected() -> Workspace {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Mixed).unwrap();
        let project = &workspace.project;
        let settings = AssociateSettings::default();
        let inputs: Vec<DetectInput> = project
            .traces()
            .iter()
            .filter_map(|trace| {
                let id = project.trace_phenomenology(trace)?.id.clone();
                DetectInput::new(project, trace, loaded(&workspace, trace)?, default_detector(id.as_str(), trace.sample_rate), settings.radius_m)
            })
            .collect();
        let outcome = run_detection(&inputs, &settings, CHUNK);
        let meta: Vec<DetectMeta> = inputs.iter().map(DetectMeta::from).collect();
        let changes = event_changes(project, &meta, &outcome, project.reference.unwrap(), true, "test");
        workspace.execute("detect", changes).unwrap();
        workspace
    }

    /// The known event nearest in origin time, within 0.5 s.
    fn truth(workspace: &Workspace, event: &Event) -> Option<&'static synthetic::MixedEvent> {
        let origin = event.origin.as_ref()?;
        let reference = workspace.project.reference.unwrap();
        synthetic::MIXED.iter().find(|m| (origin.time.seconds_since(reference) - m.origin).abs() < 0.5)
    }

    /// M6 self-check: every synthetic earthquake and explosion is found,
    /// each discriminant separates the classes, and the rules get all right.
    #[test]
    fn earthquakes_and_explosions_are_separated_by_their_discriminants() {
        let mut workspace = detected();
        let project = &workspace.project;
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let settings = ClassifySettings::default();
        let results = classify(&traces, all_events(project, &traces), &settings);

        let mut by_class: HashMap<&str, Vec<&Classified>> = HashMap::new();
        for r in &results {
            let event = project.event(r.event).unwrap();
            let truth = truth(&workspace, event).unwrap_or_else(|| panic!("unexpected event {event:#?}"));
            by_class.entry(truth.class.label()).or_default().push(r);
            assert_eq!(r.verdict.labels[0].0.as_str(), truth.class.label(), "{:?} {:?}", r.features.event, r.verdict.evidence);
        }
        assert_eq!(by_class.get(EARTHQUAKE).map(Vec::len), Some(5));
        assert_eq!(by_class.get(EXPLOSION).map(Vec::len), Some(5));

        let range = |class: &str, name: &str| {
            let values: Vec<f64> = by_class[class].iter().map(|r| r.features.event_value(name)).collect();
            assert!(values.iter().all(|v| v.is_finite()), "{class} {name} {values:?}");
            (values.iter().copied().fold(f64::INFINITY, f64::min), values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        };
        for name in ["p_s_log", "hf_lf_log", "first_motion"] {
            let (quake, blast) = (range(EARTHQUAKE, name), range(EXPLOSION, name));
            assert!(quake.1 < blast.0, "{name}: earthquakes {quake:?}, explosions {blast:?}");
        }
        assert!(range(EXPLOSION, "first_motion").0 == 1.0, "explosions push first");
        assert!(range(EXPLOSION, "depth_m").1 < 1_500.0 && range(EARTHQUAKE, "depth_m").0 > 3_000.0);
        assert!(range(EXPLOSION, "air_snr_db").0 > 20.0 && range(EARTHQUAKE, "air_snr_db").1 < 10.0);

        // Recording predictions leaves status alone and undoes; accepting takes the label.
        let changes = prediction_changes(&workspace.project, &results, &settings.hash());
        assert_eq!(changes.len(), 10);
        workspace.execute("classify", changes).unwrap();
        assert!(prediction_changes(&workspace.project, &results, &settings.hash()).is_empty(), "re-running changes nothing");
        let first = workspace.project.events()[0].clone();
        assert_eq!(first.status, EventStatus::Auto);
        let order = review_order(&workspace.project, true);
        assert_eq!(order.len(), 10);
        let uncertainty: Vec<f32> = order.iter().map(|id| workspace.project.event(*id).unwrap().uncertainty()).collect();
        assert!(uncertainty.windows(2).all(|w| w[0] >= w[1]));
        let accept = accept_change(&workspace.project, first.id).unwrap();
        workspace.execute("accept", vec![accept]).unwrap();
        let accepted = workspace.project.event(first.id).unwrap();
        assert_eq!((accepted.status, accepted.label.as_ref().map(|l| l.as_str())), (EventStatus::Reviewed, Some(truth(&workspace, &first).unwrap().class.label())));
        assert_eq!(review_order(&workspace.project, true).len(), 9);
        workspace.undo();
        workspace.undo();
        assert!(workspace.project.events().iter().all(|e| e.predictions.is_empty() && e.label.is_none()));
    }

    #[test]
    fn the_dataset_carries_classes_noise_and_features() {
        let mut workspace = detected();
        let ids: Vec<Id> = workspace.project.events().iter().map(|e| e.id).collect();
        let quarry = Label { id: LabelId::new("quarry"), name: "Quarry".into(), parent: Some(LabelId::new(EXPLOSION)), expects: vec![] };
        workspace.execute("subtype", vec![Change::DefineLabel(quarry)]).unwrap();
        let label = label_change(&workspace.project, ids[1], Some(LabelId::new("quarry"))).unwrap();
        let reject = Change::SetEvent(Event { status: EventStatus::Rejected, ..workspace.project.event(ids[2]).unwrap().clone() });
        workspace.execute("label", vec![label, reject]).unwrap();

        let project = &workspace.project;
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let results = classify(&traces, all_events(project, &traces), &ClassifySettings::default());
        let (events, members, manifest) = dataset(project, &traces, &results, false, "ana");
        assert_eq!(events.rows(), 2);
        let column = |table: &Table, name: &str| table.columns.iter().find(|c| c.0 == name).unwrap().1.clone();
        let squiggle_io::dataset::Values::Text(classes) = column(&events, "class") else { panic!() };
        assert_eq!(classes, [Some("quarry".to_string()), Some(NOISE.to_string())]);
        let squiggle_io::dataset::Values::Text(paths) = column(&events, "class_path") else { panic!() };
        assert_eq!(paths[0].as_deref(), Some("explosion/quarry"));
        assert_eq!(events.columns.len(), 11 + EVENT_FEATURES.len());
        assert!(members.rows() >= 16);
        assert_eq!(manifest.taxonomy_version, project.taxonomy().version);

        let dir = tempfile::tempdir().unwrap();
        let written = write_dataset(dir.path(), &events, &members, manifest).unwrap();
        assert_eq!(written.events, 2);
        assert!(dir.path().join(squiggle_io::dataset::MEMBERS_FILE).exists());
    }

    #[test]
    fn slugs_are_lowercase_ids() {
        assert_eq!(slug(" Quarry blast! "), "quarry_blast");
        assert_eq!(slug("Mine-collapse 2"), "mine_collapse_2");
    }
}
