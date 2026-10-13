//! Classification: event features, the rules classifier, template matching
//! and imported models, labels and the taxonomy, the review queue, and
//! training-dataset export.
//!
//! Features use each track's plotted (processed) series, as detection does.
//! Classifying replaces each event's predictions with one per model (the
//! rules, the template library, then every enabled model) and never changes
//! its status or label.
//! Labelling, or accepting a prediction, marks the event reviewed.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::classify::{RULES_MODEL, RuleSettings, Verdict, rules};
use squiggle_core::features::{kind_of, EVENT_FEATURES, EventData, FEATURE_SET, FeatureSettings, Features, MEMBER_FEATURES, OriginData, TraceData, features};
use squiggle_core::taxonomy::NOISE;
use squiggle_core::template::{Match, MatchSettings, TEMPLATES_MODEL, Template, match_all, vote};
use squiggle_core::{Change, Event, EventStatus, Id, IqComponent, Label, LabelId, Prediction, Project, SampleKind, Timestamp, Trace};
use squiggle_io::dataset::{FORMAT, Manifest, Table, write_dataset};
use squiggle_core::windows::{WindowSpec, Windows};
use squiggle_io::dataset::WindowTable;
use squiggle_models::Package;
use squiggle_models::package::Inputs;

use crate::models_ui::ModelRegistry;
use crate::templates_ui::template_change;
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
pub(crate) fn track_values<'a>(tracks: impl IntoIterator<Item = &'a Track>) -> HashMap<Id, Vec<f32>> {
    tracks
        .into_iter()
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
    pub matching: MatchSettings,
}

impl ClassifySettings {
    /// Of the features and rules.
    pub fn hash(&self) -> String {
        config_hash(&format!("{FEATURE_SET} {RULES_MODEL} {:?} {:?}", self.features, self.rules))
    }
}

/// Templates that classification matches events against.
#[derive(Clone, Debug, Default)]
pub struct Library {
    pub templates: Vec<Template>,
    pub taxonomy_version: u32,
}

impl Library {
    pub fn of(project: &Project) -> Self {
        Self { templates: project.templates().to_vec(), taxonomy_version: project.taxonomy().version }
    }

    pub fn hash(&self, settings: &MatchSettings) -> String {
        let keys: Vec<(Id, &Option<LabelId>)> = self.templates.iter().map(|t| (t.id, &t.label)).collect();
        config_hash(&format!("{TEMPLATES_MODEL} {keys:?} {settings:?}"))
    }

    /// The templates' vote for an event with these matches.
    pub fn prediction(&self, matches: &[Match], settings: &MatchSettings) -> Option<Prediction> {
        Some(Prediction {
            model: TEMPLATES_MODEL.to_string(),
            config_hash: self.hash(settings),
            taxonomy_version: self.taxonomy_version,
            labels: vote(matches, &self.templates, settings)?,
            parts: Vec::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Classified {
    pub event: Id,
    pub data: EventData,
    pub features: Features,
    pub verdict: Verdict,
    /// Library templates matched, best first.
    pub matches: Vec<Match>,
    /// The templates' (when they vote), then one per model in the order given.
    pub predictions: Vec<Prediction>,
}

/// Features and the rules verdict, the template library's vote, and each
/// model's prediction from features computed with that model's own settings
/// and, for deep models, its own waveform windows. Templates without a good
/// match, and a model that abstains (a required phenomenology is missing),
/// record nothing.
pub fn classify_one(traces: &[TraceData], event: Id, data: EventData, settings: &ClassifySettings, models: &[Arc<Package>], library: &Library) -> Classified {
    let features = features(traces, &data, &settings.features);
    let verdict = rules(&features.event, &settings.rules);
    let matches = match_all(traces, &data, &library.templates, &settings.matching);
    let mut other: Vec<(FeatureSettings, Vec<f64>)> = Vec::new();
    let mut cut: Vec<(Vec<WindowSpec>, Vec<Windows>)> = Vec::new();
    let predictions = models
        .iter()
        .filter_map(|model| {
            let wanted = model.spec.input.settings;
            if wanted != settings.features && !other.iter().any(|o| o.0 == wanted) {
                other.push((wanted, squiggle_core::features::features(traces, &data, &wanted).event));
            }
            let event = if wanted == settings.features { &features.event } else { &other.iter().find(|o| o.0 == wanted).expect("pushed above").1 };
            let specs = model.window_specs();
            if !cut.iter().any(|c| c.0 == specs) {
                cut.push((specs.clone(), squiggle_core::windows::windows(traces, &data, &specs)));
            }
            let windows = &cut.iter().find(|c| c.0 == specs).expect("pushed above").1;
            model.prediction(Inputs { features: event, windows })
        });
    let predictions = library.prediction(&matches, &settings.matching).into_iter().chain(predictions).collect();
    Classified { event, data, features, verdict, matches, predictions }
}

pub fn classify(traces: &[TraceData], events: Vec<(Id, EventData)>, settings: &ClassifySettings, models: &[Arc<Package>], library: &Library) -> Vec<Classified> {
    events.into_iter().map(|(id, data)| classify_one(traces, id, data, settings, models, library)).collect()
}

/// Every event in `project` over `traces`.
pub fn all_events(project: &Project, traces: &[TraceData]) -> Vec<(Id, EventData)> {
    project.events().iter().filter_map(|e| Some((e.id, event_data(project, e, traces)?))).collect()
}

/// Replaces each result's event's predictions with the rules' (unless
/// `rules` is `None`; else its config hash) and the models'.
pub fn prediction_changes(project: &Project, results: &[Classified], rules: Option<&str>) -> Vec<Change> {
    results
        .iter()
        .filter_map(|r| {
            let event = project.event(r.event)?;
            let mut predictions: Vec<Prediction> = rules.map(|hash| r.verdict.prediction(hash, project.taxonomy().version)).into_iter().collect();
            predictions.extend(r.predictions.iter().cloned());
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

/// Accepts `id`: reviewed, and labelled with the consensus prediction when
/// unlabelled; rejected when that consensus is noise.
pub fn accept_change(project: &Project, id: Id) -> Option<Change> {
    let event = project.event(id)?;
    let top = project.consensus(event).first().map(|l| l.0.clone());
    if event.label.is_none() && top.as_ref().is_some_and(|l| l.as_str() == NOISE) {
        return (event.status != EventStatus::Rejected).then(|| Change::SetEvent(Event { status: EventStatus::Rejected, ..event.clone() }));
    }
    let label = event.label.clone().or_else(|| top.filter(|l| project.taxonomy().get(l).is_some()));
    label_change(project, id, label)
}

/// Review order: by time, or in queue mode the automatic events, most uncertain first.
pub fn review_order(project: &Project, queue: bool) -> Vec<Id> {
    let mut events: Vec<&Event> = project.events().iter().filter(|e| !queue || e.status == EventStatus::Auto).collect();
    events.sort_by(|a, b| {
        let by_time = a.time().seconds_since(b.time()).total_cmp(&0.0);
        if queue { project.uncertainty(b).total_cmp(&project.uncertainty(a)).then(by_time) } else { by_time }
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

/// Default window specs: one per assigned phenomenology with traces, at their median rate.
pub fn default_window_specs(project: &Project) -> Vec<WindowSpec> {
    let mut specs: Vec<WindowSpec> = Vec::new();
    for phenomenology in project.phenomenologies() {
        let mut rates: Vec<f64> = project
            .traces()
            .iter()
            .filter(|t| project.trace_phenomenology(t).is_some_and(|p| p.id == phenomenology.id))
            .map(|t| t.sample_rate)
            .collect();
        if rates.is_empty() || specs.iter().any(|s| s.phenomenology == phenomenology.id) {
            continue;
        }
        rates.sort_by(f64::total_cmp);
        specs.push(WindowSpec::default_for(phenomenology.id.clone(), kind_of(&phenomenology.propagation), rates[rates.len() / 2]));
    }
    specs
}

/// Events and members tables, windows (for `specs`) and the manifest for `results`.
#[allow(clippy::too_many_arguments)]
pub fn dataset(
    project: &Project,
    traces: &[TraceData],
    results: &[Classified],
    settings: &FeatureSettings,
    specs: &[WindowSpec],
    include_unlabelled: bool,
    analyst: &str,
) -> (Table, Table, Vec<WindowTable>, Manifest) {
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

    let mut windows: Vec<WindowTable> =
        specs.iter().map(|s| WindowTable { phenomenology: s.phenomenology.0.clone(), samples: s.samples, values: Vec::new(), events: Vec::new() }).collect();
    for (row, r) in rows.iter().enumerate() {
        for (table, cut) in windows.iter_mut().zip(squiggle_core::windows::windows(traces, &r.0.data, specs)) {
            for w in cut.traces {
                table.values.extend(w);
                table.events.push(row as i64);
            }
        }
    }

    let mut classes: Vec<(Option<String>, usize)> = Vec::new();
    for (_, _, class) in &rows {
        match classes.iter_mut().find(|c| &c.0 == class) {
            Some(c) => c.1 += 1,
            None => classes.push((class.clone(), 1)),
        }
    }
    let manifest = Manifest {
        format: FORMAT.to_string(),
        project: project.name.clone(),
        created: Timestamp::now().to_string(),
        analyst: analyst.to_string(),
        feature_set: FEATURE_SET.to_string(),
        taxonomy_version: taxonomy.version,
        labels: taxonomy.labels.clone(),
        noise_class: NOISE.to_string(),
        feature_settings: Some(*settings),
        windows: specs.to_vec(),
        event_features: EVENT_FEATURES.iter().map(|s| s.to_string()).collect(),
        member_features: MEMBER_FEATURES.iter().map(|s| s.to_string()).collect(),
        classes,
        events: 0,
        members: 0,
        hashes: Vec::new(),
    };
    (events, members, windows, manifest)
}

struct ExportJob {
    project: Project,
    traces: Vec<TraceData>,
    settings: ClassifySettings,
    specs: Vec<WindowSpec>,
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
    /// Event, revision, settings and models behind the features panel.
    panel: Option<(Id, u64, ClassifySettings, String, Classified)>,
    /// Traces the panel's result indexes.
    panel_traces: Vec<TraceData>,
    new_id: String,
    new_name: String,
    new_parent: Option<LabelId>,
    include_unlabelled: bool,
    /// Phenomenologies to export windows for, with their specs.
    window_specs: Vec<(bool, WindowSpec)>,
    /// Skip recording the rules prediction.
    no_rules: bool,
    /// Skip matching the template library.
    no_templates: bool,
    /// Name of the next template saved from the panel.
    template_name: String,
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
    registry: Res<ModelRegistry>,
    mut windows: ResMut<ClassifyWindows>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    if let Some((_, task)) = &mut windows.task
        && let Some(results) = check_ready(task)
    {
        let (hash, _) = windows.task.take().expect("checked above");
        let changes = prediction_changes(&workspace.project, &results, (!windows.no_rules).then_some(hash.as_str()));
        if !changes.is_empty() {
            workspace.execute_or_report(format!("Classify {} events", changes.len()), changes);
        }
    }
    poll_export(windows, &workspace, &tracks);
    let models = registry.active(&workspace.project);
    classify_window(&ctx, windows, &workspace, &tracks, &models);
    features_window(&ctx, windows, &mut workspace, &view, &tracks, &models);
    labels_window(&ctx, windows, &mut workspace);
    export_window(&ctx, windows, &workspace);
    Ok(())
}

fn drag(ui: &mut egui::Ui, label: &str, value: &mut f64, speed: f64, suffix: &str) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).speed(speed).suffix(suffix));
    ui.end_row();
}

fn classify_window(ctx: &egui::Context, windows: &mut ClassifyWindows, workspace: &Workspace, tracks: &Query<&Track>, models: &[Arc<Package>]) {
    if !windows.classify_open {
        return;
    }
    let mut open = true;
    egui::Window::new("Classify Events").open(&mut open).default_width(420.0).show(ctx, |ui| {
        ui.weak("Features from each member's plotted series, then the rules classifier and every enabled model (Events ▸ Models). Each run replaces the events' predictions and never changes their status or label.");
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
        ui.checkbox(&mut windows.no_rules, "Leave the rules out (record only the enabled models)");
        let templates = project.templates().len();
        ui.horizontal(|ui| {
            ui.add_enabled(templates > 0, egui::Checkbox::new(&mut windows.no_templates, format!("Leave the {templates} templates out")))
                .on_hover_text("Template matching votes the label of the best-correlating templates (Events ▸ Templates)");
            let m = &mut windows.settings.matching;
            ui.add_enabled_ui(templates > 0 && !windows.no_templates, |ui| {
                ui.add(egui::DragValue::new(&mut m.min_cc).range(0.0..=1.0).speed(0.01).prefix("vote from CC "));
                ui.add(egui::DragValue::new(&mut m.max_lag_s).range(0.0..=60.0).speed(0.05).prefix("± ").suffix(" s lag"));
                ui.add(egui::DragValue::new(&mut m.min_sensors).range(1..=1000).suffix(" sensors"));
            });
        });
        ui.horizontal(|ui| {
            let busy = windows.task.is_some();
            if ui.add_enabled(n > 0 && !busy, egui::Button::new(format!("Classify {n} events"))).clicked() {
                let values = track_values(tracks);
                let traces = trace_data(project, |t| values.get(&t.id).cloned());
                let events = all_events(project, &traces);
                let settings = windows.settings;
                let models = models.to_vec();
                let library = if windows.no_templates { Library::default() } else { Library::of(project) };
                let task = AsyncComputeTaskPool::get().spawn(async move { classify(&traces, events, &settings, &models, &library) });
                windows.task = Some((settings.hash(), task));
            }
            if busy {
                ui.spinner();
            }
        });
        let names: Vec<&str> = models.iter().map(|m| m.spec.id.trim_start_matches("heidtmare.squiggle.model/")).collect();
        ui.weak(if names.is_empty() { "Rules only: no enabled models.".to_string() } else { format!("Rules, then {}.", names.join(", ")) });
        let mut ids = vec![RULES_MODEL.to_string(), TEMPLATES_MODEL.to_string()];
        ids.extend(models.iter().map(|m| m.spec.id.clone()));
        for id in ids {
            let mut counts: Vec<(String, usize)> = Vec::new();
            for event in project.events() {
                if let Some((label, _)) = event.prediction(&id).and_then(Prediction::top) {
                    match counts.iter_mut().find(|c| c.0 == label.0) {
                        Some(c) => c.1 += 1,
                        None => counts.push((label.0.clone(), 1)),
                    }
                }
            }
            if !counts.is_empty() {
                let summary: Vec<String> = counts.iter().map(|(l, n)| format!("{n} {l}")).collect();
                ui.label(format!("{} predicts: {}.", id.trim_start_matches("heidtmare.squiggle.model/"), summary.join(", ")));
            }
        }
        let disagree = project.events().iter().filter(|e| {
            let tops: Vec<&LabelId> = e.predictions.iter().filter_map(|p| p.top().map(|t| t.0)).collect();
            tops.windows(2).any(|w| w[0] != w[1])
        });
        let disagree = disagree.count();
        if disagree > 0 {
            ui.colored_label(WARN_COLOR, format!("Models disagree on {disagree} events; the review queue ranks them first."));
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

fn features_window(ctx: &egui::Context, windows: &mut ClassifyWindows, workspace: &mut Workspace, view: &TrackView, tracks: &Query<&Track>, models: &[Arc<Package>]) {
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
    let library = Library::of(&workspace.project);
    let key = format!("{:?} {}", models.iter().map(|m| m.hash.as_str()).collect::<Vec<_>>(), library.hash(&windows.settings.matching));
    let stale = !matches!(&windows.panel, Some((e, r, s, k, _)) if *e == id && *r == revision && *s == windows.settings && *k == key);
    if stale {
        let values = track_values(tracks);
        let project = &workspace.project;
        let event = project.event(id).expect("checked above");
        let member_traces: Vec<Id> = event.members.iter().map(|m| m.trace).collect();
        // Members, every air and EM trace for coupled arrivals, and every
        // trace of a phenomenology a deep model reads.
        let branches: Vec<WindowSpec> = models.iter().flat_map(|m| m.window_specs()).collect();
        let traces = trace_data(project, |t| {
            let wanted = member_traces.contains(&t.id)
                || project.trace_phenomenology(t).is_some_and(|p| {
                    matches!(p.propagation, squiggle_core::Propagation::Constant { .. }) || branches.iter().any(|b| b.phenomenology == p.id)
                });
            if wanted { values.get(&t.id).cloned() } else { None }
        });
        windows.panel =
            event_data(project, event, &traces).map(|data| (id, revision, windows.settings, key, classify_one(&traces, id, data, &windows.settings, models, &library)));
        windows.panel_traces = traces;
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
            for prediction in &result.predictions {
                ui.add_space(4.0);
                ui.strong(format!("{} (computed now)", prediction.model.trim_start_matches("heidtmare.squiggle.model/")));
                probability_bars(ui, project, &prediction.labels);
                for (part, labels) in &prediction.parts {
                    ui.weak(format!("  {part} alone"));
                    probability_bars(ui, project, labels);
                }
            }
            if !result.matches.is_empty() {
                ui.add_space(4.0);
                ui.strong("Template matches (computed now)").on_hover_text("Network correlation: the mean over shared sensors of the best normalised cross-correlation within the lag");
                egui::Grid::new("matches").num_columns(4).striped(true).show(ui, |ui| {
                    for m in result.matches.iter().take(5) {
                        let Some(t) = project.template(m.template) else { continue };
                        ui.label(&t.name);
                        ui.label(t.label.as_ref().map_or("—".to_string(), |l| project.taxonomy().path(l)));
                        ui.monospace(format!("CC {:.2}", m.cc)).on_hover_text(format!("{} sensors, mean lag {:+.2} s", m.sensors, m.lag_s));
                        ui.end_row();
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut windows.template_name).hint_text("Template name").desired_width(130.0));
                let tip = "Save each member's window from its onset as a library template, labelled as the event";
                let named = !windows.template_name.trim().is_empty();
                if ui.add_enabled(named, egui::Button::new("Save as template")).on_hover_text(tip).clicked() {
                    match template_change(project, &windows.panel_traces, &result.data, id, &windows.template_name) {
                        Some(c) => {
                            change = Some(("Save template".to_string(), c));
                            windows.template_name.clear();
                        }
                        None => {
                            ui.colored_label(WARN_COLOR, "No member windows");
                        }
                    }
                }
            });
            let consensus = project.consensus(event);
            if event.predictions.len() > 1 {
                ui.add_space(4.0);
                ui.strong("Ensemble (recorded, weighted)").on_hover_text("Each model's vote is weighted as set in Events ▸ Models; the rules count 1");
                probability_bars(ui, project, &consensus);
            }
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
                            .panel_traces
                            .get(result.data.members[m])
                            .and_then(|t| project.sensor(t.sensor))
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
        // Keep one spec per phenomenology in the project, remembering edits.
        let defaults = default_window_specs(project);
        windows.window_specs.retain(|(_, s)| defaults.iter().any(|d| d.phenomenology == s.phenomenology));
        for d in defaults {
            if !windows.window_specs.iter().any(|(_, s)| s.phenomenology == d.phenomenology) {
                windows.window_specs.push((false, d));
            }
        }
        egui::CollapsingHeader::new("Waveform windows (for deep models)").show(ui, |ui| {
            ui.weak("Per phenomenology: each trace from its onset (pick, else predicted from the origin), resampled, demeaned, unit peak; written to windows.safetensors.");
            egui::Grid::new("window_specs").num_columns(6).striped(true).show(ui, |ui| {
                for heading in ["", "Rate", "Before onset", "Samples", "Traces", "Length"] {
                    ui.strong(heading);
                }
                ui.end_row();
                for (on, spec) in &mut windows.window_specs {
                    ui.checkbox(on, spec.phenomenology.as_str());
                    ui.add(egui::DragValue::new(&mut spec.rate).range(0.1..=1e9).speed(1.0).suffix(" Hz"));
                    ui.add(egui::DragValue::new(&mut spec.pre_s).range(0.0..=1e3).speed(0.05).suffix(" s"));
                    ui.add(egui::DragValue::new(&mut spec.samples).range(32..=8192).speed(8.0));
                    ui.add(egui::DragValue::new(&mut spec.max_traces).range(1..=256));
                    ui.label(format!("{:.3} s", spec.duration()));
                    ui.end_row();
                }
            });
        });
        let busy = windows.folder.is_some() || windows.export.is_some();
        ui.horizontal(|ui| {
            let n = if windows.include_unlabelled { project.events().len() } else { labelled };
            if ui.add_enabled(n > 0 && !busy, egui::Button::new(format!("Export {n} events…"))).clicked() {
                windows.exported = None;
                windows.folder = Some(IoTaskPool::get().spawn(async {
                    if let Some(path) = crate::tour::picked() {
                        return Some(path);
                    }
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
                specs: windows.window_specs.iter().filter(|s| s.0).map(|s| s.1.clone()).collect(),
                include_unlabelled: windows.include_unlabelled,
                analyst: workspace.analyst.0.clone(),
            };
            windows.export = Some(AsyncComputeTaskPool::get().spawn(async move {
                let events: Vec<(Id, EventData)> = all_events(&job.project, &job.traces)
                    .into_iter()
                    .filter(|(id, _)| job.include_unlabelled || job.project.event(*id).and_then(training_class).is_some())
                    .collect();
                let results = classify(&job.traces, events, &job.settings, &[], &Library::default());
                let (events, members, windows, manifest) =
                    dataset(&job.project, &job.traces, &results, &job.settings.features, &job.specs, job.include_unlabelled, &job.analyst);
                write_dataset(&folder, &events, &members, &windows, manifest).map(|m| (folder, m))
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
    use squiggle_core::model::ModelRef;
    use squiggle_core::{Analyst, Change};

    fn loaded(workspace: &Workspace, trace: &Trace) -> Option<Vec<f32>> {
        match workspace.samples.get(&trace.id) {
            Some(SampleState::Loaded(samples)) => Some(samples.series(IqComponent::Magnitude).iter().collect()),
            _ => None,
        }
    }

    /// Detects and locates the mixed scene's events.
    fn detected() -> Workspace {
        detected_scene(SyntheticEvent::Mixed)
    }

    fn detected_scene(scene: SyntheticEvent) -> Workspace {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), scene).unwrap();
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

    /// The known class of the event nearest in origin time, within 0.5 s.
    fn truth_label(workspace: &Workspace, event: &Event) -> Option<LabelId> {
        let origin = event.origin.as_ref()?;
        let i = workspace.truth_events.iter().position(|(_, t)| origin.time.seconds_since(*t).abs() < 0.5)?;
        workspace.truth_labels.get(i).cloned()
    }

    /// Labels every detected event with its known class.
    fn label_all(workspace: &mut Workspace) -> usize {
        let project = &workspace.project;
        let labels: Vec<Change> = project.events().iter().filter_map(|e| label_change(project, e.id, Some(truth_label(workspace, e)?))).collect();
        let n = labels.len();
        workspace.execute("label", labels).unwrap();
        n
    }

    /// Classifies with `models` over the traces of `keep` phenomenologies; correct top labels and predictions.
    fn classify_with(workspace: &Workspace, models: &[Arc<Package>], keep: &[&str]) -> (usize, Vec<(Id, Prediction)>) {
        let project = &workspace.project;
        let traces = trace_data(project, |t| {
            let p = project.trace_phenomenology(t)?;
            if keep.contains(&p.id.as_str()) { loaded(workspace, t) } else { None }
        });
        let results = classify(&traces, all_events(project, &traces), &ClassifySettings::default(), models, &Library::default());
        let mut correct = 0;
        let mut out = Vec::new();
        for r in results {
            let event = project.event(r.event).unwrap();
            if let Some(p) = r.predictions.into_iter().next() {
                correct += usize::from(Some(p.labels[0].0.clone()) == truth_label(workspace, event));
                out.push((r.event, p));
            }
        }
        (correct, out)
    }

    /// The known family of the event nearest in origin time, within 0.5 s.
    fn truth_family(workspace: &Workspace, event: &Event) -> Option<usize> {
        let origin = event.origin.as_ref()?;
        let i = workspace.truth_events.iter().position(|(_, t)| origin.time.seconds_since(*t).abs() < 0.5)?;
        workspace.truth_families.get(i).copied()
    }

    /// Located events per known repeat (index into the truth).
    fn found(workspace: &Workspace) -> Vec<usize> {
        let mut found = vec![0; workspace.truth_events.len()];
        for event in workspace.project.events() {
            let Some(origin) = &event.origin else { continue };
            if let Some(i) = workspace.truth_events.iter().position(|(_, t)| origin.time.seconds_since(*t).abs() < 0.5) {
                found[i] += 1;
            }
        }
        found
    }

    /// M9 self-check: repeating-source families are recovered by templates
    /// (labels by network correlation, and weak repeats by the matched
    /// filter), by clustering (PCA and DBSCAN, trained offline and imported),
    /// and labelled in bulk; agreement is tracked per model version.
    #[test]
    fn synthetic_families_are_recovered() {
        use crate::events_ui::{CHUNK, DetectInput, DetectMeta, default_detector, event_changes, run_detection};
        use crate::models_ui::{agreements, import_change};
        use crate::scatter_ui::{bulk_label, scatter_points};
        use crate::templates_ui::template_change;
        use squiggle_core::dsp::MatchedSpec;
        use squiggle_core::template::TEMPLATES_MODEL;

        let mut workspace = detected_scene(SyntheticEvent::Families);
        let (truth, _) = synthetic::families();
        let weak: Vec<usize> = (0..truth.len()).filter(|&i| truth[i].scale < 0.1).collect();
        assert_eq!(weak.len(), synthetic::FAMILIES.len());
        let before = found(&workspace);
        let strong = (0..truth.len()).filter(|i| !weak.contains(i) && before[*i] > 0).count();
        assert_eq!(strong, truth.len() - weak.len(), "STA/LTA finds every strong repeat: {before:?}");
        let swarms: Vec<usize> = weak.iter().copied().filter(|&i| truth[i].class == synthetic::Class::Earthquake).collect();
        assert!(swarms.iter().all(|&i| before[i] == 0), "STA/LTA misses the weak swarm repeats: {before:?}");

        // The scene's subtype per family; templates from each family's first event.
        let labels: Vec<LabelId> = synthetic::FAMILIES.iter().map(|f| LabelId::new(slug(f.name))).collect();
        let project = &workspace.project;
        assert!(labels.iter().zip(&synthetic::FAMILIES).all(|(l, f)| project.taxonomy().is_a(l, &LabelId::new(f.class.label()))));
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let mut sources = Vec::new();
        let mut templates = Vec::new();
        for (f, label) in labels.iter().enumerate() {
            let event = project.events().iter().find(|e| truth_family(&workspace, e) == Some(f)).unwrap();
            let data = event_data(project, event, &traces).unwrap();
            sources.push((event.id, label_change(project, event.id, Some(label.clone())).unwrap()));
            templates.push(data);
        }
        workspace.execute("label sources", sources.iter().map(|s| s.1.clone()).collect()).unwrap();
        for ((id, _), data) in sources.iter().zip(&templates) {
            let change = template_change(&workspace.project, &traces, data, *id, "t").unwrap();
            workspace.execute("template", vec![change]).unwrap();
        }
        let library = Library::of(&workspace.project);
        assert_eq!(library.templates.len(), 4);
        assert!(library.templates.iter().all(|t| t.traces.len() >= 6), "{:?}", library.templates.iter().map(|t| t.traces.len()).collect::<Vec<_>>());

        // Matched filter on top of STA/LTA finds the weak repeats too.
        let project = &workspace.project;
        let settings = AssociateSettings::default();
        let inputs: Vec<DetectInput> = project
            .traces()
            .iter()
            .filter_map(|trace| {
                let id = project.trace_phenomenology(trace)?.id.clone();
                let input = DetectInput::new(project, trace, loaded(&workspace, trace)?, default_detector(id.as_str(), trace.sample_rate), settings.radius_m)?;
                Some(input.with_templates(project, MatchedSpec::default()))
            })
            .collect();
        assert!(inputs.iter().filter(|i| !i.templates.is_empty()).count() >= 8);
        let outcome = run_detection(&inputs, &settings, CHUNK);
        assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
        let meta: Vec<DetectMeta> = inputs.iter().map(DetectMeta::from).collect();
        let changes = event_changes(project, &meta, &outcome, project.reference.unwrap(), true, "matched");
        workspace.execute("detect", changes).unwrap();
        let after = found(&workspace);
        assert!(after.iter().all(|n| *n == 1), "every repeat found once, reviewed ones not again: {after:?}");
        let extra = workspace.project.events().iter().filter(|e| truth_family(&workspace, e).is_none()).count();
        assert!(extra <= 2, "{extra} events match no repeat");

        // Templates label every repeat with its family.
        let project = &workspace.project;
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let classify_settings = ClassifySettings::default();
        let results = classify(&traces, all_events(project, &traces), &classify_settings, &[], &library);
        let mut right = 0;
        let mut voted = 0;
        for r in &results {
            let event = project.event(r.event).unwrap();
            let Some(f) = truth_family(&workspace, event) else { continue };
            let Some(p) = r.predictions.iter().find(|p| p.model == TEMPLATES_MODEL) else { continue };
            voted += 1;
            right += usize::from(p.labels[0].0 == labels[f]);
        }
        assert!(voted >= truth.len() - 1 && right == voted, "{right} of {voted} voted right");
        let changes = prediction_changes(project, &results, Some(&classify_settings.hash()));
        workspace.execute("classify", changes).unwrap();

        // Analysts label every repeat; the dataset trains an embedding offline.
        let project = &workspace.project;
        let changes: Vec<Change> = project.events().iter().filter_map(|e| label_change(project, e.id, Some(labels[truth_family(&workspace, e)?].clone()))).collect();
        workspace.execute("label", changes).unwrap();
        let project = &workspace.project;
        let results = classify(&traces, all_events(project, &traces), &classify_settings, &[], &Library::default());
        let (events, members, windows, manifest) = dataset(project, &traces, &results, &classify_settings.features, &[], true, "ana");
        let dir = tempfile::tempdir().unwrap();
        write_dataset(&dir.path().join("dataset"), &events, &members, &windows, manifest).unwrap();
        let data = squiggle_train::Dataset::load(&dir.path().join("dataset")).unwrap();
        let options = squiggle_train::ClusterOptions { version: 1, ..Default::default() };
        let (package, assigned) = squiggle_train::cluster(&data, &options).unwrap();
        let family: Vec<Option<usize>> = data.events.iter().map(|id| truth_family(&workspace, project.event(id.parse().unwrap()).unwrap())).collect();
        let ari = squiggle_train::adjusted_rand(&assigned, &family);
        assert!(package.spec.output.clusters.len() == synthetic::FAMILIES.len() && ari >= 0.8, "{} clusters, adjusted Rand index {ari}", package.spec.output.clusters.len());

        // Imported, the app embeds and assigns each event as the trainer did.
        let out = dir.path().join("cluster");
        package.save(&out).unwrap();
        let import = import_change(&workspace.project, &out).unwrap();
        workspace.execute("import", vec![import]).unwrap();
        let mut registry = ModelRegistry::default();
        registry.refresh(&workspace.project);
        assert!(registry.active(&workspace.project).is_empty(), "an embedding is not a classifier");
        let embedding = registry.embeddings(&workspace.project).pop().unwrap();
        let points = scatter_points(&workspace.project, &traces, Some(&embedding));
        for (id, cluster) in data.events.iter().zip(&assigned) {
            let point = points.iter().find(|p| p.event.to_string() == *id).unwrap();
            if cluster.is_some() {
                assert_eq!(point.embedding.as_ref().unwrap().1, *cluster, "{id}");
            }
        }

        // Bulk relabel a cluster as one undoable command.
        let k = assigned.iter().flatten().next().copied().unwrap();
        let ids: Vec<Id> = points.iter().filter(|p| p.embedding.as_ref().and_then(|e| e.1) == Some(k)).map(|p| p.event).collect();
        let other = LabelId::new(squiggle_core::taxonomy::OTHER);
        workspace.execute("bulk", bulk_label(&workspace.project, &ids, Some(other.clone()))).unwrap();
        assert!(ids.iter().all(|id| workspace.project.event(*id).unwrap().label.as_ref() == Some(&other)));
        workspace.undo();
        assert!(ids.iter().all(|id| workspace.project.event(*id).unwrap().label.as_ref() != Some(&other)));

        // Agreement per model version: templates agree with the analysts.
        let rows = agreements(&workspace.project);
        let templates = rows.iter().find(|r| r.model == TEMPLATES_MODEL).unwrap();
        assert!(templates.compared >= truth.len() - 1 && templates.agree == templates.compared, "{templates:?}");
        assert!(rows.iter().any(|r| r.model == RULES_MODEL));
    }

    /// M8 self-check: a CNN trained on exported waveform windows of one
    /// synthetic catalogue classifies another scene, and its late fusion still
    /// classifies when infrasound or seismic sensors are missing. The app
    /// computes exactly the trainer's probabilities.
    #[test]
    fn fused_cnns_classify_with_a_phenomenology_missing() {
        use crate::models_ui::import_change;
        use squiggle_models::deep::{Encoder, Fusion};
        use squiggle_train::{Method, Options};

        let mut catalogue = detected_scene(SyntheticEvent::Catalogue);
        let n = label_all(&mut catalogue);
        assert!(n >= synthetic::CATALOGUE_EVENTS - 2, "{n} of {} labelled", synthetic::CATALOGUE_EVENTS);
        let project = &catalogue.project;
        let traces = trace_data(project, |t| loaded(&catalogue, t));
        let settings = ClassifySettings::default();
        let results = classify(&traces, all_events(project, &traces), &settings, &[], &Library::default());
        let specs = default_window_specs(project);
        assert_eq!(specs.iter().map(|s| s.phenomenology.as_str()).collect::<Vec<_>>(), ["seismic", "infrasound"]);
        let (events, members, windows, manifest) = dataset(project, &traces, &results, &settings.features, &specs, false, "ana");
        assert!(windows.iter().all(|w| w.events.len() >= n), "every event has windows of each");
        let dir = tempfile::tempdir().unwrap();
        write_dataset(&dir.path().join("dataset"), &events, &members, &windows, manifest).unwrap();
        let data = squiggle_train::Dataset::load(&dir.path().join("dataset")).unwrap();

        let params = squiggle_models::Params { epochs: 25, batch: 8, learning_rate: 5e-3, ..Default::default() };
        let mut held_out = detected();
        let mut packages = Vec::new();
        for (name, encoder, fusion) in [("late", Encoder::Waveform, Fusion::Late), ("early", Encoder::Spectrogram, Fusion::Early)] {
            let options = Options { method: Method::Cnn { encoder, fusion }, params: params.clone(), name: Some(name.into()), folds: 0, ..Options::default() };
            let package = squiggle_train::train(&data, &options).unwrap();
            let out = dir.path().join(name);
            package.save(&out).unwrap();
            let expected = squiggle_train::predictions(&package, &data);

            // Same probabilities in the app as in the trainer.
            let import = import_change(&catalogue.project, &out).unwrap();
            catalogue.execute("import", vec![import]).unwrap();
            let mut registry = ModelRegistry::default();
            registry.refresh(&catalogue.project);
            let loaded_package = registry.package(&package.spec.id).unwrap().clone();
            let (_, app) = classify_with(&catalogue, std::slice::from_ref(&loaded_package), &["seismic", "infrasound"]);
            assert!(app.len() >= n);
            for (event, prediction) in &app {
                let Some((_, Some(trainer))) = expected.iter().find(|e| e.0 == event.to_string()) else { continue };
                for (label, p) in package.spec.output.labels.iter().zip(trainer) {
                    assert_eq!(prediction.labels.iter().find(|l| l.0.as_str() == label).unwrap().1, *p as f32, "{name} {label}");
                }
            }

            let import = import_change(&held_out.project, &out).unwrap();
            held_out.execute("import", vec![import]).unwrap();
            packages.push((name, loaded_package));
        }

        // Held out: the fixed mixed scene, with all sensors, without infrasound, without seismic.
        for (name, package) in &packages {
            let models = std::slice::from_ref(package);
            let (all, predictions) = classify_with(&held_out, models, &["seismic", "infrasound"]);
            assert!(all >= 9, "{name}: {all}/10 with every sensor");
            if *name == "late" {
                assert!(predictions.iter().all(|(_, p)| p.parts.len() == 2));
            }
            let (seismic, predictions) = classify_with(&held_out, models, &["seismic"]);
            assert_eq!(predictions.len(), 10);
            assert!(seismic >= 9, "{name}: {seismic}/10 without infrasound");
            if *name == "late" {
                assert!(predictions.iter().all(|(_, p)| p.parts.len() == 1 && p.parts[0].0 == "seismic"));
            }
            let (infrasound, _) = classify_with(&held_out, models, &["infrasound"]);
            assert!(infrasound >= 8, "{name}: {infrasound}/10 without seismic");
        }

        // Ensemble: a zero-weight model does not move the consensus.
        let mut registry = ModelRegistry::default();
        registry.refresh(&held_out.project);
        let models = registry.active(&held_out.project);
        let project = &held_out.project;
        let traces = trace_data(project, |t| loaded(&held_out, t));
        let results = classify(&traces, all_events(project, &traces), &settings, &models, &Library::default());
        let changes = prediction_changes(project, &results, Some(&settings.hash()));
        held_out.execute("classify", changes).unwrap();
        let event = held_out.project.events()[0].clone();
        assert_eq!(event.predictions.len(), 3);
        let early = held_out.project.models().iter().find(|m| m.id.contains("early")).unwrap().clone();
        let late = held_out.project.models().iter().find(|m| m.id.contains("late")).unwrap().clone();
        held_out.execute("weights", vec![Change::SetModel(ModelRef { weight: 0.0, ..early }), Change::SetModel(ModelRef { weight: 0.0, ..late })]).unwrap();
        let rules_only = Event { predictions: vec![event.prediction(RULES_MODEL).unwrap().clone()], ..event.clone() };
        assert_eq!(held_out.project.consensus(&event), rules_only.consensus());
    }

    /// M6 self-check: every synthetic earthquake and explosion is found,
    /// each discriminant separates the classes, and the rules get all right.
    #[test]
    fn earthquakes_and_explosions_are_separated_by_their_discriminants() {
        let mut workspace = detected();
        let project = &workspace.project;
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let settings = ClassifySettings::default();
        let results = classify(&traces, all_events(project, &traces), &settings, &[], &Library::default());

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
        let changes = prediction_changes(&workspace.project, &results, Some(&settings.hash()));
        assert_eq!(changes.len(), 10);
        workspace.execute("classify", changes).unwrap();
        assert!(prediction_changes(&workspace.project, &results, Some(&settings.hash())).is_empty(), "re-running changes nothing");
        let first = workspace.project.events()[0].clone();
        assert_eq!(first.status, EventStatus::Auto);
        let order = review_order(&workspace.project, true);
        assert_eq!(order.len(), 10);
        let uncertainty: Vec<f32> = order.iter().map(|id| workspace.project.uncertainty(workspace.project.event(*id).unwrap())).collect();
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

    /// M7 self-check: export, train and import give the same predictions in
    /// the app as in the trainer.
    #[test]
    fn trained_packages_predict_in_the_app_as_in_the_trainer() {
        use crate::models_ui::{agreement, import_change};
        use squiggle_models::Kind;

        let mut workspace = detected();
        let project = &workspace.project;
        let labels: Vec<Change> = project
            .events()
            .iter()
            .filter_map(|e| label_change(project, e.id, Some(LabelId::new(truth(&workspace, e)?.class.label()))))
            .collect();
        assert_eq!(labels.len(), 10);
        workspace.execute("label", labels).unwrap();

        let project = &workspace.project;
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let settings = ClassifySettings::default();
        let results = classify(&traces, all_events(project, &traces), &settings, &[], &Library::default());
        let (events, members, windows, manifest) = dataset(project, &traces, &results, &settings.features, &[], false, "ana");
        assert!(windows.is_empty());
        let dir = tempfile::tempdir().unwrap();
        write_dataset(&dir.path().join("dataset"), &events, &members, &windows, manifest).unwrap();

        let data = squiggle_train::Dataset::load(&dir.path().join("dataset")).unwrap();
        let mut expected = Vec::new();
        for kind in [Kind::Logistic, Kind::Forest] {
            let package = squiggle_train::train(&data, &squiggle_train::Options { method: squiggle_train::Method::Classical(kind), ..Default::default() }).unwrap();
            assert_eq!(package.spec.output.labels, [EARTHQUAKE, EXPLOSION]);
            assert_eq!(package.spec.training.cross_validation.accuracy, 1.0, "{kind:?}");
            let out = dir.path().join(kind.name());
            package.save(&out).unwrap();
            expected.push((package.spec.id.clone(), package.spec.output.labels.clone(), squiggle_train::predictions(&package, &data)));
            let import = import_change(&workspace.project, &out).unwrap();
            workspace.execute("import", vec![import]).unwrap();
        }

        let mut registry = ModelRegistry::default();
        registry.refresh(&workspace.project);
        let models = registry.active(&workspace.project);
        assert_eq!(models.len(), 2);
        let project = &workspace.project;
        let results = classify(&traces, all_events(project, &traces), &settings, &models, &Library::default());
        let changes = prediction_changes(project, &results, Some(&settings.hash()));
        workspace.execute("classify", changes).unwrap();
        for event in workspace.project.events() {
            assert_eq!(event.predictions.len(), 3);
            for (id, labels, rows) in &expected {
                let trainer = rows.iter().find(|r| r.0 == event.id.to_string()).unwrap().1.as_ref().unwrap();
                let app = event.prediction(id).unwrap();
                for (label, p) in labels.iter().zip(trainer) {
                    assert_eq!(app.labels.iter().find(|l| l.0.as_str() == label).unwrap().1, *p as f32, "{id} {label}");
                }
                assert_eq!(app.top().unwrap().0, event.label.as_ref().unwrap());
            }
        }
        for (id, ..) in &expected {
            assert_eq!(agreement(&workspace.project, id), (10, 10));
        }

        // Disabling drops a model from the next run; a changed package is refused.
        let logistic = workspace.project.model(&expected[0].0).unwrap().clone();
        workspace.execute("disable", vec![Change::SetModel(squiggle_core::model::ModelRef { enabled: false, ..logistic.clone() })]).unwrap();
        registry.refresh(&workspace.project);
        assert_eq!(registry.active(&workspace.project).len(), 1);
        let spec = std::path::Path::new(&logistic.path).join(squiggle_models::package::SPEC_FILE);
        let text = std::fs::read_to_string(&spec).unwrap();
        std::fs::write(&spec, format!("{text}\n# edited\n")).unwrap();
        let mut fresh = ModelRegistry::default();
        fresh.refresh(&workspace.project);
        assert!(fresh.package(&logistic.id).unwrap_err().contains("changed"));
    }

    #[test]
    fn accepting_a_noise_consensus_rejects() {
        let mut workspace = detected();
        let id = workspace.project.events()[0].id;
        let noise = Prediction { model: "m@1".into(), config_hash: "h".into(), taxonomy_version: 1, labels: vec![(LabelId::new(NOISE), 0.8), (LabelId::new(EXPLOSION), 0.2)], parts: vec![] };
        workspace.execute("predict", vec![Change::SetPredictions { event: id, predictions: vec![noise] }]).unwrap();
        let accept = accept_change(&workspace.project, id).unwrap();
        workspace.execute("accept", vec![accept]).unwrap();
        let event = workspace.project.event(id).unwrap();
        assert_eq!((event.status, event.label.as_ref()), (EventStatus::Rejected, None));
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
        let results = classify(&traces, all_events(project, &traces), &ClassifySettings::default(), &[], &Library::default());
        let (events, members, _, manifest) = dataset(project, &traces, &results, &FeatureSettings::default(), &[], false, "ana");
        assert_eq!(events.rows(), 2);
        let column = |table: &Table, name: &str| table.columns.iter().find(|c| c.0 == name).unwrap().1.clone();
        let squiggle_io::dataset::Values::Text(classes) = column(&events, "class") else { panic!() };
        assert_eq!(classes, [Some("quarry".to_string()), Some(NOISE.to_string())]);
        let squiggle_io::dataset::Values::Text(paths) = column(&events, "class_path") else { panic!() };
        assert_eq!(paths[0].as_deref(), Some("explosion/quarry"));
        assert_eq!(events.columns.len(), 9 + EVENT_FEATURES.len());
        assert!(members.rows() >= 16);
        assert_eq!(manifest.taxonomy_version, project.taxonomy().version);

        let dir = tempfile::tempdir().unwrap();
        let written = write_dataset(dir.path(), &events, &members, &[], manifest).unwrap();
        assert_eq!(written.events, 2);
        assert!(dir.path().join(squiggle_io::dataset::MEMBERS_FILE).exists());
    }

    #[test]
    fn slugs_are_lowercase_ids() {
        assert_eq!(slug(" Quarry blast! "), "quarry_blast");
        assert_eq!(slug("Mine-collapse 2"), "mine_collapse_2");
    }
}
