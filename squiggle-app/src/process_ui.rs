//! Process menu windows: per-trace preprocessing and automatic alignment.
//!
//! Both act on the selected tracks and commit one undoable command. Alignment
//! measures on the plotted (processed) series over a time window, runs off
//! the UI thread, and lists every result for review before anything is
//! applied. Locked offsets are reported, never changed.

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::dsp::filter::MAX_ORDER;
use squiggle_core::dsp::{AlignSettings, Band, Channel, Detrend, FilterSpec, Measured, Solution, align};
use squiggle_core::sensor::Provenance;
use squiggle_core::{Alignment, Change, Id, LagMethod, Processing};

use crate::plot::Plot;
use crate::tracks::{Track, TrackView};
use crate::workspace::Workspace;

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
const GOOD_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 210, 110);
/// Analysis samples per measured pair, window plus search range.
const ALIGN_SAMPLES: usize = 1 << 20;
/// Rows whose reference pair is weaker than this (coefficient × confidence),
/// or whose pairs disagree by more than two analysis samples, start unchecked.
const MIN_WEIGHT: f64 = 0.1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BandKind {
    #[default]
    None,
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

impl BandKind {
    const ALL: [Self; 5] = [Self::None, Self::Lowpass, Self::Highpass, Self::Bandpass, Self::Bandstop];

    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Lowpass => "Low-pass",
            Self::Highpass => "High-pass",
            Self::Bandpass => "Band-pass",
            Self::Bandstop => "Band-stop",
        }
    }
}

/// Editable form of a [`Processing`].
struct ProcessingDraft {
    detrend: Detrend,
    taper_percent: f64,
    band: BandKind,
    low_hz: f64,
    high_hz: f64,
    order: u8,
    zero_phase: bool,
}

impl Default for ProcessingDraft {
    fn default() -> Self {
        Self { detrend: Detrend::Mean, taper_percent: 5.0, band: BandKind::Bandpass, low_hz: 1.0, high_hz: 10.0, order: 4, zero_phase: true }
    }
}

impl ProcessingDraft {
    fn build(&self) -> Processing {
        let band = match self.band {
            BandKind::None => None,
            BandKind::Lowpass => Some(Band::Lowpass { hz: self.high_hz }),
            BandKind::Highpass => Some(Band::Highpass { hz: self.low_hz }),
            BandKind::Bandpass => Some(Band::Bandpass { low_hz: self.low_hz, high_hz: self.high_hz }),
            BandKind::Bandstop => Some(Band::Bandstop { low_hz: self.low_hz, high_hz: self.high_hz }),
        };
        Processing {
            detrend: self.detrend,
            taper: self.taper_percent / 100.0,
            filter: band.map(|band| FilterSpec { band, order: self.order, zero_phase: self.zero_phase }),
        }
    }

    fn load(&mut self, processing: &Processing) {
        self.detrend = processing.detrend;
        self.taper_percent = processing.taper * 100.0;
        let Some(filter) = processing.filter else {
            self.band = BandKind::None;
            return;
        };
        self.order = filter.order;
        self.zero_phase = filter.zero_phase;
        match filter.band {
            Band::Lowpass { hz } => (self.band, self.high_hz) = (BandKind::Lowpass, hz),
            Band::Highpass { hz } => (self.band, self.low_hz) = (BandKind::Highpass, hz),
            Band::Bandpass { low_hz, high_hz } => (self.band, self.low_hz, self.high_hz) = (BandKind::Bandpass, low_hz, high_hz),
            Band::Bandstop { low_hz, high_hz } => (self.band, self.low_hz, self.high_hz) = (BandKind::Bandstop, low_hz, high_hz),
        }
    }
}

struct AlignDraft {
    method: LagMethod,
    from: f64,
    to: f64,
    max_lag_s: f64,
    all_pairs: bool,
    /// Window set by the analyst or from the view at least once.
    window_set: bool,
}

impl Default for AlignDraft {
    fn default() -> Self {
        Self { method: LagMethod::default(), from: 0.0, to: 1.0, max_lag_s: 1.0, all_pairs: true, window_set: false }
    }
}

/// One measured trace, in job order.
struct JobTrace {
    trace: Id,
    code: String,
    offset: f64,
    locked: bool,
}

struct AlignJob {
    traces: Vec<JobTrace>,
    reference: usize,
    settings: AlignSettings,
    all_pairs: bool,
}

struct ResultRow {
    trace: Id,
    code: String,
    /// Delay behind the reference, seconds.
    delay: Option<f64>,
    /// Against the reference, when measured directly.
    direct: Option<Measured>,
    worst_residual: Option<f64>,
    /// Weight of the direct reference pair; zero when pairs disagree.
    support: f64,
    /// Measured minus true delay; synthetic data only.
    truth_error: Option<f64>,
    offset: f64,
    locked: bool,
    is_reference: bool,
    apply: bool,
}

struct AlignResults {
    rows: Vec<ResultRow>,
    method: LagMethod,
    stage: String,
    config_hash: String,
    analysis_rate: Option<f64>,
}

#[derive(Resource, Default)]
pub struct ProcessWindows {
    pub processing_open: bool,
    pub align_open: bool,
    draft: ProcessingDraft,
    align: AlignDraft,
    task: Option<(AlignJob, Task<Solution>)>,
    results: Option<AlignResults>,
}

pub fn process_menu(ui: &mut egui::Ui, windows: &mut ProcessWindows, view: &mut TrackView) {
    if ui.button("Filter and detrend…").on_hover_text("Preprocess the selected tracks").clicked() {
        windows.processing_open = true;
        ui.close();
    }
    if ui.button("Align…").on_hover_text("Measure and apply delays between selected tracks").clicked() {
        windows.align_open = true;
        ui.close();
    }
    ui.separator();
    ui.checkbox(&mut view.show_raw, "Show raw samples").on_hover_text("Plot tracks without their processing");
}

pub fn windows(
    mut contexts: EguiContexts,
    plot: Res<Plot>,
    mut workspace: ResMut<Workspace>,
    mut view: ResMut<TrackView>,
    tracks: Query<&Track>,
    mut windows: ResMut<ProcessWindows>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    poll_alignment(windows, &workspace);
    processing_window(&ctx, windows, &mut workspace, &tracks);
    align_window(&ctx, windows, &mut workspace, &mut view, &tracks, plot.view.plot_width());
    Ok(())
}

/// Selected tracks in row order.
fn selected<'a>(tracks: &'a Query<&Track>) -> Vec<&'a Track> {
    let mut selected: Vec<&Track> = tracks.iter().filter(|t| t.selected).collect();
    selected.sort_by_key(|t| t.row.map_or(i64::MAX, |(top, _)| top as i64));
    selected
}

fn processing_window(ctx: &egui::Context, windows: &mut ProcessWindows, workspace: &mut Workspace, tracks: &Query<&Track>) {
    if !windows.processing_open {
        return;
    }
    let mut open = true;
    egui::Window::new("Filter and Detrend").open(&mut open).resizable(false).show(ctx, |ui| {
        let draft = &mut windows.draft;
        egui::Grid::new("processing").num_columns(2).show(ui, |ui| {
            ui.label("Detrend");
            egui::ComboBox::from_id_salt("detrend").selected_text(draft.detrend.label()).show_ui(ui, |ui| {
                for d in Detrend::ALL {
                    ui.selectable_value(&mut draft.detrend, d, d.label());
                }
            });
            ui.end_row();
            ui.label("Taper");
            ui.add(egui::DragValue::new(&mut draft.taper_percent).range(0.0..=50.0).speed(0.5).suffix(" % each end"));
            ui.end_row();
            ui.label("Filter");
            egui::ComboBox::from_id_salt("band").selected_text(draft.band.label()).show_ui(ui, |ui| {
                for b in BandKind::ALL {
                    ui.selectable_value(&mut draft.band, b, b.label());
                }
            });
            ui.end_row();
            if matches!(draft.band, BandKind::Highpass | BandKind::Bandpass | BandKind::Bandstop) {
                ui.label(if draft.band == BandKind::Highpass { "Corner" } else { "Low edge" });
                ui.add(egui::DragValue::new(&mut draft.low_hz).range(1e-6..=1e12).speed(0.05).suffix(" Hz"));
                ui.end_row();
            }
            if matches!(draft.band, BandKind::Lowpass | BandKind::Bandpass | BandKind::Bandstop) {
                ui.label(if draft.band == BandKind::Lowpass { "Corner" } else { "High edge" });
                ui.add(egui::DragValue::new(&mut draft.high_hz).range(1e-6..=1e12).speed(0.05).suffix(" Hz"));
                ui.end_row();
            }
            if draft.band != BandKind::None {
                ui.label("Order");
                ui.add(egui::DragValue::new(&mut draft.order).range(1..=MAX_ORDER))
                    .on_hover_text("Butterworth order; band filters have twice as many poles");
                ui.end_row();
                ui.label("Phase");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut draft.zero_phase, true, "Zero-phase").on_hover_text("Forward and backward: no delay");
                    ui.radio_value(&mut draft.zero_phase, false, "Causal").on_hover_text("One pass; delays like a real instrument filter");
                });
                ui.end_row();
            }
        });

        ui.separator();
        let processing = draft.build();
        let chosen = selected(tracks);
        if chosen.is_empty() {
            ui.label("Select tracks to process (click their rows).");
        }
        let mut changes = Vec::new();
        let mut clears = Vec::new();
        egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
            for track in &chosen {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&track.code).monospace());
                    if track.phenomenology.is_none() {
                        ui.colored_label(WARN_COLOR, "⚠ unassigned; not processed");
                        return;
                    }
                    ui.weak(track.processing.describe());
                    match processing.validate(track.rate) {
                        Ok(()) => changes.push(Change::SetProcessing { trace: track.trace, processing }),
                        Err(error) => {
                            ui.colored_label(WARN_COLOR, format!("⚠ {error}"));
                        }
                    }
                    clears.push(Change::SetProcessing { trace: track.trace, processing: Processing::default() });
                });
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            let label = format!("Apply to {} track{}", changes.len(), if changes.len() == 1 { "" } else { "s" });
            if ui.add_enabled(!changes.is_empty(), egui::Button::new(label)).clicked() {
                let n = changes.len();
                workspace.execute_or_report(format!("Process {n} track{}", if n == 1 { "" } else { "s" }), changes);
            }
            if ui.add_enabled(!clears.is_empty(), egui::Button::new("Clear")).on_hover_text("Remove processing from the selected tracks").clicked() {
                workspace.execute_or_report("Clear processing", clears);
            }
            if let Some(first) = chosen.first()
                && ui.button("Load from first").on_hover_text("Copy the first selected track's processing into this form").clicked()
            {
                windows.draft.load(&first.processing);
            }
        });
    });
    windows.processing_open = open;
}

/// FNV-1a: a short, stable id for an alignment configuration.
fn config_hash(text: &str) -> String {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    format!("{hash:016x}")
}

fn start_alignment(windows: &mut ProcessWindows, chosen: &[&Track], reference: &Track) {
    let a = &windows.align;
    let settings = AlignSettings { method: a.method, from: a.from, to: a.to, max_lag_s: a.max_lag_s, max_samples: ALIGN_SAMPLES };
    let mut members: Vec<&Track> = vec![reference];
    members.extend(chosen.iter().copied().filter(|t| t.trace != reference.trace && t.phenomenology_id == reference.phenomenology_id));
    // Resampler support, at the lowest analysis rate any pair can use.
    let lowest = members.iter().map(|t| t.rate).fold(f64::INFINITY, f64::min);
    let span = a.to - a.from + 2.0 * a.max_lag_s;
    let margin = 32.0 / lowest.min(ALIGN_SAMPLES as f64 / span);

    let mut data = Vec::new();
    let mut traces = Vec::new();
    for track in &members {
        let reach = if track.trace == reference.trace { 0.0 } else { a.max_lag_s };
        // The reference may serve as the moving side of a pair too.
        let reach = reach.max(if a.all_pairs { a.max_lag_s } else { 0.0 });
        if let Some((values, start)) = track.window(a.from - reach - margin, a.to + reach + margin) {
            data.push((values, track.rate, start));
            traces.push(JobTrace { trace: track.trace, code: track.code.clone(), offset: track.offset, locked: track.locked });
        }
    }
    if traces.first().is_none_or(|t| t.trace != reference.trace) || traces.len() < 2 {
        return;
    }
    let all_pairs = a.all_pairs;
    let task = AsyncComputeTaskPool::get().spawn(async move {
        let channels: Vec<Channel> = data.iter().map(|(values, rate, start)| Channel { values, rate: *rate, start: *start }).collect();
        align(&channels, 0, &settings, all_pairs)
    });
    windows.results = None;
    windows.task = Some((AlignJob { traces, reference: 0, settings, all_pairs }, task));
}

fn poll_alignment(windows: &mut ProcessWindows, workspace: &Workspace) {
    let Some((_, task)) = &mut windows.task else {
        return;
    };
    let Some(solution) = check_ready(task) else {
        return;
    };
    let (job, _) = windows.task.take().expect("checked above");
    let r = job.reference;
    let truth = |i: usize| workspace.truth.get(&job.traces[i].trace).copied();
    let analysis_rate = solution.pairs.iter().map(|(_, _, m)| m.analysis_rate).reduce(f64::min);
    // Two samples at the coarsest analysis rate.
    let residual_limit = analysis_rate.map_or(f64::INFINITY, |rate| 2.0 / rate);
    let rows = job
        .traces
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let delay = solution.delays[i];
            let direct = solution
                .pairs
                .iter()
                .find(|(a, b, _)| (*a, *b) == (r.min(i), r.max(i)) && i != r)
                .map(|(_, _, m)| *m);
            let worst_residual = solution
                .pairs
                .iter()
                .filter(|(a, b, _)| *a == i || *b == i)
                .filter_map(|(a, b, m)| solution.residual(*a, *b, m))
                .map(f64::abs)
                .fold(None, |w: Option<f64>, x| Some(w.map_or(x, |w| w.max(x))));
            // Expected delay at the measured positions: true delay plus the offsets already applied.
            let truth_error = match (delay, truth(i), truth(r)) {
                (Some(d), Some(ti), Some(tr)) => Some(d - (ti - tr + t.offset - job.traces[r].offset)),
                _ => None,
            };
            // Trusted when measured directly against the reference and consistent with the other pairs.
            let consistent = worst_residual.is_none_or(|r| r <= residual_limit);
            let support = if consistent { direct.map_or(0.0, |m| m.weight()) } else { 0.0 };
            ResultRow {
                trace: t.trace,
                code: t.code.clone(),
                delay,
                direct,
                worst_residual,
                truth_error,
                offset: t.offset,
                locked: t.locked,
                is_reference: i == r,
                support,
                apply: i != r && delay.is_some() && !t.locked && support >= MIN_WEIGHT,
            }
        })
        .collect();
    let s = &job.settings;
    let key = format!("{:?} {} {} {} {} {}", s.method, s.from, s.to, s.max_lag_s, job.all_pairs, job.traces[r].trace);
    windows.results = Some(AlignResults {
        rows,
        method: s.method,
        stage: format!("align.{}", if s.method == LagMethod::GccPhat { "gcc-phat" } else { "xcorr" }),
        config_hash: config_hash(&key),
        analysis_rate,
    });
}

fn seconds(value: f64) -> String {
    format!("{value:+.6}")
}

fn align_window(
    ctx: &egui::Context,
    windows: &mut ProcessWindows,
    workspace: &mut Workspace,
    view: &mut TrackView,
    tracks: &Query<&Track>,
    width: f64,
) {
    if !windows.align_open {
        return;
    }
    if !windows.align.window_set {
        let (from, to) = (view.time.start, view.time.end(width));
        windows.align = AlignDraft { from, to, max_lag_s: (to - from) / 4.0, window_set: true, ..AlignDraft::default() };
    }
    let chosen = selected(tracks);
    let reference = view
        .reference
        .and_then(|id| tracks.iter().find(|t| t.trace == id))
        .or_else(|| chosen.first().copied());

    let mut open = true;
    egui::Window::new("Align").open(&mut open).default_width(560.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Reference");
            let text = reference.map_or("none — select tracks".to_string(), |t| format!("★ {}", t.code));
            egui::ComboBox::from_id_salt("align_reference").selected_text(text).show_ui(ui, |ui| {
                for track in &chosen {
                    if ui.selectable_label(view.reference == Some(track.trace), &track.code).clicked() {
                        view.reference = Some(track.trace);
                    }
                }
            });
        });
        let a = &mut windows.align;
        egui::Grid::new("align_settings").num_columns(2).show(ui, |ui| {
            ui.label("Method");
            ui.horizontal(|ui| {
                for m in LagMethod::ALL {
                    ui.radio_value(&mut a.method, m, m.label());
                }
            });
            ui.end_row();
            ui.label("Window");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut a.from).speed(0.01).suffix(" s"));
                ui.label("to");
                ui.add(egui::DragValue::new(&mut a.to).speed(0.01).suffix(" s"));
                if ui.button("Visible range").clicked() {
                    (a.from, a.to) = (view.time.start, view.time.end(width));
                }
            });
            ui.end_row();
            ui.label("Search");
            ui.add(egui::DragValue::new(&mut a.max_lag_s).range(0.0..=1e6).speed(0.01).prefix("± ").suffix(" s"));
            ui.end_row();
            ui.label("Pairs");
            ui.checkbox(&mut a.all_pairs, "Every pair, reconciled by least squares")
                .on_hover_text("Otherwise only reference pairs are measured");
            ui.end_row();
        });

        // Who takes part.
        let mut runnable = reference.is_some() && a.to > a.from;
        if let Some(reference) = reference {
            if reference.phenomenology_id.is_none() {
                ui.colored_label(WARN_COLOR, "⚠ The reference has no phenomenology assigned.");
                runnable = false;
            }
            let mut members = 0;
            for track in chosen.iter().filter(|t| t.trace != reference.trace) {
                if track.phenomenology_id.is_none() {
                    ui.colored_label(WARN_COLOR, format!("{}: unassigned, skipped", track.code));
                } else if track.phenomenology_id != reference.phenomenology_id {
                    ui.colored_label(
                        WARN_COLOR,
                        format!("{}: different phenomenology, skipped (cross-phenomenology alignment uses picks)", track.code),
                    );
                } else if track.series.is_none() {
                    ui.colored_label(WARN_COLOR, format!("{}: still loading", track.code));
                } else {
                    members += 1;
                }
            }
            if members == 0 {
                ui.weak("Select at least one more track of the same phenomenology.");
                runnable = false;
            }
        }
        ui.horizontal(|ui| {
            let busy = windows.task.is_some();
            if ui.add_enabled(runnable && !busy, egui::Button::new("Measure")).clicked()
                && let Some(reference) = reference
            {
                view.reference = Some(reference.trace);
                start_alignment(windows, &chosen, reference);
            }
            if busy {
                ui.spinner();
                ui.label("measuring…");
            }
        });

        let mut applied = false;
        if let Some(results) = &mut windows.results {
            ui.separator();
            if let Some(rate) = results.analysis_rate {
                ui.weak(format!("{} at ≥ {} S/s on the plotted series.", results.method, trim(rate)));
            }
            if results.rows.iter().any(|r| !r.is_reference && r.delay.is_some() && r.support < MIN_WEIGHT) {
                ui.colored_label(
                    WARN_COLOR,
                    "⚠ Weak or ambiguous correlation left unchecked. Filter to the signal band or narrow the window, then measure again.",
                );
            }
            let has_truth = results.rows.iter().any(|r| r.truth_error.is_some());
            egui::Grid::new("align_results").striped(true).num_columns(if has_truth { 8 } else { 7 }).show(ui, |ui| {
                for heading in ["", "Trace", "Delay", "r", "Confidence", "Residual"] {
                    ui.strong(heading);
                }
                if has_truth {
                    ui.strong("vs truth");
                }
                ui.strong("New offset");
                ui.end_row();
                for row in &mut results.rows {
                    let enabled = !row.is_reference && !row.locked && row.delay.is_some();
                    ui.add_enabled(enabled, egui::Checkbox::without_text(&mut row.apply));
                    ui.label(egui::RichText::new(if row.is_reference { format!("★ {}", row.code) } else { row.code.clone() }).monospace());
                    match row.delay {
                        Some(delay) if !row.is_reference => ui.monospace(seconds(delay)),
                        Some(_) => ui.weak("reference"),
                        None => ui.colored_label(WARN_COLOR, "not measured"),
                    };
                    match row.direct {
                        Some(m) => ui.monospace(format!("{:.2}", m.estimate.coefficient)),
                        None => ui.label(""),
                    };
                    match row.direct {
                        Some(m) => {
                            let color = if m.estimate.confidence >= 0.3 { GOOD_COLOR } else { WARN_COLOR };
                            ui.colored_label(color, format!("{:.2}", m.estimate.confidence))
                        }
                        None => ui.label(""),
                    };
                    match row.worst_residual {
                        Some(r) => ui.monospace(format!("{:.2e} s", r)),
                        None => ui.label(""),
                    };
                    if has_truth {
                        match row.truth_error {
                            Some(e) if !row.is_reference => ui.monospace(seconds(e)),
                            _ => ui.label(""),
                        };
                    }
                    match (row.delay, row.is_reference, row.locked) {
                        (Some(delay), false, true) => ui.colored_label(
                            WARN_COLOR,
                            format!("🔒 kept {}; measured {}", seconds(row.offset), seconds(row.offset - delay)),
                        ),
                        (Some(delay), false, false) => ui.monospace(seconds(row.offset - delay)),
                        _ => ui.label(""),
                    };
                    ui.end_row();
                }
            });
            let changes: Vec<Change> = results
                .rows
                .iter()
                .filter(|r| r.apply && !r.locked)
                .filter_map(|r| {
                    let offset_s = r.offset - r.delay?;
                    let provenance = Provenance::Auto { stage: results.stage.clone(), config_hash: results.config_hash.clone() };
                    let alignment = Alignment { offset_s, provenance, locked: false };
                    Some(Change::SetAlignment { trace: r.trace, alignment: Some(alignment), revert: false })
                })
                .collect();
            ui.horizontal(|ui| {
                let n = changes.len();
                let label = format!("Apply {n} offset{}", if n == 1 { "" } else { "s" });
                if ui.add_enabled(n > 0, egui::Button::new(label)).clicked()
                    && workspace.execute_or_report(format!("Align {n} track{} ({})", if n == 1 { "" } else { "s" }, results.method), changes)
                {
                    applied = true;
                }
                ui.weak("Shifts each trace by minus its delay. Undo with ⌘Z / Ctrl+Z.");
            });
        }
        if applied {
            windows.results = None;
        }
    });
    windows.align_open = open;
}

fn trim(value: f64) -> String {
    let text = format!("{value:.3}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_round_trip_processing() {
        let mut draft = ProcessingDraft { band: BandKind::Highpass, low_hz: 0.5, order: 2, zero_phase: false, ..ProcessingDraft::default() };
        let built = draft.build();
        assert_eq!(built.filter, Some(FilterSpec { band: Band::Highpass { hz: 0.5 }, order: 2, zero_phase: false }));
        let mut other = ProcessingDraft::default();
        other.load(&built);
        assert_eq!(other.build(), built);
        draft.band = BandKind::None;
        assert!(draft.build().filter.is_none());
    }

    #[test]
    fn config_hashes_are_stable() {
        assert_eq!(config_hash(""), "cbf29ce484222325");
        assert_ne!(config_hash("a"), config_hash("b"));
    }
}
