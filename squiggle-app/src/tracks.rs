//! Stacked tracks: one row per loaded trace on a shared time axis.
//!
//! Each track draws its [`Envelope`] decimated to about one bucket per pixel,
//! cached over three screen widths so panning and moderate zoom reuse the
//! mesh. Mesh x is seconds from the cache start and the entity transform is
//! computed in `f64`, so nanosecond spacing survives far from the reference.
//!
//! The plotted series is the trace's processed samples (detrend, taper,
//! filter) unless raw display is on. Rows sit at their aligned start; Shift+drag
//! and the arrow keys shift them, as undoable commands. Picks and arrivals
//! predicted from the saved origin are drawn on each row and move with it.

use std::collections::HashSet;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::dsp::{AlignSettings, Channel, candidates};
use squiggle_core::locate::{slant_distance, travel_time};
use squiggle_core::sensor::Provenance;
use squiggle_core::timeline::format_duration;
use squiggle_core::{
    Alignment, Change, ChunkedSamples, Envelope, Id, IqComponent, LagMethod, MinMax, PhenomenologyId, Pick, Point,
    Processing, Rgba, SampleKind, Squiggle, Style, TimeView, Timestamp, TraceSamples,
};

use crate::plot::line::{self, SquiggleLineMaterial};
use crate::strips;
use crate::plot::{Plot, PlotMode, TRACK_LABEL_WIDTH};
use crate::workspace::{SampleState, UNASSIGNED_COLOR, Workspace};

/// Fraction of the row height the half range fills at unit gain.
const ROW_FILL: f64 = 0.45;
const MIN_ROW_HEIGHT: f64 = 40.0;
const LINE_WIDTH: f32 = 1.0;
const SELECTED_COLOR: Rgba = Rgba::CYAN;
const Z_TRACKS: f32 = -10.0;
const Z_STEP: f32 = 0.001;
/// Zoom per point of scroll; 50 points ≈ one wheel notch ≈ 10 %.
const ZOOM_PER_POINT: f64 = 1.0019;
/// Largest window handed to a transform.
pub const MAX_TRANSFORM_SAMPLES: usize = 1 << 20;
/// A shift snaps to a correlation peak within this many pixels.
const SNAP_PX: f64 = 8.0;
/// Weakest correlation offered as a snap target.
const SNAP_MIN: f64 = 0.3;
/// Analysis samples behind snap targets; keeps the drag start responsive.
const SNAP_SAMPLES: usize = 1 << 16;
pub const NFFT_CHOICES: [usize; 6] = [64, 128, 256, 512, 1024, 2048];

const GRID_COLOR: egui::Color32 = egui::Color32::from_gray(45);
const AXIS_COLOR: egui::Color32 = egui::Color32::from_gray(200);
const CURSOR_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 215, 0);
const SELECTED_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 60, 60, 120);
const SHIFT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 140, 60);
const MANUAL_PICK_COLOR: egui::Color32 = egui::Color32::WHITE;
const AUTO_PICK_COLOR: egui::Color32 = egui::Color32::from_rgb(100, 180, 255);
const PREDICTED_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 220, 120);

pub struct TracksPlugin;

impl Plugin for TracksPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TrackView>()
            .add_systems(
                Update,
                (
                    sync_tracks,
                    request_series,
                    poll_series,
                    layout_rows,
                    draw_tracks,
                    strips::prune_strips,
                    strips::request_strips,
                    strips::poll_strips,
                    strips::draw_strips,
                )
                    .chain(),
            );
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowOrder {
    #[default]
    Phenomenology,
    Sensor,
    Start,
    /// From the saved origin.
    Distance,
    /// First pick.
    Arrival,
}

impl RowOrder {
    pub const ALL: [Self; 5] = [Self::Phenomenology, Self::Sensor, Self::Start, Self::Distance, Self::Arrival];

    pub fn label(self) -> &'static str {
        match self {
            Self::Phenomenology => "Phenomenology",
            Self::Sensor => "Sensor",
            Self::Start => "Start time",
            Self::Distance => "Distance from origin",
            Self::Arrival => "First pick",
        }
    }
}

/// How a row maps amplitude to height.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YScale {
    /// The whole trace's range fills the row.
    #[default]
    Normalized,
    /// The visible window's range fills the row.
    Auto,
    /// `±fixed_half_range` around zero.
    Fixed,
}

impl YScale {
    pub const ALL: [Self; 3] = [Self::Normalized, Self::Auto, Self::Fixed];

    pub fn label(self) -> &'static str {
        match self {
            Self::Normalized => "Normalized",
            Self::Auto => "Auto (visible)",
            Self::Fixed => "Fixed",
        }
    }
}

/// What a row shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowDisplay {
    #[default]
    Waveform,
    Spectrogram,
    /// Waveform over its spectrogram.
    Both,
}

impl RowDisplay {
    pub const ALL: [Self; 3] = [Self::Waveform, Self::Spectrogram, Self::Both];

    pub fn label(self) -> &'static str {
        match self {
            Self::Waveform => "Waveform",
            Self::Spectrogram => "Spectrogram",
            Self::Both => "Waveform over spectrogram",
        }
    }

    pub fn has_spectrogram(self) -> bool {
        self != Self::Waveform
    }
}

#[derive(Resource)]
pub struct TrackView {
    pub time: TimeView,
    /// Multiplies every row's amplitude.
    pub gain: f64,
    /// Rows scrolled off the top, in pixels.
    pub scroll: f64,
    pub order: RowOrder,
    pub fit_requested: bool,
    /// Plot samples without their processing.
    pub show_raw: bool,
    /// Trace others align to, and snap to while shifting.
    pub reference: Option<Id>,
    /// Row whose context menu is open.
    menu_track: Option<Entity>,
    drag: Option<ShiftDrag>,
}

impl Default for TrackView {
    fn default() -> Self {
        Self {
            time: TimeView::default(),
            gain: 1.0,
            scroll: 0.0,
            order: RowOrder::default(),
            fit_requested: false,
            show_raw: false,
            reference: None,
            menu_track: None,
            drag: None,
        }
    }
}

/// A Shift+drag in progress.
struct ShiftDrag {
    entity: Entity,
    /// Unsnapped shift, seconds.
    raw: f64,
    /// Shifts that align with the reference or the origin, and why.
    snaps: Vec<(f64, String)>,
    snapped: Option<String>,
}

/// A pick as drawn on its row.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackPick {
    pub id: Id,
    pub phase: String,
    /// Clock-corrected, seconds from the project reference; the row's offset is added when drawn.
    pub at: f64,
    pub manual: bool,
}

/// An arrival predicted from the saved origin.
#[derive(Clone, Debug, PartialEq)]
pub struct Predicted {
    pub phase: String,
    /// Seconds from the project reference, before the row's offset.
    pub at: f64,
    pub travel: f64,
}

/// A trace's plotted series: processed real samples, or one component of
/// processed I/Q.
pub struct Series {
    pub component: IqComponent,
    pub processing: Processing,
    /// Processed samples, all components.
    pub samples: TraceSamples,
    pub values: ChunkedSamples,
    pub envelope: Envelope,
    pub full: MinMax,
    /// Why processing fell back to raw samples.
    pub error: Option<String>,
}

impl Series {
    fn build(raw: &TraceSamples, component: IqComponent, processing: Processing, rate: f64) -> Self {
        let (samples, error) = match raw.processed(&processing, rate) {
            Ok(samples) => (samples, None),
            Err(error) => (raw.clone(), Some(error.to_string())),
        };
        let values = samples.series(component);
        let envelope = Envelope::build(&values);
        let full = envelope.range(&values, 0, values.len());
        Self { component, processing, samples, values, envelope, full, error }
    }

    /// Samples covering `from..to` seconds of a trace starting at `start`, as
    /// a contiguous copy and its start time.
    pub fn window(&self, start: f64, rate: f64, from: f64, to: f64) -> (Vec<f32>, f64) {
        let index = |t: f64| ((t - start) * rate).clamp(0.0, self.values.len() as f64);
        let lo = index(from).floor() as usize;
        let hi = (index(to).ceil() as usize + 1).min(self.values.len());
        (self.values.range(lo, hi).collect(), start + lo as f64 / rate)
    }
}

/// The cached mesh's sample range and bucket size.
#[derive(Clone, Copy)]
struct MeshCache {
    lo: usize,
    hi: usize,
    bucket: usize,
    component: IqComponent,
}

#[derive(Component)]
pub struct Track {
    pub trace: Id,
    pub code: String,
    pub phenomenology: Option<String>,
    pub phenomenology_id: Option<PhenomenologyId>,
    pub color: Rgba,
    /// Aligned start (clock correction plus offset), seconds from the project reference.
    pub start: f64,
    /// Alignment offset included in `start`.
    pub offset: f64,
    pub locked: bool,
    pub processing: Processing,
    /// Uncommitted shift while dragging, seconds.
    pub preview: f64,
    pub display: RowDisplay,
    pub nfft: usize,
    pub picks: Vec<TrackPick>,
    pub predicted: Vec<Predicted>,
    /// Slant distance from the saved origin, metres.
    pub distance_m: Option<f64>,
    pub rate: f64,
    pub len: usize,
    pub kind: SampleKind,
    pub selected: bool,
    pub y_scale: YScale,
    pub fixed_half_range: f32,
    pub component: IqComponent,
    pub series: Option<Arc<Series>>,
    /// Top and height in plot-local pixels.
    pub row: Option<(f64, f64)>,
    index: usize,
    cache: Option<MeshCache>,
}

impl Track {
    /// Where the row is drawn: aligned start plus any drag preview.
    pub fn drawn_start(&self) -> f64 {
        self.start + self.preview
    }

    pub fn end(&self) -> f64 {
        self.drawn_start() + self.len as f64 / self.rate
    }

    /// Where a corrected time on this trace is drawn.
    pub fn drawn(&self, at: f64) -> f64 {
        at + self.offset + self.preview
    }

    /// Earliest pick, as drawn.
    pub fn first_pick(&self) -> Option<f64> {
        self.picks.iter().map(|p| self.drawn(p.at)).reduce(f64::min)
    }

    /// Processing the series should carry.
    fn wanted_processing(&self, view: &TrackView) -> Processing {
        if view.show_raw { Processing::default() } else { self.processing }
    }

    /// The trace as an aligner input over `from..to` seconds.
    pub fn window(&self, from: f64, to: f64) -> Option<(Vec<f32>, f64)> {
        let series = self.series.as_ref()?;
        let (values, start) = series.window(self.drawn_start(), self.rate, from, to);
        (values.len() > 1).then_some((values, start))
    }

    /// The component to plot; real traces ignore it.
    fn wanted(&self) -> IqComponent {
        match self.kind {
            SampleKind::Real => IqComponent::default(),
            SampleKind::Complex => self.component,
        }
    }

    /// Sample indices covering `from..to` seconds, clamped to the trace.
    pub fn indices(&self, from: f64, to: f64) -> (usize, usize) {
        let index = |t: f64| ((t - self.drawn_start()) * self.rate).clamp(0.0, self.len as f64);
        (index(from).floor() as usize, (index(to).ceil() as usize + 1).min(self.len))
    }

    pub fn name(&self) -> String {
        match self.kind {
            SampleKind::Real => self.code.clone(),
            SampleKind::Complex => format!("{} {}", self.code, self.component.label()),
        }
    }

    /// The visible window as a squiggle (x in seconds from the reference), for transforms.
    pub fn window_squiggle(&self, time: &TimeView, width: f64) -> Result<Squiggle, String> {
        let series = self.series.as_ref().ok_or("still loading")?;
        let (lo, hi) = self.indices(time.start, time.end(width));
        if hi <= lo + 1 {
            return Err("not in view".into());
        }
        if hi - lo > MAX_TRANSFORM_SAMPLES {
            return Err(format!("{} samples in view; zoom in to at most {MAX_TRANSFORM_SAMPLES}", hi - lo));
        }
        let points = series
            .values
            .range(lo, hi)
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| Point::new(self.drawn_start() + (lo + i) as f64 / self.rate, f64::from(v)));
        let mut squiggle = Squiggle::with_points(self.name(), points).with_style(Style::new(self.color, 1.5));
        squiggle.trace = Some(self.trace);
        Ok(squiggle)
    }
}

#[derive(Component)]
pub struct SeriesTask(Task<Series>);

struct Described {
    code: String,
    phenomenology: Option<String>,
    phenomenology_id: Option<PhenomenologyId>,
    color: Rgba,
    start: f64,
    offset: f64,
    locked: bool,
    processing: Processing,
    picks: Vec<TrackPick>,
    predicted: Vec<Predicted>,
    distance_m: Option<f64>,
}

/// Spawns, updates and despawns tracks to match the workspace's loaded traces.
#[allow(clippy::too_many_arguments)]
fn sync_tracks(
    mut commands: Commands,
    workspace: Res<Workspace>,
    plot: Res<Plot>,
    mut seen: Local<u64>,
    mut tracks: Query<(Entity, &mut Track)>,
    mut materials: ResMut<Assets<SquiggleLineMaterial>>,
    mut mode: ResMut<PlotMode>,
    mut view: ResMut<TrackView>,
) {
    if *seen == workspace.revision() {
        return;
    }
    *seen = workspace.revision();
    let project = &workspace.project;

    let describe = |trace: &squiggle_core::Trace| {
        let reference = project.reference.unwrap_or(trace.start);
        let phenomenology = project.trace_phenomenology(trace);
        let picks = project
            .trace_picks(trace.id)
            .map(|p| TrackPick {
                id: p.id,
                phase: p.phase.clone(),
                at: project.correct(trace, p.time).seconds_since(reference),
                manual: p.provenance.is_manual(),
            })
            .collect();
        let location = project.sensor(trace.sensor).and_then(|s| s.location);
        let (mut predicted, mut distance_m) = (Vec::new(), None);
        if let (Some(origin), Some(location), Some(phenomenology)) = (&project.origin, location, phenomenology) {
            distance_m = Some(slant_distance(origin.source(), &location));
            for (phase, speed) in phenomenology.propagation.phases() {
                let travel = travel_time(origin.source(), &location, speed);
                predicted.push(Predicted { phase: phase.to_string(), at: origin.time.seconds_since(reference) + travel, travel });
            }
        }
        Described {
            picks,
            predicted,
            distance_m,
            code: project.sensor(trace.sensor).map_or("?", |s| s.code.as_str()).to_string(),
            phenomenology: phenomenology.map(|p| p.name.clone()),
            phenomenology_id: phenomenology.map(|p| p.id.clone()),
            color: phenomenology.map_or(UNASSIGNED_COLOR, |p| p.color),
            start: project.aligned_start(trace).seconds_since(reference),
            offset: trace.offset(),
            locked: trace.is_locked(),
            processing: trace.processing,
        }
    };

    let dragging = view.drag.as_ref().map(|d| d.entity);
    let mut present = HashSet::new();
    for (entity, mut track) in &mut tracks {
        match (project.trace(track.trace), workspace.samples.get(&track.trace)) {
            (Some(trace), Some(SampleState::Loaded(_))) => {
                present.insert(track.trace);
                let d = describe(trace);
                track.code = d.code;
                track.phenomenology = d.phenomenology;
                track.phenomenology_id = d.phenomenology_id;
                track.color = d.color;
                track.start = d.start;
                track.offset = d.offset;
                track.locked = d.locked;
                track.processing = d.processing;
                track.picks = d.picks;
                track.predicted = d.predicted;
                track.distance_m = d.distance_m;
                if dragging != Some(entity) {
                    track.preview = 0.0;
                }
            }
            _ => commands.entity(entity).despawn(),
        }
    }
    if view.reference.is_some_and(|id| project.trace(id).is_none()) {
        view.reference = None;
    }

    let had_tracks = !present.is_empty();
    let mut spawned = false;
    for trace in project.traces() {
        let Some(SampleState::Loaded(samples)) = workspace.samples.get(&trace.id) else {
            continue;
        };
        if present.contains(&trace.id) {
            continue;
        }
        let d = describe(trace);
        let material = materials.add(SquiggleLineMaterial::new(d.color, LINE_WIDTH, plot.scale_factor));
        commands.spawn((
            Track {
                trace: trace.id,
                code: d.code,
                phenomenology: d.phenomenology,
                phenomenology_id: d.phenomenology_id,
                color: d.color,
                start: d.start,
                offset: d.offset,
                locked: d.locked,
                processing: d.processing,
                preview: 0.0,
                display: RowDisplay::default(),
                nfft: 256,
                picks: d.picks,
                predicted: d.predicted,
                distance_m: d.distance_m,
                rate: trace.sample_rate,
                len: samples.len(),
                kind: samples.kind(),
                selected: false,
                y_scale: YScale::default(),
                fixed_half_range: 1.0,
                component: IqComponent::default(),
                series: None,
                row: None,
                index: 0,
                cache: None,
            },
            MeshMaterial2d(material),
            Transform::default(),
            Visibility::Hidden,
        ));
        spawned = true;
    }
    if spawned && !had_tracks {
        *mode = PlotMode::Tracks;
        view.fit_requested = true;
    }
}

/// Builds each track's series off-thread when missing or its component or
/// processing changed.
fn request_series(
    mut commands: Commands,
    workspace: Res<Workspace>,
    view: Res<TrackView>,
    tracks: Query<(Entity, &Track), Without<SeriesTask>>,
) {
    for (entity, track) in &tracks {
        let (component, processing) = (track.wanted(), track.wanted_processing(&view));
        if track.series.as_ref().is_some_and(|s| s.component == component && s.processing == processing) {
            continue;
        }
        let Some(SampleState::Loaded(samples)) = workspace.samples.get(&track.trace) else {
            continue;
        };
        let (samples, rate) = (samples.clone(), track.rate);
        let task = AsyncComputeTaskPool::get().spawn(async move { Series::build(&samples, component, processing, rate) });
        commands.entity(entity).insert(SeriesTask(task));
    }
}

fn poll_series(mut commands: Commands, mut tracks: Query<(Entity, &mut Track, &mut SeriesTask)>) {
    for (entity, mut track, mut task) in &mut tracks {
        if let Some(series) = check_ready(&mut task.0) {
            track.series = Some(Arc::new(series));
            track.cache = None;
            commands.entity(entity).remove::<SeriesTask>();
        }
    }
}

/// Orders rows, sizes them to the plot, and applies a pending fit.
fn layout_rows(plot: Res<Plot>, mode: Res<PlotMode>, mut view: ResMut<TrackView>, mut tracks: Query<&mut Track>) {
    let mut rows: Vec<Mut<Track>> = tracks.iter_mut().collect();
    // Wait for the tracks gutters, which follow the mode a frame later.
    let gutters_ready = *mode == PlotMode::Tracks && plot.view.insets.left == TRACK_LABEL_WIDTH;
    if rows.is_empty() || !plot.is_usable() || !gutters_ready {
        return;
    }
    match view.order {
        RowOrder::Phenomenology => rows.sort_by(|a, b| {
            let key = |t: &Track| (t.phenomenology.is_none(), t.phenomenology.clone(), t.code.clone());
            key(a).cmp(&key(b))
        }),
        RowOrder::Sensor => rows.sort_by(|a, b| a.code.cmp(&b.code)),
        RowOrder::Start => {
            rows.sort_by(|a, b| a.drawn_start().total_cmp(&b.drawn_start()).then_with(|| a.code.cmp(&b.code)))
        }
        // Rows without a value go last.
        RowOrder::Distance => rows.sort_by(|a, b| {
            let key = |t: &Track| t.distance_m.unwrap_or(f64::INFINITY);
            key(a).total_cmp(&key(b)).then_with(|| a.code.cmp(&b.code))
        }),
        RowOrder::Arrival => rows.sort_by(|a, b| {
            let key = |t: &Track| t.first_pick().unwrap_or(f64::INFINITY);
            key(a).total_cmp(&key(b)).then_with(|| a.code.cmp(&b.code))
        }),
    }

    let area = &plot.view;
    let height = (area.plot_height() / rows.len() as f64).max(MIN_ROW_HEIGHT);
    let overflow = (height * rows.len() as f64 - area.plot_height()).max(0.0);
    view.scroll = view.scroll.clamp(0.0, overflow);
    for (index, track) in rows.iter_mut().enumerate() {
        let top = area.insets.top + index as f64 * height - view.scroll;
        if track.row != Some((top, height)) || track.index != index {
            track.row = Some((top, height));
            track.index = index;
        }
    }

    if view.fit_requested {
        view.fit_requested = false;
        let from = rows.iter().map(|t| t.drawn_start()).fold(f64::INFINITY, f64::min);
        let to = rows.iter().map(|t| t.end()).fold(f64::NEG_INFINITY, f64::max);
        view.time.fit(from, to, area.plot_width());
    }
}

/// Bucket size for `samples_per_px`: raw below 1.5, else a power of two.
fn wanted_bucket(samples_per_px: f64) -> usize {
    if samples_per_px < 1.5 { 1 } else { 1usize << (samples_per_px.log2().floor() as u32).min(62) }
}

/// Whether a cached bucket still suits the zoom, with hysteresis.
fn bucket_suits(bucket: usize, samples_per_px: f64) -> bool {
    if bucket == 1 { samples_per_px < 3.0 } else { (0.35..=2.8).contains(&(samples_per_px / bucket as f64)) }
}

#[allow(clippy::type_complexity)]
fn draw_tracks(
    mut commands: Commands,
    plot: Res<Plot>,
    mode: Res<PlotMode>,
    view: Res<TrackView>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SquiggleLineMaterial>>,
    mut tracks: Query<(Entity, &mut Track, &mut Transform, &mut Visibility, &MeshMaterial2d<SquiggleLineMaterial>)>,
) {
    let area = &plot.view;
    let width = area.plot_width();
    let active = *mode == PlotMode::Tracks && plot.is_usable();
    for (entity, mut track, mut transform, mut visibility, material) in &mut tracks {
        let waveform = track.display != RowDisplay::Spectrogram;
        let (Some(series), Some((top, height)), true, true) = (track.series.clone(), track.row, active, waveform) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        let (lo, hi) = track.indices(view.time.start, view.time.end(width));
        let below = top > area.insets.top + area.plot_height();
        if below || top + height < area.insets.top || hi <= lo {
            *visibility = Visibility::Hidden;
            continue;
        }

        let samples_per_px = track.rate / view.time.px_per_s;
        let valid = track.cache.is_some_and(|c| {
            c.component == series.component && c.lo <= lo && c.hi >= hi && bucket_suits(c.bucket, samples_per_px)
        });
        if !valid {
            let bucket = wanted_bucket(samples_per_px);
            let span = hi - lo;
            let cache_lo = lo.saturating_sub(span) / bucket * bucket;
            let cache_hi = (hi + span).min(track.len);
            let rate = track.rate;
            let points: Vec<Point> = series
                .envelope
                .decimate(&series.values, cache_lo, cache_hi, bucket)
                .into_iter()
                .map(|(i, v)| Point::new((i - cache_lo as f64) / rate, f64::from(v)))
                .collect();
            match line::build_mesh(&points) {
                Some(mesh) => commands.entity(entity).insert(Mesh2d(meshes.add(mesh))),
                None => commands.entity(entity).remove::<Mesh2d>(),
            };
            track.cache = Some(MeshCache { lo: cache_lo, hi: cache_hi, bucket, component: series.component });
        }
        let cache = track.cache.expect("set above");

        let extent = match track.y_scale {
            YScale::Normalized => series.full,
            YScale::Auto => series.envelope.range(&series.values, lo, hi),
            YScale::Fixed => MinMax { min: -track.fixed_half_range, max: track.fixed_half_range },
        };
        let (centre, half) = if extent.is_empty() {
            (0.0, 1.0)
        } else {
            let half = f64::from(extent.max - extent.min) / 2.0;
            (f64::from(extent.min) + half, if half > 0.0 { half } else { 1.0 })
        };
        let pixels_per_unit = height * ROW_FILL * view.gain / half;
        let row_centre = top + height / 2.0;
        let cache_start = track.drawn_start() + cache.lo as f64 / track.rate;
        *transform = Transform {
            translation: Vec3::new(
                (area.insets.left + view.time.x_of(cache_start) - area.width / 2.0) as f32,
                (area.height / 2.0 - row_centre - centre * pixels_per_unit) as f32,
                Z_TRACKS + track.index as f32 * Z_STEP,
            ),
            scale: Vec3::new(view.time.px_per_s as f32, pixels_per_unit as f32, 1.0),
            ..default()
        };
        *visibility = Visibility::Visible;

        let color = if track.selected { SELECTED_COLOR } else { track.color };
        let wanted = SquiggleLineMaterial::new(color, LINE_WIDTH, plot.scale_factor);
        if let Some(existing) = materials.get(&material.0)
            && existing.line.color != wanted.line.color
            && let Some(mut existing) = materials.get_mut(&material.0)
        {
            *existing = wanted;
        }
    }
}

fn color32(color: Rgba) -> egui::Color32 {
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgb(byte(color.red), byte(color.green), byte(color.blue))
}

fn manual(workspace: &Workspace) -> Provenance {
    Provenance::Manual { analyst: workspace.analyst.clone(), at: Timestamp::now() }
}

/// A manual alignment at `offset_s`, locked against automatic re-runs.
pub fn shift_change(workspace: &Workspace, trace: Id, offset_s: f64) -> Change {
    let alignment = Alignment { offset_s, provenance: manual(workspace), locked: true };
    Change::SetAlignment { trace, alignment: Some(alignment), revert: false }
}

/// A manual pick of `phase` on `track` where it is drawn at `at` seconds,
/// replacing any pick of that phase. `None` while the trace is missing.
fn manual_pick(workspace: &Workspace, track: &Track, phase: &str, at: f64, uncertainty_s: f64) -> Option<Change> {
    let project = &workspace.project;
    let trace = project.trace(track.trace)?;
    let reference = project.reference.unwrap_or(trace.start);
    // Drawn = corrected + offset; recorded = corrected − the trace's clock correction.
    let corrected = reference.add_seconds(at - track.offset);
    let correction = project.corrected_start(trace).seconds_since(trace.start);
    let id = project.trace_picks(trace.id).find(|p| p.phase == phase).map_or_else(Id::generate, |p| p.id);
    let pick = Pick {
        id,
        trace: trace.id,
        time: corrected.add_seconds(-correction),
        phase: phase.to_string(),
        uncertainty_s,
        provenance: manual(workspace),
    };
    Some(Change::SetPick { pick, revert: false })
}

/// Phase a key places on `track`: `P` the first phase, `S` the phase named S.
fn key_phase(workspace: &Workspace, track: &Track, key: egui::Key) -> Option<String> {
    let project = &workspace.project;
    let propagation = &project.trace_phenomenology(project.trace(track.trace)?)?.propagation;
    match key {
        egui::Key::P => Some(propagation.first_phase().unwrap_or("").to_string()),
        egui::Key::S => propagation.speed("S").map(|_| "S".to_string()),
        _ => None,
    }
}

fn phase_label(phase: &str) -> &str {
    if phase.is_empty() { "onset" } else { phase }
}

/// Per-track options, shown from a right-click on its row or label.
fn track_menu(ui: &mut egui::Ui, track: &mut Track, view: &mut TrackView, workspace: &mut Workspace, width: f64) {
    ui.label(egui::RichText::new(&track.code).strong());
    ui.separator();
    ui.label("Show");
    for display in RowDisplay::ALL {
        ui.radio_value(&mut track.display, display, display.label());
    }
    if track.display.has_spectrogram() {
        ui.horizontal(|ui| {
            ui.label("FFT size");
            egui::ComboBox::from_id_salt("nfft").selected_text(track.nfft.to_string()).show_ui(ui, |ui| {
                for n in NFFT_CHOICES {
                    ui.selectable_value(&mut track.nfft, n, n.to_string());
                }
            });
        });
    }
    ui.separator();
    ui.label("Y scale");
    for scale in YScale::ALL {
        ui.radio_value(&mut track.y_scale, scale, scale.label());
    }
    if track.y_scale == YScale::Fixed {
        ui.horizontal(|ui| {
            ui.label("±");
            ui.add(egui::DragValue::new(&mut track.fixed_half_range).speed(0.01).range(1e-12..=f32::MAX));
        });
    }
    if track.kind == SampleKind::Complex {
        ui.separator();
        ui.label("Plot I/Q as");
        for component in IqComponent::ALL {
            ui.radio_value(&mut track.component, component, component.label());
        }
    }

    ui.separator();
    let is_reference = view.reference == Some(track.trace);
    if ui.add_enabled(!is_reference, egui::Button::new("Use as alignment reference")).clicked() {
        view.reference = Some(track.trace);
        ui.close();
    }
    let alignment = workspace.project.trace(track.trace).and_then(|t| t.alignment.clone());
    if let Some(alignment) = alignment {
        let label = if alignment.locked { "Unlock offset" } else { "Lock offset" };
        if ui.button(label).on_hover_text("Locked offsets are left alone by automatic alignment").clicked() {
            let next = Alignment { provenance: manual(workspace), locked: !alignment.locked, ..alignment };
            let verb = if next.locked { "Lock" } else { "Unlock" };
            let change = Change::SetAlignment { trace: track.trace, alignment: Some(next), revert: false };
            workspace.execute_or_report(format!("{verb} {}", track.code), vec![change]);
            ui.close();
        }
        if ui.button("Reset offset").clicked() {
            let change = Change::SetAlignment { trace: track.trace, alignment: None, revert: false };
            workspace.execute_or_report(format!("Reset {}", track.code), vec![change]);
            ui.close();
        }
    }
    if !track.picks.is_empty() {
        ui.separator();
        ui.label("Picks");
        for pick in &track.picks {
            let kind = if pick.manual { "manual" } else { "auto" };
            if ui.button(format!("Remove {} ({kind})", phase_label(&pick.phase))).clicked() {
                let change = Change::RemovePick { pick: pick.id };
                workspace.execute_or_report(format!("Remove pick on {}", track.code), vec![change]);
                ui.close();
            }
        }
    }
    ui.separator();
    if ui.button("Fit time to this trace").clicked() {
        view.time.fit(track.drawn_start(), track.end(), width);
        ui.close();
    }
}

/// Shifts that remove `entity`'s moveout from the saved origin, put its picks
/// on the reference's, or line it up with the reference by correlation.
fn snap_targets(view: &TrackView, tracks: &Query<(Entity, &mut Track, Has<SeriesTask>)>, entity: Entity, width: f64) -> Vec<(f64, String)> {
    let Ok((_, dragged, _)) = tracks.get(entity) else {
        return Vec::new();
    };
    // Moveout: the arrival drawn at the origin time.
    let mut snaps: Vec<(f64, String)> =
        dragged.predicted.iter().map(|p| (-p.travel - dragged.offset, format!("moveout {}", phase_label(&p.phase)))).collect();
    let Some(reference) = view.reference else {
        return snaps;
    };
    let Some((_, anchor, _)) = tracks.iter().find(|(_, t, _)| t.trace == reference) else {
        return snaps;
    };
    if anchor.trace == dragged.trace {
        return snaps;
    }
    for pick in &dragged.picks {
        if let Some(target) = anchor.picks.iter().find(|p| p.phase == pick.phase) {
            let shift = anchor.drawn(target.at) - (pick.at + dragged.offset);
            snaps.push((shift, format!("on ★ {}", phase_label(&pick.phase))));
        }
    }
    if anchor.phenomenology_id.is_none() || anchor.phenomenology_id != dragged.phenomenology_id {
        return snaps;
    }
    snaps.extend(correlation_snaps(view, anchor, dragged, width));
    snaps
}

/// Correlation peaks of `dragged` against `anchor` over the visible window.
fn correlation_snaps(view: &TrackView, anchor: &Track, dragged: &Track, width: f64) -> Vec<(f64, String)> {
    let from = view.time.start.max(anchor.drawn_start());
    let to = view.time.end(width).min(anchor.end());
    if to <= from {
        return Vec::new();
    }
    let max_lag_s = (to - from) / 2.0;
    let margin = 64.0 / anchor.rate.min(dragged.rate);
    let (Some((a, a_start)), Some((b, b_start))) = (
        anchor.window(from - margin, to + margin),
        dragged.window(from - max_lag_s - margin, to + max_lag_s + margin),
    ) else {
        return Vec::new();
    };
    let settings = AlignSettings { method: LagMethod::CrossCorrelation, from, to, max_lag_s, max_samples: SNAP_SAMPLES };
    let a = Channel { values: &a, rate: anchor.rate, start: a_start };
    let b = Channel { values: &b, rate: dragged.rate, start: b_start };
    candidates(&a, &b, &settings, SNAP_MIN).into_iter().map(|(lag, r)| (-lag, format!("r = {r:.2}"))).collect()
}

/// Labels, time axis, cursor and mouse handling for the tracks view.
pub fn tracks_ui(
    mut contexts: EguiContexts,
    plot: Res<Plot>,
    mode: Res<PlotMode>,
    mut workspace: ResMut<Workspace>,
    mut view: ResMut<TrackView>,
    mut tracks: Query<(Entity, &mut Track, Has<SeriesTask>)>,
) -> Result {
    if *mode != PlotMode::Tracks || !plot.is_usable() {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?.clone();
    let v = &plot.view;
    let full = egui::Rect::from_min_size(
        egui::pos2(plot.origin.x, plot.origin.y),
        egui::vec2(v.width as f32, v.height as f32),
    );
    let area = egui::Rect::from_min_size(
        full.min + egui::vec2(v.insets.left as f32, v.insets.top as f32),
        egui::vec2(v.plot_width() as f32, v.plot_height() as f32),
    );
    let width = v.plot_width();
    let ui = egui::Ui::new(
        ctx.clone(),
        "squiggle_tracks".into(),
        egui::UiBuilder::new().layer_id(egui::LayerId::background()).max_rect(full),
    );
    let painter = ui.painter_at(full);
    let small = egui::FontId::proportional(11.0);
    let mono = egui::FontId::monospace(10.0);

    let response = ui.interact(area, ui.id().with("area"), egui::Sense::click_and_drag());
    let pointer = response.hover_pos();

    // Rows: separators and labels.
    let mut rows: Vec<(Entity, egui::Rangef)> = Vec::new();
    let mut label_hovered = false;
    for (entity, mut track, building) in &mut tracks {
        let Some((top, height)) = track.row else {
            continue;
        };
        let y = egui::Rangef::new(area.min.y + (top - v.insets.top) as f32, area.min.y + (top - v.insets.top + height) as f32);
        if y.max < area.min.y || y.min > area.max.y {
            continue;
        }
        rows.push((entity, y));
        painter.hline(full.min.x..=area.max.x, y.max, egui::Stroke::new(1.0, GRID_COLOR));

        let label = egui::Rect::from_x_y_ranges(full.min.x..=area.min.x - 1.0, y);
        let label_response = ui.interact(label, ui.id().with(("label", entity)), egui::Sense::click());
        label_hovered |= label_response.hovered();
        if label_response.clicked() {
            track.selected = !track.selected;
        }
        label_response.context_menu(|ui| track_menu(ui, &mut track, &mut view, &mut workspace, width));

        let clip = painter.with_clip_rect(label.intersect(area.with_min_x(full.min.x)));
        if track.selected {
            clip.rect_filled(label, 0.0, SELECTED_FILL);
        }
        clip.rect_filled(egui::Rect::from_x_y_ranges(label.min.x + 2.0..=label.min.x + 6.0, y.shrink(3.0)), 1.0, color32(track.color));
        let x = label.min.x + 12.0;
        let mut line_y = y.min + 4.0;
        let title = if view.reference == Some(track.trace) { format!("★ {}", track.code) } else { track.code.clone() };
        clip.text(egui::pos2(x, line_y), egui::Align2::LEFT_TOP, title, egui::FontId::proportional(12.0), egui::Color32::WHITE);
        line_y += 15.0;
        let (phenomenology, color) = match &track.phenomenology {
            Some(name) => (name.as_str(), color32(track.color)),
            None => ("unassigned", color32(UNASSIGNED_COLOR)),
        };
        let mut detail = format!("{phenomenology} · {} Hz", trim_number(track.rate));
        if track.kind == SampleKind::Complex {
            detail.push_str(&format!(" · IQ {}", track.component.label()));
        }
        if let Some(distance) = track.distance_m {
            detail.push_str(&format!(" · {:.1} km", distance / 1e3));
        }
        if building || track.series.is_none() {
            detail.push_str(" · building…");
        }
        clip.text(egui::pos2(x, line_y), egui::Align2::LEFT_TOP, detail, small.clone(), color);
        if track.display.has_spectrogram() {
            let nyquist = trim_number(track.rate / 2.0);
            let (high, low) = match track.kind {
                SampleKind::Real => (format!("{nyquist} Hz"), "0 Hz".to_string()),
                SampleKind::Complex => (format!("+{nyquist} Hz"), format!("−{nyquist} Hz")),
            };
            let scale = painter.with_clip_rect(area);
            let gray = egui::Color32::from_gray(190);
            scale.text(egui::pos2(area.max.x - 4.0, y.min + 2.0), egui::Align2::RIGHT_TOP, high, mono.clone(), gray);
            scale.text(egui::pos2(area.max.x - 4.0, y.max - 2.0), egui::Align2::RIGHT_BOTTOM, low, mono.clone(), gray);
        }
        line_y += 14.0;
        if y.span() >= 52.0 {
            let mut state = Vec::new();
            let shift = track.offset + track.preview;
            if shift != 0.0 {
                state.push(format!("Δ{} s{}", signed(shift), if track.locked { " 🔒" } else { "" }));
            }
            if !track.processing.is_identity() {
                state.push(if view.show_raw { "raw".to_string() } else { track.processing.describe() });
            }
            if let Some(error) = track.series.as_ref().and_then(|s| s.error.as_ref()) {
                state.push(format!("⚠ {error}"));
            }
            if !state.is_empty() {
                let color = if track.preview != 0.0 { SHIFT_COLOR } else { egui::Color32::from_gray(160) };
                clip.text(egui::pos2(x, line_y), egui::Align2::LEFT_TOP, state.join(" · "), small.clone(), color);
            }
        }
    }

    // Predicted arrivals (dashed) and picks (white manual, blue automatic).
    let marks = painter.with_clip_rect(area);
    for (entity, y) in &rows {
        let Ok((_, track, _)) = tracks.get(*entity) else {
            continue;
        };
        let x_of = |t: f64| area.min.x + view.time.x_of(track.drawn(t)) as f32;
        for p in &track.predicted {
            let x = x_of(p.at);
            let stroke = egui::Stroke::new(1.0, PREDICTED_COLOR);
            marks.add(egui::Shape::dashed_line(&[egui::pos2(x, y.min), egui::pos2(x, y.max)], stroke, 4.0, 3.0));
            let label = format!("{} pred", phase_label(&p.phase));
            marks.text(egui::pos2(x + 2.0, y.max - 2.0), egui::Align2::LEFT_BOTTOM, label, mono.clone(), PREDICTED_COLOR);
        }
        for p in &track.picks {
            let x = x_of(p.at);
            let color = if p.manual { MANUAL_PICK_COLOR } else { AUTO_PICK_COLOR };
            marks.vline(x, *y, egui::Stroke::new(1.5, color));
            marks.text(egui::pos2(x + 2.0, y.min + 2.0), egui::Align2::LEFT_TOP, phase_label(&p.phase), mono.clone(), color);
        }
    }

    // Time axis.
    let axis_y = area.max.y;
    painter.hline(area.x_range(), axis_y, egui::Stroke::new(1.0, AXIS_COLOR));
    for tick in view.time.ticks(width) {
        let x = area.min.x + tick.position as f32;
        let length = if tick.is_major() { 6.0 } else { 3.0 };
        painter.vline(x, axis_y..=axis_y + length, egui::Stroke::new(1.0, AXIS_COLOR));
        if let Some(label) = tick.label {
            painter.text(egui::pos2(x, axis_y + 8.0), egui::Align2::CENTER_TOP, label, mono.clone(), AXIS_COLOR);
        }
    }
    let reference = workspace.project.reference;
    let caption = match reference {
        Some(reference) => format!("seconds from\n{}", compact_time(&reference.to_string())),
        None => "seconds".to_string(),
    };
    painter.text(
        egui::pos2(full.min.x + 4.0, axis_y + 3.0),
        egui::Align2::LEFT_TOP,
        caption,
        egui::FontId::monospace(9.0),
        egui::Color32::from_gray(140),
    );

    // Mouse: scroll zooms time (rows over the labels), horizontal scroll pans,
    // pinch or Ctrl+scroll scales amplitude, drag pans time and rows.
    if response.hovered() || label_hovered {
        let (scroll, zoom) = ctx.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
        if label_hovered {
            view.scroll -= f64::from(scroll.y);
        } else if let Some(pointer) = pointer {
            if scroll.y != 0.0 {
                view.time.zoom_about(f64::from(pointer.x - area.min.x), ZOOM_PER_POINT.powf(f64::from(scroll.y)));
            }
            view.time.pan(f64::from(scroll.x));
        }
        if zoom != 1.0 {
            view.gain = (view.gain * f64::from(zoom)).clamp(1e-3, 1e6);
        }
    }
    let row_at = |y: f32| rows.iter().find(|(_, r)| r.contains(y)).map(|(e, _)| *e);
    let (shift_held, alt_held) = ctx.input(|i| (i.modifiers.shift, i.modifiers.alt));

    // Shift+drag a row: shift the trace in time, snapping to correlation
    // peaks against the reference (Alt disables snapping).
    if response.drag_started()
        && shift_held
        && let Some(entity) = pointer.and_then(|p| row_at(p.y))
    {
        let snaps = snap_targets(&view, &tracks, entity, width);
        view.drag = Some(ShiftDrag { entity, raw: 0.0, snaps, snapped: None });
    }
    if response.dragged() {
        let delta = response.drag_delta();
        let px_per_s = view.time.px_per_s;
        match &mut view.drag {
            Some(drag) => {
                drag.raw += f64::from(delta.x) / px_per_s;
                let snap = (!alt_held)
                    .then(|| drag.snaps.iter().find(|(s, _)| ((s - drag.raw) * px_per_s).abs() < SNAP_PX))
                    .flatten();
                drag.snapped = snap.map(|(_, why)| why.clone());
                let preview = snap.map_or(drag.raw, |(s, _)| *s);
                if let Ok((_, mut track, _)) = tracks.get_mut(drag.entity) {
                    track.preview = preview;
                }
            }
            None => {
                view.time.pan(f64::from(delta.x));
                view.scroll -= f64::from(delta.y);
            }
        }
    }
    if response.drag_stopped()
        && let Some(drag) = view.drag.take()
        && let Ok((_, mut track, _)) = tracks.get_mut(drag.entity)
        && track.preview != 0.0
    {
        let change = shift_change(&workspace, track.trace, track.offset + track.preview);
        if !workspace.execute_or_report(format!("Shift {}", track.code), vec![change]) {
            track.preview = 0.0;
        }
    }

    // Arrow keys nudge selected tracks by one sample (ten with Shift).
    if view.drag.is_none() && !ctx.egui_wants_keyboard_input() {
        let (left, right) = ctx.input(|i| (i.key_pressed(egui::Key::ArrowLeft), i.key_pressed(egui::Key::ArrowRight)));
        if left != right {
            let steps = if shift_held { 10.0 } else { 1.0 } * if right { 1.0 } else { -1.0 };
            let changes: Vec<Change> = tracks
                .iter()
                .filter(|(_, t, _)| t.selected)
                .map(|(_, t, _)| shift_change(&workspace, t.trace, t.offset + steps / t.rate))
                .collect();
            if !changes.is_empty() {
                let label = format!("Nudge {} sample{}", signed(steps), if steps.abs() == 1.0 { "" } else { "s" });
                workspace.execute_or_report(label, changes);
            }
        }
    }

    // Over a row: P / S place a manual pick at the cursor, Delete removes the nearest.
    if view.drag.is_none()
        && !ctx.egui_wants_keyboard_input()
        && let Some(pointer) = pointer.filter(|p| area.contains(*p))
        && let Some(entity) = row_at(pointer.y)
        && let Ok((_, track, _)) = tracks.get(entity)
    {
        let at = view.time.time_at(f64::from(pointer.x - area.min.x));
        let px_per_s = view.time.px_per_s;
        let (key, delete) = ctx.input(|i| {
            let key = [egui::Key::P, egui::Key::S].into_iter().find(|k| i.key_pressed(*k) && !i.modifiers.command);
            (key, i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete))
        });
        if let Some(phase) = key.and_then(|k| key_phase(&workspace, track, k)) {
            // A pixel at this zoom, but no finer than a sample.
            let uncertainty_s = (1.0 / track.rate).max(1.0 / px_per_s);
            if let Some(change) = manual_pick(&workspace, track, &phase, at, uncertainty_s) {
                workspace.execute_or_report(format!("Pick {} on {}", phase_label(&phase), track.code), vec![change]);
            }
        }
        let nearest = track
            .picks
            .iter()
            .map(|p| (p.id, ((track.drawn(p.at) - at) * px_per_s).abs()))
            .filter(|(_, d)| *d < SNAP_PX)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let (true, Some((pick, _))) = (delete, nearest) {
            workspace.execute_or_report(format!("Remove pick on {}", track.code), vec![Change::RemovePick { pick }]);
        }
    }

    if response.double_clicked() {
        view.fit_requested = true;
    } else if response.clicked()
        && let Some(entity) = pointer.and_then(|p| row_at(p.y))
        && let Ok((_, mut track, _)) = tracks.get_mut(entity)
    {
        track.selected = !track.selected;
    }
    if response.secondary_clicked() {
        view.menu_track = pointer.and_then(|p| row_at(p.y));
    }
    response.context_menu(|ui| {
        if let Some(entity) = view.menu_track
            && let Ok((_, mut track, _)) = tracks.get_mut(entity)
        {
            track_menu(ui, &mut track, &mut view, &mut workspace, width);
        } else {
            ui.close();
        }
    });

    // Shift readout beside the pointer.
    if let (Some(drag), Some(pointer)) = (&view.drag, pointer)
        && let Ok((_, track, _)) = tracks.get(drag.entity)
    {
        let mut text = format!("Δ{} s", signed(track.offset + track.preview));
        if let Some(why) = &drag.snapped {
            text.push_str(&format!("  snap {why}"));
        }
        let galley = painter.layout_no_wrap(text, mono.clone(), SHIFT_COLOR);
        let rect = egui::Rect::from_min_size(pointer + egui::vec2(12.0, -18.0), galley.size()).expand(3.0);
        painter.rect_filled(rect, 2.0, egui::Color32::from_black_alpha(200));
        painter.galley(rect.min + egui::vec2(3.0, 3.0), galley, SHIFT_COLOR);
    }

    // Cursor: shared vertical line, relative and absolute time.
    if let Some(pointer) = pointer.filter(|p| area.contains(*p))
        && plot.crosshairs_visible
    {
        painter.vline(pointer.x, area.y_range(), egui::Stroke::new(1.0, CURSOR_COLOR));
        let time = view.time.time_at(f64::from(pointer.x - area.min.x));
        let resolution = 1.0 / view.time.px_per_s;
        let decimals = (-resolution.log10().floor()).clamp(0.0, 12.0) as usize;
        let text = format!("{time:.decimals$} s");
        let at = egui::pos2(pointer.x, axis_y + 8.0);
        let galley = painter.layout_no_wrap(text, mono.clone(), CURSOR_COLOR);
        let rect = egui::Align2::CENTER_TOP.anchor_size(at, galley.size()).expand(2.0);
        painter.rect_filled(rect, 2.0, egui::Color32::BLACK);
        painter.galley(rect.min + egui::vec2(2.0, 2.0), galley, CURSOR_COLOR);

        let absolute = match reference {
            Some(reference) => format!("{}  (1 px = {})", reference.add_seconds(time), format_duration(resolution)),
            None => format!("1 px = {}", format_duration(resolution)),
        };
        let galley = painter.layout_no_wrap(absolute, mono, CURSOR_COLOR);
        let rect = egui::Rect::from_min_size(area.right_top() + egui::vec2(-galley.size().x - 8.0, 4.0), galley.size()).expand(3.0);
        painter.rect_filled(rect, 2.0, egui::Color32::from_black_alpha(200));
        painter.galley(rect.min + egui::vec2(3.0, 3.0), galley, CURSOR_COLOR);
    }
    Ok(())
}

/// Drops trailing zeros from an ISO timestamp's fraction: `…:00.500000000Z` → `…:00.5Z`.
fn compact_time(iso: &str) -> String {
    match iso.strip_suffix('Z').and_then(|body| body.split_once('.')) {
        Some((whole, fraction)) => {
            let fraction = fraction.trim_end_matches('0');
            if fraction.is_empty() { format!("{whole}Z") } else { format!("{whole}.{fraction}Z") }
        }
        None => iso.to_string(),
    }
}

/// `+0.8125`, `-3e-9`: a signed shift with up to 9 significant digits.
fn signed(value: f64) -> String {
    let text = if value != 0.0 && value.abs() < 1e-3 { format!("{value:+.3e}") } else { format!("{value:+.6}") };
    if text.contains('e') { text } else { text.trim_end_matches('0').trim_end_matches('.').to_string() }
}

/// `40`, `0.5`, `1e9` style.
fn trim_number(value: f64) -> String {
    if value >= 1e6 {
        format!("{value:.3e}")
    } else {
        let text = format!("{value:.3}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(rate: f64, len: usize) -> Track {
        Track {
            trace: Id::generate(),
            code: "c".into(),
            phenomenology: None,
            phenomenology_id: None,
            color: Rgba::WHITE,
            start: 10.0,
            offset: 0.0,
            locked: false,
            processing: Processing::default(),
            preview: 0.0,
            display: RowDisplay::default(),
            nfft: 256,
            picks: Vec::new(),
            predicted: Vec::new(),
            distance_m: None,
            rate,
            len,
            kind: SampleKind::Real,
            selected: false,
            y_scale: YScale::default(),
            fixed_half_range: 1.0,
            component: IqComponent::default(),
            series: Some(Arc::new(Series::build(
                &TraceSamples::Real(ChunkedSamples::from_vec((0..len).map(|i| i as f32).collect())),
                IqComponent::default(),
                Processing::default(),
                rate,
            ))),
            row: None,
            index: 0,
            cache: None,
        }
    }

    #[test]
    fn buckets_are_powers_of_two_with_hysteresis() {
        assert_eq!(wanted_bucket(0.2), 1);
        assert_eq!(wanted_bucket(5.0), 4);
        assert_eq!(wanted_bucket(1e9), 1 << 29);
        assert!(bucket_suits(4, 5.0) && bucket_suits(4, 10.0) && !bucket_suits(4, 12.0) && !bucket_suits(4, 1.0));
        assert!(bucket_suits(1, 0.001) && !bucket_suits(1, 4.0));
    }

    #[test]
    fn indices_clamp_to_the_trace() {
        let t = track(100.0, 1000);
        assert_eq!(t.indices(0.0, 100.0), (0, 1000));
        assert_eq!(t.indices(11.0, 12.0), (100, 201));
        assert_eq!(t.indices(30.0, 40.0), (1000, 1000));
    }

    #[test]
    fn transform_windows_are_bounded_and_timed() {
        let t = track(1e9, 4_000_000);
        let mut time = TimeView::default();
        time.fit(10.0, 10.0 + 1e-6, 1000.0);
        let squiggle = t.window_squiggle(&time, 1000.0).unwrap();
        assert!(squiggle.len() > 1000);
        assert!((squiggle.points()[0].x - 10.0).abs() < 1e-8);
        time.fit(10.0, 10.004, 1000.0);
        assert!(t.window_squiggle(&time, 1000.0).unwrap_err().contains("zoom in"));
    }

    #[test]
    fn timestamps_compact() {
        assert_eq!(compact_time("2024-01-01T07:05:00.000000000Z"), "2024-01-01T07:05:00Z");
        assert_eq!(compact_time("2024-01-01T07:05:00.019500000Z"), "2024-01-01T07:05:00.0195Z");
    }

    #[test]
    fn previews_move_the_drawn_window() {
        let mut t = track(100.0, 1000);
        t.preview = 1.0;
        assert_eq!(t.indices(11.0, 12.0), (0, 101));
        assert_eq!(t.end(), 21.0);
    }

    #[test]
    fn manual_picks_land_where_drawn() {
        use crate::workspace::SyntheticEvent;
        use squiggle_core::Analyst;
        use squiggle_core::sensor::ClockCorrection;

        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Located).unwrap();
        let trace = workspace.project.traces()[3].clone();
        let clock = ClockCorrection { offset_s: 0.2, drift_ppm: 0.0 };
        let changes = vec![shift_change(&workspace, trace.id, 0.75), Change::SetClockCorrection { sensor: trace.sensor, clock }];
        workspace.execute("setup", changes).unwrap();
        let mut t = track(100.0, 1000);
        (t.trace, t.offset) = (trace.id, 0.75);
        let change = manual_pick(&workspace, &t, "P", 12.5, 0.01).unwrap();
        workspace.execute("pick", vec![change]).unwrap();

        let project = &workspace.project;
        let pick = &project.picks()[0];
        let corrected = project.pick_time(pick).unwrap().seconds_since(project.reference.unwrap());
        assert!((t.drawn(corrected) - 12.5).abs() < 1e-9, "{corrected}");
        assert!(pick.provenance.is_manual());
        // A second P replaces the first.
        let again = manual_pick(&workspace, &t, "P", 13.0, 0.01).unwrap();
        workspace.execute("again", vec![again]).unwrap();
        assert_eq!(workspace.project.picks().len(), 1);
    }

    #[test]
    fn shifts_print_signed() {
        assert_eq!(signed(0.8125), "+0.8125");
        assert_eq!(signed(-2.0), "-2");
        assert_eq!(signed(3e-9), "+3.000e-9");
    }

    #[test]
    fn numbers_trim() {
        assert_eq!(trim_number(40.0), "40");
        assert_eq!(trim_number(0.5), "0.5");
        assert_eq!(trim_number(1e9), "1.000e9");
    }
}
