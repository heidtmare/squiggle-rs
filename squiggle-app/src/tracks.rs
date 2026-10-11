//! Stacked tracks: one row per loaded trace on a shared time axis.
//!
//! Each track draws its [`Envelope`] decimated to about one bucket per pixel,
//! cached over three screen widths so panning and moderate zoom reuse the
//! mesh. Mesh x is seconds from the cache start and the entity transform is
//! computed in `f64`, so nanosecond spacing survives far from the reference.

use std::collections::HashSet;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::timeline::format_duration;
use squiggle_core::{
    ChunkedSamples, Envelope, Id, IqComponent, MinMax, Point, Rgba, SampleKind, Squiggle, Style, TimeView,
    TraceSamples,
};

use crate::plot::line::{self, SquiggleLineMaterial};
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

const GRID_COLOR: egui::Color32 = egui::Color32::from_gray(45);
const AXIS_COLOR: egui::Color32 = egui::Color32::from_gray(200);
const CURSOR_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 215, 0);
const SELECTED_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 60, 60, 120);

pub struct TracksPlugin;

impl Plugin for TracksPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TrackView>()
            .add_systems(Update, (sync_tracks, request_series, poll_series, layout_rows, draw_tracks).chain());
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowOrder {
    #[default]
    Phenomenology,
    Sensor,
    Start,
}

impl RowOrder {
    pub const ALL: [Self; 3] = [Self::Phenomenology, Self::Sensor, Self::Start];

    pub fn label(self) -> &'static str {
        match self {
            Self::Phenomenology => "Phenomenology",
            Self::Sensor => "Sensor",
            Self::Start => "Start time",
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

#[derive(Resource)]
pub struct TrackView {
    pub time: TimeView,
    /// Multiplies every row's amplitude.
    pub gain: f64,
    /// Rows scrolled off the top, in pixels.
    pub scroll: f64,
    pub order: RowOrder,
    pub fit_requested: bool,
    /// Row whose context menu is open.
    menu_track: Option<Entity>,
}

impl Default for TrackView {
    fn default() -> Self {
        Self {
            time: TimeView::default(),
            gain: 1.0,
            scroll: 0.0,
            order: RowOrder::default(),
            fit_requested: false,
            menu_track: None,
        }
    }
}

/// A trace's plotted series: real samples, or one component of I/Q.
pub struct Series {
    pub component: IqComponent,
    pub values: ChunkedSamples,
    pub envelope: Envelope,
    pub full: MinMax,
}

impl Series {
    fn build(samples: &TraceSamples, component: IqComponent) -> Self {
        let values = samples.series(component);
        let envelope = Envelope::build(&values);
        let full = envelope.range(&values, 0, values.len());
        Self { component, values, envelope, full }
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
    pub color: Rgba,
    /// Clock-corrected start, seconds from the project reference.
    pub start: f64,
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
    pub fn end(&self) -> f64 {
        self.start + self.len as f64 / self.rate
    }

    /// The component to plot; real traces ignore it.
    fn wanted(&self) -> IqComponent {
        match self.kind {
            SampleKind::Real => IqComponent::default(),
            SampleKind::Complex => self.component,
        }
    }

    /// Sample indices covering `from..to` seconds, clamped to the trace.
    fn indices(&self, from: f64, to: f64) -> (usize, usize) {
        let index = |t: f64| ((t - self.start) * self.rate).clamp(0.0, self.len as f64);
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
            .map(|(i, v)| Point::new(self.start + (lo + i) as f64 / self.rate, f64::from(v)));
        let mut squiggle = Squiggle::with_points(self.name(), points).with_style(Style::new(self.color, 1.5));
        squiggle.trace = Some(self.trace);
        Ok(squiggle)
    }
}

#[derive(Component)]
pub struct SeriesTask(Task<Series>);

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
        (
            project.sensor(trace.sensor).map_or("?", |s| s.code.as_str()).to_string(),
            phenomenology.map(|p| p.name.clone()),
            phenomenology.map_or(UNASSIGNED_COLOR, |p| p.color),
            project.corrected_start(trace).seconds_since(reference),
        )
    };

    let mut present = HashSet::new();
    for (entity, mut track) in &mut tracks {
        match (project.trace(track.trace), workspace.samples.get(&track.trace)) {
            (Some(trace), Some(SampleState::Loaded(_))) => {
                present.insert(track.trace);
                let (code, phenomenology, color, start) = describe(trace);
                track.code = code;
                track.phenomenology = phenomenology;
                track.color = color;
                track.start = start;
            }
            _ => commands.entity(entity).despawn(),
        }
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
        let (code, phenomenology, color, start) = describe(trace);
        let material = materials.add(SquiggleLineMaterial::new(color, LINE_WIDTH, plot.scale_factor));
        commands.spawn((
            Track {
                trace: trace.id,
                code,
                phenomenology,
                color,
                start,
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

/// Builds each track's series off-thread when missing or its component changed.
fn request_series(
    mut commands: Commands,
    workspace: Res<Workspace>,
    tracks: Query<(Entity, &Track), Without<SeriesTask>>,
) {
    for (entity, track) in &tracks {
        let wanted = track.wanted();
        if track.series.as_ref().is_some_and(|s| s.component == wanted) {
            continue;
        }
        let Some(SampleState::Loaded(samples)) = workspace.samples.get(&track.trace) else {
            continue;
        };
        let samples = samples.clone();
        let task = AsyncComputeTaskPool::get().spawn(async move { Series::build(&samples, wanted) });
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
        RowOrder::Start => rows.sort_by(|a, b| a.start.total_cmp(&b.start).then_with(|| a.code.cmp(&b.code))),
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
        let from = rows.iter().map(|t| t.start).fold(f64::INFINITY, f64::min);
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
        let (Some(series), Some((top, height)), true) = (track.series.clone(), track.row, active) else {
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
        let cache_start = track.start + cache.lo as f64 / track.rate;
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

/// Per-track options, shown from a right-click on its row or label.
fn track_menu(ui: &mut egui::Ui, track: &mut Track, view: &mut TrackView, width: f64) {
    ui.label(egui::RichText::new(&track.code).strong());
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
    if ui.button("Fit time to this trace").clicked() {
        view.time.fit(track.start, track.end(), width);
        ui.close();
    }
}

/// Labels, time axis, cursor and mouse handling for the tracks view.
pub fn tracks_ui(
    mut contexts: EguiContexts,
    plot: Res<Plot>,
    mode: Res<PlotMode>,
    workspace: Res<Workspace>,
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
        label_response.context_menu(|ui| track_menu(ui, &mut track, &mut view, width));

        let clip = painter.with_clip_rect(label.intersect(area.with_min_x(full.min.x)));
        if track.selected {
            clip.rect_filled(label, 0.0, SELECTED_FILL);
        }
        clip.rect_filled(egui::Rect::from_x_y_ranges(label.min.x + 2.0..=label.min.x + 6.0, y.shrink(3.0)), 1.0, color32(track.color));
        let x = label.min.x + 12.0;
        let mut line_y = y.min + 4.0;
        clip.text(egui::pos2(x, line_y), egui::Align2::LEFT_TOP, &track.code, egui::FontId::proportional(12.0), egui::Color32::WHITE);
        line_y += 15.0;
        let (phenomenology, color) = match &track.phenomenology {
            Some(name) => (name.as_str(), color32(track.color)),
            None => ("unassigned", color32(UNASSIGNED_COLOR)),
        };
        let mut detail = format!("{phenomenology} · {} Hz", trim_number(track.rate));
        if track.kind == SampleKind::Complex {
            detail.push_str(&format!(" · IQ {}", track.component.label()));
        }
        if building || track.series.is_none() {
            detail.push_str(" · building…");
        }
        clip.text(egui::pos2(x, line_y), egui::Align2::LEFT_TOP, detail, small.clone(), color);
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
    if response.dragged() {
        let delta = response.drag_delta();
        view.time.pan(f64::from(delta.x));
        view.scroll -= f64::from(delta.y);
    }
    let row_at = |y: f32| rows.iter().find(|(_, r)| r.contains(y)).map(|(e, _)| *e);
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
            track_menu(ui, &mut track, &mut view, width);
        } else {
            ui.close();
        }
    });

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
            color: Rgba::WHITE,
            start: 10.0,
            rate,
            len,
            kind: SampleKind::Real,
            selected: false,
            y_scale: YScale::default(),
            fixed_half_range: 1.0,
            component: IqComponent::default(),
            series: Some(Arc::new(Series::build(&TraceSamples::Real(ChunkedSamples::from_vec((0..len).map(|i| i as f32).collect())), IqComponent::default()))),
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
    fn numbers_trim() {
        assert_eq!(trim_number(40.0), "40");
        assert_eq!(trim_number(0.5), "0.5");
        assert_eq!(trim_number(1e9), "1.000e9");
    }
}
