//! Scatter view of events: any two event features, or the PCA coordinates of
//! an imported cluster package (`squiggle-train cluster`), coloured by
//! analyst label, consensus, cluster or status. Click or lasso to select;
//! label, accept or reject the selection in one command. Double-click
//! reviews an event.

use std::collections::HashSet;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::features::{EVENT_FEATURES, FeatureSettings, TraceData, features};
use squiggle_core::{Change, EventStatus, Id, LabelId, Project};
use squiggle_models::Package;

use crate::classify_ui::{accept_change, all_events, label_change, label_picker, trace_data, track_values};
use crate::events_ui::{open_event, status_change};
use crate::models_ui::ModelRegistry;
use crate::plot::Plot;
use crate::tracks::{Track, TrackView};
use crate::workspace::Workspace;

const PALETTE: [egui::Color32; 10] = [
    egui::Color32::from_rgb(110, 170, 240),
    egui::Color32::from_rgb(240, 140, 70),
    egui::Color32::from_rgb(120, 200, 120),
    egui::Color32::from_rgb(210, 110, 200),
    egui::Color32::from_rgb(240, 210, 90),
    egui::Color32::from_rgb(90, 210, 210),
    egui::Color32::from_rgb(230, 100, 110),
    egui::Color32::from_rgb(160, 140, 240),
    egui::Color32::from_rgb(190, 160, 110),
    egui::Color32::from_rgb(150, 220, 170),
];
const NONE_COLOR: egui::Color32 = egui::Color32::from_gray(130);

/// One event's coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterPoint {
    pub event: Id,
    /// In `EVENT_FEATURES` order.
    pub features: Vec<f64>,
    /// PCA coordinates and cluster, from the chosen package.
    pub embedding: Option<(Vec<f64>, Option<usize>)>,
}

/// Every event's features (with the package's feature settings when given) and embedding.
pub fn scatter_points(project: &Project, traces: &[TraceData], package: Option<&Package>) -> Vec<ScatterPoint> {
    let settings = package.map_or_else(FeatureSettings::default, |p| p.spec.input.settings);
    all_events(project, traces)
        .into_iter()
        .map(|(event, data)| {
            let event_features = features(traces, &data, &settings).event;
            let embedding = package.and_then(|p| p.embed(&event_features));
            ScatterPoint { event, features: event_features, embedding }
        })
        .collect()
}

/// Labels every event in `ids` (marking them reviewed).
pub fn bulk_label(project: &Project, ids: &[Id], label: Option<LabelId>) -> Vec<Change> {
    ids.iter().filter_map(|id| label_change(project, *id, label.clone())).collect()
}

/// Whether `p` lies inside the closed polygon `polygon` (even-odd rule).
pub fn inside(polygon: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut inside = false;
    for (i, a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % polygon.len()];
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) {
            inside = !inside;
        }
    }
    inside
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ColorBy {
    #[default]
    Label,
    Consensus,
    Cluster,
    Status,
}

impl ColorBy {
    const ALL: [Self; 4] = [Self::Label, Self::Consensus, Self::Cluster, Self::Status];

    fn name(self) -> &'static str {
        match self {
            Self::Label => "Analyst label",
            Self::Consensus => "Consensus",
            Self::Cluster => "Cluster",
            Self::Status => "Status",
        }
    }
}

#[derive(Resource)]
pub struct ScatterWindows {
    pub open: bool,
    /// Cluster package whose coordinates are plotted; features when `None`.
    source: Option<String>,
    x: usize,
    y: usize,
    color: ColorBy,
    /// Revision and source behind `points` or the running task.
    key: Option<(u64, Option<String>)>,
    task: Option<Task<Vec<ScatterPoint>>>,
    points: Vec<ScatterPoint>,
    selected: HashSet<Id>,
    /// In data coordinates.
    lasso: Vec<[f64; 2]>,
}

impl Default for ScatterWindows {
    fn default() -> Self {
        let index = |name: &str| EVENT_FEATURES.iter().position(|n| *n == name).unwrap_or(0);
        Self {
            open: false,
            source: None,
            x: index("p_s_log"),
            y: index("hf_lf_log"),
            color: ColorBy::default(),
            key: None,
            task: None,
            points: Vec::new(),
            selected: HashSet::new(),
            lasso: Vec::new(),
        }
    }
}

pub fn scatter_menu(ui: &mut egui::Ui, windows: &mut ScatterWindows) {
    if ui.button("Scatter…").on_hover_text("Events by two features or a cluster package's embedding; lasso to label in bulk").clicked() {
        windows.open = true;
        ui.close();
    }
}

fn palette(i: usize) -> egui::Color32 {
    PALETTE[i % PALETTE.len()]
}

/// Index of `key` among the distinct keys seen so far.
fn slot(keys: &mut Vec<String>, key: String) -> usize {
    keys.iter().position(|k| *k == key).unwrap_or_else(|| {
        keys.push(key);
        keys.len() - 1
    })
}

#[allow(clippy::too_many_arguments)]
pub fn windows(
    mut contexts: EguiContexts,
    plot: Res<Plot>,
    mut workspace: ResMut<Workspace>,
    mut view: ResMut<TrackView>,
    mut tracks: Query<&mut Track>,
    registry: Res<ModelRegistry>,
    mut windows: ResMut<ScatterWindows>,
) -> Result {
    if !windows.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    let embeddings = registry.embeddings(&workspace.project);
    if windows.source.as_ref().is_some_and(|s| !embeddings.iter().any(|p| &p.spec.id == s)) {
        windows.source = None;
    }
    let package: Option<Arc<Package>> = windows.source.as_ref().and_then(|s| embeddings.iter().find(|p| &p.spec.id == s).cloned());
    if let Some(task) = &mut windows.task
        && let Some(points) = check_ready(task)
    {
        windows.task = None;
        windows.points = points;
    }
    let key = (workspace.revision(), windows.source.clone());
    if windows.key.as_ref() != Some(&key) && windows.task.is_none() {
        let values = track_values(&tracks);
        let project = workspace.project.clone();
        let traces = trace_data(&project, |t| values.get(&t.id).cloned());
        let package = package.clone();
        windows.task = Some(AsyncComputeTaskPool::get().spawn(async move { scatter_points(&project, &traces, package.as_deref()) }));
        windows.key = Some(key);
    }

    let mut open = true;
    let mut change: Option<(String, Vec<Change>)> = None;
    let mut review: Option<Id> = None;
    egui::Window::new("Event Scatter").open(&mut open).default_width(620.0).show(&ctx, |ui| {
        let project = &workspace.project;
        let dims: Vec<String> = match &package {
            Some(p) => {
                let n = p.spec.cluster.map_or(2, |c| c.components);
                (1..=n).map(|k| format!("PC{k}")).collect()
            }
            None => EVENT_FEATURES.iter().map(|s| s.to_string()).collect(),
        };
        ui.horizontal(|ui| {
            let before = windows.source.clone();
            egui::ComboBox::from_id_salt("scatter_source")
                .selected_text(windows.source.as_deref().map_or("Event features", |s| s.trim_start_matches("heidtmare.squiggle.model/")))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut windows.source, None, "Event features");
                    for p in &embeddings {
                        ui.selectable_value(&mut windows.source, Some(p.spec.id.clone()), p.spec.id.trim_start_matches("heidtmare.squiggle.model/"));
                    }
                })
                .response
                .on_hover_text("Import a cluster package (squiggle-train cluster) in Events ▸ Models to plot its PCA embedding");
            if windows.source != before {
                (windows.x, windows.y) = if windows.source.is_some() {
                    (0, 1)
                } else {
                    let defaults = ScatterWindows::default();
                    (defaults.x, defaults.y)
                };
            }
            windows.x = windows.x.min(dims.len().saturating_sub(1));
            windows.y = windows.y.min(dims.len().saturating_sub(1));
            for (salt, axis) in [("scatter_x", &mut windows.x), ("scatter_y", &mut windows.y)] {
                egui::ComboBox::from_id_salt(salt).selected_text(dims.get(*axis).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                    for (i, d) in dims.iter().enumerate() {
                        ui.selectable_value(axis, i, d);
                    }
                });
            }
            egui::ComboBox::from_id_salt("scatter_color").selected_text(windows.color.name()).show_ui(ui, |ui| {
                for c in ColorBy::ALL {
                    ui.selectable_value(&mut windows.color, c, c.name());
                }
            });
            if windows.task.is_some() {
                ui.spinner();
            }
        });

        let coords = |p: &ScatterPoint| -> Option<[f64; 2]> {
            let values: &[f64] = match (&package, &p.embedding) {
                (Some(_), Some((e, _))) => e,
                (Some(_), None) => return None,
                (None, _) => &p.features,
            };
            let (x, y) = (*values.get(windows.x)?, *values.get(windows.y)?);
            (x.is_finite() && y.is_finite()).then_some([x, y])
        };
        let shown: Vec<(&ScatterPoint, [f64; 2])> = windows.points.iter().filter(|p| project.event(p.event).is_some()).filter_map(|p| Some((p, coords(p)?))).collect();
        let missing = windows.points.len() - shown.len();

        // Colours and their legend.
        let mut keys: Vec<String> = Vec::new();
        let colour: Vec<egui::Color32> = shown
            .iter()
            .map(|(p, _)| {
                let event = project.event(p.event).expect("filtered");
                let key = match windows.color {
                    ColorBy::Label => event.label.as_ref().map(|l| project.taxonomy().path(l)),
                    ColorBy::Consensus => project.consensus(event).first().map(|l| l.0.to_string()),
                    ColorBy::Cluster => p.embedding.as_ref().and_then(|e| e.1).map(|k| format!("cluster {k}")),
                    ColorBy::Status => Some(event.status.label().to_string()),
                };
                key.map_or(NONE_COLOR, |k| palette(slot(&mut keys, k)))
            })
            .collect();

        let size = egui::vec2(ui.available_width().max(300.0), 360.0);
        let (response, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
        let rect = response.rect.shrink(28.0);
        painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::from_gray(90)), egui::StrokeKind::Inside);
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (_, c) in &shown {
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        for k in 0..2 {
            if !lo[k].is_finite() {
                (lo[k], hi[k]) = (0.0, 1.0);
            }
            let pad = ((hi[k] - lo[k]) * 0.05).max(1e-9);
            (lo[k], hi[k]) = (lo[k] - pad, hi[k] + pad);
        }
        let to_screen = |c: [f64; 2]| {
            egui::pos2(
                rect.left() + ((c[0] - lo[0]) / (hi[0] - lo[0])) as f32 * rect.width(),
                rect.bottom() - ((c[1] - lo[1]) / (hi[1] - lo[1])) as f32 * rect.height(),
            )
        };
        let to_data = |p: egui::Pos2| {
            [lo[0] + f64::from((p.x - rect.left()) / rect.width()) * (hi[0] - lo[0]), lo[1] + f64::from((rect.bottom() - p.y) / rect.height()) * (hi[1] - lo[1])]
        };
        let text = egui::TextStyle::Small.resolve(ui.style());
        let weak = ui.visuals().weak_text_color();
        let label = |v: f64| if v.abs() >= 1e4 || (v != 0.0 && v.abs() < 1e-2) { format!("{v:.2e}") } else { format!("{v:.2}") };
        painter.text(rect.left_bottom() + egui::vec2(0.0, 4.0), egui::Align2::LEFT_TOP, label(lo[0]), text.clone(), weak);
        painter.text(rect.right_bottom() + egui::vec2(0.0, 4.0), egui::Align2::RIGHT_TOP, label(hi[0]), text.clone(), weak);
        painter.text(rect.center_bottom() + egui::vec2(0.0, 4.0), egui::Align2::CENTER_TOP, &dims[windows.x], text.clone(), weak);
        painter.text(rect.left_bottom() - egui::vec2(4.0, 0.0), egui::Align2::RIGHT_BOTTOM, label(lo[1]), text.clone(), weak);
        painter.text(rect.left_top() - egui::vec2(4.0, 0.0), egui::Align2::RIGHT_TOP, label(hi[1]), text.clone(), weak);
        painter.text(rect.left_top() + egui::vec2(4.0, 2.0), egui::Align2::LEFT_TOP, &dims[windows.y], text.clone(), weak);

        let nearest = response.hover_pos().and_then(|pos| {
            shown.iter().map(|(p, c)| (p.event, to_screen(*c).distance(pos))).filter(|(_, d)| *d < 7.0).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(id, _)| id)
        });
        for ((p, c), colour) in shown.iter().zip(&colour) {
            let at = to_screen(*c);
            painter.circle_filled(at, 4.0, *colour);
            if windows.selected.contains(&p.event) {
                painter.circle_stroke(at, 6.5, egui::Stroke::new(1.5, egui::Color32::WHITE));
            }
            if view.event == Some(p.event) {
                painter.circle_stroke(at, 9.0, egui::Stroke::new(1.0, egui::Color32::YELLOW));
            }
        }

        let shift = ui.input(|i| i.modifiers.shift);
        if response.drag_started() {
            windows.lasso.clear();
        }
        if response.dragged()
            && let Some(pos) = response.interact_pointer_pos()
        {
            windows.lasso.push(to_data(pos));
        }
        if windows.lasso.len() > 1 {
            let line: Vec<egui::Pos2> = windows.lasso.iter().map(|c| to_screen(*c)).collect();
            painter.add(egui::Shape::closed_line(line, egui::Stroke::new(1.0, egui::Color32::from_gray(220))));
        }
        if response.drag_stopped() {
            if !shift {
                windows.selected.clear();
            }
            if windows.lasso.len() > 2 {
                windows.selected.extend(shown.iter().filter(|(_, c)| inside(&windows.lasso, *c)).map(|(p, _)| p.event));
            }
            windows.lasso.clear();
        }
        if response.double_clicked()
            && let Some(id) = nearest
        {
            review = Some(id);
        } else if response.clicked() {
            match nearest {
                Some(id) if shift => {
                    if !windows.selected.remove(&id) {
                        windows.selected.insert(id);
                    }
                }
                Some(id) => windows.selected = HashSet::from([id]),
                None if !shift => windows.selected.clear(),
                None => {}
            }
        }
        if let Some(id) = nearest
            && let Some(event) = project.event(id)
        {
            let point = shown.iter().find(|(p, _)| p.event == id);
            let at = project.reference.map_or(event.time().to_string(), |r| format!("{:+.1} s", event.time().seconds_since(r)));
            let mut tip = format!("Event at {at}\n{}", event.status.label());
            if let Some(l) = &event.label {
                tip.push_str(&format!(", {}", project.taxonomy().path(l)));
            }
            if let Some((_, Some(k))) = point.and_then(|p| p.0.embedding.as_ref()) {
                tip.push_str(&format!("\ncluster {k}"));
            }
            response.clone().on_hover_text_at_pointer(tip);
        }

        ui.horizontal_wrapped(|ui| {
            for (i, k) in keys.iter().enumerate() {
                ui.colored_label(palette(i), format!("● {k}"));
            }
            if colour.contains(&NONE_COLOR) {
                ui.colored_label(NONE_COLOR, "● none");
            }
        });
        ui.weak(format!(
            "{} events plotted{}. Drag to lasso, click to pick (Shift adds), double-click to review.",
            shown.len(),
            if missing > 0 { format!(", {missing} without both values") } else { String::new() }
        ));
        if let Some(p) = &package {
            let clusters: Vec<String> = p
                .spec
                .output
                .clusters
                .iter()
                .map(|c| {
                    let classes: Vec<String> = c.classes.iter().take(2).map(|k| format!("{} {}", k.events, k.label)).collect();
                    format!("{}: {} trained{}", c.cluster, c.events, if classes.is_empty() { String::new() } else { format!(" ({})", classes.join(", ")) })
                })
                .collect();
            ui.weak(format!("Clusters from training: {}", clusters.join("; ")));
        }

        ui.separator();
        let ids: Vec<Id> = {
            let mut ids: Vec<Id> = windows.selected.iter().copied().filter(|id| project.event(*id).is_some()).collect();
            ids.sort();
            ids
        };
        ui.horizontal(|ui| {
            ui.label(format!("{} selected", ids.len()));
            if package.is_some() {
                let clusters: Vec<usize> = {
                    let mut c: Vec<usize> = windows.points.iter().filter_map(|p| p.embedding.as_ref()?.1).collect();
                    c.sort_unstable();
                    c.dedup();
                    c
                };
                egui::ComboBox::from_id_salt("scatter_cluster").selected_text("Select cluster").show_ui(ui, |ui| {
                    for k in clusters {
                        if ui.selectable_label(false, format!("cluster {k}")).clicked() {
                            let members = windows.points.iter().filter(|p| p.embedding.as_ref().and_then(|e| e.1) == Some(k)).map(|p| p.event);
                            if !shift {
                                windows.selected.clear();
                            }
                            windows.selected.extend(members);
                        }
                    }
                });
            }
            ui.add_enabled_ui(!ids.is_empty(), |ui| {
                ui.label("Label");
                if let Some(label) = label_picker(ui, "scatter_label", project, None) {
                    let changes = bulk_label(project, &ids, label);
                    change = Some((format!("Label {} events", changes.len()), changes));
                }
                if ui.button("Accept").on_hover_text("Reviewed; unlabelled events take the consensus label").clicked() {
                    let changes: Vec<Change> = ids.iter().filter_map(|id| accept_change(project, *id)).collect();
                    change = Some((format!("Accept {} events", changes.len()), changes));
                }
                if ui.button("Reject").clicked() {
                    let changes: Vec<Change> = ids.iter().filter_map(|id| status_change(project, *id, EventStatus::Rejected)).collect();
                    change = Some((format!("Reject {} events", changes.len()), changes));
                }
                if ui.button("Clear").clicked() {
                    windows.selected.clear();
                }
            });
        });
    });
    if let Some(id) = review {
        open_event(&mut view, &mut tracks, &workspace.project, id, plot.view.plot_width());
    }
    if let Some((label, changes)) = change.filter(|c| !c.1.is_empty()) {
        workspace.execute_or_report(label, changes);
    }
    windows.open = open;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lasso_polygons_contain_points_by_the_even_odd_rule() {
        let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        assert!(inside(&square, [1.0, 1.0]));
        assert!(!inside(&square, [3.0, 1.0]));
        let bow = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [3.0, 4.0], [3.0, 1.0], [1.0, 1.0], [1.0, 4.0], [0.0, 4.0]];
        assert!(!inside(&bow, [2.0, 3.0]), "the notch");
        assert!(inside(&bow, [0.5, 3.0]));
    }
}
