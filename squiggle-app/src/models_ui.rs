//! Model manager: imports packages written by `squiggle-train`, enables and
//! disables them, and shows their training scores and how often analysts
//! agree with each model version (the rules and templates included). Packages stay on disk; the project records their path and
//! hash, and a package that changed since import is refused.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::model::ModelRef;
use squiggle_core::taxonomy::NOISE;
use squiggle_core::{Change, EventStatus, LabelId, Project};
use squiggle_models::Package;

use crate::workspace::Workspace;
use crate::theme;


/// A referenced package as read from disk.
struct Loaded {
    path: String,
    hash: String,
    package: Result<Arc<Package>, String>,
}

/// Packages the project references, loaded once per path and hash.
#[derive(Resource, Default)]
pub struct ModelRegistry {
    loaded: HashMap<String, Loaded>,
    /// Shared-file cache: a package missing from its path is looked up here by hash.
    pub cache: Option<std::path::PathBuf>,
}

impl ModelRegistry {
    /// Loads newly referenced packages and forgets removed ones.
    pub fn refresh(&mut self, project: &Project) {
        self.loaded.retain(|id, _| project.model(id).is_some());
        for model in project.models() {
            if self.loaded.get(&model.id).is_some_and(|l| l.path == model.path && l.hash == model.hash) {
                continue;
            }
            let cached = self.cache.as_ref().map(|c| squiggle_io::sync::blob_dir(c, &model.hash)).filter(|d| d.is_dir());
            let dir = match cached {
                Some(dir) if !Path::new(&model.path).is_dir() => dir,
                _ => Path::new(&model.path).to_path_buf(),
            };
            let package = Package::load(&dir).and_then(|p| {
                if p.hash != model.hash {
                    Err("the package changed since it was imported; import it again".to_string())
                } else if p.spec.id != model.id {
                    Err(format!("the package is now {}", p.spec.id))
                } else {
                    Ok(Arc::new(p))
                }
            });
            self.loaded.insert(model.id.clone(), Loaded { path: model.path.clone(), hash: model.hash.clone(), package });
        }
    }

    pub fn package(&self, id: &str) -> Result<&Arc<Package>, String> {
        match self.loaded.get(id) {
            Some(loaded) => loaded.package.as_ref().map_err(String::clone),
            None => Err("not loaded".to_string()),
        }
    }

    /// Enabled classifiers that load and fit the project's taxonomy.
    pub fn active(&self, project: &Project) -> Vec<Arc<Package>> {
        project
            .models()
            .iter()
            .filter(|m| m.enabled)
            .filter_map(|m| self.package(&m.id).ok().filter(|p| p.is_classifier() && fits(project, p).is_ok()).cloned())
            .collect()
    }

    /// Embedding (cluster) packages that load.
    pub fn embeddings(&self, project: &Project) -> Vec<Arc<Package>> {
        project.models().iter().filter_map(|m| self.package(&m.id).ok().filter(|p| !p.is_classifier()).cloned()).collect()
    }
}

/// Refuses a package with labels the project lacks; warns on a taxonomy
/// version change and on phenomenologies the project has no traces of.
pub fn fits(project: &Project, package: &Package) -> Result<Option<String>, String> {
    let taxonomy = project.taxonomy();
    let missing: Vec<&str> =
        package.spec.output.labels.iter().map(String::as_str).filter(|l| *l != NOISE && taxonomy.get(&LabelId::new(*l)).is_none()).collect();
    if !missing.is_empty() {
        return Err(format!("labels missing from the project: {}", missing.join(", ")));
    }
    let mut warnings = Vec::new();
    if package.spec.output.taxonomy_version != taxonomy.version {
        warnings.push(format!("trained on taxonomy version {}; the project is at {}", package.spec.output.taxonomy_version, taxonomy.version));
    }
    for branch in &package.spec.input.branches {
        let id = &branch.window.phenomenology;
        if !project.traces().iter().any(|t| project.trace_phenomenology(t).is_some_and(|p| &p.id == id)) {
            warnings.push(format!("no {id} traces{}", if branch.required { ": the model is required to abstain" } else { ": fused from the rest" }));
        }
    }
    Ok((!warnings.is_empty()).then(|| warnings.join("\n")))
}

/// Kind and inputs, e.g. "cnn (late): seismic*, infrasound" (* required).
pub fn describe(package: &Package) -> String {
    let spec = &package.spec;
    if let Some(c) = spec.cluster {
        return format!("cluster: PCA ({} components) and DBSCAN on {} features", c.components, spec.input.features.len());
    }
    if spec.input.branches.is_empty() {
        return format!("{} on {} features", spec.kind, spec.input.features.len());
    }
    let branches: Vec<String> = spec
        .input
        .branches
        .iter()
        .map(|b| format!("{}{} ({:?})", b.window.phenomenology, if b.required { "*" } else { "" }, b.encoder).to_lowercase())
        .collect();
    let fusion = spec.input.fusion.map_or(String::new(), |f| format!(" {f:?}").to_lowercase());
    format!("{}{fusion}: {}", spec.kind, branches.join(", "))
}

/// Registers the package in `dir`, enabled.
pub fn import_change(project: &Project, dir: &Path) -> Result<Change, String> {
    let package = Package::load(dir)?;
    fits(project, &package)?;
    let path = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    Ok(Change::SetModel(ModelRef { id: package.spec.id.clone(), path: path.to_string_lossy().into_owned(), hash: package.hash.clone(), enabled: true, weight: 1.0 }))
}

/// Reviewed or rejected events `model` predicted, and how many it got right
/// (its top label is the analyst's label or an ancestor of it; noise for rejected).
pub fn agreement(project: &Project, model: &str) -> (usize, usize) {
    let mut agree = 0;
    let mut total = 0;
    for event in project.events() {
        let Some((top, _)) = event.prediction(model).and_then(|p| p.top()) else { continue };
        let right = match (event.status, &event.label) {
            (EventStatus::Rejected, _) => top.as_str() == NOISE,
            (EventStatus::Reviewed, Some(label)) => project.taxonomy().is_a(label, top),
            _ => continue,
        };
        total += 1;
        agree += usize::from(right);
    }
    (agree, total)
}

/// How often analysts agree with one model version.
#[derive(Clone, Debug, PartialEq)]
pub struct Agreement {
    pub model: String,
    pub agree: usize,
    pub compared: usize,
    /// Over the `RECENT` latest compared events (by event time).
    pub recent: (usize, usize),
    /// (analyst class, predicted, events), most first.
    pub confusion: Vec<(String, String, usize)>,
}

const RECENT: usize = 20;

/// Agreement of every model with a recorded prediction on a reviewed or
/// rejected event, by name then version.
pub fn agreements(project: &Project) -> Vec<Agreement> {
    let mut models: Vec<&str> = project.events().iter().flat_map(|e| e.predictions.iter().map(|p| p.model.as_str())).collect();
    models.sort_by_key(|m| version_key(m));
    models.dedup();
    let mut events: Vec<&squiggle_core::Event> = project.events().iter().collect();
    events.sort_by(|a, b| a.time().seconds_since(b.time()).total_cmp(&0.0));
    models
        .into_iter()
        .filter_map(|model| {
            let mut outcomes: Vec<bool> = Vec::new();
            let mut confusion: Vec<(String, String, usize)> = Vec::new();
            for event in &events {
                let Some((top, _)) = event.prediction(model).and_then(|p| p.top()) else { continue };
                let (truth, right) = match (event.status, &event.label) {
                    (EventStatus::Rejected, _) => (NOISE.to_string(), top.as_str() == NOISE),
                    (EventStatus::Reviewed, Some(label)) => (label.to_string(), project.taxonomy().is_a(label, top)),
                    _ => continue,
                };
                outcomes.push(right);
                match confusion.iter_mut().find(|c| c.0 == truth && c.1 == top.as_str()) {
                    Some(c) => c.2 += 1,
                    None => confusion.push((truth, top.to_string(), 1)),
                }
            }
            if outcomes.is_empty() {
                return None;
            }
            confusion.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
            let recent = &outcomes[outcomes.len().saturating_sub(RECENT)..];
            Some(Agreement {
                model: model.to_string(),
                agree: outcomes.iter().filter(|o| **o).count(),
                compared: outcomes.len(),
                recent: (recent.iter().filter(|o| **o).count(), recent.len()),
                confusion,
            })
        })
        .collect()
}

/// Name and numeric version of `heidtmare.squiggle.model/<name>@<version>`.
fn version_key(id: &str) -> (String, u64) {
    let short = id.trim_start_matches("heidtmare.squiggle.model/");
    match short.rsplit_once('@') {
        Some((name, version)) => (name.to_string(), version.parse().unwrap_or(0)),
        None => (short.to_string(), 0),
    }
}

fn percent(agree: usize, of: usize) -> String {
    if of == 0 { "—".to_string() } else { format!("{:.0}% of {of}", agree as f64 * 100.0 / of as f64) }
}

#[derive(Resource, Default)]
pub struct ModelWindows {
    pub open: bool,
    folder: Option<Task<Option<PathBuf>>>,
    error: Option<String>,
    weights: HashMap<String, f32>,
}

pub fn models_menu(ui: &mut egui::Ui, windows: &mut ModelWindows) {
    if ui.button("Models…").on_hover_text("Import, enable and compare trained model packages").clicked() {
        windows.open = true;
        ui.close();
    }
}

pub fn windows(mut contexts: EguiContexts, mut workspace: ResMut<Workspace>, mut registry: ResMut<ModelRegistry>, mut windows: ResMut<ModelWindows>) -> Result {
    let cache = workspace.cache_dir();
    if registry.cache != cache {
        registry.cache = cache;
        registry.loaded.clear();
    }
    registry.refresh(&workspace.project);
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    if let Some(task) = &mut windows.folder
        && let Some(folder) = check_ready(task)
    {
        windows.folder = None;
        if let Some(folder) = folder {
            match import_change(&workspace.project, &folder) {
                Ok(change) => {
                    windows.error = None;
                    workspace.execute_or_report("Import model", vec![change]);
                    registry.refresh(&workspace.project);
                }
                Err(error) => windows.error = Some(format!("{}: {error}", folder.display())),
            }
        }
    }
    if !windows.open {
        return Ok(());
    }
    let mut open = true;
    let mut change: Option<(&str, Change)> = None;
    let weights = &mut windows.weights;
    theme::tool_window(&ctx, "Models").open(&mut open).default_width(640.0).show(&ctx, |ui| {
        let project = &workspace.project;
        ui.weak("Packages written by squiggle-train. Events ▸ Classify runs the rules and every enabled model; predictions record each model's id. The catalogue, queue and Accept use the weighted ensemble.");
        if project.models().is_empty() {
            ui.label("No models imported.");
        }
        egui::Grid::new("models").num_columns(9).striped(true).show(ui, |ui| {
            if !project.models().is_empty() {
                for heading in ["On", "Model", "Inputs", "Labels", "Trained on", "Cross-validated", "Analysts agree", "Weight", ""] {
                    ui.strong(heading);
                }
                ui.end_row();
            }
            for model in project.models() {
                let package = registry.package(&model.id);
                let fit = package.as_ref().map_err(String::clone).and_then(|p| fits(project, p));
                let mut enabled = model.enabled;
                if ui.add_enabled(fit.is_ok(), egui::Checkbox::without_text(&mut enabled)).changed() {
                    change = Some(("Toggle model", Change::SetModel(ModelRef { enabled, ..model.clone() })));
                }
                ui.horizontal(|ui| {
                    ui.monospace(model.id.trim_start_matches("heidtmare.squiggle.model/")).on_hover_text(format!("{}\n{}", model.id, model.path));
                    match &fit {
                        Err(error) => {
                            ui.colored_label(theme::BAD, "✖").on_hover_text(error);
                        }
                        Ok(Some(warning)) => {
                            ui.colored_label(theme::WARN, "⚠").on_hover_text(warning);
                        }
                        Ok(None) => {}
                    }
                });
                match package {
                    Ok(p) => {
                        let t = &p.spec.training;
                        ui.label(describe(p)).on_hover_text("* required phenomenology");
                        if p.is_classifier() {
                            ui.label(p.spec.output.labels.join(", ")).on_hover_text(format!("{} level", p.spec.output.level));
                        } else {
                            ui.label(format!("{} clusters", p.spec.output.clusters.len())).on_hover_text("Plotted in Events ▸ Scatter; not used by Classify");
                        }
                        ui.label(format!("{} events", t.events)).on_hover_text(format!("{}\ndataset {}", t.project, &t.dataset[..t.dataset.len().min(12)]));
                        if t.folds > 0 {
                            ui.label(format!("{:.0}% ({}-fold)", t.cross_validation.accuracy * 100.0, t.folds))
                                .on_hover_text(confusion(&p.spec.output.labels, &t.cross_validation.confusion));
                        } else {
                            ui.weak("—");
                        }
                    }
                    Err(_) => {
                        for _ in 0..4 {
                            ui.weak("—");
                        }
                    }
                }
                let (agree, total) = agreement(project, &model.id);
                ui.label(percent(agree, total));
                // Edited locally while dragging; one undoable change on release.
                let weight = weights.entry(model.id.clone()).or_insert(model.weight);
                let response = ui.add(egui::DragValue::new(weight).range(0.0..=10.0).speed(0.05)).on_hover_text("Vote in the ensemble consensus (the rules count 1)");
                let done = response.drag_stopped() || response.lost_focus() || (response.changed() && !response.dragged());
                if done && *weight != model.weight {
                    change = Some(("Model weight", Change::SetModel(ModelRef { weight: *weight, ..model.clone() })));
                }
                if !response.dragged() && !response.has_focus() && !done {
                    *weight = model.weight;
                }
                if ui.small_button("🗑").on_hover_text("Remove from the project (the package stays on disk)").clicked() {
                    change = Some(("Remove model", Change::RemoveModel { model: model.id.clone() }));
                }
                ui.end_row();
            }
        });
        let rows = agreements(project);
        egui::CollapsingHeader::new("Analyst agreement by model version").default_open(!rows.is_empty()).show(ui, |ui| {
            ui.weak(format!("Each recorded prediction's top label against reviewed labels (a model trained on a parent label agrees with its subtypes) and rejections (noise). Recent: the latest {RECENT} by event time."));
            egui::Grid::new("agreement").num_columns(5).striped(true).show(ui, |ui| {
                for heading in ["Model", "Version", "Agree", "Recent", "Change"] {
                    ui.strong(heading);
                }
                ui.end_row();
                let mut previous: Option<(String, f64)> = None;
                for row in &rows {
                    let (name, version) = version_key(&row.model);
                    let rate = row.agree as f64 / row.compared as f64;
                    ui.monospace(&name);
                    ui.label(version.to_string());
                    let mut matrix = String::from("Analyst → model: events\n");
                    for (truth, predicted, n) in &row.confusion {
                        matrix.push_str(&format!("{truth} → {predicted}: {n}\n"));
                    }
                    ui.label(percent(row.agree, row.compared)).on_hover_text(matrix);
                    ui.label(percent(row.recent.0, row.recent.1));
                    match &previous {
                        Some((p, r)) if *p == name => {
                            let delta = (rate - r) * 100.0;
                            ui.label(format!("{delta:+.0} pts")).on_hover_text("Against the previous version");
                        }
                        _ => {
                            ui.weak("—");
                        }
                    }
                    previous = Some((name, rate));
                    ui.end_row();
                }
            });
        });
        ui.separator();
        ui.horizontal(|ui| {
            let busy = windows.folder.is_some();
            if ui.add_enabled(!busy, egui::Button::new("Import package…")).clicked() {
                windows.error = None;
                windows.folder = Some(IoTaskPool::get().spawn(async {
                    if let Some(path) = crate::tour::picked() {
                        return Some(path);
                    }
                    let folder = rfd::AsyncFileDialog::new().set_title("Import model package (folder with spec.toml)").pick_folder().await?;
                    Some(folder.path().to_path_buf())
                }));
            }
            if busy {
                ui.spinner();
            }
        });
        if let Some(error) = &windows.error {
            ui.colored_label(theme::BAD, error);
        }
    });
    if let Some((label, c)) = change {
        workspace.execute_or_report(label, vec![c]);
    }
    windows.open = open;
    Ok(())
}

/// Confusion matrix as text: rows true, columns predicted.
fn confusion(labels: &[String], matrix: &[Vec<usize>]) -> String {
    let mut text = String::from("Out-of-fold confusion (rows true, columns predicted):\n");
    for (label, row) in labels.iter().zip(matrix) {
        let cells: Vec<String> = row.iter().map(|n| format!("{n:>4}")).collect();
        text.push_str(&format!("{label:>12} {}\n", cells.join("")));
    }
    text
}
