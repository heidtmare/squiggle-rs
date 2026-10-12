//! Model manager: imports packages written by `squiggle-train`, enables and
//! disables them, and shows their training scores and how often analysts
//! agree with them. Packages stay on disk; the project records their path and
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

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
const FAIL_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 80, 80);

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
}

impl ModelRegistry {
    /// Loads newly referenced packages and forgets removed ones.
    pub fn refresh(&mut self, project: &Project) {
        self.loaded.retain(|id, _| project.model(id).is_some());
        for model in project.models() {
            if self.loaded.get(&model.id).is_some_and(|l| l.path == model.path && l.hash == model.hash) {
                continue;
            }
            let package = Package::load(Path::new(&model.path)).and_then(|p| {
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

    /// Enabled packages that load and fit the project's taxonomy.
    pub fn active(&self, project: &Project) -> Vec<Arc<Package>> {
        project
            .models()
            .iter()
            .filter(|m| m.enabled)
            .filter_map(|m| self.package(&m.id).ok().filter(|p| fits(project, p).is_ok()).cloned())
            .collect()
    }
}

/// Refuses a package with labels the project lacks; warns on a taxonomy version change.
pub fn fits(project: &Project, package: &Package) -> Result<Option<String>, String> {
    let taxonomy = project.taxonomy();
    let missing: Vec<&str> =
        package.spec.output.labels.iter().map(String::as_str).filter(|l| *l != NOISE && taxonomy.get(&LabelId::new(*l)).is_none()).collect();
    if !missing.is_empty() {
        return Err(format!("labels missing from the project: {}", missing.join(", ")));
    }
    Ok((package.spec.output.taxonomy_version != taxonomy.version)
        .then(|| format!("trained on taxonomy version {}; the project is at {}", package.spec.output.taxonomy_version, taxonomy.version)))
}

/// Registers the package in `dir`, enabled.
pub fn import_change(project: &Project, dir: &Path) -> Result<Change, String> {
    let package = Package::load(dir)?;
    fits(project, &package)?;
    let path = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    Ok(Change::SetModel(ModelRef { id: package.spec.id.clone(), path: path.to_string_lossy().into_owned(), hash: package.hash.clone(), enabled: true }))
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

#[derive(Resource, Default)]
pub struct ModelWindows {
    pub open: bool,
    folder: Option<Task<Option<PathBuf>>>,
    error: Option<String>,
}

pub fn models_menu(ui: &mut egui::Ui, windows: &mut ModelWindows) {
    if ui.button("Models…").on_hover_text("Import, enable and compare trained model packages").clicked() {
        windows.open = true;
        ui.close();
    }
}

pub fn windows(mut contexts: EguiContexts, mut workspace: ResMut<Workspace>, mut registry: ResMut<ModelRegistry>, mut windows: ResMut<ModelWindows>) -> Result {
    registry.refresh(&workspace.project);
    let ctx = contexts.ctx_mut()?.clone();
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
    egui::Window::new("Models").open(&mut open).default_width(640.0).show(&ctx, |ui| {
        let project = &workspace.project;
        ui.weak("Packages written by squiggle-train. Events ▸ Classify runs the rules and every enabled model; predictions record each model's id.");
        if project.models().is_empty() {
            ui.label("No models imported.");
        }
        egui::Grid::new("models").num_columns(7).striped(true).show(ui, |ui| {
            if !project.models().is_empty() {
                for heading in ["On", "Model", "Labels", "Trained on", "Cross-validated", "Analysts agree", ""] {
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
                            ui.colored_label(FAIL_COLOR, "✖").on_hover_text(error);
                        }
                        Ok(Some(warning)) => {
                            ui.colored_label(WARN_COLOR, "⚠").on_hover_text(warning);
                        }
                        Ok(None) => {}
                    }
                });
                match package {
                    Ok(p) => {
                        let t = &p.spec.training;
                        ui.label(p.spec.output.labels.join(", ")).on_hover_text(format!("{} level", p.spec.output.level));
                        ui.label(format!("{} events", t.events)).on_hover_text(format!("{}\ndataset {}", t.project, &t.dataset[..t.dataset.len().min(12)]));
                        if t.folds > 0 {
                            ui.label(format!("{:.0}% ({}-fold)", t.cross_validation.accuracy * 100.0, t.folds))
                                .on_hover_text(confusion(&p.spec.output.labels, &t.cross_validation.confusion));
                        } else {
                            ui.weak("—");
                        }
                    }
                    Err(_) => {
                        for _ in 0..3 {
                            ui.weak("—");
                        }
                    }
                }
                match agreement(project, &model.id) {
                    (_, 0) => ui.weak("—"),
                    (agree, total) => ui.label(format!("{:.0}% of {total}", agree as f64 * 100.0 / total as f64)),
                };
                if ui.small_button("🗑").on_hover_text("Remove from the project (the package stays on disk)").clicked() {
                    change = Some(("Remove model", Change::RemoveModel { model: model.id.clone() }));
                }
                ui.end_row();
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            let busy = windows.folder.is_some();
            if ui.add_enabled(!busy, egui::Button::new("Import package…")).clicked() {
                windows.error = None;
                windows.folder = Some(IoTaskPool::get().spawn(async {
                    let folder = rfd::AsyncFileDialog::new().set_title("Import model package (folder with spec.toml)").pick_folder().await?;
                    Some(folder.path().to_path_buf())
                }));
            }
            if busy {
                ui.spinner();
            }
        });
        if let Some(error) = &windows.error {
            ui.colored_label(FAIL_COLOR, error);
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
