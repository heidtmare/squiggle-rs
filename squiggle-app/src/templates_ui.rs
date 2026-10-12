//! Template library: analysts save events as templates (each member's
//! window from its onset, per sensor), name and label them, and remove them.
//! Classify matches every event against the library and records the vote of
//! the best-correlating labels; Detect runs a matched filter on sensors that
//! have templates.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use squiggle_core::features::{EventData, TraceData};
use squiggle_core::template::{Template, template_traces};
use squiggle_core::{Change, Id, Project};

use crate::classify_ui::{default_window_specs, label_picker};
use crate::events_ui::open_event;
use crate::plot::Plot;
use crate::tracks::{Track, TrackView};
use crate::workspace::Workspace;

/// A template named `name` from `event`'s members over `traces`, labelled as the event; `None` without member windows.
pub fn template_change(project: &Project, traces: &[TraceData], data: &EventData, event: Id, name: &str) -> Option<Change> {
    let specs = default_window_specs(project);
    let traces = template_traces(traces, data, &specs);
    if traces.is_empty() || name.trim().is_empty() {
        return None;
    }
    let label = project.event(event)?.label.clone();
    Some(Change::SetTemplate(Template { id: Id::generate(), name: name.trim().to_string(), label, event: Some(event), traces }))
}

#[derive(Resource, Default)]
pub struct TemplateWindows {
    pub open: bool,
    /// Names being edited.
    names: HashMap<Id, String>,
}

pub fn templates_menu(ui: &mut egui::Ui, windows: &mut TemplateWindows) {
    if ui.button("Templates…").on_hover_text("The template library: rename, relabel, remove; save new ones from Features").clicked() {
        windows.open = true;
        ui.close();
    }
}

pub fn windows(
    mut contexts: EguiContexts,
    plot: Res<Plot>,
    mut workspace: ResMut<Workspace>,
    mut view: ResMut<TrackView>,
    mut tracks: Query<&mut Track>,
    mut windows: ResMut<TemplateWindows>,
) -> Result {
    if !windows.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?.clone();
    let windows = &mut *windows;
    let mut open = true;
    let mut change: Option<(&str, Change)> = None;
    let mut review: Option<Id> = None;
    egui::Window::new("Templates").open(&mut open).default_width(560.0).show(&ctx, |ui| {
        let project = &workspace.project;
        ui.weak("Save an event as a template from Events ▸ Features. Classify votes the label of the best network correlation; Detect's matched filter finds repeats on sensors with templates.");
        if project.templates().is_empty() {
            ui.label("No templates.");
            return;
        }
        egui::Grid::new("templates").num_columns(5).striped(true).show(ui, |ui| {
            for heading in ["Name", "Label", "Sensors", "From", ""] {
                ui.strong(heading);
            }
            ui.end_row();
            for template in project.templates() {
                let name = windows.names.entry(template.id).or_insert_with(|| template.name.clone());
                let response = ui.add(egui::TextEdit::singleline(name).desired_width(140.0));
                if response.lost_focus() && !name.trim().is_empty() && name.trim() != template.name {
                    change = Some(("Rename template", Change::SetTemplate(Template { name: name.trim().to_string(), ..template.clone() })));
                }
                if !response.has_focus() && !response.lost_focus() {
                    name.clone_from(&template.name);
                }
                if let Some(label) = label_picker(ui, ("template_label", template.id), project, template.label.as_ref()) {
                    change = Some(("Relabel template", Change::SetTemplate(Template { label, ..template.clone() })));
                }
                let mut kinds: Vec<&str> = template.traces.iter().map(|t| t.phenomenology.as_str()).collect();
                kinds.dedup();
                let codes: Vec<&str> = template.traces.iter().filter_map(|t| project.sensor(t.sensor)).map(|s| s.code.as_str()).collect();
                ui.label(format!("{} ({})", template.traces.len(), kinds.join(", "))).on_hover_text(codes.join("\n"));
                match template.event.and_then(|e| project.event(e)) {
                    Some(event) => {
                        let at = project.reference.map_or(event.time().to_string(), |r| format!("{:+.1} s", event.time().seconds_since(r)));
                        if ui.small_button(at).on_hover_text("Review the event it was cut from").clicked() {
                            review = Some(event.id);
                        }
                    }
                    None => {
                        ui.weak("—");
                    }
                }
                if ui.small_button("🗑").on_hover_text("Remove from the library").clicked() {
                    change = Some(("Remove template", Change::RemoveTemplate { template: template.id }));
                }
                ui.end_row();
            }
        });
    });
    if let Some(id) = review {
        open_event(&mut view, &mut tracks, &workspace.project, id, plot.view.plot_width());
    }
    if let Some((label, c)) = change {
        workspace.execute_or_report(label, vec![c]);
    }
    windows.open = open;
    Ok(())
}
