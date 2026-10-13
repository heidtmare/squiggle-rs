//! Project windows: import, sensors, phenomenology registry.

use std::collections::HashMap;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use squiggle_core::phenomenology::Phase;
use squiggle_core::sensor::Provenance;
use squiggle_core::{
    Change, Id, Location, NumberFormat, Phenomenology, PhenomenologyId, Project, Propagation, Rgba, SampleKind,
};

use crate::workspace::{ImportQueue, SampleState, UNASSIGNED_COLOR, Workspace, WorkspaceRequest};
use crate::theme;


#[derive(Resource, Default)]
pub struct ProjectWindows {
    pub sensors_open: bool,
    pub phenomenologies_open: bool,
    editor: Option<PhenomenologyDraft>,
    bulk_start: String,
    bulk_phenomenology: Option<PhenomenologyId>,
    /// Latitude, longitude, elevation text being edited, per sensor, and the
    /// location it was loaded from; reloaded when that changes (e.g. undo).
    location_drafts: HashMap<Id, ([String; 3], Option<Location>)>,
}

/// Parses latitude and longitude in degrees and elevation in metres (blank: 0).
fn parse_location([lat, lon, elevation]: &[String; 3]) -> Result<Location, String> {
    let number = |text: &str, what: &str| text.trim().parse::<f64>().map_err(|_| format!("{what} must be a number"));
    let elevation_m = if elevation.trim().is_empty() { 0.0 } else { number(elevation, "elevation")? };
    let location = Location { latitude: number(lat, "latitude")?, longitude: number(lon, "longitude")?, elevation_m };
    if location.is_valid() { Ok(location) } else { Err("latitude ±90°, longitude ±180°".to_string()) }
}

/// What the menu bar needs from the project.
#[derive(SystemParam)]
pub struct ProjectAccess<'w> {
    pub workspace: ResMut<'w, Workspace>,
    pub windows: ResMut<'w, ProjectWindows>,
    pub requests: MessageWriter<'w, WorkspaceRequest>,
}

pub fn file_menu(ui: &mut egui::Ui, access: &mut ProjectAccess) {
    let mut request = |ui: &mut egui::Ui, label: &str, request: WorkspaceRequest| {
        if ui.button(label).clicked() {
            access.requests.write(request);
            ui.close();
        }
    };
    request(ui, "New Project", WorkspaceRequest::New);
    request(ui, "Open Project…", WorkspaceRequest::OpenDialog);
    request(ui, "Save Project As…", WorkspaceRequest::SaveAsDialog);
    ui.separator();
    request(ui, "Import Waveforms…", WorkspaceRequest::ImportDialog);
}

pub fn project_menu(ui: &mut egui::Ui, access: &mut ProjectAccess) {
    let unassigned = access.workspace.project.sensors().iter().filter(|s| s.assignment.is_none()).count();
    let label = if unassigned > 0 { format!("Sensors… ({unassigned} unassigned)") } else { "Sensors…".to_string() };
    if ui.button(label).clicked() {
        access.windows.sensors_open = true;
        ui.close();
    }
    if ui.button("Phenomenologies…").clicked() {
        access.windows.phenomenologies_open = true;
        ui.close();
    }
}

pub fn windows(
    mut contexts: EguiContexts,
    mut workspace: ResMut<Workspace>,
    mut queue: ResMut<ImportQueue>,
    mut windows: ResMut<ProjectWindows>,
    mut requests: MessageWriter<WorkspaceRequest>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    import_window(&ctx, &mut workspace, &mut queue, &mut windows, &mut requests);
    sensors_window(&ctx, &mut workspace, &mut windows);
    phenomenologies_window(&ctx, &workspace, &mut windows);
    editor_window(&ctx, &mut workspace, &mut windows);
    discard_window(&ctx, &mut workspace, &mut requests);
    Ok(())
}

fn phenomenology_label(project: &Project, id: Option<&PhenomenologyId>) -> egui::RichText {
    match id.and_then(|id| project.phenomenology(id)) {
        Some(p) => egui::RichText::new(&p.name).color(theme::color32(p.color)),
        None => egui::RichText::new("⚠ unassigned").color(theme::WARN),
    }
}

/// Drop-down over the project's phenomenologies; `allow_none` adds "unassigned".
fn phenomenology_choice(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    project: &Project,
    value: &mut Option<PhenomenologyId>,
    allow_none: bool,
) -> bool {
    let before = value.clone();
    egui::ComboBox::from_id_salt(id).selected_text(phenomenology_label(project, value.as_ref())).show_ui(ui, |ui| {
        if allow_none {
            ui.selectable_value(value, None, "unassigned");
        }
        for p in project.phenomenologies() {
            ui.selectable_value(value, Some(p.id.clone()), egui::RichText::new(&p.name).color(theme::color32(p.color)));
        }
    });
    *value != before
}

/// Layout editor for files no reader recognised.
fn raw_section(ui: &mut egui::Ui, queue: &mut ImportQueue, requests: &mut MessageWriter<WorkspaceRequest>) {
    if queue.raw_pending.is_empty() {
        return;
    }
    ui.strong("Unrecognised files");
    ui.label("No reader recognised these. Describe their layout to read them as raw binary samples; \
              the layout is saved with each trace.");
    for path in &queue.raw_pending {
        let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        ui.label(egui::RichText::new(name).monospace()).on_hover_text(path.display().to_string());
    }
    let layout = &mut queue.raw_layout;
    egui::Grid::new("raw_layout").num_columns(2).show(ui, |ui| {
        ui.label("Sample format");
        egui::ComboBox::from_id_salt("raw_format").selected_text(layout.format.label()).show_ui(ui, |ui| {
            for format in NumberFormat::ALL {
                ui.selectable_value(&mut layout.format, format, format.label());
            }
        });
        ui.end_row();
        ui.label("Byte order");
        ui.horizontal(|ui| {
            ui.radio_value(&mut layout.big_endian, false, "Little-endian");
            ui.radio_value(&mut layout.big_endian, true, "Big-endian");
        });
        ui.end_row();
        ui.label("Samples");
        ui.checkbox(&mut layout.complex, "Complex (interleaved I, Q)");
        ui.end_row();
        ui.label("Channels");
        ui.add(egui::DragValue::new(&mut layout.channels).range(1..=1024)).on_hover_text("Interleaved per frame");
        ui.end_row();
        ui.label("Header");
        ui.add(egui::DragValue::new(&mut layout.header_bytes).suffix(" bytes"));
        ui.end_row();
        ui.label("Scale");
        ui.add(egui::DragValue::new(&mut layout.scale).speed(0.001)).on_hover_text("Physical units per stored value");
        ui.end_row();
    });
    ui.label(egui::RichText::new(format!("{} bytes per frame", layout.frame_size())).weak());
    ui.horizontal(|ui| {
        if ui.button("Read as raw").clicked() {
            requests.write(WorkspaceRequest::ImportRaw(queue.raw_pending.clone(), queue.raw_layout));
        }
        if ui.button("Skip").clicked() {
            queue.raw_pending.clear();
        }
    });
    ui.separator();
}

fn import_window(
    ctx: &egui::Context,
    workspace: &mut Workspace,
    queue: &mut ImportQueue,
    windows: &mut ProjectWindows,
    requests: &mut MessageWriter<WorkspaceRequest>,
) {
    if !queue.is_open() {
        return;
    }

    let mut open = true;
    let mut confirm = false;
    theme::dialog(ctx, "Import Waveforms")
        .default_width(980.0)
        .resizable(true)
        .open(&mut open)
        .show(ctx, |ui| {
            let project = &workspace.project;
            for failure in &queue.failures {
                ui.label(egui::RichText::new(format!("✘ {failure}")).color(theme::BAD));
            }
            raw_section(ui, queue, requests);
            if queue.rows.is_empty() {
                return;
            }
            ui.label("Confirm each channel's sensor code, start time (UTC), sample rate and phenomenology. \
                      Unassigned sensors are plotted but not processed.");

            ui.horizontal(|ui| {
                ui.label("Start for all:");
                ui.add(egui::TextEdit::singleline(&mut windows.bulk_start).hint_text("2026-10-11T08:30:00Z").desired_width(220.0));
                if ui.button("Apply").clicked() {
                    for row in &mut queue.rows {
                        row.start.clone_from(&windows.bulk_start);
                        row.start_origin = "entered";
                    }
                }
                ui.separator();
                ui.label("Phenomenology for all:");
                phenomenology_choice(ui, "bulk_phenomenology", project, &mut windows.bulk_phenomenology, true);
                if ui.button("Apply").clicked() {
                    for row in &mut queue.rows {
                        row.phenomenology.clone_from(&windows.bulk_phenomenology);
                    }
                }
            });
            ui.separator();

            egui::ScrollArea::both().max_height(420.0).show(ui, |ui| {
                egui::Grid::new("import_rows").striped(true).num_columns(8).show(ui, |ui| {
                    for heading in ["", "File", "Sensor code", "Start (UTC)", "Rate (Hz)", "Samples", "Phenomenology", ""] {
                        ui.strong(heading);
                    }
                    ui.end_row();

                    for (index, row) in queue.rows.iter_mut().enumerate() {
                        ui.checkbox(&mut row.include, "");
                        ui.label(&row.file);
                        let height = ui.spacing().interact_size.y;
                        ui.add_sized([150.0, height], egui::TextEdit::singleline(&mut row.code));
                        ui.horizontal(|ui| {
                            ui.add_sized([240.0, height], egui::TextEdit::singleline(&mut row.start))
                                .on_hover_text(format!("From: {}", row.start_origin));
                            if row.start_origin != "file" && row.start_origin != "entered" {
                                ui.label(egui::RichText::new("⚠").color(theme::WARN))
                                    .on_hover_text(format!("Not in the file ({}); check before importing", row.start_origin));
                            }
                        });
                        ui.add_sized([80.0, height], egui::TextEdit::singleline(&mut row.rate));
                        match row.samples.kind() {
                            SampleKind::Real => ui.label(row.samples.len().to_string()),
                            SampleKind::Complex => ui.label(format!("{} IQ", row.samples.len())),
                        };

                        let known = project.sensor_by_code(row.code.trim()).and_then(|s| s.assignment.as_ref());
                        match known {
                            Some(assignment) => {
                                ui.label(phenomenology_label(project, Some(&assignment.phenomenology)))
                                    .on_hover_text("Known sensor: assignment comes from the project");
                            }
                            None => {
                                ui.horizontal(|ui| {
                                    phenomenology_choice(ui, ("import_phenomenology", index), project, &mut row.phenomenology, true);
                                    if let Some(suggestion) = &row.suggestion
                                        && row.phenomenology.as_ref() == Some(&suggestion.phenomenology)
                                    {
                                        ui.label(egui::RichText::new("suggested").weak().italics())
                                            .on_hover_text(format!("From {}; confirmed on import", suggestion.reason));
                                    }
                                });
                            }
                        }

                        match &row.problem {
                            Some(problem) => ui.label(egui::RichText::new(problem).color(theme::BAD)),
                            None => ui.label(""),
                        };
                        ui.end_row();
                    }
                });
            });

            ui.separator();
            ui.horizontal(|ui| {
                let count = queue.rows.iter().filter(|r| r.include).count();
                confirm = ui.add_enabled(count > 0, theme::primary(format!("Import {count}"))).clicked();
            });
        });

    if confirm {
        queue.confirm(workspace);
    }
    if !open {
        queue.cancel();
    }
}

fn sensors_window(ctx: &egui::Context, workspace: &mut Workspace, windows: &mut ProjectWindows) {
    if !windows.sensors_open {
        return;
    }

    let mut changes = Vec::new();
    let mut located: Option<(String, Change)> = None;
    let drafts = &mut windows.location_drafts;
    theme::tool_window(ctx, "Sensors").default_width(760.0).open(&mut windows.sensors_open).show(ctx, |ui| {
        let project = &workspace.project;
        if project.sensors().is_empty() {
            ui.label("No sensors. Import waveforms with File ▸ Import Waveforms…, or drop files on the window.");
            return;
        }
        egui::ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
            for sensor in project.sensors() {
                ui.horizontal(|ui| {
                    ui.strong(&sensor.code);
                    let mut chosen = sensor.assignment.as_ref().map(|a| a.phenomenology.clone());
                    if phenomenology_choice(ui, ("sensor", sensor.id), project, &mut chosen, false)
                        && let Some(phenomenology) = chosen
                    {
                        changes.push(Change::AssignPhenomenology { sensor: sensor.id, phenomenology });
                    }
                    if let Some(assignment) = &sensor.assignment {
                        let by = match &assignment.provenance {
                            Provenance::Manual { analyst, at } => format!("{analyst}, {at}"),
                            Provenance::Auto { stage, .. } => format!("auto: {stage}"),
                        };
                        ui.label(egui::RichText::new(format!("v{} · {by}", assignment.version)).weak());
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    let load = || match sensor.location {
                        Some(l) => ([l.latitude.to_string(), l.longitude.to_string(), l.elevation_m.to_string()], sensor.location),
                        None => (Default::default(), None),
                    };
                    let entry = drafts.entry(sensor.id).or_insert_with(load);
                    if entry.1 != sensor.location {
                        *entry = load();
                    }
                    let draft = &mut entry.0;
                    ui.label("Location");
                    for (text, hint, width) in [(0, "latitude °", 90.0), (1, "longitude °", 90.0), (2, "elevation m", 70.0)] {
                        ui.add(egui::TextEdit::singleline(&mut draft[text]).hint_text(hint).desired_width(width));
                    }
                    let parsed = parse_location(draft);
                    let unchanged = parsed.as_ref().ok() == sensor.location.as_ref();
                    if ui.add_enabled(parsed.is_ok() && !unchanged, egui::Button::new("Set")).clicked()
                        && let Ok(location) = parsed.clone()
                    {
                        let change = Change::SetSensorLocation { sensor: sensor.id, location: Some(location) };
                        located = Some((format!("Locate {}", sensor.code), change));
                    }
                    if sensor.location.is_some() && ui.button("Clear").clicked() {
                        located = Some((format!("Clear location of {}", sensor.code), Change::SetSensorLocation { sensor: sensor.id, location: None }));
                    }
                    match (&parsed, sensor.location) {
                        (Err(why), _) if draft.iter().any(|t| !t.trim().is_empty()) => {
                            ui.colored_label(theme::WARN, why);
                        }
                        (_, None) => {
                            ui.weak("none: not used to locate");
                        }
                        _ => {}
                    }
                });

                for trace in project.traces().iter().filter(|t| t.sensor == sensor.id) {
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        let (status, color) = match workspace.samples.get(&trace.id) {
                            Some(SampleState::Loaded(_)) => ("loaded".to_string(), None),
                            Some(SampleState::Loading) => ("loading…".to_string(), None),
                            Some(SampleState::Failed(error)) => (format!("✘ {error}"), Some(theme::BAD)),
                            None => ("not loaded".to_string(), Some(theme::WARN)),
                        };
                        ui.label(format!(
                            "{} · {} Hz · {:.3} s · {}",
                            trace.start,
                            trace.sample_rate,
                            trace.duration(),
                            trace.source.describe()
                        ));
                        let status = egui::RichText::new(status);
                        ui.label(match color {
                            Some(color) => status.color(color),
                            None => status.weak(),
                        });
                        if theme::danger(ui, true, egui::Button::new("✕").small()).on_hover_text("Remove trace from project").clicked() {
                            changes.push(Change::RemoveTrace { trace: trace.id });
                        }
                    });
                }
                ui.separator();
            }
        });
    });

    for change in changes {
        workspace.commit_or_report(change);
    }
    if let Some((label, change)) = located {
        workspace.execute_or_report(label, vec![change]);
    }
}

fn phenomenologies_window(ctx: &egui::Context, workspace: &Workspace, windows: &mut ProjectWindows) {
    if !windows.phenomenologies_open {
        return;
    }

    let mut edit = None;
    theme::tool_window(ctx, "Phenomenologies").default_width(640.0).open(&mut windows.phenomenologies_open).show(ctx, |ui| {
        egui::Grid::new("phenomenologies").striped(true).num_columns(7).show(ui, |ui| {
            for heading in ["Name", "Id", "Quantity", "Samples", "Propagation", "Ver", ""] {
                ui.strong(heading);
            }
            ui.end_row();
            for p in workspace.project.phenomenologies() {
                ui.label(egui::RichText::new(&p.name).color(theme::color32(p.color)));
                ui.label(egui::RichText::new(p.id.as_str()).monospace());
                ui.label(format!("{} ({})", p.quantity, p.units));
                ui.label(p.sample_kind.label());
                ui.label(propagation_summary(&p.propagation));
                ui.label(p.version.to_string());
                if ui.small_button("Edit").clicked() {
                    edit = Some(PhenomenologyDraft::from(p));
                }
                ui.end_row();
            }
        });
        ui.separator();
        if ui.button("New…").clicked() {
            edit = Some(PhenomenologyDraft::default());
        }
    });
    if edit.is_some() {
        windows.editor = edit;
    }
}

fn propagation_summary(propagation: &Propagation) -> String {
    match propagation {
        Propagation::Constant { speed_m_s } => format!("{speed_m_s} m/s"),
        Propagation::Phases(phases) => {
            phases.iter().map(|p| format!("{} {} m/s", p.name, p.speed_m_s)).collect::<Vec<_>>().join(", ")
        }
        Propagation::Unknown => "unknown".to_string(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PropagationKind {
    Unknown,
    Constant,
    Phases,
}

struct PhenomenologyDraft {
    /// Version being replaced; `None` for a new definition.
    replaces: Option<u32>,
    id: String,
    name: String,
    quantity: String,
    units: String,
    sample_kind: SampleKind,
    propagation: PropagationKind,
    speed_m_s: f64,
    phases: Vec<(String, f64)>,
    color: [f32; 3],
    problem: Option<String>,
}

impl Default for PhenomenologyDraft {
    fn default() -> Self {
        Self {
            replaces: None,
            id: String::new(),
            name: String::new(),
            quantity: String::new(),
            units: String::new(),
            sample_kind: SampleKind::Real,
            propagation: PropagationKind::Unknown,
            speed_m_s: 343.0,
            phases: Vec::new(),
            color: [UNASSIGNED_COLOR.red, UNASSIGNED_COLOR.green, UNASSIGNED_COLOR.blue],
            problem: None,
        }
    }
}

impl From<&Phenomenology> for PhenomenologyDraft {
    fn from(p: &Phenomenology) -> Self {
        let (propagation, speed_m_s, phases) = match &p.propagation {
            Propagation::Unknown => (PropagationKind::Unknown, 343.0, Vec::new()),
            Propagation::Constant { speed_m_s } => (PropagationKind::Constant, *speed_m_s, Vec::new()),
            Propagation::Phases(phases) => {
                (PropagationKind::Phases, 343.0, phases.iter().map(|ph| (ph.name.clone(), ph.speed_m_s)).collect())
            }
        };
        Self {
            replaces: Some(p.version),
            id: p.id.0.clone(),
            name: p.name.clone(),
            quantity: p.quantity.clone(),
            units: p.units.clone(),
            sample_kind: p.sample_kind,
            propagation,
            speed_m_s,
            phases,
            color: [p.color.red, p.color.green, p.color.blue],
            problem: None,
        }
    }
}

impl PhenomenologyDraft {
    fn build(&self, project: &Project) -> Result<Phenomenology, String> {
        let id = self.id.trim();
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') {
            return Err("id must be lowercase letters, digits, - or _".to_string());
        }
        if self.name.trim().is_empty() {
            return Err("name required".to_string());
        }
        if self.replaces.is_none() && project.phenomenology(&PhenomenologyId::new(id)).is_some() {
            return Err(format!("{id} already exists"));
        }
        let propagation = match self.propagation {
            PropagationKind::Unknown => Propagation::Unknown,
            PropagationKind::Constant => Propagation::Constant { speed_m_s: self.speed_m_s },
            PropagationKind::Phases if self.phases.is_empty() => return Err("add at least one phase".to_string()),
            PropagationKind::Phases => Propagation::Phases(
                self.phases.iter().map(|(name, speed_m_s)| Phase { name: name.clone(), speed_m_s: *speed_m_s }).collect(),
            ),
        };
        Ok(Phenomenology {
            id: PhenomenologyId::new(id),
            name: self.name.trim().to_string(),
            quantity: self.quantity.trim().to_string(),
            units: self.units.trim().to_string(),
            sample_kind: self.sample_kind,
            propagation,
            color: Rgba::new(self.color[0], self.color[1], self.color[2]),
            version: self.replaces.map_or(1, |v| v + 1),
        })
    }
}

fn editor_window(ctx: &egui::Context, workspace: &mut Workspace, windows: &mut ProjectWindows) {
    let Some(draft) = &mut windows.editor else {
        return;
    };

    let mut open = true;
    let mut save = false;
    let title = if draft.replaces.is_some() { "Edit Phenomenology" } else { "New Phenomenology" };
    theme::dialog(ctx, title).open(&mut open).show(ctx, |ui| {
        egui::Grid::new("phenomenology_editor").num_columns(2).show(ui, |ui| {
            ui.label("Id");
            ui.add_enabled(draft.replaces.is_none(), egui::TextEdit::singleline(&mut draft.id).hint_text("e.g. hydroacoustic"));
            ui.end_row();
            ui.label("Name");
            ui.text_edit_singleline(&mut draft.name);
            ui.end_row();
            ui.label("Quantity");
            ui.text_edit_singleline(&mut draft.quantity);
            ui.end_row();
            ui.label("Units");
            ui.text_edit_singleline(&mut draft.units);
            ui.end_row();
            ui.label("Samples");
            ui.horizontal(|ui| {
                for kind in SampleKind::ALL {
                    ui.radio_value(&mut draft.sample_kind, kind, kind.label());
                }
            });
            ui.end_row();
            ui.label("Colour");
            ui.color_edit_button_rgb(&mut draft.color);
            ui.end_row();
            ui.label("Propagation");
            ui.horizontal(|ui| {
                ui.radio_value(&mut draft.propagation, PropagationKind::Unknown, "Unknown");
                ui.radio_value(&mut draft.propagation, PropagationKind::Constant, "Constant speed");
                ui.radio_value(&mut draft.propagation, PropagationKind::Phases, "Phases");
            });
            ui.end_row();
        });

        match draft.propagation {
            PropagationKind::Unknown => {
                ui.label(egui::RichText::new("Aligned by picks or manual offsets only.").weak());
            }
            PropagationKind::Constant => {
                ui.horizontal(|ui| {
                    ui.label("Speed");
                    ui.add(egui::DragValue::new(&mut draft.speed_m_s).range(1e-3..=3e8).suffix(" m/s"));
                });
            }
            PropagationKind::Phases => {
                let mut remove = None;
                for (index, (name, speed)) in draft.phases.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(name).desired_width(80.0));
                        ui.add(egui::DragValue::new(speed).range(1e-3..=3e8).suffix(" m/s"));
                        if theme::danger(ui, true, egui::Button::new("✕").small()).clicked() {
                            remove = Some(index);
                        }
                    });
                }
                if let Some(index) = remove {
                    draft.phases.remove(index);
                }
                if ui.small_button("Add phase").clicked() {
                    draft.phases.push((format!("Phase {}", draft.phases.len() + 1), 1000.0));
                }
            }
        }

        if let Some(problem) = &draft.problem {
            ui.label(egui::RichText::new(problem).color(theme::BAD));
        }
        ui.separator();
        if let Some(version) = draft.replaces {
            ui.label(egui::RichText::new(format!("Saves as v{}; existing assignments keep v{version}.", version + 1)).weak());
        }
        save = ui.add(theme::primary("Save")).clicked();
    });

    if save {
        match draft.build(&workspace.project) {
            Ok(definition) => {
                if workspace.commit_or_report(Change::DefinePhenomenology(definition)) {
                    windows.editor = None;
                }
            }
            Err(problem) => draft.problem = Some(problem),
        }
    } else if !open {
        windows.editor = None;
    }
}

fn discard_window(ctx: &egui::Context, workspace: &mut Workspace, requests: &mut MessageWriter<WorkspaceRequest>) {
    let Some(request) = &workspace.confirm_discard else {
        return;
    };

    let mut decided = None;
    theme::dialog(ctx, "Unsaved Project").show(ctx, |ui| {
        ui.label("This project has not been saved. Discard it?");
        ui.horizontal(|ui| {
            if theme::danger(ui, true, egui::Button::new("Discard")).clicked() {
                decided = Some(true);
            }
            if ui.add(theme::primary("Save As…")).clicked() {
                decided = Some(false);
                requests.write(WorkspaceRequest::SaveAsDialog);
            }
            if ui.button("Cancel").clicked() {
                decided = Some(false);
            }
        });
    });

    match decided {
        Some(true) => {
            requests.write(WorkspaceRequest::Discard(Box::new(request.clone())));
            workspace.confirm_discard = None;
        }
        Some(false) => workspace.confirm_discard = None,
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::Analyst;

    #[test]
    fn locations_parse_with_optional_elevation() {
        let text = |a: &str, b: &str, c: &str| [a.to_string(), b.to_string(), c.to_string()];
        assert_eq!(parse_location(&text("35", "-106.5", "")), Ok(Location { latitude: 35.0, longitude: -106.5, elevation_m: 0.0 }));
        assert!(parse_location(&text("95", "0", "0")).is_err());
        assert!(parse_location(&text("x", "0", "0")).is_err());
    }

    #[test]
    fn editing_a_phenomenology_bumps_its_version() {
        let (project, _) = Project::create("p", &Analyst("a".to_string()));
        let seismic = project.phenomenology(&PhenomenologyId::new("seismic")).unwrap();
        let mut draft = PhenomenologyDraft::from(seismic);
        draft.name = "Seismic (regional)".to_string();
        let built = draft.build(&project).unwrap();
        assert_eq!(built.version, 2);
        assert_eq!(built.propagation, seismic.propagation);
    }

    #[test]
    fn new_phenomenologies_need_a_fresh_valid_id() {
        let (project, _) = Project::create("p", &Analyst("a".to_string()));
        let draft = |id: &str| PhenomenologyDraft { id: id.to_string(), name: "X".to_string(), ..Default::default() };
        assert!(draft("seismic").build(&project).is_err());
        assert!(draft("Bad Id").build(&project).is_err());
        assert_eq!(draft("hydro").build(&project).unwrap().version, 1);
    }
}
