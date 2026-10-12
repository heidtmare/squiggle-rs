//! The application chrome: menu bar, toolbar and squiggle sidebar.
//!
//! The Java version described this layout in FXML, with a custom `Sidebar`
//! control whose rotated handles toggled a content pane. egui expresses the
//! same arrangement directly, and the sidebar's squiggle list -- an empty
//! placeholder pane in the original -- is filled in here.

use bevy::camera::{CameraOutputMode, Viewport, visibility::RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::render_resource::BlendState;
use bevy::window::PrimaryWindow;
use bevy_egui::{
    EguiContext, EguiContexts, EguiGlobalSettings, EguiPrimaryContextPass, PrimaryEguiContext, egui,
};
use squiggle_core::{
    Component, Project, Rgba, Squiggle, Transform, TransformOptions, Wavelet, WaveletOutput, Window as Taper,
};

use crate::demo::{self, DemoOptions};
use crate::plot::{AddSquiggle, FitToData, Plot, PlotCamera, PlotMode, RemoveSquiggle, SquiggleModel};
use crate::classify_ui::{self, ClassifyWindows};
use crate::events_ui::{self, EventWindows};
use crate::locate_ui::{self, LocateWindows};
use crate::process_ui::{self, ProcessWindows};
use crate::project_ui::{self, ProjectAccess, ProjectWindows};
use crate::workspace::{SyntheticEvent, WorkspaceRequest};
use crate::showcase::{LoadScene, Scene, Showcase};
use crate::tracks::{self, RowOrder, Track, TrackView};

const HANDLE_WIDTH: f32 = 26.0;
const SIDEBAR_WIDTH: f32 = 300.0;
const SHOWCASE_WIDTH: f32 = 340.0;
const UNDO: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT), egui::Key::Z);
const PASS_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 210, 110);
const FAIL_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 80, 80);

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Chrome>()
            .init_resource::<ProjectWindows>()
            .init_resource::<ProcessWindows>()
            .init_resource::<LocateWindows>()
            .init_resource::<EventWindows>()
            .init_resource::<ClassifyWindows>()
            .add_systems(Startup, setup)
            .add_systems(
                EguiPrimaryContextPass,
                (
                    chrome,
                    tracks::tracks_ui,
                    project_ui::windows,
                    process_ui::windows,
                    locate_ui::windows,
                    events_ui::windows,
                    classify_ui::windows,
                )
                    .chain(),
            );
    }
}

/// What the chrome can ask the plot to do. Grouping the writers keeps them out
/// of the argument list of every function that needs one.
#[derive(SystemParam)]
pub struct PlotCommands<'w> {
    add: MessageWriter<'w, AddSquiggle>,
    remove: MessageWriter<'w, RemoveSquiggle>,
    fit: MessageWriter<'w, FitToData>,
    load: MessageWriter<'w, LoadScene>,
    exit: MessageWriter<'w, AppExit>,
}

/// The tracks view, as the chrome needs it.
#[derive(SystemParam)]
pub struct TrackAccess<'w, 's> {
    mode: ResMut<'w, PlotMode>,
    view: ResMut<'w, TrackView>,
    tracks: Query<'w, 's, &'static mut Track>,
    process: ResMut<'w, ProcessWindows>,
    locate: ResMut<'w, LocateWindows>,
    events: ResMut<'w, EventWindows>,
    classify: ResMut<'w, ClassifyWindows>,
}

/// Which parts of the chrome are open.
#[derive(Resource, Default)]
pub struct Chrome {
    pub sidebar_open: bool,
    pub about_open: bool,
    pub transform_settings_open: bool,
    pub transforms: TransformSettings,
    /// Why the last transform failed, until dismissed.
    pub transform_error: Option<String>,
}

/// The parameters behind the Transform menu.
pub struct TransformSettings {
    pub options: TransformOptions,
    pub stft_window: Taper,
    pub stft_size: usize,
    pub stft_hop: usize,
    pub wavelet: Wavelet,
    pub wavelet_levels: usize,
    pub wavelet_output: WaveletOutput,
    /// Delete the squiggles a transform was applied to.
    pub replace_source: bool,
    /// Frame the plot on the results; a spectrum rarely shares a scale with
    /// the signal it came from.
    pub fit_after: bool,
}

impl Default for TransformSettings {
    fn default() -> Self {
        Self {
            options: TransformOptions::default(),
            stft_window: Taper::default(),
            stft_size: 256,
            stft_hop: 128,
            wavelet: Wavelet::default(),
            wavelet_levels: 4,
            wavelet_output: WaveletOutput::default(),
            replace_source: false,
            fit_after: true,
        }
    }
}

impl TransformSettings {
    /// Every transform the menu offers, configured from these settings.
    fn transforms(&self) -> [Transform; 6] {
        [
            Transform::Fourier,
            Transform::Dft,
            Transform::Fft,
            Transform::Stft { window: self.stft_window, size: self.stft_size, hop: self.stft_hop },
            Transform::InverseFourier,
            Transform::Wavelet {
                family: self.wavelet,
                levels: self.wavelet_levels,
                output: self.wavelet_output,
            },
        ]
    }
}

fn setup(mut commands: Commands, mut egui_settings: ResMut<EguiGlobalSettings>) {
    // The plot camera must stay free of egui, so the context gets its own.
    egui_settings.auto_create_primary_context = false;

    commands.spawn((
        PrimaryEguiContext,
        Camera2d,
        // Nothing in the world belongs on this camera; it draws only the chrome.
        RenderLayers::none(),
        Camera {
            order: 1,
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
    ));
}

/// Draws the chrome and hands the space left over to the plot camera.
// A Bevy system declares everything it touches in its signature; eleven is
// what drawing the whole chrome needs.
#[allow(clippy::too_many_arguments)]
fn chrome(
    mut contexts: EguiContexts,
    mut chrome: ResMut<Chrome>,
    mut plot: ResMut<Plot>,
    mut camera: Single<&mut Camera, (With<PlotCamera>, Without<EguiContext>)>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut squiggles: Query<(Entity, &mut SquiggleModel)>,
    options: Res<DemoOptions>,
    mut showcase: ResMut<Showcase>,
    mut commands: PlotCommands,
    mut project: ProjectAccess,
    mut tracks: TrackAccess,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    ctx.set_visuals(egui::Visuals::dark());

    // Undo and redo, unless a text field has the keyboard.
    if !ctx.egui_wants_keyboard_input() {
        if ctx.input_mut(|i| i.consume_shortcut(&REDO)) {
            project.workspace.redo();
        } else if ctx.input_mut(|i| i.consume_shortcut(&UNDO)) {
            project.workspace.undo();
        }
    }

    // Panels are laid out inside a background Ui covering the whole viewport;
    // whatever rectangle survives is where the plot goes.
    let mut root = egui::Ui::new(
        ctx.clone(),
        "squiggle_chrome".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    let top = egui::Panel::top("squiggle_top")
        .show(&mut root, |ui| {
            menu_bar(ui, &mut chrome, &mut squiggles, &mut commands, &mut project, &mut tracks, &plot);
            ui.separator();
            toolbar(ui, &mut plot, &squiggles, &options, &mut commands, &mut tracks);
        })
        .response
        .rect
        .height();

    let mut left = egui::Panel::left("squiggle_sidebar_handle")
        .resizable(false)
        .exact_size(HANDLE_WIDTH)
        .show_separator_line(false)
        .show(&mut root, |ui| {
            // The Java sidebar stacked rotated toggle buttons down its edge.
            ui.toggle_value(&mut chrome.sidebar_open, vertical("Squiggles"))
                .on_hover_text("Show the squiggles in this plot");
        })
        .response
        .rect
        .width();

    if chrome.sidebar_open {
        left += egui::Panel::left("squiggle_sidebar")
            .default_size(SIDEBAR_WIDTH)
            .min_size(180.0)
            .show(&mut root, |ui| squiggle_list(ui, &mut squiggles, &mut commands))
            .response
            .rect
            .width();
    }

    let right = match &showcase.current {
        Some(_) => egui::Panel::right("squiggle_showcase")
            .default_size(SHOWCASE_WIDTH)
            .min_size(220.0)
            .show(&mut root, |ui| showcase_panel(ui, &mut showcase, &mut commands))
            .response
            .rect
            .width(),
        None => 0.0,
    };

    about_window(&ctx, &mut chrome);
    transform_window(&ctx, &mut chrome);
    transform_error_window(&ctx, &mut chrome);

    hand_viewport_to_plot(&window, &mut camera, &mut plot, left, right, top);
    Ok(())
}

fn menu_bar(
    ui: &mut egui::Ui,
    chrome: &mut Chrome,
    squiggles: &mut Query<(Entity, &mut SquiggleModel)>,
    commands: &mut PlotCommands,
    project: &mut ProjectAccess,
    tracks: &mut TrackAccess,
    plot: &Plot,
) {
    egui::MenuBar::new().ui(ui, |ui| {
        ui.menu_button("File", |ui| {
            project_ui::file_menu(ui, project);
            ui.separator();
            if ui.button("Close").clicked() {
                commands.exit.write(AppExit::Success);
                ui.close();
            }
        });
        ui.menu_button("Edit", |ui| {
            let workspace = &mut project.workspace;
            let undo = workspace.undo_label().map_or("Undo".to_string(), |l| format!("Undo {l}"));
            let shortcut = ui.ctx().format_shortcut(&UNDO);
            if ui.add_enabled(workspace.undo_label().is_some(), egui::Button::new(undo).shortcut_text(shortcut)).clicked() {
                workspace.undo();
                ui.close();
            }
            let redo = workspace.redo_label().map_or("Redo".to_string(), |l| format!("Redo {l}"));
            let shortcut = ui.ctx().format_shortcut(&REDO);
            if ui.add_enabled(workspace.redo_label().is_some(), egui::Button::new(redo).shortcut_text(shortcut)).clicked() {
                workspace.redo();
                ui.close();
            }
            ui.separator();
            let selected: Vec<Entity> = squiggles
                .iter()
                .filter(|(_, model)| model.selected)
                .map(|(entity, _)| entity)
                .collect();

            if ui
                .add_enabled(!selected.is_empty(), egui::Button::new("Delete selected"))
                .clicked()
            {
                for entity in selected {
                    commands.remove.write(RemoveSquiggle(entity));
                }
                ui.close();
            }
        });
        ui.menu_button("Project", |ui| project_ui::project_menu(ui, project));
        ui.menu_button("Process", |ui| {
            process_ui::process_menu(ui, &mut tracks.process, &mut tracks.view);
            ui.separator();
            locate_ui::locate_menu(ui, &mut tracks.locate);
            let locate = tracks.locate.pick_open || tracks.locate.locate_open;
            if tracks.process.processing_open || tracks.process.align_open || locate {
                *tracks.mode = PlotMode::Tracks;
            }
        });
        ui.menu_button("Events", |ui| {
            events_ui::events_menu(ui, &mut tracks.events, &mut tracks.view);
            ui.separator();
            classify_ui::classify_menu(ui, &mut tracks.classify);
            if tracks.events.detect_open || tracks.events.catalogue_open || tracks.classify.features_open {
                *tracks.mode = PlotMode::Tracks;
            }
        });
        ui.menu_button("Transform", |ui| {
            let sources = TransformSources { squiggles, tracks, width: plot.view.plot_width() };
            transform_menu(ui, chrome, sources, commands, &project.workspace.project);
        });
        ui.menu_button("Demo", |ui| {
            if ui
                .button("Synthetic event (alignment)")
                .on_hover_text("A new project: seven sensors recording one event with known delays")
                .clicked()
            {
                project.requests.write(WorkspaceRequest::Synthetic(SyntheticEvent::Alignment));
                ui.close();
            }
            if ui
                .button("Synthetic located event")
                .on_hover_text("A new project: EMP, seismic and infrasound sensors at known positions recording one explosion")
                .clicked()
            {
                project.requests.write(WorkspaceRequest::Synthetic(SyntheticEvent::Located));
                ui.close();
            }
            if ui
                .button("Synthetic swarm")
                .on_hover_text("A new project: two minutes on eight seismic stations with six events and glitches")
                .clicked()
            {
                project.requests.write(WorkspaceRequest::Synthetic(SyntheticEvent::Swarm));
                ui.close();
            }
            if ui
                .button("Synthetic earthquakes and explosions")
                .on_hover_text("A new project: seven minutes of five earthquakes and five explosions on seismic and infrasound sensors")
                .clicked()
            {
                project.requests.write(WorkspaceRequest::Synthetic(SyntheticEvent::Mixed));
                ui.close();
            }
            ui.separator();
            for scene in Scene::ALL {
                if ui
                    .button(scene.label())
                    .on_hover_text("Replace the plot with a known signal, its transform and checks on the result")
                    .clicked()
                {
                    commands.load.write(LoadScene(scene));
                    ui.close();
                }
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.button("About").clicked() {
                chrome.about_open = true;
                ui.close();
            }
        });
    });
}

/// What a transform runs on: selected squiggles in overlay mode, the visible
/// window of selected tracks in tracks mode.
struct TransformSources<'a, 'b, 'w, 's, 'q, 'tw, 'ts> {
    squiggles: &'a Query<'w, 's, (Entity, &'q mut SquiggleModel)>,
    tracks: &'b mut TrackAccess<'tw, 'ts>,
    width: f64,
}

impl TransformSources<'_, '_, '_, '_, '_, '_, '_> {
    fn any_selected(&self) -> bool {
        match *self.tracks.mode {
            PlotMode::Overlay => self.squiggles.iter().any(|(_, model)| model.selected),
            PlotMode::Tracks => self.tracks.tracks.iter().any(|track| track.selected),
        }
    }

    /// `(name, squiggle entity to replace, input or why not)` per selected source.
    fn inputs(&self, project: &Project) -> Vec<(String, Option<Entity>, Result<Squiggle, String>)> {
        let unassigned = "no phenomenology assigned (Project ▸ Sensors…)".to_string();
        match *self.tracks.mode {
            PlotMode::Overlay => self
                .squiggles
                .iter()
                .filter(|(_, model)| model.selected)
                .map(|(entity, model)| {
                    let blocked = model
                        .trace
                        .and_then(|id| project.trace(id))
                        .is_some_and(|trace| project.trace_phenomenology(trace).is_none());
                    let input = if blocked { Err(unassigned.clone()) } else { Ok(model.0.clone()) };
                    (model.name.clone(), Some(entity), input)
                })
                .collect(),
            PlotMode::Tracks => self
                .tracks
                .tracks
                .iter()
                .filter(|track| track.selected)
                .map(|track| {
                    let input = match track.phenomenology {
                        None => Err(unassigned.clone()),
                        Some(_) => track.window_squiggle(&self.tracks.view.time, self.width),
                    };
                    (track.name(), None, input)
                })
                .collect(),
        }
    }
}

fn transform_menu(
    ui: &mut egui::Ui,
    chrome: &mut Chrome,
    mut sources: TransformSources,
    commands: &mut PlotCommands,
    project: &Project,
) {
    let any_selected = sources.any_selected();
    let hint = |transform: &Transform| match transform {
        Transform::Fourier => "Continuous Fourier integral, honouring uneven sample spacing",
        Transform::Dft => "Discrete Fourier transform, summed directly — slow on long squiggles",
        Transform::Fft => "Discrete Fourier transform by FFT",
        Transform::Stft { .. } => "A spectrum per windowed frame, as a waterfall",
        Transform::InverseFourier => "Back from a spectrum to a signal",
        Transform::Wavelet { .. } => "Split into wavelet levels, coarse to fine",
    };

    for transform in chrome.transforms.transforms() {
        if matches!(transform, Transform::InverseFourier | Transform::Stft { .. }) {
            ui.separator();
        }
        let response = ui
            .add_enabled(any_selected, egui::Button::new(transform.label()))
            .on_hover_text(hint(&transform))
            .on_disabled_hover_text("Select a squiggle or track to transform");
        if response.clicked() {
            apply_transform(transform, chrome, &mut sources, commands, project);
            ui.close();
        }
    }

    ui.separator();
    if ui.button("Settings…").clicked() {
        chrome.transform_settings_open = true;
        ui.close();
    }
}

/// Transforms every selected source, adding the results to the overlay.
/// Traces whose sensor has no phenomenology are refused.
fn apply_transform(
    transform: Transform,
    chrome: &mut Chrome,
    sources: &mut TransformSources,
    commands: &mut PlotCommands,
    project: &Project,
) {
    let settings = &chrome.transforms;
    let mut failures = Vec::new();
    let mut added = false;

    for (name, entity, input) in sources.inputs(project) {
        let outputs = input.and_then(|squiggle| transform.apply(&squiggle, &settings.options).map_err(|e| e.to_string()));
        match outputs {
            Ok(outputs) => {
                added |= !outputs.is_empty();
                for output in outputs {
                    commands.add.write(AddSquiggle(output));
                }
                if let (true, Some(entity)) = (settings.replace_source, entity) {
                    commands.remove.write(RemoveSquiggle(entity));
                }
            }
            Err(error) => failures.push(format!("{name}: {error}")),
        }
    }

    if added {
        *sources.tracks.mode = PlotMode::Overlay;
        if settings.fit_after {
            commands.fit.write(FitToData);
        }
    }
    chrome.transform_error = (!failures.is_empty())
        .then(|| format!("{} failed on\n{}", transform.label(), failures.join("\n")));
}

fn transform_window(ctx: &egui::Context, chrome: &mut Chrome) {
    if !chrome.transform_settings_open {
        return;
    }

    let settings = &mut chrome.transforms;
    egui::Window::new("Transform Settings")
        .collapsible(false)
        .resizable(false)
        .open(&mut chrome.transform_settings_open)
        .show(ctx, |ui| {
            ui.heading("Spectra");
            egui::Grid::new("transform_spectra").num_columns(2).show(ui, |ui| {
                ui.label("Plot");
                choice(ui, "component", &mut settings.options.component, &Component::ALL, |c| c.label());
                ui.end_row();

                ui.label("Frequencies");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut settings.options.one_sided, true, "One-sided");
                    ui.radio_value(&mut settings.options.one_sided, false, "Two-sided");
                });
                ui.end_row();
            });

            ui.separator();
            ui.heading("Short-time Fourier");
            egui::Grid::new("transform_stft").num_columns(2).show(ui, |ui| {
                ui.label("Window");
                choice(ui, "window", &mut settings.stft_window, &Taper::ALL, |w| w.label());
                ui.end_row();

                ui.label("Frame size");
                ui.add(egui::DragValue::new(&mut settings.stft_size).range(2..=65_536).suffix(" samples"));
                ui.end_row();

                ui.label("Hop");
                ui.add(egui::DragValue::new(&mut settings.stft_hop).range(1..=65_536).suffix(" samples"));
                ui.end_row();
            });

            ui.separator();
            ui.heading("Wavelet");
            egui::Grid::new("transform_wavelet").num_columns(2).show(ui, |ui| {
                ui.label("Family");
                choice(ui, "wavelet", &mut settings.wavelet, &Wavelet::ALL, |w| w.label());
                ui.end_row();

                ui.label("Levels");
                ui.add(egui::DragValue::new(&mut settings.wavelet_levels).range(1..=16))
                    .on_hover_text("Clamped to as many as the squiggle is long enough for");
                ui.end_row();

                ui.label("Plot");
                choice(ui, "wavelet_output", &mut settings.wavelet_output, &WaveletOutput::ALL, |o| {
                    o.label()
                });
                ui.end_row();
            });

            ui.separator();
            ui.checkbox(&mut settings.options.stack, "Stack frames and levels")
                .on_hover_text("Offset STFT frames and wavelet levels so they do not overlap");
            ui.checkbox(&mut settings.replace_source, "Replace the source squiggles");
            ui.checkbox(&mut settings.fit_after, "Fit the plot to the results");
        });
}

fn transform_error_window(ctx: &egui::Context, chrome: &mut Chrome) {
    let Some(message) = &chrome.transform_error else {
        return;
    };

    let mut open = true;
    egui::Window::new("Transform Failed")
        .collapsible(false)
        .resizable(false)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label(message);
        });
    if !open {
        chrome.transform_error = None;
    }
}

/// A drop-down over a fixed set of values.
fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut T,
    options: &[T],
    label: impl Fn(T) -> &'static str,
) {
    egui::ComboBox::from_id_salt(id).selected_text(label(*value)).show_ui(ui, |ui| {
        for option in options {
            ui.selectable_value(value, *option, label(*option));
        }
    });
}

fn toolbar(
    ui: &mut egui::Ui,
    plot: &mut Plot,
    squiggles: &Query<(Entity, &mut SquiggleModel)>,
    options: &DemoOptions,
    commands: &mut PlotCommands,
    tracks: &mut TrackAccess,
) {
    ui.horizontal(|ui| {
        let mode = &mut *tracks.mode;
        ui.selectable_value(mode, PlotMode::Tracks, "Tracks")
            .on_hover_text("Imported traces stacked on a shared time axis");
        ui.selectable_value(mode, PlotMode::Overlay, "Overlay")
            .on_hover_text("Free squiggles: demos and transform results");
        ui.separator();

        if ui.button("Add Squiggle").clicked() {
            let index = squiggles.iter().count();
            commands.add.write(AddSquiggle(demo::generate(index, options.sample_count)));
            *tracks.mode = PlotMode::Overlay;
        }
        let (any, hint) = match *tracks.mode {
            PlotMode::Overlay => (squiggles.iter().count() > 0, "Frame every squiggle"),
            PlotMode::Tracks => (!tracks.tracks.is_empty(), "Frame every track's time span (or double-click)"),
        };
        if ui.add_enabled(any, egui::Button::new("Fit")).on_hover_text(hint).clicked() {
            match *tracks.mode {
                PlotMode::Overlay => {
                    commands.fit.write(FitToData);
                }
                PlotMode::Tracks => {
                    tracks.view.fit_requested = true;
                    tracks.view.gain = 1.0;
                }
            }
        }
        if *tracks.mode == PlotMode::Tracks {
            ui.label("Rows by");
            choice(ui, "row_order", &mut tracks.view.order, &RowOrder::ALL, |o| o.label());
            ui.checkbox(&mut tracks.view.show_raw, "Raw").on_hover_text("Plot tracks without their processing");
        }

        ui.separator();
        ui.checkbox(&mut plot.axes_visible, "Axes");
        ui.checkbox(&mut plot.crosshairs_visible, "Crosshairs");
        ui.checkbox(&mut plot.controls_enabled, "Controls")
            .on_hover_text("Drag to pan, scroll to zoom, click to select");

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(&plot.name).weak());
        });
    });
}

fn squiggle_list(
    ui: &mut egui::Ui,
    squiggles: &mut Query<(Entity, &mut SquiggleModel)>,
    commands: &mut PlotCommands,
) {
    ui.heading("Squiggles");
    ui.separator();

    if squiggles.is_empty() {
        ui.label("No squiggles yet. Import waveforms from the File menu or drop files on the window; “Add Squiggle” makes a demo one.");
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        for (entity, mut model) in squiggles.iter_mut() {
            ui.horizontal(|ui| {
                swatch(ui, model.style.stroke);

                let label = format!("{} ({} pts)", model.name, model.len());
                // Selecting here is the same state the plot's picking toggles.
                let mut selected = model.selected;
                if ui.toggle_value(&mut selected, label).changed() {
                    model.selected = selected;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("✕").on_hover_text("Remove").clicked() {
                        commands.remove.write(RemoveSquiggle(entity));
                    }
                });
            });
        }
    });
}

/// The demo scene's legend, and its checks marked pass or fail.
fn showcase_panel(ui: &mut egui::Ui, showcase: &mut Showcase, commands: &mut PlotCommands) {
    let Some(loaded) = &showcase.current else {
        return;
    };

    let mut close = false;
    ui.horizontal(|ui| {
        ui.heading(loaded.title);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            close = ui.small_button("Close").on_hover_text("Close the demo panel").clicked();
        });
    });

    ui.horizontal(|ui| {
        for scene in Scene::ALL {
            if ui.selectable_label(loaded.scene == scene, scene.label()).clicked() && loaded.scene != scene {
                commands.load.write(LoadScene(scene));
            }
        }
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label(loaded.description);
        ui.add_space(6.0);
        for (color, meaning) in &loaded.legend {
            ui.horizontal(|ui| {
                swatch(ui, *color);
                ui.label(*meaning);
            });
        }

        ui.separator();
        let passed = loaded.checks.iter().filter(|check| check.passed).count();
        let (summary, color) = if loaded.all_passed() {
            (format!("All {passed} checks pass"), PASS_COLOR)
        } else {
            (format!("{passed} of {} checks pass", loaded.checks.len()), FAIL_COLOR)
        };
        ui.label(egui::RichText::new(summary).strong().color(color));
        ui.add_space(4.0);

        for check in &loaded.checks {
            ui.horizontal_top(|ui| {
                let (mark, color) = if check.passed { ("✔", PASS_COLOR) } else { ("✘", FAIL_COLOR) };
                ui.label(egui::RichText::new(mark).color(color).strong());
                ui.vertical(|ui| {
                    ui.label(&check.claim);
                    ui.label(egui::RichText::new(&check.measured).weak().monospace());
                });
            });
            ui.add_space(2.0);
        }
    });

    if close {
        showcase.current = None;
    }
}

fn about_window(ctx: &egui::Context, chrome: &mut Chrome) {
    if !chrome.about_open {
        return;
    }

    egui::Window::new("About Squiggle")
        .collapsible(false)
        .resizable(false)
        .open(&mut chrome.about_open)
        .show(ctx, |ui| {
            ui.label("Squiggle — a waveform plot.");
            ui.label("A Rust and Bevy port of the original JavaFX application.");
            ui.separator();
            ui.strong("Tracks");
            ui.label("Scroll to zoom time, drag to pan time and rows, pinch or Control-scroll to scale amplitude.");
            ui.label("Click a row to select it; right-click for display, y scale, I/Q component, reference and offset lock. Double-click to fit.");
            ui.label("Shift+drag a row to shift it in time (snaps to moveout from the saved origin, to the reference's picks and to correlation peaks; Alt disables). Arrow keys nudge selected rows one sample, ten with Shift.");
            ui.label("Over a row, P places a first-phase pick at the cursor, S an S pick; Delete removes the nearest pick.");
            ui.strong("Overlay");
            ui.label("Drag to pan. Scroll to zoom — hold Alt to keep the x scale, Control to keep the y scale.");
            ui.label("Click a squiggle to select it and show its samples.");
            ui.separator();
            ui.label("Process ▸ Filter and detrend preprocesses selected tracks; Process ▸ Align measures and applies delays.");
            ui.label("Process ▸ Pick finds onsets; Process ▸ Locate fits source position and origin time, and removes moveout.");
            ui.label("Events ▸ Detect finds and associates events; Events ▸ Catalogue reviews them. ⌘/Ctrl+drag makes an event; drag its edges to resize; right-click to split. While reviewing: ] / [ next and previous, A accept, R reject, 1–9 label, Esc close.");
            ui.label("Events ▸ Classify runs features and the rules classifier; Features shows the reviewed event's evidence; Labels edits the taxonomy; Export dataset writes Parquet for training.");
            ui.label("Edits to processing, offsets, picks, sensor locations, origins and events undo with ⌘Z / Ctrl+Z and redo with ⇧⌘Z.");
            ui.label("Transform turns the selected squiggles, or the visible part of selected tracks, into spectra or wavelet levels.");
        });
}

/// Points the plot camera at the rectangle the panels did not claim, and tells
/// the plot how big that rectangle is.
fn hand_viewport_to_plot(
    window: &Window,
    camera: &mut Camera,
    plot: &mut Plot,
    left: f32,
    right: f32,
    top: f32,
) {
    let logical_size = Vec2::new(
        (window.width() - left - right).max(0.0),
        (window.height() - top).max(0.0),
    );

    plot.origin = Vec2::new(left, top);
    plot.view.width = logical_size.x as f64;
    plot.view.height = logical_size.y as f64;

    let scale = window.scale_factor();
    let position = (Vec2::new(left, top) * scale).as_uvec2();
    let size = (logical_size * scale).as_uvec2();

    // A zero-sized viewport is invalid; drop it until there is room again.
    camera.viewport = (size.x > 0 && size.y > 0)
        .then(|| Viewport { physical_position: position, physical_size: size, ..default() });
}

fn swatch(ui: &mut egui::Ui, color: Rgba) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    let to_byte = |channel: f32| (channel.clamp(0.0, 1.0) * 255.0).round() as u8;
    ui.painter().rect_filled(
        rect,
        egui::CornerRadius::same(2),
        egui::Color32::from_rgb(to_byte(color.red), to_byte(color.green), to_byte(color.blue)),
    );
}

/// Stacks a label's characters so it reads down a narrow strip, standing in
/// for the rotated handle of the Java sidebar.
fn vertical(text: &str) -> String {
    text.chars().map(String::from).collect::<Vec<_>>().join("\n")
}
