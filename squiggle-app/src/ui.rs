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
use crate::models_ui::{self, ModelRegistry, ModelWindows};
use crate::live_ui::{self, LiveState};
use crate::sync_ui::{self, SyncState};
use crate::scatter_ui::{self, ScatterWindows};
use crate::templates_ui::{self, TemplateWindows};
use crate::process_ui::{self, ProcessWindows};
use crate::project_ui::{self, ProjectAccess, ProjectWindows};
use crate::workspace::{SampleState, SyntheticEvent, WorkspaceRequest};
use crate::showcase::{LoadScene, Scene, Showcase};
use crate::layout::Layout;
use crate::status::{self, Readout, Status};
use crate::theme;
use crate::tracks::{self, RowOrder, Track, TrackView};

const HANDLE_WIDTH: f32 = 26.0;
const SIDEBAR_WIDTH: f32 = 300.0;
const SHOWCASE_WIDTH: f32 = 340.0;
const UNDO: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT), egui::Key::Z);

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Chrome>()
            .init_resource::<Status>()
            .add_systems(Update, status::track_drags)
            .init_resource::<ProjectWindows>()
            .init_resource::<ProcessWindows>()
            .init_resource::<LocateWindows>()
            .init_resource::<EventWindows>()
            .init_resource::<ClassifyWindows>()
            .init_resource::<ModelWindows>()
            .init_resource::<ModelRegistry>()
            .init_resource::<TemplateWindows>()
            .init_resource::<ScatterWindows>()
            .init_resource::<LiveState>()
            .init_resource::<SyncState>()
            .add_systems(Startup, sync_ui::startup)
            .add_systems(Update, sync_ui::poll_sync)
            .add_systems(Startup, live_ui::startup)
            .add_systems(Update, live_ui::poll_live)
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
                    models_ui::windows,
                    templates_ui::windows,
                    scatter_ui::windows,
                    live_ui::window,
                    sync_ui::window,
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
    models: ResMut<'w, ModelWindows>,
    templates: ResMut<'w, TemplateWindows>,
    scatter: ResMut<'w, ScatterWindows>,
    live: ResMut<'w, LiveState>,
    sync: ResMut<'w, SyncState>,
}

/// Which parts of the chrome are open.
#[derive(Resource, Default)]
pub struct Chrome {
    pub sidebar_open: bool,
    pub about_open: bool,
    pub shortcuts_open: bool,
    pub transform_settings_open: bool,
    pub transforms: TransformSettings,
    /// Why the last transform failed, until posted as a notice.
    pub transform_error: Option<String>,
    pub demos_open: bool,
    /// The style is installed and the saved layout restored.
    started: bool,
}

/// The parameters behind Process ▸ Transform.
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
// A Bevy system declares everything it touches in its signature; thirteen is
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
    mut status: ResMut<Status>,
    mut layout: ResMut<Layout>,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    if !chrome.started {
        if let Some(memory) = layout.take_memory() {
            ctx.memory_mut(|m| {
                let options = std::mem::take(&mut m.options);
                *m = memory;
                m.options = options;
            });
        }
        theme::install(&ctx);
        chrome.started = true;
    }

    // Undo, redo, the File menu and the shortcuts sheet, unless a text field has the keyboard.
    if !ctx.egui_wants_keyboard_input() {
        if ctx.input_mut(|i| i.consume_shortcut(&REDO)) {
            project.workspace.redo();
        } else if ctx.input_mut(|i| i.consume_shortcut(&UNDO)) {
            project.workspace.undo();
        }
        project_ui::file_keys(&ctx, &mut project);
        if ctx.input(|i| i.key_pressed(egui::Key::Questionmark) || i.key_pressed(egui::Key::F1)) {
            chrome.shortcuts_open = !chrome.shortcuts_open;
        }
    }

    // Failures become notices; edits, undos and redos show in the status bar.
    if let Some(error) = project.workspace.error.take() {
        status.error("Something went wrong", error);
    }
    if let Some(error) = chrome.transform_error.take() {
        status.error("Transform failed", error);
    }
    let keys = (ctx.format_shortcut(&UNDO), ctx.format_shortcut(&REDO));
    status.observe_history(&project.workspace, (&keys.0, &keys.1), ctx.input(|i| i.time));

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
            let unsaved = project.workspace.is_unsaved();
            toolbar(ui, &mut plot, &squiggles, &options, &mut commands, &mut tracks, unsaved);
        })
        .response
        .rect
        .height();

    let readout = readout(&plot, &squiggles, &mut project, &mut tracks);
    status.observe_busy(&readout.busy, ctx.input(|i| i.time));
    let bottom = egui::Panel::bottom("squiggle_status")
        .frame(egui::Frame::side_top_panel(root.style()).inner_margin(egui::Margin::symmetric(8, 3)))
        .show(&mut root, |ui| status::bar(ui, &status, readout))
        .response
        .rect
        .height();

    let mut left = egui::Panel::left("squiggle_sidebar_handle")
        .resizable(false)
        .exact_size(HANDLE_WIDTH)
        .show_separator_line(false)
        .frame(egui::Frame::side_top_panel(root.style()).inner_margin(egui::Margin::symmetric(3, 6)))
        .show(&mut root, |ui| {
            // The Java sidebar stacked rotated toggle buttons down its edge.
            sidebar_handle(ui, &mut chrome.sidebar_open, "Squiggles")
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
    shortcuts_window(&ctx, &mut chrome);
    transform_window(&ctx, &mut chrome);
    demos_window(&ctx, &mut chrome, &mut project, &mut tracks, &mut commands);

    hand_viewport_to_plot(&window, &mut camera, &mut plot, left, right, top, bottom);
    let plot_rect = egui::Rect::from_min_size(plot.origin.to_array().into(), egui::vec2(plot.view.width as f32, plot.view.height as f32));
    theme::set_plot_rect(&ctx, plot_rect);

    if project.workspace.project.traces().is_empty() && squiggles.is_empty() && showcase.current.is_none() {
        empty_state(&ctx, plot_rect, &mut project, &mut chrome.demos_open);
    }
    status::notices(&ctx, &mut status, plot_rect);
    if status.dragging {
        status::drop_target(&ctx, plot_rect);
    }
    Ok(())
}

/// What the status bar reports this frame.
fn readout(
    plot: &Plot,
    squiggles: &Query<(Entity, &mut SquiggleModel)>,
    project: &mut ProjectAccess,
    tracks: &mut TrackAccess,
) -> Readout {
    let mut busy = Vec::new();
    tracks.process.busy(&mut busy);
    tracks.locate.busy(&mut busy);
    tracks.events.busy(&mut busy);
    tracks.classify.busy(&mut busy);
    tracks.scatter.busy(&mut busy);
    tracks.sync.busy(&mut busy);
    if project.workspace.samples.values().any(|s| matches!(s, SampleState::Loading)) {
        busy.push("Loading samples");
    }

    let plural = |n: usize, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
    let (cursor, selection) = match *tracks.mode {
        PlotMode::Tracks => {
            let total = tracks.tracks.iter().count();
            let selected = tracks.tracks.iter().filter(|t| t.selected).count();
            let selection = match (total, selected) {
                (0, _) => None,
                (_, 0) => Some(plural(total, "track")),
                _ => Some(format!("{selected} of {} selected", plural(total, "track"))),
            };
            (tracks.view.hover.clone(), selection)
        }
        PlotMode::Overlay => {
            let cursor = plot.cursor.filter(|c| plot.plot_area_contains(*c)).map(|c| {
                let at = plot.view.screen_to_cart(c);
                format!("x {}  y {}", short(at.x), short(at.y))
            });
            let total = squiggles.iter().count();
            let selected = squiggles.iter().filter(|(_, m)| m.selected).count();
            let selection = match (total, selected) {
                (0, _) => None,
                (_, 0) => Some(plural(total, "squiggle")),
                _ => Some(format!("{selected} of {} selected", plural(total, "squiggle"))),
            };
            (cursor, selection)
        }
    };

    Readout {
        busy,
        cursor,
        selection,
        live: tracks.live.summary(),
        sync: SyncState::summary(&project.workspace),
    }
}

/// A number short enough for the status bar.
fn short(value: f64) -> String {
    if value != 0.0 && !(1e-3..1e5).contains(&value.abs()) { format!("{value:.3e}") } else { format!("{value:.4}") }
}

/// What to do with an empty project, over the empty plot.
fn empty_state(ctx: &egui::Context, plot: egui::Rect, project: &mut ProjectAccess, demos_open: &mut bool) {
    egui::Area::new("squiggle_empty".into())
        .order(egui::Order::Background)
        .pivot(egui::Align2::CENTER_CENTER)
        .fixed_pos(plot.center())
        .show(ctx, |ui| {
            egui::Frame::window(ui.style()).inner_margin(20).show(ui, |ui| {
                ui.set_width(340.0);
                ui.vertical_centered(|ui| {
                    ui.heading("No traces yet");
                    ui.add_space(4.0);
                    ui.weak("Drop waveform files or a project on the window, or:");
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        let width = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                        let size = egui::vec2(width, 26.0);
                        if ui.add_sized(size, theme::primary("Import Waveforms…")).clicked() {
                            project.requests.write(WorkspaceRequest::ImportDialog);
                        }
                        if ui.add_sized(size, egui::Button::new("Open Project…")).clicked() {
                            project.requests.write(WorkspaceRequest::OpenDialog);
                        }
                    });
                    ui.add_space(6.0);
                    if ui.link("Or start from a demo…").clicked() {
                        *demos_open = true;
                    }
                });
            });
        });
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
            project_ui::file_menu(ui, project, &mut chrome.demos_open);
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
        ui.menu_button("Project", |ui| {
            project_ui::project_menu(ui, project);
            ui.separator();
            live_ui::live_menu(ui, &mut tracks.live);
            sync_ui::sync_menu(ui, &mut tracks.sync);
        });
        ui.menu_button("Process", |ui| {
            process_ui::process_menu(ui, &mut tracks.process, &mut tracks.view);
            ui.separator();
            locate_ui::locate_menu(ui, &mut tracks.locate);
            let locate = tracks.locate.pick_open || tracks.locate.locate_open;
            if tracks.process.processing_open || tracks.process.align_open || locate {
                *tracks.mode = PlotMode::Tracks;
            }
            ui.separator();
            ui.menu_button("Transform", |ui| {
                let sources = TransformSources { squiggles, tracks, width: plot.view.plot_width() };
                transform_menu(ui, chrome, sources, commands, &project.workspace.project);
            });
        });
        ui.menu_button("Events", |ui| {
            events_ui::events_menu(ui, &mut tracks.events, &mut tracks.view);
            ui.separator();
            classify_ui::classify_menu(ui, &mut tracks.classify);
            models_ui::models_menu(ui, &mut tracks.models);
            templates_ui::templates_menu(ui, &mut tracks.templates);
            scatter_ui::scatter_menu(ui, &mut tracks.scatter);
            if tracks.events.detect_open || tracks.events.catalogue_open || tracks.classify.features_open {
                *tracks.mode = PlotMode::Tracks;
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.add(egui::Button::new("Keyboard Shortcuts").shortcut_text("?")).clicked() {
                chrome.shortcuts_open = true;
                ui.close();
            }
            if ui.button("About").clicked() {
                chrome.about_open = true;
                ui.close();
            }
        });
    });
}

/// What a demo opens.
#[derive(Clone, Copy)]
enum Demo {
    Project(SyntheticEvent),
    LiveReplay,
    Scene(Scene),
}

/// The demo projects, each replacing the open project: name and what is in it.
fn demo_projects() -> [(&'static str, String, Demo); 7] {
    [
        ("Synthetic event (alignment)", "Seven sensors recording one event with known delays".into(), Demo::Project(SyntheticEvent::Alignment)),
        ("Synthetic located event", "EMP, seismic and infrasound sensors at known positions recording one explosion".into(), Demo::Project(SyntheticEvent::Located)),
        ("Synthetic swarm", "Two minutes on eight seismic stations with six events and glitches".into(), Demo::Project(SyntheticEvent::Swarm)),
        ("Synthetic earthquakes and explosions", "Seven minutes of five earthquakes and five explosions on seismic and infrasound sensors".into(), Demo::Project(SyntheticEvent::Mixed)),
        ("Synthetic training catalogue", "27 minutes of 40 varied earthquakes and explosions on the same sensors, to label, export and train on".into(), Demo::Project(SyntheticEvent::Catalogue)),
        ("Synthetic event families", "16 minutes of repeats from two quarries and two earthquake swarms, one weak repeat in each, for clustering, templates and the matched filter".into(), Demo::Project(SyntheticEvent::Families)),
        ("Live replay of event families", format!("The event families fed live at ×{} by a local replay: recorded, plotted and detected as they arrive", live_ui::DEMO_SPEED), Demo::LiveReplay),
    ]
}

fn open_demo(demo: Demo, project: &mut ProjectAccess, tracks: &mut TrackAccess, commands: &mut PlotCommands) {
    match demo {
        Demo::Project(event) => {
            project.requests.write(WorkspaceRequest::Synthetic(event));
        }
        Demo::LiveReplay => live_ui::request_demo(&mut tracks.live, &mut project.requests),
        Demo::Scene(scene) => {
            commands.load.write(LoadScene(scene));
        }
    }
}

/// File ▸ New from Demo…: every demo, with what is in it.
fn demos_window(ctx: &egui::Context, chrome: &mut Chrome, project: &mut ProjectAccess, tracks: &mut TrackAccess, commands: &mut PlotCommands) {
    if !chrome.demos_open {
        return;
    }
    let mut chosen = None;
    let mut entry = |ui: &mut egui::Ui, name: &'static str, about: &str, demo: Demo| {
        if ui.add(egui::Button::new(egui::RichText::new(name).strong()).min_size(egui::vec2(230.0, 0.0))).clicked() {
            chosen = Some(demo);
        }
        ui.add(egui::Label::new(egui::RichText::new(about).weak()).wrap());
        ui.end_row();
    };
    theme::dialog(ctx, "New from Demo").open(&mut chrome.demos_open).show(ctx, |ui| {
        ui.set_width(600.0);
        ui.strong("Projects");
        ui.weak("Each replaces the open project.");
        egui::Grid::new("demo_projects").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            for (name, about, demo) in demo_projects() {
                entry(ui, name, &about, demo);
            }
        });
        ui.separator();
        ui.strong("Transform scenes");
        ui.weak("A known signal on the plot, its transform and checks on the result.");
        egui::Grid::new("demo_scenes").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            for scene in Scene::ALL {
                entry(ui, scene.label(), scene.summary(), Demo::Scene(scene));
            }
        });
    });
    if let Some(demo) = chosen {
        open_demo(demo, project, tracks, commands);
        chrome.demos_open = false;
    }
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
    theme::tool_window(ctx, "Transform Settings")
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

/// A drop-down over a fixed set of values.
fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut T,
    options: &[T],
    label: impl Fn(T) -> &'static str,
) -> egui::Response {
    egui::ComboBox::from_id_salt(id)
        .selected_text(label(*value))
        .show_ui(ui, |ui| {
            for option in options {
                ui.selectable_value(value, *option, label(*option));
            }
        })
        .response
}

/// Toolbar controls that belong together, on one strip.
fn group(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(ui.visuals().extreme_bg_color)
        .corner_radius(5)
        .inner_margin(2)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.horizontal(add);
        });
}

fn toolbar(
    ui: &mut egui::Ui,
    plot: &mut Plot,
    squiggles: &Query<(Entity, &mut SquiggleModel)>,
    options: &DemoOptions,
    commands: &mut PlotCommands,
    tracks: &mut TrackAccess,
    unsaved: bool,
) {
    ui.horizontal(|ui| {
        group(ui, |ui| {
            let mode = &mut *tracks.mode;
            ui.selectable_value(mode, PlotMode::Tracks, "Tracks")
                .on_hover_text("Imported traces stacked on a shared time axis");
            ui.selectable_value(mode, PlotMode::Overlay, "Overlay")
                .on_hover_text("Free squiggles: demos and transform results");
        });

        // How the plot is framed and what it shows.
        group(ui, |ui| {
            let (any, hint) = match *tracks.mode {
                PlotMode::Overlay => (squiggles.iter().count() > 0, "Frame every squiggle"),
                PlotMode::Tracks => (!tracks.tracks.is_empty(), "Frame every track's time span (or double-click)"),
            };
            if ui.add_enabled(any, egui::Button::new("Fit").frame(false)).on_hover_text(hint).clicked() {
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
                choice(ui, "row_order", &mut tracks.view.order, &RowOrder::ALL, |o| o.label())
                    .on_hover_text("Order rows by");
                ui.toggle_value(&mut tracks.view.show_raw, "Raw").on_hover_text("Plot tracks without their processing");
            }
        });

        // What is drawn over the plot, and whether it answers the pointer.
        group(ui, |ui| {
            ui.toggle_value(&mut plot.axes_visible, "∟").on_hover_text("Axes");
            ui.toggle_value(&mut plot.crosshairs_visible, "✛").on_hover_text("Crosshairs");
            ui.toggle_value(&mut plot.controls_enabled, "✋")
                .on_hover_text("Controls: drag to pan, scroll to zoom, click to select");
        });

        if *tracks.mode == PlotMode::Overlay
            && ui.button("+ Squiggle").on_hover_text("Add a demo squiggle").clicked()
        {
            let index = squiggles.iter().count();
            commands.add.write(AddSquiggle(demo::generate(index, options.sample_count)));
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if unsaved {
                ui.label(egui::RichText::new(&plot.name).color(theme::WARN))
                    .on_hover_text("Never saved; File ▸ Save Project As… keeps it");
                ui.label(egui::RichText::new("●").color(theme::WARN));
            } else {
                ui.label(egui::RichText::new(&plot.name).weak());
            }
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
        ui.label("No squiggles yet. Import waveforms from the File menu or drop files on the window; “+ Squiggle” in Overlay makes a demo one.");
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
                    if theme::danger(ui, true, egui::Button::new("✕").small()).on_hover_text("Remove").clicked() {
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
            (format!("All {passed} checks pass"), theme::GOOD)
        } else {
            (format!("{passed} of {} checks pass", loaded.checks.len()), theme::BAD)
        };
        ui.label(egui::RichText::new(summary).strong().color(color));
        ui.add_space(4.0);

        for check in &loaded.checks {
            ui.horizontal_top(|ui| {
                let (mark, color) = if check.passed { ("✔", theme::GOOD) } else { ("✘", theme::BAD) };
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

    theme::dialog(ctx, "About Squiggle")
        .open(&mut chrome.about_open)
        .show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.label("Squiggle — a waveform plot.");
            ui.label("A Rust and Bevy port of the original JavaFX application.");
            ui.separator();
            ui.label("Process ▸ Filter and detrend preprocesses selected tracks; Process ▸ Align measures and applies delays.");
            ui.label("Process ▸ Pick finds onsets; Process ▸ Locate fits source position and origin time, and removes moveout.");
            ui.label("Events ▸ Detect finds and associates events; Events ▸ Catalogue reviews them.");
            ui.label("Events ▸ Classify runs features and the rules classifier; Features shows the reviewed event's evidence; Labels edits the taxonomy; Export dataset writes Parquet for training; Models imports packages from squiggle-train, whose predictions Classify then records.");
            ui.label("Edits to processing, offsets, picks, sensor locations, origins and events can be undone.");
            ui.label("Process ▸ Transform turns the selected squiggles, or the visible part of selected tracks, into spectra or wavelet levels.");
            ui.separator();
            ui.weak("Help ▸ Keyboard Shortcuts (?) lists every key and gesture.");
        });
}

/// Every key and gesture, by where it works.
fn shortcuts_window(ctx: &egui::Context, chrome: &mut Chrome) {
    if !chrome.shortcuts_open {
        return;
    }

    let mac = ctx.os() == egui::os::OperatingSystem::Mac;
    let (command, control, alt) = if mac { ("⌘", "⌃", "⌥") } else { ("Ctrl+", "Ctrl+", "Alt+") };
    let file = project_ui::file_shortcuts().map(|(shortcut, label, _)| (ctx.format_shortcut(&shortcut), label.trim_end_matches('…')));
    let sections: [(&str, Vec<(String, &str)>); 5] = [
        ("General", vec![
            (ctx.format_shortcut(&UNDO), "Undo"),
            (ctx.format_shortcut(&REDO), "Redo"),
            ("?".into(), "This sheet"),
            ("Esc".into(), "Close a menu"),
        ]),
        ("Tracks", vec![
            ("Click".into(), "Select a row"),
            ("Right-click, ⋯".into(), "Row display, y scale, reference, lock"),
            ("Double-click".into(), "Fit every track"),
            ("Scroll".into(), "Zoom time (over labels: scroll rows)"),
            ("Drag".into(), "Pan time and rows"),
            (format!("Pinch, {command}scroll"), "Scale amplitude"),
            ("⇧ drag".into(), "Shift a row in time, snapping"),
            (format!("⇧{alt}drag"), "Shift without snapping"),
            ("← →".into(), "Nudge selected rows a sample"),
            ("⇧← ⇧→".into(), "Nudge ten samples"),
            ("P / S".into(), "Pick a first or S phase at the cursor"),
            ("Delete".into(), "Remove the nearest pick"),
            (format!("{command}drag"), "Make an event"),
            ("Drag an edge".into(), "Resize the reviewed event"),
        ]),
        ("File", file.into()),
        ("Reviewing an event", vec![
            ("] / [".into(), "Next and previous event"),
            ("A / R".into(), "Accept, reject"),
            ("1–9".into(), "Label"),
            ("Right-click".into(), "Split the event"),
            ("Esc".into(), "Close the review"),
        ]),
        ("Overlay", vec![
            ("Drag".into(), "Pan"),
            ("Scroll".into(), "Zoom"),
            (format!("{alt}scroll"), "Zoom y only"),
            (format!("{control}scroll"), "Zoom x only"),
            ("Click".into(), "Select a squiggle"),
        ]),
    ];

    theme::dialog(ctx, "Keyboard Shortcuts")
        .open(&mut chrome.shortcuts_open)
        .show(ctx, |ui| {
            ui.horizontal_top(|ui| {
                for column in [&sections[..2], &sections[2..]] {
                    ui.vertical(|ui| {
                        for (title, rows) in column {
                            ui.strong(*title);
                            egui::Grid::new(title).num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                                for (keys, what) in rows {
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.label(egui::RichText::new(keys).monospace().background_color(ui.visuals().faint_bg_color));
                                    });
                                    ui.label(*what);
                                    ui.end_row();
                                }
                            });
                            ui.add_space(8.0);
                        }
                    });
                    ui.add_space(16.0);
                }
            });
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
    bottom: f32,
) {
    let logical_size = Vec2::new(
        (window.width() - left - right).max(0.0),
        (window.height() - top - bottom).max(0.0),
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
    ui.painter().rect_filled(rect, egui::CornerRadius::same(2), theme::color32(color));
}

/// A toggle whose label reads up a narrow strip, standing in for the rotated
/// handle of the Java sidebar.
fn sidebar_handle(ui: &mut egui::Ui, open: &mut bool, text: &str) -> egui::Response {
    const PADDING: f32 = 12.0;
    let galley = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(13.0), egui::Color32::PLACEHOLDER);
    let size = egui::vec2(ui.available_width(), galley.size().x + 2.0 * PADDING);
    let (rect, mut response) = ui.allocate_exact_size(size, egui::Sense::click());
    if response.clicked() {
        *open = !*open;
        response.mark_changed();
    }
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, *open, text));

    let visuals = ui.style().interact_selectable(&response, *open);
    let painter = ui.painter();
    painter.rect(rect, visuals.corner_radius, visuals.weak_bg_fill, visuals.bg_stroke, egui::StrokeKind::Inside);
    let corner = rect.center() - galley.size() / 2.0;
    painter.add(
        egui::epaint::TextShape::new(corner, galley, visuals.text_color())
            .with_angle_and_anchor(-std::f32::consts::FRAC_PI_2, egui::Align2::CENTER_CENTER),
    );
    response
}
