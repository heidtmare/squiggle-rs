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
use squiggle_core::Rgba;

use crate::demo::{self, DemoOptions};
use crate::plot::{AddSquiggle, FitToData, Plot, PlotCamera, RemoveSquiggle, SquiggleModel};

const HANDLE_WIDTH: f32 = 26.0;
const SIDEBAR_WIDTH: f32 = 300.0;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Chrome>()
            .add_systems(Startup, setup)
            .add_systems(EguiPrimaryContextPass, chrome);
    }
}

/// What the chrome can ask the plot to do. Grouping the writers keeps them out
/// of the argument list of every function that needs one.
#[derive(SystemParam)]
pub struct PlotCommands<'w> {
    add: MessageWriter<'w, AddSquiggle>,
    remove: MessageWriter<'w, RemoveSquiggle>,
    fit: MessageWriter<'w, FitToData>,
    exit: MessageWriter<'w, AppExit>,
}

/// Which parts of the chrome are open.
#[derive(Resource, Default)]
pub struct Chrome {
    pub sidebar_open: bool,
    pub about_open: bool,
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
// A Bevy system declares everything it touches in its signature; eight is what
// drawing the whole chrome needs.
#[allow(clippy::too_many_arguments)]
fn chrome(
    mut contexts: EguiContexts,
    mut chrome: ResMut<Chrome>,
    mut plot: ResMut<Plot>,
    mut camera: Single<&mut Camera, (With<PlotCamera>, Without<EguiContext>)>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut squiggles: Query<(Entity, &mut SquiggleModel)>,
    options: Res<DemoOptions>,
    mut commands: PlotCommands,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    ctx.set_visuals(egui::Visuals::dark());

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
            menu_bar(ui, &mut chrome, &mut squiggles, &mut commands);
            ui.separator();
            toolbar(ui, &mut plot, &squiggles, &options, &mut commands);
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

    about_window(&ctx, &mut chrome);

    hand_viewport_to_plot(&window, &mut camera, &mut plot, left, top);
    Ok(())
}

fn menu_bar(
    ui: &mut egui::Ui,
    chrome: &mut Chrome,
    squiggles: &mut Query<(Entity, &mut SquiggleModel)>,
    commands: &mut PlotCommands,
) {
    egui::MenuBar::new().ui(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.button("Close").clicked() {
                commands.exit.write(AppExit::Success);
                ui.close();
            }
        });
        ui.menu_button("Edit", |ui| {
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
        ui.menu_button("Help", |ui| {
            if ui.button("About").clicked() {
                chrome.about_open = true;
                ui.close();
            }
        });
    });
}

fn toolbar(
    ui: &mut egui::Ui,
    plot: &mut Plot,
    squiggles: &Query<(Entity, &mut SquiggleModel)>,
    options: &DemoOptions,
    commands: &mut PlotCommands,
) {
    ui.horizontal(|ui| {
        if ui.button("Add Squiggle").clicked() {
            let index = squiggles.iter().count();
            commands.add.write(AddSquiggle(demo::generate(index, options.sample_count)));
        }
        if ui
            .add_enabled(squiggles.iter().count() > 0, egui::Button::new("Fit"))
            .on_hover_text("Frame every squiggle")
            .clicked()
        {
            commands.fit.write(FitToData);
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
        ui.label("No squiggles yet. Use “Add Squiggle” to make one.");
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
            ui.label("Drag to pan. Scroll to zoom — hold Alt to keep the x scale, Control to keep the y scale.");
            ui.label("Click a squiggle to select it and show its samples.");
        });
}

/// Points the plot camera at the rectangle the panels did not claim, and tells
/// the plot how big that rectangle is.
fn hand_viewport_to_plot(
    window: &Window,
    camera: &mut Camera,
    plot: &mut Plot,
    left: f32,
    top: f32,
) {
    let logical_size = Vec2::new(
        (window.width() - left).max(0.0),
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
