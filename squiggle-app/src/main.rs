//! Squiggle: an interactive waveform plot.
//!
//! A port of the JavaFX application of the same name. The plot model and its
//! coordinate math live in `squiggle-core`; this crate renders them with Bevy
//! and wraps them in an egui shell.

mod demo;
mod plot;
mod ui;

use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_egui::EguiPlugin;

fn main() -> AppExit {
    let options = match demo::DemoOptions::from_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            return AppExit::error();
        }
    };

    App::new()
        // The Java plot painted onto a black canvas.
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Squiggle".to_string(),
                resolution: WindowResolution::new(1152, 648),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(plot::PlotPlugin)
        .add_plugins(ui::UiPlugin)
        .insert_resource(options)
        .add_systems(Startup, demo::seed)
        .run()
}
