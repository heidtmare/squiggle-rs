//! Squiggle: an interactive waveform plot.
//!
//! A port of the JavaFX application of the same name. The plot model and its
//! coordinate math live in `squiggle-core`; this crate renders them with Bevy
//! and wraps them in an egui shell.

mod demo;
mod plot;
mod classify_ui;
mod live;
mod live_ui;
mod sync_ui;
mod theme;
mod models_ui;
mod scatter_ui;
mod templates_ui;
mod events_ui;
mod locate_ui;
mod process_ui;
mod project_ui;
mod showcase;
mod status;
mod strips;
mod synthetic;
mod tour;
mod tracks;
mod ui;
mod workspace;

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

    // Tours run headless: the window is never opened, only rendered offscreen
    // at its size, laid out at 720p and drawn at 1080p.
    let tour = options.tour.is_some();
    let resolution = if tour {
        WindowResolution::new(1920, 1080).with_scale_factor_override(1.5)
    } else {
        WindowResolution::new(1152, 648)
    };
    let mut plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: "Squiggle".to_string(),
                name: Some(squiggle_io::APP_ID.to_string()),
                resolution,
                ..default()
            }),
            // Closing asks first when the project is unsaved (workspace::handle_close).
            close_when_requested: false,
            ..default()
        })
        .build();
    if tour {
        plugins = plugins
            .disable::<bevy::winit::WinitPlugin>()
            .add(bevy::app::ScheduleRunnerPlugin::run_loop(std::time::Duration::ZERO));
    }

    App::new()
        // The Java plot painted onto a black canvas.
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(plugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(plot::PlotPlugin)
        .add_plugins(ui::UiPlugin)
        .add_plugins(workspace::WorkspacePlugin)
        .add_plugins(tracks::TracksPlugin)
        .add_plugins(showcase::ShowcasePlugin)
        .insert_resource(options)
        .add_plugins(tour::TourPlugin)
        .add_systems(Startup, demo::seed)
        .add_systems(Update, demo::screenshot)
        .run()
}
