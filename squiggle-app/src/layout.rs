//! The layout kept between sessions: the plot mode, the toolbar's toggles,
//! the sidebar, and egui's memory of where windows sat and how wide panels
//! were. Tours and screenshots neither read nor write it, so they always
//! start the same way.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use serde::{Deserialize, Serialize};

use crate::demo::DemoOptions;
use crate::plot::{Plot, PlotMode};
use crate::ui::Chrome;

pub struct LayoutPlugin;

impl Plugin for LayoutPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Layout>()
            .add_systems(PreStartup, restore)
            .add_systems(Last, save_on_exit);
    }
}

/// Where the layout is kept, and egui's memory until the chrome first runs.
#[derive(Resource, Default)]
pub struct Layout {
    path: Option<PathBuf>,
    memory: Option<egui::Memory>,
}

impl Layout {
    /// egui's memory from last time, once: windows and panels as they were left.
    pub fn take_memory(&mut self) -> Option<egui::Memory> {
        self.memory.take()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    tracks: bool,
    sidebar_open: bool,
    axes: bool,
    crosshairs: bool,
    controls: bool,
    egui: Option<egui::Memory>,
}

impl Default for Saved {
    fn default() -> Self {
        let plot = Plot::default();
        Self {
            tracks: false,
            sidebar_open: false,
            axes: plot.axes_visible,
            crosshairs: plot.crosshairs_visible,
            controls: plot.controls_enabled,
            egui: None,
        }
    }
}

/// The per-user settings folder: Application Support on macOS, the roaming
/// AppData on Windows, the XDG config folder elsewhere.
fn path() -> Option<PathBuf> {
    let var = |name| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(target_os = "macos") {
        var("HOME")?.join("Library/Application Support")
    } else if cfg!(windows) {
        var("APPDATA")?
    } else {
        var("XDG_CONFIG_HOME").or_else(|| var("HOME").map(|home| home.join(".config")))?
    };
    Some(base.join(squiggle_io::APP_ID).join("layout.ron"))
}

/// Before anything else starts, so command-line options still override it.
fn restore(
    options: Res<DemoOptions>,
    mut layout: ResMut<Layout>,
    mut mode: ResMut<PlotMode>,
    mut plot: ResMut<Plot>,
    mut chrome: ResMut<Chrome>,
) {
    if options.tour.is_some() || options.screenshot.is_some() {
        return;
    }
    layout.path = path();
    let Some(text) = layout.path.as_ref().and_then(|path| std::fs::read_to_string(path).ok()) else {
        return;
    };
    let saved: Saved = match ron::from_str(&text) {
        Ok(saved) => saved,
        Err(error) => {
            warn!("ignoring the saved layout: {error}");
            return;
        }
    };
    *mode = if saved.tracks { PlotMode::Tracks } else { PlotMode::Overlay };
    chrome.sidebar_open = saved.sidebar_open;
    plot.axes_visible = saved.axes;
    plot.crosshairs_visible = saved.crosshairs;
    plot.controls_enabled = saved.controls;
    layout.memory = saved.egui;
}

fn save_on_exit(
    mut exits: MessageReader<AppExit>,
    layout: Res<Layout>,
    mode: Res<PlotMode>,
    plot: Res<Plot>,
    chrome: Res<Chrome>,
    mut contexts: EguiContexts,
) {
    if exits.read().count() == 0 {
        return;
    }
    let Some(path) = &layout.path else {
        return;
    };
    let saved = Saved {
        tracks: *mode == PlotMode::Tracks,
        sidebar_open: chrome.sidebar_open,
        axes: plot.axes_visible,
        crosshairs: plot.crosshairs_visible,
        controls: plot.controls_enabled,
        egui: contexts.ctx_mut().ok().map(|ctx| ctx.memory(|m| m.clone())),
    };
    let written = ron::to_string(&saved)
        .map_err(|e| e.to_string())
        .and_then(|json| {
            path.parent().map_or(Ok(()), std::fs::create_dir_all).map_err(|e| e.to_string())?;
            std::fs::write(path, json).map_err(|e| e.to_string())
        });
    if let Err(error) = written {
        warn!("could not save the layout to {}: {error}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_positions_survive_a_round_trip() {
        let ctx = egui::Context::default();
        let moved = egui::pos2(321.0, 123.0);
        let frame = |ctx: &egui::Context, run: &mut dyn FnMut(&egui::Context)| {
            ctx.run_ui(egui::RawInput::default(), |ui| run(ui.ctx())).textures_delta.clear();
        };
        frame(&ctx, &mut |ctx| {
            egui::Window::new("Moved").current_pos(moved).show(ctx, |ui| ui.label("here"));
        });
        let saved = Saved { egui: Some(ctx.memory(|m| m.clone())), ..Saved::default() };
        let text = ron::to_string(&saved).unwrap();
        let restored: Saved = ron::from_str(&text).unwrap();

        let ctx = egui::Context::default();
        ctx.memory_mut(|m| *m = restored.egui.unwrap());
        let mut at = None;
        frame(&ctx, &mut |ctx| {
            at = egui::Window::new("Moved").show(ctx, |ui| ui.label("here")).map(|r| r.response.rect.min);
        });
        assert_eq!(at, Some(moved));
    }
}
