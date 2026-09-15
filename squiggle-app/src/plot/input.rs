//! Mouse handling: pan, zoom and pick.
//!
//! The Java control installed and removed JavaFX event handlers as its
//! `controlsEnabled` property changed. Systems run unconditionally instead and
//! check the flag, which keeps the wiring in one place.

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use squiggle_core::Coord;

use super::{Plot, SquiggleModel};

/// Zoom applied per scroll notch. The original stepped the scale by a flat
/// 0.05, which crawls when zoomed in and jumps when zoomed out; a constant
/// ratio feels the same at every magnification.
const ZOOM_PER_NOTCH: f64 = 1.1;

/// Scroll reported in pixels, as trackpads do, divided into notches.
const PIXELS_PER_NOTCH: f64 = 50.0;

/// How far the pointer may travel before a click counts as a drag. This is the
/// role JavaFX's `isStillSincePress` played.
const DRAG_SLOP: f64 = 3.0;

/// How close a click must land to count as hitting a squiggle, in pixels.
const PICK_TOLERANCE: f64 = 5.0;

/// Where the current press started and whether it has become a drag.
#[derive(Resource, Default)]
pub struct DragState {
    start: Option<Coord>,
    previous: Option<Coord>,
    travelled: f64,
}

impl DragState {
    fn begin(&mut self, at: Coord) {
        self.start = Some(at);
        self.previous = Some(at);
        self.travelled = 0.0;
    }

    fn end(&mut self) -> Option<Coord> {
        self.previous = None;
        self.travelled = 0.0;
        self.start.take()
    }

    fn is_drag(&self) -> bool {
        self.travelled > DRAG_SLOP
    }
}

/// Records the pointer in plot-local pixels, or `None` when it is elsewhere.
pub fn track_cursor(window: Option<Single<&Window, With<PrimaryWindow>>>, mut plot: ResMut<Plot>) {
    let Some(window) = window else {
        plot.cursor = None;
        return;
    };

    plot.scale_factor = window.scale_factor();

    let Some(position) = window.cursor_position() else {
        plot.cursor = None;
        return;
    };

    let local = Coord::new(
        position.x as f64 - plot.origin.x as f64,
        position.y as f64 - plot.origin.y as f64,
    );
    let inside = local.x >= 0.0
        && local.x <= plot.view.width
        && local.y >= 0.0
        && local.y <= plot.view.height;

    plot.cursor = inside.then_some(local);
}

/// Drags the plot under the cursor.
pub fn pan(
    mut plot: ResMut<Plot>,
    mut drag: ResMut<DragState>,
    buttons: Res<ButtonInput<MouseButton>>,
) {
    if !plot.controls_enabled {
        drag.end();
        return;
    }

    if buttons.just_pressed(MouseButton::Left)
        && let Some(cursor) = plot.cursor
    {
        drag.begin(cursor);
        return;
    }

    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    let (Some(cursor), Some(previous)) = (plot.cursor, drag.previous) else {
        return;
    };

    let delta_x = cursor.x - previous.x;
    // Screen y grows downwards; the view's translation grows upwards.
    let delta_y = previous.y - cursor.y;
    drag.travelled += delta_x.hypot(delta_y);
    drag.previous = Some(cursor);

    if drag.is_drag() {
        plot.view.pan(delta_x, delta_y);
    }
}

/// Zooms about the cursor. Alt holds the x scale still and Control holds y,
/// matching the original's modifiers.
pub fn zoom(
    mut plot: ResMut<Plot>,
    mut scrolls: MessageReader<MouseWheel>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let notches: f64 = scrolls
        .read()
        .map(|scroll| match scroll.unit {
            MouseScrollUnit::Line => scroll.y as f64,
            MouseScrollUnit::Pixel => scroll.y as f64 / PIXELS_PER_NOTCH,
        })
        .sum();

    if notches == 0.0 || !plot.controls_enabled {
        return;
    }

    let Some(cursor) = plot.cursor else {
        return;
    };

    let factor = ZOOM_PER_NOTCH.powf(notches);
    let hold_x = keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);
    let hold_y = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);

    plot.view.zoom_about(
        cursor,
        if hold_x { 1.0 } else { factor },
        if hold_y { 1.0 } else { factor },
    );
}

/// Selects the squiggle under a click that did not turn into a drag.
pub fn pick(
    plot: Res<Plot>,
    mut drag: ResMut<DragState>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut squiggles: Query<(Entity, &mut SquiggleModel)>,
) {
    if !buttons.just_released(MouseButton::Left) {
        return;
    }

    let was_drag = drag.is_drag();
    let Some(start) = drag.end() else {
        return;
    };

    if was_drag || !plot.controls_enabled || !plot.plot_area_contains(start) {
        return;
    }

    // Nearest wins, so a click between two squiggles takes the closer one.
    let mut nearest: Option<(Entity, f64)> = None;
    for (entity, model) in &squiggles {
        if let Some(distance) = model.screen_distance_within(&plot.view, start, PICK_TOLERANCE)
            && nearest.is_none_or(|(_, best)| distance < best)
        {
            nearest = Some((entity, distance));
        }
    }

    if let Some((entity, _)) = nearest
        && let Ok((_, mut model)) = squiggles.get_mut(entity)
    {
        model.toggle_selected();
    }
}
