//! Axes, crosshairs and selection markers, drawn with gizmos.
//!
//! These are cheap, change every frame and never number more than a few
//! hundred lines, so immediate-mode drawing suits them. The squiggles
//! themselves take the opposite route and live in meshes; see [`super::line`].

use bevy::math::{Isometry2d, Rot2};
use bevy::prelude::*;
use squiggle_core::{Axis, Coord, Orientation, PlotView, Rgba};

use super::{Plot, SquiggleModel};

const MAJOR_TICK_LENGTH: f64 = 6.0;
const MINOR_TICK_LENGTH: f64 = 3.0;
const LABEL_SIZE: f32 = 9.0;
/// Room a label needs per character, as a fraction of its size. The stroke
/// font is near enough monospaced for masking purposes.
const LABEL_ADVANCE: f32 = 0.62;
const POINT_RADIUS: f32 = 2.5;
/// Selected points closer together than this on screen are not worth drawing
/// individually; the line already shows where they are. The Java renderer made
/// the same trade by averaging points that landed on one screen column.
const POINT_SPACING: f64 = 6.0;

pub fn draw(plot: Res<Plot>, squiggles: Query<&SquiggleModel>, mut gizmos: Gizmos) {
    if !plot.is_usable() {
        return;
    }

    if plot.axes_visible {
        for axis in &plot.axes {
            draw_axis(&plot, axis, &mut gizmos);
        }
    }

    draw_selected_points(&plot, &squiggles, &mut gizmos);

    if plot.crosshairs_visible {
        draw_crosshairs(&plot, &mut gizmos);
    }
}

fn draw_axis(plot: &Plot, axis: &Axis, gizmos: &mut Gizmos) {
    let view = &plot.view;
    let origin = view.plot_origin();
    let color = to_color(Rgba::WHITE);

    let line = |gizmos: &mut Gizmos, from: Coord, to: Coord| {
        gizmos.line_2d(plot.to_world(from), plot.to_world(to), color);
    };

    match axis.orientation {
        Orientation::X => {
            // The axis line runs along the top of the bottom gutter.
            let axis_y = origin.y + view.plot_height();
            line(gizmos, Coord::new(origin.x, axis_y), Coord::new(origin.x + view.plot_width(), axis_y));

            for tick in axis.ticks(view) {
                let x = PlotView::snap(tick.position);
                let length =
                    if tick.is_major() { MAJOR_TICK_LENGTH } else { MINOR_TICK_LENGTH };
                line(gizmos, Coord::new(x, axis_y), Coord::new(x, axis_y + length));

                if let Some(label) = tick.label {
                    let at = Coord::new(x, view.height - view.insets.bottom / 2.0);
                    text(plot, gizmos, at, &label, Rot2::IDENTITY, color);
                }
            }
        }
        Orientation::Y => {
            // The axis line runs down the right edge of the left gutter.
            let axis_x = origin.x;
            line(gizmos, Coord::new(axis_x, origin.y), Coord::new(axis_x, origin.y + view.plot_height()));

            for tick in axis.ticks(view) {
                let y = PlotView::snap(tick.position);
                let length =
                    if tick.is_major() { MAJOR_TICK_LENGTH } else { MINOR_TICK_LENGTH };
                line(gizmos, Coord::new(axis_x, y), Coord::new(axis_x - length, y));

                if let Some(label) = tick.label {
                    let at = Coord::new(view.insets.left / 2.0, y);
                    // Reading downwards, as the Java renderer's rotation did.
                    text(plot, gizmos, at, &label, Rot2::degrees(-90.0), color);
                }
            }
        }
    }
}

/// Marks the individual samples of selected squiggles, as the Java renderer
/// did when a squiggle was picked.
fn draw_selected_points(plot: &Plot, squiggles: &Query<&SquiggleModel>, gizmos: &mut Gizmos) {
    let view = &plot.view;

    for model in squiggles.iter().filter(|model| model.selected) {
        let color = to_color(model.style.fill);
        let mut last: Option<Coord> = None;

        for point in model.points() {
            let screen = view.cart_to_screen(point.coord());
            if view.is_culled(screen) || !plot.plot_area_contains(screen) {
                continue;
            }
            // Skip samples that would land on top of the previous marker.
            if last.is_some_and(|prev| (screen.x - prev.x).hypot(screen.y - prev.y) < POINT_SPACING)
            {
                continue;
            }
            last = Some(screen);
            gizmos.circle_2d(plot.to_world(screen), POINT_RADIUS, color);
        }
    }
}

fn draw_crosshairs(plot: &Plot, gizmos: &mut Gizmos) {
    let view = &plot.view;
    let plot_origin = view.plot_origin();
    let cart_origin = view.cart_origin();
    let (left, right) = (plot_origin.x, plot_origin.x + view.plot_width());
    let (top, bottom) = (plot_origin.y, plot_origin.y + view.plot_height());

    let cross = |gizmos: &mut Gizmos, at: Coord, color: Color| {
        // Each arm is suppressed when it would stray over an axis gutter.
        if at.y >= top && at.y <= bottom {
            let y = PlotView::snap(at.y);
            gizmos.line_2d(
                plot.to_world(Coord::new(left, y)),
                plot.to_world(Coord::new(right, y)),
                color,
            );
        }
        if at.x >= left && at.x <= right {
            let x = PlotView::snap(at.x);
            gizmos.line_2d(
                plot.to_world(Coord::new(x, top)),
                plot.to_world(Coord::new(x, bottom)),
                color,
            );
        }
    };

    // Where the data's origin currently sits.
    cross(gizmos, cart_origin, to_color(Rgba::WHITE));

    let Some(cursor) = plot.cursor else {
        return;
    };

    let gold = to_color(Rgba::GOLD);
    cross(gizmos, cursor, gold);

    // Read out the cursor's position in data units, in the gutters.
    let position = view.screen_to_cart(cursor);
    let x_label = squiggle_core::axis::format_value(position.x);
    let y_label = squiggle_core::axis::format_value(position.y);

    let x_at = Coord::new(cursor.x, view.height - view.insets.bottom / 2.0);
    masked_text(plot, gizmos, x_at, &x_label, Rot2::IDENTITY, gold);

    let y_at = Coord::new(view.insets.left / 2.0, cursor.y);
    masked_text(plot, gizmos, y_at, &y_label, Rot2::degrees(-90.0), gold);
}

fn text(plot: &Plot, gizmos: &mut Gizmos, at: Coord, label: &str, rotation: Rot2, color: Color) {
    gizmos.text_2d(
        Isometry2d { translation: plot.to_world(at), rotation },
        label,
        LABEL_SIZE,
        Vec2::ZERO,
        color,
    );
}

/// Draws a label over an opaque patch, so the cursor readout stays legible
/// where it crosses an axis label. The Java renderer filled a black rectangle
/// behind the text for the same reason.
fn masked_text(
    plot: &Plot,
    gizmos: &mut Gizmos,
    at: Coord,
    label: &str,
    rotation: Rot2,
    color: Color,
) {
    let width = label.chars().count() as f32 * LABEL_SIZE * LABEL_ADVANCE;
    let centre = plot.to_world(at);
    let black = to_color(Rgba::BLACK);

    // Gizmos draw lines, not fills, so the patch is a short stack of them.
    let half_height = LABEL_SIZE / 2.0 + 1.0;
    // Half-pixel steps so the one-pixel lines overlap rather than stripe.
    const ROW_STEP: f32 = 0.5;
    let rows = (half_height * 2.0 / ROW_STEP).ceil() as i32;
    for row in 0..=rows {
        let offset = -half_height + row as f32 * ROW_STEP;
        let a = centre + rotation * Vec2::new(-width / 2.0, offset);
        let b = centre + rotation * Vec2::new(width / 2.0, offset);
        gizmos.line_2d(a, b, black);
    }

    text(plot, gizmos, at, label, rotation, color);
}

fn to_color(rgba: Rgba) -> Color {
    Color::srgba(rgba.red, rgba.green, rgba.blue, rgba.alpha)
}
