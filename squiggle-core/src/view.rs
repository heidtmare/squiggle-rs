//! Mapping between cartesian data space and plot-screen pixels.
//!
//! This is the transform half of the Java `RenderingContext`. Screen space uses
//! the JavaFX convention: pixels, origin top-left, +y down. Cartesian space is
//! the data's own space with +y up.

use core::fmt;

/// Screen-space margin outside which geometry is considered off-plot.
pub const DEFAULT_CULLING_MARGIN: f64 = 100.0;

/// Smallest permitted axis scale; stops a zoom-out from collapsing the plot.
pub const MIN_SCALE: f64 = 0.05;

/// Largest permitted axis scale; stops a zoom-in from losing float precision.
pub const MAX_SCALE: f64 = 1.0e6;

/// A point in either screen or cartesian space, depending on context.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl Coord {
    pub const ZERO: Self = Self::new(0.0, 0.0);

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.x, self.y)
    }
}

/// Screen-space padding reserved around the plot area, typically for axes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Insets {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Insets {
    pub const EMPTY: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    pub const fn new(top: f64, right: f64, bottom: f64, left: f64) -> Self {
        Self { top, right, bottom, left }
    }
}

/// The pan/zoom state of a plot plus the size of the surface it draws into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotView {
    /// Width of the drawing surface, in pixels.
    pub width: f64,
    /// Height of the drawing surface, in pixels.
    pub height: f64,
    /// Gutters reserved for the axes.
    pub insets: Insets,
    /// Pixels per cartesian unit, per axis. Zoom is per-axis by design: the
    /// original plots waveforms, where x and y rarely share a unit.
    pub scale: Coord,
    /// Pan offset in pixels. +x moves the data right, +y moves it up.
    pub translation: Coord,
    /// Margin beyond the surface within which geometry is still drawn.
    pub culling_margin: f64,
}

impl Default for PlotView {
    fn default() -> Self {
        Self {
            width: 0.0,
            height: 0.0,
            insets: Insets::EMPTY,
            scale: Coord::new(1.0, 1.0),
            translation: Coord::ZERO,
            culling_margin: DEFAULT_CULLING_MARGIN,
        }
    }
}

impl PlotView {
    /// Nudges a coordinate onto a pixel centre so that odd-width lines render
    /// crisply rather than straddling two pixels.
    pub fn snap(value: f64) -> f64 {
        (value as i64) as f64 + 0.5
    }

    pub fn cart_to_screen(&self, cart: Coord) -> Coord {
        Coord::new(self.cart_x_to_screen_x(cart.x), self.cart_y_to_screen_y(cart.y))
    }

    pub fn cart_x_to_screen_x(&self, cart_x: f64) -> f64 {
        self.insets.left + self.translation.x + self.scale.x * cart_x
    }

    pub fn cart_y_to_screen_y(&self, cart_y: f64) -> f64 {
        self.height - self.insets.bottom - self.translation.y - self.scale.y * cart_y
    }

    pub fn screen_to_cart(&self, screen: Coord) -> Coord {
        Coord::new(self.screen_x_to_cart_x(screen.x), self.screen_y_to_cart_y(screen.y))
    }

    pub fn screen_x_to_cart_x(&self, screen_x: f64) -> f64 {
        (screen_x - self.translation.x - self.insets.left) / self.scale.x
    }

    pub fn screen_y_to_cart_y(&self, screen_y: f64) -> f64 {
        (screen_y - self.height + self.insets.bottom + self.translation.y) / -self.scale.y
    }

    /// Whether a screen-space point lies far enough outside the surface to skip.
    pub fn is_culled(&self, screen: Coord) -> bool {
        screen.x < -self.culling_margin
            || screen.x > self.width + self.culling_margin
            || screen.y < -self.culling_margin
            || screen.y > self.height + self.culling_margin
    }

    /// Top-left corner of the plot area, inside the axis gutters.
    pub fn plot_origin(&self) -> Coord {
        Coord::new(self.insets.left, self.insets.top)
    }

    pub fn plot_width(&self) -> f64 {
        self.width - self.insets.left - self.insets.right
    }

    pub fn plot_height(&self) -> f64 {
        self.height - self.insets.top - self.insets.bottom
    }

    /// Where cartesian (0, 0) currently sits on screen.
    pub fn cart_origin(&self) -> Coord {
        self.cart_to_screen(Coord::ZERO)
    }

    /// Pans by a screen-space delta. `delta_y` is positive upwards, matching
    /// the cartesian sense of [`PlotView::translation`].
    pub fn pan(&mut self, delta_x: f64, delta_y: f64) {
        self.translation.x += delta_x;
        self.translation.y += delta_y;
    }

    /// Multiplies each axis scale by its factor, keeping the result in range.
    /// Pass `1.0` to leave an axis untouched.
    pub fn zoom(&mut self, factor_x: f64, factor_y: f64) {
        self.scale.x = (self.scale.x * factor_x).clamp(MIN_SCALE, MAX_SCALE);
        self.scale.y = (self.scale.y * factor_y).clamp(MIN_SCALE, MAX_SCALE);
    }

    /// Zooms about a fixed screen-space point, so whatever is under the cursor
    /// stays under the cursor.
    pub fn zoom_about(&mut self, anchor: Coord, factor_x: f64, factor_y: f64) {
        let target = self.screen_to_cart(anchor);
        self.zoom(factor_x, factor_y);
        // Solve each axis mapping for the translation that puts `target` back
        // under `anchor` at the new scale.
        self.translation.x = anchor.x - self.insets.left - self.scale.x * target.x;
        self.translation.y = self.height - self.insets.bottom - anchor.y - self.scale.y * target.y;
    }

    /// Pans and scales so that `bounds` fills the plot area with a margin.
    pub fn fit(&mut self, bounds: crate::Bounds, margin_fraction: f64) {
        let (plot_w, plot_h) = (self.plot_width(), self.plot_height());
        if plot_w <= 0.0 || plot_h <= 0.0 {
            return;
        }

        let pad = 1.0 + margin_fraction.max(0.0) * 2.0;
        let span_x = (bounds.max_x - bounds.min_x).abs().max(f64::EPSILON) * pad;
        let span_y = (bounds.max_y - bounds.min_y).abs().max(f64::EPSILON) * pad;

        self.scale.x = (plot_w / span_x).clamp(MIN_SCALE, MAX_SCALE);
        self.scale.y = (plot_h / span_y).clamp(MIN_SCALE, MAX_SCALE);

        // Centre the bounds inside the plot area.
        let centre_x = (bounds.min_x + bounds.max_x) / 2.0;
        let centre_y = (bounds.min_y + bounds.max_y) / 2.0;
        self.translation.x = plot_w / 2.0 - self.scale.x * centre_x;
        self.translation.y = plot_h / 2.0 - self.scale.y * centre_y;
    }
}

/// Shortest distance from `point` to the segment `start`..`end`.
///
/// Used for picking: the Java version rendered the scene a second time in
/// unique colours and read back the pixel under the cursor, which cost a full
/// extra draw and a GPU readback per click. Measuring the distance directly is
/// both cheaper and exact.
pub fn distance_to_segment(point: Coord, start: Coord, end: Coord) -> f64 {
    let (dx, dy) = (end.x - start.x, end.y - start.y);
    let length_squared = dx * dx + dy * dy;

    // A zero-length segment is just a point.
    if length_squared <= f64::EPSILON {
        return (point.x - start.x).hypot(point.y - start.y);
    }

    let t = (((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared).clamp(0.0, 1.0);
    (point.x - (start.x + t * dx)).hypot(point.y - (start.y + t * dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> PlotView {
        PlotView {
            width: 200.0,
            height: 100.0,
            insets: Insets::new(0.0, 0.0, 20.0, 20.0),
            ..PlotView::default()
        }
    }

    #[test]
    fn cartesian_origin_sits_at_the_plot_corner() {
        let v = view();
        // x = left inset, y = height - bottom inset.
        assert_eq!(v.cart_origin(), Coord::new(20.0, 80.0));
    }

    #[test]
    fn y_grows_upwards_on_screen() {
        let v = view();
        assert!(v.cart_y_to_screen_y(10.0) < v.cart_y_to_screen_y(0.0));
    }

    #[test]
    fn screen_and_cartesian_round_trip() {
        let mut v = view();
        v.scale = Coord::new(2.5, 0.75);
        v.translation = Coord::new(-30.0, 12.0);

        let cart = Coord::new(13.25, -7.5);
        let back = v.screen_to_cart(v.cart_to_screen(cart));
        assert!((back.x - cart.x).abs() < 1e-9, "{back:?}");
        assert!((back.y - cart.y).abs() < 1e-9, "{back:?}");
    }

    #[test]
    fn culling_respects_the_margin() {
        let v = PlotView { culling_margin: 10.0, ..view() };
        assert!(!v.is_culled(Coord::new(-9.0, 50.0)));
        assert!(v.is_culled(Coord::new(-11.0, 50.0)));
        assert!(v.is_culled(Coord::new(100.0, 111.0)));
    }

    #[test]
    fn plot_area_excludes_the_axis_gutters() {
        let v = view();
        assert_eq!(v.plot_width(), 180.0);
        assert_eq!(v.plot_height(), 80.0);
        assert_eq!(v.plot_origin(), Coord::new(20.0, 0.0));
    }

    #[test]
    fn zoom_is_clamped_to_the_usable_range() {
        let mut v = view();
        v.zoom(1e-9, 1e12);
        assert_eq!(v.scale.x, MIN_SCALE);
        assert_eq!(v.scale.y, MAX_SCALE);
    }

    #[test]
    fn zoom_about_keeps_the_anchor_pinned() {
        let mut v = view();
        let anchor = Coord::new(140.0, 30.0);
        let before = v.screen_to_cart(anchor);
        v.zoom_about(anchor, 2.0, 3.0);
        let after = v.screen_to_cart(anchor);
        assert!((before.x - after.x).abs() < 1e-9, "{before:?} {after:?}");
        assert!((before.y - after.y).abs() < 1e-9, "{before:?} {after:?}");
    }

    #[test]
    fn snap_lands_on_pixel_centres() {
        assert_eq!(PlotView::snap(10.0), 10.5);
        assert_eq!(PlotView::snap(10.9), 10.5);
    }
}
