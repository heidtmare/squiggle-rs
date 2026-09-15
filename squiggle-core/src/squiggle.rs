//! The plot's data model: points, the polylines they form, and how to draw them.

use crate::color::Rgba;
use crate::view::{Coord, PlotView};

/// A single sample. `z` and `m` are optional measures carried alongside the
/// plotted `x`/`y`; the Java original signalled their absence with `NaN`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
    pub m: Option<f64>,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y, z: None, m: None }
    }

    pub const fn with_z(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z: Some(z), m: None }
    }

    pub const fn with_z_m(x: f64, y: f64, z: f64, m: f64) -> Self {
        Self { x, y, z: Some(z), m: Some(m) }
    }

    pub const fn coord(self) -> Coord {
        Coord::new(self.x, self.y)
    }
}

/// The cartesian extent of a set of points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    /// Extent of `points`, or `None` when there is nothing to measure.
    /// Non-finite samples are ignored rather than poisoning the result.
    pub fn of(points: impl IntoIterator<Item = Point>) -> Option<Self> {
        let mut bounds: Option<Self> = None;
        for point in points {
            if !point.x.is_finite() || !point.y.is_finite() {
                continue;
            }
            bounds = Some(match bounds {
                None => Self { min_x: point.x, min_y: point.y, max_x: point.x, max_y: point.y },
                Some(b) => Self {
                    min_x: b.min_x.min(point.x),
                    min_y: b.min_y.min(point.y),
                    max_x: b.max_x.max(point.x),
                    max_y: b.max_y.max(point.y),
                },
            });
        }
        bounds
    }

    /// The smallest extent containing both.
    pub fn union(self, other: Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }
}

/// How a squiggle is painted. Ported from `RenderingAttributes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    pub stroke: Rgba,
    pub fill: Rgba,
    pub line_width: f32,
}

impl Style {
    /// The `SimpleSquiggle` default: a two-pixel white line.
    pub const SQUIGGLE: Self = Self { stroke: Rgba::WHITE, fill: Rgba::WHITE, line_width: 2.0 };

    pub const fn new(color: Rgba, line_width: f32) -> Self {
        Self { stroke: color, fill: color, line_width }
    }
}

impl Default for Style {
    fn default() -> Self {
        Self { stroke: Rgba::WHITE, fill: Rgba::WHITE, line_width: 1.0 }
    }
}

/// A named polyline through a series of [`Point`]s.
///
/// Bounds are cached because the plot asks for them every frame, while points
/// change only when the caller edits them.
#[derive(Clone, Debug)]
pub struct Squiggle {
    pub name: String,
    pub style: Style,
    pub selected: bool,
    points: Vec<Point>,
    bounds: Option<Bounds>,
}

impl Squiggle {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            style: Style::SQUIGGLE,
            selected: false,
            points: Vec::new(),
            bounds: None,
        }
    }

    pub fn with_points(name: impl Into<String>, points: impl IntoIterator<Item = Point>) -> Self {
        let mut squiggle = Self::new(name);
        squiggle.extend(points);
        squiggle
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn points(&self) -> &[Point] {
        &self.points
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Cartesian extent, or `None` while the squiggle has no finite points.
    pub fn bounds(&self) -> Option<Bounds> {
        self.bounds
    }

    pub fn extend(&mut self, points: impl IntoIterator<Item = Point>) {
        let before = self.points.len();
        self.points.extend(points);
        // Growing only ever widens the extent, so merge instead of rescanning.
        if let Some(added) = Bounds::of(self.points[before..].iter().copied()) {
            self.bounds = Some(match self.bounds {
                Some(existing) => existing.union(added),
                None => added,
            });
        }
    }

    pub fn push(&mut self, point: Point) {
        self.extend([point]);
    }

    /// Keeps only the points matching `predicate`, then recomputes the extent.
    pub fn retain(&mut self, predicate: impl FnMut(&Point) -> bool) {
        self.points.retain(predicate);
        self.bounds = Bounds::of(self.points.iter().copied());
    }

    pub fn clear(&mut self) {
        self.points.clear();
        self.bounds = None;
    }

    /// Flips selection, as the Java `pick` handler did.
    pub fn toggle_selected(&mut self) {
        self.selected = !self.selected;
    }

    /// The squiggle's screen-space extent under `view`, if it has any points.
    pub fn screen_bounds(&self, view: &PlotView) -> Option<Bounds> {
        let bounds = self.bounds()?;
        let a = view.cart_to_screen(Coord::new(bounds.min_x, bounds.min_y));
        let b = view.cart_to_screen(Coord::new(bounds.max_x, bounds.max_y));
        Some(Bounds {
            min_x: a.x.min(b.x),
            min_y: a.y.min(b.y),
            max_x: a.x.max(b.x),
            max_y: a.y.max(b.y),
        })
    }

    /// Distance in screen pixels from `screen` to the nearest point on this
    /// squiggle, or `None` if nothing lies within `tolerance`.
    ///
    /// Segments entirely off-screen are skipped, which is what the Java
    /// renderer's culling margin did for drawing.
    pub fn screen_distance_within(
        &self,
        view: &PlotView,
        screen: Coord,
        tolerance: f64,
    ) -> Option<f64> {
        // Reject the whole squiggle before touching its points.
        let box_ = self.screen_bounds(view)?;
        if screen.x < box_.min_x - tolerance
            || screen.x > box_.max_x + tolerance
            || screen.y < box_.min_y - tolerance
            || screen.y > box_.max_y + tolerance
        {
            return None;
        }

        let mut nearest = f64::INFINITY;
        let mut previous: Option<Coord> = None;
        for point in &self.points {
            let current = view.cart_to_screen(point.coord());
            if let Some(start) = previous
                && !(view.is_culled(start) && view.is_culled(current))
            {
                nearest = nearest.min(crate::view::distance_to_segment(screen, start, current));
            }
            previous = Some(current);
        }

        (nearest <= tolerance).then_some(nearest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_squiggle_has_no_bounds() {
        assert_eq!(Squiggle::new("empty").bounds(), None);
    }

    #[test]
    fn bounds_track_appended_points() {
        let mut s = Squiggle::with_points("s", [Point::new(1.0, 5.0), Point::new(3.0, 2.0)]);
        assert_eq!(s.bounds(), Some(Bounds { min_x: 1.0, min_y: 2.0, max_x: 3.0, max_y: 5.0 }));

        s.push(Point::new(-4.0, 9.0));
        assert_eq!(s.bounds(), Some(Bounds { min_x: -4.0, min_y: 2.0, max_x: 3.0, max_y: 9.0 }));
    }

    #[test]
    fn bounds_shrink_back_after_removal() {
        let mut s = Squiggle::with_points("s", [Point::new(0.0, 0.0), Point::new(100.0, 100.0)]);
        s.retain(|p| p.x < 50.0);
        assert_eq!(s.bounds(), Some(Bounds { min_x: 0.0, min_y: 0.0, max_x: 0.0, max_y: 0.0 }));
    }

    #[test]
    fn non_finite_points_do_not_poison_bounds() {
        let s = Squiggle::with_points(
            "s",
            [Point::new(f64::NAN, 1.0), Point::new(2.0, 4.0), Point::new(6.0, f64::INFINITY)],
        );
        assert_eq!(s.bounds(), Some(Bounds { min_x: 2.0, min_y: 4.0, max_x: 2.0, max_y: 4.0 }));
    }

    #[test]
    fn optional_measures_stay_absent_unless_given() {
        assert_eq!(Point::new(1.0, 2.0).z, None);
        assert_eq!(Point::with_z(1.0, 2.0, 3.0).m, None);
        assert_eq!(Point::with_z_m(1.0, 2.0, 3.0, 4.0).m, Some(4.0));
    }

    fn pick_view() -> PlotView {
        PlotView { width: 200.0, height: 100.0, ..PlotView::default() }
    }

    #[test]
    fn a_click_on_the_line_picks_it() {
        let s = Squiggle::with_points("s", [Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
        let view = pick_view();
        // The line runs along cartesian y = 0, which is screen y = height.
        let on_the_line = Coord::new(50.0, 100.0);
        assert_eq!(s.screen_distance_within(&view, on_the_line, 5.0), Some(0.0));
    }

    #[test]
    fn a_click_beside_the_line_misses() {
        let s = Squiggle::with_points("s", [Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
        let view = pick_view();
        assert_eq!(s.screen_distance_within(&view, Coord::new(50.0, 80.0), 5.0), None);
    }

    #[test]
    fn picking_measures_to_the_segment_not_its_ends() {
        // A click level with the middle of a long segment is close to the line
        // even though it is far from either end point.
        let s = Squiggle::with_points("s", [Point::new(0.0, 0.0), Point::new(1000.0, 0.0)]);
        let view = pick_view();
        let distance = s.screen_distance_within(&view, Coord::new(500.0, 97.0), 5.0);
        assert_eq!(distance, Some(3.0));
    }

    #[test]
    fn an_empty_squiggle_cannot_be_picked() {
        assert_eq!(Squiggle::new("s").screen_distance_within(&pick_view(), Coord::ZERO, 1e9), None);
    }

    #[test]
    fn picking_toggles_selection() {
        let mut s = Squiggle::new("s");
        s.toggle_selected();
        assert!(s.selected);
        s.toggle_selected();
        assert!(!s.selected);
    }
}
