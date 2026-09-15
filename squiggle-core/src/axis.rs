//! Axis definitions and tick generation.
//!
//! The Java original repeated a near-identical tick loop in each of its four
//! axis classes. Here a single generator drives every axis, and the axis only
//! decides where its ticks sit and what they are called.

use crate::view::PlotView;

/// Upper bound on ticks per axis, so a tiny step cannot stall the frame.
const MAX_TICKS: usize = 4096;

/// Whether a measurement is usable: strictly positive and finite.
///
/// Written as a positive test so that NaN, which compares false against
/// everything, is rejected rather than slipping through a negated `<=`.
fn is_usable(value: f64) -> bool {
    value > 0.0 && value.is_finite()
}

/// Which screen dimension an axis runs along.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Orientation {
    /// Horizontal, drawn along the bottom of the plot.
    X,
    /// Vertical, drawn up the left of the plot.
    Y,
}

/// What the tick sequence is measured from.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Anchor {
    /// Ticks start at the edge of the plot and keep a fixed pixel spacing,
    /// so they do not move when the plot is panned.
    Screen,
    /// Ticks start at cartesian zero and scale with the data, so a tick always
    /// marks the same value.
    #[default]
    Value,
}

/// How tick values progress along an axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AxisKind {
    /// Equal pixel steps carry equal value steps.
    Linear,
    /// Equal pixel steps carry equal *multiples*: each major step is one
    /// decade of `base`. This completes what the Java `Logrithmic*Axis`
    /// classes started; there the minor ticks were left as a TODO and the
    /// vertical variant never applied a log scale at all.
    Logarithmic { base: f64 },
}

impl AxisKind {
    /// Base-10 decades, the usual choice.
    pub const DECADE: Self = Self::Logarithmic { base: 10.0 };
}

/// One mark on an axis. Minor ticks carry no label.
#[derive(Clone, Debug, PartialEq)]
pub struct Tick {
    /// Position along the axis in screen pixels: x for [`Orientation::X`],
    /// y for [`Orientation::Y`].
    pub position: f64,
    /// Text for a major tick; `None` marks a minor tick.
    pub label: Option<String>,
}

impl Tick {
    fn minor(position: f64) -> Self {
        Self { position, label: None }
    }

    fn major(position: f64, label: String) -> Self {
        Self { position, label: Some(label) }
    }

    pub fn is_major(&self) -> bool {
        self.label.is_some()
    }
}

/// A labelled axis along one edge of the plot.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub name: String,
    pub orientation: Orientation,
    pub kind: AxisKind,
    pub anchor: Anchor,
    /// Spacing between labelled ticks, in cartesian units for
    /// [`Anchor::Value`] and in pixels for [`Anchor::Screen`].
    pub major_step: f64,
    /// Spacing between unlabelled ticks, in the same units as `major_step`.
    pub minor_step: f64,
}

impl Axis {
    /// A linear axis with the Java defaults: major every 100 units, minor
    /// every 10, anchored to the data.
    pub fn linear(name: impl Into<String>, orientation: Orientation) -> Self {
        Self {
            name: name.into(),
            orientation,
            kind: AxisKind::Linear,
            anchor: Anchor::Value,
            major_step: 100.0,
            minor_step: 10.0,
        }
    }

    /// A base-10 logarithmic axis, one decade per major step.
    pub fn logarithmic(name: impl Into<String>, orientation: Orientation) -> Self {
        Self { kind: AxisKind::DECADE, ..Self::linear(name, orientation) }
    }

    pub fn with_anchor(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }

    pub fn with_steps(mut self, major: f64, minor: f64) -> Self {
        self.major_step = major;
        self.minor_step = minor;
        self
    }

    /// The ticks visible in `view`, ordered minor-then-major as the renderer
    /// wants them: minor ticks first so major ticks and their labels win.
    pub fn ticks(&self, view: &PlotView) -> Vec<Tick> {
        let Some(span) = self.span(view) else {
            return Vec::new();
        };

        match self.kind {
            AxisKind::Linear => self.linear_ticks(view, &span),
            AxisKind::Logarithmic { base } => self.log_ticks(&span, base),
        }
    }

    /// Resolves the axis into the screen-space window it occupies, or `None`
    /// when the plot is too small or the steps are unusable.
    fn span(&self, view: &PlotView) -> Option<Span> {
        let plot_origin = view.plot_origin();
        let (min, max, scale, screen_anchor) = match self.orientation {
            Orientation::X => {
                let min = plot_origin.x;
                (min, min + view.plot_width(), view.scale.x, min)
            }
            Orientation::Y => {
                let min = plot_origin.y;
                let max = min + view.plot_height();
                // Screen-anchored vertical ticks count up from the bottom.
                (min, max, view.scale.y, max)
            }
        };

        if !is_usable(max - min) {
            return None;
        }

        let (origin, step_scale) = match self.anchor {
            Anchor::Screen => (screen_anchor, 1.0),
            Anchor::Value => {
                let cart_origin = view.cart_origin();
                let origin = match self.orientation {
                    Orientation::X => cart_origin.x,
                    Orientation::Y => cart_origin.y,
                };
                (origin, scale)
            }
        };

        let major = self.major_step * step_scale;
        let minor = self.minor_step * step_scale;
        if !origin.is_finite() || !is_usable(major) {
            return None;
        }

        Some(Span { min, max, origin, major, minor })
    }

    fn linear_ticks(&self, view: &PlotView, span: &Span) -> Vec<Tick> {
        let mut ticks = Vec::new();
        // Each sweep draws from one shared budget, so a pathological step
        // cannot run the axis away with us.
        macro_rules! budget {
            () => {
                MAX_TICKS.saturating_sub(ticks.len())
            };
        }

        // Minor ticks first, skipping the origin itself.
        if span.minor > 0.0 {
            span.walk_up(span.origin + span.minor, span.minor, budget!(), &mut |p| {
                ticks.push(Tick::minor(p));
            });
            span.walk_down(span.origin - span.minor, span.minor, budget!(), &mut |p| {
                ticks.push(Tick::minor(p));
            });
        }

        // Major ticks, the downward sweep including the origin.
        let label_at = |position: f64| {
            let value = match self.orientation {
                Orientation::X => view.screen_x_to_cart_x(position),
                Orientation::Y => view.screen_y_to_cart_y(position),
            };
            format_value(value)
        };
        span.walk_up(span.origin + span.major, span.major, budget!(), &mut |p| {
            ticks.push(Tick::major(p, label_at(p)));
        });
        span.walk_down(span.origin, span.major, budget!(), &mut |p| {
            ticks.push(Tick::major(p, label_at(p)));
        });

        ticks
    }

    fn log_ticks(&self, span: &Span, base: f64) -> Vec<Tick> {
        if !is_usable(base - 1.0) {
            return Vec::new();
        }

        // Minor ticks subdivide each decade at log-spaced fractions, which is
        // what makes a log axis readable between the powers.
        let subdivisions: Vec<f64> = (2..=9)
            .map(f64::from)
            .filter(|k| *k < base)
            .map(|k| k.log(base))
            .collect();

        let mut ticks = Vec::new();
        let push_decade = |decade: i32, ticks: &mut Vec<Tick>| {
            let anchor = span.origin + f64::from(decade) * span.major;
            for fraction in &subdivisions {
                let position = anchor + fraction * span.major;
                if span.contains(position) {
                    ticks.push(Tick::minor(position));
                }
            }
            if span.contains(anchor) {
                ticks.push(Tick::major(anchor, format_value(base.powi(decade))));
            }
        };

        // Sweep outwards from the origin's decade until both ends leave the span.
        let first = ((span.min - span.origin) / span.major).floor() as i32;
        let last = ((span.max - span.origin) / span.major).ceil() as i32;
        for decade in first..=last {
            if ticks.len() >= MAX_TICKS {
                break;
            }
            push_decade(decade, &mut ticks);
        }
        ticks.truncate(MAX_TICKS);

        ticks
    }
}

/// The screen-space window an axis draws into, with its steps already in pixels.
struct Span {
    min: f64,
    max: f64,
    origin: f64,
    major: f64,
    minor: f64,
}

impl Span {
    fn contains(&self, position: f64) -> bool {
        position >= self.min && position <= self.max
    }

    /// Steps forward from `start` until past `max`, reporting positions that
    /// fall inside the span. Positions before `min` are stepped over, matching
    /// the original's behaviour when the origin sits off the left of the plot.
    fn walk_up(&self, start: f64, step: f64, budget: usize, visit: &mut impl FnMut(f64)) {
        let mut position = start;
        let mut emitted = 0;
        while position <= self.max && emitted < budget {
            if position >= self.min {
                visit(position);
                emitted += 1;
            }
            position += step;
        }
    }

    fn walk_down(&self, start: f64, step: f64, budget: usize, visit: &mut impl FnMut(f64)) {
        let mut position = start;
        let mut emitted = 0;
        while position >= self.min && emitted < budget {
            if position <= self.max {
                visit(position);
                emitted += 1;
            }
            position -= step;
        }
    }
}

/// Formats a tick value like the Java `DecimalFormat("#.###")`: at most three
/// decimals, no trailing zeros, and no lone minus sign on a rounded-away value.
pub fn format_value(value: f64) -> String {
    if !value.is_finite() {
        return "-".to_string();
    }

    let text = format!("{value:.3}");
    let trimmed = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text.as_str()
    };

    match trimmed {
        "" | "-" | "-0" => "0".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{Coord, Insets};

    fn view() -> PlotView {
        PlotView {
            width: 420.0,
            height: 220.0,
            insets: Insets::new(0.0, 0.0, 20.0, 20.0),
            ..PlotView::default()
        }
    }

    fn majors(ticks: &[Tick]) -> Vec<f64> {
        let mut positions: Vec<f64> =
            ticks.iter().filter(|t| t.is_major()).map(|t| t.position).collect();
        positions.sort_by(f64::total_cmp);
        positions
    }

    #[test]
    fn value_anchored_majors_land_on_round_numbers() {
        let axis = Axis::linear("x", Orientation::X);
        let view = view();
        let ticks = axis.ticks(&view);

        for tick in ticks.iter().filter(|t| t.is_major()) {
            let value = view.screen_x_to_cart_x(tick.position);
            assert!((value % 100.0).abs() < 1e-6, "{value} is not a multiple of the major step");
        }
        // Plot is 400px wide starting at cartesian 0: 0, 100, 200, 300, 400.
        assert_eq!(majors(&ticks).len(), 5);
    }

    #[test]
    fn the_cartesian_origin_gets_a_major_tick() {
        let axis = Axis::linear("x", Orientation::X);
        let view = view();
        let origin = view.cart_origin().x;
        assert!(majors(&axis.ticks(&view)).iter().any(|p| (p - origin).abs() < 1e-9));
    }

    #[test]
    fn ticks_stay_inside_the_plot_area() {
        let axis = Axis::linear("y", Orientation::Y);
        let mut view = view();
        view.translation = Coord::new(0.0, -137.0);

        let (min, max) = (view.plot_origin().y, view.plot_origin().y + view.plot_height());
        for tick in axis.ticks(&view) {
            assert!(tick.position >= min && tick.position <= max, "{tick:?} outside {min}..{max}");
        }
    }

    #[test]
    fn zooming_out_keeps_the_same_values_labelled() {
        let axis = Axis::linear("x", Orientation::X);
        let mut view = view();
        view.scale = Coord::new(0.5, 0.5);

        for tick in axis.ticks(&view).iter().filter(|t| t.is_major()) {
            let value = view.screen_x_to_cart_x(tick.position);
            assert!((value % 100.0).abs() < 1e-6, "{value}");
        }
    }

    #[test]
    fn screen_anchored_ticks_ignore_pan() {
        let axis = Axis::linear("x", Orientation::X).with_anchor(Anchor::Screen);
        let mut view = view();
        let before = majors(&axis.ticks(&view));
        view.translation = Coord::new(73.0, 0.0);
        assert_eq!(before, majors(&axis.ticks(&view)));
    }

    #[test]
    fn a_zero_step_yields_no_ticks_instead_of_hanging() {
        let axis = Axis::linear("x", Orientation::X).with_steps(0.0, 0.0);
        assert!(axis.ticks(&view()).is_empty());
    }

    #[test]
    fn a_tiny_step_is_capped_rather_than_stalling() {
        let axis = Axis::linear("x", Orientation::X).with_steps(1e-9, 1e-9);
        let ticks = axis.ticks(&view());
        assert!(ticks.len() <= MAX_TICKS, "{} ticks", ticks.len());
    }

    #[test]
    fn log_majors_are_powers_of_the_base() {
        let axis = Axis::logarithmic("x", Orientation::X)
            .with_anchor(Anchor::Screen)
            .with_steps(100.0, 100.0);
        let labels: Vec<String> = axis
            .ticks(&view())
            .into_iter()
            .filter_map(|t| t.label)
            .collect();
        assert!(labels.contains(&"1".to_string()), "{labels:?}");
        assert!(labels.contains(&"10".to_string()), "{labels:?}");
        assert!(labels.contains(&"100".to_string()), "{labels:?}");
    }

    #[test]
    fn log_axes_subdivide_each_decade() {
        let axis = Axis::logarithmic("x", Orientation::X)
            .with_anchor(Anchor::Screen)
            .with_steps(100.0, 100.0);
        let ticks = axis.ticks(&view());
        assert!(ticks.iter().any(|t| !t.is_major()), "the Java version never drew these");
    }

    #[test]
    fn values_format_like_the_java_decimal_pattern() {
        assert_eq!(format_value(0.0), "0");
        assert_eq!(format_value(-0.0), "0");
        assert_eq!(format_value(100.0), "100");
        assert_eq!(format_value(1.5), "1.5");
        assert_eq!(format_value(1.23456), "1.235");
        assert_eq!(format_value(-0.0001), "0");
        assert_eq!(format_value(-12.25), "-12.25");
    }
}
