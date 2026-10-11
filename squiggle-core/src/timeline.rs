//! The shared time axis of stacked tracks.
//!
//! Time is `f64` seconds from the project reference. `px_per_s` spans
//! 1 px/day to 1 px/10 ps, so a 100 Hz trace and a 1 GS/s capture share it.

use crate::axis::Tick;

pub const MIN_PX_PER_S: f64 = 1.0 / 86_400.0;
pub const MAX_PX_PER_S: f64 = 1.0e11;

/// Target spacing between labelled ticks, in pixels.
const MAJOR_SPACING_PX: f64 = 120.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeView {
    /// Time at the left edge of the plot area.
    pub start: f64,
    pub px_per_s: f64,
}

impl Default for TimeView {
    fn default() -> Self {
        Self { start: 0.0, px_per_s: 100.0 }
    }
}

impl TimeView {
    /// Pixels from the left edge of the plot area.
    pub fn x_of(&self, time: f64) -> f64 {
        (time - self.start) * self.px_per_s
    }

    pub fn time_at(&self, x: f64) -> f64 {
        self.start + x / self.px_per_s
    }

    pub fn end(&self, width: f64) -> f64 {
        self.time_at(width)
    }

    /// Positive `dx` moves the data right.
    pub fn pan(&mut self, dx: f64) {
        self.start -= dx / self.px_per_s;
    }

    /// Scales about `x`, keeping the time under it fixed.
    pub fn zoom_about(&mut self, x: f64, factor: f64) {
        let anchor = self.time_at(x);
        self.px_per_s = (self.px_per_s * factor).clamp(MIN_PX_PER_S, MAX_PX_PER_S);
        self.start = anchor - x / self.px_per_s;
    }

    /// Frames `from..to` in `width` pixels with a 2 % margin each side.
    pub fn fit(&mut self, from: f64, to: f64, width: f64) {
        if !(width > 0.0 && from.is_finite() && to.is_finite()) {
            return;
        }
        let span = (to - from).max(1.0 / MAX_PX_PER_S);
        let margin = span * 0.02;
        self.px_per_s = (width / (span + 2.0 * margin)).clamp(MIN_PX_PER_S, MAX_PX_PER_S);
        self.start = from - margin;
    }

    /// Major ticks (labelled in seconds) and minor ticks across `width` pixels.
    pub fn ticks(&self, width: f64) -> Vec<Tick> {
        if !(width > 0.0 && self.px_per_s > 0.0) {
            return Vec::new();
        }
        let (major, minor) = nice_step(MAJOR_SPACING_PX / self.px_per_s);
        let decimals = (-major.log10().floor()).max(0.0) as usize;
        let mut ticks = Vec::new();
        let first = (self.start / minor).floor() as i64;
        let last = (self.end(width) / minor).ceil() as i64;
        let per_major = (major / minor).round() as i64;
        for n in first..=last {
            let x = self.x_of(n as f64 * minor);
            if !(0.0..=width).contains(&x) {
                continue;
            }
            if n.rem_euclid(per_major) == 0 {
                let value = n as f64 * minor;
                // Avoid "-0".
                let value = if value.abs() < minor / 2.0 { 0.0 } else { value };
                ticks.push(Tick { position: x, label: Some(format!("{value:.decimals$}")) });
            } else {
                ticks.push(Tick { position: x, label: None });
            }
        }
        ticks
    }
}

/// The 1-2-5 step at or above `raw`, and its minor subdivision.
pub fn nice_step(raw: f64) -> (f64, f64) {
    let power = 10f64.powf(raw.log10().floor());
    let mantissa = raw / power;
    let (major, divisions) = match mantissa {
        m if m <= 1.0 => (1.0, 5.0),
        m if m <= 2.0 => (2.0, 4.0),
        m if m <= 5.0 => (5.0, 5.0),
        _ => (10.0, 5.0),
    };
    (major * power, major * power / divisions)
}

/// Seconds with an SI prefix: `1.5 ms`, `20 ns`.
pub fn format_duration(seconds: f64) -> String {
    let magnitude = seconds.abs();
    let (scale, unit) = match magnitude {
        m if m >= 1.0 || m == 0.0 => (1.0, "s"),
        m if m >= 1e-3 => (1e3, "ms"),
        m if m >= 1e-6 => (1e6, "µs"),
        m if m >= 1e-9 => (1e9, "ns"),
        _ => (1e12, "ps"),
    };
    let value = seconds * scale;
    let text = format!("{value:.3}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    format!("{text} {unit}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_follow_one_two_five() {
        assert_eq!(nice_step(0.7), (1.0, 0.2));
        assert_eq!(nice_step(1.5), (2.0, 0.5));
        assert_eq!(nice_step(30.0), (50.0, 10.0));
        let (major, _) = nice_step(3e-9);
        assert!((major - 5e-9).abs() < 1e-20);
    }

    #[test]
    fn zoom_keeps_the_anchor_time() {
        let mut view = TimeView { start: 10.0, px_per_s: 50.0 };
        let before = view.time_at(200.0);
        view.zoom_about(200.0, 1e6);
        assert!((view.time_at(200.0) - before).abs() < 1e-12);
    }

    #[test]
    fn nanosecond_ticks_far_from_the_reference_are_distinct() {
        let mut view = TimeView::default();
        view.fit(100.0, 100.0 + 1e-6, 1000.0);
        let labels: Vec<String> = view.ticks(1000.0).into_iter().filter_map(|t| t.label).collect();
        assert!(labels.len() >= 4, "{labels:?}");
        let mut unique = labels.clone();
        unique.dedup();
        assert_eq!(unique, labels);
        assert!(labels[0].starts_with("100.000000"), "{labels:?}");
    }

    #[test]
    fn ticks_stay_inside_the_width() {
        let view = TimeView { start: -0.5, px_per_s: 300.0 };
        let ticks = view.ticks(800.0);
        assert!(ticks.iter().all(|t| (0.0..=800.0).contains(&t.position)));
        assert!(ticks.iter().any(|t| t.label.as_deref() == Some("0.0")));
    }

    #[test]
    fn durations_pick_a_prefix() {
        assert_eq!(format_duration(0.0015), "1.5 ms");
        assert_eq!(format_duration(2e-8), "20 ns");
        assert_eq!(format_duration(12.0), "12 s");
    }
}
