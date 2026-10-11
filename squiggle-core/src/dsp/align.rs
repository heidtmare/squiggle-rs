//! Lag measurement between traces of any rates and start times.
//!
//! Both traces are resampled onto one grid at a common analysis rate (the
//! lower rate, reduced further to stay under a sample budget). The reference
//! keeps only the analysis window; the other trace spans the window plus the
//! search range on each side.

use super::correlate::{LagEstimate, LagMethod, PairLag, correlation, estimate_lag, solve_delays};
use super::resample::resample_at;

/// One trace as the aligner sees it: its plotted series and timing.
#[derive(Clone, Copy, Debug)]
pub struct Channel<'a> {
    pub values: &'a [f32],
    pub rate: f64,
    /// Seconds from the project reference, alignment included.
    pub start: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignSettings {
    pub method: LagMethod,
    /// Analysis window, seconds from the reference.
    pub from: f64,
    pub to: f64,
    /// Search range either side, seconds.
    pub max_lag_s: f64,
    /// Cap on analysis samples across window plus search range.
    pub max_samples: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measured {
    /// `b`'s delay behind `a`, seconds.
    pub lag_s: f64,
    pub estimate: LagEstimate,
    pub analysis_rate: f64,
}

impl Measured {
    /// Weight for the consistency solve: positive correlation, unambiguous peak.
    pub fn weight(&self) -> f64 {
        self.estimate.coefficient.max(0.0) * self.estimate.confidence
    }
}

pub fn analysis_rate(a: f64, b: f64, settings: &AlignSettings) -> f64 {
    let span = settings.to - settings.from + 2.0 * settings.max_lag_s;
    a.min(b).min(settings.max_samples as f64 / span)
}

/// Both channels on one grid at the analysis rate, plus the search range in
/// grid samples. `None` when the window misses either trace.
fn grids(a: &Channel, b: &Channel, settings: &AlignSettings) -> Option<(Vec<f32>, Vec<f32>, f64, usize)> {
    if settings.to <= settings.from || settings.max_lag_s < 0.0 {
        return None;
    }
    let rate = analysis_rate(a.rate, b.rate, settings);
    let grid_start = settings.from - settings.max_lag_s;
    let count = ((settings.to - settings.from + 2.0 * settings.max_lag_s) * rate).ceil() as usize + 1;
    let lag_samples = (settings.max_lag_s * rate).ceil() as usize;

    let mut reference = resample_at(a.values, a.rate, a.start, grid_start, rate, count);
    // Outside the window the reference is silent, so only the window is matched.
    for (k, v) in reference.iter_mut().enumerate() {
        let t = grid_start + k as f64 / rate;
        if t < settings.from || t > settings.to {
            *v = f32::NAN;
        }
    }
    let other = resample_at(b.values, b.rate, b.start, grid_start, rate, count);
    if reference.iter().all(|v| !v.is_finite()) || other.iter().all(|v| !v.is_finite()) {
        return None;
    }
    Some((reference, other, rate, lag_samples))
}

/// Delay of `b` behind `a` within the settings' window, or `None` when the
/// window misses either trace or no positive peak exists.
pub fn measure(a: &Channel, b: &Channel, settings: &AlignSettings) -> Option<Measured> {
    let (reference, other, rate, lag_samples) = grids(a, b, settings)?;
    let estimate = estimate_lag(&reference, &other, lag_samples, settings.method)?;
    Some(Measured { lag_s: estimate.lag / rate, estimate, analysis_rate: rate })
}

/// Local maxima of the normalised cross-correlation at or above `min`, as
/// `(lag seconds, coefficient)`, strongest first. Used to snap manual shifts.
pub fn candidates(a: &Channel, b: &Channel, settings: &AlignSettings, min: f64) -> Vec<(f64, f64)> {
    let Some((reference, other, rate, lag_samples)) = grids(a, b, settings) else {
        return Vec::new();
    };
    let r = correlation(&reference, &other, lag_samples, LagMethod::CrossCorrelation);
    let mut peaks: Vec<(f64, f64)> = (1..r.len().saturating_sub(1))
        .filter(|&i| r[i] >= min && r[i] > r[i - 1] && r[i] >= r[i + 1])
        .map(|i| {
            let denom = r[i - 1] - 2.0 * r[i] + r[i + 1];
            let offset = if denom < 0.0 { (0.5 * (r[i - 1] - r[i + 1]) / denom).clamp(-0.5, 0.5) } else { 0.0 };
            ((i as f64 + offset - lag_samples as f64) / rate, r[i])
        })
        .collect();
    peaks.sort_by(|x, y| y.1.total_cmp(&x.1));
    peaks
}

/// Result of aligning several channels to one.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    /// Per channel: delay behind the reference in seconds (shift by its negative to align).
    pub delays: Vec<Option<f64>>,
    /// Every measured pair, `(a, b, measurement)`.
    pub pairs: Vec<(usize, usize, Measured)>,
}

impl Solution {
    /// `measured − (delay_b − delay_a)` for a pair, seconds.
    pub fn residual(&self, a: usize, b: usize, measured: &Measured) -> Option<f64> {
        Some(measured.lag_s - (self.delays.get(b).copied()?? - self.delays.get(a).copied()??))
    }
}

/// Measures the reference against every channel, or every pair when
/// `all_pairs`, then reconciles the lags by weighted least squares.
pub fn align(channels: &[Channel], reference: usize, settings: &AlignSettings, all_pairs: bool) -> Solution {
    let mut pairs = Vec::new();
    for a in 0..channels.len() {
        for b in a + 1..channels.len() {
            if !all_pairs && a != reference && b != reference {
                continue;
            }
            if let Some(m) = measure(&channels[a], &channels[b], settings) {
                pairs.push((a, b, m));
            }
        }
    }
    let lags: Vec<PairLag> = pairs.iter().map(|&(a, b, m)| PairLag { a, b, lag: m.lag_s, weight: m.weight() }).collect();
    Solution { delays: solve_delays(channels.len(), &lags, reference), pairs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    /// A band-limited event (energy 1–12 Hz) centred at `t0`.
    fn event(t: f64, t0: f64) -> f64 {
        let envelope = (-((t - t0) / 0.6).powi(2)).exp();
        let tones = [(1.3, 0.2, 1.0), (3.7, 1.1, 0.8), (6.1, 2.3, 0.6), (9.4, 0.7, 0.5), (11.8, 4.0, 0.3)];
        envelope * tones.iter().map(|(f, phase, a)| a * (TAU * f * t + phase).sin()).sum::<f64>()
    }

    fn sampled(rate: f64, start: f64, len: usize, delay: f64) -> Vec<f32> {
        (0..len).map(|i| event(start + i as f64 / rate - delay, 5.0) as f32).collect()
    }

    #[test]
    fn known_delays_at_mixed_rates_are_recovered_within_a_sample() {
        // (rate, start, true delay): different rates, starts and delays.
        let truth = [(100.0, 0.0, 0.0), (40.0, 0.37, 0.8125), (250.0, -1.2, -0.433), (100.0, 2.0, 1.9)];
        let data: Vec<Vec<f32>> = truth.iter().map(|&(rate, start, delay)| sampled(rate, start, (12.0 * rate) as usize, delay)).collect();
        let channels: Vec<Channel> = truth
            .iter()
            .zip(&data)
            .map(|(&(rate, start, _), values)| Channel { values, rate, start })
            .collect();
        for method in LagMethod::ALL {
            let settings = AlignSettings { method, from: 2.5, to: 7.5, max_lag_s: 2.5, max_samples: 1 << 20 };
            for all_pairs in [false, true] {
                let solution = align(&channels, 0, &settings, all_pairs);
                for (i, &(_, _, delay)) in truth.iter().enumerate() {
                    let found = solution.delays[i].unwrap();
                    // Within one sample of the 40 Hz analysis rate.
                    assert!((found - delay).abs() < 1.0 / 40.0, "{method} pairs={all_pairs} #{i}: {found} vs {delay}");
                }
                for &(a, b, m) in &solution.pairs {
                    assert!(solution.residual(a, b, &m).unwrap().abs() < 1.0 / 40.0, "{method} {a}-{b}: {m:?}");
                    assert!(m.estimate.confidence > 0.3, "{method} {a}-{b}: {m:?}");
                }
            }
        }
    }

    #[test]
    fn windows_that_miss_a_trace_measure_nothing() {
        let values = sampled(100.0, 0.0, 1000, 0.0);
        let a = Channel { values: &values, rate: 100.0, start: 0.0 };
        let b = Channel { values: &values, rate: 100.0, start: 100.0 };
        let settings = AlignSettings { method: LagMethod::CrossCorrelation, from: 2.0, to: 8.0, max_lag_s: 1.0, max_samples: 1 << 16 };
        assert!(measure(&a, &b, &settings).is_none());
    }

    #[test]
    fn candidates_include_the_true_delay_first() {
        let a = sampled(100.0, 0.0, 1000, 0.0);
        let b = sampled(100.0, 0.0, 1000, 0.42);
        let settings = AlignSettings { method: LagMethod::CrossCorrelation, from: 3.0, to: 7.0, max_lag_s: 2.0, max_samples: 1 << 16 };
        let peaks = candidates(&Channel { values: &a, rate: 100.0, start: 0.0 }, &Channel { values: &b, rate: 100.0, start: 0.0 }, &settings, 0.2);
        assert!((peaks[0].0 - 0.42).abs() < 0.01, "{peaks:?}");
        assert!(peaks.len() > 1);
    }

    #[test]
    fn the_sample_budget_lowers_the_analysis_rate() {
        let settings = AlignSettings { method: LagMethod::GccPhat, from: 0.0, to: 1.0, max_lag_s: 0.5, max_samples: 1000 };
        assert_eq!(analysis_rate(1e9, 1e6, &settings), 500.0);
    }
}
