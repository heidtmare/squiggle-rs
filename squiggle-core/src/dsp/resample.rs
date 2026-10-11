//! Band-limited resampling by windowed-sinc interpolation at any ratio.
//!
//! The kernel is a Kaiser-windowed sinc, tabulated once. Downsampling widens
//! it by `in/out` so it doubles as the anti-alias filter. Each output sample
//! is normalised by its weight sum, so DC passes exactly and truncated edges
//! stay unbiased. Any NaN in the support yields NaN.

use core::f64::consts::PI;
use std::sync::OnceLock;

/// Kernel half-width, in zero crossings at the cutoff.
const HALF_WIDTH: usize = 16;
/// Table points per zero crossing.
const OVERSAMPLE: usize = 256;
const KAISER_BETA: f64 = 8.0;
/// Cutoff as a fraction of the lower Nyquist, leaving room for the transition band.
const ROLLOFF: f64 = 0.92;

fn table() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let i0_beta = bessel_i0(KAISER_BETA);
        (0..=HALF_WIDTH * OVERSAMPLE + 1)
            .map(|i| {
                let x = i as f64 / OVERSAMPLE as f64;
                let sinc = if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) };
                let r = x / HALF_WIDTH as f64;
                let window = if r >= 1.0 { 0.0 } else { bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / i0_beta };
                sinc * window
            })
            .collect()
    })
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..50 {
        term *= (half / k as f64).powi(2);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn kernel(x: f64) -> f64 {
    let pos = x.abs() * OVERSAMPLE as f64;
    let i = pos as usize;
    if i >= HALF_WIDTH * OVERSAMPLE {
        return 0.0;
    }
    let t = table();
    let frac = pos - i as f64;
    t[i] + (t[i + 1] - t[i]) * frac
}

/// Cutoff in cycles per input sample, relative to the input Nyquist.
fn cutoff(in_rate: f64, out_rate: f64) -> f64 {
    (out_rate / in_rate).min(1.0) * ROLLOFF
}

/// Support half-width in input samples.
fn support(cutoff: f64) -> f64 {
    HALF_WIDTH as f64 / cutoff
}

/// Interpolates `values` (absolute indices `base..base + values.len()`) at
/// fractional input index `u`.
fn evaluate(values: &[f32], base: usize, u: f64, cutoff: f64) -> f32 {
    let reach = support(cutoff);
    let lo = ((u - reach).ceil().max(base as f64)) as usize;
    let hi = ((u + reach).floor() as isize).min((base + values.len()) as isize - 1);
    if hi < lo as isize {
        return f32::NAN;
    }
    let hi = hi as usize;
    let (mut sum, mut weights) = (0.0, 0.0);
    for n in lo..=hi {
        let w = kernel((u - n as f64) * cutoff);
        sum += w * f64::from(values[n - base]);
        weights += w;
    }
    if weights.abs() < 1e-12 { f32::NAN } else { (sum / weights) as f32 }
}

/// Samples of `values` (rate `in_rate`, first sample at `in_start` seconds) at
/// `out_start + k / out_rate` for `k < count`. Times more than half a sample
/// outside the input are NaN.
pub fn resample_at(values: &[f32], in_rate: f64, in_start: f64, out_start: f64, out_rate: f64, count: usize) -> Vec<f32> {
    let c = cutoff(in_rate, out_rate);
    let last = values.len() as f64 - 0.5;
    (0..count)
        .map(|k| {
            let u = (out_start + k as f64 / out_rate - in_start) * in_rate;
            if values.is_empty() || u < -0.5 || u > last { f32::NAN } else { evaluate(values, 0, u, c) }
        })
        .collect()
}

/// Whole-buffer resampling; identical to streaming through [`Resampler`].
pub fn resample(values: &[f32], in_rate: f64, out_rate: f64) -> Vec<f32> {
    let mut resampler = Resampler::new(in_rate, out_rate);
    let mut out = resampler.process(values);
    out.extend(resampler.finish());
    out
}

/// Streaming resampler: keeps the tail of the input needed by future outputs.
/// Output `k` is at input time `k · in_rate / out_rate` samples.
#[derive(Clone, Debug)]
pub struct Resampler {
    step: f64,
    cutoff: f64,
    history: Vec<f32>,
    /// Absolute index of `history[0]`.
    base: usize,
    /// Input samples seen.
    total: usize,
    next: u64,
}

impl Resampler {
    pub fn new(in_rate: f64, out_rate: f64) -> Self {
        Self { step: in_rate / out_rate, cutoff: cutoff(in_rate, out_rate), history: Vec::new(), base: 0, total: 0, next: 0 }
    }

    fn position(&self) -> f64 {
        self.next as f64 * self.step
    }

    pub fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        self.history.extend_from_slice(chunk);
        self.total += chunk.len();
        let reach = support(self.cutoff);
        let mut out = Vec::new();
        while self.position() + reach <= self.total as f64 - 1.0 {
            out.push(evaluate(&self.history, self.base, self.position(), self.cutoff));
            self.next += 1;
        }
        // Drop input no future output can reach.
        let keep_from = ((self.position() - reach).floor().max(0.0) as usize).max(self.base);
        self.history.drain(..keep_from - self.base);
        self.base = keep_from;
        out
    }

    /// Outputs up to the last input sample, with the kernel truncated at the end.
    pub fn finish(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        while self.total > 0 && self.position() <= self.total as f64 - 1.0 {
            out.push(evaluate(&self.history, self.base, self.position(), self.cutoff));
            self.next += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: f64, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * hz * i as f64 / rate).sin() as f32).collect()
    }

    #[test]
    fn chunked_matches_whole() {
        let input: Vec<f32> = (0..5000).map(|i| ((i * 7919) % 211) as f32).collect();
        for (a, b) in [(100.0, 40.0), (40.0, 100.0), (1e6, 3e5)] {
            let whole = resample(&input, a, b);
            let mut r = Resampler::new(a, b);
            let mut pieces = Vec::new();
            for chunk in input.chunks(333) {
                pieces.extend(r.process(chunk));
            }
            pieces.extend(r.finish());
            assert_eq!(whole.len(), pieces.len());
            assert!(whole.iter().zip(&pieces).all(|(x, y)| x.to_bits() == y.to_bits()), "{a}->{b}");
        }
    }

    #[test]
    fn in_band_tones_survive_and_aliases_are_removed() {
        let n = 20_000;
        let kept = resample(&tone(5.0, 1000.0, n), 1000.0, 100.0);
        assert_eq!(kept.len(), 2000);
        let expected = tone(5.0, 100.0, 2000);
        let error = kept[100..1900].iter().zip(&expected[100..1900]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(error < 2e-3, "{error}");
        // 130 Hz would alias to 30 Hz at 100 S/s.
        let alias = resample(&tone(130.0, 1000.0, n), 1000.0, 100.0);
        assert!(alias[100..1900].iter().all(|v| v.abs() < 2e-3));
    }

    #[test]
    fn upsampling_interpolates_smoothly() {
        let up = resample(&tone(3.0, 100.0, 1000), 100.0, 1000.0);
        let expected = tone(3.0, 1000.0, up.len());
        let error = up[500..9000].iter().zip(&expected[500..9000]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(error < 1e-3, "{error}");
    }

    #[test]
    fn evaluation_at_arbitrary_times_honours_start() {
        let values = tone(2.0, 100.0, 1000);
        // 0.0137 s later than the input start, at 50 S/s.
        let out = resample_at(&values, 100.0, 1.0, 1.0137, 50.0, 400);
        let expected: Vec<f32> = (0..400).map(|k| (2.0 * PI * 2.0 * (0.0137 + k as f64 / 50.0)).sin() as f32).collect();
        let error = out[20..380].iter().zip(&expected[20..380]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(error < 1e-3, "{error}");
        assert!(resample_at(&values, 100.0, 1.0, 0.0, 50.0, 1)[0].is_nan());
    }

    #[test]
    fn dc_passes_exactly() {
        let out = resample(&[2.5; 300], 48_000.0, 44_100.0);
        assert!(out.iter().all(|v| (v - 2.5).abs() < 1e-6));
    }
}
