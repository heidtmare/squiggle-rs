//! Spectrograms at a fixed column budget: each column averages the power of
//! the overlapping frames that start inside it (Welch), so a long trace
//! squeezed into few columns still shows every burst.

use crate::transform::{Complex, Window, fft};

const FLOOR_DB: f32 = -240.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Spectrogram {
    pub columns: usize,
    /// Frequency bins, lowest first. Two-sided spectra run `−fs/2..fs/2`.
    pub bins: usize,
    /// Power in dB, `columns × bins`, column-major. NaN where a column has no
    /// gap-free frame.
    pub db: Vec<f32>,
    /// Input samples per column.
    pub hop: usize,
    pub two_sided: bool,
}

impl Spectrogram {
    pub fn at(&self, column: usize, bin: usize) -> f32 {
        self.db[column * self.bins + bin]
    }

    /// The `lo` and `hi` quantiles of the finite values, for a colour range.
    pub fn db_range(&self, lo: f64, hi: f64) -> (f32, f32) {
        let mut finite: Vec<f32> = self.db.iter().copied().filter(|v| v.is_finite() && *v > FLOOR_DB).collect();
        if finite.is_empty() {
            return (-100.0, 0.0);
        }
        finite.sort_by(f32::total_cmp);
        let pick = |q: f64| finite[((finite.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize];
        let (a, b) = (pick(lo), pick(hi));
        if b > a { (a, b) } else { (a - 1.0, a + 1.0) }
    }
}

/// Spectrogram of `input` with frames of `nfft` samples (50 % overlap) and
/// `hop` input samples per column. Real input (`two_sided = false`) gives
/// `nfft/2 + 1` bins; complex input gives `nfft`, centred on DC.
pub fn spectrogram(input: &[Complex], nfft: usize, hop: usize, window: Window, two_sided: bool) -> Spectrogram {
    let nfft = nfft.max(2);
    let hop = hop.max(1);
    let bins = if two_sided { nfft } else { nfft / 2 + 1 };
    let columns = input.len().div_ceil(hop);
    let taper = window.coefficients(nfft);
    let gain: f64 = taper.iter().sum::<f64>().powi(2);
    let step = (nfft / 2).max(1);
    let last_start = input.len().saturating_sub(nfft);

    let mut db = Vec::with_capacity(columns * bins);
    let mut frame = vec![Complex::ZERO; nfft];
    let mut power = vec![0.0f64; bins];
    for c in 0..columns {
        let (lo, hi) = (c * hop, ((c + 1) * hop).min(input.len()));
        // Frames starting in the column, or one centred on it when the column is narrower than a frame.
        let starts: Vec<usize> = if hop >= nfft {
            (lo..hi).step_by(step).map(|s| s.min(last_start)).collect()
        } else {
            vec![((lo + hi) / 2).saturating_sub(nfft / 2).min(last_start)]
        };
        power.fill(0.0);
        let mut frames = 0;
        for start in starts {
            let samples = &input[start..(start + nfft).min(input.len())];
            if samples.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
                continue;
            }
            for (k, slot) in frame.iter_mut().enumerate() {
                *slot = samples.get(k).map_or(Complex::ZERO, |z| z.scale(taper[k]));
            }
            fft(&mut frame);
            for (b, p) in power.iter_mut().enumerate() {
                let k = if two_sided { (b + nfft - nfft / 2) % nfft } else { b };
                let mut value = frame[k].norm_sqr();
                // Fold negative frequencies into one-sided bins.
                if !two_sided && b != 0 && !(nfft.is_multiple_of(2) && b == nfft / 2) {
                    value *= 2.0;
                }
                *p += value;
            }
            frames += 1;
        }
        for p in &power {
            db.push(if frames == 0 {
                f32::NAN
            } else {
                let mean = p / frames as f64 / gain;
                if mean > 0.0 { ((10.0 * mean.log10()) as f32).max(FLOOR_DB) } else { FLOOR_DB }
            });
        }
    }
    Spectrogram { columns, bins, db, hop, two_sided }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    fn peak_bin(s: &Spectrogram, column: usize) -> usize {
        (0..s.bins).max_by(|&a, &b| s.at(column, a).total_cmp(&s.at(column, b))).unwrap()
    }

    #[test]
    fn tones_land_in_their_bin() {
        let rate = 1000.0;
        // 125 Hz for the first half, 250 Hz after.
        let input: Vec<Complex> = (0..8192)
            .map(|i| {
                let hz = if i < 4096 { 125.0 } else { 250.0 };
                Complex::from((TAU * hz * i as f64 / rate).sin())
            })
            .collect();
        let s = spectrogram(&input, 256, 1024, Window::Hann, false);
        assert_eq!((s.columns, s.bins), (8, 129));
        assert_eq!(peak_bin(&s, 1), 32);
        assert_eq!(peak_bin(&s, 6), 64);
        // A unit sine is 0.5 power: −3 dB.
        assert!((s.at(1, 32) + 3.01).abs() < 0.1, "{}", s.at(1, 32));
    }

    #[test]
    fn complex_tones_keep_their_sign() {
        let input: Vec<Complex> = (0..4096).map(|i| Complex::cis(-TAU * 0.125 * i as f64)).collect();
        let s = spectrogram(&input, 64, 512, Window::Hann, true);
        assert_eq!(s.bins, 64);
        // −fs/8 sits a quarter below the centre bin.
        assert_eq!(peak_bin(&s, 3), 32 - 8);
    }

    #[test]
    fn gaps_leave_empty_columns() {
        let mut input = vec![Complex::from(1.0); 1024];
        input[256..512].fill(Complex::from(f64::NAN));
        let s = spectrogram(&input, 64, 256, Window::Hann, false);
        assert!(s.at(1, 0).is_nan());
        assert!(s.at(0, 0).is_finite());
        let (lo, hi) = s.db_range(0.05, 0.99);
        assert!(lo < hi);
    }
}
