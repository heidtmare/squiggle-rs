//! Frequency-domain results, and the way back from them.

use core::f64::consts::TAU;

use super::complex::Complex;
use super::fft;

/// The complex spectrum of a real signal, kept on a transformed squiggle so
/// that the inverse transform can reproduce the original exactly rather than
/// guess at it from the plotted magnitudes.
///
/// Only the non-negative frequencies are stored: a real signal's spectrum is
/// conjugate-symmetric, so the rest is implied.
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum {
    /// Coefficients for bins `0..=sample_count / 2`.
    pub coefficients: Vec<Complex>,
    /// How many samples the spectrum was computed from.
    pub sample_count: usize,
    pub basis: Basis,
}

/// Which transform produced a [`Spectrum`], and so which inverse undoes it.
#[derive(Clone, Debug, PartialEq)]
pub enum Basis {
    /// From the DFT of evenly spaced samples: `x[n] = 1/N Σ X[k]·e^(2πikn/N)`,
    /// at `start + n·interval`.
    Discrete { start: f64, interval: f64 },
    /// From the continuous Fourier integral, approximated over the samples'
    /// actual positions: `x(t) = ∫ X(f)·e^(2πift) df`, reconstructed at `times`.
    Continuous { times: Vec<f64> },
}

impl Spectrum {
    /// Spacing between frequency bins, in cycles per x unit.
    pub fn resolution(&self) -> f64 {
        let n = self.sample_count as f64;
        match &self.basis {
            Basis::Discrete { interval, .. } => 1.0 / (n * interval),
            Basis::Continuous { times } => {
                let span = times.last().copied().unwrap_or(0.0) - times.first().copied().unwrap_or(0.0);
                (n - 1.0) / (n * span)
            }
        }
    }

    /// Whether the last stored bin is the Nyquist bin, which has no negative
    /// twin and so counts once rather than twice.
    pub fn has_nyquist_bin(&self) -> bool {
        self.sample_count.is_multiple_of(2)
    }

    /// How many times bin `k` appears in the full two-sided spectrum.
    pub(super) fn multiplicity(&self, k: usize) -> f64 {
        if k == 0 || (self.has_nyquist_bin() && k == self.sample_count / 2) { 1.0 } else { 2.0 }
    }

    /// The coefficient at signed bin `k`, using conjugate symmetry for `k < 0`.
    pub(super) fn bin(&self, k: isize) -> Complex {
        let c = self.coefficients[k.unsigned_abs()];
        if k < 0 { c.conj() } else { c }
    }

    /// Reconstructs the signal: the x positions and the values at them.
    pub fn invert(&self) -> (Vec<f64>, Vec<f64>) {
        match &self.basis {
            Basis::Discrete { start, interval } => {
                let n = self.sample_count;
                let mut full = vec![Complex::ZERO; n];
                for (k, c) in self.coefficients.iter().enumerate() {
                    full[k] = *c;
                    if k != 0 {
                        full[n - k] = c.conj();
                    }
                }
                fft::ifft(&mut full);
                let times = (0..n).map(|i| start + i as f64 * interval).collect();
                (times, full.iter().map(|c| c.re).collect())
            }
            Basis::Continuous { times } => {
                let df = self.resolution();
                let values = times
                    .iter()
                    .map(|&t| {
                        // Step the phasor rather than calling sin/cos per bin.
                        let step = Complex::cis(TAU * df * t);
                        let mut phasor = Complex::ONE;
                        let mut sum = 0.0;
                        for (k, c) in self.coefficients.iter().enumerate() {
                            sum += self.multiplicity(k) * (*c * phasor).re;
                            phasor *= step;
                        }
                        df * sum
                    })
                    .collect();
                (times.clone(), values)
            }
        }
    }
}

/// The DFT of evenly spaced real samples, by summation or by FFT.
pub fn discrete(values: &[f64], start: f64, interval: f64, fast: bool) -> Spectrum {
    let n = values.len();
    let bins = n / 2 + 1;
    let input: Vec<Complex> = values.iter().map(|&v| Complex::from(v)).collect();
    let coefficients = if fast {
        let mut data = input;
        fft::fft(&mut data);
        data.truncate(bins);
        data
    } else {
        fft::dft(&input, bins)
    };
    Spectrum { coefficients, sample_count: n, basis: Basis::Discrete { start, interval } }
}

/// The continuous Fourier transform `X(f) = ∫ x(t)·e^(-2πift) dt`, approximated
/// by a Riemann sum over the samples where they actually lie. Unlike the DFT
/// this honours uneven spacing, and its phase is referenced to x = 0 rather
/// than to the first sample.
///
/// It is evaluated on the same frequency grid the DFT would use, so on evenly
/// spaced data it equals the DFT scaled by the sample interval.
pub fn continuous(times: &[f64], values: &[f64]) -> Spectrum {
    let n = times.len();
    // Each sample stands for the half-way cell around it.
    let weights: Vec<f64> = (0..n)
        .map(|i| match i {
            0 => times[1] - times[0],
            i if i == n - 1 => times[n - 1] - times[n - 2],
            i => (times[i + 1] - times[i - 1]) / 2.0,
        })
        .collect();

    let mut spectrum = Spectrum {
        coefficients: Vec::new(),
        sample_count: n,
        basis: Basis::Continuous { times: times.to_vec() },
    };
    let df = spectrum.resolution();

    let steps: Vec<Complex> = times.iter().map(|&t| Complex::cis(-TAU * df * t)).collect();
    let mut phasors = vec![Complex::ONE; n];
    spectrum.coefficients = (0..n / 2 + 1)
        .map(|_| {
            let mut sum = Complex::ZERO;
            for i in 0..n {
                sum += phasors[i].scale(values[i] * weights[i]);
                phasors[i] *= steps[i];
            }
            sum
        })
        .collect();
    spectrum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.4).sin() + 0.3 * (i as f64 * 2.1).cos() + 1.5).collect()
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-8, "{x} vs {y}");
        }
    }

    #[test]
    fn discrete_spectra_invert_exactly() {
        for (n, fast) in [(128, true), (101, true), (50, false)] {
            let values = wave(n);
            let spectrum = discrete(&values, 7.0, 0.5, fast);
            let (times, back) = spectrum.invert();
            assert_close(&back, &values);
            assert_eq!(times[0], 7.0);
            assert_eq!(times[n - 1], 7.0 + (n - 1) as f64 * 0.5);
        }
    }

    #[test]
    fn continuous_spectra_invert_exactly_on_even_spacing() {
        for n in [64, 63] {
            let values = wave(n);
            let times: Vec<f64> = (0..n).map(|i| 3.0 + i as f64 * 0.25).collect();
            let (back_times, back) = continuous(&times, &values).invert();
            assert_eq!(back_times, times);
            assert_close(&back, &values);
        }
    }

    #[test]
    fn the_continuous_transform_is_the_dft_scaled_by_the_interval() {
        // With samples starting at x = 0 the two share a phase reference too.
        let values = wave(40);
        let times: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
        let c = continuous(&times, &values);
        let d = discrete(&values, 0.0, 0.5, true);
        for (a, b) in c.coefficients.iter().zip(&d.coefficients) {
            assert!((*a - b.scale(0.5)).norm() < 1e-9, "{a:?} vs {b:?}");
        }
        assert!((c.resolution() - d.resolution()).abs() < 1e-12);
    }
}
