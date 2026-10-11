//! Time-delay estimation: FFT cross-correlation and GCC-PHAT, with sub-sample
//! peaks, plus a weighted least-squares solve that makes pairwise lags
//! consistent across many channels.
//!
//! Convention: a positive lag means `b` is late, `b(t) ≈ a(t − lag)`.

use core::fmt;

use crate::transform::{Complex, fft, ifft};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LagMethod {
    /// Normalised cross-correlation; the peak is a correlation coefficient.
    #[default]
    CrossCorrelation,
    /// Generalised cross-correlation with phase transform: whitened, so the
    /// peak is sharp for broadband signals and robust to coloured noise.
    GccPhat,
}

impl LagMethod {
    pub const ALL: [Self; 2] = [Self::CrossCorrelation, Self::GccPhat];

    pub fn label(self) -> &'static str {
        match self {
            Self::CrossCorrelation => "Cross-correlation",
            Self::GccPhat => "GCC-PHAT",
        }
    }
}

impl fmt::Display for LagMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagEstimate {
    /// Samples, sub-sample by parabolic interpolation.
    pub lag: f64,
    /// Normalised cross-correlation at the nearest whole lag, in `-1..=1`.
    pub coefficient: f64,
    /// `1 − second peak / main peak`, in `0..=1`: how unambiguous the peak is.
    pub confidence: f64,
}

/// `values` with NaN as zero, minus the mean of the finite samples.
fn centred(values: &[f32]) -> Vec<f64> {
    let finite: Vec<f64> = values.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v)).collect();
    let mean = if finite.is_empty() { 0.0 } else { finite.iter().sum::<f64>() / finite.len() as f64 };
    values.iter().map(|&v| if v.is_finite() { f64::from(v) - mean } else { 0.0 }).collect()
}

/// Correlation of `a` and `b` for lags `-max_lag..=max_lag` (index `i` is lag
/// `i − max_lag`). Cross-correlation is normalised by the signal energies.
pub fn correlation(a: &[f32], b: &[f32], max_lag: usize, method: LagMethod) -> Vec<f64> {
    let (a, b) = (centred(a), centred(b));
    let n = (a.len() + b.len()).max(2).next_power_of_two();
    let spectrum = |x: &[f64]| {
        let mut data: Vec<Complex> = x.iter().map(|&v| Complex::from(v)).collect();
        data.resize(n, Complex::ZERO);
        fft(&mut data);
        data
    };
    let (fa, fb) = (spectrum(&a), spectrum(&b));
    let mut cross: Vec<Complex> = fa.iter().zip(&fb).map(|(x, y)| x.conj() * *y).collect();
    let scale = match method {
        LagMethod::CrossCorrelation => {
            let energy = a.iter().map(|v| v * v).sum::<f64>() * b.iter().map(|v| v * v).sum::<f64>();
            if energy > 0.0 { 1.0 / energy.sqrt() } else { 0.0 }
        }
        LagMethod::GccPhat => {
            // Bins 80 dB below the strongest carry no signal; whitening them
            // would amplify rounding noise into a flat floor.
            let floor = cross.iter().map(|c| c.norm()).fold(0.0, f64::max) * 1e-4;
            for c in &mut cross {
                let m = c.norm();
                *c = if m > floor { c.scale(1.0 / m) } else { Complex::ZERO };
            }
            1.0
        }
    };
    ifft(&mut cross);
    let max_lag = max_lag.min(n / 2 - 1);
    (0..=2 * max_lag)
        .map(|i| {
            let lag = i as isize - max_lag as isize;
            cross[lag.rem_euclid(n as isize) as usize].re * scale
        })
        .collect()
}

/// Lag of `b` relative to `a`, searched within `±max_lag` samples.
pub fn estimate_lag(a: &[f32], b: &[f32], max_lag: usize, method: LagMethod) -> Option<LagEstimate> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let max_lag = max_lag.min(a.len().max(b.len()) - 1);
    let r = correlation(a, b, max_lag, method);
    let peak = (0..r.len()).max_by(|&i, &j| r[i].total_cmp(&r[j]))?;
    // A maximum on the search edge is a slope, not a peak.
    if !r[peak].is_finite() || r[peak] <= 0.0 || peak == 0 || peak + 1 == r.len() {
        return None;
    }

    // Parabolic refinement.
    let (l, c, h) = (r[peak - 1], r[peak], r[peak + 1]);
    let denom = l - 2.0 * c + h;
    let offset = if denom < 0.0 { (0.5 * (l - h) / denom).clamp(-0.5, 0.5) } else { 0.0 };

    // Main lobe: downhill from the peak on both sides.
    let mut left = peak;
    while left > 0 && r[left - 1] < r[left] {
        left -= 1;
    }
    let mut right = peak;
    while right + 1 < r.len() && r[right + 1] < r[right] {
        right += 1;
    }
    let second = r[..left].iter().chain(&r[right + 1..]).copied().fold(0.0, f64::max);
    let confidence = (1.0 - second / r[peak]).clamp(0.0, 1.0);

    let coefficient = match method {
        LagMethod::CrossCorrelation => r[peak],
        LagMethod::GccPhat => {
            let whole = peak as isize - max_lag as isize;
            let reach = whole.unsigned_abs();
            correlation(a, b, reach, LagMethod::CrossCorrelation)[(whole + reach as isize) as usize]
        }
    };
    Some(LagEstimate { lag: peak as f64 + offset - max_lag as f64, coefficient, confidence })
}

/// One measured lag between channels `a` and `b` (`t_b − t_a`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairLag {
    pub a: usize,
    pub b: usize,
    pub lag: f64,
    pub weight: f64,
}

/// Delays `t_i` relative to `reference` (`t_reference = 0`) minimising
/// `Σ w (t_b − t_a − lag)²`. Channels not connected to the reference by
/// positive-weight pairs are `None`.
pub fn solve_delays(channels: usize, pairs: &[PairLag], reference: usize) -> Vec<Option<f64>> {
    let pairs: Vec<&PairLag> = pairs
        .iter()
        .filter(|p| p.weight > 0.0 && p.a != p.b && p.a < channels && p.b < channels && p.lag.is_finite())
        .collect();

    // Channels reachable from the reference.
    let mut reached = vec![false; channels];
    if reference < channels {
        reached[reference] = true;
        let mut changed = true;
        while changed {
            changed = false;
            for p in &pairs {
                if reached[p.a] != reached[p.b] {
                    reached[p.a] = true;
                    reached[p.b] = true;
                    changed = true;
                }
            }
        }
    }
    let unknowns: Vec<usize> = (0..channels).filter(|&i| reached[i] && i != reference).collect();
    let column = |i: usize| unknowns.iter().position(|&u| u == i);
    let m = unknowns.len();

    // Normal equations: a weighted graph Laplacian with the reference removed.
    let mut matrix = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];
    for p in pairs.iter().filter(|p| reached[p.a]) {
        let (ia, ib) = (column(p.a), column(p.b));
        if let Some(a) = ia {
            matrix[a][a] += p.weight;
            rhs[a] -= p.weight * p.lag;
        }
        if let Some(b) = ib {
            matrix[b][b] += p.weight;
            rhs[b] += p.weight * p.lag;
        }
        if let (Some(a), Some(b)) = (ia, ib) {
            matrix[a][b] -= p.weight;
            matrix[b][a] -= p.weight;
        }
    }
    let solution = solve(matrix, rhs);

    (0..channels)
        .map(|i| {
            if i == reference && i < channels {
                Some(0.0)
            } else {
                column(i).and_then(|c| solution.as_ref().map(|s| s[c]))
            }
        })
        .collect()
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            if factor != 0.0 {
                let (upper, lower) = a.split_at_mut(row);
                for (x, p) in lower[0][col..].iter_mut().zip(&upper[col][col..]) {
                    *x -= factor * p;
                }
                b[row] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Band-limited noise burst: deterministic, broadband enough for PHAT.
    pub(crate) fn burst(len: usize, centre: f64, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 2001) as f64 / 1000.0 - 1.0
        };
        let raw: Vec<f64> = (0..len + 4).map(|_| noise()).collect();
        (0..len)
            .map(|i| {
                let smooth = (raw[i] + 2.0 * raw[i + 1] + raw[i + 2]) / 4.0;
                let envelope = (-((i as f64 - centre) / 40.0).powi(2)).exp();
                (smooth * envelope) as f32
            })
            .collect()
    }

    fn shifted(signal: &[f32], by: isize) -> Vec<f32> {
        (0..signal.len() as isize)
            .map(|i| signal.get((i - by) as usize).copied().filter(|_| i - by >= 0).unwrap_or(0.0))
            .collect()
    }

    #[test]
    fn known_delays_are_recovered() {
        let a = burst(2000, 800.0, 7);
        for method in LagMethod::ALL {
            for delay in [-37isize, 0, 5, 123] {
                let b = shifted(&a, delay);
                let estimate = estimate_lag(&a, &b, 300, method).unwrap();
                assert!((estimate.lag - delay as f64).abs() < 0.5, "{method}: {delay} vs {}", estimate.lag);
                assert!(estimate.coefficient > 0.9, "{method}: {}", estimate.coefficient);
                assert!(estimate.confidence > 0.3, "{method}: {}", estimate.confidence);
            }
        }
    }

    #[test]
    fn sub_sample_delays_are_interpolated() {
        // A smooth pulse delayed by 10.3 samples.
        let pulse = |shift: f64| -> Vec<f32> {
            (0..512).map(|i| (-((i as f64 - 200.0 - shift) / 6.0).powi(2)).exp() as f32).collect()
        };
        let estimate = estimate_lag(&pulse(0.0), &pulse(10.3), 50, LagMethod::CrossCorrelation).unwrap();
        assert!((estimate.lag - 10.3).abs() < 0.05, "{}", estimate.lag);
    }

    #[test]
    fn noise_is_not_confident() {
        let a = burst(2000, 1000.0, 3);
        let b = burst(2000, 1000.0, 99);
        let estimate = estimate_lag(&a, &b, 500, LagMethod::CrossCorrelation);
        // Short bursts can correlate by chance; the ambiguity shows in confidence.
        assert!(estimate.is_none_or(|e| e.confidence < 0.3));
    }

    #[test]
    fn gaps_do_not_break_estimation() {
        let a = burst(2000, 800.0, 11);
        let mut b = shifted(&a, 20);
        b[100..140].fill(f32::NAN);
        let estimate = estimate_lag(&a, &b, 100, LagMethod::GccPhat).unwrap();
        assert!((estimate.lag - 20.0).abs() < 0.5);
    }

    #[test]
    fn inconsistent_pairs_are_reconciled() {
        // True delays 0, 10, 25; the 0→2 measurement is off by 3.
        let pairs = [
            PairLag { a: 0, b: 1, lag: 10.0, weight: 1.0 },
            PairLag { a: 1, b: 2, lag: 15.0, weight: 1.0 },
            PairLag { a: 0, b: 2, lag: 28.0, weight: 1.0 },
        ];
        let delays = solve_delays(4, &pairs, 0);
        assert_eq!(delays[0], Some(0.0));
        assert!((delays[1].unwrap() - 11.0).abs() < 1e-9);
        assert!((delays[2].unwrap() - 27.0).abs() < 1e-9);
        assert_eq!(delays[3], None);
        // Reference need not be channel 0.
        let delays = solve_delays(3, &pairs, 1);
        assert!((delays[0].unwrap() + 11.0).abs() < 1e-9);
    }
}
