//! The discrete wavelet transform: orthonormal Daubechies filter banks with
//! periodic extension at the ends.

/// A wavelet family, by its low-pass decomposition filter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wavelet {
    /// The step-shaped wavelet: two taps, sharp in time, leaky in frequency.
    Haar,
    /// Daubechies with two vanishing moments (four taps, "D4").
    Daubechies2,
    /// Daubechies with three vanishing moments (six taps).
    Daubechies3,
    /// Daubechies with four vanishing moments (eight taps).
    #[default]
    Daubechies4,
}

impl Wavelet {
    pub const ALL: [Self; 4] = [Self::Haar, Self::Daubechies2, Self::Daubechies3, Self::Daubechies4];

    pub fn label(self) -> &'static str {
        match self {
            Self::Haar => "Haar",
            Self::Daubechies2 => "Db2",
            Self::Daubechies3 => "Db3",
            Self::Daubechies4 => "Db4",
        }
    }

    fn lowpass(self) -> &'static [f64] {
        match self {
            Self::Haar => &[core::f64::consts::FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2],
            Self::Daubechies2 => &[
                0.482_962_913_144_690_25,
                0.836_516_303_737_469,
                0.224_143_868_041_857_35,
                -0.129_409_522_550_921_45,
            ],
            Self::Daubechies3 => &[
                0.332_670_552_950_956_9,
                0.806_891_509_313_338_8,
                0.459_877_502_119_331_3,
                -0.135_011_020_010_390_84,
                -0.085_441_273_882_241_49,
                0.035_226_291_882_100_656,
            ],
            Self::Daubechies4 => &[
                0.230_377_813_308_855_23,
                0.714_846_570_552_541_5,
                0.630_880_767_929_590_4,
                -0.027_983_769_416_983_85,
                -0.187_034_811_718_881_14,
                0.030_841_381_835_986_965,
                0.032_883_011_666_982_945,
                -0.010_597_401_784_997_278,
            ],
        }
    }

    /// The matching high-pass filter: the low-pass reversed with alternating sign.
    fn highpass(self) -> Vec<f64> {
        let h = self.lowpass();
        (0..h.len())
            .map(|k| if k % 2 == 0 { h[h.len() - 1 - k] } else { -h[h.len() - 1 - k] })
            .collect()
    }

    /// How many levels a signal of `len` samples can be split into before a
    /// level is shorter than the filter.
    pub fn max_levels(self, len: usize) -> usize {
        let taps = self.lowpass().len().max(2);
        let mut len = len;
        let mut levels = 0;
        while len >= taps {
            len = len.div_ceil(2);
            levels += 1;
        }
        levels
    }
}

/// A multi-level decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct Decomposition {
    /// The coarsest approximation.
    pub approximation: Vec<f64>,
    /// Detail coefficients, finest (level 1) first.
    pub details: Vec<Vec<f64>>,
    /// The signal length entering each level, before padding; used to trim the
    /// padding back off on reconstruction.
    lengths: Vec<usize>,
    family: Wavelet,
}

impl Decomposition {
    pub fn levels(&self) -> usize {
        self.details.len()
    }
}

/// Splits `signal` into `levels` detail bands and a final approximation.
/// Odd-length levels are padded by repeating their last sample.
pub fn decompose(signal: &[f64], family: Wavelet, levels: usize) -> Decomposition {
    let h = family.lowpass();
    let g = family.highpass();
    let mut current = signal.to_vec();
    let mut details = Vec::with_capacity(levels);
    let mut lengths = Vec::with_capacity(levels);

    for _ in 0..levels {
        lengths.push(current.len());
        if current.len() % 2 == 1 {
            current.push(*current.last().expect("a level is never empty"));
        }
        let len = current.len();
        let half = len / 2;
        let mut approximation = vec![0.0; half];
        let mut detail = vec![0.0; half];
        for i in 0..half {
            for k in 0..h.len() {
                let sample = current[(2 * i + k) % len];
                approximation[i] += h[k] * sample;
                detail[i] += g[k] * sample;
            }
        }
        details.push(detail);
        current = approximation;
    }

    Decomposition { approximation: current, details, lengths, family }
}

/// Inverts [`decompose`].
pub fn reconstruct(decomposition: &Decomposition) -> Vec<f64> {
    let h = decomposition.family.lowpass();
    let g = decomposition.family.highpass();
    let mut current = decomposition.approximation.clone();

    for level in (0..decomposition.levels()).rev() {
        let detail = &decomposition.details[level];
        let len = 2 * current.len();
        let mut next = vec![0.0; len];
        for i in 0..current.len() {
            for k in 0..h.len() {
                next[(2 * i + k) % len] += h[k] * current[i] + g[k] * detail[i];
            }
        }
        next.truncate(decomposition.lengths[level]);
        current = next;
    }

    current
}

/// The multiresolution analysis: one signal-length band per level, each the
/// reconstruction from that level's coefficients alone. The bands sum to the
/// original signal. Ordered coarsest approximation first, then details from
/// coarsest to finest.
pub fn bands(decomposition: &Decomposition) -> Vec<Vec<f64>> {
    let zeroed = Decomposition {
        approximation: vec![0.0; decomposition.approximation.len()],
        details: decomposition.details.iter().map(|d| vec![0.0; d.len()]).collect(),
        ..decomposition.clone()
    };

    let mut bands = Vec::with_capacity(decomposition.levels() + 1);

    let mut only_approximation = zeroed.clone();
    only_approximation.approximation = decomposition.approximation.clone();
    bands.push(reconstruct(&only_approximation));

    for level in (0..decomposition.levels()).rev() {
        let mut only_detail = zeroed.clone();
        only_detail.details[level] = decomposition.details[level].clone();
        bands.push(reconstruct(&only_detail));
    }

    bands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0 + (i as f64 * 0.05).cos() + i as f64 * 0.01).collect()
    }

    #[test]
    fn haar_splits_a_constant_into_pure_approximation() {
        let d = decompose(&[1.0, 1.0, 1.0, 1.0], Wavelet::Haar, 1);
        let root2 = 2f64.sqrt();
        assert!(d.approximation.iter().all(|a| (a - root2).abs() < 1e-12), "{d:?}");
        assert!(d.details[0].iter().all(|x| x.abs() < 1e-12), "{d:?}");
    }

    #[test]
    fn every_family_is_orthonormal() {
        // Energy is preserved by an orthonormal transform.
        let x = signal(256);
        let energy: f64 = x.iter().map(|v| v * v).sum();
        for family in Wavelet::ALL {
            let d = decompose(&x, family, 3);
            let coefficients: f64 = d.approximation.iter().chain(d.details.iter().flatten()).map(|v| v * v).sum();
            assert!((coefficients - energy).abs() < 1e-8 * energy, "{family:?}");
        }
    }

    #[test]
    fn reconstruction_is_exact_even_with_odd_lengths() {
        for n in [64, 1000, 999, 37] {
            let x = signal(n);
            for family in Wavelet::ALL {
                let levels = family.max_levels(n);
                let back = reconstruct(&decompose(&x, family, levels));
                assert_eq!(back.len(), n);
                for (a, b) in back.iter().zip(&x) {
                    assert!((a - b).abs() < 1e-9, "{family:?} n={n}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn the_bands_sum_to_the_signal() {
        let x = signal(301);
        let d = decompose(&x, Wavelet::Daubechies2, 4);
        let bands = bands(&d);
        assert_eq!(bands.len(), 5);
        for i in 0..x.len() {
            let sum: f64 = bands.iter().map(|band| band[i]).sum();
            assert!((sum - x[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn daubechies_details_vanish_on_a_ramp() {
        // Two vanishing moments: the detail filter cannot see a straight line.
        // Only coefficients whose support wraps around the end are non-zero.
        let ramp: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let d = decompose(&ramp, Wavelet::Daubechies2, 1);
        let interior = &d.details[0][..d.details[0].len() - 2];
        assert!(interior.iter().all(|x| x.abs() < 1e-9), "{interior:?}");
    }

    #[test]
    fn levels_stop_before_the_filter_outgrows_the_signal() {
        assert_eq!(Wavelet::Haar.max_levels(8), 3);
        assert_eq!(Wavelet::Daubechies4.max_levels(8), 1);
        assert_eq!(Wavelet::Daubechies4.max_levels(7), 0);
    }
}
