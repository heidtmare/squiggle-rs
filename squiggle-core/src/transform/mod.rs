//! Transforms that turn one squiggle into others: Fourier spectra, a
//! short-time spectrogram, the inverse back to a signal, and wavelet bands.
//!
//! Everything here treats a squiggle as a real-valued signal `y(x)`. The DFT,
//! FFT, STFT and wavelets assume the samples are evenly spaced and take the
//! interval from the first and last x; the continuous Fourier transform uses
//! each sample's actual position.

mod complex;
mod fft;
mod spectrum;
mod wavelet;

use core::f64::consts::TAU;
use core::fmt;

pub use complex::Complex;
pub use spectrum::{Basis, Spectrum};
pub use wavelet::Wavelet;

use crate::squiggle::{Point, Squiggle};

/// Amplitudes below this read as this in decibels, so silent bins sit at
/// -240 dB instead of negative infinity.
const DECIBEL_FLOOR: f64 = 1e-12;

/// One way of transforming a squiggle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    /// The continuous Fourier integral, approximated at the samples' real positions.
    Fourier,
    /// The discrete Fourier transform, summed directly: O(N²).
    Dft,
    /// The same transform by FFT: O(N log N), at any length.
    Fft,
    /// Spectra of successive windowed frames.
    Stft { window: Window, size: usize, hop: usize },
    /// Back from a spectrum to a signal.
    InverseFourier,
    /// A multi-level discrete wavelet transform.
    Wavelet { family: Wavelet, levels: usize, output: WaveletOutput },
}

/// What to plot from a spectrum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Component {
    /// In the signal's units: a sine of amplitude A shows as a peak of height A.
    /// For the continuous transform this is `|X(f)|`, a density.
    #[default]
    Amplitude,
    /// Amplitude squared.
    Power,
    /// `20·log10(amplitude)`.
    Decibels,
    /// Phase in radians.
    Phase,
}

impl Component {
    pub const ALL: [Self; 4] = [Self::Amplitude, Self::Power, Self::Decibels, Self::Phase];

    pub fn label(self) -> &'static str {
        match self {
            Self::Amplitude => "Amplitude",
            Self::Power => "Power",
            Self::Decibels => "Decibels",
            Self::Phase => "Phase",
        }
    }
}

/// The taper applied to each STFT frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Window {
    Rectangular,
    #[default]
    Hann,
    Hamming,
    Blackman,
}

impl Window {
    pub const ALL: [Self; 4] = [Self::Rectangular, Self::Hann, Self::Hamming, Self::Blackman];

    pub fn label(self) -> &'static str {
        match self {
            Self::Rectangular => "Rectangular",
            Self::Hann => "Hann",
            Self::Hamming => "Hamming",
            Self::Blackman => "Blackman",
        }
    }

    /// The periodic form of the window, which is what spectral analysis wants.
    fn coefficients(self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|j| {
                let phase = TAU * j as f64 / size as f64;
                match self {
                    Self::Rectangular => 1.0,
                    Self::Hann => 0.5 - 0.5 * phase.cos(),
                    Self::Hamming => 0.54 - 0.46 * phase.cos(),
                    Self::Blackman => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
                }
            })
            .collect()
    }
}

/// What a wavelet transform plots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaveletOutput {
    /// One signal-length band per level, in the signal's units and aligned
    /// with it. The bands add up to the original.
    #[default]
    Bands,
    /// The raw coefficients of each level, placed at the centre of the stretch
    /// of signal each one summarises.
    Coefficients,
}

impl WaveletOutput {
    pub const ALL: [Self; 2] = [Self::Bands, Self::Coefficients];

    pub fn label(self) -> &'static str {
        match self {
            Self::Bands => "Bands",
            Self::Coefficients => "Coefficients",
        }
    }
}

/// Options shared by every transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformOptions {
    pub component: Component,
    /// Plot only non-negative frequencies, folding the negative half in.
    pub one_sided: bool,
    /// Offset multi-squiggle results (STFT frames, wavelet levels) vertically
    /// so they do not overlap.
    pub stack: bool,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self { component: Component::default(), one_sided: true, stack: true }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformError {
    TooFewSamples { needed: usize, found: usize },
    /// Every sample has the same x, so there is no interval to measure.
    ZeroSpan,
    /// The inverse was asked of something that does not read as a spectrum.
    NotASpectrum,
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSamples { needed, found } => {
                write!(f, "needs at least {needed} samples, has {found}")
            }
            Self::ZeroSpan => write!(f, "all samples share one x value"),
            Self::NotASpectrum => write!(f, "does not look like a spectrum (no positive frequencies)"),
        }
    }
}

impl std::error::Error for TransformError {}

impl Transform {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fourier => "Fourier",
            Self::Dft => "DFT",
            Self::Fft => "FFT",
            Self::Stft { .. } => "STFT",
            Self::InverseFourier => "Inverse Fourier",
            Self::Wavelet { .. } => "Wavelet",
        }
    }

    /// Applies the transform to `source`. Most transforms yield one squiggle;
    /// the STFT yields one per frame and wavelets one per level.
    pub fn apply(
        &self,
        source: &Squiggle,
        options: &TransformOptions,
    ) -> Result<Vec<Squiggle>, TransformError> {
        let output = |label: String, points: Vec<Point>| {
            Squiggle::with_points(format!("{label}({})", source.name), points).with_style(source.style)
        };

        if let Self::InverseFourier = self {
            let (times, values) = match source.spectrum() {
                Some(spectrum) => spectrum.invert(),
                None => synthesize(source.points())?,
            };
            return Ok(vec![output("IFT".into(), zip(&times, &values))]);
        }

        let (times, values) = samples(source);
        let found = times.len();
        let needed = 2;
        if found < needed {
            return Err(TransformError::TooFewSamples { needed, found });
        }
        let start = times[0];
        let interval = (times[found - 1] - start) / (found - 1) as f64;
        if !(interval.is_finite() && interval > 0.0) {
            return Err(TransformError::ZeroSpan);
        }

        match *self {
            Self::Fourier | Self::Dft | Self::Fft => {
                let (spectrum, norm) = match self {
                    Self::Fourier => (spectrum::continuous(&times, &values), None),
                    _ => (
                        spectrum::discrete(&values, start, interval, *self == Self::Fft),
                        Some(found as f64),
                    ),
                };
                let points = spectrum_points(&spectrum, norm, options);
                Ok(vec![output(self.label().into(), points).with_spectrum(spectrum)])
            }
            Self::Stft { window, size, hop } => {
                let frames = stft(&values, start, interval, window, size, hop)?;
                let mut curves: Vec<_> = frames
                    .iter()
                    .map(|(spectrum, gain)| spectrum_points(spectrum, Some(*gain), options))
                    .collect();
                if options.stack {
                    stack(&mut curves);
                }
                Ok(frames
                    .into_iter()
                    .zip(curves)
                    .map(|((spectrum, _), points)| {
                        let Basis::Discrete { start: frame_start, .. } = &spectrum.basis else {
                            unreachable!("STFT frames are discrete spectra")
                        };
                        let centre = *frame_start + (spectrum.sample_count - 1) as f64 * interval / 2.0;
                        output(format!("STFT t={centre:.4}"), points).with_spectrum(spectrum)
                    })
                    .collect())
            }
            Self::Wavelet { family, levels, output: kind } => {
                let available = family.max_levels(found);
                if available == 0 {
                    return Err(TransformError::TooFewSamples {
                        needed: family_taps(family),
                        found,
                    });
                }
                let levels = levels.clamp(1, available);
                let decomposition = wavelet::decompose(&values, family, levels);

                let names = core::iter::once(format!("A{levels}"))
                    .chain((1..=levels).rev().map(|level| format!("D{level}")));

                let mut curves: Vec<Vec<Point>> = match kind {
                    WaveletOutput::Bands => {
                        wavelet::bands(&decomposition).iter().map(|band| zip(&times, band)).collect()
                    }
                    WaveletOutput::Coefficients => {
                        let place = |level: usize, coefficients: &[f64]| {
                            let width = (1usize << level) as f64;
                            coefficients
                                .iter()
                                .enumerate()
                                .map(|(i, &c)| {
                                    Point::new(start + (i as f64 * width + (width - 1.0) / 2.0) * interval, c)
                                })
                                .collect()
                        };
                        core::iter::once(place(levels, &decomposition.approximation))
                            .chain(
                                (1..=levels)
                                    .rev()
                                    .map(|level| place(level, &decomposition.details[level - 1])),
                            )
                            .collect()
                    }
                };
                if options.stack {
                    stack(&mut curves);
                }

                Ok(names
                    .zip(curves)
                    .map(|(band, points)| output(format!("{} {band}", family.label()), points))
                    .collect())
            }
            Self::InverseFourier => unreachable!("handled above"),
        }
    }
}

fn family_taps(family: Wavelet) -> usize {
    (1..).find(|&n| family.max_levels(n) > 0).expect("some length always fits")
}

/// The squiggle's finite samples as parallel x and y arrays.
fn samples(source: &Squiggle) -> (Vec<f64>, Vec<f64>) {
    source
        .points()
        .iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite())
        .map(|p| (p.x, p.y))
        .unzip()
}

fn zip(xs: &[f64], ys: &[f64]) -> Vec<Point> {
    xs.iter().zip(ys).map(|(&x, &y)| Point::new(x, y)).collect()
}

/// Plots a spectrum. `norm` turns a discrete coefficient into an amplitude
/// (the sample count, or a window's gain); `None` marks a continuous density,
/// which is plotted as-is and not folded.
fn spectrum_points(spectrum: &Spectrum, norm: Option<f64>, options: &TransformOptions) -> Vec<Point> {
    let n = spectrum.sample_count as isize;
    let df = spectrum.resolution();

    let point = |k: isize| {
        let c = spectrum.bin(k);
        let amplitude = match norm {
            Some(norm) if options.one_sided => c.norm() / norm * spectrum.multiplicity(k.unsigned_abs()),
            Some(norm) => c.norm() / norm,
            None => c.norm(),
        };
        let y = match options.component {
            Component::Amplitude => amplitude,
            Component::Power => amplitude * amplitude,
            Component::Decibels => 20.0 * amplitude.max(DECIBEL_FLOOR).log10(),
            Component::Phase => c.arg(),
        };
        Point::new(k as f64 * df, y)
    };

    if options.one_sided {
        (0..=n / 2).map(point).collect()
    } else {
        (-(n - 1) / 2..=n / 2).map(point).collect()
    }
}

/// Spectra of overlapping windowed frames, each with its window's gain.
fn stft(
    values: &[f64],
    start: f64,
    interval: f64,
    window: Window,
    size: usize,
    hop: usize,
) -> Result<Vec<(Spectrum, f64)>, TransformError> {
    let size = size.min(values.len());
    if size < 2 {
        return Err(TransformError::TooFewSamples { needed: 2, found: values.len() });
    }
    let taper = window.coefficients(size);
    let gain: f64 = taper.iter().sum();

    Ok((0..=values.len() - size)
        .step_by(hop.max(1))
        .map(|offset| {
            let frame: Vec<f64> =
                values[offset..offset + size].iter().zip(&taper).map(|(v, w)| v * w).collect();
            let spectrum = spectrum::discrete(&frame, start + offset as f64 * interval, interval, true);
            (spectrum, gain)
        })
        .collect())
}

/// Lifts each curve above the one before it, by the tallest curve's height.
fn stack(curves: &mut [Vec<Point>]) {
    let extent = |curve: &Vec<Point>| {
        let (low, high) = curve
            .iter()
            .filter(|p| p.y.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
        if low <= high { high - low } else { 0.0 }
    };
    let tallest = curves.iter().map(extent).fold(0.0, f64::max);
    let spacing = if tallest > 0.0 { tallest * 1.1 } else { 1.0 };

    for (i, curve) in curves.iter_mut().enumerate() {
        for point in curve {
            point.y += i as f64 * spacing;
        }
    }
}

/// The inverse for a squiggle that carries no spectrum: each point `(f, a)`
/// is read as a cosine of frequency `f` and amplitude `a`, which is what a
/// one-sided amplitude plot means. A two-sided plot works too, since each half
/// carries half the amplitude.
///
/// The result spans one period of the lowest frequency step, sampled at twice
/// the highest frequency.
fn synthesize(points: &[Point]) -> Result<(Vec<f64>, Vec<f64>), TransformError> {
    let terms: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite())
        .map(|p| (p.x, p.y))
        .collect();

    let highest = terms.iter().map(|(f, _)| f.abs()).fold(0.0, f64::max);
    let mut positive: Vec<f64> = terms.iter().map(|(f, _)| f.abs()).filter(|f| *f > 0.0).collect();
    positive.sort_by(f64::total_cmp);
    positive.dedup();
    if highest <= 0.0 {
        return Err(TransformError::NotASpectrum);
    }

    // The bin spacing: the smallest gap between distinct frequencies, which
    // for a single tone is the tone itself.
    let step = positive
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(positive[0], f64::min);

    let count = ((2.0 * highest / step).round() as usize).max(2);
    let dt = 1.0 / (count as f64 * step);

    let times: Vec<f64> = (0..count).map(|i| i as f64 * dt).collect();
    let values = times
        .iter()
        .map(|&t| terms.iter().map(|(f, a)| a * (TAU * f * t).cos()).sum())
        .collect();
    Ok((times, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone of `amplitude` at `cycles` per `n` samples, sampled every `interval`.
    fn tone(n: usize, cycles: f64, amplitude: f64, interval: f64) -> Squiggle {
        Squiggle::with_points(
            "tone",
            (0..n).map(|i| {
                Point::new(i as f64 * interval, amplitude * (TAU * cycles * i as f64 / n as f64).sin())
            }),
        )
    }

    fn peak(squiggle: &Squiggle) -> Point {
        *squiggle.points().iter().max_by(|a, b| a.y.total_cmp(&b.y)).expect("not empty")
    }

    #[test]
    fn the_fft_amplitude_spectrum_reads_in_signal_units() {
        // 8 cycles over 256 samples at interval 0.5: f = 8 / (256 · 0.5).
        let out = Transform::Fft.apply(&tone(256, 8.0, 3.0, 0.5), &TransformOptions::default()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "FFT(tone)");
        assert_eq!(out[0].len(), 129);
        let p = peak(&out[0]);
        assert!((p.x - 8.0 / 128.0).abs() < 1e-12, "{p:?}");
        assert!((p.y - 3.0).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn dft_and_fft_plot_the_same_spectrum() {
        let source = tone(100, 5.0, 1.0, 1.0);
        let options = TransformOptions::default();
        let slow = Transform::Dft.apply(&source, &options).unwrap();
        let fast = Transform::Fft.apply(&source, &options).unwrap();
        for (a, b) in slow[0].points().iter().zip(fast[0].points()) {
            assert!((a.y - b.y).abs() < 1e-9);
        }
    }

    #[test]
    fn a_two_sided_spectrum_splits_the_amplitude() {
        let options = TransformOptions { one_sided: false, ..TransformOptions::default() };
        let out = Transform::Fft.apply(&tone(64, 4.0, 2.0, 1.0), &options).unwrap();
        let points = out[0].points();
        assert_eq!(points.len(), 64);
        assert!(points.first().unwrap().x < 0.0);
        let at = |f: f64| points.iter().find(|p| (p.x - f).abs() < 1e-12).unwrap().y;
        assert!((at(4.0 / 64.0) - 1.0).abs() < 1e-9);
        assert!((at(-4.0 / 64.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn decibels_floor_silence() {
        let options = TransformOptions { component: Component::Decibels, ..TransformOptions::default() };
        let silence = Squiggle::with_points("s", (0..16).map(|i| Point::new(i as f64, 0.0)));
        let out = Transform::Fft.apply(&silence, &options).unwrap();
        assert!(out[0].points().iter().all(|p| (p.y + 240.0).abs() < 1e-9));
    }

    #[test]
    fn the_inverse_undoes_every_forward_transform() {
        let source = tone(90, 7.0, 1.5, 0.25);
        let options = TransformOptions { component: Component::Phase, ..TransformOptions::default() };
        for forward in [Transform::Fourier, Transform::Dft, Transform::Fft] {
            let spectrum = forward.apply(&source, &options).unwrap().remove(0);
            let back = Transform::InverseFourier.apply(&spectrum, &options).unwrap().remove(0);
            assert_eq!(back.len(), source.len(), "{forward:?}");
            for (a, b) in back.points().iter().zip(source.points()) {
                assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-8, "{forward:?}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn the_continuous_transform_honours_uneven_spacing() {
        // The same tone, sampled densely in its first half and sparsely in its
        // second, still peaks at its own frequency.
        let points = (0..200)
            .map(|i| i as f64 * 0.05)
            .chain((0..50).map(|i| 10.0 + i as f64 * 0.2))
            .map(|t| Point::new(t, (TAU * 2.0 * t).cos()));
        let source = Squiggle::with_points("uneven", points);
        let out = Transform::Fourier.apply(&source, &TransformOptions::default()).unwrap();
        let p = peak(&out[0]);
        assert!((p.x - 2.0).abs() < 0.1, "{p:?}");
        // A unit cosine over 20 x units has |X(2)| = 10; the grid misses 2
        // slightly, which costs a little of the peak.
        assert!(p.y > 8.0 && p.y < 10.5, "{p:?}");
    }

    #[test]
    fn the_inverse_of_a_plain_squiggle_synthesises_cosines() {
        let spectrum = Squiggle::with_points("s", [Point::new(0.0, 0.0), Point::new(0.25, 2.0)]);
        let out = Transform::InverseFourier.apply(&spectrum, &TransformOptions::default()).unwrap();
        let points = out[0].points();
        // One period of a 0.25 Hz cosine, sampled at its Nyquist rate.
        assert_eq!(points, &[Point::new(0.0, 2.0), Point::new(2.0, -2.0)]);
    }

    #[test]
    fn the_inverse_rejects_what_cannot_be_a_spectrum() {
        let flat = Squiggle::with_points("s", [Point::new(0.0, 1.0)]);
        assert_eq!(
            Transform::InverseFourier.apply(&flat, &TransformOptions::default()).err(),
            Some(TransformError::NotASpectrum)
        );
    }

    #[test]
    fn the_stft_follows_a_change_in_pitch() {
        // Low tone for the first half, high tone for the second.
        let n = 1024;
        let points = (0..n).map(|i| {
            let cycles = if i < n / 2 { 8.0 } else { 64.0 };
            Point::new(i as f64, (TAU * cycles * i as f64 / 256.0).sin())
        });
        let source = Squiggle::with_points("chirp", points);
        let transform = Transform::Stft { window: Window::Hann, size: 256, hop: 128 };
        let options = TransformOptions { stack: false, ..TransformOptions::default() };
        let frames = transform.apply(&source, &options).unwrap();

        assert_eq!(frames.len(), 7);
        assert_eq!(frames[0].name, "STFT t=127.5000(chirp)");
        let first = peak(&frames[0]);
        let last = peak(&frames[6]);
        assert!((first.x - 8.0 / 256.0).abs() < 1e-12, "{first:?}");
        assert!((last.x - 64.0 / 256.0).abs() < 1e-12, "{last:?}");
        // The window's gain is divided out, so the tone reads at amplitude 1.
        assert!((first.y - 1.0).abs() < 1e-9, "{first:?}");
    }

    #[test]
    fn stacked_results_do_not_overlap() {
        let transform = Transform::Stft { window: Window::Hann, size: 64, hop: 64 };
        let frames = transform.apply(&tone(256, 16.0, 1.0, 1.0), &TransformOptions::default()).unwrap();
        for pair in frames.windows(2) {
            let below = pair[0].bounds().unwrap();
            let above = pair[1].bounds().unwrap();
            assert!(above.min_y > below.max_y, "{below:?} {above:?}");
        }
    }

    #[test]
    fn wavelet_bands_rebuild_the_signal() {
        let source = tone(500, 9.0, 2.0, 0.1);
        let transform =
            Transform::Wavelet { family: Wavelet::Daubechies4, levels: 3, output: WaveletOutput::Bands };
        let options = TransformOptions { stack: false, ..TransformOptions::default() };
        let bands = transform.apply(&source, &options).unwrap();

        let names: Vec<_> = bands.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, ["Db4 A3(tone)", "Db4 D3(tone)", "Db4 D2(tone)", "Db4 D1(tone)"]);
        for i in 0..source.len() {
            let sum: f64 = bands.iter().map(|b| b.points()[i].y).sum();
            assert!((sum - source.points()[i].y).abs() < 1e-9);
            assert_eq!(bands[0].points()[i].x, source.points()[i].x);
        }
    }

    #[test]
    fn wavelet_coefficients_shrink_by_level_and_sit_under_their_samples() {
        let source = tone(64, 2.0, 1.0, 1.0);
        let transform = Transform::Wavelet { family: Wavelet::Haar, levels: 2, output: WaveletOutput::Coefficients };
        let options = TransformOptions { stack: false, ..TransformOptions::default() };
        let levels = transform.apply(&source, &options).unwrap();
        let lengths: Vec<_> = levels.iter().map(Squiggle::len).collect();
        assert_eq!(lengths, [16, 16, 32]);
        // A level-1 Haar coefficient covers samples 0 and 1.
        assert_eq!(levels[2].points()[0].x, 0.5);
    }

    #[test]
    fn wavelet_levels_are_clamped_to_what_fits() {
        let transform =
            Transform::Wavelet { family: Wavelet::Haar, levels: 50, output: WaveletOutput::Bands };
        let out = transform.apply(&tone(16, 1.0, 1.0, 1.0), &TransformOptions::default()).unwrap();
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn too_little_data_is_an_error() {
        let one = Squiggle::with_points("s", [Point::new(0.0, 1.0)]);
        assert_eq!(
            Transform::Fft.apply(&one, &TransformOptions::default()).err(),
            Some(TransformError::TooFewSamples { needed: 2, found: 1 })
        );
        let stacked = Squiggle::with_points("s", [Point::new(1.0, 1.0), Point::new(1.0, 2.0)]);
        assert_eq!(
            Transform::Dft.apply(&stacked, &TransformOptions::default()).err(),
            Some(TransformError::ZeroSpan)
        );
        let short = Squiggle::with_points("s", (0..4).map(|i| Point::new(i as f64, 0.0)));
        let db4 = Transform::Wavelet { family: Wavelet::Daubechies4, levels: 1, output: WaveletOutput::Bands };
        assert_eq!(
            db4.apply(&short, &TransformOptions::default()).err(),
            Some(TransformError::TooFewSamples { needed: 8, found: 4 })
        );
    }
}
