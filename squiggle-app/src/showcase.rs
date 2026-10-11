//! Demo scenes that show the transforms working on signals whose answers are
//! known in advance.
//!
//! Each scene builds its test signal, runs it through the same
//! [`Transform::apply`] the Transform menu uses, and lays the results out on
//! the plot beside markers for what they ought to be. Alongside the picture it
//! carries a list of numerical checks, so the claim that the transforms work
//! does not rest on squinting at curves.

use core::f64::consts::{FRAC_PI_4, PI, TAU};

use bevy::prelude::*;
use squiggle_core::{
    Component, Point, Rgba, Squiggle, Style, Transform, TransformOptions, Wavelet, WaveletOutput,
};

use crate::plot::{AddSquiggle, FitToData, Plot, PlotMode, RemoveSquiggle, SquiggleModel};

/// Both scenes sample 16 x units at 32 samples per unit. A power of two keeps
/// the wavelet levels free of padding, and with a Nyquist frequency of 16 the
/// time and frequency axes of the Fourier scene span the same range.
const SAMPLE_COUNT: usize = 512;
const INTERVAL: f64 = 1.0 / 32.0;

/// How close a computed value must come to the exact one. The arithmetic is
/// all double precision, so anything looser would hide a real error.
const TOLERANCE: f64 = 1e-9;

const SIGNAL_COLOR: Rgba = Rgba::new(0.55, 0.55, 0.55);
const RECONSTRUCTION_COLOR: Rgba = Rgba::new(1.0, 0.6, 0.1);
const SPECTRUM_COLOR: Rgba = Rgba::new(0.35, 0.7, 1.0);
const EXPECTED_COLOR: Rgba = Rgba::new(0.9, 0.2, 0.3);
const BAND_COLORS: [Rgba; 5] = [
    Rgba::new(0.35, 0.7, 1.0),
    Rgba::new(0.45, 0.85, 0.45),
    Rgba::new(0.95, 0.85, 0.3),
    Rgba::new(0.85, 0.5, 0.95),
    Rgba::new(1.0, 0.45, 0.45),
];

/// A demo scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scene {
    Fourier,
    Wavelet,
}

impl Scene {
    pub const ALL: [Self; 2] = [Self::Fourier, Self::Wavelet];

    pub fn label(self) -> &'static str {
        match self {
            Self::Fourier => "Fourier",
            Self::Wavelet => "Wavelet",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|scene| scene.label().eq_ignore_ascii_case(name))
    }

    pub fn build(self) -> Built {
        match self {
            Self::Fourier => fourier(),
            Self::Wavelet => wavelet(),
        }
    }
}

/// A scene ready to plot.
pub struct Built {
    pub title: &'static str,
    pub description: &'static str,
    pub squiggles: Vec<Squiggle>,
    /// What each colour on the plot means.
    pub legend: Vec<(Rgba, &'static str)>,
    pub checks: Vec<Check>,
}

/// One numerical claim about a transform's output, and whether it held.
pub struct Check {
    pub claim: String,
    pub measured: String,
    pub passed: bool,
}

impl Check {
    fn new(claim: impl Into<String>, measured: impl Into<String>, passed: bool) -> Self {
        Self { claim: claim.into(), measured: measured.into(), passed }
    }

    /// Passes when `error` is within [`TOLERANCE`].
    fn exact(claim: impl Into<String>, error: f64) -> Self {
        Self::new(claim, format!("error {error:.1e}"), error <= TOLERANCE)
    }
}

fn times() -> impl Iterator<Item = f64> {
    (0..SAMPLE_COUNT).map(|i| i as f64 * INTERVAL)
}

fn styled(name: &str, points: impl IntoIterator<Item = Point>, color: Rgba, width: f32) -> Squiggle {
    Squiggle::with_points(name, points).with_style(Style::new(color, width))
}

fn restyle(squiggle: Squiggle, color: Rgba, width: f32) -> Squiggle {
    squiggle.with_style(Style::new(color, width))
}

/// Moves a squiggle up by `offset`. The copy drops any attached spectrum,
/// which no longer describes the shifted points.
fn lifted(squiggle: &Squiggle, offset: f64) -> Squiggle {
    Squiggle::with_points(
        squiggle.name.clone(),
        squiggle.points().iter().map(|p| Point::new(p.x, p.y + offset)),
    )
    .with_style(squiggle.style)
}

fn max_error<'a>(a: impl IntoIterator<Item = &'a Point>, b: impl IntoIterator<Item = &'a Point>) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| (a.y - b.y).abs().max((a.x - b.x).abs())).fold(0.0, f64::max)
}

/// The y value a curve takes at `x`, which must be one of its sample positions.
fn at(squiggle: &Squiggle, x: f64) -> f64 {
    squiggle
        .points()
        .iter()
        .find(|p| (p.x - x).abs() < 1e-9)
        .unwrap_or_else(|| panic!("{} has no sample at {x}", squiggle.name))
        .y
}

fn transform(kind: Transform, source: &Squiggle, options: &TransformOptions) -> Vec<Squiggle> {
    kind.apply(source, options).unwrap_or_else(|error| panic!("{} failed: {error}", kind.label()))
}

/// A tone in the Fourier test signal.
struct Tone {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

const TONES: [Tone; 3] = [
    Tone { frequency: 1.0, amplitude: 3.0, phase: 0.0 },
    Tone { frequency: 3.0, amplitude: 1.5, phase: FRAC_PI_4 },
    Tone { frequency: 6.0, amplitude: 0.75, phase: 0.0 },
];

/// Where the signal sits above the spectrum, clear of its tallest peak.
const FOURIER_SIGNAL_OFFSET: f64 = 10.0;

/// Three sines at frequencies that land exactly on FFT bins, so the spectrum
/// has three clean spikes of known height and nothing anywhere else.
fn fourier() -> Built {
    let signal = Squiggle::with_points(
        "signal",
        times().map(|t| {
            let y = TONES
                .iter()
                .map(|tone| tone.amplitude * (TAU * tone.frequency * t + tone.phase).sin())
                .sum();
            Point::new(t, y)
        }),
    );

    let amplitude = TransformOptions::default();
    let phase = TransformOptions { component: Component::Phase, ..amplitude };

    let fft = transform(Transform::Fft, &signal, &amplitude).remove(0);
    let dft = transform(Transform::Dft, &signal, &amplitude).remove(0);
    let continuous = transform(Transform::Fourier, &signal, &amplitude).remove(0);
    let fft_phase = transform(Transform::Fft, &signal, &phase).remove(0);
    let inverse = transform(Transform::InverseFourier, &fft, &amplitude).remove(0);

    let mut checks = Vec::new();
    for tone in &TONES {
        let measured = at(&fft, tone.frequency);
        checks.push(Check::new(
            format!("FFT peak at {} Hz has amplitude {}", tone.frequency, tone.amplitude),
            format!("{measured:.6}"),
            (measured - tone.amplitude).abs() <= TOLERANCE,
        ));
    }

    let leakage = fft
        .points()
        .iter()
        .filter(|p| !TONES.iter().any(|tone| (p.x - tone.frequency).abs() < 1e-9))
        .map(|p| p.y)
        .fold(0.0, f64::max);
    checks.push(Check::new(
        "Every other bin is empty",
        format!("largest {leakage:.1e}"),
        leakage <= TOLERANCE,
    ));

    // sin(ωt + φ) is a cosine at phase φ − π/2.
    let tone = &TONES[1];
    let expected_phase = tone.phase - PI / 2.0;
    let measured_phase = at(&fft_phase, tone.frequency);
    checks.push(Check::new(
        format!("Phase at {} Hz is φ − π/2 = {expected_phase:.4} rad", tone.frequency),
        format!("{measured_phase:.6} rad"),
        (measured_phase - expected_phase).abs() <= TOLERANCE,
    ));

    checks.push(Check::exact("DFT and FFT agree", max_error(dft.points(), fft.points())));

    // On even samples the continuous transform is the DFT scaled by the
    // interval, so a sine of amplitude A over T units reads A·T/2.
    let span = SAMPLE_COUNT as f64 * INTERVAL;
    let tone = &TONES[0];
    let density = at(&continuous, tone.frequency);
    let expected_density = tone.amplitude * span / 2.0;
    checks.push(Check::new(
        format!("Continuous |X({} Hz)| is A·T/2 = {expected_density}", tone.frequency),
        format!("{density:.6}"),
        (density - expected_density).abs() <= TOLERANCE,
    ));

    // Parseval: the mean square of the signal is the power in its spectrum.
    let mean_square = signal.points().iter().map(|p| p.y * p.y).sum::<f64>() / SAMPLE_COUNT as f64;
    let spectral_power: f64 = fft.points().iter().map(|p| p.y * p.y / 2.0).sum();
    checks.push(Check::exact(
        format!("Parseval: mean square {mean_square:.4} equals spectral power"),
        (mean_square - spectral_power).abs(),
    ));

    checks.push(Check::exact(
        "Inverse FFT reproduces the signal",
        max_error(inverse.points(), signal.points()),
    ));

    // A stem and crossbar at each expected peak, wide enough to show round the
    // spectrum drawn over it.
    let markers = TONES.iter().map(|tone| {
        let (f, a) = (tone.frequency, tone.amplitude);
        styled(
            &format!("expected {f} Hz"),
            [Point::new(f, 0.0), Point::new(f, a), Point::new(f - 0.25, a), Point::new(f + 0.25, a)],
            EXPECTED_COLOR,
            6.0,
        )
    });

    // Draw order is add order: the reconstruction goes over the signal it
    // should cover exactly, and the spectrum over its markers.
    let squiggles = markers
        .chain([
            restyle(fft, SPECTRUM_COLOR, 2.0),
            restyle(lifted(&signal, FOURIER_SIGNAL_OFFSET), SIGNAL_COLOR, 5.0),
            restyle(lifted(&inverse, FOURIER_SIGNAL_OFFSET), RECONSTRUCTION_COLOR, 1.5),
        ])
        .collect();

    Built {
        title: "Fourier transform",
        description: "Top: 3·sin(2π·t) + 1.5·sin(2π·3t + π/4) + 0.75·sin(2π·6t), 512 samples over 16 s. \
                      Bottom: its one-sided FFT amplitude spectrum, x in Hz.",
        squiggles,
        legend: vec![
            (SIGNAL_COLOR, "Signal (t in s)"),
            (RECONSTRUCTION_COLOR, "Inverse FFT of the spectrum"),
            (SPECTRUM_COLOR, "FFT amplitude (f in Hz)"),
            (EXPECTED_COLOR, "Expected peaks"),
        ],
        checks,
    }
}

const WAVELET: Wavelet = Wavelet::Daubechies4;
const WAVELET_LEVELS: usize = 4;

/// A Hann-tapered tone between `from` and `to`, zero elsewhere. The taper
/// keeps the burst's edges from splashing into every band.
fn burst(t: f64, from: f64, to: f64, frequency: f64) -> f64 {
    if !(from..to).contains(&t) {
        return 0.0;
    }
    let envelope = 0.5 - 0.5 * (TAU * (t - from) / (to - from)).cos();
    envelope * (TAU * frequency * t).sin()
}

/// Where each feature of the wavelet signal lives.
const SLOW_BURST: (f64, f64) = (2.0, 5.0);
const FAST_BURST: (f64, f64) = (8.0, 11.0);
/// A one-sample click, rather than a step: the transform wraps the signal
/// round periodically, and a step would add a second edge at the ends.
const CLICK_AT: usize = 426;
const CLICK_HEIGHT: f64 = 2.0;

/// A slow swell, two bursts and a click. At 32 samples per unit each detail
/// level covers an octave — D1 8–16 Hz, D2 4–8, D3 2–4, D4 1–2 — and the
/// approximation A4 everything under 1 Hz, so each feature should turn up in
/// its own band, at its own time.
fn wavelet() -> Built {
    let signal = Squiggle::with_points(
        "signal",
        times().enumerate().map(|(i, t)| {
            // Exactly two periods, so the swell meets itself at the ends.
            let swell = 1.5 * (TAU * 0.125 * t).sin();
            let click = if i == CLICK_AT { CLICK_HEIGHT } else { 0.0 };
            let y = swell + burst(t, SLOW_BURST.0, SLOW_BURST.1, 3.0) + burst(t, FAST_BURST.0, FAST_BURST.1, 12.0)
                + click;
            Point::new(t, y)
        }),
    );

    let flat = TransformOptions { stack: false, ..TransformOptions::default() };
    let stacked = TransformOptions::default();
    let kind = |output| Transform::Wavelet { family: WAVELET, levels: WAVELET_LEVELS, output };

    // Coarsest first: A4, D4, D3, D2, D1.
    let bands = transform(kind(WaveletOutput::Bands), &signal, &flat);
    let coefficients = transform(kind(WaveletOutput::Coefficients), &signal, &flat);
    let display = transform(kind(WaveletOutput::Bands), &signal, &stacked);

    let mut checks = Vec::new();
    checks.push(Check::new(
        format!("{} splits into A{WAVELET_LEVELS} and D{WAVELET_LEVELS}…D1", WAVELET.label()),
        format!("{} bands", bands.len()),
        bands.len() == WAVELET_LEVELS + 1,
    ));

    let sum: Vec<Point> = times()
        .enumerate()
        .map(|(i, t)| Point::new(t, bands.iter().map(|band| band.points()[i].y).sum()))
        .collect();
    checks.push(Check::exact("The bands add up to the signal", max_error(&sum, signal.points())));

    // An orthonormal transform keeps the signal's energy.
    let energy = |points: &[Point]| points.iter().map(|p| p.y * p.y).sum::<f64>();
    let signal_energy = energy(signal.points());
    let coefficient_energy: f64 = coefficients.iter().map(|level| energy(level.points())).sum();
    checks.push(Check::exact(
        format!("Coefficients keep the signal's energy, {signal_energy:.3}"),
        (signal_energy - coefficient_energy).abs() / signal_energy,
    ));

    // Each burst should land in its own band at its own time. The click is
    // broadband and turns up in every band, so it is the one other place a
    // band's energy may go.
    let click_at = CLICK_AT as f64 * INTERVAL;
    let click_window = (click_at - 0.5, click_at + 0.5);
    let share = |band: &Squiggle, windows: &[(f64, f64)]| {
        let inside = band
            .points()
            .iter()
            .filter(|p| windows.iter().any(|(from, to)| (*from..*to).contains(&p.x)))
            .map(|p| p.y * p.y)
            .sum::<f64>();
        inside / energy(band.points())
    };
    for (level, window, frequency) in [(1, FAST_BURST, 12.0), (3, SLOW_BURST, 3.0)] {
        let band = &bands[WAVELET_LEVELS + 1 - level];
        let burst = share(band, &[window]);
        let accounted = share(band, &[window, click_window]);
        checks.push(Check::new(
            format!("D{level} holds the {frequency} Hz burst at {}–{} s, and the click", window.0, window.1),
            format!("{:.1}% burst + {:.1}% click", burst * 100.0, (accounted - burst) * 100.0),
            burst >= 0.5 && accounted >= 0.98,
        ));
    }

    // The click is the one feature every detail band should see, and see in
    // the same place.
    let finest = (1..=WAVELET_LEVELS).map(|level| &bands[WAVELET_LEVELS + 1 - level]);
    let peaks: Vec<f64> = finest
        .map(|band| {
            band.points()
                .iter()
                .filter(|p| !(FAST_BURST.0..FAST_BURST.1).contains(&p.x) && !(SLOW_BURST.0..SLOW_BURST.1).contains(&p.x))
                .max_by(|a, b| a.y.abs().total_cmp(&b.y.abs()))
                .map(|p| p.x)
                .unwrap_or(f64::NAN)
        })
        .collect();
    let located = peaks.iter().all(|x| (click_window.0..click_window.1).contains(x));
    checks.push(Check::new(
        format!("Outside the bursts, every detail band peaks at the click, t = {click_at}"),
        peaks.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(", "),
        located,
    ));

    // Put the signal and its reconstruction above the stacked bands.
    let top = display
        .iter()
        .flat_map(|band| band.points())
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let low = signal.points().iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let offset = top - low + 1.0;
    let sum = Squiggle::with_points("sum of bands", sum);

    let squiggles = display
        .into_iter()
        .zip(BAND_COLORS)
        .map(|(band, color)| restyle(band, color, 1.5))
        .chain([
            restyle(lifted(&signal, offset), SIGNAL_COLOR, 5.0),
            restyle(lifted(&sum, offset), RECONSTRUCTION_COLOR, 1.5),
        ])
        .collect();

    Built {
        title: "Wavelet transform",
        description: "Top: a 0.125 Hz swell, a 3 Hz burst at 2–5 s, a 12 Hz burst at 8–11 s and a click at 13.31 s. \
                      Below: its Db4 bands, coarsest at the bottom, stacked.",
        squiggles,
        legend: vec![
            (SIGNAL_COLOR, "Signal"),
            (RECONSTRUCTION_COLOR, "Sum of the bands"),
            (BAND_COLORS[0], "A4: under 1 Hz"),
            (BAND_COLORS[1], "D4: 1–2 Hz"),
            (BAND_COLORS[2], "D3: 2–4 Hz"),
            (BAND_COLORS[3], "D2: 4–8 Hz"),
            (BAND_COLORS[4], "D1: 8–16 Hz"),
        ],
        checks,
    }
}

/// Ask for a scene to replace whatever is on the plot.
#[derive(Message)]
pub struct LoadScene(pub Scene);

/// The scene on screen, if any, and what the demo panel shows about it.
#[derive(Resource, Default)]
pub struct Showcase {
    pub current: Option<Loaded>,
    /// Squiggles still to arrive before the view can be framed on them.
    awaiting_fit: Option<usize>,
}

pub struct Loaded {
    pub scene: Scene,
    pub title: &'static str,
    pub description: &'static str,
    pub legend: Vec<(Rgba, &'static str)>,
    pub checks: Vec<Check>,
}

impl Loaded {
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}

pub struct ShowcasePlugin;

impl Plugin for ShowcasePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Showcase>()
            .add_message::<LoadScene>()
            .add_systems(Update, (load_scene, fit_when_ready).chain());
    }
}

fn load_scene(
    mut requests: MessageReader<LoadScene>,
    mut showcase: ResMut<Showcase>,
    mut plot: ResMut<Plot>,
    mut mode: ResMut<PlotMode>,
    existing: Query<Entity, With<SquiggleModel>>,
    mut add: MessageWriter<AddSquiggle>,
    mut remove: MessageWriter<RemoveSquiggle>,
) {
    let Some(LoadScene(scene)) = requests.read().last() else {
        return;
    };

    for entity in &existing {
        remove.write(RemoveSquiggle(entity));
    }
    *mode = PlotMode::Overlay;

    let built = scene.build();
    for check in &built.checks {
        let verdict = if check.passed { "pass" } else { "FAIL" };
        info!("{} demo: [{verdict}] {} — {}", scene.label(), check.claim, check.measured);
    }

    showcase.awaiting_fit = Some(built.squiggles.len());
    for squiggle in built.squiggles {
        add.write(AddSquiggle(squiggle));
    }

    plot.name = built.title.to_string();
    showcase.current = Some(Loaded {
        scene: *scene,
        title: built.title,
        description: built.description,
        legend: built.legend,
        checks: built.checks,
    });
}

/// Frames the scene once its squiggles are all on the plot and the plot has
/// been given room to draw. At startup neither is true on the first frame,
/// and a fit asked for too early is dropped.
fn fit_when_ready(
    mut showcase: ResMut<Showcase>,
    plot: Res<Plot>,
    squiggles: Query<(), With<SquiggleModel>>,
    mut fit: MessageWriter<FitToData>,
) {
    let Some(expected) = showcase.awaiting_fit else {
        return;
    };
    if plot.is_usable() && squiggles.iter().count() == expected {
        fit.write(FitToData);
        showcase.awaiting_fit = None;
    }
}

/// Whether the scene has been framed, which is when it is worth a screenshot.
pub fn settled(showcase: &Showcase) -> bool {
    showcase.current.is_some() && showcase.awaiting_fit.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_pass(scene: Scene) {
        let built = scene.build();
        let failures: Vec<String> = built
            .checks
            .iter()
            .filter(|check| !check.passed)
            .map(|check| format!("{} — {}", check.claim, check.measured))
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
        assert!(!built.squiggles.is_empty());
    }

    #[test]
    fn every_fourier_check_passes() {
        assert_all_pass(Scene::Fourier);
    }

    #[test]
    fn every_wavelet_check_passes() {
        assert_all_pass(Scene::Wavelet);
    }

    #[test]
    fn scenes_parse_by_name_in_any_case() {
        assert_eq!(Scene::parse("fourier"), Some(Scene::Fourier));
        assert_eq!(Scene::parse("WAVELET"), Some(Scene::Wavelet));
        assert_eq!(Scene::parse("laplace"), None);
    }

    #[test]
    fn the_fourier_signal_clears_the_spectrum() {
        let built = Scene::Fourier.build();
        let signal = built.squiggles.iter().find(|s| s.name == "signal").unwrap();
        let spectrum = built.squiggles.iter().find(|s| s.name.starts_with("FFT")).unwrap();
        assert!(signal.bounds().unwrap().min_y > spectrum.bounds().unwrap().max_y);
    }
}
