//! Butterworth IIR filters as second-order sections, plus detrend and taper.
//!
//! Design: analog prototype poles → band transform → bilinear transform with
//! prewarping → sections, normalised to unit gain in the passband. Sections
//! run in transposed direct form II with `f64` state.
//!
//! [`SosFilter`] is causal and keeps state across calls, so a stream filtered
//! chunk by chunk matches the whole file. [`filtfilt`] is zero-phase and
//! needs the whole run.

use core::f64::consts::PI;
use core::fmt;

use crate::transform::Complex;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Band {
    Lowpass { hz: f64 },
    Highpass { hz: f64 },
    Bandpass { low_hz: f64, high_hz: f64 },
    Bandstop { low_hz: f64, high_hz: f64 },
}

impl Band {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Lowpass { .. } => "Low-pass",
            Self::Highpass { .. } => "High-pass",
            Self::Bandpass { .. } => "Band-pass",
            Self::Bandstop { .. } => "Band-stop",
        }
    }

    /// Lowest corner, which sets how long the filter rings.
    fn lowest_hz(&self) -> f64 {
        match *self {
            Self::Lowpass { hz } | Self::Highpass { hz } => hz,
            Self::Bandpass { low_hz, .. } | Self::Bandstop { low_hz, .. } => low_hz,
        }
    }

    pub fn describe(&self) -> String {
        match *self {
            Self::Lowpass { hz } => format!("LP {hz} Hz"),
            Self::Highpass { hz } => format!("HP {hz} Hz"),
            Self::Bandpass { low_hz, high_hz } => format!("BP {low_hz}–{high_hz} Hz"),
            Self::Bandstop { low_hz, high_hz } => format!("BS {low_hz}–{high_hz} Hz"),
        }
    }
}

pub const MAX_ORDER: u8 = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FilterSpec {
    pub band: Band,
    /// Butterworth order, 1..=[`MAX_ORDER`]. Band filters have twice as many poles.
    pub order: u8,
    /// Forward-backward: no phase shift, squared magnitude response.
    pub zero_phase: bool,
}

impl FilterSpec {
    pub fn describe(&self) -> String {
        let phase = if self.zero_phase { "zero-phase" } else { "causal" };
        format!("{} order {} {phase}", self.band.describe(), self.order)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Detrend {
    #[default]
    None,
    Mean,
    Linear,
}

impl Detrend {
    pub const ALL: [Self; 3] = [Self::None, Self::Mean, Self::Linear];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Mean => "Remove mean",
            Self::Linear => "Remove linear trend",
        }
    }
}

/// Per-trace preprocessing: detrend, taper, filter, in that order. Runs on
/// each gap-free run separately; gaps stay NaN.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Processing {
    pub detrend: Detrend,
    /// Hann taper length at each end, as a fraction of the run (0..=0.5).
    pub taper: f64,
    pub filter: Option<FilterSpec>,
}

impl Processing {
    pub fn is_identity(&self) -> bool {
        self.detrend == Detrend::None && self.taper <= 0.0 && self.filter.is_none()
    }

    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.detrend != Detrend::None {
            parts.push(self.detrend.label().to_lowercase());
        }
        if self.taper > 0.0 {
            parts.push(format!("taper {:.0} %", self.taper * 100.0));
        }
        if let Some(filter) = &self.filter {
            parts.push(filter.describe());
        }
        if parts.is_empty() { "raw".to_string() } else { parts.join(", ") }
    }

    /// Checks the filter against `rate` without running it.
    pub fn validate(&self, rate: f64) -> Result<(), DesignError> {
        match &self.filter {
            Some(spec) => butterworth(spec.band, spec.order, rate).map(|_| ()),
            None => Ok(()),
        }
    }

    pub fn apply(&self, values: &mut [f32], rate: f64) -> Result<(), DesignError> {
        let sos: Option<(Sos, FilterSpec)> = self.filter.map(|spec| butterworth(spec.band, spec.order, rate).map(|s| (s, spec))).transpose()?;
        for run in finite_runs(values) {
            let run = &mut values[run];
            match self.detrend {
                Detrend::None => {}
                Detrend::Mean => remove_mean(run),
                Detrend::Linear => remove_trend(run),
            }
            taper(run, self.taper);
            if let Some((sos, spec)) = &sos {
                if spec.zero_phase {
                    let pad = pad_length(sos, spec.band.lowest_hz(), rate);
                    filtfilt(sos, run, pad);
                } else {
                    SosFilter::new(sos.clone()).process(run);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DesignError {
    Order(u8),
    /// Corner outside `(0, rate/2)`.
    Corner { hz: f64, nyquist: f64 },
    /// Band edges out of order.
    Edges,
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Order(order) => write!(f, "order {order} outside 1..={MAX_ORDER}"),
            Self::Corner { hz, nyquist } => write!(f, "corner {hz} Hz outside (0, {nyquist}) Hz"),
            Self::Edges => write!(f, "low edge must be below high edge"),
        }
    }
}

impl std::error::Error for DesignError {}

/// `H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    pub b: [f64; 3],
    pub a: [f64; 2],
}

impl Biquad {
    fn response(&self, w: f64) -> Complex {
        let z1 = Complex::cis(-w);
        let z2 = Complex::cis(-2.0 * w);
        let num = Complex::from(self.b[0]) + z1.scale(self.b[1]) + z2.scale(self.b[2]);
        let den = Complex::ONE + z1.scale(self.a[0]) + z2.scale(self.a[1]);
        num / den
    }

    fn dc_gain(&self) -> f64 {
        (self.b[0] + self.b[1] + self.b[2]) / (1.0 + self.a[0] + self.a[1])
    }
}

pub type Sos = Vec<Biquad>;

/// Complex magnitude response at `hz`.
pub fn response(sos: &[Biquad], hz: f64, rate: f64) -> f64 {
    let w = 2.0 * PI * hz / rate;
    sos.iter().map(|s| s.response(w).norm()).product()
}

pub fn butterworth(band: Band, order: u8, rate: f64) -> Result<Sos, DesignError> {
    if !(1..=MAX_ORDER).contains(&order) {
        return Err(DesignError::Order(order));
    }
    let nyquist = rate / 2.0;
    let check = |hz: f64| {
        if hz.is_finite() && hz > 0.0 && hz < nyquist { Ok(hz) } else { Err(DesignError::Corner { hz, nyquist }) }
    };
    let n = usize::from(order);
    let fs2 = 2.0 * rate;
    let warp = |hz: f64| fs2 * (PI * hz / rate).tan();
    let prototype: Vec<Complex> =
        (0..n).map(|k| Complex::cis(PI * (2 * k + n + 1) as f64 / (2 * n) as f64)).collect();

    // Analog zeros and poles, and the digital frequency (rad/sample) where gain is 1.
    let (zeros, poles, w_ref) = match band {
        Band::Lowpass { hz } => {
            let w = warp(check(hz)?);
            (Vec::new(), prototype.iter().map(|p| p.scale(w)).collect(), 0.0)
        }
        Band::Highpass { hz } => {
            let w = warp(check(hz)?);
            (vec![Complex::ZERO; n], prototype.iter().map(|p| Complex::from(w) / *p).collect(), PI)
        }
        Band::Bandpass { low_hz, high_hz } | Band::Bandstop { low_hz, high_hz } => {
            let (lo, hi) = (warp(check(low_hz)?), warp(check(high_hz)?));
            if lo >= hi {
                return Err(DesignError::Edges);
            }
            let (w0, bw) = ((lo * hi).sqrt(), hi - lo);
            let w0_sq = Complex::from(w0 * w0);
            let split = |a: Complex| {
                let root = (a * a - w0_sq).sqrt();
                [a + root, a - root]
            };
            if matches!(band, Band::Bandpass { .. }) {
                let poles: Vec<Complex> = prototype.iter().flat_map(|p| split(p.scale(bw / 2.0))).collect();
                (vec![Complex::ZERO; n], poles, 2.0 * (w0 / fs2).atan())
            } else {
                let poles: Vec<Complex> = prototype.iter().flat_map(|p| split(Complex::from(bw / 2.0) / *p)).collect();
                let zeros = (0..n).flat_map(|_| [Complex::new(0.0, w0), Complex::new(0.0, -w0)]).collect();
                (zeros, poles, 0.0)
            }
        }
    };

    let bilinear = |s: &Complex| (Complex::from(fs2) + *s) / (Complex::from(fs2) - *s);
    let poles: Vec<Complex> = poles.iter().map(bilinear).collect();
    let mut zeros: Vec<Complex> = zeros.iter().map(bilinear).collect();
    // Zeros at infinity land on Nyquist.
    zeros.resize(poles.len(), Complex::from(-1.0));

    let pole_groups = group(&poles);
    let zero_groups = group(&zeros);
    debug_assert_eq!(pole_groups.len(), zero_groups.len());
    let mut sos: Sos = zero_groups
        .iter()
        .zip(&pole_groups)
        .map(|(z, p)| Biquad { b: *z, a: [p[1], p[2]] })
        .collect();

    let gain: f64 = response(&sos, w_ref * rate / (2.0 * PI), rate);
    if gain > 0.0 && gain.is_finite() {
        for b in &mut sos[0].b {
            *b /= gain;
        }
    }
    Ok(sos)
}

/// Monic polynomials (`[1, c1, c2]`) from roots: conjugate pairs, then real
/// pairs, then at most one first-order factor.
fn group(roots: &[Complex]) -> Vec<[f64; 3]> {
    let tolerance = 1e-9;
    let mut groups = Vec::new();
    let mut reals = Vec::new();
    for r in roots {
        if r.im.abs() <= tolerance * r.norm().max(1.0) {
            reals.push(r.re);
        } else if r.im > 0.0 {
            groups.push([1.0, -2.0 * r.re, r.norm_sqr()]);
        }
    }
    reals.sort_by(f64::total_cmp);
    let mut pairs = reals.chunks_exact(2);
    for pair in pairs.by_ref() {
        groups.push([1.0, -(pair[0] + pair[1]), pair[0] * pair[1]]);
    }
    if let [r] = pairs.remainder() {
        groups.push([1.0, -r, 0.0]);
    }
    groups
}

/// Stateful cascade. A NaN input yields NaN and restarts the filter.
#[derive(Clone, Debug)]
pub struct SosFilter {
    sections: Sos,
    state: Vec<[f64; 2]>,
}

impl SosFilter {
    pub fn new(sections: Sos) -> Self {
        let state = vec![[0.0; 2]; sections.len()];
        Self { sections, state }
    }

    /// Sets the state to the steady response to a constant input `x`.
    fn settle(&mut self, x: f64) {
        let mut x = x;
        for (s, state) in self.sections.iter().zip(&mut self.state) {
            let y = s.dc_gain() * x;
            *state = [y - s.b[0] * x, s.b[2] * x - s.a[1] * y];
            x = y;
        }
    }

    fn step(&mut self, x: f64) -> f64 {
        let mut x = x;
        for (s, [s1, s2]) in self.sections.iter().zip(&mut self.state) {
            let y = s.b[0] * x + *s1;
            *s1 = s.b[1] * x - s.a[0] * y + *s2;
            *s2 = s.b[2] * x - s.a[1] * y;
            x = y;
        }
        x
    }

    pub fn process(&mut self, values: &mut [f32]) {
        for v in values {
            if v.is_finite() {
                *v = self.step(f64::from(*v)) as f32;
            } else {
                self.state.fill([0.0; 2]);
                *v = f32::NAN;
            }
        }
    }
}

/// Reflection padding, long enough for the slowest corner to settle.
fn pad_length(sos: &[Biquad], lowest_hz: f64, rate: f64) -> usize {
    let ring = (2.0 * rate / lowest_hz).ceil().min(1e7) as usize;
    ring.max(3 * (2 * sos.len() + 1))
}

/// Zero-phase filtering of a gap-free run: odd reflection padding, steady-state
/// initial conditions, forward then backward.
pub fn filtfilt(sos: &[Biquad], values: &mut [f32], pad: usize) {
    let n = values.len();
    if n == 0 {
        return;
    }
    let pad = pad.min(n - 1);
    let x: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    let (first, last) = (x[0], x[n - 1]);
    let mut ext = Vec::with_capacity(n + 2 * pad);
    ext.extend((1..=pad).rev().map(|i| 2.0 * first - x[i]));
    ext.extend_from_slice(&x);
    ext.extend((1..=pad).map(|i| 2.0 * last - x[n - 1 - i]));

    let run = |data: &mut Vec<f64>| {
        let mut filter = SosFilter::new(sos.to_vec());
        filter.settle(data[0]);
        for v in data.iter_mut() {
            *v = filter.step(*v);
        }
    };
    run(&mut ext);
    ext.reverse();
    run(&mut ext);
    ext.reverse();
    for (v, y) in values.iter_mut().zip(&ext[pad..pad + n]) {
        *v = *y as f32;
    }
}

/// Index ranges of consecutive finite samples.
pub fn finite_runs(values: &[f32]) -> Vec<core::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut start = None;
    for (i, v) in values.iter().enumerate() {
        match (v.is_finite(), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                runs.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push(s..values.len());
    }
    runs
}

pub fn remove_mean(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    for v in values {
        *v = (f64::from(*v) - mean) as f32;
    }
}

/// Least-squares line removal.
pub fn remove_trend(values: &mut [f32]) {
    let n = values.len();
    if n < 2 {
        remove_mean(values);
        return;
    }
    let nf = n as f64;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = values.iter().map(|&v| f64::from(v)).sum::<f64>() / nf;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, &v) in values.iter().enumerate() {
        let dx = i as f64 - mean_x;
        sxy += dx * (f64::from(v) - mean_y);
        sxx += dx * dx;
    }
    let slope = sxy / sxx;
    for (i, v) in values.iter_mut().enumerate() {
        *v = (f64::from(*v) - mean_y - slope * (i as f64 - mean_x)) as f32;
    }
}

/// Hann taper over `fraction` of the run at each end.
pub fn taper(values: &mut [f32], fraction: f64) {
    let n = values.len();
    let m = ((fraction.clamp(0.0, 0.5) * n as f64) as usize).min(n / 2);
    for i in 0..m {
        let w = (0.5 - 0.5 * (PI * i as f64 / m as f64).cos()) as f32;
        values[i] *= w;
        values[n - 1 - i] *= w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: f64, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * hz * i as f64 / rate).sin() as f32).collect()
    }

    fn rms(values: &[f32]) -> f64 {
        (values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
    }

    #[test]
    fn butterworth_gains_match_the_analytic_response() {
        let rate = 100.0;
        for order in 1..=8u8 {
            let lp = butterworth(Band::Lowpass { hz: 10.0 }, order, rate).unwrap();
            assert!((response(&lp, 0.0, rate) - 1.0).abs() < 1e-9);
            // −3 dB at the corner.
            assert!((response(&lp, 10.0, rate) - 0.5f64.sqrt()).abs() < 1e-9, "order {order}");
            let hp = butterworth(Band::Highpass { hz: 10.0 }, order, rate).unwrap();
            assert!((response(&hp, 50.0, rate) - 1.0).abs() < 1e-9);
            assert!((response(&hp, 10.0, rate) - 0.5f64.sqrt()).abs() < 1e-9);
        }
        let bp = butterworth(Band::Bandpass { low_hz: 2.0, high_hz: 8.0 }, 4, rate).unwrap();
        assert_eq!(bp.len(), 4);
        assert!((response(&bp, 2.0, rate) - 0.5f64.sqrt()).abs() < 1e-6);
        assert!((response(&bp, 8.0, rate) - 0.5f64.sqrt()).abs() < 1e-6);
        assert!(response(&bp, 30.0, rate) < 1e-3);
        let bs = butterworth(Band::Bandstop { low_hz: 45.0, high_hz: 55.0 }, 3, 1000.0).unwrap();
        // The notch sits at the geometric centre of the prewarped edges.
        let warped = |hz: f64| (PI * hz / 1000.0).tan();
        let notch = (warped(45.0) * warped(55.0)).sqrt().atan() * 1000.0 / PI;
        assert!(response(&bs, notch, 1000.0) < 1e-6);
        assert!((response(&bs, 45.0, 1000.0) - 0.5f64.sqrt()).abs() < 1e-6);
        assert!((response(&bs, 0.0, 1000.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bad_designs_are_rejected() {
        assert_eq!(butterworth(Band::Lowpass { hz: 60.0 }, 2, 100.0), Err(DesignError::Corner { hz: 60.0, nyquist: 50.0 }));
        assert_eq!(butterworth(Band::Lowpass { hz: 1.0 }, 0, 100.0), Err(DesignError::Order(0)));
        assert_eq!(butterworth(Band::Bandpass { low_hz: 5.0, high_hz: 2.0 }, 2, 100.0), Err(DesignError::Edges));
    }

    #[test]
    fn filtering_passes_and_stops_tones() {
        let rate = 1000.0;
        let sos = butterworth(Band::Lowpass { hz: 50.0 }, 4, rate).unwrap();
        let mut low = tone(5.0, rate, 4000);
        let mut high = tone(200.0, rate, 4000);
        SosFilter::new(sos.clone()).process(&mut low);
        SosFilter::new(sos).process(&mut high);
        assert!((rms(&low[1000..]) - 0.5f64.sqrt()).abs() < 0.01);
        assert!(rms(&high[1000..]) < 0.01);
    }

    #[test]
    fn chunked_filtering_matches_whole() {
        let sos = butterworth(Band::Bandpass { low_hz: 1.0, high_hz: 10.0 }, 3, 100.0).unwrap();
        let input: Vec<f32> = (0..1000).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
        let mut whole = input.clone();
        SosFilter::new(sos.clone()).process(&mut whole);
        let mut filter = SosFilter::new(sos);
        let mut pieces = input;
        for chunk in pieces.chunks_mut(37) {
            filter.process(chunk);
        }
        assert_eq!(whole, pieces);
    }

    #[test]
    fn zero_phase_filtering_does_not_shift_a_pulse() {
        let rate = 100.0;
        let n = 2001;
        let mut pulse: Vec<f32> = (0..n).map(|i| (-((i as f64 - 1000.0) / 20.0).powi(2)).exp() as f32).collect();
        let processing = Processing {
            detrend: Detrend::None,
            taper: 0.0,
            filter: Some(FilterSpec { band: Band::Lowpass { hz: 5.0 }, order: 4, zero_phase: true }),
        };
        processing.apply(&mut pulse, rate).unwrap();
        let peak = pulse.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(peak, 1000);
        // Ends settle without a transient.
        assert!(pulse[0].abs() < 1e-3 && pulse[n - 1].abs() < 1e-3);
    }

    #[test]
    fn processing_skips_gaps() {
        let mut values = vec![1.0, 2.0, 3.0, f32::NAN, 10.0, 10.0];
        Processing { detrend: Detrend::Mean, ..Processing::default() }.apply(&mut values, 1.0).unwrap();
        assert_eq!(&values[..3], &[-1.0, 0.0, 1.0]);
        assert!(values[3].is_nan());
        assert_eq!(&values[4..], &[0.0, 0.0]);
    }

    #[test]
    fn linear_trends_are_removed() {
        let mut values: Vec<f32> = (0..100).map(|i| 3.0 + 0.5 * i as f32).collect();
        remove_trend(&mut values);
        assert!(values.iter().all(|v| v.abs() < 1e-4));
    }

    #[test]
    fn tapers_reach_zero_at_the_ends() {
        let mut values = vec![1.0; 100];
        taper(&mut values, 0.1);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[50], 1.0);
        assert!(values[5] > 0.0 && values[5] < 1.0);
    }
}
