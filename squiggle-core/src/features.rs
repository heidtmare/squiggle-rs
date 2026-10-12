//! Event features: per member (time domain, spectral, wavelet) and per event
//! (across sensors, across phenomenologies, and earthquake/explosion
//! discriminants). The trainer computes them with this same code, so training
//! and inference cannot drift. Missing values are NaN.
//!
//! Times are clock-corrected seconds after a common reference. Phenomenologies
//! are classed by propagation, not by name: phases named `P` and `S` are
//! seismic-like, constant speeds under 1 km/s travel through air, over
//! 10⁸ m/s are electromagnetic.

use crate::id::Id;
use crate::locate::{Source, travel_time};
use crate::phenomenology::{PhenomenologyId, Propagation};
use crate::sensor::Location;
use crate::transform::wavelet::{Wavelet, decompose};
use crate::transform::{Complex, fft, ifft};

/// Version of the feature definitions below; recorded with every export.
pub const FEATURE_SET: &str = "heidtmare.squiggle.features/1";

const WAVELET_LEVELS: usize = 6;

pub const MEMBER_FEATURES: [&str; 20] = [
    "snr_db",
    "peak",
    "rms",
    "duration_s",
    "rise_s",
    "decay_s",
    "zcr_hz",
    "kurtosis",
    "dominant_hz",
    "centroid_hz",
    "bandwidth_hz",
    "rolloff_hz",
    "flatness",
    "hf_lf_log",
    "wavelet_1",
    "wavelet_2",
    "wavelet_3",
    "wavelet_4",
    "wavelet_5",
    "wavelet_6",
];

pub const EVENT_FEATURES: [&str; 15] = [
    "members",
    "sensors",
    "phenomenologies",
    "located",
    "depth_m",
    "rms_s",
    "arrival_spread_s",
    "apparent_velocity_m_s",
    "back_azimuth_deg",
    "amplitude_spread_log",
    "p_s_log",
    "hf_lf_log",
    "first_motion",
    "air_snr_db",
    "em_snr_db",
];

/// Index of `name` in [`EVENT_FEATURES`].
pub fn event_index(name: &str) -> Option<usize> {
    EVENT_FEATURES.iter().position(|n| *n == name)
}

/// Index of `name` in [`MEMBER_FEATURES`].
pub fn member_index(name: &str) -> Option<usize> {
    MEMBER_FEATURES.iter().position(|n| *n == name)
}

/// One trace, as plotted (processed).
#[derive(Clone, Debug)]
pub struct TraceData {
    pub trace: Id,
    pub sensor: Id,
    pub phenomenology: PhenomenologyId,
    pub propagation: Propagation,
    pub values: Vec<f32>,
    pub rate: f64,
    /// Corrected time of `values[0]`.
    pub start: f64,
    pub location: Option<Location>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Seismic,
    Air,
    Electromagnetic,
    Other,
}

/// Propagation class: P and S phases are seismic-like, under 1 km/s air, over 10⁸ m/s EM.
pub fn kind_of(propagation: &Propagation) -> Kind {
    let named = |n: &str| propagation.phases().iter().any(|p| p.0 == n);
    match propagation {
        Propagation::Phases(_) if named("P") && named("S") => Kind::Seismic,
        Propagation::Constant { speed_m_s } if *speed_m_s < 1e3 => Kind::Air,
        Propagation::Constant { speed_m_s } if *speed_m_s > 1e8 => Kind::Electromagnetic,
        _ => Kind::Other,
    }
}

impl TraceData {
    pub fn kind(&self) -> Kind {
        kind_of(&self.propagation)
    }


    fn index(&self, t: f64) -> f64 {
        (t - self.start) * self.rate
    }

    /// Samples over `from..to`, clipped to the trace; `None` when under `min` samples.
    fn slice(&self, from: f64, to: f64, min: usize) -> Option<&[f32]> {
        let a = self.index(from).round().max(0.0) as usize;
        let b = (self.index(to).round().max(0.0) as usize).min(self.values.len());
        (b > a && b - a >= min).then(|| &self.values[a..b])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OriginData {
    pub source: Source,
    pub time: f64,
    pub rms_s: f64,
}

/// An event over `traces`.
#[derive(Clone, Debug, PartialEq)]
pub struct EventData {
    pub start: f64,
    pub end: f64,
    pub origin: Option<OriginData>,
    /// Indices into the traces.
    pub members: Vec<usize>,
    /// Trace index, phase and corrected time.
    pub picks: Vec<(usize, String, f64)>,
}

impl EventData {
    fn pick(&self, trace: usize, phase: &str) -> Option<f64> {
        self.picks.iter().find(|p| p.0 == trace && p.1 == phase).map(|p| p.2)
    }

    fn first_pick(&self, trace: usize) -> Option<f64> {
        self.picks.iter().filter(|p| p.0 == trace).map(|p| p.2).min_by(f64::total_cmp)
    }

    /// Picked, else predicted from the origin.
    fn arrival(&self, trace: &TraceData, index: usize, phase: &str) -> Option<f64> {
        self.pick(index, phase).or_else(|| {
            let origin = self.origin?;
            Some(origin.time + travel_time(origin.source, trace.location.as_ref()?, trace.propagation.speed(phase)?))
        })
    }
}

/// Frequency bands, Hz, for the spectral ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bands {
    pub low: (f64, f64),
    pub high: (f64, f64),
}

impl Bands {
    /// Regional defaults for seismic-like traces; fractions of the rate otherwise.
    pub fn default_for(kind: Kind, rate: f64) -> Self {
        match kind {
            Kind::Seismic => Self { low: (1.0, 4.0), high: (8.0, 16.0) },
            Kind::Air => Self { low: (0.5, 2.0), high: (4.0, 8.0) },
            _ => Self { low: (0.01 * rate, 0.05 * rate), high: (0.1 * rate, 0.4 * rate) },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureSettings {
    /// Spectral-ratio bands for seismic-like traces.
    pub seismic_bands: Bands,
    /// Location error behind the coupling search window, metres.
    pub coupling_tolerance_m: f64,
    /// Longest P and S windows for the P/S ratio, seconds.
    pub phase_window_s: f64,
}

impl Default for FeatureSettings {
    fn default() -> Self {
        Self { seismic_bands: Bands::default_for(Kind::Seismic, 0.0), coupling_tolerance_m: 1_000.0, phase_window_s: 2.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Features {
    pub event: Vec<f64>,
    /// In [`EventData::members`] order.
    pub members: Vec<Vec<f64>>,
}

impl Features {
    pub fn event_value(&self, name: &str) -> f64 {
        event_index(name).map_or(f64::NAN, |i| self.event[i])
    }
}

pub fn features(traces: &[TraceData], event: &EventData, settings: &FeatureSettings) -> Features {
    let members: Vec<Vec<f64>> = event
        .members
        .iter()
        .map(|&i| {
            let trace = &traces[i];
            let bands = match trace.kind() {
                Kind::Seismic => settings.seismic_bands,
                kind => Bands::default_for(kind, trace.rate),
            };
            match trace.slice(event.start, event.end, 16) {
                Some(window) => {
                    let onset = event.first_pick(i).map(|t| ((t - event.start.max(trace.start)) * trace.rate).round().max(0.0) as usize);
                    member_features(window, trace.rate, onset, &bands)
                }
                None => vec![f64::NAN; MEMBER_FEATURES.len()],
            }
        })
        .collect();
    Features { event: event_features(traces, event, &members, settings), members }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { 0.5 * (values[n / 2 - 1] + values[n / 2]) }
}

fn rms(values: &[f64]) -> f64 {
    mean(values.iter().map(|v| v * v)).sqrt()
}

/// Magnitude of the analytic signal.
pub fn envelope(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut data: Vec<Complex> = values.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft(&mut data);
    for (k, c) in data.iter_mut().enumerate() {
        let factor = match k {
            0 => 1.0,
            k if 2 * k == n => 1.0,
            k if 2 * k < n => 2.0,
            _ => 0.0,
        };
        *c = c.scale(factor);
    }
    ifft(&mut data);
    data.iter().map(|c| c.norm()).collect()
}

/// One-sided power spectrum of a Hann-tapered signal: (frequency, power) for bins 1..=n/2.
pub fn power_spectrum(values: &[f64], rate: f64) -> Vec<(f64, f64)> {
    let n = values.len();
    let taper = crate::transform::Window::Hann.coefficients(n);
    let mut data: Vec<Complex> = values.iter().zip(&taper).map(|(v, w)| Complex::new(v * w, 0.0)).collect();
    fft(&mut data);
    (1..=n / 2).map(|k| (k as f64 * rate / n as f64, data[k].norm_sqr())).collect()
}

fn band_energy(spectrum: &[(f64, f64)], band: (f64, f64)) -> f64 {
    spectrum.iter().filter(|(f, _)| *f >= band.0 && *f < band.1).map(|(_, p)| p).sum()
}

/// Features of one member over its event window. `onset` indexes `window`;
/// without one, the envelope's first rise above a tenth of its peak.
pub fn member_features(window: &[f32], rate: f64, onset: Option<usize>, bands: &Bands) -> Vec<f64> {
    let mut out = vec![f64::NAN; MEMBER_FEATURES.len()];
    let set = |out: &mut Vec<f64>, name: &str, value: f64| out[member_index(name).expect("known feature")] = value;
    let offset = mean(window.iter().map(|&v| f64::from(v)));
    let x: Vec<f64> = window.iter().map(|&v| f64::from(v) - offset).collect();
    let env = envelope(&x);
    let (peak_at, peak_env) = env.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.0));
    let first_above = env.iter().position(|&e| e >= 0.1 * peak_env).unwrap_or(0);
    let onset = onset.unwrap_or(first_above).min(x.len().saturating_sub(1));
    let signal = &x[onset..];
    let noise = &x[..onset];
    if signal.len() < 8 {
        return out;
    }

    if noise.len() >= 8 {
        set(&mut out, "snr_db", 20.0 * (rms(signal) / rms(noise)).log10());
    }
    set(&mut out, "peak", signal.iter().fold(0.0, |m, v| m.max(v.abs())));
    set(&mut out, "rms", rms(signal));
    let last_above = env.iter().rposition(|&e| e >= 0.1 * peak_env).unwrap_or(0);
    set(&mut out, "duration_s", (last_above.saturating_sub(first_above)) as f64 / rate);
    set(&mut out, "rise_s", (peak_at as f64 - onset as f64) / rate);
    // Exponential time constant: a log-linear fit from the peak down to a tenth of it.
    let tail: Vec<(f64, f64)> = env[peak_at..]
        .iter()
        .take_while(|&&e| e >= 0.1 * peak_env)
        .enumerate()
        .map(|(i, e)| (i as f64 / rate, e.ln()))
        .collect();
    if tail.len() >= 4 {
        let (mt, ml) = (mean(tail.iter().map(|p| p.0)), mean(tail.iter().map(|p| p.1)));
        let slope = tail.iter().map(|(t, l)| (t - mt) * (l - ml)).sum::<f64>() / tail.iter().map(|(t, _)| (t - mt).powi(2)).sum::<f64>();
        if slope < 0.0 {
            set(&mut out, "decay_s", -1.0 / slope);
        }
    }
    let crossings = signal.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
    set(&mut out, "zcr_hz", crossings as f64 / (signal.len() as f64 / rate));
    let mu = mean(signal.iter().copied());
    let m2 = mean(signal.iter().map(|v| (v - mu).powi(2)));
    let m4 = mean(signal.iter().map(|v| (v - mu).powi(4)));
    set(&mut out, "kurtosis", m4 / (m2 * m2));

    let spectrum = power_spectrum(signal, rate);
    let total: f64 = spectrum.iter().map(|(_, p)| p).sum();
    if total > 0.0 {
        let dominant = spectrum.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(f64::NAN, |(f, _)| *f);
        let centroid = spectrum.iter().map(|(f, p)| f * p).sum::<f64>() / total;
        let spread = (spectrum.iter().map(|(f, p)| (f - centroid).powi(2) * p).sum::<f64>() / total).sqrt();
        let mut cumulative = 0.0;
        let rolloff = spectrum
            .iter()
            .find(|(_, p)| {
                cumulative += p;
                cumulative >= 0.85 * total
            })
            .map_or(f64::NAN, |(f, _)| *f);
        let tiny = total * 1e-12 / spectrum.len() as f64;
        let geometric = mean(spectrum.iter().map(|(_, p)| (p + tiny).ln())).exp();
        set(&mut out, "dominant_hz", dominant);
        set(&mut out, "centroid_hz", centroid);
        set(&mut out, "bandwidth_hz", spread);
        set(&mut out, "rolloff_hz", rolloff);
        set(&mut out, "flatness", geometric / (total / spectrum.len() as f64));
        let (low, high) = (band_energy(&spectrum, bands.low), band_energy(&spectrum, bands.high));
        if low > 0.0 && high > 0.0 {
            set(&mut out, "hf_lf_log", (high / low).log10());
        }
    }

    let levels = Wavelet::Daubechies4.max_levels(signal.len()).min(WAVELET_LEVELS);
    let decomposition = decompose(signal, Wavelet::Daubechies4, levels);
    let energy = |c: &[f64]| c.iter().map(|v| v * v).sum::<f64>();
    let total = decomposition.details.iter().map(|d| energy(d)).sum::<f64>() + energy(&decomposition.approximation);
    if total > 0.0 {
        for (level, detail) in decomposition.details.iter().enumerate() {
            set(&mut out, MEMBER_FEATURES[member_index("wavelet_1").expect("known") + level], energy(detail) / total);
        }
    }
    out
}

/// Least-squares plane wave through (east, north, time): apparent velocity
/// and back-azimuth (degrees clockwise from north, toward the source).
pub fn plane_wave(points: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let (me, mn, mt) = points.iter().fold((0.0, 0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n, a.2 + p.2 / n));
    let (mut see, mut snn, mut sen, mut set, mut snt) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(e, nn, t) in points {
        let (e, nn, t) = (e - me, nn - mn, t - mt);
        see += e * e;
        snn += nn * nn;
        sen += e * nn;
        set += e * t;
        snt += nn * t;
    }
    let det = see * snn - sen * sen;
    if det.abs() < 1e-9 * (see * snn).max(1e-300) {
        return None;
    }
    let sx = (set * snn - snt * sen) / det;
    let sy = (snt * see - set * sen) / det;
    let slowness = sx.hypot(sy);
    if slowness <= 0.0 {
        return None;
    }
    let travel = sx.atan2(sy).to_degrees();
    Some((1.0 / slowness, (travel + 180.0).rem_euclid(360.0)))
}

/// East and north of `location` from `origin`, metres (local flat earth).
fn local(origin: &Location, location: &Location) -> (f64, f64) {
    let r = crate::locate::EARTH_RADIUS_M;
    let north = (location.latitude - origin.latitude).to_radians() * r;
    let east = (location.longitude - origin.longitude).to_radians() * r * origin.latitude.to_radians().cos();
    (east, north)
}

/// Peak of `trace` over `from..to` relative to the peak over `noise_s` before it, in dB.
fn arrival_snr(trace: &TraceData, from: f64, to: f64, noise_s: f64) -> Option<f64> {
    let signal = trace.slice(from, to, 4)?;
    let noise = trace.slice(from - noise_s, from, 4)?;
    let mu = mean(noise.iter().map(|&v| f64::from(v)));
    let peak = |s: &[f32]| s.iter().fold(0.0f64, |m, &v| m.max((f64::from(v) - mu).abs()));
    let (s, n) = (peak(signal), peak(noise));
    (n > 0.0).then(|| 20.0 * (s / n).log10())
}

fn event_features(traces: &[TraceData], event: &EventData, members: &[Vec<f64>], settings: &FeatureSettings) -> Vec<f64> {
    let mut out = vec![f64::NAN; EVENT_FEATURES.len()];
    let set = |out: &mut Vec<f64>, name: &str, value: f64| out[event_index(name).expect("known feature")] = value;
    let member = |m: usize, name: &str| members[m][member_index(name).expect("known feature")];

    let mut sensors: Vec<Id> = event.members.iter().map(|&i| traces[i].sensor).collect();
    sensors.sort_unstable();
    sensors.dedup();
    let mut kinds: Vec<&PhenomenologyId> = event.members.iter().map(|&i| &traces[i].phenomenology).collect();
    kinds.sort();
    kinds.dedup();
    set(&mut out, "members", event.members.len() as f64);
    set(&mut out, "sensors", sensors.len() as f64);
    set(&mut out, "phenomenologies", kinds.len() as f64);
    set(&mut out, "located", if event.origin.is_some() { 1.0 } else { 0.0 });
    if let Some(origin) = event.origin {
        set(&mut out, "depth_m", origin.source.depth_m);
        set(&mut out, "rms_s", origin.rms_s);
    }
    let firsts: Vec<f64> = event.members.iter().filter_map(|&i| event.first_pick(i)).collect();
    if firsts.len() >= 2 {
        let spread = firsts.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v)) - firsts.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        set(&mut out, "arrival_spread_s", spread);
    }

    // The phenomenology with the most members carries the cross-sensor measures.
    let dominant = kinds.iter().max_by_key(|k| event.members.iter().filter(|&&i| &&traces[i].phenomenology == *k).count()).copied();
    let same: Vec<usize> = (0..event.members.len()).filter(|&m| Some(&traces[event.members[m]].phenomenology) == dominant).collect();
    let picked: Vec<(Location, f64)> = same
        .iter()
        .filter_map(|&m| {
            let i = event.members[m];
            Some((traces[i].location?, event.first_pick(i)?))
        })
        .collect();
    if let Some((anchor, _)) = picked.first() {
        let points: Vec<(f64, f64, f64)> = picked.iter().map(|(l, t)| {
            let (e, n) = local(anchor, l);
            (e, n, *t)
        }).collect();
        if let Some((velocity, azimuth)) = plane_wave(&points) {
            set(&mut out, "apparent_velocity_m_s", velocity);
            set(&mut out, "back_azimuth_deg", azimuth);
        }
    }
    let peaks: Vec<f64> = same.iter().map(|&m| member(m, "peak")).filter(|p| p.is_finite() && *p > 0.0).collect();
    if peaks.len() >= 2 {
        let (lo, hi) = peaks.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        set(&mut out, "amplitude_spread_log", (hi / lo).log10());
    }

    // Discriminants on seismic-like members.
    let seismic: Vec<usize> = (0..event.members.len()).filter(|&m| traces[event.members[m]].kind() == Kind::Seismic).collect();
    let mut ratios = Vec::new();
    let mut polarities = Vec::new();
    for &m in &seismic {
        let i = event.members[m];
        let trace = &traces[i];
        let Some(p) = event.arrival(trace, i, "P") else { continue };
        if let Some(s) = event.arrival(trace, i, "S") {
            let length = settings.phase_window_s.min(0.9 * (s - p));
            if length * trace.rate >= 5.0 {
                let peak = |from: f64| trace.slice(from, from + length, 4).map(|w| w.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs())));
                if let (Some(a), Some(b)) = (peak(p), peak(s))
                    && a > 0.0
                    && b > 0.0
                {
                    ratios.push((a / b).log10());
                }
            }
        }
        if let (Some(before), Some(after)) = (trace.slice(p - 0.5, p - 0.05, 4), trace.slice(p - 0.05, p + 0.5, 8)) {
            let baseline = mean(before.iter().map(|&v| f64::from(v)));
            let motion: Vec<f64> = after.iter().map(|&v| f64::from(v) - baseline).collect();
            let largest = motion.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if let Some(first) = motion.iter().find(|v| v.abs() >= 0.3 * largest && largest > 0.0) {
                polarities.push(first.signum());
            }
        }
    }
    set(&mut out, "p_s_log", median(ratios));
    set(&mut out, "hf_lf_log", median(seismic.iter().map(|&m| member(m, "hf_lf_log")).collect()));
    if polarities.len() >= 3 {
        set(&mut out, "first_motion", mean(polarities.into_iter()));
    }

    // Coupled air and electromagnetic arrivals, on any trace of those kinds.
    for (kind, name) in [(Kind::Air, "air_snr_db"), (Kind::Electromagnetic, "em_snr_db")] {
        let mut best = f64::NAN;
        for (i, trace) in traces.iter().enumerate().filter(|(_, t)| t.kind() == kind) {
            let snr = match (event.origin, trace.location, trace.propagation.speed("")) {
                (Some(origin), Some(location), Some(speed)) => {
                    let at = origin.time + travel_time(origin.source, &location, speed);
                    let tolerance = (settings.coupling_tolerance_m / speed).max(20.0 / trace.rate);
                    let length = 2.0 * tolerance + 100.0 / trace.rate;
                    arrival_snr(trace, at - tolerance, at - tolerance + length, 4.0 * length)
                }
                _ => event.members.iter().position(|&j| j == i).map(|m| member(m, "snr_db")),
            };
            if let Some(snr) = snr.filter(|s| s.is_finite()) {
                best = if best.is_nan() { snr } else { best.max(snr) };
            }
        }
        set(&mut out, name, best);
    }
    out
}

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use super::*;

    const RATE: f64 = 100.0;

    fn signal(seconds: f64, f: impl Fn(f64) -> f64) -> Vec<f32> {
        (0..(seconds * RATE) as usize).map(|i| f(i as f64 / RATE) as f32).collect()
    }

    fn value(features: &[f64], name: &str) -> f64 {
        features[member_index(name).unwrap()]
    }

    fn noise(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            // Box–Muller over a 64-bit LCG.
            let mut uniform = || {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (uniform(), uniform());
            (-2.0 * a.ln()).sqrt() * (TAU * b).cos()
        }
    }

    #[test]
    fn a_sine_matches_its_analytic_features() {
        let bands = Bands { low: (1.0, 3.0), high: (4.0, 8.0) };
        let x = signal(10.0, |t| 2.0 * (TAU * 5.0 * t).sin());
        let f = member_features(&x, RATE, Some(0), &bands);
        assert!((value(&f, "peak") - 2.0).abs() < 1e-3);
        assert!((value(&f, "rms") - 2f64.sqrt()).abs() < 1e-3);
        assert!((value(&f, "zcr_hz") - 10.0).abs() < 0.2);
        assert!((value(&f, "kurtosis") - 1.5).abs() < 1e-3, "{}", value(&f, "kurtosis"));
        assert!((value(&f, "dominant_hz") - 5.0).abs() < 0.11);
        assert!((value(&f, "centroid_hz") - 5.0).abs() < 0.05);
        assert!(value(&f, "bandwidth_hz") < 0.2);
        assert!(value(&f, "flatness") < 1e-3);
        assert!(value(&f, "hf_lf_log") > 3.0);
        assert!(value(&f, "snr_db").is_nan(), "no noise before the onset");
    }

    #[test]
    fn white_noise_is_gaussian_and_flat() {
        let mut n = noise(7);
        let x: Vec<f32> = (0..20_000).map(|_| n() as f32).collect();
        let f = member_features(&x, RATE, Some(0), &Bands { low: (5.0, 10.0), high: (30.0, 35.0) });
        assert!((value(&f, "kurtosis") - 3.0).abs() < 0.15, "{}", value(&f, "kurtosis"));
        assert!(value(&f, "flatness") > 0.5);
        assert!((value(&f, "centroid_hz") - 25.0).abs() < 1.0);
        assert!(value(&f, "hf_lf_log").abs() < 0.1, "equal-width bands carry equal energy");
    }

    #[test]
    fn a_decaying_burst_has_its_onset_rise_and_decay() {
        let x = signal(8.0, |t| if t < 2.0 { 0.0 } else { (-(t - 2.0) / 0.5).exp() * (TAU * 20.0 * (t - 2.0)).sin() });
        let mut n = noise(3);
        let x: Vec<f32> = x.iter().map(|v| v + 1e-3 * n() as f32).collect();
        let f = member_features(&x, RATE, Some(200), &Bands::default_for(Kind::Seismic, RATE));
        assert!(value(&f, "rise_s").abs() <= 0.05, "{}", value(&f, "rise_s"));
        assert!((value(&f, "decay_s") - 0.5).abs() < 0.05, "{}", value(&f, "decay_s"));
        assert!(value(&f, "snr_db") > 30.0);
        assert!((value(&f, "duration_s") - 0.5 * 10f64.ln()).abs() < 0.1, "{}", value(&f, "duration_s"));
    }

    #[test]
    fn two_tones_give_their_band_ratio_and_wavelet_level() {
        // 2 Hz at 1, 10 Hz at 0.1: high/low energy is 0.01.
        let x = signal(20.0, |t| (TAU * 2.0 * t).sin() + 0.1 * (TAU * 10.0 * t).sin());
        let f = member_features(&x, RATE, Some(0), &Bands { low: (1.0, 4.0), high: (8.0, 16.0) });
        assert!((value(&f, "hf_lf_log") + 2.0).abs() < 0.02, "{}", value(&f, "hf_lf_log"));
        // 30 Hz sits in the first detail level (25–50 Hz at 100 Hz).
        let x = signal(10.0, |t| (TAU * 30.0 * t).sin());
        let f = member_features(&x, RATE, Some(0), &Bands::default_for(Kind::Seismic, RATE));
        assert!(value(&f, "wavelet_1") > 0.8, "{f:?}");
    }

    #[test]
    fn a_plane_wave_gives_its_speed_and_back_azimuth() {
        // Travelling toward 30° (from a source at 210°) at 3 km/s.
        let (s, theta) = (1.0 / 3000.0, 30f64.to_radians());
        let points: Vec<(f64, f64, f64)> =
            [(0.0, 0.0), (500.0, 100.0), (-200.0, 700.0), (300.0, -400.0)].iter().map(|&(e, n)| (e, n, s * (e * theta.sin() + n * theta.cos()))).collect();
        let (v, az) = plane_wave(&points).unwrap();
        assert!((v - 3000.0).abs() < 1e-6 && (az - 210.0).abs() < 1e-6, "{v} {az}");
        assert!(plane_wave(&points[..2]).is_none());
    }
}
