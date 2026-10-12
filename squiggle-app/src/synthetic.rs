//! Synthetic events with known answers, written as SAC to the temp directory
//! so a saved project reloads them like any import.
//!
//! - [`SPECS`]: five seismic stations at mixed rates and two infrasound
//!   sensors record one band-limited event, each delayed by a known amount,
//!   over noise and a microseism below 0.3 Hz. Checks alignment.
//! - [`STATIONS`]: a shallow explosion recorded by EMP, seismic (P and S) and
//!   infrasound sensors at known positions. Arrivals follow the built-in
//!   propagation speeds exactly. Checks picking and location.
//! - [`SWARM`]: two minutes on eight seismic stations with six events (one
//!   too weak to detect) and single-station glitches. Checks detection and
//!   association.
//! - [`MIXED`]: seven minutes of earthquakes and explosions on eight seismic
//!   stations and two infrasound sensors, with the physical signatures that
//!   tell them apart. Checks features and classification.
//! - [`catalogue`]: the same sensors over a longer, seeded run of varied
//!   earthquakes and explosions, for training models.

use std::f64::consts::TAU;
use std::path::PathBuf;

use squiggle_core::locate::{Source, travel_time};
use squiggle_core::phenomenology::builtins;
use squiggle_core::{Location, Timestamp};

/// Event onset in each trace before its delay, seconds after [`ORIGIN`].
const ONSET: f64 = 10.0;
const DURATION: f64 = 30.0;
pub const ORIGIN: &str = "2026-01-01T00:00:00Z";

pub struct Spec {
    pub code: [&'static str; 4],
    pub phenomenology: &'static str,
    pub rate: f64,
    /// True delay of the event, seconds. Comparable within a phenomenology.
    pub delay: f64,
    /// Recording start after the origin, seconds.
    pub start: f64,
    amplitude: f64,
    seed: u32,
}

pub const SPECS: [Spec; 7] = [
    Spec { code: ["SY", "STA01", "", "BHZ"], phenomenology: "seismic", rate: 100.0, delay: 0.0, start: 0.0, amplitude: 1.0, seed: 1 },
    Spec { code: ["SY", "STA02", "", "BHZ"], phenomenology: "seismic", rate: 40.0, delay: 0.8125, start: 0.37, amplitude: 0.6, seed: 2 },
    Spec { code: ["SY", "STA03", "", "HHZ"], phenomenology: "seismic", rate: 200.0, delay: -0.433, start: -1.2, amplitude: 1.4, seed: 3 },
    Spec { code: ["SY", "STA04", "", "BHZ"], phenomenology: "seismic", rate: 100.0, delay: 1.9, start: 2.0, amplitude: 0.8, seed: 4 },
    Spec { code: ["SY", "STA05", "", "BHZ"], phenomenology: "seismic", rate: 50.0, delay: 0.27, start: 0.81, amplitude: 1.1, seed: 5 },
    Spec { code: ["SY", "IS01", "", "BDF"], phenomenology: "infrasound", rate: 20.0, delay: 0.0, start: 0.5, amplitude: 1.0, seed: 6 },
    Spec { code: ["SY", "IS02", "", "BDF"], phenomenology: "infrasound", rate: 20.0, delay: 3.3, start: 1.0, amplitude: 0.7, seed: 7 },
];

impl Spec {
    pub fn code(&self) -> String {
        self.code.join(".")
    }

    fn signal(&self, t: f64) -> f64 {
        let t = t - ONSET - self.delay;
        let event = if self.phenomenology == "infrasound" {
            let envelope = (-(t / 1.8).powi(2)).exp();
            envelope * ((TAU * 0.7 * t + 0.3).sin() + 0.6 * (TAU * 1.9 * t + 1.2).sin() + 0.3 * (TAU * 3.1 * t).sin())
        } else {
            let envelope = (-(t / 0.6).powi(2)).exp();
            let tones = [(1.3, 0.2, 1.0), (3.7, 1.1, 0.8), (6.1, 2.3, 0.6), (9.4, 0.7, 0.5), (11.8, 4.0, 0.3)];
            envelope * tones.iter().map(|(f, phase, a)| a * (TAU * f * t + phase).sin()).sum::<f64>()
        };
        self.amplitude * event
    }

    fn samples(&self) -> Vec<f32> {
        let mut state = self.seed.wrapping_mul(0x9E37_79B9) | 1;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state % 2001) / 1000.0 - 1.0
        };
        let len = (DURATION * self.rate) as usize;
        let phase = f64::from(self.seed);
        (0..len)
            .map(|i| {
                let t = self.start + i as f64 / self.rate;
                let microseism = 0.8 * (TAU * 0.15 * t + phase).sin() + 0.3 * (TAU * 0.27 * t).sin();
                (self.signal(t) + 0.04 * noise() + microseism) as f32
            })
            .collect()
    }
}

/// Writes every trace as SAC into `dir`; returns the paths in [`SPECS`] order.
pub fn write(dir: &std::path::Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let origin: Timestamp = ORIGIN.parse().expect("valid constant");
    SPECS
        .iter()
        .map(|spec| {
            let path = dir.join(format!("{}.sac", spec.code()));
            let bytes = squiggle_io::sac::encode(&spec.samples(), origin.add_seconds(spec.start), spec.rate, spec.code, None);
            write_atomic(&path, &bytes)?;
            Ok(path)
        })
        .collect()
}

/// Writes via a uniquely named temporary and a rename, so concurrent writers
/// of the same file never expose a partial one.
fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_extension(format!("{}.{n}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes).map_err(|e| format!("{}: {e}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn directory() -> PathBuf {
    std::env::temp_dir().join("squiggle-synthetic")
}

/// The located event's source.
pub const SOURCE: Source = Source { latitude: 35.0, longitude: -106.0, depth_m: 500.0 };
/// Its origin time, seconds after [`ORIGIN`].
pub const ORIGIN_TIME: f64 = 10.0;

pub struct Station {
    pub code: [&'static str; 4],
    pub phenomenology: &'static str,
    pub rate: f64,
    /// Position relative to the source, kilometres.
    east_km: f64,
    north_km: f64,
    elevation_m: f64,
    /// Recording start and length, seconds after [`ORIGIN`].
    start: f64,
    duration: f64,
    seed: u32,
}

const fn station(code: [&'static str; 4], phenomenology: &'static str, rate: f64, east_km: f64, north_km: f64, seed: u32) -> Station {
    let (start, duration) = match phenomenology.as_bytes()[0] {
        b'e' => (ORIGIN_TIME - 0.005, 0.02),
        _ => (5.0, 30.0),
    };
    Station { code, phenomenology, rate, east_km, north_km, elevation_m: 0.0, start, duration, seed }
}

pub const STATIONS: [Station; 12] = [
    station(["SY", "EMP1", "", "EFZ"], "emp", 1e6, 4.0, 3.0, 11),
    station(["SY", "EMP2", "", "EFZ"], "emp", 1e6, -6.0, -2.0, 12),
    station(["SY", "EMP3", "", "EFZ"], "emp", 1e6, 1.0, -9.0, 13),
    station(["SY", "SE01", "", "HHZ"], "seismic", 100.0, 12.0, 5.0, 21),
    station(["SY", "SE02", "", "HHZ"], "seismic", 100.0, -8.0, 14.0, 22),
    station(["SY", "SE03", "", "HHZ"], "seismic", 200.0, -15.0, -6.0, 23),
    station(["SY", "SE04", "", "BHZ"], "seismic", 50.0, 5.0, -18.0, 24),
    station(["SY", "SE05", "", "HHZ"], "seismic", 100.0, 22.0, -9.0, 25),
    station(["SY", "SE06", "", "HHZ"], "seismic", 100.0, -3.0, 30.0, 26),
    station(["SY", "IS01", "", "BDF"], "infrasound", 50.0, 1.5, 0.8, 31),
    station(["SY", "IS02", "", "BDF"], "infrasound", 50.0, -2.0, 1.5, 32),
    station(["SY", "IS03", "", "BDF"], "infrasound", 50.0, 0.5, -3.0, 33),
];

impl Station {
    pub fn code(&self) -> String {
        self.code.join(".")
    }

    pub fn location(&self) -> Location {
        let latitude = SOURCE.latitude + (self.north_km * 1e3 / squiggle_core::locate::EARTH_RADIUS_M).to_degrees();
        let longitude = SOURCE.longitude
            + (self.east_km * 1e3 / (squiggle_core::locate::EARTH_RADIUS_M * SOURCE.latitude.to_radians().cos())).to_degrees();
        Location { latitude, longitude, elevation_m: self.elevation_m }
    }

    /// Phase name and true arrival, seconds after [`ORIGIN`], at the stored
    /// (f32) coordinates the reader returns.
    pub fn arrivals(&self) -> Vec<(String, f64)> {
        let stored = self.location();
        let location = Location { latitude: f64::from(stored.latitude as f32), longitude: f64::from(stored.longitude as f32), ..stored };
        let definition = builtins().into_iter().find(|p| p.id.as_str() == self.phenomenology).expect("built-in");
        definition
            .propagation
            .phases()
            .into_iter()
            .filter(|(name, _)| matches!(*name, "" | "P" | "S"))
            .map(|(name, speed)| (name.to_string(), ORIGIN_TIME + travel_time(SOURCE, &location, speed)))
            .collect()
    }

    fn samples(&self) -> Vec<f32> {
        let mut state = self.seed.wrapping_mul(0x9E37_79B9) | 1;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state % 2001) / 1000.0 - 1.0
        };
        let arrivals = self.arrivals();
        let distance_km = self.east_km.hypot(self.north_km);
        let len = (self.duration * self.rate) as usize;
        (0..len)
            .map(|i| {
                let t = self.start + i as f64 / self.rate;
                let signal: f64 = arrivals
                    .iter()
                    .map(|(phase, at)| {
                        let t = t - at;
                        if t < 0.0 {
                            return 0.0;
                        }
                        match (self.phenomenology, phase.as_str()) {
                            // Double exponential: 1 µs rise, 20 µs decay.
                            ("emp", _) => 1.05 * ((-t / 20e-6).exp() - (-t / 1e-6).exp()),
                            // Friedlander overpressure.
                            ("infrasound", _) => (1.0 - t / 0.25) * (-t / 0.25).exp() * 2.0 / distance_km.sqrt(),
                            (_, "P") => {
                                let envelope = (1.0 - (-t / 0.01).exp()) * (-t / 0.4).exp();
                                envelope * ((TAU * 6.0 * t + 1.0).sin() + 0.6 * (TAU * 9.5 * t + 2.0).sin()) * 3.0 / distance_km.sqrt()
                            }
                            (_, _) => {
                                let envelope = (1.0 - (-t / 0.02).exp()) * (-t / 0.8).exp();
                                envelope * ((TAU * 3.0 * t + 0.5).sin() + 0.5 * (TAU * 4.5 * t).sin()) * 6.0 / distance_km.sqrt()
                            }
                        }
                    })
                    .sum();
                (signal + 0.01 * noise()) as f32
            })
            .collect()
    }
}

/// A swarm event: position relative to [`SOURCE`], origin time after
/// [`ORIGIN`], and amplitude (1 is the located event's).
pub struct SwarmEvent {
    pub east_km: f64,
    pub north_km: f64,
    pub origin: f64,
    pub amplitude: f64,
}

pub const SWARM: [SwarmEvent; 6] = [
    SwarmEvent { east_km: 2.0, north_km: 3.0, origin: 12.0, amplitude: 1.0 },
    SwarmEvent { east_km: -6.0, north_km: 8.0, origin: 32.0, amplitude: 0.7 },
    SwarmEvent { east_km: 10.0, north_km: -4.0, origin: 52.0, amplitude: 1.2 },
    // Below the noise: not expected to be found.
    SwarmEvent { east_km: -3.0, north_km: -10.0, origin: 72.0, amplitude: 0.004 },
    SwarmEvent { east_km: 4.0, north_km: 12.0, origin: 92.0, amplitude: 0.8 },
    SwarmEvent { east_km: -9.0, north_km: -2.0, origin: 110.0, amplitude: 1.0 },
];

/// Events in [`SWARM`] strong enough to detect.
#[cfg(test)]
pub const SWARM_DETECTABLE: usize = 5;

/// Single-station glitches: station index and time after [`ORIGIN`].
const GLITCHES: [(usize, f64); 4] = [(2, 20.0), (5, 65.0), (7, 100.0), (0, 121.0)];

const SWARM_DURATION: f64 = 130.0;
const SWARM_RATE: f64 = 100.0;
const SWARM_STATIONS: [(&str, f64, f64); 8] = [
    ("SW01", 12.0, 5.0),
    ("SW02", -8.0, 14.0),
    ("SW03", -15.0, -6.0),
    ("SW04", 5.0, -18.0),
    ("SW05", 22.0, -9.0),
    ("SW06", -3.0, 25.0),
    ("SW07", 15.0, 15.0),
    ("SW08", -18.0, 8.0),
];

impl SwarmEvent {
    pub fn source(&self) -> Source {
        let l = offset_location(self.east_km, self.north_km);
        Source { latitude: l.latitude, longitude: l.longitude, depth_m: 3_000.0 }
    }
}

fn offset_location(east_km: f64, north_km: f64) -> Location {
    let r = squiggle_core::locate::EARTH_RADIUS_M;
    let latitude = SOURCE.latitude + (north_km * 1e3 / r).to_degrees();
    let longitude = SOURCE.longitude + (east_km * 1e3 / (r * SOURCE.latitude.to_radians().cos())).to_degrees();
    Location { latitude, longitude, elevation_m: 0.0 }
}

/// Samples of swarm station `index`, starting at [`ORIGIN`].
fn swarm_samples(index: usize) -> Vec<f32> {
    let (_, east, north) = SWARM_STATIONS[index];
    let stored = offset_location(east, north);
    let location = Location { latitude: f64::from(stored.latitude as f32), longitude: f64::from(stored.longitude as f32), ..stored };
    let mut state = (index as u32 + 41).wrapping_mul(0x9E37_79B9) | 1;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state % 2001) / 1000.0 - 1.0
    };
    let arrivals: Vec<(f64, f64, f64)> = SWARM
        .iter()
        .flat_map(|e| {
            let source = e.source();
            let distance_km = squiggle_core::locate::slant_distance(source, &location) / 1e3;
            let scale = e.amplitude / distance_km.sqrt();
            [
                (e.origin + travel_time(source, &location, 6000.0), 3.0 * scale, 0.0),
                (e.origin + travel_time(source, &location, 3500.0), 6.0 * scale, 1.0),
            ]
        })
        .collect();
    let len = (SWARM_DURATION * SWARM_RATE) as usize;
    (0..len)
        .map(|i| {
            let t = i as f64 / SWARM_RATE;
            let mut value = 0.01 * noise();
            for &(at, amplitude, s_wave) in &arrivals {
                let t = t - at;
                if (0.0..6.0).contains(&t) {
                    value += if s_wave == 0.0 {
                        amplitude * (1.0 - (-t / 0.01).exp()) * (-t / 0.4).exp() * ((TAU * 6.0 * t + 1.0).sin() + 0.6 * (TAU * 9.5 * t + 2.0).sin())
                    } else {
                        amplitude * (1.0 - (-t / 0.02).exp()) * (-t / 0.8).exp() * ((TAU * 3.0 * t + 0.5).sin() + 0.5 * (TAU * 4.5 * t).sin())
                    };
                }
            }
            for &(station, at) in &GLITCHES {
                let t = t - at;
                if station == index && (0.0..0.1).contains(&t) {
                    value += (TAU * 20.0 * t).sin();
                }
            }
            value as f32
        })
        .collect()
}

/// Writes the swarm as SAC with coordinates; paths in station order.
pub fn write_swarm(dir: &std::path::Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let origin: Timestamp = ORIGIN.parse().expect("valid constant");
    SWARM_STATIONS
        .iter()
        .enumerate()
        .map(|(index, &(station, east, north))| {
            let path = dir.join(format!("SY.{station}..HHZ.sac"));
            let bytes = squiggle_io::sac::encode(&swarm_samples(index), origin, SWARM_RATE, ["SY", station, "", "HHZ"], Some(offset_location(east, north)));
            write_atomic(&path, &bytes)?;
            Ok(path)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Earthquake,
    Explosion,
}

impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Self::Earthquake => squiggle_core::taxonomy::EARTHQUAKE,
            Self::Explosion => squiggle_core::taxonomy::EXPLOSION,
        }
    }
}

/// A source in the mixed scene, relative to [`SOURCE`].
pub struct MixedEvent {
    pub class: Class,
    pub east_km: f64,
    pub north_km: f64,
    pub depth_km: f64,
    pub origin: f64,
    /// Fault strike, degrees; earthquakes only. Sets the P radiation pattern.
    strike: f64,
    /// Amplitude factor.
    scale: f64,
    /// Frequency factor.
    tone: f64,
}

const fn quake(east_km: f64, north_km: f64, depth_km: f64, origin: f64, strike: f64) -> MixedEvent {
    MixedEvent { class: Class::Earthquake, east_km, north_km, depth_km, origin, strike, scale: 1.0, tone: 1.0 }
}

const fn blast(east_km: f64, north_km: f64, depth_km: f64, origin: f64) -> MixedEvent {
    MixedEvent { class: Class::Explosion, east_km, north_km, depth_km, origin, strike: 0.0, scale: 1.0, tone: 1.0 }
}

/// Seed of the training catalogue.
pub const CATALOGUE_SEED: u64 = 2026;
pub const CATALOGUE_EVENTS: usize = 40;

/// `n` events 40 s apart from 10 s, either class at random, within 5 km of
/// [`SOURCE`], earthquakes 4–12 km and explosions 0.1–0.6 km deep, with
/// amplitude ×0.6–1.6 and frequency ×0.8–1.25.
pub fn catalogue(seed: u64, n: usize) -> Vec<MixedEvent> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|i| {
            let class = if uniform() < 0.5 { Class::Earthquake } else { Class::Explosion };
            let (east_km, north_km) = (10.0 * uniform() - 5.0, 10.0 * uniform() - 5.0);
            let depth_km = match class {
                Class::Earthquake => 4.0 + 8.0 * uniform(),
                Class::Explosion => 0.1 + 0.5 * uniform(),
            };
            let strike = 180.0 * uniform();
            MixedEvent { class, east_km, north_km, depth_km, origin: 10.0 + 40.0 * i as f64, strike, scale: 0.6 + uniform(), tone: 0.8 + 0.45 * uniform() }
        })
        .collect()
}

/// Seconds covering `events`.
pub fn mixed_duration(events: &[MixedEvent]) -> f64 {
    events.iter().map(|e| e.origin).fold(0.0, f64::max) + 50.0
}

/// Earthquakes: deep, weak and low-frequency P with a quadrant first-motion
/// pattern, strong S, no air wave. Explosions: shallow, strong high-frequency
/// compressional P, weak S, and an infrasound arrival.
pub const MIXED: [MixedEvent; 10] = [
    quake(3.0, 2.0, 8.0, 10.0, 20.0),
    blast(-2.0, 4.0, 0.3, 50.0),
    quake(-5.0, -3.0, 6.0, 90.0, 70.0),
    blast(4.0, -4.0, 0.5, 130.0),
    quake(1.0, 5.0, 11.0, 170.0, 130.0),
    blast(-4.0, 1.0, 0.2, 210.0),
    quake(5.0, 0.0, 5.0, 250.0, 160.0),
    blast(0.0, -5.0, 0.4, 290.0),
    quake(-3.0, -5.0, 9.0, 330.0, 100.0),
    blast(2.0, 3.0, 0.3, 370.0),
];

const MIXED_DURATION: f64 = 420.0;
const INFRASOUND_RATE: f64 = 20.0;
const MIXED_INFRASOUND: [(&str, f64, f64); 2] = [("IS01", 1.0, 0.5), ("IS02", -1.5, -1.0)];

impl MixedEvent {
    pub fn source(&self) -> Source {
        let l = offset_location(self.east_km, self.north_km);
        Source { latitude: l.latitude, longitude: l.longitude, depth_m: self.depth_km * 1e3 }
    }
}

fn stored(east_km: f64, north_km: f64) -> Location {
    let l = offset_location(east_km, north_km);
    Location { latitude: f64::from(l.latitude as f32), longitude: f64::from(l.longitude as f32), ..l }
}

fn xorshift(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state % 2001) / 1000.0 - 1.0
    }
}

/// A phase starting at `t = 0`: sharp rise, exponential decay, two tones
/// starting upward, so the first motion has the sign of `amplitude`.
fn wavelet(t: f64, amplitude: f64, decay: f64, f: f64, g: f64) -> f64 {
    if t < 0.0 || t > 8.0 * decay {
        return 0.0;
    }
    amplitude * (1.0 - (-t / 0.01).exp()) * (-t / decay).exp() * ((TAU * f * t).sin() + 0.5 * (TAU * g * t).sin())
}

fn mixed_seismic(index: usize, events: &[MixedEvent], duration: f64) -> Vec<f32> {
    let (_, east, north) = SWARM_STATIONS[index];
    let location = stored(east, north);
    let mut noise = xorshift(index as u32 + 61);
    let len = (duration * SWARM_RATE) as usize;
    let mut values: Vec<f64> = (0..len).map(|_| 0.01 * noise()).collect();
    for e in events {
        let source = e.source();
        let scale = e.scale / (squiggle_core::locate::slant_distance(source, &location) / 1e3).sqrt();
        let f = e.tone;
        let p = e.origin + travel_time(source, &location, 6000.0);
        let s = e.origin + travel_time(source, &location, 3500.0);
        // Amplitude, decay and two tones of P, then of S.
        let (pw, sw) = match e.class {
            Class::Earthquake => {
                let azimuth = (east - e.east_km).atan2(north - e.north_km);
                let radiation = (2.0 * (azimuth - e.strike.to_radians())).sin();
                let p_amplitude = 1.5 * scale * radiation.signum() * (0.4 + 0.6 * radiation.abs());
                ((p_amplitude, 0.4, 3.0 * f, 5.0 * f), (4.0 * scale, 1.0, 2.0 * f, 3.0 * f))
            }
            Class::Explosion => ((3.0 * scale, 0.3, 10.0 * f, 14.0 * f), (0.6 * scale, 0.6, 6.0 * f, 7.0 * f)),
        };
        let from = (p * SWARM_RATE) as usize;
        let to = ((s + 8.0) * SWARM_RATE).min(len as f64) as usize;
        for (i, v) in values.iter_mut().enumerate().take(to).skip(from) {
            let t = i as f64 / SWARM_RATE;
            *v += wavelet(t - p, pw.0, pw.1, pw.2, pw.3) + wavelet(t - s, sw.0, sw.1, sw.2, sw.3);
        }
    }
    values.into_iter().map(|v| v as f32).collect()
}

fn mixed_infrasound(index: usize, events: &[MixedEvent], duration: f64) -> Vec<f32> {
    let (_, east, north) = MIXED_INFRASOUND[index];
    let location = stored(east, north);
    let mut noise = xorshift(index as u32 + 71);
    let len = (duration * INFRASOUND_RATE) as usize;
    let arrivals: Vec<(f64, f64)> = events
        .iter()
        .filter(|e| e.class == Class::Explosion)
        .map(|e| {
            let distance = squiggle_core::locate::slant_distance(e.source(), &location);
            (e.origin + distance / 340.0, 2.0 * e.scale / (distance / 1e3).sqrt())
        })
        .collect();
    (0..len)
        .map(|i| {
            let t = i as f64 / INFRASOUND_RATE;
            let mut value = 0.01 * noise();
            for &(at, amplitude) in &arrivals {
                let t = t - at;
                if (0.0..5.0).contains(&t) {
                    // Friedlander overpressure, 0.5 s positive phase.
                    value += amplitude * (1.0 - t / 0.5) * (-t / 0.5).exp();
                }
            }
            value as f32
        })
        .collect()
}

/// Writes the mixed scene as SAC with coordinates: seismic, then infrasound.
pub fn write_mixed(dir: &std::path::Path) -> Result<Vec<(PathBuf, &'static str)>, String> {
    write_mixed_events(dir, &MIXED, MIXED_DURATION)
}

/// Writes `events` over `duration` seconds on the mixed scene's sensors.
pub fn write_mixed_events(dir: &std::path::Path, events: &[MixedEvent], duration: f64) -> Result<Vec<(PathBuf, &'static str)>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let origin: Timestamp = ORIGIN.parse().expect("valid constant");
    let mut paths = Vec::new();
    for (index, &(station, east, north)) in SWARM_STATIONS.iter().enumerate() {
        let path = dir.join(format!("SY.{station}..HHZ.sac"));
        let bytes = squiggle_io::sac::encode(&mixed_seismic(index, events, duration), origin, SWARM_RATE, ["SY", station, "", "HHZ"], Some(offset_location(east, north)));
        write_atomic(&path, &bytes)?;
        paths.push((path, "seismic"));
    }
    for (index, &(station, east, north)) in MIXED_INFRASOUND.iter().enumerate() {
        let path = dir.join(format!("SY.{station}..BDF.sac"));
        let bytes = squiggle_io::sac::encode(&mixed_infrasound(index, events, duration), origin, INFRASOUND_RATE, ["SY", station, "", "BDF"], Some(offset_location(east, north)));
        write_atomic(&path, &bytes)?;
        paths.push((path, "infrasound"));
    }
    Ok(paths)
}

/// Writes every located-event trace as SAC with its coordinates; paths in [`STATIONS`] order.
pub fn write_located(dir: &std::path::Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let origin: Timestamp = ORIGIN.parse().expect("valid constant");
    STATIONS
        .iter()
        .map(|station| {
            let path = dir.join(format!("{}.sac", station.code()));
            let start = origin.add_seconds(station.start);
            let bytes = squiggle_io::sac::encode(&station.samples(), start, station.rate, station.code, Some(station.location()));
            write_atomic(&path, &bytes)?;
            Ok(path)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_read_back_with_their_timing() {
        let dir = tempfile::tempdir().unwrap();
        let paths = write(dir.path()).unwrap();
        let read = squiggle_io::read_file(&paths[1]).unwrap();
        let recording = &read.recordings[0];
        assert_eq!(recording.code, "SY.STA02..BHZ");
        let origin: Timestamp = ORIGIN.parse().unwrap();
        assert!((recording.start.unwrap().seconds_since(origin) - 0.37).abs() < 1e-6);
        assert_eq!(recording.samples.len(), 1200);
    }

    #[test]
    fn located_files_carry_positions_and_arrivals() {
        let dir = tempfile::tempdir().unwrap();
        let paths = write_located(dir.path()).unwrap();
        let read = squiggle_io::read_file(&paths[3]).unwrap();
        let recording = &read.recordings[0];
        let location = recording.location.unwrap();
        assert!((location.latitude - STATIONS[3].location().latitude).abs() < 1e-5);
        let arrivals = STATIONS[3].arrivals();
        assert_eq!(arrivals.len(), 2);
        // P at 13 km and 6 km/s, S later.
        assert!((arrivals[0].1 - ORIGIN_TIME - 13f64.hypot(0.5) / 6.0).abs() < 0.01, "{arrivals:?}");
        assert!(arrivals[1].1 > arrivals[0].1);
        let emp = &squiggle_io::read_file(&paths[0]).unwrap().recordings[0];
        assert_eq!(emp.samples.len(), 20_000);
    }
}
