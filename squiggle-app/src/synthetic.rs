//! Synthetic events with known answers, written as SAC to the temp directory
//! so a saved project reloads them like any import.
//!
//! - [`SPECS`]: five seismic stations at mixed rates and two infrasound
//!   sensors record one band-limited event, each delayed by a known amount,
//!   over noise and a microseism below 0.3 Hz. Checks alignment.
//! - [`STATIONS`]: a shallow explosion recorded by EMP, seismic (P and S) and
//!   infrasound sensors at known positions. Arrivals follow the built-in
//!   propagation speeds exactly. Checks picking and location.

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
