//! A synthetic event with known delays, for checking alignment by hand.
//!
//! Five seismic stations at mixed rates and two infrasound sensors record one
//! band-limited event, each delayed by a known amount, over noise and a
//! microseism below 0.3 Hz. Files are SAC in the temp directory, so a saved
//! project reloads them like any import.

use std::f64::consts::TAU;
use std::path::PathBuf;

use squiggle_core::Timestamp;

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
            let bytes = squiggle_io::sac::encode(&spec.samples(), origin.add_seconds(spec.start), spec.rate, spec.code);
            std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(path)
        })
        .collect()
}

pub fn directory() -> PathBuf {
    std::env::temp_dir().join("squiggle-synthetic")
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
}
