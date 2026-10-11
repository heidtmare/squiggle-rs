//! Sensors, the traces they record, and where each value came from.

use core::fmt;

use crate::id::Id;
use crate::phenomenology::{PhenomenologyId, SampleKind};
use crate::samples::RawLayout;
use crate::time::Timestamp;

/// Who made a change. The local OS user until accounts exist.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Analyst(pub String);

impl Analyst {
    /// `$USER` / `$USERNAME`, else `unknown`.
    pub fn local() -> Self {
        let name = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "unknown".to_string());
        Self(name)
    }
}

impl fmt::Display for Analyst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Origin of a value. Manual values are locked against automatic re-runs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Provenance {
    Manual { analyst: Analyst, at: Timestamp },
    Auto { stage: String, config_hash: String },
}

impl Provenance {
    pub fn is_manual(&self) -> bool {
        matches!(self, Self::Manual { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Assignment {
    pub phenomenology: PhenomenologyId,
    /// Phenomenology version at assignment time.
    pub version: u32,
    pub provenance: Provenance,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub elevation_m: f64,
}

/// `true = recorded + offset + drift · (recorded − reference)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClockCorrection {
    pub offset_s: f64,
    /// Parts per million; zero until drift estimation exists.
    pub drift_ppm: f64,
}

impl ClockCorrection {
    pub fn apply(self, recorded: Timestamp, reference: Timestamp) -> Timestamp {
        let drift = self.drift_ppm * 1e-6 * recorded.seconds_since(reference);
        recorded.add_seconds(self.offset_s + drift)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sensor {
    pub id: Id,
    /// Native identifier, e.g. `NET.STA.LOC.CHA`. Unique within a project.
    pub code: String,
    /// `None` until an analyst assigns one; unassigned sensors are not processed.
    pub assignment: Option<Assignment>,
    pub location: Option<Location>,
    pub clock: ClockCorrection,
}

/// Where a trace's samples live. Samples are never copied into the project.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TraceSource {
    File {
        path: String,
        /// Reader name, e.g. `wav`.
        format: String,
        /// Channel index within the file.
        channel: u32,
        /// BLAKE3 of the file and any companion files, hex.
        content_hash: String,
        /// Analyst-supplied layout for headerless binary files.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        layout: Option<RawLayout>,
    },
}

impl TraceSource {
    pub fn describe(&self) -> String {
        match self {
            Self::File { path, channel, .. } => format!("{path} #{channel}"),
        }
    }
}

/// Trace metadata, as persisted. Samples are held separately in [`crate::samples::TraceSamples`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trace {
    pub id: Id,
    pub sensor: Id,
    /// First sample, as recorded (before clock correction).
    pub start: Timestamp,
    pub sample_rate: f64,
    pub len: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    pub kind: SampleKind,
    pub source: TraceSource,
}

impl Trace {
    pub fn duration(&self) -> f64 {
        self.len as f64 / self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_correction_applies_offset_and_drift() {
        let reference = Timestamp::new(1000, 0.0);
        let clock = ClockCorrection { offset_s: 0.5, drift_ppm: 10.0 };
        let corrected = clock.apply(Timestamp::new(2000, 0.0), reference);
        // 0.5 s offset + 10 ppm over 1000 s = 0.51 s.
        assert!((corrected.seconds_since(Timestamp::new(2000, 0.0)) - 0.51).abs() < 1e-12);
    }
}
