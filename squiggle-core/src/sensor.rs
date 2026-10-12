//! Sensors, the traces they record, and where each value came from.

use core::fmt;

use crate::dsp::Processing;
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

/// WGS84 degrees; elevation above sea level, metres.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub elevation_m: f64,
}

impl Location {
    pub fn is_valid(&self) -> bool {
        (-90.0..=90.0).contains(&self.latitude) && (-180.0..=180.0).contains(&self.longitude) && self.elevation_m.is_finite()
    }
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
    /// Recorded from a live feed: little-endian f32 samples (I/Q pairs when
    /// complex) appended to `path` as they arrive. The trace covers the first
    /// `bytes`, whose BLAKE3 is `content_hash`; a recording that ran on
    /// past its last checkpoint still loads.
    Stream {
        endpoint: String,
        /// Channel code as the feed named it.
        code: String,
        path: String,
        bytes: u64,
        content_hash: String,
    },
}

impl TraceSource {
    pub fn describe(&self) -> String {
        match self {
            Self::File { path, channel, .. } => format!("{path} #{channel}"),
            Self::Stream { endpoint, code, .. } => format!("{endpoint} {code}"),
        }
    }

    pub fn is_stream(&self) -> bool {
        matches!(self, Self::Stream { .. })
    }
}

/// A trace's alignment shift, added to its clock-corrected start. Global until
/// events exist; event members will carry their own.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Alignment {
    pub offset_s: f64,
    pub provenance: Provenance,
    /// Automatic alignment leaves a locked offset alone.
    pub locked: bool,
}

/// An arrival marked on a trace. One per trace, phase and event.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pick {
    pub id: Id,
    pub trace: Id,
    /// Onset on the trace's own clock (before clock correction and alignment),
    /// so the pick moves with its waveform.
    pub time: Timestamp,
    /// Phase name from the phenomenology's propagation; empty for a single speed.
    pub phase: String,
    /// One standard deviation, seconds.
    pub uncertainty_s: f64,
    /// Manual picks are locked against automatic re-picks.
    pub provenance: Provenance,
    /// The event it belongs to; `None` outside any event.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub event: Option<Id>,
}

impl Pick {
    /// `P`, or `onset` for an unnamed phase.
    pub fn phase_label(&self) -> &str {
        if self.phase.is_empty() { "onset" } else { &self.phase }
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
    #[cfg_attr(feature = "serde", serde(default))]
    pub processing: Processing,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub alignment: Option<Alignment>,
}

impl Trace {
    pub fn duration(&self) -> f64 {
        self.len as f64 / self.sample_rate
    }

    pub fn offset(&self) -> f64 {
        self.alignment.as_ref().map_or(0.0, |a| a.offset_s)
    }

    pub fn is_locked(&self) -> bool {
        self.alignment.as_ref().is_some_and(|a| a.locked)
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
