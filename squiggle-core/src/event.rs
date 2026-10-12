//! Events: a time window over traces that recorded one source, with each
//! member's alignment, the event's picks (on [`crate::sensor::Pick::event`])
//! and its located origin.

use crate::id::Id;
use crate::locate::Origin;
use crate::sensor::{Alignment, Provenance};
use crate::time::Timestamp;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventStatus {
    /// From detection; replaced when detection re-runs.
    #[default]
    Auto,
    /// Accepted or created by an analyst; kept by re-runs.
    Reviewed,
    /// A false trigger or noise; kept, and exported as the noise class.
    Rejected,
}

impl EventStatus {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Reviewed, Self::Rejected];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Automatic",
            Self::Reviewed => "Reviewed",
            Self::Rejected => "Rejected",
        }
    }
}

/// A trace in an event, with its alignment within that event.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Member {
    pub trace: Id,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub alignment: Option<Alignment>,
}

impl Member {
    pub fn new(trace: Id) -> Self {
        Self { trace, alignment: None }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Event {
    pub id: Id,
    /// Window in clock-corrected time.
    pub start: Timestamp,
    pub end: Timestamp,
    pub members: Vec<Member>,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub origin: Option<Origin>,
    pub status: EventStatus,
    pub provenance: Provenance,
}

impl Event {
    pub fn member(&self, trace: Id) -> Option<&Member> {
        self.members.iter().find(|m| m.trace == trace)
    }

    pub fn member_mut(&mut self, trace: Id) -> Option<&mut Member> {
        self.members.iter_mut().find(|m| m.trace == trace)
    }

    pub fn duration(&self) -> f64 {
        self.end.seconds_since(self.start)
    }

    /// Origin time when located, else the window start.
    pub fn time(&self) -> Timestamp {
        self.origin.as_ref().map_or(self.start, |o| o.time)
    }
}
