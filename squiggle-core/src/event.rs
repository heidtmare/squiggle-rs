//! Events: a time window over traces that recorded one source, with each
//! member's alignment, the event's picks (on [`crate::sensor::Pick::event`])
//! its located origin, the analyst's label and model predictions.

use crate::id::Id;
use crate::locate::Origin;
use crate::sensor::{Alignment, Provenance};
use crate::taxonomy::LabelId;
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
    /// The analyst's label.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub label: Option<LabelId>,
    /// At most one per model.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub predictions: Vec<Prediction>,
}

/// A model's class probabilities for an event.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Prediction {
    /// `heidtmare.squiggle.model/<name>@<version>`.
    pub model: String,
    /// Hash of the model's settings.
    pub config_hash: String,
    pub taxonomy_version: u32,
    /// Sum to one, highest first.
    pub labels: Vec<(LabelId, f32)>,
    /// Probabilities from each part of a fusion model (e.g. per phenomenology), highest first.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub parts: Vec<(String, Vec<(LabelId, f32)>)>,
}

impl Prediction {
    pub fn top(&self) -> Option<(&LabelId, f32)> {
        self.labels.first().map(|(l, p)| (l, *p))
    }

    /// Top probability minus the runner-up's.
    pub fn margin(&self) -> f32 {
        match self.labels.as_slice() {
            [] => 0.0,
            [(_, p)] => *p,
            [(_, a), (_, b), ..] => a - b,
        }
    }
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

    pub fn prediction(&self, model: &str) -> Option<&Prediction> {
        self.predictions.iter().find(|p| p.model == model)
    }

    /// Mean probabilities over models, highest first.
    pub fn consensus(&self) -> Vec<(LabelId, f32)> {
        self.consensus_by(|_| 1.0)
    }

    /// Probabilities averaged with each model's weight (an ensemble vote), highest first.
    pub fn consensus_by(&self, weight: impl Fn(&str) -> f32) -> Vec<(LabelId, f32)> {
        let mut sum: Vec<(LabelId, f32)> = Vec::new();
        let mut total = 0.0;
        for prediction in &self.predictions {
            let w = weight(&prediction.model).max(0.0);
            total += w;
            for (label, p) in &prediction.labels {
                match sum.iter_mut().find(|(l, _)| l == label) {
                    Some(entry) => entry.1 += w * p,
                    None => sum.push((label.clone(), w * p)),
                }
            }
        }
        if total > 0.0 {
            for entry in &mut sum {
                entry.1 /= total;
            }
        }
        sum.sort_by(|a, b| b.1.total_cmp(&a.1));
        sum
    }

    /// Review priority in 0..=1: low consensus margin, and models whose top
    /// labels differ, rank first. Unpredicted events score 1.
    pub fn uncertainty(&self) -> f32 {
        self.uncertainty_by(|_| 1.0)
    }

    /// [`Self::uncertainty`] over the weighted consensus.
    pub fn uncertainty_by(&self, weight: impl Fn(&str) -> f32) -> f32 {
        if self.predictions.is_empty() {
            return 1.0;
        }
        let consensus = self.consensus_by(weight);
        let margin = match consensus.as_slice() {
            [(_, a), (_, b), ..] => a - b,
            [(_, a)] => *a,
            [] => 0.0,
        };
        let tops: Vec<&LabelId> = self.predictions.iter().filter_map(|p| p.top().map(|t| t.0)).collect();
        let disagree = tops.windows(2).any(|w| w[0] != w[1]);
        ((1.0 - margin) * if disagree { 1.0 } else { 0.8 } + if disagree { 0.2 } else { 0.0 }).clamp(0.0, 1.0)
    }

    /// Replaces `prediction`'s model's entry.
    pub fn with_prediction(&self, prediction: Prediction) -> Vec<Prediction> {
        let mut predictions: Vec<Prediction> = self.predictions.iter().filter(|p| p.model != prediction.model).cloned().collect();
        predictions.push(prediction);
        predictions
    }

    /// Origin time when located, else the window start.
    pub fn time(&self) -> Timestamp {
        self.origin.as_ref().map_or(self.start, |o| o.time)
    }
}
