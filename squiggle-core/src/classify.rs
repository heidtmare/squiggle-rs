//! The rules classifier: a transparent, untrained baseline over the
//! earthquake/explosion discriminants in [`crate::features`]. Each rule adds
//! log-odds toward explosion (negative: toward earthquake), clamped so no
//! single measure decides alone. The share left to `other` grows as fewer
//! discriminants are available.

use crate::event::Prediction;
use crate::features::event_index;
use crate::taxonomy::{EARTHQUAKE, EXPLOSION, LabelId, OTHER};

pub const RULES_MODEL: &str = "heidtmare.squiggle.model/rules@1";

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuleSettings {
    /// log10 P/S peak ratio above which explosion is favoured, and its scale.
    pub p_s_log: f64,
    pub p_s_scale: f64,
    /// log10 high/low band energy above which explosion is favoured, and its scale.
    pub hf_lf_log: f64,
    pub hf_lf_scale: f64,
    /// Sources shallower than this favour explosion; each kilometre deeper favours earthquake.
    pub shallow_m: f64,
    /// Mean first-motion sign above which compressional (explosion-like) onsets dominate.
    pub first_motion: f64,
    /// Coupled-arrival SNR that counts as present.
    pub coupled_db: f64,
}

impl Default for RuleSettings {
    fn default() -> Self {
        Self { p_s_log: 0.0, p_s_scale: 0.25, hf_lf_log: 0.0, hf_lf_scale: 0.5, shallow_m: 2_000.0, first_motion: 0.6, coupled_db: 10.0 }
    }
}

/// One rule's contribution.
#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    pub feature: &'static str,
    pub value: f64,
    /// Log-odds toward explosion.
    pub weight: f64,
    pub reason: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    /// Highest first; sums to one.
    pub labels: Vec<(LabelId, f32)>,
    pub evidence: Vec<Evidence>,
}

impl Verdict {
    pub fn prediction(&self, config_hash: &str, taxonomy_version: u32) -> Prediction {
        Prediction { model: RULES_MODEL.to_string(), config_hash: config_hash.to_string(), taxonomy_version, labels: self.labels.clone() }
    }
}

/// Applies the rules to an event feature vector ([`crate::features::EVENT_FEATURES`] order).
pub fn rules(event: &[f64], settings: &RuleSettings) -> Verdict {
    let get = |name: &str| event_index(name).map_or(f64::NAN, |i| event[i]);
    let mut evidence = Vec::new();
    let mut add = |feature: &'static str, value: f64, weight: f64, reason: &'static str| {
        if value.is_finite() {
            evidence.push(Evidence { feature, value, weight, reason });
        }
    };

    let v = get("p_s_log");
    add("p_s_log", v, ((v - settings.p_s_log) / settings.p_s_scale).clamp(-3.0, 3.0), "P strong relative to S");
    let v = get("hf_lf_log");
    add("hf_lf_log", v, ((v - settings.hf_lf_log) / settings.hf_lf_scale).clamp(-3.0, 3.0), "rich in high frequencies");
    let v = if get("located") == 1.0 { get("depth_m") } else { f64::NAN };
    add("depth_m", v, ((settings.shallow_m - v) / 1_000.0).clamp(-3.0, 1.5), "shallow source");
    let v = get("first_motion");
    add("first_motion", v, ((v - settings.first_motion) * 4.0).clamp(-3.0, 1.5), "compressional first motions");
    let v = get("air_snr_db");
    add("air_snr_db", v, if v >= settings.coupled_db { 1.5 } else { -0.5 }, "coupled air arrival");
    let v = get("em_snr_db");
    add("em_snr_db", v, if v >= settings.coupled_db { 1.0 } else { -0.25 }, "coincident EM signal");

    let z: f64 = evidence.iter().map(|e| e.weight).sum();
    let other = 0.05 + 0.6 * (1.0 - evidence.len() as f64 / 3.0).max(0.0);
    let explosion = (1.0 - other) / (1.0 + (-z).exp());
    let earthquake = 1.0 - other - explosion;
    let mut labels = vec![
        (LabelId::new(EARTHQUAKE), earthquake as f32),
        (LabelId::new(EXPLOSION), explosion as f32),
        (LabelId::new(OTHER), other as f32),
    ];
    labels.sort_by(|a, b| b.1.total_cmp(&a.1));
    Verdict { labels, evidence }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::EVENT_FEATURES;

    fn vector(values: &[(&str, f64)]) -> Vec<f64> {
        let mut v = vec![f64::NAN; EVENT_FEATURES.len()];
        for (name, value) in values {
            v[event_index(name).unwrap()] = *value;
        }
        v
    }

    #[test]
    fn discriminants_vote_and_missing_ones_leave_room_for_other() {
        let settings = RuleSettings::default();
        let blast = rules(&vector(&[("p_s_log", 0.6), ("hf_lf_log", 1.0), ("located", 1.0), ("depth_m", 300.0), ("air_snr_db", 30.0)]), &settings);
        assert_eq!(blast.labels[0].0.as_str(), EXPLOSION);
        assert!(blast.labels[0].1 > 0.9);
        let quake = rules(&vector(&[("p_s_log", -0.5), ("hf_lf_log", -1.5), ("located", 1.0), ("depth_m", 8_000.0), ("first_motion", 0.0)]), &settings);
        assert_eq!(quake.labels[0].0.as_str(), EARTHQUAKE);
        let unknown = rules(&vector(&[]), &settings);
        assert_eq!(unknown.labels[0].0.as_str(), OTHER);
        assert!(unknown.evidence.is_empty());
        let total: f32 = quake.labels.iter().map(|l| l.1).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
