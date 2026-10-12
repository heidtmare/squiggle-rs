//! Model packages: a directory holding `spec.toml` (inputs, labels,
//! training provenance and scores) and `model.json` (preprocessing and
//! weights, checksummed in the spec).

use std::path::Path;

use serde::{Deserialize, Serialize};
use squiggle_core::features::{FEATURE_SET, FeatureSettings, event_index};
use squiggle_core::{LabelId, Prediction};

use crate::Trained;
use crate::classical::{Kind, Params};
use crate::metrics::Scores;

pub const FORMAT: &str = "heidtmare.squiggle.model/1";
pub const SPEC_FILE: &str = "spec.toml";
pub const WEIGHTS_FILE: &str = "model.json";
/// Input kind of every classical model.
pub const EVENT_FEATURES: &str = "event_features";

/// `heidtmare.squiggle.model/<name>@<version>`.
pub fn model_id(name: &str, version: u32) -> String {
    format!("heidtmare.squiggle.model/{name}@{version}")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    pub format: String,
    pub id: String,
    pub kind: Kind,
    pub created: String,
    /// `squiggle-models` version that wrote the package.
    pub models_version: String,
    pub input: Input,
    pub output: Output,
    pub params: Params,
    pub training: Training,
    pub weights: Weights,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    pub kind: String,
    pub feature_set: String,
    pub preprocess: String,
    /// Event features, in model column order.
    pub features: Vec<String>,
    /// Feature settings the dataset was computed with; inference uses them too.
    pub settings: FeatureSettings,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub taxonomy_version: u32,
    /// `leaf` (the analyst's label) or `top` (its root ancestor).
    pub level: String,
    /// In model output order; may include the noise class.
    pub labels: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Training {
    pub project: String,
    /// blake3 of the dataset's events table.
    pub dataset: String,
    pub events: usize,
    pub seed: u64,
    pub folds: usize,
    pub classes: Vec<ClassCount>,
    /// Scores on the training rows.
    pub fit: Scores,
    /// Out-of-fold scores; empty when there were too few rows to fold.
    pub cross_validation: Scores,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassCount {
    pub label: String,
    pub events: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    pub file: String,
    pub blake3: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Package {
    pub spec: Spec,
    pub trained: Trained,
    /// blake3 of `spec.toml` then `model.json`.
    pub hash: String,
    /// Index into the event features of each model column.
    columns: Vec<usize>,
}

impl Package {
    /// Fills the spec's weights checksum and checks the inputs.
    pub fn new(mut spec: Spec, trained: Trained) -> Result<Self, String> {
        let weights = serde_json::to_vec(&trained).map_err(|e| e.to_string())?;
        spec.weights = Weights { file: WEIGHTS_FILE.to_string(), blake3: blake3::hash(&weights).to_hex().to_string() };
        let text = toml::to_string(&spec).map_err(|e| e.to_string())?;
        Self::from_bytes(text.as_bytes(), &weights)
    }

    fn from_bytes(bytes: &[u8], weights: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        let spec: Spec = toml::from_str(text).map_err(|e| format!("{SPEC_FILE}: {e}"))?;
        if spec.format != FORMAT {
            return Err(format!("format {} is not {FORMAT}", spec.format));
        }
        if blake3::hash(weights).to_hex().as_str() != spec.weights.blake3 {
            return Err(format!("{WEIGHTS_FILE} does not match its checksum"));
        }
        let trained: Trained = serde_json::from_slice(weights).map_err(|e| format!("{WEIGHTS_FILE}: {e}"))?;
        if spec.input.kind != EVENT_FEATURES {
            return Err(format!("input {} is not supported", spec.input.kind));
        }
        if spec.input.feature_set != FEATURE_SET {
            return Err(format!("feature set {} is not {FEATURE_SET}", spec.input.feature_set));
        }
        let columns = spec
            .input
            .features
            .iter()
            .map(|name| event_index(name).ok_or_else(|| format!("unknown feature {name}")))
            .collect::<Result<Vec<usize>, String>>()?;
        if trained.preprocess.mean.len() != columns.len() {
            return Err("weights and spec disagree on the feature count".to_string());
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytes);
        hasher.update(weights);
        Ok(Self { spec, trained, hash: hasher.finalize().to_hex().to_string(), columns })
    }

    /// Writes the package into `dir`.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let weights = serde_json::to_vec(&self.trained).map_err(|e| e.to_string())?;
        for (name, bytes) in [(SPEC_FILE, spec_bytes(&self.spec)?.into_bytes()), (WEIGHTS_FILE, weights)] {
            std::fs::write(dir.join(name), bytes).map_err(|e| format!("{}: {e}", dir.join(name).display()))?;
        }
        Ok(())
    }

    /// Reads and verifies the package in `dir`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()));
        Self::from_bytes(&read(SPEC_FILE)?, &read(WEIGHTS_FILE)?)
    }

    /// The model's inputs from event features in `squiggle_core::features::EVENT_FEATURES` order.
    pub fn row(&self, event: &[f64]) -> Vec<f64> {
        self.columns.iter().map(|&c| event.get(c).copied().unwrap_or(f64::NAN)).collect()
    }

    /// Probabilities by label, highest first.
    pub fn predict(&self, event: &[f64]) -> Vec<(LabelId, f32)> {
        let p = self.trained.predict(&self.row(event));
        let mut labels: Vec<(LabelId, f32)> = self.spec.output.labels.iter().zip(p).map(|(l, p)| (LabelId::new(l.clone()), p as f32)).collect();
        labels.sort_by(|a, b| b.1.total_cmp(&a.1));
        labels
    }

    pub fn prediction(&self, event: &[f64]) -> Prediction {
        Prediction {
            model: self.spec.id.clone(),
            config_hash: self.hash[..12].to_string(),
            taxonomy_version: self.spec.output.taxonomy_version,
            labels: self.predict(event),
        }
    }
}

fn spec_bytes(spec: &Spec) -> Result<String, String> {
    toml::to_string(spec).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::features::EVENT_FEATURES as FEATURES;

    fn package() -> Package {
        let features = vec!["p_s_log".to_string(), "depth_m".to_string()];
        let x = vec![vec![0.6, 300.0], vec![0.7, f64::NAN], vec![-0.5, 8000.0], vec![-0.6, 6000.0]];
        let y = vec![1, 1, 0, 0];
        let trained = Trained::fit(Kind::Logistic, &Params::default(), &x, &y, 2, 1);
        let spec = Spec {
            format: FORMAT.to_string(),
            id: model_id("logistic", 1),
            kind: Kind::Logistic,
            created: "now".into(),
            models_version: env!("CARGO_PKG_VERSION").into(),
            input: Input {
                kind: EVENT_FEATURES.into(),
                feature_set: FEATURE_SET.into(),
                preprocess: crate::preprocess::DESCRIPTION.into(),
                features,
                settings: FeatureSettings::default(),
            },
            output: Output { taxonomy_version: 3, level: "leaf".into(), labels: vec!["earthquake".into(), "explosion".into()] },
            params: Params::default(),
            training: Training {
                project: "p".into(),
                dataset: "d".into(),
                events: 4,
                seed: 1,
                folds: 2,
                classes: vec![ClassCount { label: "earthquake".into(), events: 2 }],
                fit: Scores { accuracy: 1.0, log_loss: 0.1, confusion: vec![vec![2, 0], vec![0, 2]] },
                cross_validation: Scores::default(),
            },
            weights: Weights { file: String::new(), blake3: String::new() },
        };
        Package::new(spec, trained).unwrap()
    }

    #[test]
    fn packages_round_trip_and_predict_by_feature_name() {
        let package = package();
        let dir = tempfile::tempdir().unwrap();
        package.save(dir.path()).unwrap();
        let loaded = Package::load(dir.path()).unwrap();
        assert_eq!(loaded, package);

        let mut event = vec![f64::NAN; FEATURES.len()];
        event[event_index("p_s_log").unwrap()] = 0.65;
        event[event_index("depth_m").unwrap()] = 200.0;
        let prediction = loaded.prediction(&event);
        assert_eq!(prediction.labels[0].0.as_str(), "explosion");
        assert_eq!(prediction.taxonomy_version, 3);
        assert_eq!(loaded.predict(&event), package.predict(&event));
    }

    #[test]
    fn tampered_or_mismatched_packages_are_refused() {
        let package = package();
        let dir = tempfile::tempdir().unwrap();
        package.save(dir.path()).unwrap();
        let weights = dir.path().join(WEIGHTS_FILE);
        let text = std::fs::read_to_string(&weights).unwrap();
        std::fs::write(&weights, text.replacen("0.", "1.", 1)).unwrap();
        assert!(Package::load(dir.path()).unwrap_err().contains("checksum"));

        let mut spec = package.spec.clone();
        spec.input.features[0] = "nope".into();
        assert!(Package::new(spec, package.trained.clone()).unwrap_err().contains("unknown feature"));
        let mut spec = package.spec.clone();
        spec.input.feature_set = "other/1".into();
        assert!(Package::new(spec, package.trained.clone()).is_err());
    }
}
