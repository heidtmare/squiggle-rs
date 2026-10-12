//! Model packages: a directory holding `spec.toml` (inputs, labels,
//! training provenance and scores) and one weights file, checksummed in the
//! spec: `model.json` (classical: preprocessing and weights),
//! `model.safetensors` (deep) or `model.onnx` (imported, `onnx` feature).

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use squiggle_core::features::{FEATURE_SET, FeatureSettings, event_index};
use squiggle_core::windows::{WINDOW_SET, WindowSpec, Windows};
use squiggle_core::{LabelId, Prediction};

use crate::Trained;
use crate::classical::{Kind, Params};
use crate::deep::{Architecture, Branch, DeepModel, Fusion};
use crate::metrics::Scores;

pub const FORMAT: &str = "heidtmare.squiggle.model/1";
pub const SPEC_FILE: &str = "spec.toml";
pub const WEIGHTS_FILE: &str = "model.json";
pub const DEEP_WEIGHTS_FILE: &str = "model.safetensors";
pub const ONNX_WEIGHTS_FILE: &str = "model.onnx";
/// Input kinds.
pub const EVENT_FEATURES: &str = "event_features";
pub const WINDOWS: &str = "windows";
/// Model kinds besides the classical ones.
pub const CNN: &str = "cnn";
pub const ONNX: &str = "onnx";

/// `heidtmare.squiggle.model/<name>@<version>`.
pub fn model_id(name: &str, version: u32) -> String {
    format!("heidtmare.squiggle.model/{name}@{version}")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    pub format: String,
    pub id: String,
    /// A classical kind (`logistic`, `svm`, `knn`, `tree`, `forest`), `cnn` or `onnx`.
    pub kind: String,
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
    /// `event_features` or `windows`.
    pub kind: String,
    pub feature_set: String,
    pub preprocess: String,
    /// Event features, in model column order (`event_features`).
    #[serde(default)]
    pub features: Vec<String>,
    /// Window definitions version (`windows`).
    #[serde(default)]
    pub window_set: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion: Option<Fusion>,
    /// Feature settings the dataset was computed with; inference uses them too.
    pub settings: FeatureSettings,
    /// One per phenomenology (`windows`).
    #[serde(default)]
    pub branches: Vec<Branch>,
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

impl Weights {
    pub fn empty() -> Self {
        Self { file: String::new(), blake3: String::new() }
    }
}

/// Probabilities in label order, and each present fusion part's by name.
pub type Probabilities = (Vec<f64>, Vec<(String, Vec<f64>)>);

/// What a model reads for one event.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    /// In `squiggle_core::features::EVENT_FEATURES` order.
    pub features: &'a [f64],
    /// In the spec's branch order; empty for feature models.
    pub windows: &'a [Windows],
}

#[derive(Debug)]
pub enum Backend {
    Classical {
        trained: Trained,
        /// Index into the event features of each model column.
        columns: Vec<usize>,
    },
    Deep(Box<DeepModel>),
    #[cfg(feature = "onnx")]
    Onnx(crate::onnx::OnnxModel),
}

#[derive(Debug)]
pub struct Package {
    pub spec: Spec,
    pub backend: Backend,
    /// blake3 of `spec.toml` then the weights file.
    pub hash: String,
    spec_text: String,
    weights: Arc<[u8]>,
}

impl PartialEq for Package {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.spec == other.spec
    }
}

impl Package {
    /// A classical model; fills the spec's weights entry.
    pub fn classical(spec: Spec, trained: &Trained) -> Result<Self, String> {
        let weights = serde_json::to_vec(trained).map_err(|e| e.to_string())?;
        Self::assemble(spec, WEIGHTS_FILE, weights)
    }

    /// A deep model; fills the spec's weights entry.
    pub fn deep(spec: Spec, model: &DeepModel) -> Result<Self, String> {
        Self::assemble(spec, DEEP_WEIGHTS_FILE, model.to_bytes()?)
    }

    /// Any model from its weights file's bytes; fills the spec's weights entry.
    pub fn assemble(mut spec: Spec, file: &str, weights: Vec<u8>) -> Result<Self, String> {
        spec.weights = Weights { file: file.to_string(), blake3: blake3::hash(&weights).to_hex().to_string() };
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
            return Err(format!("{} does not match its checksum", spec.weights.file));
        }
        if spec.input.feature_set != FEATURE_SET {
            return Err(format!("feature set {} is not {FEATURE_SET}", spec.input.feature_set));
        }
        let columns = || {
            spec.input
                .features
                .iter()
                .map(|name| event_index(name).ok_or_else(|| format!("unknown feature {name}")))
                .collect::<Result<Vec<usize>, String>>()
        };
        let backend = match (spec.kind.as_str(), spec.input.kind.as_str()) {
            (kind, EVENT_FEATURES) if Kind::parse(kind).is_some() => {
                let trained: Trained = serde_json::from_slice(weights).map_err(|e| format!("{}: {e}", spec.weights.file))?;
                let columns = columns()?;
                if trained.preprocess.mean.len() != columns.len() {
                    return Err("weights and spec disagree on the feature count".to_string());
                }
                Backend::Classical { trained, columns }
            }
            (CNN, WINDOWS) => {
                if spec.input.window_set != WINDOW_SET {
                    return Err(format!("window set {} is not {WINDOW_SET}", spec.input.window_set));
                }
                Backend::Deep(Box::new(DeepModel::from_bytes(&architecture(&spec)?, weights)?))
            }
            #[cfg(feature = "onnx")]
            (ONNX, EVENT_FEATURES) => Backend::Onnx(crate::onnx::OnnxModel::load(weights, columns()?, spec.output.labels.len())?),
            #[cfg(not(feature = "onnx"))]
            (ONNX, _) => return Err("ONNX models need squiggle built with the onnx feature".to_string()),
            (kind, input) => return Err(format!("{kind} on {input} is not supported")),
        };
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytes);
        hasher.update(weights);
        Ok(Self { spec, backend, hash: hasher.finalize().to_hex().to_string(), spec_text: text.to_string(), weights: weights.into() })
    }

    /// Writes the package into `dir`.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (name, bytes) in [(SPEC_FILE, self.spec_text.as_bytes()), (self.spec.weights.file.as_str(), &self.weights[..])] {
            std::fs::write(dir.join(name), bytes).map_err(|e| format!("{}: {e}", dir.join(name).display()))?;
        }
        Ok(())
    }

    /// Reads and verifies the package in `dir`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()));
        let spec = read(SPEC_FILE)?;
        let text = std::str::from_utf8(&spec).map_err(|e| e.to_string())?;
        let file: Spec = toml::from_str(text).map_err(|e| format!("{SPEC_FILE}: {e}"))?;
        if file.weights.file.contains(['/', '\\']) || file.weights.file.starts_with('.') {
            return Err(format!("weights file {} must sit beside the spec", file.weights.file));
        }
        Self::from_bytes(&spec, &read(&file.weights.file)?)
    }

    /// Window specs a deep model reads, in branch order; empty for feature models.
    pub fn window_specs(&self) -> Vec<WindowSpec> {
        self.spec.input.branches.iter().map(|b| b.window.clone()).collect()
    }

    /// The model's columns from event features in `squiggle_core::features::EVENT_FEATURES` order.
    pub fn row(&self, event: &[f64]) -> Vec<f64> {
        match &self.backend {
            Backend::Classical { columns, .. } => columns.iter().map(|&c| event.get(c).copied().unwrap_or(f64::NAN)).collect(),
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => model.row(event),
            Backend::Deep(_) => Vec::new(),
        }
    }

    /// Probabilities in label order, and each present fusion part's; `None`
    /// when a deep model lacks a required (or any) phenomenology.
    pub fn probabilities(&self, inputs: Inputs) -> Option<Probabilities> {
        match &self.backend {
            Backend::Classical { trained, .. } => Some((trained.predict(&self.row(inputs.features)), Vec::new())),
            Backend::Deep(model) => {
                let (p, parts) = model.predict(inputs.windows)?;
                let parts = parts.into_iter().map(|(b, p)| (self.spec.input.branches[b].window.phenomenology.0.clone(), p)).collect();
                Some((p, parts))
            }
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => model.predict(inputs.features).map(|p| (p, Vec::new())),
        }
    }

    /// Probabilities by label, highest first.
    fn labelled(&self, p: &[f64]) -> Vec<(LabelId, f32)> {
        let mut labels: Vec<(LabelId, f32)> = self.spec.output.labels.iter().zip(p).map(|(l, p)| (LabelId::new(l.clone()), *p as f32)).collect();
        labels.sort_by(|a, b| b.1.total_cmp(&a.1));
        labels
    }

    pub fn predict(&self, inputs: Inputs) -> Option<Vec<(LabelId, f32)>> {
        self.probabilities(inputs).map(|(p, _)| self.labelled(&p))
    }

    pub fn prediction(&self, inputs: Inputs) -> Option<Prediction> {
        let (p, parts) = self.probabilities(inputs)?;
        Some(Prediction {
            model: self.spec.id.clone(),
            config_hash: self.hash[..12].to_string(),
            taxonomy_version: self.spec.output.taxonomy_version,
            labels: self.labelled(&p),
            parts: parts.into_iter().map(|(name, p)| (name, self.labelled(&p))).collect(),
        })
    }
}

/// The network a deep spec describes.
pub fn architecture(spec: &Spec) -> Result<Architecture, String> {
    if spec.input.branches.is_empty() {
        return Err("a deep model needs at least one branch".to_string());
    }
    Ok(Architecture {
        fusion: spec.input.fusion.ok_or("a deep model needs a fusion")?,
        classes: spec.output.labels.len(),
        width: spec.params.width,
        hidden: spec.params.hidden,
        branches: spec.input.branches.clone(),
    })
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
            kind: "logistic".into(),
            created: "now".into(),
            models_version: env!("CARGO_PKG_VERSION").into(),
            input: Input {
                kind: EVENT_FEATURES.into(),
                feature_set: FEATURE_SET.into(),
                preprocess: crate::preprocess::DESCRIPTION.into(),
                features,
                window_set: String::new(),
                fusion: None,
                settings: FeatureSettings::default(),
                branches: vec![],
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
            weights: Weights::empty(),
        };
        Package::classical(spec, &trained).unwrap()
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
        let inputs = Inputs { features: &event, windows: &[] };
        let prediction = loaded.prediction(inputs).unwrap();
        assert_eq!(prediction.labels[0].0.as_str(), "explosion");
        assert_eq!(prediction.taxonomy_version, 3);
        assert_eq!(loaded.predict(inputs), package.predict(inputs));
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
        let Backend::Classical { trained, .. } = &package.backend else { panic!() };
        assert!(Package::classical(spec, trained).unwrap_err().contains("unknown feature"));
        let mut spec = package.spec.clone();
        spec.input.feature_set = "other/1".into();
        assert!(Package::classical(spec, trained).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_packages_round_trip() {
        let mut spec = package().spec;
        spec.kind = ONNX.into();
        let onnx = crate::onnx::tests::linear_softmax([1.0, -1.0, 0.0, 0.0], [0.0, 0.0]);
        let package = Package::assemble(spec, ONNX_WEIGHTS_FILE, onnx).unwrap();
        let dir = tempfile::tempdir().unwrap();
        package.save(dir.path()).unwrap();
        let loaded = Package::load(dir.path()).unwrap();
        let mut event = vec![f64::NAN; FEATURES.len()];
        event[event_index("p_s_log").unwrap()] = 2.0;
        let p = loaded.predict(Inputs { features: &event, windows: &[] }).unwrap();
        assert_eq!(p[0].0.as_str(), "earthquake");
        assert!((p[0].1 - 0.982_013_8).abs() < 1e-5, "{p:?}");
    }
}
