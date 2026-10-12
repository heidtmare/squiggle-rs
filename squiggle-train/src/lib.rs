//! Offline training: reads a dataset exported by the app, fits a classical
//! model on its event features, cross-validates, and writes a model package
//! the app imports. Nothing here runs inside the app.

use std::path::Path;

use squiggle_core::features::{EVENT_FEATURES, FEATURE_SET, event_index};
use squiggle_core::taxonomy::{NOISE, Taxonomy};
use squiggle_core::{LabelId, Timestamp};
use squiggle_io::dataset::{EVENTS_FILE, Manifest, Values, read_dataset};
use squiggle_models::metrics::{Scores, cross_validate};
use squiggle_models::package::{self, ClassCount, Input, Output, Spec, Training, Weights};
use squiggle_models::{Kind, Package, Params, Trained};

/// An exported dataset's events.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub manifest: Manifest,
    pub events: Vec<String>,
    /// Training class per event; `None` when unlabelled.
    pub classes: Vec<Option<String>>,
    /// Event features in `EVENT_FEATURES` order; NaN when missing.
    pub rows: Vec<Vec<f64>>,
}

impl Dataset {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let (events, _, manifest) = read_dataset(dir)?;
        if manifest.feature_set != FEATURE_SET {
            return Err(format!("feature set {} is not {FEATURE_SET}", manifest.feature_set));
        }
        let text = |name: &str| match events.column(name) {
            Some(Values::Text(v)) => Ok(v.clone()),
            _ => Err(format!("{EVENTS_FILE} has no text column {name}")),
        };
        let ids = text("event")?;
        let mut rows = vec![vec![f64::NAN; EVENT_FEATURES.len()]; events.rows()];
        for (j, name) in EVENT_FEATURES.iter().enumerate() {
            if let Some(Values::Number(v)) = events.column(name) {
                for (row, value) in rows.iter_mut().zip(v) {
                    row[j] = *value;
                }
            }
        }
        Ok(Self { events: ids.into_iter().map(Option::unwrap_or_default).collect(), classes: text("class")?, rows, manifest })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// The analyst's label.
    Leaf,
    /// Its root ancestor.
    Top,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Top => "top",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub kind: Kind,
    pub params: Params,
    /// Defaults to the kind's name.
    pub name: Option<String>,
    pub version: u32,
    pub level: Level,
    pub seed: u64,
    /// Upper bound; lowered to the smallest class's size.
    pub folds: usize,
    /// Train on rejected events as the noise class.
    pub noise: bool,
    /// Event features to use; all when `None`.
    pub features: Option<Vec<String>>,
}

impl Default for Options {
    fn default() -> Self {
        Self { kind: Kind::Forest, params: Params::default(), name: None, version: 1, level: Level::Leaf, seed: 1, folds: 5, noise: true, features: None }
    }
}

/// Fits `options.kind` to the labelled events of `dataset`.
pub fn train(dataset: &Dataset, options: &Options) -> Result<Package, String> {
    let taxonomy = Taxonomy { labels: dataset.manifest.labels.clone(), version: dataset.manifest.taxonomy_version };
    let features: Vec<String> = match &options.features {
        Some(names) => names.clone(),
        None => EVENT_FEATURES.iter().map(|s| s.to_string()).collect(),
    };
    let columns =
        features.iter().map(|n| event_index(n).ok_or_else(|| format!("unknown feature {n}"))).collect::<Result<Vec<usize>, String>>()?;

    let class_of = |class: &str| -> Option<String> {
        if class == NOISE {
            return options.noise.then(|| class.to_string());
        }
        let id = LabelId::new(class);
        match options.level {
            Level::Leaf => Some(class.to_string()),
            Level::Top => taxonomy.lineage(&id).last().map(|l| l.0.clone()),
        }
    };
    let labelled: Vec<(usize, String)> =
        dataset.classes.iter().enumerate().filter_map(|(i, c)| Some((i, class_of(c.as_deref()?)?))).collect();
    let rank = |label: &str| {
        let ordered = taxonomy.ordered();
        ordered.iter().position(|l| l.id.as_str() == label).unwrap_or(ordered.len())
    };
    let mut labels: Vec<String> = labelled.iter().map(|(_, c)| c.clone()).collect();
    labels.sort_by(|a, b| rank(a).cmp(&rank(b)).then(a.cmp(b)));
    labels.dedup();
    if labels.len() < 2 {
        return Err(format!("needs at least two classes; found {}", labels.len()));
    }

    let x: Vec<Vec<f64>> = labelled.iter().map(|(i, _)| columns.iter().map(|&c| dataset.rows[*i][c]).collect()).collect();
    let y: Vec<usize> = labelled.iter().map(|(_, c)| labels.iter().position(|l| l == c).expect("collected above")).collect();
    let counts: Vec<usize> = (0..labels.len()).map(|c| y.iter().filter(|&&v| v == c).count()).collect();
    let folds = options.folds.min(counts.iter().copied().min().unwrap_or(0));
    let cross_validation = if folds >= 2 {
        Scores::new(&cross_validate(options.kind, &options.params, &x, &y, labels.len(), folds, options.seed), &y, labels.len())
    } else {
        Scores::default()
    };
    let trained = Trained::fit(options.kind, &options.params, &x, &y, labels.len(), options.seed);
    let fit = Scores::new(&x.iter().map(|r| trained.predict(r)).collect::<Vec<_>>(), &y, labels.len());

    let spec = Spec {
        format: package::FORMAT.to_string(),
        id: package::model_id(options.name.as_deref().unwrap_or(options.kind.name()), options.version),
        kind: options.kind,
        created: Timestamp::now().to_string(),
        models_version: env!("CARGO_PKG_VERSION").to_string(),
        input: Input {
            kind: package::EVENT_FEATURES.to_string(),
            feature_set: dataset.manifest.feature_set.clone(),
            preprocess: squiggle_models::preprocess::DESCRIPTION.to_string(),
            features,
            settings: dataset.manifest.feature_settings.unwrap_or_default(),
        },
        output: Output { taxonomy_version: dataset.manifest.taxonomy_version, level: options.level.name().to_string(), labels: labels.clone() },
        params: options.params.clone(),
        training: Training {
            project: dataset.manifest.project.clone(),
            dataset: dataset.manifest.hashes.iter().find(|h| h.0 == EVENTS_FILE).map(|h| h.1.clone()).unwrap_or_default(),
            events: y.len(),
            seed: options.seed,
            folds: if folds >= 2 { folds } else { 0 },
            classes: labels.iter().zip(&counts).map(|(l, n)| ClassCount { label: l.clone(), events: *n }).collect(),
            fit,
            cross_validation,
        },
        weights: Weights { file: String::new(), blake3: String::new() },
    };
    Package::new(spec, trained)
}

/// Each event's probabilities in the package's label order.
pub fn predictions(package: &Package, dataset: &Dataset) -> Vec<(String, Vec<f64>)> {
    dataset.events.iter().zip(&dataset.rows).map(|(id, row)| (id.clone(), package.trained.predict(&package.row(row)))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::features::FeatureSettings;
    use squiggle_io::dataset::{FORMAT, Table, write_dataset};
    use squiggle_models::rng::Rng;

    /// 24 synthetic events: quarry blasts, earthquakes and noise, with gaps.
    fn dataset(dir: &Path) -> Dataset {
        let mut rng = Rng::new(5);
        let mut events = Table::default();
        let mut classes = Vec::new();
        let mut p_s = Vec::new();
        let mut depth = Vec::new();
        for i in 0..24 {
            let (class, ps, d) = match i % 3 {
                0 => ("quarry", 0.6, 300.0),
                1 => ("earthquake", -0.5, 8000.0),
                _ => (NOISE, 0.0, f64::NAN),
            };
            classes.push(Some(class.to_string()));
            p_s.push(if i == 3 { f64::NAN } else { ps + 0.1 * rng.normal() });
            depth.push(d + 200.0 * rng.normal());
        }
        events.text("event", (0..24).map(|i| Some(format!("e{i}"))).collect());
        events.text("class", classes);
        events.number("p_s_log", p_s);
        events.number("depth_m", depth);
        let mut labels = squiggle_core::taxonomy::seed();
        labels.push(squiggle_core::Label { id: LabelId::new("quarry"), name: "Quarry".into(), parent: Some(LabelId::new("explosion")), expects: vec![] });
        let manifest = Manifest {
            format: FORMAT.into(),
            project: "p".into(),
            created: "now".into(),
            analyst: "ana".into(),
            feature_set: FEATURE_SET.into(),
            taxonomy_version: 4,
            labels,
            noise_class: NOISE.into(),
            feature_settings: Some(FeatureSettings::default()),
            event_features: vec!["p_s_log".into(), "depth_m".into()],
            member_features: vec![],
            classes: vec![],
            events: 0,
            members: 0,
            hashes: vec![],
        };
        let mut members = Table::default();
        members.text("event", vec![]);
        write_dataset(dir, &events, &members, manifest).unwrap();
        Dataset::load(dir).unwrap()
    }

    #[test]
    fn trains_cross_validates_and_packages() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = dataset(dir.path());
        for kind in Kind::ALL {
            let options = Options { kind, ..Options::default() };
            let package = train(&dataset, &options).unwrap();
            let spec = &package.spec;
            assert_eq!(spec.output.labels, ["earthquake", "quarry", NOISE]);
            assert_eq!((spec.training.events, spec.training.folds, spec.output.taxonomy_version), (24, 5, 4));
            assert!(spec.training.cross_validation.accuracy > 0.9, "{kind:?} {:?}", spec.training.cross_validation);

            let out = tempfile::tempdir().unwrap();
            package.save(out.path()).unwrap();
            let loaded = Package::load(out.path()).unwrap();
            assert_eq!(predictions(&loaded, &dataset), predictions(&package, &dataset));
        }
    }

    #[test]
    fn top_level_and_noise_options() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = dataset(dir.path());
        let options = Options { level: Level::Top, noise: false, features: Some(vec!["p_s_log".into()]), ..Options::default() };
        let package = train(&dataset, &options).unwrap();
        assert_eq!(package.spec.output.labels, ["earthquake", "explosion"]);
        assert_eq!(package.spec.training.events, 16);
        assert_eq!(package.spec.input.features, ["p_s_log"]);
        let only_noise = Options { features: Some(vec!["nope".into()]), ..Options::default() };
        assert!(train(&dataset, &only_noise).is_err());
    }
}
