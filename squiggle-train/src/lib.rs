//! Offline training: reads a dataset exported by the app, fits a classical
//! model on its event features or a CNN on its waveform windows,
//! cross-validates, and writes a model package the app imports. Nothing here
//! runs inside the app.

use std::path::Path;

use squiggle_core::features::{COMPATIBLE_FEATURE_SETS, EVENT_FEATURES, FEATURE_SET, event_index};
use squiggle_core::taxonomy::{NOISE, Taxonomy};
use squiggle_core::windows::{WINDOW_SET, WindowSpec, Windows};
use squiggle_core::{LabelId, Timestamp};
use squiggle_io::dataset::{EVENTS_FILE, Manifest, Values, WINDOWS_FILE, read_dataset, read_windows};
use squiggle_models::deep::{self, Architecture, Branch, DeepModel, Encoder, Example, Fusion};
use squiggle_models::metrics::{Scores, cross_validate, stratified_folds};
use squiggle_models::cluster::{ClusterParams, Embedding};
use squiggle_models::package::{self, ClassCount, ClusterSummary, Input, Inputs, Output, Spec, Training, Weights};
use squiggle_models::{Kind, Package, Params, Trained};

/// One phenomenology's windows in a dataset.
#[derive(Clone, Debug)]
pub struct WindowSet {
    pub spec: WindowSpec,
    /// Per event row, its windows.
    pub events: Vec<Vec<Vec<f32>>>,
}

/// An exported dataset's events.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub manifest: Manifest,
    pub events: Vec<String>,
    /// Training class per event; `None` when unlabelled.
    pub classes: Vec<Option<String>>,
    /// Event features in `EVENT_FEATURES` order; NaN when missing.
    pub rows: Vec<Vec<f64>>,
    /// In manifest order; empty when the dataset has no windows.
    pub windows: Vec<WindowSet>,
}

impl Dataset {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let (events, _, manifest) = read_dataset(dir)?;
        if !COMPATIBLE_FEATURE_SETS.contains(&manifest.feature_set.as_str()) {
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
        let mut windows = Vec::new();
        if !manifest.windows.is_empty() {
            let tables = read_windows(&dir.join(WINDOWS_FILE))?;
            for spec in &manifest.windows {
                let mut set = WindowSet { spec: spec.clone(), events: vec![Vec::new(); rows.len()] };
                if let Some(table) = tables.iter().find(|t| t.phenomenology == spec.phenomenology.0) {
                    for (w, &event) in table.values.chunks_exact(table.samples.max(1)).zip(&table.events) {
                        let slot = set.events.get_mut(event as usize).ok_or_else(|| format!("window for missing event row {event}"))?;
                        slot.push(w.to_vec());
                    }
                }
                windows.push(set);
            }
        }
        Ok(Self { events: ids.into_iter().map(Option::unwrap_or_default).collect(), classes: text("class")?, rows, manifest, windows })
    }

    /// Row `row`'s windows for `specs` (by phenomenology; empty when absent).
    pub fn windows_for(&self, row: usize, specs: &[WindowSpec]) -> Vec<Windows> {
        specs
            .iter()
            .map(|spec| Windows {
                traces: self.windows.iter().find(|w| w.spec.phenomenology == spec.phenomenology).map(|w| w.events[row].clone()).unwrap_or_default(),
                sources: Vec::new(),
            })
            .collect()
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// On event features.
    Classical(Kind),
    /// On waveform windows, one branch per phenomenology.
    Cnn { encoder: Encoder, fusion: Fusion },
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Self::Classical(kind) => kind.name(),
            Self::Cnn { .. } => package::CNN,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub method: Method,
    pub params: Params,
    /// Defaults to the method's name.
    pub name: Option<String>,
    pub version: u32,
    pub level: Level,
    pub seed: u64,
    /// Upper bound; lowered to the smallest class's size.
    pub folds: usize,
    /// Train on rejected events as the noise class.
    pub noise: bool,
    /// Event features to use (classical); all when `None`.
    pub features: Option<Vec<String>>,
    /// Phenomenologies to use (cnn); every one with windows when `None`.
    pub branches: Option<Vec<String>>,
    /// Phenomenologies a prediction requires (cnn).
    pub require: Vec<String>,
    /// Train on the GPU (cnn; needs the `gpu` feature).
    pub gpu: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            method: Method::Classical(Kind::Forest),
            params: Params::default(),
            name: None,
            version: 1,
            level: Level::Leaf,
            seed: 1,
            folds: 5,
            noise: true,
            features: None,
            branches: None,
            require: Vec::new(),
            gpu: false,
        }
    }
}

/// Labelled rows, their class indices, and the label names in taxonomy order.
type Classes = (Vec<usize>, Vec<usize>, Vec<String>);

fn classes(dataset: &Dataset, options: &Options) -> Result<Classes, String> {
    let taxonomy = Taxonomy { labels: dataset.manifest.labels.clone(), version: dataset.manifest.taxonomy_version };
    let class_of = |class: &str| -> Option<String> {
        if class == NOISE {
            return options.noise.then(|| class.to_string());
        }
        match options.level {
            Level::Leaf => Some(class.to_string()),
            Level::Top => taxonomy.lineage(&LabelId::new(class)).last().map(|l| l.0.clone()),
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
    let y = labelled.iter().map(|(_, c)| labels.iter().position(|l| l == c).expect("collected above")).collect();
    Ok((labelled.into_iter().map(|(i, _)| i).collect(), y, labels))
}

/// Fits `options.method` to the labelled events of `dataset`.
pub fn train(dataset: &Dataset, options: &Options) -> Result<Package, String> {
    let (rows, y, labels) = classes(dataset, options)?;
    let counts: Vec<usize> = (0..labels.len()).map(|c| y.iter().filter(|&&v| v == c).count()).collect();
    let folds = options.folds.min(counts.iter().copied().min().unwrap_or(0));
    let folds = if folds >= 2 { folds } else { 0 };
    let mut spec = Spec {
        format: package::FORMAT.to_string(),
        id: package::model_id(options.name.as_deref().unwrap_or(options.method.name()), options.version),
        kind: options.method.name().to_string(),
        created: Timestamp::now().to_string(),
        models_version: env!("CARGO_PKG_VERSION").to_string(),
        input: Input {
            kind: package::EVENT_FEATURES.to_string(),
            feature_set: dataset.manifest.feature_set.clone(),
            preprocess: String::new(),
            features: Vec::new(),
            window_set: String::new(),
            fusion: None,
            settings: dataset.manifest.feature_settings.unwrap_or_default(),
            branches: Vec::new(),
        },
        output: Output { taxonomy_version: dataset.manifest.taxonomy_version, level: options.level.name().to_string(), labels: labels.clone(), clusters: Vec::new() },
        params: options.params.clone(),
        cluster: None,
        training: Training {
            project: dataset.manifest.project.clone(),
            dataset: dataset.manifest.hashes.iter().find(|h| h.0 == EVENTS_FILE).map(|h| h.1.clone()).unwrap_or_default(),
            events: y.len(),
            seed: options.seed,
            folds,
            classes: labels.iter().zip(&counts).map(|(l, n)| ClassCount { label: l.clone(), events: *n }).collect(),
            fit: Scores::default(),
            cross_validation: Scores::default(),
        },
        weights: Weights::empty(),
    };

    match options.method {
        Method::Classical(kind) => {
            let features: Vec<String> = match &options.features {
                Some(names) => names.clone(),
                None => EVENT_FEATURES.iter().map(|s| s.to_string()).collect(),
            };
            let columns =
                features.iter().map(|n| event_index(n).ok_or_else(|| format!("unknown feature {n}"))).collect::<Result<Vec<usize>, String>>()?;
            let x: Vec<Vec<f64>> = rows.iter().map(|&i| columns.iter().map(|&c| dataset.rows[i][c]).collect()).collect();
            if folds > 0 {
                spec.training.cross_validation =
                    Scores::new(&cross_validate(kind, &options.params, &x, &y, labels.len(), folds, options.seed), &y, labels.len());
            }
            let trained = Trained::fit(kind, &options.params, &x, &y, labels.len(), options.seed);
            spec.training.fit = Scores::new(&x.iter().map(|r| trained.predict(r)).collect::<Vec<_>>(), &y, labels.len());
            spec.input.preprocess = squiggle_models::preprocess::DESCRIPTION.to_string();
            spec.input.features = features;
            Package::classical(spec, &trained)
        }
        Method::Cnn { encoder, fusion } => {
            if dataset.windows.is_empty() {
                return Err("the dataset has no waveform windows; export it with windows".to_string());
            }
            let sets: Vec<&WindowSet> = match &options.branches {
                Some(names) => names
                    .iter()
                    .map(|n| dataset.windows.iter().find(|w| w.spec.phenomenology.0 == *n).ok_or_else(|| format!("the dataset has no {n} windows")))
                    .collect::<Result<_, _>>()?,
                None => dataset.windows.iter().collect(),
            };
            if let Some(r) = options.require.iter().find(|r| !sets.iter().any(|s| s.spec.phenomenology.0 == **r)) {
                return Err(format!("required {r} is not a branch"));
            }
            let branches: Vec<Branch> = sets
                .iter()
                .map(|s| Branch { required: options.require.contains(&s.spec.phenomenology.0), encoder, window: s.spec.clone() })
                .collect();
            let arch = Architecture { fusion, classes: labels.len(), width: options.params.width, hidden: options.params.hidden, branches };
            let examples: Vec<Example> =
                rows.iter().zip(&y).map(|(&i, &class)| Example { windows: sets.iter().map(|s| s.events[i].clone()).collect(), class }).collect();
            let device = deep::device(options.gpu)?;
            let specs: Vec<WindowSpec> = arch.branches.iter().map(|b| b.window.clone()).collect();
            let windows = |i: usize| -> Vec<Windows> { examples[i].windows.iter().map(|w| Windows { traces: w.clone(), sources: Vec::new() }).collect() };
            let uniform = vec![1.0 / labels.len() as f64; labels.len()];
            let predict = |model: &DeepModel, i: usize| model.predict(&windows(i)).map_or_else(|| uniform.clone(), |p| p.0);
            if folds > 0 {
                let fold = stratified_folds(&y, folds, options.seed);
                let mut out = vec![Vec::new(); y.len()];
                for k in 0..folds {
                    let train: Vec<Example> = (0..y.len()).filter(|&i| fold[i] != k).map(|i| examples[i].clone()).collect();
                    let model = DeepModel::train(&arch, &train, &options.params, options.seed, &device);
                    for i in (0..y.len()).filter(|&i| fold[i] == k) {
                        out[i] = predict(&model, i);
                    }
                }
                spec.training.cross_validation = Scores::new(&out, &y, labels.len());
            }
            let model = DeepModel::train(&arch, &examples, &options.params, options.seed, &device);
            spec.training.fit = Scores::new(&(0..y.len()).map(|i| predict(&model, i)).collect::<Vec<_>>(), &y, labels.len());
            spec.input.kind = package::WINDOWS.to_string();
            spec.input.preprocess = "windows demeaned to unit peak; log-spectrograms standardized per window".to_string();
            spec.input.window_set = WINDOW_SET.to_string();
            spec.input.fusion = Some(fusion);
            spec.input.branches = arch.branches.clone();
            debug_assert_eq!(specs.len(), spec.input.branches.len());
            Package::deep(spec, &model)
        }
    }
}

/// Each event's probabilities in the package's label order; `None` where the model abstains.
pub fn predictions(package: &Package, dataset: &Dataset) -> Vec<(String, Option<Vec<f64>>)> {
    let specs = package.window_specs();
    dataset
        .events
        .iter()
        .zip(&dataset.rows)
        .enumerate()
        .map(|(i, (id, row))| {
            let windows = dataset.windows_for(i, &specs);
            (id.clone(), package.probabilities(Inputs { features: row, windows: &windows }).map(|p| p.0))
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct ClusterOptions {
    /// Defaults to `cluster`.
    pub name: Option<String>,
    pub version: u32,
    /// Event features to use; all when `None`.
    pub features: Option<Vec<String>>,
    pub params: ClusterParams,
}

/// Fits PCA and DBSCAN to every event of `dataset`, labelled or not;
/// returns the package and each event's cluster.
pub fn cluster(dataset: &Dataset, options: &ClusterOptions) -> Result<(Package, Vec<Option<usize>>), String> {
    let features: Vec<String> = match &options.features {
        Some(names) => names.clone(),
        None => EVENT_FEATURES.iter().map(|s| s.to_string()).collect(),
    };
    let columns = features.iter().map(|n| event_index(n).ok_or_else(|| format!("unknown feature {n}"))).collect::<Result<Vec<usize>, String>>()?;
    if dataset.rows.len() < options.params.min_points.max(2) {
        return Err(format!("needs at least {} events; found {}", options.params.min_points.max(2), dataset.rows.len()));
    }
    let x: Vec<Vec<f64>> = dataset.rows.iter().map(|r| columns.iter().map(|&c| r[c]).collect()).collect();
    let (embedding, assigned) = Embedding::fit(&x, &options.params);
    let count = |rows: &mut dyn Iterator<Item = usize>| {
        let mut classes: Vec<ClassCount> = Vec::new();
        for i in rows {
            let Some(class) = &dataset.classes[i] else { continue };
            match classes.iter_mut().find(|c| &c.label == class) {
                Some(c) => c.events += 1,
                None => classes.push(ClassCount { label: class.clone(), events: 1 }),
            }
        }
        classes.sort_by(|a, b| b.events.cmp(&a.events).then(a.label.cmp(&b.label)));
        classes
    };
    let clusters = (0..embedding.clusters)
        .map(|k| ClusterSummary {
            cluster: k,
            events: assigned.iter().filter(|a| **a == Some(k)).count(),
            classes: count(&mut (0..assigned.len()).filter(|&i| assigned[i] == Some(k))),
        })
        .collect();
    let spec = Spec {
        format: package::FORMAT.to_string(),
        id: package::model_id(options.name.as_deref().unwrap_or(package::CLUSTER), options.version.max(1)),
        kind: package::CLUSTER.to_string(),
        created: Timestamp::now().to_string(),
        models_version: env!("CARGO_PKG_VERSION").to_string(),
        input: Input {
            kind: package::EVENT_FEATURES.to_string(),
            feature_set: dataset.manifest.feature_set.clone(),
            preprocess: squiggle_models::preprocess::DESCRIPTION.to_string(),
            features,
            window_set: String::new(),
            fusion: None,
            settings: dataset.manifest.feature_settings.unwrap_or_default(),
            branches: Vec::new(),
        },
        output: Output { taxonomy_version: dataset.manifest.taxonomy_version, level: Level::Leaf.name().to_string(), labels: Vec::new(), clusters },
        params: Params::default(),
        cluster: Some(ClusterParams { components: embedding.components.len(), eps: embedding.eps, min_points: embedding.min_points }),
        training: Training {
            project: dataset.manifest.project.clone(),
            dataset: dataset.manifest.hashes.iter().find(|h| h.0 == EVENTS_FILE).map(|h| h.1.clone()).unwrap_or_default(),
            events: dataset.rows.len(),
            seed: 0,
            folds: 0,
            classes: count(&mut (0..dataset.rows.len())),
            fit: Scores::default(),
            cross_validation: Scores::default(),
        },
        weights: Weights::empty(),
    };
    Ok((Package::cluster(spec, &embedding)?, assigned))
}

/// Adjusted Rand index of two labellings (`None` is its own singleton): 1 when identical, about 0 when random.
pub fn adjusted_rand(a: &[Option<usize>], b: &[Option<usize>]) -> f64 {
    let n = a.len().min(b.len());
    let key = |l: Option<usize>, i: usize| l.map_or(usize::MAX - i, |v| v);
    let mut table: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
    let mut rows: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut cols: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for i in 0..n {
        let (x, y) = (key(a[i], i), key(b[i], i));
        *table.entry((x, y)).or_default() += 1.0;
        *rows.entry(x).or_default() += 1.0;
        *cols.entry(y).or_default() += 1.0;
    }
    let pairs = |v: f64| v * (v - 1.0) / 2.0;
    let index: f64 = table.values().copied().map(pairs).sum();
    let (sa, sb): (f64, f64) = (rows.values().copied().map(pairs).sum(), cols.values().copied().map(pairs).sum());
    let expected = sa * sb / pairs(n as f64).max(1.0);
    let max = (sa + sb) / 2.0;
    if (max - expected).abs() < 1e-12 { 1.0 } else { (index - expected) / (max - expected) }
}

/// Packages an ONNX graph on event features: `[1, features]` in, `[1, labels]` out.
pub fn package_onnx(onnx: Vec<u8>, features: Vec<String>, labels: Vec<String>, name: &str, version: u32, taxonomy_version: u32) -> Result<Package, String> {
    if let Some(f) = features.iter().find(|f| event_index(f).is_none()) {
        return Err(format!("unknown feature {f}"));
    }
    let spec = Spec {
        format: package::FORMAT.to_string(),
        id: package::model_id(name, version),
        kind: package::ONNX.to_string(),
        created: Timestamp::now().to_string(),
        models_version: env!("CARGO_PKG_VERSION").to_string(),
        input: Input {
            kind: package::EVENT_FEATURES.to_string(),
            feature_set: FEATURE_SET.to_string(),
            preprocess: "none; missing values as 0".to_string(),
            features,
            window_set: String::new(),
            fusion: None,
            settings: Default::default(),
            branches: Vec::new(),
        },
        output: Output { taxonomy_version, level: Level::Leaf.name().to_string(), labels, clusters: Vec::new() },
        params: Params::default(),
        cluster: None,
        training: Training {
            project: String::new(),
            dataset: String::new(),
            events: 0,
            seed: 0,
            folds: 0,
            classes: Vec::new(),
            fit: Scores::default(),
            cross_validation: Scores::default(),
        },
        weights: Weights::empty(),
    };
    Package::assemble(spec, package::ONNX_WEIGHTS_FILE, onnx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::features::FeatureSettings;
    use squiggle_io::dataset::{FORMAT, Table, WindowTable, write_dataset};
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
            windows: ["seismic", "infrasound"]
                .iter()
                .map(|p| WindowSpec { phenomenology: squiggle_core::PhenomenologyId::new(*p), rate: 20.0, pre_s: 0.5, samples: 64, max_traces: 4 })
                .collect(),
            event_features: vec!["p_s_log".into(), "depth_m".into()],
            member_features: vec![],
            classes: vec![],
            events: 0,
            members: 0,
            hashes: vec![],
        };
        let mut members = Table::default();
        members.text("event", vec![]);
        // Seismic: two traces, low tone for earthquakes, high for quarries, noise for noise.
        // Infrasound: one trace with a pulse for quarries only.
        let mut seismic = WindowTable { phenomenology: "seismic".into(), samples: 64, values: vec![], events: vec![] };
        let mut infrasound = WindowTable { phenomenology: "infrasound".into(), samples: 64, ..seismic.clone() };
        for i in 0..24 {
            for _ in 0..2 {
                let f = [6.0, 1.5, 0.0][i % 3];
                let phase = rng.uniform() * 6.0;
                seismic.values.extend((0..64).map(|t| ((std::f64::consts::TAU * f * t as f64 / 20.0 + phase).sin() + 0.3 * rng.normal()) as f32));
                seismic.events.push(i as i64);
            }
            infrasound.values.extend((0..64).map(|t| if i % 3 == 0 && (10..16).contains(&t) { 1.0 } else { 0.1 * rng.normal() as f32 }));
            infrasound.events.push(i as i64);
        }
        write_dataset(dir, &events, &members, &[seismic, infrasound], manifest).unwrap();
        Dataset::load(dir).unwrap()
    }

    #[test]
    fn trains_cross_validates_and_packages() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = dataset(dir.path());
        for kind in Kind::ALL {
            let options = Options { method: Method::Classical(kind), ..Options::default() };
            let package = train(&dataset, &options).unwrap();
            let spec = &package.spec;
            assert_eq!(spec.output.labels, ["earthquake", "quarry", NOISE]);
            assert_eq!((spec.training.events, spec.training.folds, spec.output.taxonomy_version), (24, 5, 4));
            assert!(spec.training.cross_validation.accuracy > 0.9, "{kind:?} {:?}", spec.training.cross_validation);

            let out = tempfile::tempdir().unwrap();
            package.save(out.path()).unwrap();
            let loaded = Package::load(out.path()).unwrap();
            assert_eq!(predictions(&loaded, &dataset), predictions(&package, &dataset));
            assert!(predictions(&loaded, &dataset).iter().all(|p| p.1.is_some()));
        }
    }

    #[test]
    fn cnns_train_on_windows_and_respect_required_branches() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = dataset(dir.path());
        assert_eq!(dataset.windows.iter().map(|w| w.events[0].len()).collect::<Vec<_>>(), [2, 1]);
        let params = Params { epochs: 30, batch: 8, learning_rate: 0.01, ..Params::default() };
        let options = Options { method: Method::Cnn { encoder: Encoder::Waveform, fusion: Fusion::Late }, params, folds: 3, ..Options::default() };
        let package = train(&dataset, &options).unwrap();
        let spec = &package.spec;
        assert_eq!((spec.kind.as_str(), spec.input.kind.as_str(), spec.input.branches.len()), ("cnn", "windows", 2));
        assert!(spec.training.cross_validation.accuracy >= 0.8, "{:?}", spec.training.cross_validation);
        assert!(spec.training.fit.accuracy >= 0.9, "{:?}", spec.training.fit);

        let out = tempfile::tempdir().unwrap();
        package.save(out.path()).unwrap();
        assert!(out.path().join(squiggle_models::package::DEEP_WEIGHTS_FILE).exists());
        let loaded = Package::load(out.path()).unwrap();
        assert_eq!(predictions(&loaded, &dataset), predictions(&package, &dataset));

        let required = Options { require: vec!["infrasound".into()], branches: Some(vec!["infrasound".into()]), folds: 0, ..options.clone() };
        let package = train(&dataset, &required).unwrap();
        let none = dataset.windows_for(0, &[WindowSpec { phenomenology: squiggle_core::PhenomenologyId::new("seismic"), ..dataset.windows[0].spec.clone() }]);
        assert!(package.probabilities(Inputs { features: &dataset.rows[0], windows: &[Windows { traces: vec![], sources: vec![] }] }).is_none());
        assert_eq!(none[0].traces.len(), 2);
        assert!(train(&dataset, &Options { branches: Some(vec!["emp".into()]), ..options }).is_err());
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
