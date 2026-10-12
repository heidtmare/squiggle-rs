//! Classical classifiers on standardized feature rows, each giving class
//! probabilities. Pure Rust and deterministic for a seed, so the trainer and
//! the app compute the same numbers from the same weights.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Logistic,
    Svm,
    Knn,
    Tree,
    Forest,
}

impl Kind {
    pub const ALL: [Self; 5] = [Self::Logistic, Self::Svm, Self::Knn, Self::Tree, Self::Forest];

    pub fn name(self) -> &'static str {
        match self {
            Self::Logistic => "logistic",
            Self::Svm => "svm",
            Self::Knn => "knn",
            Self::Tree => "tree",
            Self::Forest => "forest",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// Hyperparameters; each kind reads its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    /// L2 penalty (logistic, svm).
    pub l2: f64,
    /// Gradient steps (logistic) or epochs (svm).
    pub iterations: usize,
    /// Step size (logistic).
    pub rate: f64,
    /// Neighbours (knn).
    pub k: usize,
    /// Deepest split (tree, forest).
    pub max_depth: usize,
    /// Fewest rows in a leaf (tree, forest).
    pub min_leaf: usize,
    /// Trees (forest).
    pub trees: usize,
    /// Passes over the data (deep).
    pub epochs: usize,
    /// Events per step (deep).
    pub batch: usize,
    /// Adam step size (deep).
    pub learning_rate: f64,
    /// Chance of dropping each present branch per step, keeping one (deep).
    pub branch_dropout: f64,
    /// Encoder channels (deep).
    pub width: usize,
    /// Classifier hidden units (deep).
    pub hidden: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            l2: 0.01,
            iterations: 500,
            rate: 0.5,
            k: 5,
            max_depth: 8,
            min_leaf: 1,
            trees: 100,
            epochs: 40,
            batch: 16,
            learning_rate: 3e-3,
            branch_dropout: 0.3,
            width: 16,
            hidden: 32,
        }
    }
}

/// Multinomial logistic regression, full-batch gradient descent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Logistic {
    /// Per class: one weight per feature, then the bias.
    pub weights: Vec<Vec<f64>>,
}

impl Logistic {
    pub fn fit(x: &[Vec<f64>], y: &[usize], classes: usize, params: &Params) -> Self {
        let d = x.first().map_or(0, Vec::len);
        let mut weights = vec![vec![0.0; d + 1]; classes];
        let n = x.len().max(1) as f64;
        for _ in 0..params.iterations {
            let mut gradient = vec![vec![0.0; d + 1]; classes];
            for (row, &class) in x.iter().zip(y) {
                let p = softmax(&scores(&weights, row));
                for (c, g) in gradient.iter_mut().enumerate() {
                    let error = p[c] - if c == class { 1.0 } else { 0.0 };
                    for (g, v) in g.iter_mut().zip(row.iter().chain([&1.0])) {
                        *g += error * v / n;
                    }
                }
            }
            for (w, g) in weights.iter_mut().zip(&gradient) {
                for j in 0..=d {
                    let penalty = if j < d { params.l2 * w[j] } else { 0.0 };
                    w[j] -= params.rate * (g[j] + penalty);
                }
            }
        }
        Self { weights }
    }

    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        softmax(&scores(&self.weights, row))
    }
}

/// Linear SVM, one-vs-rest (Pegasos, averaged); probabilities are a softmax
/// of the margins with a scale fitted to the training log loss.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Svm {
    pub weights: Vec<Vec<f64>>,
    pub scale: f64,
}

impl Svm {
    pub fn fit(x: &[Vec<f64>], y: &[usize], classes: usize, params: &Params, rng: &mut Rng) -> Self {
        let d = x.first().map_or(0, Vec::len);
        let lambda = params.l2.max(1e-6);
        let mut order: Vec<usize> = (0..x.len()).collect();
        let mut weights = Vec::with_capacity(classes);
        for c in 0..classes {
            let mut w = vec![0.0; d + 1];
            let mut mean = vec![0.0; d + 1];
            let mut t = 0usize;
            for _ in 0..params.iterations {
                rng.shuffle(&mut order);
                for &i in &order {
                    t += 1;
                    let eta = 1.0 / (lambda * t as f64);
                    let target = if y[i] == c { 1.0 } else { -1.0 };
                    let margin = target * dot(&w, &x[i]);
                    for v in &mut w {
                        *v *= 1.0 - eta * lambda;
                    }
                    if margin < 1.0 {
                        for (v, xi) in w.iter_mut().zip(x[i].iter().chain([&1.0])) {
                            *v += eta * target * xi;
                        }
                    }
                    for (m, v) in mean.iter_mut().zip(&w) {
                        *m += (v - *m) / t as f64;
                    }
                }
            }
            weights.push(mean);
        }
        let mut svm = Self { weights, scale: 1.0 };
        let loss = |scale: f64| -> f64 {
            x.iter().zip(y).map(|(row, &c)| -softmax(&scaled(&scores(&svm.weights, row), scale))[c].max(1e-12).ln()).sum()
        };
        svm.scale = (0..=60).map(|i| 10f64.powf(-1.0 + i as f64 / 20.0)).min_by(|a, b| loss(*a).total_cmp(&loss(*b))).unwrap_or(1.0);
        svm
    }

    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        softmax(&scaled(&scores(&self.weights, row), self.scale))
    }
}

/// k nearest neighbours, inverse-distance weighted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Knn {
    pub k: usize,
    pub classes: usize,
    pub rows: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
}

impl Knn {
    pub fn fit(x: &[Vec<f64>], y: &[usize], classes: usize, params: &Params) -> Self {
        Self { k: params.k.max(1), classes, rows: x.to_vec(), labels: y.to_vec() }
    }

    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        let mut nearest: Vec<(f64, usize)> =
            self.rows.iter().zip(&self.labels).map(|(r, &c)| (r.iter().zip(row).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt(), c)).collect();
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut p = vec![0.0; self.classes];
        for (distance, class) in nearest.into_iter().take(self.k) {
            p[class] += 1.0 / (distance + 1e-9);
        }
        normalise(p)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Node {
    Leaf { p: Vec<f64> },
    /// Rows with `feature <= threshold` go left.
    Split { feature: usize, threshold: f64, left: usize, right: usize },
}

/// CART on Gini impurity; leaves hold class frequencies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    /// `features`: candidates drawn per split (all when `None`).
    pub fn fit(x: &[Vec<f64>], y: &[usize], rows: Vec<usize>, classes: usize, params: &Params, features: Option<usize>, rng: &mut Rng) -> Self {
        let mut tree = Self { nodes: Vec::new() };
        tree.grow(x, y, rows, classes, params, features, rng, 0);
        tree
    }

    #[allow(clippy::too_many_arguments)]
    fn grow(&mut self, x: &[Vec<f64>], y: &[usize], rows: Vec<usize>, classes: usize, params: &Params, features: Option<usize>, rng: &mut Rng, depth: usize) -> usize {
        let counts = count(y, &rows, classes);
        let index = self.nodes.len();
        self.nodes.push(Node::Leaf { p: normalise(counts.iter().map(|&c| c as f64).collect()) });
        if depth >= params.max_depth || counts.iter().filter(|&&c| c > 0).count() < 2 {
            return index;
        }
        let d = x.first().map_or(0, Vec::len);
        let mut candidates: Vec<usize> = (0..d).collect();
        if let Some(m) = features {
            rng.shuffle(&mut candidates);
            candidates.truncate(m.clamp(1, d.max(1)));
            candidates.sort_unstable();
        }
        let Some((feature, threshold)) = best_split(x, y, &rows, classes, &candidates, params.min_leaf.max(1)) else {
            return index;
        };
        let (left, right): (Vec<usize>, Vec<usize>) = rows.into_iter().partition(|&i| x[i][feature] <= threshold);
        let left = self.grow(x, y, left, classes, params, features, rng, depth + 1);
        let right = self.grow(x, y, right, classes, params, features, rng, depth + 1);
        self.nodes[index] = Node::Split { feature, threshold, left, right };
        index
    }

    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        let mut i = 0;
        loop {
            match &self.nodes[i] {
                Node::Leaf { p } => return p.clone(),
                Node::Split { feature, threshold, left, right } => i = if row[*feature] <= *threshold { *left } else { *right },
            }
        }
    }
}

/// Bagged trees, √d candidate features per split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Forest {
    pub trees: Vec<Tree>,
}

impl Forest {
    pub fn fit(x: &[Vec<f64>], y: &[usize], classes: usize, params: &Params, rng: &mut Rng) -> Self {
        let d = x.first().map_or(0, Vec::len);
        let m = ((d as f64).sqrt().round() as usize).max(1);
        let trees = (0..params.trees.max(1))
            .map(|_| {
                let rows: Vec<usize> = (0..x.len()).map(|_| rng.below(x.len())).collect();
                Tree::fit(x, y, rows, classes, params, Some(m), rng)
            })
            .collect();
        Self { trees }
    }

    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        let mut p = Vec::new();
        for tree in &self.trees {
            let q = tree.predict(row);
            p.resize(q.len(), 0.0);
            for (a, b) in p.iter_mut().zip(q) {
                *a += b;
            }
        }
        normalise(p)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Model {
    Logistic(Logistic),
    Svm(Svm),
    Knn(Knn),
    Tree(Tree),
    Forest(Forest),
}

impl Model {
    /// Fits `kind` to rows `x` with class indices `y` below `classes`.
    pub fn fit(kind: Kind, x: &[Vec<f64>], y: &[usize], classes: usize, params: &Params, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        match kind {
            Kind::Logistic => Self::Logistic(Logistic::fit(x, y, classes, params)),
            Kind::Svm => Self::Svm(Svm::fit(x, y, classes, params, &mut rng)),
            Kind::Knn => Self::Knn(Knn::fit(x, y, classes, params)),
            Kind::Tree => Self::Tree(Tree::fit(x, y, (0..x.len()).collect(), classes, params, None, &mut rng)),
            Kind::Forest => Self::Forest(Forest::fit(x, y, classes, params, &mut rng)),
        }
    }

    pub fn kind(&self) -> Kind {
        match self {
            Self::Logistic(_) => Kind::Logistic,
            Self::Svm(_) => Kind::Svm,
            Self::Knn(_) => Kind::Knn,
            Self::Tree(_) => Kind::Tree,
            Self::Forest(_) => Kind::Forest,
        }
    }

    /// Class probabilities for a standardized row.
    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        match self {
            Self::Logistic(m) => m.predict(row),
            Self::Svm(m) => m.predict(row),
            Self::Knn(m) => m.predict(row),
            Self::Tree(m) => m.predict(row),
            Self::Forest(m) => m.predict(row),
        }
    }
}

/// `w · [row, 1]`.
fn dot(w: &[f64], row: &[f64]) -> f64 {
    row.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + w.last().copied().unwrap_or(0.0)
}

fn scores(weights: &[Vec<f64>], row: &[f64]) -> Vec<f64> {
    weights.iter().map(|w| dot(w, row)).collect()
}

fn scaled(v: &[f64], scale: f64) -> Vec<f64> {
    v.iter().map(|x| x * scale).collect()
}

pub(crate) fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    normalise(z.iter().map(|v| (v - max).exp()).collect())
}

fn normalise(mut p: Vec<f64>) -> Vec<f64> {
    let sum: f64 = p.iter().sum();
    if sum > 0.0 {
        for v in &mut p {
            *v /= sum;
        }
    }
    p
}

fn count(y: &[usize], rows: &[usize], classes: usize) -> Vec<usize> {
    let mut counts = vec![0; classes];
    for &i in rows {
        counts[y[i]] += 1;
    }
    counts
}

fn gini(counts: &[usize], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    1.0 - counts.iter().map(|&c| (c as f64 / n as f64).powi(2)).sum::<f64>()
}

/// Feature and threshold with the lowest weighted Gini, if any improves on the parent.
fn best_split(x: &[Vec<f64>], y: &[usize], rows: &[usize], classes: usize, features: &[usize], min_leaf: usize) -> Option<(usize, f64)> {
    let n = rows.len();
    let total = count(y, rows, classes);
    let parent = gini(&total, n);
    let mut best: Option<(f64, usize, f64)> = None;
    let mut sorted = rows.to_vec();
    for &f in features {
        sorted.sort_by(|&a, &b| x[a][f].total_cmp(&x[b][f]).then(a.cmp(&b)));
        let mut left = vec![0; classes];
        for split in 1..n {
            left[y[sorted[split - 1]]] += 1;
            let (a, b) = (x[sorted[split - 1]][f], x[sorted[split]][f]);
            if a == b || split < min_leaf || n - split < min_leaf {
                continue;
            }
            let right: Vec<usize> = total.iter().zip(&left).map(|(t, l)| t - l).collect();
            let impurity = (split as f64 * gini(&left, split) + (n - split) as f64 * gini(&right, n - split)) / n as f64;
            if impurity < parent - 1e-12 && best.is_none_or(|b| impurity < b.0) {
                best = Some((impurity, f, a + (b - a) / 2.0));
            }
        }
    }
    best.map(|b| (b.1, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three Gaussian blobs in 2-D, 30 rows each.
    fn blobs() -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = Rng::new(7);
        let centres = [(-2.0, 0.0), (2.0, 0.0), (0.0, 3.0)];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (c, (cx, cy)) in centres.iter().enumerate() {
            for _ in 0..30 {
                x.push(vec![cx + rng.normal() * 0.5, cy + rng.normal() * 0.5]);
                y.push(c);
            }
        }
        (x, y)
    }

    #[test]
    fn every_kind_separates_blobs_with_probabilities() {
        let (x, y) = blobs();
        for kind in Kind::ALL {
            let model = Model::fit(kind, &x, &y, 3, &Params::default(), 1);
            let correct = x.iter().zip(&y).filter(|(row, c)| {
                let p = model.predict(row);
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{kind:?} {p:?}");
                crate::argmax(&p) == **c
            });
            assert!(correct.count() >= 85, "{kind:?}");
            let centre = model.predict(&[0.0, 3.0]);
            assert_eq!(crate::argmax(&centre), 2, "{kind:?} {centre:?}");
        }
    }

    #[test]
    fn fitting_is_deterministic_for_a_seed() {
        let (x, y) = blobs();
        for kind in [Kind::Svm, Kind::Forest] {
            assert_eq!(Model::fit(kind, &x, &y, 3, &Params::default(), 3), Model::fit(kind, &x, &y, 3, &Params::default(), 3));
        }
        let a = Model::fit(Kind::Forest, &x, &y, 3, &Params::default(), 3);
        let b = Model::fit(Kind::Forest, &x, &y, 3, &Params::default(), 4);
        assert_ne!(a, b);
    }
}
