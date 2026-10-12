//! Classifiers and the model package format shared by `squiggle-train`
//! (fitting) and the app (inference only). Both compute features with
//! `squiggle-core`, preprocess and predict with this crate, so a package
//! gives the same probabilities in either.

pub mod classical;
pub mod deep;
pub mod metrics;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod package;
pub mod preprocess;
pub mod rng;

use serde::{Deserialize, Serialize};

pub use classical::{Kind, Model, Params};
pub use package::{Package, Spec};
pub use preprocess::Standardize;

/// Preprocessing plus a fitted model: raw feature rows in, class probabilities out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    pub preprocess: Standardize,
    pub model: Model,
}

impl Trained {
    pub fn fit(kind: Kind, params: &Params, x: &[Vec<f64>], y: &[usize], classes: usize, seed: u64) -> Self {
        let preprocess = Standardize::fit(x);
        let z: Vec<Vec<f64>> = x.iter().map(|r| preprocess.apply(r)).collect();
        Self { model: Model::fit(kind, &z, y, classes, params, seed), preprocess }
    }

    pub fn predict(&self, row: &[f64]) -> Vec<f64> {
        self.model.predict(&self.preprocess.apply(row))
    }
}

/// Index of the largest value; the first on ties.
pub fn argmax(p: &[f64]) -> usize {
    p.iter().enumerate().fold(0, |best, (i, v)| if *v > p[best] { i } else { best })
}
