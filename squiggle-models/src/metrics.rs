//! Stratified cross-validation and its scores.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::{Trained, argmax};
use crate::classical::{Kind, Params};

/// Fold of each row: every class is shuffled and dealt round-robin.
pub fn stratified_folds(y: &[usize], folds: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut fold = vec![0; y.len()];
    let classes = y.iter().max().map_or(0, |m| m + 1);
    let mut next = 0;
    for c in 0..classes {
        let mut rows: Vec<usize> = (0..y.len()).filter(|&i| y[i] == c).collect();
        rng.shuffle(&mut rows);
        for i in rows {
            fold[i] = next % folds.max(1);
            next += 1;
        }
    }
    fold
}

/// Out-of-fold probabilities: each fold is predicted by a model fitted (with
/// its own preprocessing) on the other folds.
pub fn cross_validate(kind: Kind, params: &Params, x: &[Vec<f64>], y: &[usize], classes: usize, folds: usize, seed: u64) -> Vec<Vec<f64>> {
    let fold = stratified_folds(y, folds, seed);
    let mut out = vec![Vec::new(); x.len()];
    for k in 0..folds {
        let train: Vec<usize> = (0..x.len()).filter(|&i| fold[i] != k).collect();
        if train.is_empty() || train.len() == x.len() {
            continue;
        }
        let tx: Vec<Vec<f64>> = train.iter().map(|&i| x[i].clone()).collect();
        let ty: Vec<usize> = train.iter().map(|&i| y[i]).collect();
        let model = Trained::fit(kind, params, &tx, &ty, classes, seed);
        for i in (0..x.len()).filter(|&i| fold[i] == k) {
            out[i] = model.predict(&x[i]);
        }
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub accuracy: f64,
    pub log_loss: f64,
    /// Rows: true class; columns: predicted.
    pub confusion: Vec<Vec<usize>>,
}

impl Scores {
    pub fn new(probabilities: &[Vec<f64>], y: &[usize], classes: usize) -> Self {
        let mut confusion = vec![vec![0; classes]; classes];
        let mut loss = 0.0;
        for (p, &c) in probabilities.iter().zip(y) {
            confusion[c][argmax(p)] += 1;
            loss -= p.get(c).copied().unwrap_or(0.0).max(1e-12).ln();
        }
        let n = y.len().max(1) as f64;
        let correct: usize = (0..classes).map(|c| confusion[c][c]).sum();
        Self { accuracy: correct as f64 / n, log_loss: loss / n, confusion }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_are_balanced_per_class() {
        let y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let fold = stratified_folds(&y, 2, 9);
        for c in 0..2 {
            let in_first = (0..y.len()).filter(|&i| y[i] == c && fold[i] == 0).count();
            let total = y.iter().filter(|&&v| v == c).count();
            assert_eq!(in_first, total / 2);
        }
    }

    #[test]
    fn scores_count_the_confusion() {
        let p = vec![vec![0.9, 0.1], vec![0.4, 0.6], vec![0.2, 0.8]];
        let s = Scores::new(&p, &[0, 0, 1], 2);
        assert_eq!(s.confusion, [[1, 1], [0, 1]]);
        assert!((s.accuracy - 2.0 / 3.0).abs() < 1e-12);
    }
}
