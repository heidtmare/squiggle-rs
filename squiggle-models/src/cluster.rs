//! Unsupervised embedding of events: standardized event features, PCA, and
//! DBSCAN clusters in the PCA space. New events take the cluster of their
//! nearest core point within `eps`, or none.

use serde::{Deserialize, Serialize};

use crate::Standardize;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    pub preprocess: Standardize,
    /// Principal axes (rows), unit length, largest variance first.
    pub components: Vec<Vec<f64>>,
    /// Fraction of the total variance along each component.
    pub explained: Vec<f64>,
    pub eps: f64,
    pub min_points: usize,
    /// Core points in PCA space with their cluster.
    pub core: Vec<(Vec<f64>, usize)>,
    pub clusters: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterParams {
    /// Components kept; 0 keeps enough for 90 % of the variance (at least 2).
    pub components: usize,
    /// Neighbourhood radius in PCA space; 0 picks three times the median distance to the `min_points − 1`-th neighbour.
    pub eps: f64,
    /// Points within `eps`, itself included, that make a core point.
    pub min_points: usize,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self { components: 0, eps: 0.0, min_points: 3 }
    }
}

impl Embedding {
    /// Fits to `x` (rows of raw features); returns the model and each row's cluster.
    pub fn fit(x: &[Vec<f64>], params: &ClusterParams) -> (Self, Vec<Option<usize>>) {
        let preprocess = Standardize::fit(x);
        let z: Vec<Vec<f64>> = x.iter().map(|r| preprocess.apply(r)).collect();
        let d = preprocess.mean.len();
        let n = z.len().max(1) as f64;
        let mut covariance = vec![vec![0.0; d]; d];
        for row in &z {
            for i in 0..d {
                for j in 0..d {
                    covariance[i][j] += row[i] * row[j] / n;
                }
            }
        }
        let (values, vectors) = jacobi(covariance);
        let total: f64 = values.iter().map(|v| v.max(0.0)).sum::<f64>().max(1e-12);
        let mut order: Vec<usize> = (0..d).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        let keep = if params.components > 0 {
            params.components.min(d)
        } else {
            let mut sum = 0.0;
            let k = order.iter().take_while(|&&i| {
                let before = sum;
                sum += values[i].max(0.0) / total;
                before < 0.9
            });
            k.count().clamp(2.min(d), d)
        };
        let components: Vec<Vec<f64>> = order[..keep]
            .iter()
            .map(|&i| {
                let mut v: Vec<f64> = (0..d).map(|r| vectors[r][i]).collect();
                // Sign: the largest entry is positive.
                let big = v.iter().copied().fold(0.0f64, |m, x| if x.abs() > m.abs() { x } else { m });
                if big < 0.0 {
                    v.iter_mut().for_each(|x| *x = -*x);
                }
                v
            })
            .collect();
        let explained = order[..keep].iter().map(|&i| values[i].max(0.0) / total).collect();
        let mut model = Self { preprocess, components, explained, eps: params.eps, min_points: params.min_points.max(1), core: Vec::new(), clusters: 0 };
        let points: Vec<Vec<f64>> = z.iter().map(|r| model.project(r)).collect();
        if model.eps.is_nan() || model.eps <= 0.0 {
            model.eps = auto_eps(&points, model.min_points);
        }
        let (labels, core) = dbscan(&points, model.eps, model.min_points);
        model.clusters = labels.iter().flatten().max().map_or(0, |m| m + 1);
        model.core = core.into_iter().map(|i| (points[i].clone(), labels[i].expect("core points are clustered"))).collect();
        (model, labels)
    }

    fn project(&self, z: &[f64]) -> Vec<f64> {
        self.components.iter().map(|c| c.iter().zip(z).map(|(a, b)| a * b).sum()).collect()
    }

    /// PCA coordinates of a raw feature row.
    pub fn embed(&self, row: &[f64]) -> Vec<f64> {
        self.project(&self.preprocess.apply(row))
    }

    /// Cluster of the nearest core point within `eps`.
    pub fn assign(&self, point: &[f64]) -> Option<usize> {
        self.core
            .iter()
            .map(|(c, k)| (distance(c, point), *k))
            .filter(|(d, _)| *d <= self.eps)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, k)| k)
    }
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

/// Three times the median distance to the `min_points − 1`-th neighbour.
fn auto_eps(points: &[Vec<f64>], min_points: usize) -> f64 {
    let k = min_points.saturating_sub(1).max(1);
    let mut kth: Vec<f64> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let mut d: Vec<f64> = points.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, q)| distance(p, q)).collect();
            d.sort_by(f64::total_cmp);
            d.get(k - 1).copied()
        })
        .collect();
    if kth.is_empty() {
        return 1.0;
    }
    kth.sort_by(f64::total_cmp);
    (3.0 * kth[kth.len() / 2]).max(1e-9)
}

/// DBSCAN: each point's cluster (`None`: noise) and the core points' indices.
pub fn dbscan(points: &[Vec<f64>], eps: f64, min_points: usize) -> (Vec<Option<usize>>, Vec<usize>) {
    let neighbours: Vec<Vec<usize>> =
        points.iter().map(|p| points.iter().enumerate().filter(|(_, q)| distance(p, q) <= eps).map(|(j, _)| j).collect()).collect();
    let core: Vec<bool> = neighbours.iter().map(|n| n.len() >= min_points).collect();
    let mut labels: Vec<Option<usize>> = vec![None; points.len()];
    let mut next = 0;
    for start in 0..points.len() {
        if !core[start] || labels[start].is_some() {
            continue;
        }
        let mut stack = vec![start];
        labels[start] = Some(next);
        while let Some(i) = stack.pop() {
            if !core[i] {
                continue;
            }
            for &j in &neighbours[i] {
                if labels[j].is_none() {
                    labels[j] = Some(next);
                    stack.push(j);
                }
            }
        }
        next += 1;
    }
    (labels, (0..points.len()).filter(|&i| core[i]).collect())
}

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix, by cyclic Jacobi rotations.
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (rp, rq) = (a[p].clone(), a[q].clone());
                a[p] = rp.iter().zip(&rq).map(|(x, y)| c * x - s * y).collect();
                a[q] = rp.iter().zip(&rq).map(|(x, y)| s * x + c * y).collect();
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn jacobi_diagonalises() {
        let a = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 1.0]];
        let (values, v) = jacobi(a.clone());
        for k in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i][j] * v[j][k]).sum();
                assert!((av - values[k] * v[i][k]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn separated_blobs_are_recovered_and_new_points_assigned() {
        let mut rng = Rng::new(5);
        let centres = [[0.0, 0.0, 5.0], [6.0, 1.0, 5.0], [0.0, 7.0, 5.0]];
        let mut x = Vec::new();
        let mut truth = Vec::new();
        for (k, c) in centres.iter().enumerate() {
            for _ in 0..8 {
                x.push(vec![c[0] + 0.4 * rng.normal(), c[1] + 0.4 * rng.normal(), c[2]]);
                truth.push(k);
            }
        }
        // An outlier and a missing value.
        x.push(vec![30.0, -20.0, 5.0]);
        x[3][2] = f64::NAN;
        let (model, labels) = Embedding::fit(&x, &ClusterParams::default());
        assert_eq!(model.clusters, 3, "{labels:?}");
        assert_eq!(labels[24], None, "the outlier is noise");
        for k in 0..3 {
            let first = labels[k * 8];
            assert!(first.is_some() && labels[k * 8..k * 8 + 8].iter().all(|l| *l == first), "{labels:?}");
        }
        assert!(model.explained[0] >= model.explained[1]);
        let new = model.embed(&[6.2, 0.8, 5.0]);
        assert_eq!(model.assign(&new), labels[8]);
        assert_eq!(model.assign(&model.embed(&[-40.0, 40.0, 5.0])), None);
        let text = serde_json::to_string(&model).unwrap();
        assert_eq!(serde_json::from_str::<Embedding>(&text).unwrap(), model);
    }
}
