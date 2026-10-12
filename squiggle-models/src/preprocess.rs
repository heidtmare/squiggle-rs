//! Feature preprocessing fitted on the training rows: missing (non-finite)
//! values take the column median, then each column is standardized.

use serde::{Deserialize, Serialize};

pub const DESCRIPTION: &str = "median fill of missing values, then standardize";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Standardize {
    pub fill: Vec<f64>,
    pub mean: Vec<f64>,
    /// Population standard deviation; 1 for constant columns.
    pub scale: Vec<f64>,
}

impl Standardize {
    pub fn fit(x: &[Vec<f64>]) -> Self {
        let d = x.first().map_or(0, Vec::len);
        let mut out = Self { fill: Vec::with_capacity(d), mean: Vec::with_capacity(d), scale: Vec::with_capacity(d) };
        for j in 0..d {
            let mut known: Vec<f64> = x.iter().map(|r| r[j]).filter(|v| v.is_finite()).collect();
            known.sort_by(f64::total_cmp);
            let fill = match known.len() {
                0 => 0.0,
                n if n % 2 == 1 => known[n / 2],
                n => (known[n / 2 - 1] + known[n / 2]) / 2.0,
            };
            let filled: Vec<f64> = x.iter().map(|r| if r[j].is_finite() { r[j] } else { fill }).collect();
            let n = filled.len().max(1) as f64;
            let mean = filled.iter().sum::<f64>() / n;
            let sd = (filled.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
            out.fill.push(fill);
            out.mean.push(mean);
            out.scale.push(if sd > 1e-12 { sd } else { 1.0 });
        }
        out
    }

    pub fn apply(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, &v)| (if v.is_finite() { v } else { self.fill[j] } - self.mean[j]) / self.scale[j])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_values_take_the_median_then_standardize() {
        let x = vec![vec![1.0, f64::NAN, 5.0], vec![3.0, f64::NAN, 5.0], vec![f64::NAN, f64::NAN, 5.0], vec![2.0, f64::NAN, 5.0]];
        let s = Standardize::fit(&x);
        assert_eq!(s.fill, [2.0, 0.0, 5.0]);
        assert_eq!(s.scale[1..], [1.0, 1.0]);
        let row = s.apply(&[f64::NAN, f64::INFINITY, 7.0]);
        assert_eq!(row, [0.0, 0.0, 2.0]);
        let z: Vec<f64> = x.iter().map(|r| s.apply(r)[0]).collect();
        assert!(z.iter().sum::<f64>().abs() < 1e-12);
    }
}
