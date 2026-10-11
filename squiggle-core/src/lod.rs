//! Min/max pyramid for drawing long traces at any zoom.
//!
//! Level `k` holds the min/max of each aligned block of `BASE · FANOUT^k`
//! samples. Appends are O(1) amortised, so a stream can extend it in place.
//! NaN marks a missing sample and is ignored; an all-NaN block is empty.

use crate::samples::ChunkedSamples;

const BASE: usize = 16;
const FANOUT: usize = 4;

/// Extent of a run of samples. Empty when every sample was NaN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMax {
    pub min: f32,
    pub max: f32,
}

impl MinMax {
    pub const EMPTY: Self = Self { min: f32::INFINITY, max: f32::NEG_INFINITY };

    pub fn is_empty(self) -> bool {
        self.min > self.max
    }

    fn add(&mut self, value: f32) {
        // NaN fails both comparisons.
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn merge(&mut self, other: Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

#[derive(Clone, Debug)]
struct Level {
    blocks: Vec<MinMax>,
    /// The block being filled.
    partial: MinMax,
    /// Children folded into `partial`.
    count: usize,
}

impl Level {
    fn new() -> Self {
        Self { blocks: Vec::new(), partial: MinMax::EMPTY, count: 0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Envelope {
    levels: Vec<Level>,
    len: usize,
}

/// One decimated vertex: sample index (fractional at bucket centres) and value.
/// A NaN value breaks the line.
pub type LodPoint = (f64, f32);

impl Envelope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(samples: &ChunkedSamples) -> Self {
        let mut envelope = Self::new();
        for chunk in samples.chunks() {
            envelope.append(chunk);
        }
        envelope
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn append(&mut self, samples: &[f32]) {
        if self.levels.is_empty() {
            self.levels.push(Level::new());
        }
        for &value in samples {
            let base = &mut self.levels[0];
            base.partial.add(value);
            base.count += 1;
            if base.count == BASE {
                self.carry(0);
            }
        }
        self.len += samples.len();
    }

    /// Closes level `k`'s partial block and folds it into level `k + 1`.
    fn carry(&mut self, mut k: usize) {
        loop {
            let level = &mut self.levels[k];
            let block = level.partial;
            level.blocks.push(block);
            level.partial = MinMax::EMPTY;
            level.count = 0;

            if self.levels.len() == k + 1 {
                self.levels.push(Level::new());
            }
            let up = &mut self.levels[k + 1];
            up.partial.merge(block);
            up.count += 1;
            if up.count < FANOUT {
                return;
            }
            k += 1;
        }
    }

    fn block_size(k: usize) -> usize {
        BASE * FANOUT.pow(k as u32)
    }

    /// Extent of samples `start..end`, in O(levels²) block reads plus at most
    /// `2·BASE` raw samples.
    pub fn range(&self, samples: &ChunkedSamples, start: usize, end: usize) -> MinMax {
        let end = end.min(self.len).min(samples.len());
        let mut extent = MinMax::EMPTY;
        let mut i = start;
        while i < end {
            let block = (0..self.levels.len()).rev().find_map(|k| {
                let size = Self::block_size(k);
                let index = i / size;
                (i.is_multiple_of(size) && i + size <= end).then(|| self.levels[k].blocks.get(index).map(|b| (*b, size))).flatten()
            });
            match block {
                Some((block, size)) => {
                    extent.merge(block);
                    i += size;
                }
                None => {
                    // Up to the next base boundary, or the end.
                    let next = ((i / BASE + 1) * BASE).min(end);
                    for value in samples.range(i, next) {
                        extent.add(value);
                    }
                    i = next;
                }
            }
        }
        extent
    }

    /// Vertices for samples `start..end` with `bucket` samples per bucket.
    /// `bucket <= 1` returns raw samples; larger buckets are aligned to
    /// multiples of `bucket` so panning does not shimmer. Each bucket gives its
    /// min and max, alternating order (even buckets min first) so the line
    /// steps up, across, down, across rather than doubling back.
    pub fn decimate(&self, samples: &ChunkedSamples, start: usize, end: usize, bucket: usize) -> Vec<LodPoint> {
        let end = end.min(self.len).min(samples.len());
        if start >= end {
            return Vec::new();
        }
        if bucket <= 1 {
            return samples.range(start, end).enumerate().map(|(i, v)| ((start + i) as f64, v)).collect();
        }
        let first = start / bucket * bucket;
        let mut points = Vec::with_capacity(2 * (end - first).div_ceil(bucket));
        for lo in (first..end).step_by(bucket) {
            let hi = (lo + bucket).min(self.len);
            let centre = (lo + hi) as f64 / 2.0;
            let extent = self.range(samples, lo, hi);
            if extent.is_empty() {
                points.push((centre, f32::NAN));
            } else if (lo / bucket).is_multiple_of(2) {
                points.push((centre, extent.min));
                points.push((centre, extent.max));
            } else {
                points.push((centre, extent.max));
                points.push((centre, extent.min));
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random samples with a few NaN gaps.
    fn signal(len: usize) -> Vec<f32> {
        let mut state = 0x2545_f491_u32;
        (0..len)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if i % 997 < 7 { f32::NAN } else { (state % 2001) as f32 - 1000.0 }
            })
            .collect()
    }

    fn brute(values: &[f32]) -> MinMax {
        let mut extent = MinMax::EMPTY;
        values.iter().for_each(|&v| extent.add(v));
        extent
    }

    #[test]
    fn ranges_match_a_brute_force_scan() {
        let values = signal(20_000);
        let samples = ChunkedSamples::from_vec(values.clone());
        let envelope = Envelope::build(&samples);
        for (start, end) in [(0, 20_000), (1, 19_999), (15, 17), (64, 1088), (333, 15_000), (19_990, 25_000), (5, 5)] {
            assert_eq!(envelope.range(&samples, start, end), brute(&values[start.min(20_000)..end.min(20_000)]), "{start}..{end}");
        }
    }

    #[test]
    fn many_chunks_decimate_like_one() {
        let values = signal(50_000);
        let whole = ChunkedSamples::from_vec(values.clone());
        let mut pieces = ChunkedSamples::new();
        let mut streamed = Envelope::new();
        for chunk in values.chunks(777) {
            pieces.append(chunk.to_vec());
            streamed.append(chunk);
        }
        let built = Envelope::build(&whole);
        for bucket in [1, 3, 16, 100, 1024] {
            let a = built.decimate(&whole, 1234, 48_000, bucket);
            let b = streamed.decimate(&pieces, 1234, 48_000, bucket);
            // NaN != NaN, so compare bit patterns.
            let bits = |p: &[LodPoint]| p.iter().map(|(x, y)| (x.to_bits(), y.to_bits())).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "bucket {bucket}");
        }
    }

    #[test]
    fn buckets_carry_min_and_max_and_mark_gaps() {
        let mut values = vec![0.0; 64];
        values[5] = 9.0;
        values[6] = -3.0;
        values[32..48].fill(f32::NAN);
        let samples = ChunkedSamples::from_vec(values);
        let envelope = Envelope::build(&samples);
        let points = envelope.decimate(&samples, 0, 64, 16);
        assert_eq!(points[0], (8.0, -3.0));
        assert_eq!(points[1], (8.0, 9.0));
        assert!(points[4].1.is_nan());
        assert_eq!(points.len(), 7);
        // Odd buckets give max first.
        assert_eq!((points[5].1, points[6].1), (0.0, 0.0));
        let points = envelope.decimate(&samples, 16, 64, 16);
        assert_eq!(points[0], (24.0, 0.0));
        let mut ramp = ChunkedSamples::new();
        ramp.append((0..64).map(|i| i as f32).collect::<Vec<_>>());
        let ramp_points = Envelope::build(&ramp).decimate(&ramp, 0, 64, 16);
        assert_eq!(ramp_points[2..4], [(24.0, 31.0), (24.0, 16.0)]);
    }

    #[test]
    fn raw_buckets_return_samples() {
        let samples = ChunkedSamples::from_vec(vec![1.0, 2.0, 3.0]);
        let envelope = Envelope::build(&samples);
        assert_eq!(envelope.decimate(&samples, 1, 9, 1), [(1.0, 2.0), (2.0, 3.0)]);
    }
}
