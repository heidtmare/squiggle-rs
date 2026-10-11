//! Sample storage: append-only chunks, real or I/Q, and raw binary layouts.

use std::sync::Arc;

use crate::phenomenology::SampleKind;

/// One complex sample: `[I, Q]`.
pub type Iq = [f32; 2];

/// Append-only samples. Chunks never move or change, so readers can hold them
/// while a stream appends.
#[derive(Clone, Debug)]
pub struct ChunkedSamples<T = f32> {
    chunks: Vec<Arc<[T]>>,
    /// Index of each chunk's first sample.
    offsets: Vec<usize>,
    len: usize,
}

impl<T> Default for ChunkedSamples<T> {
    fn default() -> Self {
        Self { chunks: Vec::new(), offsets: Vec::new(), len: 0 }
    }
}

impl<T: Copy> ChunkedSamples<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_vec(samples: Vec<T>) -> Self {
        let mut chunked = Self::new();
        chunked.append(samples);
        chunked
    }

    pub fn append(&mut self, chunk: impl Into<Arc<[T]>>) {
        let chunk = chunk.into();
        if !chunk.is_empty() {
            self.offsets.push(self.len);
            self.len += chunk.len();
            self.chunks.push(chunk);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn chunks(&self) -> &[Arc<[T]>] {
        &self.chunks
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.chunks.iter().flat_map(|chunk| chunk.iter().copied())
    }

    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let chunk = self.offsets.partition_point(|&start| start <= index) - 1;
        Some(self.chunks[chunk][index - self.offsets[chunk]])
    }

    /// Samples `start..end`, clamped to the stored range.
    pub fn range(&self, start: usize, end: usize) -> impl Iterator<Item = T> + '_ {
        let end = end.min(self.len);
        let start = start.min(end);
        let first = self.offsets.partition_point(|&s| s <= start).saturating_sub(1);
        self.chunks[first.min(self.chunks.len())..]
            .iter()
            .zip(&self.offsets[first.min(self.offsets.len())..])
            .flat_map(move |(chunk, &offset)| {
                let lo = start.saturating_sub(offset).min(chunk.len());
                let hi = end.saturating_sub(offset).min(chunk.len());
                chunk[lo..hi].iter().copied()
            })
            .take(end - start)
    }

    /// Maps every sample, keeping chunk boundaries.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> ChunkedSamples<U> {
        let mut out = ChunkedSamples::new();
        for chunk in &self.chunks {
            out.append(chunk.iter().map(|&s| f(s)).collect::<Vec<_>>());
        }
        out
    }
}

/// A decoded block of samples, as readers produce them.
#[derive(Clone, Debug, PartialEq)]
pub enum SampleBuf {
    Real(Vec<f32>),
    Complex(Vec<Iq>),
}

impl SampleBuf {
    pub fn len(&self) -> usize {
        match self {
            Self::Real(s) => s.len(),
            Self::Complex(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn kind(&self) -> SampleKind {
        match self {
            Self::Real(_) => SampleKind::Real,
            Self::Complex(_) => SampleKind::Complex,
        }
    }

    /// `count` missing samples (NaN) of `kind`.
    pub fn gap(kind: SampleKind, count: usize) -> Self {
        match kind {
            SampleKind::Real => Self::Real(vec![f32::NAN; count]),
            SampleKind::Complex => Self::Complex(vec![[f32::NAN; 2]; count]),
        }
    }
}

/// A trace's samples in memory.
#[derive(Clone, Debug)]
pub enum TraceSamples {
    Real(ChunkedSamples<f32>),
    Complex(ChunkedSamples<Iq>),
}

impl TraceSamples {
    pub fn new(kind: SampleKind) -> Self {
        match kind {
            SampleKind::Real => Self::Real(ChunkedSamples::new()),
            SampleKind::Complex => Self::Complex(ChunkedSamples::new()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Real(s) => s.len(),
            Self::Complex(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn kind(&self) -> SampleKind {
        match self {
            Self::Real(_) => SampleKind::Real,
            Self::Complex(_) => SampleKind::Complex,
        }
    }

    /// Appends a block; `false` if its kind does not match.
    pub fn append(&mut self, buf: SampleBuf) -> bool {
        match (self, buf) {
            (Self::Real(s), SampleBuf::Real(b)) => s.append(b),
            (Self::Complex(s), SampleBuf::Complex(b)) => s.append(b),
            _ => return false,
        }
        true
    }

    /// The real series plotted for `component`. Real samples ignore it.
    pub fn series(&self, component: IqComponent) -> ChunkedSamples<f32> {
        match self {
            Self::Real(s) => s.clone(),
            Self::Complex(s) => s.map(|iq| component.of(iq)),
        }
    }
}

impl From<SampleBuf> for TraceSamples {
    fn from(buf: SampleBuf) -> Self {
        match buf {
            SampleBuf::Real(s) => Self::Real(ChunkedSamples::from_vec(s)),
            SampleBuf::Complex(s) => Self::Complex(ChunkedSamples::from_vec(s)),
        }
    }
}

/// What a complex trace plots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IqComponent {
    #[default]
    Magnitude,
    /// Radians.
    Phase,
    I,
    Q,
}

impl IqComponent {
    pub const ALL: [Self; 4] = [Self::Magnitude, Self::Phase, Self::I, Self::Q];

    pub fn label(self) -> &'static str {
        match self {
            Self::Magnitude => "Magnitude",
            Self::Phase => "Phase",
            Self::I => "I",
            Self::Q => "Q",
        }
    }

    pub fn of(self, [i, q]: Iq) -> f32 {
        match self {
            Self::Magnitude => i.hypot(q),
            Self::Phase => q.atan2(i),
            Self::I => i,
            Self::Q => q,
        }
    }
}

/// Numeric encoding of one sample component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumberFormat {
    I8,
    U8,
    #[default]
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl NumberFormat {
    pub const ALL: [Self; 8] = [Self::I8, Self::U8, Self::I16, Self::U16, Self::I32, Self::U32, Self::F32, Self::F64];

    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

/// Layout of a headerless binary file, supplied by the analyst and stored
/// with each trace so the file can be decoded again.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RawLayout {
    pub format: NumberFormat,
    pub big_endian: bool,
    /// Interleaved I/Q pairs.
    pub complex: bool,
    /// Interleaved channels per frame.
    pub channels: u32,
    /// Bytes skipped before the first frame.
    pub header_bytes: u64,
    /// Multiplier from stored value to physical units.
    pub scale: f64,
}

impl Default for RawLayout {
    fn default() -> Self {
        Self { format: NumberFormat::I16, big_endian: false, complex: false, channels: 1, header_bytes: 0, scale: 1.0 }
    }
}

impl RawLayout {
    /// Bytes per frame (all channels, both components when complex).
    pub fn frame_size(&self) -> usize {
        self.format.size() * self.channels.max(1) as usize * if self.complex { 2 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(samples: &[f32], sizes: &[usize]) -> ChunkedSamples {
        let mut chunked = ChunkedSamples::new();
        let mut at = 0;
        for &size in sizes {
            chunked.append(samples[at..at + size].to_vec());
            at += size;
        }
        chunked
    }

    #[test]
    fn chunks_concatenate_in_order() {
        let mut samples = ChunkedSamples::from_vec(vec![1.0, 2.0]);
        samples.append(vec![]);
        samples.append(vec![3.0]);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples.chunks().len(), 2);
        assert_eq!(samples.iter().collect::<Vec<_>>(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn random_access_and_ranges_cross_chunks() {
        let values: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let chunked = split(&values, &[3, 1, 4, 2]);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(chunked.get(i), Some(*v));
        }
        assert_eq!(chunked.get(10), None);
        for start in 0..=10 {
            for end in start..=12 {
                let got: Vec<f32> = chunked.range(start, end).collect();
                assert_eq!(got, values[start..end.min(10)], "{start}..{end}");
            }
        }
        assert_eq!(ChunkedSamples::<f32>::new().range(0, 5).count(), 0);
    }

    #[test]
    fn complex_series_follow_the_component() {
        let samples = TraceSamples::from(SampleBuf::Complex(vec![[3.0, 4.0], [0.0, -1.0]]));
        assert_eq!(samples.kind(), SampleKind::Complex);
        assert_eq!(samples.series(IqComponent::Magnitude).iter().collect::<Vec<_>>(), [5.0, 1.0]);
        assert_eq!(samples.series(IqComponent::Q).iter().collect::<Vec<_>>(), [4.0, -1.0]);
        let phase = samples.series(IqComponent::Phase).get(1).unwrap();
        assert!((phase + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn mismatched_kinds_are_not_appended() {
        let mut samples = TraceSamples::new(SampleKind::Real);
        assert!(samples.append(SampleBuf::Real(vec![1.0])));
        assert!(!samples.append(SampleBuf::Complex(vec![[1.0, 0.0]])));
        assert_eq!(samples.len(), 1);
    }
}
