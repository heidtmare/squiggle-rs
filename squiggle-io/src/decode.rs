//! Numeric decoding shared by SigMF and raw binary.

use squiggle_core::{NumberFormat, SampleBuf};

/// Decodes packed values. Trailing bytes short of one value are ignored.
pub fn values(bytes: &[u8], format: NumberFormat, big_endian: bool) -> Vec<f64> {
    macro_rules! read {
        ($t:ty) => {
            bytes
                .chunks_exact(size_of::<$t>())
                .map(|b| {
                    let b = b.try_into().expect("exact chunk");
                    (if big_endian { <$t>::from_be_bytes(b) } else { <$t>::from_le_bytes(b) }) as f64
                })
                .collect()
        };
    }
    match format {
        NumberFormat::I8 => bytes.iter().map(|&b| f64::from(b as i8)).collect(),
        NumberFormat::U8 => bytes.iter().map(|&b| f64::from(b)).collect(),
        NumberFormat::I16 => read!(i16),
        NumberFormat::U16 => read!(u16),
        NumberFormat::I32 => read!(i32),
        NumberFormat::U32 => read!(u32),
        NumberFormat::F32 => read!(f32),
        NumberFormat::F64 => read!(f64),
    }
}

/// `(offset, scale)` mapping integer formats onto `[-1, 1]`; floats unchanged.
pub fn full_scale(format: NumberFormat) -> (f64, f64) {
    match format {
        NumberFormat::I8 => (0.0, 1.0 / 128.0),
        NumberFormat::U8 => (-128.0, 1.0 / 128.0),
        NumberFormat::I16 => (0.0, 1.0 / 32_768.0),
        NumberFormat::U16 => (-32_768.0, 1.0 / 32_768.0),
        NumberFormat::I32 => (0.0, 1.0 / 2_147_483_648.0),
        NumberFormat::U32 => (-2_147_483_648.0, 1.0 / 2_147_483_648.0),
        NumberFormat::F32 | NumberFormat::F64 => (0.0, 1.0),
    }
}

/// Splits interleaved frames into one buffer per channel. A trailing partial frame is dropped.
pub fn deinterleave(values: &[f64], channels: usize, complex: bool, map: impl Fn(f64) -> f32) -> Vec<SampleBuf> {
    let channels = channels.max(1);
    let width = if complex { 2 } else { 1 };
    let frames = values.len() / (channels * width);
    (0..channels)
        .map(|c| {
            let at = |frame: usize, k: usize| map(values[(frame * channels + c) * width + k]);
            if complex {
                SampleBuf::Complex((0..frames).map(|f| [at(f, 0), at(f, 1)]).collect())
            } else {
                SampleBuf::Real((0..frames).map(|f| at(f, 0)).collect())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endianness_and_signedness_decode() {
        assert_eq!(values(&[0x01, 0x00, 0xff, 0xff], NumberFormat::I16, false), [1.0, -1.0]);
        assert_eq!(values(&[0x00, 0x01, 0xff, 0xff], NumberFormat::U16, true), [1.0, 65_535.0]);
        assert_eq!(values(&1.5f32.to_be_bytes(), NumberFormat::F32, true), [1.5]);
        assert_eq!(values(&[0x80, 0x7f], NumberFormat::I8, false), [-128.0, 127.0]);
        assert_eq!(values(&[1, 2, 3], NumberFormat::I16, false).len(), 1);
    }

    #[test]
    fn frames_split_by_channel_and_component() {
        let v: Vec<f64> = (0..9).map(f64::from).collect();
        let bufs = deinterleave(&v, 2, true, |x| x as f32);
        assert_eq!(bufs[0], SampleBuf::Complex(vec![[0.0, 1.0], [4.0, 5.0]]));
        assert_eq!(bufs[1], SampleBuf::Complex(vec![[2.0, 3.0], [6.0, 7.0]]));
    }
}
