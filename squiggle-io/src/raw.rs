//! Headerless binary decoded with an analyst-supplied [`RawLayout`].
//! Values are multiplied by `scale`; integers are not normalised.

use std::path::Path;

use squiggle_core::RawLayout;

use crate::decode::{deinterleave, values};
use crate::{ReadError, Recording};

pub const FORMAT: &str = "raw";

pub fn read(path: &Path, bytes: &[u8], layout: &RawLayout) -> Result<Vec<Recording>, ReadError> {
    let error = |message: String| ReadError::Format(format!("{}: {message}", path.display()));
    let header = usize::try_from(layout.header_bytes).unwrap_or(usize::MAX);
    let body = bytes.get(header..).ok_or_else(|| error(format!("header of {header} bytes exceeds the file")))?;
    if body.len() < layout.frame_size() {
        return Err(error("no complete frame after the header".into()));
    }
    let scale = layout.scale;
    let buffers = deinterleave(
        &values(body, layout.format, layout.big_endian),
        layout.channels as usize,
        layout.complex,
        |v| (v * scale) as f32,
    );
    let stem = path.file_stem().map_or_else(|| "raw".into(), |s| s.to_string_lossy().into_owned());
    let channels = buffers.len();
    Ok(buffers
        .into_iter()
        .enumerate()
        .map(|(channel, samples)| Recording {
            code: if channels == 1 { stem.clone() } else { format!("{stem}.ch{channel}") },
            channel: channel as u32,
            start: None,
            sample_rate: None,
            samples,
            suggestion: None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::{NumberFormat, SampleBuf};

    #[test]
    fn header_channels_and_scale_apply() {
        let mut bytes = vec![0xAA; 3];
        bytes.extend([1i16, -2, 3, -4, 5].iter().flat_map(|v| v.to_be_bytes()));
        let layout = RawLayout { format: NumberFormat::I16, big_endian: true, channels: 2, header_bytes: 3, scale: 0.5, ..RawLayout::default() };
        let r = read(Path::new("/x/dump.bin"), &bytes, &layout).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].code, "dump.ch0");
        assert_eq!(r[0].samples, SampleBuf::Real(vec![0.5, 1.5]));
        assert_eq!(r[1].samples, SampleBuf::Real(vec![-1.0, -2.0]));
    }

    #[test]
    fn too_short_files_are_errors() {
        let layout = RawLayout { format: NumberFormat::F64, ..RawLayout::default() };
        assert!(read(Path::new("x"), &[0; 7], &layout).is_err());
        assert!(read(Path::new("x"), &[0; 7], &RawLayout { header_bytes: 9, ..layout }).is_err());
    }
}
