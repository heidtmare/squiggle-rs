//! WAV via `hound`. Integer samples are scaled to `[-1, 1]`. WAV carries no
//! start time or sensor identity: the analyst supplies both.

use std::io::Cursor;
use std::path::Path;

use squiggle_core::SampleBuf;

use crate::{ReadError, Recording, WaveformReader};

pub struct WavReader;

impl WaveformReader for WavReader {
    fn name(&self) -> &'static str {
        "wav"
    }

    fn probe(&self, _path: &Path, head: &[u8]) -> bool {
        head.len() >= 12 && &head[0..4] == b"RIFF" && &head[8..12] == b"WAVE"
    }

    fn read(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Recording>, ReadError> {
        let format = |e: hound::Error| ReadError::Format(format!("{}: {e}", path.display()));
        let mut reader = hound::WavReader::new(Cursor::new(bytes)).map_err(format)?;
        let spec = reader.spec();
        let channels = usize::from(spec.channels.max(1));

        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>().map_err(format)?,
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|s| s as f32 * scale))
                    .collect::<Result<_, _>>()
                    .map_err(format)?
            }
        };

        let stem = path.file_stem().map_or_else(|| "wav".into(), |s| s.to_string_lossy().into_owned());
        Ok((0..channels)
            .map(|channel| Recording {
                code: if channels == 1 { stem.clone() } else { format!("{stem}.ch{channel}") },
                channel: channel as u32,
                start: None,
                sample_rate: Some(f64::from(spec.sample_rate)),
                samples: SampleBuf::Real(interleaved.iter().skip(channel).step_by(channels).copied().collect()),
                suggestion: None,
                location: None,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(channels: u16, frames: &[[i16; 2]]) -> Vec<u8> {
        let spec = hound::WavSpec { channels, sample_rate: 8000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for frame in frames {
            for sample in &frame[..usize::from(channels)] {
                writer.write_sample(*sample).unwrap();
            }
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn stereo_is_split_into_scaled_channels() {
        let bytes = write_wav(2, &[[16384, -32768], [0, 8192]]);
        assert!(WavReader.probe(Path::new("x"), &bytes));
        let recordings = WavReader.read(Path::new("/data/mic.wav"), &bytes).unwrap();
        assert_eq!(recordings.len(), 2);
        assert_eq!(recordings[0].code, "mic.ch0");
        assert_eq!(recordings[0].samples, SampleBuf::Real(vec![0.5, 0.0]));
        assert_eq!(recordings[1].samples, SampleBuf::Real(vec![-1.0, 0.25]));
        assert_eq!(recordings[1].sample_rate, Some(8000.0));
        assert_eq!(recordings[1].start, None);
    }

    #[test]
    fn mono_uses_the_file_stem() {
        let bytes = write_wav(1, &[[0, 0]]);
        let recordings = WavReader.read(Path::new("boom.wav"), &bytes).unwrap();
        assert_eq!(recordings[0].code, "boom");
    }
}
