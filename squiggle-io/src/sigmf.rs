//! SigMF: a `.sigmf-meta` JSON beside a `.sigmf-data` file.
//!
//! Reads `core:datatype`, `core:sample_rate`, `core:num_channels` and the first
//! capture's `core:datetime`. Integer types are scaled to `[-1, 1]`. Later
//! captures are not treated as discontinuities.

use std::path::{Path, PathBuf};

use serde_json::Value;
use squiggle_core::{Location, NumberFormat, PhenomenologyId, Suggestion, Timestamp};

use crate::decode::{deinterleave, full_scale, values};
use crate::{ReadError, Recording, WaveformReader};

pub const META_EXTENSION: &str = "sigmf-meta";
pub const DATA_EXTENSION: &str = "sigmf-data";

pub struct SigmfReader;

impl WaveformReader for SigmfReader {
    fn name(&self) -> &'static str {
        "sigmf"
    }

    fn probe(&self, path: &Path, _head: &[u8]) -> bool {
        path.extension().is_some_and(|e| e == META_EXTENSION)
    }

    fn companions(&self, path: &Path) -> Vec<PathBuf> {
        vec![path.with_extension(DATA_EXTENSION)]
    }

    fn read(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Recording>, ReadError> {
        let error = |message: String| ReadError::Format(format!("{}: {message}", path.display()));
        let meta: Value = serde_json::from_slice(bytes).map_err(|e| error(e.to_string()))?;
        let global = &meta["global"];
        let datatype = global["core:datatype"].as_str().ok_or_else(|| error("missing core:datatype".into()))?;
        let (complex, format, big_endian) = parse_datatype(datatype).ok_or_else(|| error(format!("unsupported datatype {datatype}")))?;
        let channels = global["core:num_channels"].as_u64().unwrap_or(1).max(1) as usize;
        let sample_rate = global["core:sample_rate"].as_f64().filter(|r| *r > 0.0);
        // GeoJSON point: [longitude, latitude, altitude?].
        let point = &global["core:geolocation"]["coordinates"];
        let location = match (point[0].as_f64(), point[1].as_f64()) {
            (Some(longitude), Some(latitude)) => {
                Some(Location { latitude, longitude, elevation_m: point[2].as_f64().unwrap_or(0.0) }).filter(Location::is_valid)
            }
            _ => None,
        };
        let start = meta["captures"][0]["core:datetime"]
            .as_str()
            .map(|t| t.parse::<Timestamp>().map_err(|e| error(e.to_string())))
            .transpose()?;

        let data = std::fs::read(path.with_extension(DATA_EXTENSION))?;
        let (offset, scale) = full_scale(format);
        let buffers = deinterleave(&values(&data, format, big_endian), channels, complex, |v| ((v + offset) * scale) as f32);

        let stem = path.file_stem().map_or_else(|| "sigmf".into(), |s| s.to_string_lossy().into_owned());
        Ok(buffers
            .into_iter()
            .enumerate()
            .map(|(channel, samples)| Recording {
                code: if channels == 1 { stem.clone() } else { format!("{stem}.ch{channel}") },
                channel: channel as u32,
                start,
                sample_rate,
                samples,
                suggestion: Some(Suggestion { phenomenology: PhenomenologyId::new("rf"), reason: "SigMF recording".into() }),
                location,
            })
            .collect())
    }
}

/// `cf32_le` → (complex, F32, little-endian). Byte order defaults to little.
fn parse_datatype(text: &str) -> Option<(bool, NumberFormat, bool)> {
    let (body, big_endian) = match text.rsplit_once('_') {
        Some((body, "le")) => (body, false),
        Some((body, "be")) => (body, true),
        None => (text, false),
        _ => return None,
    };
    let complex = match body.as_bytes().first()? {
        b'c' => true,
        b'r' => false,
        _ => return None,
    };
    let format = match &body[1..] {
        "i8" => NumberFormat::I8,
        "u8" => NumberFormat::U8,
        "i16" => NumberFormat::I16,
        "u16" => NumberFormat::U16,
        "i32" => NumberFormat::I32,
        "u32" => NumberFormat::U32,
        "f32" => NumberFormat::F32,
        "f64" => NumberFormat::F64,
        _ => return None,
    };
    Some((complex, format, big_endian))
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::SampleBuf;

    #[test]
    fn datatypes_parse() {
        assert_eq!(parse_datatype("cf32_le"), Some((true, NumberFormat::F32, false)));
        assert_eq!(parse_datatype("ri16_be"), Some((false, NumberFormat::I16, true)));
        assert_eq!(parse_datatype("cu8"), Some((true, NumberFormat::U8, false)));
        assert_eq!(parse_datatype("xf32_le"), None);
        assert_eq!(parse_datatype("cf32_me"), None);
    }

    #[test]
    fn complex_int16_reads_scaled() {
        let dir = tempfile::tempdir().unwrap();
        let meta = dir.path().join("rx.sigmf-meta");
        std::fs::write(
            &meta,
            r#"{"global":{"core:datatype":"ci16_le","core:sample_rate":1e6,"core:version":"1.0.0",
                "core:geolocation":{"type":"Point","coordinates":[-106.5,35.25,1650]}},
                "captures":[{"core:sample_start":0,"core:datetime":"2026-10-11T00:00:00.000001Z"}],"annotations":[]}"#,
        )
        .unwrap();
        let data: Vec<u8> = [16_384i16, -32_768, 0, 8192].iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.path().join("rx.sigmf-data"), data).unwrap();

        let r = &SigmfReader.read(&meta, &std::fs::read(&meta).unwrap()).unwrap()[0];
        assert_eq!(r.code, "rx");
        assert_eq!(r.samples, SampleBuf::Complex(vec![[0.5, -1.0], [0.0, 0.25]]));
        assert_eq!(r.sample_rate, Some(1e6));
        assert_eq!(r.start.unwrap().to_string(), "2026-10-11T00:00:00.000001000Z");
        assert_eq!(r.suggestion.as_ref().unwrap().phenomenology.as_str(), "rf");
        assert_eq!(r.location, Some(Location { latitude: 35.25, longitude: -106.5, elevation_m: 1650.0 }));
    }
}
