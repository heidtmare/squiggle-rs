//! SAC binary, either byte order: evenly spaced time series only (`iftype`
//! ITIME, `leven` true). Start is the reference time `nz*` plus `b`.

use std::path::Path;

use squiggle_core::phenomenology::suggest_from_seed_code;
use squiggle_core::{Location, SampleBuf, Timestamp};

use crate::{ReadError, Recording, WaveformReader};

const HEADER_LEN: usize = 632;
const UNDEFINED: i32 = -12_345;
const ITIME: i32 = 1;

// Word indices (floats 0..70, ints 70..110) and string byte offsets.
const DELTA: usize = 0;
const B: usize = 5;
const STLA: usize = 31;
const STLO: usize = 32;
const STEL: usize = 33;
const NZYEAR: usize = 70;
const NVHDR: usize = 76;
const NPTS: usize = 79;
const IFTYPE: usize = 85;
const LEVEN: usize = 105;
const KSTNM: usize = 440;
const KHOLE: usize = 464;
const KCMPNM: usize = 600;
const KNETWK: usize = 608;

pub struct SacReader;

impl WaveformReader for SacReader {
    fn name(&self) -> &'static str {
        "sac"
    }

    fn probe(&self, path: &Path, head: &[u8]) -> bool {
        path.extension().is_some_and(|e| e.eq_ignore_ascii_case("sac")) || byte_order(head).is_some()
    }

    fn read(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Recording>, ReadError> {
        let error = |message: &str| ReadError::Format(format!("{}: {message}", path.display()));
        let be = byte_order(bytes).ok_or_else(|| error("not a SAC file (nvhdr is not 6 or 7)"))?;
        let word = |i: usize| {
            let b = bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes");
            if be { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) }
        };
        let float = |i: usize| f32::from_bits(word(i));
        let int = |i: usize| word(i) as i32;
        let text = |at: usize| {
            let s = String::from_utf8_lossy(&bytes[at..at + 8]).trim_matches(|c: char| c == ' ' || c == '\0').to_string();
            if s == "-12345" { String::new() } else { s }
        };

        if int(IFTYPE) != ITIME || int(LEVEN) != 1 {
            return Err(error("only evenly spaced time series are supported"));
        }
        let npts = usize::try_from(int(NPTS)).map_err(|_| error("negative npts"))?;
        let delta = f64::from(float(DELTA));
        if !(delta > 0.0 && delta.is_finite()) {
            return Err(error("delta must be positive"));
        }
        let data = bytes.get(HEADER_LEN..HEADER_LEN + npts * 4).ok_or_else(|| error("file shorter than npts"))?;
        let samples = data
            .chunks_exact(4)
            .map(|b| {
                let b = b.try_into().expect("4 bytes");
                f32::from_bits(if be { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
            })
            .collect();

        let nz: Vec<i32> = (NZYEAR..NZYEAR + 6).map(int).collect();
        let start = (!nz.contains(&UNDEFINED) && nz[1] > 0).then(|| {
            let b = f64::from(float(B));
            let b = if b == f64::from(UNDEFINED) { 0.0 } else { b };
            let [year, day, hour, minute, second, msec] = [nz[0], nz[1], nz[2], nz[3], nz[4], nz[5]];
            Timestamp::from_ordinal(year.into(), day as u32, hour as u32, minute as u32, second as u32, f64::from(msec) * 1e-3)
                .add_seconds(b)
        });

        let defined = |i: usize| Some(f64::from(float(i))).filter(|v| *v != f64::from(UNDEFINED));
        let position = match (defined(STLA), defined(STLO)) {
            (Some(latitude), Some(longitude)) => {
                Some(Location { latitude, longitude, elevation_m: defined(STEL).unwrap_or(0.0) }).filter(Location::is_valid)
            }
            _ => None,
        };

        let (network, station, location, component) = (text(KNETWK), text(KSTNM), text(KHOLE), text(KCMPNM));
        let code = if station.is_empty() {
            path.file_stem().map_or_else(|| "sac".into(), |s| s.to_string_lossy().into_owned())
        } else {
            format!("{network}.{station}.{location}.{component}")
        };
        Ok(vec![Recording {
            suggestion: suggest_from_seed_code(&component),
            code,
            channel: 0,
            start,
            sample_rate: Some(1.0 / delta),
            samples: SampleBuf::Real(samples),
            location: position,
        }])
    }
}

/// A little-endian SAC file of an evenly spaced series. Codes longer than 8
/// characters are truncated, as the format requires.
pub fn encode(
    samples: &[f32],
    start: Timestamp,
    rate: f64,
    [network, station, location, component]: [&str; 4],
    position: Option<Location>,
) -> Vec<u8> {
    let mut words = [0u32; 110];
    for (i, w) in words.iter_mut().enumerate() {
        *w = if i < 70 { (UNDEFINED as f32).to_bits() } else { UNDEFINED as u32 };
    }
    let (year, day, hour, minute, second, frac) = start.to_ordinal();
    words[DELTA] = ((1.0 / rate) as f32).to_bits();
    words[B] = (frac as f32).to_bits();
    for (i, v) in [year as i32, day as i32, hour as i32, minute as i32, second as i32, 0].into_iter().enumerate() {
        words[NZYEAR + i] = v as u32;
    }
    if let Some(p) = position {
        words[STLA] = (p.latitude as f32).to_bits();
        words[STLO] = (p.longitude as f32).to_bits();
        words[STEL] = (p.elevation_m as f32).to_bits();
    }
    words[NVHDR] = 6;
    words[NPTS] = samples.len() as u32;
    words[IFTYPE] = ITIME as u32;
    words[LEVEN] = 1;
    let mut out: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut strings = vec![b' '; HEADER_LEN - 440];
    for at in (0..strings.len()).step_by(8) {
        strings[at..at + 6].copy_from_slice(b"-12345");
    }
    for (at, text) in [(KSTNM, station), (KHOLE, location), (KCMPNM, component), (KNETWK, network)] {
        let field: String = format!("{text:<8}").chars().take(8).collect();
        strings[at - 440..at - 440 + 8].copy_from_slice(field.as_bytes());
    }
    out.extend(strings);
    out.extend(samples.iter().flat_map(|s| s.to_le_bytes()));
    out
}

/// `Some(big_endian)` when the header is plausible in one byte order:
/// `nvhdr` 6 or 7, a known `iftype`, boolean `leven`, positive `delta`.
fn byte_order(head: &[u8]) -> Option<bool> {
    if head.len() < HEADER_LEN {
        return None;
    }
    [false, true].into_iter().find(|&be| {
        let word = |i: usize| {
            let b = head[i * 4..i * 4 + 4].try_into().expect("4 bytes");
            if be { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) }
        };
        let delta = f32::from_bits(word(DELTA));
        matches!(word(NVHDR), 6 | 7) && (1..=4).contains(&word(IFTYPE)) && word(LEVEN) <= 1 && delta > 0.0 && delta.is_finite()
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn write_sac(samples: &[f32], big_endian: bool, station: &str, component: &str) -> Vec<u8> {
        let mut words = [0u32; 158];
        for (i, w) in words.iter_mut().enumerate().take(110) {
            *w = if i < 70 { (UNDEFINED as f32).to_bits() } else { UNDEFINED as u32 };
        }
        words[DELTA] = 0.01f32.to_bits();
        words[B] = 1.5f32.to_bits();
        for (i, v) in [2026, 284, 1, 2, 3, 250].into_iter().enumerate() {
            words[NZYEAR + i] = v as u32;
        }
        words[NVHDR] = 6;
        words[NPTS] = samples.len() as u32;
        words[IFTYPE] = ITIME as u32;
        words[LEVEN] = 1;
        let mut out: Vec<u8> = words[..110].iter().flat_map(|w| if big_endian { w.to_be_bytes() } else { w.to_le_bytes() }).collect();
        let mut strings = vec![b' '; 192];
        for at in (0..192).step_by(8) {
            strings[at..at + 6].copy_from_slice(b"-12345");
        }
        let mut put = |at: usize, text: &str| {
            strings[at - 440..at - 440 + 8].copy_from_slice(format!("{text:<8}").as_bytes());
        };
        put(KSTNM, station);
        put(KCMPNM, component);
        put(KNETWK, "IU");
        out.extend(strings);
        out.extend(samples.iter().flat_map(|s| if big_endian { s.to_be_bytes() } else { s.to_le_bytes() }));
        out
    }

    #[test]
    fn both_byte_orders_read() {
        for be in [false, true] {
            let bytes = write_sac(&[1.0, -2.5, 3.0], be, "ANMO", "BHZ");
            assert!(SacReader.probe(Path::new("x"), &bytes));
            let r = &SacReader.read(Path::new("x.sac"), &bytes).unwrap()[0];
            assert_eq!(r.code, "IU.ANMO..BHZ");
            assert_eq!(r.samples, SampleBuf::Real(vec![1.0, -2.5, 3.0]));
            assert!((r.sample_rate.unwrap() - 100.0).abs() < 1e-3);
            assert_eq!(r.start.unwrap().to_string(), "2026-10-11T01:02:04.750000000Z");
            assert_eq!(r.suggestion.as_ref().unwrap().phenomenology.as_str(), "seismic");
        }
    }

    #[test]
    fn encoded_files_read_back() {
        let start: Timestamp = "2026-01-01T00:00:09.63Z".parse().unwrap();
        let at = Location { latitude: 35.1, longitude: -106.6, elevation_m: 1600.0 };
        let bytes = encode(&[1.0, 2.0], start, 40.0, ["SY", "STA02", "", "BHZ"], Some(at));
        let r = &SacReader.read(Path::new("x.sac"), &bytes).unwrap()[0];
        assert_eq!(r.code, "SY.STA02..BHZ");
        let read = r.location.unwrap();
        assert!((read.latitude - 35.1).abs() < 1e-5 && (read.longitude + 106.6).abs() < 1e-5 && read.elevation_m == 1600.0);
        let bytes = encode(&[1.0], start, 40.0, ["SY", "STA02", "", "BHZ"], None);
        assert_eq!(SacReader.read(Path::new("x.sac"), &bytes).unwrap()[0].location, None);
        assert!((r.start.unwrap().seconds_since(start)).abs() < 1e-7);
        assert!((r.sample_rate.unwrap() - 40.0).abs() < 1e-4);
    }

    #[test]
    fn spectral_files_are_refused() {
        let mut bytes = write_sac(&[1.0], false, "A", "BHZ");
        bytes[IFTYPE * 4..IFTYPE * 4 + 4].copy_from_slice(&2u32.to_le_bytes());
        assert!(SacReader.read(Path::new("x.sac"), &bytes).is_err());
    }
}
