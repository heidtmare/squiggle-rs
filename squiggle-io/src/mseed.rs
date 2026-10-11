//! miniSEED 2: fixed header, blockettes 1000/1001, encodings int16, int32,
//! float32, float64, Steim1 and Steim2.
//!
//! Records are grouped by `NET.STA.LOC.CHA`, sorted by start and merged.
//! Gaps become NaN; overlapping samples are dropped. Steim frames are always
//! big-endian; other encodings follow blockette 1000's word order.

use std::collections::HashMap;
use std::path::Path;

use squiggle_core::phenomenology::suggest_from_seed_code;
use squiggle_core::{SampleBuf, Timestamp};

use crate::{ReadError, Recording, WaveformReader};

const HEADER_LEN: usize = 48;
/// Longest gap filled with NaN; longer ones are refused.
const MAX_GAP_SAMPLES: f64 = 50_000_000.0;

pub struct MseedReader;

impl WaveformReader for MseedReader {
    fn name(&self) -> &'static str {
        "mseed"
    }

    fn probe(&self, path: &Path, head: &[u8]) -> bool {
        let extension = path.extension().is_some_and(|e| {
            ["mseed", "miniseed", "msd"].iter().any(|x| e.eq_ignore_ascii_case(x))
        });
        extension || looks_like_header(head)
    }

    fn read(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Recording>, ReadError> {
        let error = |offset: usize, message: String| ReadError::Format(format!("{} @{offset}: {message}", path.display()));
        if bytes.starts_with(b"MS\x03") {
            return Err(error(0, "miniSEED 3 is not supported".into()));
        }

        let mut order: Vec<String> = Vec::new();
        let mut records: HashMap<String, Vec<Record>> = HashMap::new();
        let mut offset = 0;
        while offset + HEADER_LEN <= bytes.len() {
            let rest = &bytes[offset..];
            if !looks_like_header(rest) {
                if rest.iter().all(|&b| b == 0 || b == b' ') {
                    break;
                }
                return Err(error(offset, "not a miniSEED record".into()));
            }
            let (record, length) = parse_record(rest).map_err(|m| error(offset, m))?;
            offset += length;
            if let Some(record) = record {
                if !records.contains_key(&record.code) {
                    order.push(record.code.clone());
                }
                records.entry(record.code.clone()).or_default().push(record);
            }
        }
        if order.is_empty() {
            return Err(error(0, "no data records".into()));
        }

        order
            .into_iter()
            .enumerate()
            .map(|(channel, code)| {
                let mut list = records.remove(&code).expect("ordered codes have records");
                list.sort_by(|a, b| a.start.partial_cmp(&b.start).expect("finite times"));
                let (start, sample_rate, samples) = merge(&list).map_err(|m| error(0, format!("{code}: {m}")))?;
                Ok(Recording {
                    suggestion: suggest_from_seed_code(&code),
                    code,
                    channel: channel as u32,
                    start: Some(start),
                    sample_rate: Some(sample_rate),
                    samples: SampleBuf::Real(samples),
                })
            })
            .collect()
    }
}

struct Record {
    code: String,
    start: Timestamp,
    rate: f64,
    samples: Vec<f32>,
}

fn looks_like_header(head: &[u8]) -> bool {
    head.len() >= HEADER_LEN
        && head[..6].iter().all(|b| b.is_ascii_digit() || *b == b' ' || *b == 0)
        && b"DRQM".contains(&head[6])
        && (head[7] == b' ' || head[7] == 0)
}

/// Parses one record. Returns `None` for records without samples (logs, opaque).
fn parse_record(rec: &[u8]) -> Result<(Option<Record>, usize), String> {
    let year_be = u16::from_be_bytes([rec[20], rec[21]]);
    let day_be = u16::from_be_bytes([rec[22], rec[23]]);
    let be = (1900..=2100).contains(&year_be) && (1..=366).contains(&day_be);
    let u16_at = |o: usize| {
        let b = [rec[o], rec[o + 1]];
        if be { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) }
    };
    let i16_at = |o: usize| u16_at(o) as i16;
    let i32_at = |o: usize| {
        let b = [rec[o], rec[o + 1], rec[o + 2], rec[o + 3]];
        if be { i32::from_be_bytes(b) } else { i32::from_le_bytes(b) }
    };
    let text = |range: std::ops::Range<usize>| String::from_utf8_lossy(&rec[range]).trim().to_string();

    let code = format!("{}.{}.{}.{}", text(18..20), text(8..13), text(13..15), text(15..18));
    let samples = usize::from(u16_at(30));
    let rate = sample_rate(i16_at(32), i16_at(34));
    let activity = rec[36];
    let blockettes = rec[39];
    let correction = i32_at(40);
    let data_offset = usize::from(u16_at(44));

    let mut encoding = None;
    let mut word_order_be = be;
    let mut length = None;
    let mut micros = 0i8;
    let mut at = usize::from(u16_at(46));
    for _ in 0..blockettes {
        if at == 0 || at + 8 > rec.len() {
            break;
        }
        match u16_at(at) {
            1000 => {
                encoding = Some(rec[at + 4]);
                word_order_be = rec[at + 5] == 1;
                length = Some(1usize << rec[at + 6].min(20));
            }
            1001 => micros = rec[at + 5] as i8,
            _ => {}
        }
        let next = usize::from(u16_at(at + 2));
        if next <= at {
            break;
        }
        at = next;
    }

    let length = length.unwrap_or_else(|| guess_length(rec));
    if length < HEADER_LEN || length > rec.len() {
        return Err(format!("record length {length} exceeds the file"));
    }

    let year = i64::from(u16_at(20));
    let frac = f64::from(u16_at(28)) * 1e-4 + f64::from(micros) * 1e-6;
    let mut start = Timestamp::from_ordinal(year, u32::from(u16_at(22)), rec[24].into(), rec[25].into(), rec[26].into(), frac);
    // Bit 1: correction already applied.
    if activity & 0x02 == 0 {
        start = start.add_seconds(f64::from(correction) * 1e-4);
    }

    if samples == 0 || rate <= 0.0 || data_offset == 0 {
        return Ok((None, length));
    }
    let data = rec.get(data_offset..length).ok_or("data offset past record end")?;
    let encoding = encoding.ok_or("no blockette 1000: encoding unknown")?;
    let values = decode(data, encoding, word_order_be, samples)?;
    Ok((Some(Record { code, start, rate, samples: values }), length))
}

/// SEED rate factor/multiplier rules.
fn sample_rate(factor: i16, multiplier: i16) -> f64 {
    let (f, m) = (f64::from(factor), f64::from(multiplier));
    match (factor.signum(), multiplier.signum()) {
        (0, _) => 0.0,
        (1, 1) | (1, 0) => f * m.max(1.0),
        (1, -1) => -f / m,
        (-1, 1) | (-1, 0) => -m.max(1.0) / f,
        _ => 1.0 / (f * m),
    }
}

/// Without blockette 1000, the next header's offset, else the rest of the file.
fn guess_length(rec: &[u8]) -> usize {
    (8..=16)
        .map(|p| 1usize << p)
        .find(|&len| rec.get(len..).is_some_and(looks_like_header))
        .unwrap_or(rec.len())
}

fn decode(data: &[u8], encoding: u8, be: bool, count: usize) -> Result<Vec<f32>, String> {
    let need = |size: usize| {
        if data.len() < count * size { Err(format!("{count} samples need {} bytes, record has {}", count * size, data.len())) } else { Ok(()) }
    };
    macro_rules! fixed {
        ($t:ty) => {{
            need(size_of::<$t>())?;
            data.chunks_exact(size_of::<$t>())
                .take(count)
                .map(|b| {
                    let b = b.try_into().expect("exact chunk");
                    (if be { <$t>::from_be_bytes(b) } else { <$t>::from_le_bytes(b) }) as f32
                })
                .collect()
        }};
    }
    Ok(match encoding {
        1 => fixed!(i16),
        3 => fixed!(i32),
        4 => fixed!(f32),
        5 => fixed!(f64),
        10 => steim(data, count, 1)?.into_iter().map(|v| v as f32).collect(),
        11 => steim(data, count, 2)?.into_iter().map(|v| v as f32).collect(),
        other => return Err(format!("unsupported encoding {other}")),
    })
}

fn sign_extend(value: u32, bits: u32) -> i32 {
    ((value << (32 - bits)) as i32) >> (32 - bits)
}

/// Unpacks `fields` signed values of `bits` each from the low bits of `word`, most significant first.
fn unpack(word: u32, fields: u32, bits: u32, out: &mut Vec<i32>) {
    for j in 0..fields {
        let shift = bits * (fields - 1 - j);
        let mask = if bits == 32 { u32::MAX } else { (1 << bits) - 1 };
        out.push(sign_extend((word >> shift) & mask, bits));
    }
}

/// Steim1/2 decoding: integrates differences from the forward constant X0 and checks against Xn.
pub(crate) fn steim(data: &[u8], count: usize, version: u8) -> Result<Vec<i32>, String> {
    let mut diffs = Vec::with_capacity(count + 8);
    let (mut first, mut last) = (None, None);
    'frames: for (f, frame) in data.chunks_exact(64).enumerate() {
        let word = |i: usize| u32::from_be_bytes(frame[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        let control = word(0);
        for i in 1..16 {
            let w = word(i);
            if f == 0 && i == 1 {
                first = Some(w as i32);
                continue;
            }
            if f == 0 && i == 2 {
                last = Some(w as i32);
                continue;
            }
            match ((control >> (30 - 2 * i)) & 3, version) {
                (0, _) => {}
                (1, _) => unpack(w, 4, 8, &mut diffs),
                (2, 1) => unpack(w, 2, 16, &mut diffs),
                (3, 1) => unpack(w, 1, 32, &mut diffs),
                (2, _) => match w >> 30 {
                    1 => unpack(w, 1, 30, &mut diffs),
                    2 => unpack(w, 2, 15, &mut diffs),
                    3 => unpack(w, 3, 10, &mut diffs),
                    _ => return Err("invalid Steim2 sub-code".into()),
                },
                (_, _) => match w >> 30 {
                    0 => unpack(w, 5, 6, &mut diffs),
                    1 => unpack(w, 6, 5, &mut diffs),
                    2 => unpack(w, 7, 4, &mut diffs),
                    _ => return Err("invalid Steim2 sub-code".into()),
                },
            }
            if diffs.len() >= count {
                break 'frames;
            }
        }
    }
    let first = first.ok_or("no Steim frames")?;
    if diffs.len() < count {
        return Err(format!("Steim data holds {} of {count} samples", diffs.len()));
    }
    // The first difference refers to the previous record and is skipped.
    let mut values = Vec::with_capacity(count);
    values.push(first);
    for &d in &diffs[1..count] {
        values.push(values[values.len() - 1].wrapping_add(d));
    }
    if Some(values[count - 1]) != last {
        return Err("Steim reverse integration constant does not match".into());
    }
    Ok(values)
}

/// Concatenates sorted records into one series.
fn merge(records: &[Record]) -> Result<(Timestamp, f64, Vec<f32>), String> {
    let rate = records[0].rate;
    let start = records[0].start;
    let mut samples: Vec<f32> = Vec::new();
    for record in records {
        if ((record.rate - rate) / rate).abs() > 1e-6 {
            return Err(format!("sample rate changes from {rate} to {} Hz", record.rate));
        }
        let expected = samples.len() as f64;
        let offset = record.start.seconds_since(start) * rate - expected;
        if offset >= 0.5 {
            if offset > MAX_GAP_SAMPLES {
                return Err(format!("gap of {:.0} s is too long to fill", offset / rate));
            }
            samples.resize(samples.len() + offset.round() as usize, f32::NAN);
            samples.extend_from_slice(&record.samples);
        } else {
            let skip = (-offset).round().max(0.0) as usize;
            samples.extend(record.samples.iter().skip(skip));
        }
    }
    Ok((start, rate, samples))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Steim2 encoder for tests: greedy choice of the densest packing that fits.
    pub fn steim2_encode(values: &[i32], previous: i32) -> Vec<u8> {
        let mut diffs = vec![values[0].wrapping_sub(previous)];
        diffs.extend(values.windows(2).map(|w| w[1].wrapping_sub(w[0])));
        let fits = |d: &[i32], bits: u32| d.iter().all(|&v| v >= -(1 << (bits - 1)) && v < (1 << (bits - 1)));
        // (count, bits, nibble, dnib)
        const PACKINGS: [(usize, u32, u32, Option<u32>); 7] =
            [(7, 4, 3, Some(2)), (6, 5, 3, Some(1)), (5, 6, 3, Some(0)), (4, 8, 1, None), (3, 10, 2, Some(3)), (2, 15, 2, Some(2)), (1, 30, 2, Some(1))];
        let mut words: Vec<(u32, u32)> = Vec::new();
        let mut i = 0;
        while i < diffs.len() {
            let &(count, bits, nibble, dnib) = PACKINGS
                .iter()
                .find(|(count, bits, ..)| i + count <= diffs.len() && fits(&diffs[i..i + count], *bits))
                .expect("difference fits 30 bits");
            let mut word = dnib.map_or(0, |d| d << 30);
            for (j, &d) in diffs[i..i + count].iter().enumerate() {
                word |= ((d as u32) & ((1 << bits) - 1)) << (bits * (count as u32 - 1 - j as u32));
            }
            words.push((nibble, word));
            i += count;
        }
        frames(values, words)
    }

    /// Steim1 encoder for tests: one 32-bit difference per word.
    pub fn steim1_encode(values: &[i32]) -> Vec<u8> {
        let mut diffs = vec![0];
        diffs.extend(values.windows(2).map(|w| w[1].wrapping_sub(w[0])));
        frames(values, diffs.into_iter().map(|d| (3, d as u32)).collect())
    }

    fn frames(values: &[i32], words: Vec<(u32, u32)>) -> Vec<u8> {
        let mut out = Vec::new();
        let mut words = words.into_iter();
        let mut frame_index = 0;
        loop {
            let mut frame = [0u32; 16];
            let mut control = 0u32;
            let mut slot = 1;
            if frame_index == 0 {
                frame[1] = values[0] as u32;
                frame[2] = values[values.len() - 1] as u32;
                slot = 3;
            }
            let mut wrote = false;
            for (nibble, word) in words.by_ref() {
                frame[slot] = word;
                control |= nibble << (30 - 2 * slot);
                wrote = true;
                slot += 1;
                if slot == 16 {
                    break;
                }
            }
            frame[0] = control;
            out.extend(frame.iter().flat_map(|w| w.to_be_bytes()));
            frame_index += 1;
            if !wrote || slot < 16 {
                break;
            }
        }
        out
    }

    /// A 512-byte record with blockette 1000 at offset 48 and data at 64.
    pub fn record(code: [&str; 4], start: (u16, u16, u8, u8, u8, u16), rate: (i16, i16), encoding: u8, samples: u16, data: &[u8], big_endian: bool) -> Vec<u8> {
        let mut r = vec![0u8; 512];
        r[..8].copy_from_slice(b"000001D ");
        let put = |r: &mut Vec<u8>, at: usize, text: &str, width: usize| {
            let padded = format!("{text:<width$}");
            r[at..at + width].copy_from_slice(padded.as_bytes());
        };
        let [net, sta, loc, cha] = code;
        put(&mut r, 8, sta, 5);
        put(&mut r, 13, loc, 2);
        put(&mut r, 15, cha, 3);
        put(&mut r, 18, net, 2);
        let u16b = |v: u16| if big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
        r[20..22].copy_from_slice(&u16b(start.0));
        r[22..24].copy_from_slice(&u16b(start.1));
        r[24] = start.2;
        r[25] = start.3;
        r[26] = start.4;
        r[28..30].copy_from_slice(&u16b(start.5));
        r[30..32].copy_from_slice(&u16b(samples));
        r[32..34].copy_from_slice(&u16b(rate.0 as u16));
        r[34..36].copy_from_slice(&u16b(rate.1 as u16));
        r[36] = 0x02;
        r[39] = 1;
        r[44..46].copy_from_slice(&u16b(64));
        r[46..48].copy_from_slice(&u16b(48));
        r[48..50].copy_from_slice(&u16b(1000));
        r[52] = encoding;
        r[53] = u8::from(big_endian);
        r[54] = 9;
        r[64..64 + data.len()].copy_from_slice(data);
        r
    }

    #[test]
    fn steim2_round_trips_every_packing() {
        let mut values = vec![0i32];
        for d in [1, -2, 3, 7, -8, 15, -16, 31, -32, 127, -128, 511, -512, 16_383, -16_384, 536_870_911, -536_870_912, 0, 0, 0, 0, 0, 0, 0] {
            values.push(values[values.len() - 1].wrapping_add(d));
        }
        values.extend((0..200).map(|i| (i * 37 % 101) - 50));
        let data = steim2_encode(&values, 0);
        assert_eq!(steim(&data, values.len(), 2).unwrap(), values);
    }

    #[test]
    fn steim1_round_trips_and_checks_the_last_value() {
        let values: Vec<i32> = (0..100).map(|i| i * i - 3000).collect();
        let mut data = steim1_encode(&values);
        assert_eq!(steim(&data, values.len(), 1).unwrap(), values);
        data[11] ^= 1;
        assert!(steim(&data, values.len(), 1).is_err());
    }

    #[test]
    fn rate_factor_rules() {
        assert_eq!(sample_rate(100, 1), 100.0);
        assert_eq!(sample_rate(1, -10), 0.1);
        assert_eq!(sample_rate(-10, 1), 0.1);
        assert_eq!(sample_rate(-10, -10), 0.01);
    }

    #[test]
    fn records_merge_with_gaps_and_channels() {
        let a: Vec<i32> = (0..50).collect();
        let b: Vec<i32> = (100..120).collect();
        let code = ["IU", "ANMO", "00", "BHZ"];
        let mut file = record(code, (2026, 284, 0, 0, 0, 0), (20, 1), 11, 50, &steim2_encode(&a, 0), true);
        // Starts at 3.5 s, 1 s after the first record ends.
        file.extend(record(code, (2026, 284, 0, 0, 3, 5000), (20, 1), 11, 20, &steim2_encode(&b, 49), true));
        let ints: Vec<u8> = (0..4i32).flat_map(|v| v.to_le_bytes()).collect();
        file.extend(record(["XX", "MIC", "", "HDF"], (2026, 284, 0, 0, 1, 0), (-10, 1), 3, 4, &ints, false));

        let recordings = MseedReader.read(Path::new("x.mseed"), &file).unwrap();
        assert_eq!(recordings.len(), 2);
        let bhz = &recordings[0];
        assert_eq!(bhz.code, "IU.ANMO.00.BHZ");
        assert_eq!(bhz.start, Some("2026-10-11T00:00:00Z".parse().unwrap()));
        assert_eq!(bhz.sample_rate, Some(20.0));
        let SampleBuf::Real(samples) = &bhz.samples else { panic!() };
        // Second record starts at 3.5 s = sample 70: 20 NaN fill 50..70.
        assert_eq!(samples.len(), 90);
        assert!(samples[50..70].iter().all(|s| s.is_nan()));
        assert_eq!(samples[70], 100.0);
        assert_eq!(bhz.suggestion.as_ref().unwrap().phenomenology.as_str(), "seismic");

        let mic = &recordings[1];
        assert_eq!(mic.code, "XX.MIC..HDF");
        assert_eq!(mic.sample_rate, Some(0.1));
        assert_eq!(mic.samples, SampleBuf::Real(vec![0.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn probe_recognises_headers_without_an_extension() {
        let r = record(["IU", "A", "", "BHZ"], (2026, 1, 0, 0, 0, 0), (1, 1), 1, 0, &[], true);
        assert!(MseedReader.probe(Path::new("data.bin"), &r));
        assert!(!MseedReader.probe(Path::new("x.csv"), b"time,a\n0,1\n"));
    }
}
