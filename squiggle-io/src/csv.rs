//! CSV: a header row, one column per channel, optional time column.
//!
//! - Channel columns: header is the sensor code; empty cells read as NaN.
//! - Time column (`time`, `t`, `timestamp`, `seconds`): ISO 8601 values give
//!   the start and rate; numeric values give the rate only. Spacing must be
//!   uniform to within 1 %.
//! - Lines starting with `#` are comments.

use std::path::Path;

use squiggle_core::phenomenology::suggest_from_seed_code;
use squiggle_core::{SampleBuf, Timestamp};

use crate::{ReadError, Recording, WaveformReader};

const TIME_HEADERS: [&str; 4] = ["time", "t", "timestamp", "seconds"];
const SPACING_TOLERANCE: f64 = 0.01;

pub struct CsvReader;

impl WaveformReader for CsvReader {
    fn name(&self) -> &'static str {
        "csv"
    }

    fn probe(&self, path: &Path, _head: &[u8]) -> bool {
        path.extension().is_some_and(|e| e.eq_ignore_ascii_case("csv"))
    }

    fn read(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Recording>, ReadError> {
        let error = |line: u64, message: String| ReadError::Format(format!("{}:{line}: {message}", path.display()));
        let mut reader = ::csv::ReaderBuilder::new().comment(Some(b'#')).trim(::csv::Trim::All).from_reader(bytes);

        let headers: Vec<String> = reader.headers().map_err(|e| error(1, e.to_string()))?.iter().map(String::from).collect();
        let time_column = headers.iter().position(|h| TIME_HEADERS.contains(&h.to_ascii_lowercase().as_str()));
        let channels: Vec<usize> = (0..headers.len()).filter(|i| Some(*i) != time_column).collect();
        if channels.is_empty() {
            return Err(error(1, "no channel columns".to_string()));
        }

        let mut times = Vec::new();
        let mut columns = vec![Vec::new(); channels.len()];
        for record in reader.records() {
            let record = record.map_err(|e| error(e.position().map_or(0, |p| p.line()), e.to_string()))?;
            let line = record.position().map_or(0, |p| p.line());
            if let Some(column) = time_column {
                times.push(record.get(column).unwrap_or_default().to_string());
            }
            for (samples, &column) in columns.iter_mut().zip(&channels) {
                let cell = record.get(column).unwrap_or_default();
                let value = if cell.is_empty() {
                    f32::NAN
                } else {
                    cell.parse().map_err(|_| error(line, format!("not a number: {cell}")))?
                };
                samples.push(value);
            }
        }

        let (start, sample_rate) = parse_times(&times).map_err(|m| error(0, m))?;
        Ok(channels
            .iter()
            .zip(columns)
            .enumerate()
            .map(|(channel, (&column, samples))| Recording {
                code: headers[column].clone(),
                channel: channel as u32,
                start,
                sample_rate,
                samples: SampleBuf::Real(samples),
                suggestion: suggest_from_seed_code(&headers[column]),
                location: None,
            })
            .collect())
    }
}

/// Start (ISO only) and sample rate from a time column.
fn parse_times(times: &[String]) -> Result<(Option<Timestamp>, Option<f64>), String> {
    if times.len() < 2 {
        return Ok((None, None));
    }
    let (start, offsets): (Option<Timestamp>, Vec<f64>) = if times.iter().all(|t| t.parse::<f64>().is_ok()) {
        (None, times.iter().map(|t| t.parse().expect("checked")).collect())
    } else {
        let stamps: Vec<Timestamp> = times
            .iter()
            .map(|t| t.parse::<Timestamp>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        (Some(stamps[0]), stamps.iter().map(|t| t.seconds_since(stamps[0])).collect())
    };

    let interval = (offsets[offsets.len() - 1] - offsets[0]) / (offsets.len() - 1) as f64;
    if !(interval.is_finite() && interval > 0.0) {
        return Err("time column must increase".to_string());
    }
    if let Some(bad) = offsets.windows(2).position(|w| ((w[1] - w[0]) / interval - 1.0).abs() > SPACING_TOLERANCE) {
        return Err(format!("uneven sample spacing at row {}", bad + 2));
    }
    Ok((start, Some(1.0 / interval)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Vec<Recording>, ReadError> {
        CsvReader.read(Path::new("x.csv"), text.as_bytes())
    }

    #[test]
    fn iso_time_column_gives_start_and_rate() {
        let recordings = read(
            "# comment\ntime,IU.ANMO.00.BHZ,mic\n\
             2026-10-11T00:00:00Z,1,4\n2026-10-11T00:00:00.01Z,2,\n2026-10-11T00:00:00.02Z,3,6\n",
        )
        .unwrap();
        assert_eq!(recordings.len(), 2);
        assert_eq!(recordings[0].start, Some("2026-10-11T00:00:00Z".parse().unwrap()));
        assert!((recordings[0].sample_rate.unwrap() - 100.0).abs() < 1e-6);
        assert_eq!(recordings[0].samples, SampleBuf::Real(vec![1.0, 2.0, 3.0]));
        assert_eq!(recordings[0].suggestion.as_ref().map(|s| s.phenomenology.as_str()), Some("seismic"));
        assert!(matches!(&recordings[1].samples, SampleBuf::Real(s) if s[1].is_nan()));
        assert_eq!(recordings[1].channel, 1);
        assert_eq!(recordings[1].suggestion, None);
    }

    #[test]
    fn numeric_time_column_gives_rate_only() {
        let recordings = read("t,a\n0,1\n0.5,2\n1.0,3\n").unwrap();
        assert_eq!(recordings[0].start, None);
        assert_eq!(recordings[0].sample_rate, Some(2.0));
    }

    #[test]
    fn without_a_time_column_both_are_unknown() {
        let recordings = read("a,b\n1,2\n").unwrap();
        assert_eq!((recordings[0].start, recordings[0].sample_rate), (None, None));
    }

    #[test]
    fn uneven_spacing_and_bad_numbers_are_errors() {
        assert!(read("t,a\n0,1\n1,2\n3,3\n").is_err());
        assert!(read("a\n1\nabc\n").is_err());
        assert!(read("time\n0\n1\n").is_err());
    }
}
