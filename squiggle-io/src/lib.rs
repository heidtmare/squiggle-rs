//! Waveform readers, sources and project persistence.
//!
//! Readers decode whole files (captures are seconds long) and are replayed
//! through [`FileSource`]. Each file and its companions are hashed so a
//! project can detect when a referenced file has changed.

mod csv;
pub mod dataset;
mod decode;
mod mseed;
pub mod raw;
pub mod sac;
mod sigmf;
mod source;
mod store;
pub mod stream;
mod wav;

use core::fmt;
use std::path::{Path, PathBuf};

use squiggle_core::{Location, RawLayout, SampleBuf, Suggestion, Timestamp, TraceSamples, TraceSource};

pub use crate::csv::CsvReader;
pub use crate::mseed::MseedReader;
pub use crate::sac::SacReader;
pub use crate::sigmf::SigmfReader;
pub use crate::source::{FileSource, Header, SourceEvent, WaveformSource, ingest};
pub use crate::store::{ProjectStore, SqliteStore, StoreError};
pub use crate::wav::WavReader;

/// Application identifier.
pub const APP_ID: &str = "heidtmare.squiggle";
/// Project file extension.
pub const PROJECT_EXTENSION: &str = "squiggle";

/// One channel decoded from a file. `None` fields must be supplied by the analyst.
#[derive(Clone, Debug, PartialEq)]
pub struct Recording {
    /// Proposed sensor code.
    pub code: String,
    /// Channel index within the file.
    pub channel: u32,
    pub start: Option<Timestamp>,
    pub sample_rate: Option<f64>,
    pub samples: SampleBuf,
    pub suggestion: Option<Suggestion>,
    /// Sensor position from the file's metadata.
    pub location: Option<Location>,
}

/// A decoded file.
#[derive(Clone, Debug, PartialEq)]
pub struct FileRead {
    pub path: PathBuf,
    pub format: &'static str,
    pub content_hash: String,
    pub recordings: Vec<Recording>,
    /// Set for raw binary.
    pub layout: Option<RawLayout>,
}

impl FileRead {
    pub fn source(&self, channel: u32) -> TraceSource {
        TraceSource::File {
            path: self.path.to_string_lossy().into_owned(),
            format: self.format.to_string(),
            channel,
            content_hash: self.content_hash.clone(),
            layout: self.layout,
        }
    }
}

#[derive(Debug)]
pub enum ReadError {
    Io(std::io::Error),
    UnknownFormat(PathBuf),
    Format(String),
    /// The file no longer matches the hash recorded when it was imported.
    Changed(PathBuf),
    MissingChannel(u32),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::UnknownFormat(path) => write!(f, "no reader recognises {}", path.display()),
            Self::Format(message) => f.write_str(message),
            Self::Changed(path) => write!(f, "{} changed since import", path.display()),
            Self::MissingChannel(channel) => write!(f, "channel {channel} not found"),
        }
    }
}

impl std::error::Error for ReadError {}

impl From<std::io::Error> for ReadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub trait WaveformReader: Send + Sync {
    /// Stable name, stored in [`TraceSource::File::format`].
    fn name(&self) -> &'static str;
    /// Whether this reader handles the file. `head` is its first bytes.
    fn probe(&self, path: &Path, head: &[u8]) -> bool;
    fn read(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Recording>, ReadError>;
    /// Other files the read depends on; they are hashed with the primary.
    fn companions(&self, _path: &Path) -> Vec<PathBuf> {
        Vec::new()
    }
}

/// Probe order: content signatures first, extensions last. Unrecognised files
/// fall through to [`read_raw`] with an analyst-supplied layout.
pub fn readers() -> [&'static dyn WaveformReader; 5] {
    [&MseedReader, &SacReader, &SigmfReader, &WavReader, &CsvReader]
}

fn reader(name: &str) -> Option<&'static dyn WaveformReader> {
    readers().into_iter().find(|r| r.name() == name)
}

/// File extensions offered by the import dialog.
pub const EXTENSIONS: [&str; 10] = ["mseed", "miniseed", "msd", "sac", "sigmf-meta", "sigmf-data", "wav", "csv", "bin", "raw"];

/// BLAKE3 over `bytes` then each companion file, hex.
fn content_hash(bytes: &[u8], companions: &[PathBuf]) -> Result<String, ReadError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(bytes);
    for companion in companions {
        hasher.update(&std::fs::read(companion)?);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// The file a reader opens: a `.sigmf-data` path resolves to its `.sigmf-meta`.
pub fn primary_path(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|e| e == sigmf::DATA_EXTENSION) {
        let meta = path.with_extension(sigmf::META_EXTENSION);
        if meta.exists() {
            return meta;
        }
    }
    path.to_path_buf()
}

/// Decodes a file with the first reader that recognises it.
/// [`ReadError::UnknownFormat`] means the analyst must describe it for [`read_raw`].
pub fn read_file(path: &Path) -> Result<FileRead, ReadError> {
    let path = primary_path(path);
    let bytes = std::fs::read(&path)?;
    let head = &bytes[..bytes.len().min(1024)];
    let reader = readers()
        .into_iter()
        .find(|r| r.probe(&path, head))
        .ok_or_else(|| ReadError::UnknownFormat(path.clone()))?;
    Ok(FileRead {
        content_hash: content_hash(&bytes, &reader.companions(&path))?,
        format: reader.name(),
        recordings: reader.read(&path, &bytes)?,
        path,
        layout: None,
    })
}

pub fn read_raw(path: &Path, layout: RawLayout) -> Result<FileRead, ReadError> {
    let bytes = std::fs::read(path)?;
    Ok(FileRead {
        path: path.to_path_buf(),
        format: raw::FORMAT,
        content_hash: content_hash(&bytes, &[])?,
        recordings: raw::read(path, &bytes, &layout)?,
        layout: Some(layout),
    })
}

/// Reopens a trace's file as a source, refusing one that changed since import.
pub fn open_source(source: &TraceSource) -> Result<FileSource, ReadError> {
    let TraceSource::File { path, format, content_hash: expected, layout, .. } = source else {
        return Err(ReadError::Format("a stream recording has no file source; use load_trace".to_string()));
    };
    let path = Path::new(path);
    let bytes = std::fs::read(path)?;
    let recordings = if format == raw::FORMAT {
        let layout = layout.ok_or_else(|| ReadError::Format(format!("{}: raw layout missing", path.display())))?;
        if content_hash(&bytes, &[])? != *expected {
            return Err(ReadError::Changed(path.to_path_buf()));
        }
        raw::read(path, &bytes, &layout)?
    } else {
        let reader = reader(format).ok_or_else(|| ReadError::UnknownFormat(path.to_path_buf()))?;
        if content_hash(&bytes, &reader.companions(path))? != *expected {
            return Err(ReadError::Changed(path.to_path_buf()));
        }
        reader.read(path, &bytes)?
    };
    Ok(FileSource::new(recordings))
}

/// Reloads a trace's samples: from its file, or its stream recording.
pub fn load_trace(trace: &squiggle_core::Trace) -> Result<TraceSamples, ReadError> {
    match &trace.source {
        TraceSource::Stream { .. } => stream::load_stream(&trace.source, trace.kind),
        file => load_samples(file),
    }
}

/// Reloads one file trace's samples through its source.
pub fn load_samples(source: &TraceSource) -> Result<TraceSamples, ReadError> {
    let TraceSource::File { channel, .. } = source else {
        return Err(ReadError::Format("a stream recording needs its sample kind; use load_trace".to_string()));
    };
    ingest(&mut open_source(source)?)?
        .into_iter()
        .find(|(header, _)| header.channel == *channel)
        .map(|(_, samples)| samples)
        .ok_or(ReadError::MissingChannel(*channel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_changed_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.csv");
        std::fs::write(&path, "x\n1\n2\n").unwrap();
        let read = read_file(&path).unwrap();
        let source = read.source(0);
        let TraceSamples::Real(samples) = load_samples(&source).unwrap() else { panic!() };
        assert_eq!(samples.iter().collect::<Vec<_>>(), [1.0, 2.0]);

        std::fs::write(&path, "x\n1\n3\n").unwrap();
        assert!(matches!(load_samples(&source), Err(ReadError::Changed(_))));
    }

    #[test]
    fn unknown_files_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.bin");
        std::fs::write(&path, [0u8, 1, 2]).unwrap();
        assert!(matches!(read_file(&path), Err(ReadError::UnknownFormat(_))));
    }

    #[test]
    fn phase_one_traces_still_deserialize() {
        let json = r#"{"id":"0192f0c4-0000-7000-8000-000000000001","sensor":"0192f0c4-0000-7000-8000-000000000002",
            "start":{"secs":0,"frac":0.0},"sample_rate":100.0,"len":4,
            "source":{"File":{"path":"a.wav","format":"wav","channel":0,"content_hash":"00"}}}"#;
        let trace: squiggle_core::Trace = serde_json::from_str(json).unwrap();
        assert_eq!(trace.kind, squiggle_core::SampleKind::Real);
        assert!(matches!(trace.source, TraceSource::File { layout: None, .. }));
    }

    #[test]
    fn raw_files_reload_with_their_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dump.bin");
        std::fs::write(&path, [1u8, 0, 2, 0]).unwrap();
        let read = read_raw(&path, RawLayout::default()).unwrap();
        let TraceSamples::Real(samples) = load_samples(&read.source(0)).unwrap() else { panic!() };
        assert_eq!(samples.iter().collect::<Vec<_>>(), [1.0, 2.0]);
    }

    #[test]
    fn sigmf_resolves_and_hashes_its_data_file() {
        let dir = tempfile::tempdir().unwrap();
        let meta = dir.path().join("a.sigmf-meta");
        let data = dir.path().join("a.sigmf-data");
        std::fs::write(&meta, r#"{"global":{"core:datatype":"rf32_le","core:sample_rate":10}}"#).unwrap();
        std::fs::write(&data, 1.0f32.to_le_bytes()).unwrap();
        let read = read_file(&data).unwrap();
        assert_eq!(read.path, meta);
        let source = read.source(0);
        assert!(load_samples(&source).is_ok());
        std::fs::write(&data, 2.0f32.to_le_bytes()).unwrap();
        assert!(matches!(load_samples(&source), Err(ReadError::Changed(_))));
    }

    #[test]
    fn mseed_and_sac_load_through_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let mseed = dir.path().join("a.mseed");
        let values: Vec<i32> = (0..30).collect();
        let data = mseed::tests::steim2_encode(&values, 0);
        std::fs::write(&mseed, mseed::tests::record(["IU", "A", "", "BHZ"], (2026, 1, 0, 0, 0, 0), (40, 1), 11, 30, &data, true)).unwrap();
        let sac = dir.path().join("b.sac");
        std::fs::write(&sac, sac::tests::write_sac(&[1.0, 2.0], true, "B", "BDF")).unwrap();
        for (path, format, len) in [(mseed, "mseed", 30), (sac, "sac", 2)] {
            let read = read_file(&path).unwrap();
            assert_eq!(read.format, format);
            assert_eq!(load_samples(&read.source(0)).unwrap().len(), len);
        }
    }
}
