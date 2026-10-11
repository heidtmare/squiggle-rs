//! Waveform readers and project persistence.
//!
//! Readers decode whole files (captures are seconds long). Each file is hashed
//! so a project can detect when a referenced file has changed.

mod csv;
mod store;
mod wav;

use core::fmt;
use std::path::{Path, PathBuf};

use squiggle_core::{Suggestion, Timestamp, TraceSource};

pub use crate::csv::CsvReader;
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
    pub samples: Vec<f32>,
    pub suggestion: Option<Suggestion>,
}

/// A decoded file.
#[derive(Clone, Debug, PartialEq)]
pub struct FileRead {
    pub path: PathBuf,
    pub format: &'static str,
    pub content_hash: String,
    pub recordings: Vec<Recording>,
}

impl FileRead {
    pub fn source(&self, channel: u32) -> TraceSource {
        TraceSource::File {
            path: self.path.to_string_lossy().into_owned(),
            format: self.format.to_string(),
            channel,
            content_hash: self.content_hash.clone(),
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
}

pub fn readers() -> [&'static dyn WaveformReader; 2] {
    [&WavReader, &CsvReader]
}

fn reader(name: &str) -> Option<&'static dyn WaveformReader> {
    readers().into_iter().find(|r| r.name() == name)
}

pub fn content_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Decodes a file with the first reader that recognises it.
pub fn read_file(path: &Path) -> Result<FileRead, ReadError> {
    let bytes = std::fs::read(path)?;
    let head = &bytes[..bytes.len().min(512)];
    let reader = readers()
        .into_iter()
        .find(|r| r.probe(path, head))
        .ok_or_else(|| ReadError::UnknownFormat(path.to_path_buf()))?;
    Ok(FileRead {
        path: path.to_path_buf(),
        format: reader.name(),
        content_hash: content_hash(&bytes),
        recordings: reader.read(path, &bytes)?,
    })
}

/// Reloads a trace's samples, refusing a file that changed since import.
pub fn load_samples(source: &TraceSource) -> Result<Vec<f32>, ReadError> {
    let TraceSource::File { path, format, channel, content_hash: expected } = source;
    let path = Path::new(path);
    let bytes = std::fs::read(path)?;
    if content_hash(&bytes) != *expected {
        return Err(ReadError::Changed(path.to_path_buf()));
    }
    let reader = reader(format).ok_or_else(|| ReadError::UnknownFormat(path.to_path_buf()))?;
    reader
        .read(path, &bytes)?
        .into_iter()
        .find(|r| r.channel == *channel)
        .map(|r| r.samples)
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
        assert_eq!(load_samples(&source).unwrap(), [1.0, 2.0]);

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
}
