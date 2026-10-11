//! Sources: where samples come from, as a stream of events.
//!
//! Downstream code consumes [`WaveformSource`]s only, so a live feed can
//! replace [`FileSource`] without touching ingest or display.

use std::collections::VecDeque;

use squiggle_core::{Location, SampleBuf, SampleKind, Suggestion, Timestamp, TraceSamples};

use crate::{ReadError, Recording};

/// Describes a channel before its samples arrive.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub channel: u32,
    pub code: String,
    pub start: Option<Timestamp>,
    pub sample_rate: Option<f64>,
    pub kind: SampleKind,
    pub suggestion: Option<Suggestion>,
    pub location: Option<Location>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SourceEvent {
    Header(Header),
    Chunk { channel: u32, samples: SampleBuf },
    /// `samples` missing samples, stored as NaN.
    Gap { channel: u32, samples: usize },
    /// No more events; repeated on further calls.
    End,
}

pub trait WaveformSource: Send {
    fn next_event(&mut self) -> Result<SourceEvent, ReadError>;
}

/// A decoded file replayed as events: each channel's header, its samples
/// (optionally split into chunks), then `End`.
pub struct FileSource {
    events: VecDeque<SourceEvent>,
}

impl FileSource {
    pub fn new(recordings: Vec<Recording>) -> Self {
        Self::chunked(recordings, usize::MAX)
    }

    /// Splits each channel into chunks of at most `chunk_len` samples.
    pub fn chunked(recordings: Vec<Recording>, chunk_len: usize) -> Self {
        let chunk_len = chunk_len.max(1);
        let mut events = VecDeque::new();
        for recording in recordings {
            let channel = recording.channel;
            events.push_back(SourceEvent::Header(Header {
                channel,
                code: recording.code,
                start: recording.start,
                sample_rate: recording.sample_rate,
                kind: recording.samples.kind(),
                suggestion: recording.suggestion,
                location: recording.location,
            }));
            let pieces: Vec<SampleBuf> = match recording.samples {
                SampleBuf::Real(s) => s.chunks(chunk_len).map(|c| SampleBuf::Real(c.to_vec())).collect(),
                SampleBuf::Complex(s) => s.chunks(chunk_len).map(|c| SampleBuf::Complex(c.to_vec())).collect(),
            };
            events.extend(pieces.into_iter().map(|samples| SourceEvent::Chunk { channel, samples }));
        }
        Self { events }
    }
}

impl WaveformSource for FileSource {
    fn next_event(&mut self) -> Result<SourceEvent, ReadError> {
        Ok(self.events.pop_front().unwrap_or(SourceEvent::End))
    }
}

fn find(channels: &mut [(Header, TraceSamples)], channel: u32) -> Result<&mut TraceSamples, ReadError> {
    channels
        .iter_mut()
        .find(|(h, _)| h.channel == channel)
        .map(|(_, s)| s)
        .ok_or(ReadError::MissingChannel(channel))
}

/// Drains a source into per-channel samples, in header order.
pub fn ingest(source: &mut dyn WaveformSource) -> Result<Vec<(Header, TraceSamples)>, ReadError> {
    let mut channels: Vec<(Header, TraceSamples)> = Vec::new();
    loop {
        let event = source.next_event()?;
        match event {
            SourceEvent::Header(header) => {
                let samples = TraceSamples::new(header.kind);
                channels.push((header, samples));
            }
            SourceEvent::Chunk { channel, samples } => {
                if !find(&mut channels, channel)?.append(samples) {
                    return Err(ReadError::Format(format!("channel {channel}: sample kind changed mid-stream")));
                }
            }
            SourceEvent::Gap { channel, samples } => {
                let target = find(&mut channels, channel)?;
                target.append(SampleBuf::gap(target.kind(), samples));
            }
            SourceEvent::End => return Ok(channels),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording(channel: u32, samples: SampleBuf) -> Recording {
        Recording {
            code: format!("c{channel}"),
            channel,
            start: None,
            sample_rate: Some(10.0),
            samples,
            suggestion: None,
            location: None,
        }
    }

    #[test]
    fn chunking_does_not_change_what_is_ingested() {
        let recordings = vec![
            recording(0, SampleBuf::Real((0..1000).map(|i| i as f32).collect())),
            recording(1, SampleBuf::Complex(vec![[1.0, 2.0]; 37])),
        ];
        let whole = ingest(&mut FileSource::new(recordings.clone())).unwrap();
        let split = ingest(&mut FileSource::chunked(recordings, 7)).unwrap();
        assert_eq!(whole.len(), 2);
        let TraceSamples::Real(a) = &whole[0].1 else { panic!() };
        let TraceSamples::Real(b) = &split[0].1 else { panic!() };
        assert_eq!(a.chunks().len(), 1);
        assert_eq!(b.chunks().len(), 143);
        assert_eq!(a.iter().collect::<Vec<_>>(), b.iter().collect::<Vec<_>>());
        assert_eq!(split[1].1.len(), 37);
        assert_eq!(split[1].0.kind, SampleKind::Complex);
    }

    struct Scripted(VecDeque<SourceEvent>);

    impl WaveformSource for Scripted {
        fn next_event(&mut self) -> Result<SourceEvent, ReadError> {
            Ok(self.0.pop_front().unwrap_or(SourceEvent::End))
        }
    }

    #[test]
    fn gaps_become_nan() {
        let header = Header {
            channel: 3,
            code: "x".into(),
            start: None,
            sample_rate: None,
            kind: SampleKind::Real,
            suggestion: None,
            location: None,
        };
        let mut source = Scripted(VecDeque::from([
            SourceEvent::Header(header),
            SourceEvent::Chunk { channel: 3, samples: SampleBuf::Real(vec![1.0]) },
            SourceEvent::Gap { channel: 3, samples: 2 },
            SourceEvent::Chunk { channel: 3, samples: SampleBuf::Real(vec![4.0]) },
        ]));
        let TraceSamples::Real(samples) = &ingest(&mut source).unwrap()[0].1 else { panic!() };
        let values: Vec<f32> = samples.iter().collect();
        assert_eq!(values.len(), 4);
        assert!(values[1].is_nan() && values[2].is_nan());
        assert_eq!(values[3], 4.0);
    }

    #[test]
    fn chunks_for_unknown_channels_are_errors() {
        let mut source = Scripted(VecDeque::from([SourceEvent::Gap { channel: 0, samples: 1 }]));
        assert!(matches!(ingest(&mut source), Err(ReadError::MissingChannel(0))));
    }
}
