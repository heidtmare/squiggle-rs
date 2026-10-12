//! Live feeds over TCP (`heidtmare.squiggle.stream/1`) and their recording.
//!
//! A server greets with `SQGL` and a little-endian `u16` version, then sends
//! frames: a type byte, a little-endian `u32` payload length, the payload.
//!
//! | type | payload |
//! | --- | --- |
//! | 1 header | JSON: `channel`, `code`, `start` (RFC 3339), `sample_rate`, `complex`, optional `latitude`, `longitude`, `elevation_m` |
//! | 2 chunk | `u32` channel, then f32 samples (I/Q pairs when complex) |
//! | 3 gap | `u32` channel, `u64` missing samples |
//! | 4 end | empty |
//!
//! [`TcpSource`] turns a connection into [`SourceEvent`]s; [`serve`] replays
//! recordings as a feed paced by their timestamps; [`Recorder`] appends a
//! channel's samples to disk for [`TraceSource::Stream`].

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use squiggle_core::phenomenology::suggest_from_seed_code;
use squiggle_core::{Location, SampleBuf, SampleKind, Timestamp, TraceSamples, TraceSource};

use crate::source::{Header, SourceEvent, WaveformSource};
use crate::{ReadError, Recording};

pub const PROTOCOL: &str = "heidtmare.squiggle.stream/1";
const MAGIC: &[u8; 4] = b"SQGL";
const VERSION: u16 = 1;
const HEADER: u8 = 1;
const CHUNK: u8 = 2;
const GAP: u8 = 3;
const END: u8 = 4;
/// Largest payload accepted: 64 MiB.
const MAX_FRAME: u32 = 64 << 20;

#[derive(Serialize, Deserialize)]
struct WireHeader {
    channel: u32,
    code: String,
    start: Option<String>,
    sample_rate: Option<f64>,
    complex: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    latitude: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    longitude: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    elevation_m: Option<f64>,
}

/// The greeting a server sends first.
pub fn greeting() -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend(VERSION.to_le_bytes());
    out
}

/// One event as a frame. `End` included.
pub fn encode(event: &SourceEvent) -> Vec<u8> {
    let (kind, payload) = match event {
        SourceEvent::Header(h) => {
            let wire = WireHeader {
                channel: h.channel,
                code: h.code.clone(),
                start: h.start.map(|t| t.to_string()),
                sample_rate: h.sample_rate,
                complex: h.kind == SampleKind::Complex,
                latitude: h.location.map(|l| l.latitude),
                longitude: h.location.map(|l| l.longitude),
                elevation_m: h.location.map(|l| l.elevation_m),
            };
            (HEADER, serde_json::to_vec(&wire).expect("plain data"))
        }
        SourceEvent::Chunk { channel, samples } => {
            let mut payload = channel.to_le_bytes().to_vec();
            payload.extend(sample_bytes(samples));
            (CHUNK, payload)
        }
        SourceEvent::Gap { channel, samples } => {
            let mut payload = channel.to_le_bytes().to_vec();
            payload.extend((*samples as u64).to_le_bytes());
            (GAP, payload)
        }
        SourceEvent::End => (END, Vec::new()),
    };
    let mut out = vec![kind];
    out.extend((payload.len() as u32).to_le_bytes());
    out.extend(payload);
    out
}

/// Samples as little-endian f32, I then Q.
pub fn sample_bytes(samples: &SampleBuf) -> Vec<u8> {
    match samples {
        SampleBuf::Real(s) => s.iter().flat_map(|v| v.to_le_bytes()).collect(),
        SampleBuf::Complex(s) => s.iter().flat_map(|[i, q]| i.to_le_bytes().into_iter().chain(q.to_le_bytes())).collect(),
    }
}

fn decode_samples(bytes: &[u8], kind: SampleKind) -> Result<SampleBuf, ReadError> {
    let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    match kind {
        SampleKind::Real if bytes.len().is_multiple_of(4) => Ok(SampleBuf::Real(values)),
        SampleKind::Complex if bytes.len().is_multiple_of(8) => Ok(SampleBuf::Complex(values.chunks_exact(2).map(|p| [p[0], p[1]]).collect())),
        _ => Err(ReadError::Format("sample bytes do not fill whole samples".to_string())),
    }
}

/// Decodes frames from any reader: a socket, or bytes in a test.
pub struct FrameReader<R> {
    reader: R,
    kinds: Vec<(u32, SampleKind)>,
    greeted: bool,
    ended: bool,
}

impl<R: Read> FrameReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader, kinds: Vec::new(), greeted: false, ended: false }
    }

    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<bool, ReadError> {
        match self.reader.read_exact(buf) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

impl<R: Read + Send> WaveformSource for FrameReader<R> {
    /// A closed connection ends the stream.
    fn next_event(&mut self) -> Result<SourceEvent, ReadError> {
        if self.ended {
            return Ok(SourceEvent::End);
        }
        if !self.greeted {
            let mut hello = [0u8; 6];
            if !self.read_exact(&mut hello)? || &hello[..4] != MAGIC {
                return Err(ReadError::Format(format!("not a {PROTOCOL} feed")));
            }
            let version = u16::from_le_bytes([hello[4], hello[5]]);
            if version != VERSION {
                return Err(ReadError::Format(format!("stream version {version} is not {VERSION}")));
            }
            self.greeted = true;
        }
        let mut head = [0u8; 5];
        if !self.read_exact(&mut head)? {
            self.ended = true;
            return Ok(SourceEvent::End);
        }
        let len = u32::from_le_bytes([head[1], head[2], head[3], head[4]]);
        if len > MAX_FRAME {
            return Err(ReadError::Format(format!("frame of {len} bytes")));
        }
        let mut payload = vec![0u8; len as usize];
        if !self.read_exact(&mut payload)? {
            return Err(ReadError::Format("feed closed mid-frame".to_string()));
        }
        let channel = || payload.get(..4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| ReadError::Format("short frame".to_string()));
        match head[0] {
            HEADER => {
                let wire: WireHeader = serde_json::from_slice(&payload).map_err(|e| ReadError::Format(format!("header: {e}")))?;
                let start = match &wire.start {
                    Some(text) => Some(text.parse::<Timestamp>().map_err(|e| ReadError::Format(format!("header start: {e}")))?),
                    None => None,
                };
                let kind = if wire.complex { SampleKind::Complex } else { SampleKind::Real };
                self.kinds.retain(|k| k.0 != wire.channel);
                self.kinds.push((wire.channel, kind));
                let location = match (wire.latitude, wire.longitude) {
                    (Some(latitude), Some(longitude)) => Some(Location { latitude, longitude, elevation_m: wire.elevation_m.unwrap_or(0.0) }),
                    _ => None,
                };
                let suggestion = suggest_from_seed_code(&wire.code);
                Ok(SourceEvent::Header(Header { channel: wire.channel, code: wire.code, start, sample_rate: wire.sample_rate, kind, suggestion, location }))
            }
            CHUNK => {
                let channel = channel()?;
                let kind = self.kinds.iter().find(|k| k.0 == channel).map(|k| k.1).ok_or(ReadError::MissingChannel(channel))?;
                Ok(SourceEvent::Chunk { channel, samples: decode_samples(&payload[4..], kind)? })
            }
            GAP => {
                let channel = channel()?;
                let count = payload.get(4..12).map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes"))).ok_or_else(|| ReadError::Format("short gap".to_string()))?;
                Ok(SourceEvent::Gap { channel, samples: count as usize })
            }
            END => {
                self.ended = true;
                Ok(SourceEvent::End)
            }
            other => Err(ReadError::Format(format!("unknown frame type {other}"))),
        }
    }
}

/// A live feed over TCP.
pub type TcpSource = FrameReader<TcpStream>;

/// Connects to a feed at `endpoint` (`host:port`).
pub fn connect(endpoint: &str, timeout: Duration) -> Result<TcpSource, ReadError> {
    let address = endpoint
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| ReadError::Format(format!("{endpoint}: no address")))?;
    let stream = TcpStream::connect_timeout(&address, timeout)?;
    stream.set_nodelay(true)?;
    Ok(FrameReader::new(stream))
}

/// How a replay is paced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayOptions {
    /// Stream seconds per wall-clock second; infinite sends as fast as possible.
    pub speed: f64,
    /// Stream seconds per chunk.
    pub chunk_s: f64,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self { speed: 1.0, chunk_s: 0.5 }
    }
}

/// Every recording's events in stream order: headers, then chunks of
/// `chunk_s` interleaved by time, then `End`. Each event carries its stream
/// time (seconds from the earliest start).
pub fn replay_events(recordings: &[Recording], chunk_s: f64) -> Vec<(f64, SourceEvent)> {
    let starts: Vec<Option<Timestamp>> = recordings.iter().map(|r| r.start).collect();
    let first = starts.iter().flatten().copied().reduce(|a, b| if b.seconds_since(a) < 0.0 { b } else { a });
    let mut events: Vec<(f64, SourceEvent)> = recordings
        .iter()
        .map(|r| {
            (
                f64::NEG_INFINITY,
                SourceEvent::Header(Header {
                    channel: r.channel,
                    code: r.code.clone(),
                    start: r.start,
                    sample_rate: r.sample_rate,
                    kind: r.samples.kind(),
                    suggestion: r.suggestion.clone(),
                    location: r.location,
                }),
            )
        })
        .collect();
    let mut chunks: Vec<(f64, usize, SourceEvent)> = Vec::new();
    for (k, r) in recordings.iter().enumerate() {
        let rate = r.sample_rate.unwrap_or(1.0);
        let offset = match (r.start, first) {
            (Some(s), Some(f)) => s.seconds_since(f),
            _ => 0.0,
        };
        let n = ((chunk_s * rate).round() as usize).max(1);
        let pieces: Vec<SampleBuf> = match &r.samples {
            SampleBuf::Real(s) => s.chunks(n).map(|c| SampleBuf::Real(c.to_vec())).collect(),
            SampleBuf::Complex(s) => s.chunks(n).map(|c| SampleBuf::Complex(c.to_vec())).collect(),
        };
        for (i, samples) in pieces.into_iter().enumerate() {
            // Sent once its last sample has been "recorded".
            let at = offset + ((i * n + samples.len()) as f64) / rate;
            chunks.push((at, k, SourceEvent::Chunk { channel: r.channel, samples }));
        }
    }
    chunks.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    events.extend(chunks.into_iter().map(|(at, _, e)| (at, e)));
    let last = events.last().map_or(0.0, |e| e.0.max(0.0));
    events.push((last, SourceEvent::End));
    events
}

/// Replays `recordings` to each client of `listener` until `stop` is set.
/// Each client gets the whole replay from the start, paced by `options`.
pub fn serve(listener: TcpListener, recordings: Arc<Vec<Recording>>, options: ReplayOptions, stop: Arc<AtomicBool>) -> std::io::Result<()> {
    listener.set_nonblocking(true)?;
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let (recordings, stop) = (recordings.clone(), stop.clone());
                std::thread::spawn(move || {
                    let _ = replay_to(stream, &recordings, options, &stop);
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn replay_to(mut stream: TcpStream, recordings: &[Recording], options: ReplayOptions, stop: &AtomicBool) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.write_all(&greeting())?;
    let began = Instant::now();
    for (at, event) in replay_events(recordings, options.chunk_s) {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if options.speed.is_finite() && at.is_finite() {
            let due = Duration::from_secs_f64((at / options.speed).max(0.0));
            if let Some(wait) = due.checked_sub(began.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        stream.write_all(&encode(&event))?;
    }
    stream.flush()
}

/// Appends one channel's samples to a file, little-endian f32 (I/Q pairs).
pub struct Recorder {
    path: PathBuf,
    file: std::fs::File,
    kind: SampleKind,
    bytes: u64,
}

impl Recorder {
    pub fn create(path: &Path, kind: SampleKind) -> Result<Self, ReadError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self { path: path.to_path_buf(), file: std::fs::File::create(path)?, kind, bytes: 0 })
    }

    pub fn append(&mut self, samples: &SampleBuf) -> Result<(), ReadError> {
        if samples.kind() != self.kind {
            return Err(ReadError::Format("sample kind changed mid-stream".to_string()));
        }
        let bytes = sample_bytes(samples);
        self.file.write_all(&bytes)?;
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Flushes, and returns the recorded length and the hash of its bytes.
    pub fn checkpoint(&mut self) -> Result<(u64, String), ReadError> {
        self.file.flush()?;
        let bytes = std::fs::read(&self.path)?;
        let n = self.bytes.min(bytes.len() as u64) as usize;
        Ok((n as u64, blake3::hash(&bytes[..n]).to_hex().to_string()))
    }
}

/// Samples of a stream-recorded trace: its first `bytes`, checked against their hash.
pub fn load_stream(source: &TraceSource, kind: SampleKind) -> Result<TraceSamples, ReadError> {
    let TraceSource::Stream { path, bytes, content_hash, .. } = source else {
        return Err(ReadError::Format("not a stream recording".to_string()));
    };
    let data = std::fs::read(path)?;
    let n = *bytes as usize;
    if data.len() < n || blake3::hash(&data[..n]).to_hex().as_str() != content_hash {
        return Err(ReadError::Changed(PathBuf::from(path)));
    }
    let mut samples = TraceSamples::new(kind);
    samples.append(decode_samples(&data[..n], kind)?);
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ingest;

    fn recording(channel: u32, start: f64, rate: f64, samples: SampleBuf) -> Recording {
        Recording {
            code: format!("SY.S{channel}..HHZ"),
            channel,
            start: Some(Timestamp::new(1_000, 0.0).add_seconds(start)),
            sample_rate: Some(rate),
            samples,
            suggestion: None,
            location: Some(Location { latitude: 35.0, longitude: -106.0, elevation_m: 12.0 }),
        }
    }

    fn recordings() -> Vec<Recording> {
        vec![
            recording(0, 0.0, 100.0, SampleBuf::Real((0..1234).map(|i| i as f32).collect())),
            recording(1, 0.37, 20.0, SampleBuf::Complex((0..77).map(|i| [i as f32, -(i as f32)]).collect())),
        ]
    }

    #[test]
    fn frames_round_trip_into_the_same_samples() {
        let mut bytes = greeting();
        for (_, event) in replay_events(&recordings(), 0.25) {
            bytes.extend(encode(&event));
        }
        let mut source = FrameReader::new(std::io::Cursor::new(bytes));
        let channels = ingest(&mut source).unwrap();
        assert_eq!(channels.len(), 2);
        let (header, samples) = &channels[1];
        assert_eq!((header.code.as_str(), header.kind, header.sample_rate), ("SY.S1..HHZ", SampleKind::Complex, Some(20.0)));
        assert!((header.start.unwrap().seconds_since(Timestamp::new(1_000, 0.0)) - 0.37).abs() < 1e-6);
        assert_eq!(header.location.unwrap().elevation_m, 12.0);
        assert_eq!(header.suggestion.as_ref().map(|s| s.phenomenology.clone()), Some(squiggle_core::PhenomenologyId::new("seismic")));
        assert_eq!(samples.len(), 77);
        let TraceSamples::Real(real) = &channels[0].1 else { panic!() };
        assert_eq!(real.iter().collect::<Vec<_>>(), (0..1234).map(|i| i as f32).collect::<Vec<_>>());
        assert!(real.chunks().len() > 40, "arrived in chunks");
    }

    #[test]
    fn replay_orders_chunks_by_time_and_bad_feeds_are_refused() {
        let events = replay_events(&recordings(), 0.5);
        let times: Vec<f64> = events.iter().map(|e| e.0).filter(|t| t.is_finite()).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
        assert!(matches!(events.last(), Some((_, SourceEvent::End))));
        let mut bad = FrameReader::new(std::io::Cursor::new(b"HTTP/1.1".to_vec()));
        assert!(bad.next_event().is_err());
        let mut truncated = greeting();
        truncated.extend(&encode(&events[0].1)[..7]);
        assert!(FrameReader::new(std::io::Cursor::new(truncated)).next_event().is_err());
    }

    #[test]
    fn a_served_replay_arrives_over_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let stop = Arc::new(AtomicBool::new(false));
        let server = {
            let (stop, recordings) = (stop.clone(), Arc::new(recordings()));
            std::thread::spawn(move || serve(listener, recordings, ReplayOptions { speed: f64::INFINITY, chunk_s: 0.1 }, stop))
        };
        let mut source = connect(&endpoint, Duration::from_secs(5)).unwrap();
        let channels = ingest(&mut source).unwrap();
        stop.store(true, Ordering::Relaxed);
        server.join().unwrap().unwrap();
        assert_eq!(channels.iter().map(|c| c.1.len()).collect::<Vec<_>>(), [1234, 77]);
    }

    #[test]
    fn recordings_reload_up_to_their_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live").join("a.f32");
        let mut recorder = Recorder::create(&path, SampleKind::Real).unwrap();
        recorder.append(&SampleBuf::Real(vec![1.0, 2.0])).unwrap();
        let (bytes, content_hash) = recorder.checkpoint().unwrap();
        recorder.append(&SampleBuf::Real(vec![3.0])).unwrap();
        recorder.checkpoint().unwrap();
        let source = TraceSource::Stream { endpoint: "e".into(), code: "c".into(), path: path.to_string_lossy().into(), bytes, content_hash };
        let TraceSamples::Real(samples) = load_stream(&source, SampleKind::Real).unwrap() else { panic!() };
        assert_eq!(samples.iter().collect::<Vec<_>>(), [1.0, 2.0], "the checkpoint, not what followed");
        std::fs::write(&path, [0u8; 12]).unwrap();
        assert!(matches!(load_stream(&source, SampleKind::Real), Err(ReadError::Changed(_))));
    }
}
