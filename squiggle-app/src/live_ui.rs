//! Live feeds: connect to a `heidtmare.squiggle.stream/1` server, record each
//! channel to disk as it arrives (a stream trace, checkpointed into the
//! project every 10 s and when the feed stops), plot it as it grows, follow
//! the newest data, and detect events on a sliding window, each added once
//! it is final.

use std::collections::HashMap;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Duration;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use squiggle_core::associate::AssociateSettings;
use squiggle_core::dsp::MatchedSpec;
use squiggle_core::{Change, Id, PhenomenologyId, SampleBuf, SampleKind, Timestamp, Trace, TraceSamples, TraceSource};
use squiggle_io::{Header, SourceEvent, WaveformSource};
use squiggle_io::stream::{Recorder, ReplayOptions, connect, serve};

use crate::demo::DemoOptions;
use crate::events_ui::{DetectInput, DetectMeta, default_detector, event_changes};
use crate::live::LiveDetection;
use crate::plot::Plot;
use crate::synthetic;
use crate::tracks::{Track, TrackView};
use crate::workspace::{SampleState, Workspace, WorkspaceRequest};

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
/// Most feed messages handled per frame.
const PER_FRAME: usize = 20_000;
const CHECKPOINT_S: f64 = 10.0;
/// Stream seconds per second of the demo replay.
pub const DEMO_SPEED: f64 = 10.0;

enum Message {
    Connected(TcpStream),
    Event(SourceEvent),
    Failed(String),
}

struct Channel {
    trace: Id,
    kind: SampleKind,
    recorder: Recorder,
}

pub struct Session {
    pub endpoint: String,
    project: Option<Id>,
    receiver: Receiver<Message>,
    stream: Option<TcpStream>,
    channels: HashMap<u32, Channel>,
    dir: PathBuf,
    detection: Option<LiveDetection>,
    /// Assign each new sensor its suggested phenomenology (demo replays).
    assign_suggested: bool,
    /// New sensors awaiting the analyst, with the feed's suggestion.
    suggestions: Vec<(Id, PhenomenologyId)>,
    checkpointed: f64,
    samples: u64,
    /// Stops the demo replay server.
    server: Option<Arc<AtomicBool>>,
    connected: bool,
}

#[derive(Resource)]
pub struct LiveState {
    pub open: bool,
    /// Behind a mutex only so the resource is `Sync`; accessed through `&mut`.
    session: Option<Mutex<Session>>,
    /// Keep the newest samples at the right edge.
    pub follow: bool,
    /// Detect events as they become final.
    pub detect: bool,
    endpoint: String,
    /// Why the last session ended, and what it found.
    status: Option<String>,
    /// Start a demo replay once a new project is open.
    pending_demo: bool,
    /// Connect once the app is up.
    pending_connect: Option<String>,
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            open: false,
            session: None,
            follow: true,
            detect: true,
            endpoint: "127.0.0.1:18000".to_string(),
            status: None,
            pending_demo: false,
            pending_connect: None,
        }
    }
}

pub fn live_menu(ui: &mut egui::Ui, live: &mut LiveState) {
    if ui.button("Live feed…").on_hover_text("Connect to a live stream; record, plot and detect as data arrives").clicked() {
        live.open = true;
        ui.close();
    }
}

/// Requests a new project, then replays the families scene into it as a live feed.
pub fn request_demo(live: &mut LiveState, requests: &mut MessageWriter<WorkspaceRequest>) {
    requests.write(WorkspaceRequest::New);
    live.pending_demo = true;
    live.open = true;
}

/// Startup options: `--live HOST:PORT`, `--live-demo`.
pub fn startup(options: Res<DemoOptions>, mut live: ResMut<LiveState>) {
    if let Some(endpoint) = &options.live {
        live.pending_connect = Some(endpoint.clone());
        live.open = true;
    }
    if options.live_demo {
        live.pending_demo = true;
        live.open = true;
    }
}

/// Serves the families scene on a free local port; returns the endpoint and the stop flag.
fn demo_server() -> Result<(String, Arc<AtomicBool>), String> {
    let (events, _) = synthetic::families();
    let written = synthetic::write_mixed_events(&synthetic::directory().join("families"), &events, synthetic::mixed_duration(&events))?;
    let mut recordings = Vec::new();
    for (path, _) in written {
        recordings.extend(squiggle_io::read_file(&path).map_err(|e| e.to_string())?.recordings);
    }
    for (i, r) in recordings.iter_mut().enumerate() {
        r.channel = i as u32;
    }
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let endpoint = listener.local_addr().map_err(|e| e.to_string())?.to_string();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    std::thread::spawn(move || serve(listener, Arc::new(recordings), ReplayOptions { speed: DEMO_SPEED, chunk_s: 0.5 }, flag));
    Ok((endpoint, stop))
}

/// Where a session records: beside the saved project, else in the temp directory.
fn recording_dir(workspace: &Workspace) -> PathBuf {
    let session = Timestamp::now().to_string().replace([':', '.'], "-");
    match &workspace.path {
        Some(path) => path.with_extension("live").join(session),
        None => std::env::temp_dir().join("squiggle-live").join(session),
    }
}

fn start(endpoint: String, workspace: &Workspace, detect: bool, assign_suggested: bool, server: Option<Arc<AtomicBool>>) -> Session {
    let (sender, receiver) = channel();
    let target = endpoint.clone();
    std::thread::spawn(move || {
        let mut source = match connect(&target, Duration::from_secs(5)) {
            Ok(source) => source,
            Err(error) => {
                let _ = sender.send(Message::Failed(format!("{target}: {error}")));
                return;
            }
        };
        if let Ok(stream) = source.get_ref().try_clone() {
            let _ = sender.send(Message::Connected(stream));
        }
        loop {
            let message = match source.next_event() {
                Ok(SourceEvent::End) => {
                    let _ = sender.send(Message::Event(SourceEvent::End));
                    return;
                }
                Ok(event) => Message::Event(event),
                Err(error) => {
                    let _ = sender.send(Message::Failed(error.to_string()));
                    return;
                }
            };
            if sender.send(message).is_err() {
                return;
            }
        }
    });
    Session {
        endpoint,
        project: workspace.project.id,
        receiver,
        stream: None,
        channels: HashMap::new(),
        dir: recording_dir(workspace),
        detection: detect.then(|| LiveDetection::new(Vec::new(), AssociateSettings::default())),
        assign_suggested,
        suggestions: Vec::new(),
        checkpointed: 0.0,
        samples: 0,
        server,
        connected: false,
    }
}

impl Session {
    /// A channel's header: its sensor (added when new), the reference time, and a recorded trace.
    fn header(&mut self, workspace: &mut Workspace, h: Header) -> Result<(), String> {
        let rate = h.sample_rate.filter(|r| r.is_finite() && *r > 0.0).ok_or_else(|| format!("{}: no sample rate", h.code))?;
        let sensor = match workspace.project.sensor_by_code(&h.code) {
            Some(s) => s.id,
            None => {
                let id = Id::generate();
                workspace.commit(Change::AddSensor { id, code: h.code.clone() })?;
                id
            }
        };
        let known = workspace.project.sensor(sensor).expect("added above");
        if let Some(location) = h.location.filter(|_| known.location.is_none()) {
            workspace.commit(Change::SetSensorLocation { sensor, location: Some(location) })?;
        }
        if workspace.project.sensor(sensor).is_some_and(|s| s.assignment.is_none())
            && let Some(suggestion) = &h.suggestion
        {
            if self.assign_suggested {
                workspace.commit(Change::AssignPhenomenology { sensor, phenomenology: suggestion.phenomenology.clone() })?;
            } else {
                self.suggestions.push((sensor, suggestion.phenomenology.clone()));
            }
        }
        let start = h.start.unwrap_or_else(Timestamp::now);
        if workspace.project.reference.is_none() {
            workspace.commit(Change::SetReference { reference: start })?;
        }
        let file: String = h.code.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
        let mut recorder = Recorder::create(&self.dir.join(format!("{file}-{}.f32", h.channel)), h.kind).map_err(|e| e.to_string())?;
        let (bytes, content_hash) = recorder.checkpoint().map_err(|e| e.to_string())?;
        let trace = Trace {
            id: Id::generate(),
            sensor,
            start,
            sample_rate: rate,
            len: 0,
            kind: h.kind,
            source: TraceSource::Stream {
                endpoint: self.endpoint.clone(),
                code: h.code.clone(),
                path: recorder.path().to_string_lossy().into_owned(),
                bytes,
                content_hash,
            },
            processing: Default::default(),
            alignment: None,
        };
        workspace.samples.insert(trace.id, SampleState::Loaded(TraceSamples::new(h.kind)));
        self.channels.insert(h.channel, Channel { trace: trace.id, kind: h.kind, recorder });
        workspace.commit(Change::AddTrace(trace))
    }

    fn samples(&mut self, workspace: &mut Workspace, channel: u32, samples: SampleBuf) -> Result<(), String> {
        let c = self.channels.get_mut(&channel).ok_or_else(|| format!("samples for unannounced channel {channel}"))?;
        c.recorder.append(&samples).map_err(|e| e.to_string())?;
        self.samples += samples.len() as u64;
        if let Some(SampleState::Loaded(stored)) = workspace.samples.get_mut(&c.trace) {
            stored.append(samples);
        }
        Ok(())
    }

    /// Records each channel's length and hash in the project.
    fn checkpoint(&mut self, workspace: &mut Workspace) -> Result<(), String> {
        for c in self.channels.values_mut() {
            let (bytes, content_hash) = c.recorder.checkpoint().map_err(|e| e.to_string())?;
            let size = match c.kind {
                SampleKind::Real => 4,
                SampleKind::Complex => 8,
            };
            let len = (bytes / size) as usize;
            if workspace.project.trace(c.trace).is_some_and(|t| t.len < len) {
                workspace.commit(Change::ExtendTrace { trace: c.trace, len, bytes, content_hash })?;
            }
        }
        Ok(())
    }

    /// Feeds each live track's new plotted samples to detection; adds final events.
    fn detect(&mut self, workspace: &mut Workspace, tracks: &Query<&Track>, finish: bool) -> usize {
        let Some(detection) = &mut self.detection else { return 0 };
        let Some(reference) = workspace.project.reference else { return 0 };
        let live: Vec<Id> = self.channels.values().map(|c| c.trace).collect();
        for track in tracks.iter().filter(|t| live.contains(&t.trace)) {
            let (Some(series), Some(phenomenology)) = (&track.series, &track.phenomenology_id) else { continue };
            let index = match detection.input_of(track.trace) {
                Some(index) => index,
                None => {
                    let project = &workspace.project;
                    let Some(trace) = project.trace(track.trace) else { continue };
                    let spec = default_detector(phenomenology.as_str(), trace.sample_rate);
                    let Some(input) = DetectInput::new(project, trace, Vec::new(), spec, AssociateSettings::default().radius_m) else { continue };
                    detection.add(input.with_templates(project, MatchedSpec::default()))
                }
            };
            let fed = detection.fed(index);
            if series.values.len() > fed {
                let new: Vec<f32> = series.values.range(fed, series.values.len()).collect();
                detection.push(index, &new);
            }
        }
        let outcome = if finish { detection.finish() } else { detection.close() };
        if outcome.candidates.is_empty() {
            return 0;
        }
        let meta: Vec<DetectMeta> = detection.inputs().into_iter().map(DetectMeta::from).collect();
        let changes = event_changes(&workspace.project, &meta, &outcome, reference, false, "live");
        let n = changes.iter().filter(|c| matches!(c, Change::AddEvent(_))).count();
        if n > 0 {
            workspace.execute_or_report(format!("Live: {n} event{}", if n == 1 { "" } else { "s" }), changes);
        }
        n
    }

    /// Handles queued feed messages; why the feed ended, if it did.
    fn drain(&mut self, workspace: &mut Workspace) -> Option<String> {
        let mut ended: Option<String> = None;
        for _ in 0..PER_FRAME {
            let outcome = match self.receiver.try_recv() {
                Ok(Message::Connected(stream)) => {
                    self.stream = Some(stream);
                    self.connected = true;
                    Ok(())
                }
                Ok(Message::Event(SourceEvent::Header(h))) => self.header(workspace, h),
                Ok(Message::Event(SourceEvent::Chunk { channel, samples })) => self.samples(workspace, channel, samples),
                Ok(Message::Event(SourceEvent::Gap { channel, samples })) => match self.channels.get(&channel) {
                    Some(c) => {
                        let gap = SampleBuf::gap(c.kind, samples);
                        self.samples(workspace, channel, gap)
                    }
                    None => Err(format!("gap for unannounced channel {channel}")),
                },
                Ok(Message::Event(SourceEvent::End)) => {
                    ended = Some("The feed ended.".to_string());
                    break;
                }
                Ok(Message::Failed(error)) => {
                    ended = Some(format!("Feed failed: {error}"));
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    ended = Some("Disconnected.".to_string());
                    break;
                }
            };
            if let Err(error) = outcome {
                ended = Some(format!("Stopped: {error}"));
                break;
            }
        }
        ended
    }

    fn stop(&mut self) {
        if let Some(stream) = &self.stream {
            let _ = stream.shutdown(Shutdown::Both);
        }
        if let Some(server) = &self.server {
            server.store(true, Ordering::Relaxed);
        }
    }
}

/// Drains the feed, records, detects and follows.
#[allow(clippy::too_many_arguments)]
pub fn poll_live(
    mut workspace: ResMut<Workspace>,
    mut live: ResMut<LiveState>,
    tracks: Query<&Track>,
    mut view: ResMut<TrackView>,
    plot: Res<Plot>,
    time: Res<Time>,
) {
    let live = &mut *live;
    if let Some(endpoint) = live.pending_connect.take() {
        live.session = Some(Mutex::new(start(endpoint, &workspace, live.detect, false, None)));
    }
    if live.pending_demo && live.session.is_none() && workspace.project.traces().is_empty() && workspace.confirm_discard.is_none() {
        live.pending_demo = false;
        match demo_server() {
            Ok((endpoint, stop)) => {
                live.endpoint = endpoint.clone();
                live.session = Some(Mutex::new(start(endpoint, &workspace, live.detect, true, Some(stop))));
            }
            Err(error) => live.status = Some(error),
        }
    }
    let Some(session) = live.session.as_mut().map(|m| m.get_mut().unwrap_or_else(|p| p.into_inner())) else { return };
    if session.project != workspace.project.id {
        session.stop();
        live.session = None;
        live.status = Some("Stopped: the project was replaced.".to_string());
        return;
    }
    let mut ended = session.drain(&mut workspace);

    let now = time.elapsed_secs_f64();
    session.detect(&mut workspace, &tracks, false);
    if now - session.checkpointed >= CHECKPOINT_S || ended.is_some() {
        session.checkpointed = now;
        if let Err(error) = session.checkpoint(&mut workspace) {
            ended = Some(format!("Not recorded: {error}"));
        }
    }

    if live.follow {
        let newest = tracks
            .iter()
            .filter(|t| session.channels.values().any(|c| c.trace == t.trace))
            .map(Track::end)
            .fold(f64::NEG_INFINITY, f64::max);
        let width = plot.view.plot_width();
        if newest.is_finite() && width > 0.0 && view.time.px_per_s > 0.0 {
            view.time.start = newest - 0.97 * width / view.time.px_per_s;
        }
    }

    if let Some(reason) = ended {
        session.detect(&mut workspace, &tracks, true);
        session.stop();
        let events = session.detection.as_ref().map_or(0, |d| d.events);
        live.status = Some(format!(
            "{reason} {} channels, {} samples recorded in {}; {events} events detected live.",
            session.channels.len(),
            session.samples,
            session.dir.display()
        ));
        live.session = None;
    }
}

/// Stops the session from the window.
pub fn stop(live: &mut LiveState, workspace: &mut Workspace, tracks: &Query<&Track>) {
    if let Some(session) = live.session.as_mut().map(|m| m.get_mut().unwrap_or_else(|p| p.into_inner())) {
        session.stop();
        session.detect(workspace, tracks, true);
        let _ = session.checkpoint(workspace);
        live.status = Some(format!("Stopped. {} samples recorded in {}.", session.samples, session.dir.display()));
        live.session = None;
    }
}

pub fn window(mut contexts: EguiContexts, mut workspace: ResMut<Workspace>, mut live: ResMut<LiveState>, tracks: Query<&Track>) -> Result {
    if !live.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?.clone();
    let live = &mut *live;
    let mut open = true;
    let mut assign: Vec<Change> = Vec::new();
    let mut stop_now = false;
    egui::Window::new("Live Feed").open(&mut open).default_width(460.0).show(&ctx, |ui| {
        ui.weak(format!(
            "A {} server (e.g. squiggle-stream replaying files). Each channel is recorded to disk as it arrives and checkpointed into the project.",
            squiggle_io::stream::PROTOCOL
        ));
        ui.horizontal(|ui| {
            ui.label("Server");
            ui.add_enabled(live.session.is_none(), egui::TextEdit::singleline(&mut live.endpoint).desired_width(180.0));
            if live.session.is_none() {
                if ui.button("Connect").clicked() {
                    live.status = None;
                    live.session = Some(Mutex::new(start(live.endpoint.clone(), &workspace, live.detect, false, None)));
                }
            } else if ui.button("Stop").clicked() {
                stop_now = true;
            }
        });
        ui.checkbox(&mut live.follow, "Follow live").on_hover_text("Keep the newest samples at the right edge");
        ui.add_enabled(live.session.is_none(), egui::Checkbox::new(&mut live.detect, "Detect events as they become final"))
            .on_hover_text("Each assigned trace's default detector, plus the matched filter where templates exist; association over a sliding window");
        if let Some(session) = live.session.as_mut().map(|m| &*m.get_mut().unwrap_or_else(|p| p.into_inner())) {
            ui.separator();
            ui.label(if session.connected { format!("Connected to {}", session.endpoint) } else { format!("Connecting to {}…", session.endpoint) });
            ui.label(format!("{} channels, {} samples recorded in {}", session.channels.len(), session.samples, session.dir.display()));
            if let Some(d) = &session.detection {
                ui.label(format!("{} events; {} arrivals open; events close {:.0} s after their first arrival", d.events, d.open(), d.horizon));
            }
            let waiting: Vec<&(Id, PhenomenologyId)> =
                session.suggestions.iter().filter(|(s, _)| workspace.project.sensor(*s).is_some_and(|s| s.assignment.is_none())).collect();
            if !waiting.is_empty() {
                ui.colored_label(WARN_COLOR, format!("{} new sensors have no phenomenology; they are recorded but not processed.", waiting.len()));
                let names: Vec<String> = waiting
                    .iter()
                    .map(|(s, p)| format!("{} → {p}", workspace.project.sensor(*s).map_or("?", |s| s.code.as_str())))
                    .collect();
                if ui.button("Assign the suggested phenomenologies").on_hover_text(names.join("\n")).clicked() {
                    assign = waiting.iter().map(|(sensor, phenomenology)| Change::AssignPhenomenology { sensor: *sensor, phenomenology: phenomenology.clone() }).collect();
                }
            }
        }
        if let Some(status) = &live.status {
            ui.separator();
            ui.label(status);
        }
    });
    if stop_now {
        stop(live, &mut workspace, &tracks);
    }
    if !assign.is_empty() {
        workspace.execute_or_report("Assign live sensors", assign);
    }
    live.open = open;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::Analyst;

    /// A served replay is recorded channel by channel, its sensors placed and
    /// assigned, checkpointed into the project, and reloads from disk exactly.
    #[test]
    fn a_live_feed_is_recorded_and_reloads_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let paths = synthetic::write_swarm(&dir.path().join("swarm")).unwrap();
        let mut recordings = Vec::new();
        for path in &paths {
            recordings.extend(squiggle_io::read_file(path).unwrap().recordings);
        }
        for (i, r) in recordings.iter_mut().enumerate() {
            r.channel = i as u32;
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let stop = Arc::new(AtomicBool::new(false));
        let served = Arc::new(recordings.clone());
        let flag = stop.clone();
        std::thread::spawn(move || serve(listener, served, ReplayOptions { speed: f64::INFINITY, chunk_s: 0.25 }, flag));

        let mut workspace = Workspace::new(Analyst("ana".into()));
        let mut session = start(endpoint, &workspace, false, true, Some(stop.clone()));
        session.dir = dir.path().join("live");
        let began = std::time::Instant::now();
        let reason = loop {
            if let Some(reason) = session.drain(&mut workspace) {
                break reason;
            }
            assert!(began.elapsed() < Duration::from_secs(60), "feed timed out");
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(reason, "The feed ended.");
        session.checkpoint(&mut workspace).unwrap();
        session.stop();

        let project = &workspace.project;
        assert_eq!(project.traces().len(), recordings.len());
        assert!(project.reference.is_some());
        for (trace, recording) in project.traces().iter().zip(&recordings) {
            let sensor = project.sensor(trace.sensor).unwrap();
            assert_eq!(sensor.code, recording.code);
            assert!(sensor.location.is_some() && project.trace_phenomenology(trace).is_some_and(|p| p.id.as_str() == "seismic"));
            assert!(trace.source.is_stream());
            assert_eq!(trace.len, recording.samples.len());
            let SampleBuf::Real(expected) = &recording.samples else { panic!() };
            let TraceSamples::Real(reloaded) = squiggle_io::load_trace(trace).unwrap() else { panic!() };
            assert_eq!(&reloaded.iter().collect::<Vec<_>>(), expected);
            let Some(SampleState::Loaded(TraceSamples::Real(live))) = workspace.samples.get(&trace.id) else { panic!() };
            assert_eq!(&live.iter().collect::<Vec<_>>(), expected);
        }
    }
}
