//! Shared projects: share the open project on a `heidtmare.squiggle.sync/1`
//! server or open one others share, then sync edits both ways (every few
//! seconds, or on demand). Trace and model files travel with the project and
//! land beside the project file (`<project>.files/<hash>/`). Edits the server
//! holds back are listed for the analyst to settle.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Duration;

use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task, futures::check_ready};
use bevy_egui::{EguiContexts, egui};
use squiggle_core::sync::{Conflict, describe, fields};
use squiggle_core::{Analyst, Change, Id, Op, Project, TraceSource};
use squiggle_io::sync::{Files, ProjectInfo, Remote, SyncClient, Synced, blob_dir, cached_source, source_hash, store_blob, trace_files};
use squiggle_io::{PROJECT_EXTENSION, ProjectStore, SqliteStore};
use squiggle_models::package::{Package, SPEC_FILE};

use crate::demo::DemoOptions;
use crate::workspace::{Workspace, WorkspaceRequest};
use crate::theme;

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
/// Seconds between automatic syncs.
const AUTO_S: f64 = 5.0;
const TIMEOUT: Duration = Duration::from_secs(5);

/// A file set the project refers to by hash.
#[derive(Clone, Debug, PartialEq)]
enum Blob {
    Trace(TraceSource),
    /// Package directory and hash.
    Model(String, String),
}

impl Blob {
    fn hash(&self) -> &str {
        match self {
            Self::Trace(source) => source_hash(source),
            Self::Model(_, hash) => hash,
        }
    }

    /// Where the files are on this machine: their own path.
    fn local(&self) -> &Path {
        match self {
            Self::Trace(TraceSource::File { path, .. } | TraceSource::Stream { path, .. }) | Self::Model(path, _) => Path::new(path),
        }
    }

    /// Reads the files from their own path, else from the cache.
    fn files(&self, cache: Option<&Path>) -> Option<Files> {
        let read = |blob: &Self| match blob {
            Self::Trace(source) => trace_files(source).ok(),
            Self::Model(dir, _) => model_files(Path::new(dir)).ok(),
        };
        read(self).filter(|f| squiggle_io::sync::files_hash(f) == self.hash()).or_else(|| {
            let cache = cache?;
            let cached = match self {
                Self::Trace(source) => Self::Trace(cached_source(source, cache)),
                Self::Model(_, hash) => Self::Model(blob_dir(cache, hash).to_string_lossy().into_owned(), hash.clone()),
            };
            read(&cached)
        })
    }
}

/// A model package's files in hash order: the spec, then the weights.
fn model_files(dir: &Path) -> Result<Files, String> {
    let package = Package::load(dir)?;
    let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| e.to_string());
    Ok(vec![(SPEC_FILE.to_string(), read(SPEC_FILE)?), (package.spec.weights.file.clone(), read(&package.spec.weights.file)?)])
}

/// Files behind the traces and models that `ops` add or change, or behind the whole project.
fn blobs(project: &Project, ops: Option<&[Op]>) -> Vec<Blob> {
    let mut out: Vec<Blob> = Vec::new();
    let mut add = |blob: Blob| {
        if !out.iter().any(|b| b.hash() == blob.hash()) {
            out.push(blob);
        }
    };
    match ops {
        None => {
            project.traces().iter().for_each(|t| add(Blob::Trace(t.source.clone())));
            project.models().iter().for_each(|m| add(Blob::Model(m.path.clone(), m.hash.clone())));
        }
        Some(ops) => {
            for op in ops {
                match &op.change {
                    Change::AddTrace(t) => project.trace(t.id).into_iter().for_each(|t| add(Blob::Trace(t.source.clone()))),
                    Change::ExtendTrace { trace, .. } => project.trace(*trace).into_iter().for_each(|t| add(Blob::Trace(t.source.clone()))),
                    Change::SetModel(m) => project.model(&m.id).into_iter().for_each(|m| add(Blob::Model(m.path.clone(), m.hash.clone()))),
                    _ => {}
                }
            }
        }
    }
    out
}

/// Sends the blobs the server lacks; returns how many.
fn upload(client: &mut SyncClient, blobs: &[Blob], cache: Option<&Path>) -> Result<usize, String> {
    let hashes: Vec<String> = blobs.iter().map(|b| b.hash().to_string()).collect();
    let missing = client.missing(&hashes).map_err(|e| e.to_string())?;
    let mut sent = 0;
    for blob in blobs.iter().filter(|b| missing.iter().any(|h| h == b.hash())) {
        // Files missing here too are someone else's to send.
        if let Some(files) = blob.files(cache) {
            client.put_blob(blob.hash(), &files).map_err(|e| e.to_string())?;
            sent += 1;
        }
    }
    Ok(sent)
}

/// Fetches into `cache` the blobs whose own path is missing here; returns how many.
fn download(client: &mut SyncClient, project: &Project, cache: &Path) -> Result<usize, String> {
    let mut fetched = 0;
    for blob in blobs(project, None) {
        if blob.local().exists() || blob_dir(cache, blob.hash()).is_dir() {
            continue;
        }
        match client.get_blob(blob.hash()) {
            Ok(files) => {
                store_blob(cache, blob.hash(), &files).map_err(|e| e.to_string())?;
                fetched += 1;
            }
            // Not uploaded yet (its owner is offline): try again next sync.
            Err(squiggle_io::sync::SyncError::Server(_)) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(fetched)
}

/// Shares a project's whole log and files; returns its link.
pub fn share(endpoint: &str, analyst: &Analyst, log: &[Op], project: &Project, cache: Option<&Path>) -> Result<Remote, String> {
    let id = project.id.ok_or("the project has no id")?;
    let mut client = SyncClient::connect(endpoint, analyst, TIMEOUT).map_err(|e| e.to_string())?;
    upload(&mut client, &blobs(project, None), cache)?;
    let head = client.create(log).map_err(|e| e.to_string())?;
    Ok(Remote { endpoint: endpoint.to_string(), project: id, confirmed: head })
}

/// Pushes the pending ops, pulls the rest, and fetches files now referenced.
pub fn sync(analyst: &Analyst, remote: &Remote, log: &[Op], project: &Project, cache: Option<&Path>) -> Result<(Synced, usize), String> {
    let base = remote.confirmed.min(log.len());
    let pending = &log[base..];
    let mut client = SyncClient::connect(&remote.endpoint, analyst, TIMEOUT).map_err(|e| e.to_string())?;
    upload(&mut client, &blobs(project, Some(pending)), cache)?;
    let synced = client.sync(remote.project, base, pending).map_err(|e| e.to_string())?;
    let fetched = match cache {
        Some(cache) => {
            let merged = Project::replay(log[..base].iter().chain(&synced.ops)).map_err(|e| e.to_string())?;
            download(&mut client, &merged, cache)?
        }
        None => 0,
    };
    Ok((synced, fetched))
}

pub fn list(endpoint: &str, analyst: &Analyst) -> Result<Vec<ProjectInfo>, String> {
    SyncClient::connect(endpoint, analyst, TIMEOUT).and_then(|mut c| c.list()).map_err(|e| e.to_string())
}

/// Saves a shared project to `path`, with its files beside it, ready to open.
pub fn join(endpoint: &str, analyst: &Analyst, project: Id, path: &Path) -> Result<(), String> {
    let mut client = SyncClient::connect(endpoint, analyst, TIMEOUT).map_err(|e| e.to_string())?;
    let synced = client.sync(project, 0, &[]).map_err(|e| e.to_string())?;
    let replayed = Project::replay(&synced.ops).map_err(|e| e.to_string())?;
    download(&mut client, &replayed, &path.with_extension("files"))?;
    let mut store = SqliteStore::create(path).map_err(|e| e.to_string())?;
    store.append(&synced.ops).map_err(|e| e.to_string())?;
    let remote = Remote { endpoint: endpoint.to_string(), project, confirmed: synced.head };
    Remote::save(&mut store, Some(&remote)).map_err(|e| e.to_string())
}

enum Done {
    Shared(Result<Remote, String>),
    /// The sync and how many pending ops it sent.
    Synced(Result<(Synced, usize), String>, usize),
    Listed(Result<Vec<ProjectInfo>, String>),
    Joined(Result<PathBuf, String>),
}

struct Job {
    project: Option<Id>,
    receiver: Receiver<Done>,
}

fn spawn(project: Option<Id>, work: impl FnOnce() -> Done + Send + 'static) -> Mutex<Job> {
    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        let _ = sender.send(work());
    });
    Mutex::new(Job { project, receiver })
}

#[derive(Resource)]
pub struct SyncState {
    pub open: bool,
    endpoint: String,
    /// Sync every few seconds while the project is shared.
    pub auto: bool,
    /// Behind a mutex only so the resource is `Sync`; accessed through `&mut`.
    job: Option<Mutex<Job>>,
    status: Option<String>,
    /// When the last sync finished, app seconds.
    last: f64,
    projects: Vec<ProjectInfo>,
    /// The save-as dialog for a project being opened from the server.
    dialog: Option<(Id, Task<Option<PathBuf>>)>,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            open: false,
            endpoint: "127.0.0.1:18100".to_string(),
            auto: true,
            job: None,
            status: None,
            last: f64::NEG_INFINITY,
            projects: Vec::new(),
            dialog: None,
        }
    }
}

pub fn sync_menu(ui: &mut egui::Ui, state: &mut SyncState) {
    if ui.button("Shared project…").on_hover_text("Share this project on a server, open one others share, and sync edits").clicked() {
        state.open = true;
        ui.close();
    }
}

/// Startup options: `--analyst NAME`, `--server HOST:PORT`.
pub fn startup(options: Res<DemoOptions>, mut workspace: ResMut<Workspace>, mut state: ResMut<SyncState>) {
    if let Some(name) = &options.analyst {
        workspace.analyst = Analyst(name.clone());
    }
    if let Some(server) = &options.server {
        state.endpoint = server.clone();
        state.open = true;
    }
}

fn start_sync(workspace: &Workspace) -> Option<Mutex<Job>> {
    let remote = workspace.remote()?.clone();
    let (analyst, log, project, cache) = (workspace.analyst.clone(), workspace.log().to_vec(), workspace.project.clone(), workspace.cache_dir());
    let sent = workspace.pending().len();
    Some(spawn(workspace.project.id, move || Done::Synced(sync(&analyst, &remote, &log, &project, cache.as_deref()), sent)))
}

pub fn poll_sync(mut workspace: ResMut<Workspace>, mut state: ResMut<SyncState>, time: Res<Time>, mut requests: MessageWriter<WorkspaceRequest>) {
    let state = &mut *state;
    let now = time.elapsed_secs_f64();
    if let Some((project, task)) = &mut state.dialog
        && let Some(path) = check_ready(task)
    {
        let project = *project;
        state.dialog = None;
        if let Some(path) = path {
            let path = path.with_extension(PROJECT_EXTENSION);
            let (endpoint, analyst) = (state.endpoint.clone(), workspace.analyst.clone());
            state.status = Some(format!("Opening into {}…", path.display()));
            state.job = Some(spawn(None, move || Done::Joined(join(&endpoint, &analyst, project, &path).map(|()| path))));
        }
    }
    if let Some(job) = state.job.as_mut().map(|m| m.get_mut().unwrap_or_else(|p| p.into_inner())) {
        let done = match job.receiver.try_recv() {
            Ok(done) => done,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Done::Listed(Err("the sync thread stopped".to_string())),
        };
        let current = job.project.is_none() || job.project == workspace.project.id;
        state.job = None;
        state.last = now;
        match done {
            Done::Shared(Ok(remote)) if current => {
                state.status = Some(format!("Shared on {}.", remote.endpoint));
                if let Err(error) = workspace.link(remote) {
                    state.status = Some(error);
                }
            }
            Done::Synced(Ok((synced, fetched)), sent) if current => match workspace.adopt(sent, synced) {
                Ok(adopted) => {
                    if !adopted.reload.is_empty() {
                        requests.write(WorkspaceRequest::Reload(adopted.reload.clone()));
                    }
                    let mut parts = vec![format!("{} sent", adopted.accepted), format!("{} received", adopted.remote_ops)];
                    if fetched > 0 {
                        parts.push(format!("{fetched} files fetched"));
                    }
                    if adopted.superseded > 0 {
                        parts.push(format!("{} superseded by later edits", adopted.superseded));
                    }
                    if adopted.conflicts > 0 {
                        parts.push(format!("{} new conflicts", adopted.conflicts));
                    }
                    state.status = Some(format!("Synced: {}.", parts.join(", ")));
                }
                Err(error) => state.status = Some(error),
            },
            Done::Listed(Ok(projects)) => {
                state.status = Some(format!("{} shared projects.", projects.len()));
                state.projects = projects;
            }
            Done::Joined(Ok(path)) => {
                state.status = Some(format!("Opened {}.", path.display()));
                requests.write(WorkspaceRequest::Open(path));
            }
            Done::Shared(Err(error)) | Done::Synced(Err(error), _) | Done::Listed(Err(error)) | Done::Joined(Err(error)) => {
                state.status = Some(error);
            }
            Done::Shared(Ok(_)) | Done::Synced(Ok(_), _) => state.status = Some("The project was replaced before the sync finished.".to_string()),
        }
        return;
    }
    if state.auto && workspace.remote().is_some() && now - state.last >= AUTO_S {
        state.job = start_sync(&workspace);
    }
}

/// `SetEvent on the event at …`.
fn summary(change: &Change, project: &Project) -> String {
    let debug = format!("{change:?}");
    let name = debug.split(['(', ' ', '{']).next().unwrap_or_default().to_string();
    match fields(change, project).first() {
        Some(field) => format!("{name} on {}", describe(field, project)),
        None => name,
    }
}

fn conflict_ui(ui: &mut egui::Ui, conflict: &Conflict, project: &Project) -> Option<bool> {
    let mut choice = None;
    ui.colored_label(WARN_COLOR, &conflict.reason);
    ui.label(format!("Mine ({}, {}): {}", conflict.mine.analyst, conflict.mine.at, summary(&conflict.mine.change, project)));
    for op in &conflict.theirs {
        ui.label(format!("Theirs ({}, {}): {}", op.analyst, op.at, summary(&op.change, project)));
    }
    ui.horizontal(|ui| {
        if ui.button("Keep theirs").on_hover_text("Drop my edit").clicked() {
            choice = Some(false);
        }
        if ui.button("Apply mine").on_hover_text("Make my edit again, on top of theirs").clicked() {
            choice = Some(true);
        }
    });
    choice
}

pub fn window(mut contexts: EguiContexts, mut workspace: ResMut<Workspace>, mut state: ResMut<SyncState>) -> Result {
    if !state.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?.clone();
    let state = &mut *state;
    let mut open = true;
    let mut settled: Option<(usize, bool)> = None;
    let mut unlink = false;
    theme::tool_window(&ctx, "Shared Project").open(&mut open).default_width(480.0).show(&ctx, |ui| {
        let busy = state.job.is_some() || state.dialog.is_some();
        ui.label(format!("Analyst: {}", workspace.analyst)).on_hover_text("Set with --analyst NAME or $SQUIGGLE_ANALYST");
        match workspace.remote().cloned() {
            Some(remote) => {
                ui.label(format!("Shared on {} as {}", remote.endpoint, remote.project));
                ui.label(format!("{} shared ops, {} pending", remote.confirmed, workspace.pending().len()));
                ui.horizontal(|ui| {
                    if ui.add_enabled(!busy, egui::Button::new("Sync now")).clicked() {
                        state.job = start_sync(&workspace);
                    }
                    ui.checkbox(&mut state.auto, format!("Every {AUTO_S} s"));
                    if ui.add_enabled(!busy, egui::Button::new("Stop sharing")).on_hover_text("Keep the project, stop syncing it").clicked() {
                        unlink = true;
                    }
                });
            }
            None => {
                ui.weak(format!("A {} server (squiggle-server). Edits sync per field; the later edit wins, except where locked or reviewed, which needs an analyst.", squiggle_io::sync::PROTOCOL));
                ui.horizontal(|ui| {
                    ui.label("Server");
                    ui.add_enabled(!busy, egui::TextEdit::singleline(&mut state.endpoint).desired_width(180.0));
                });
                ui.horizontal(|ui| {
                    let saved = workspace.cache_dir().is_some();
                    let button = ui.add_enabled(!busy && saved, theme::primary("Share this project"));
                    let button = if saved { button } else { button.on_disabled_hover_text("Save the project first") };
                    if button.clicked() {
                        let (endpoint, analyst) = (state.endpoint.clone(), workspace.analyst.clone());
                        let (log, project, cache) = (workspace.log().to_vec(), workspace.project.clone(), workspace.cache_dir());
                        state.status = Some("Sharing…".to_string());
                        state.job = Some(spawn(workspace.project.id, move || Done::Shared(share(&endpoint, &analyst, &log, &project, cache.as_deref()))));
                    }
                    if ui.add_enabled(!busy, egui::Button::new("List shared projects")).clicked() {
                        let (endpoint, analyst) = (state.endpoint.clone(), workspace.analyst.clone());
                        state.job = Some(spawn(None, move || Done::Listed(list(&endpoint, &analyst))));
                    }
                });
                for info in state.projects.clone() {
                    ui.horizontal(|ui| {
                        let mine = workspace.project.id == Some(info.id);
                        if ui.add_enabled(!busy && !mine, egui::Button::new("Open…")).clicked() {
                            let name = format!("{}.{PROJECT_EXTENSION}", info.name);
                            let task = IoTaskPool::get().spawn(async move {
                                let file = rfd::AsyncFileDialog::new()
                                    .add_filter("Squiggle project", &[PROJECT_EXTENSION])
                                    .set_title("Save the shared project as")
                                    .set_file_name(name)
                                    .save_file()
                                    .await?;
                                Some(file.path().to_path_buf())
                            });
                            state.dialog = Some((info.id, task));
                        }
                        ui.label(format!("{} — {} ops by {}", info.name, info.ops, info.analysts.join(", ")));
                    });
                }
            }
        }
        if busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.weak("Working…");
            });
        }
        if let Some(status) = &state.status {
            ui.weak(status);
        }
        if !workspace.conflicts.is_empty() {
            ui.separator();
            ui.heading(format!("Conflicts ({})", workspace.conflicts.len()));
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                for (i, conflict) in workspace.conflicts.iter().enumerate() {
                    if let Some(apply) = conflict_ui(ui, conflict, &workspace.project) {
                        settled = Some((i, apply));
                    }
                    ui.separator();
                }
            });
        }
    });
    state.open = open;
    if unlink && let Err(error) = workspace.unlink() {
        workspace.error = Some(error);
    }
    if let Some((i, apply)) = settled {
        settle(&mut workspace, i, apply);
    }
    Ok(())
}

/// Settles conflict `i`: drops my edit, or makes it again on the current state.
pub fn settle(workspace: &mut Workspace, i: usize, apply: bool) {
    if i >= workspace.conflicts.len() {
        return;
    }
    if apply {
        let change = workspace.conflicts[i].mine.change.clone();
        if !workspace.commit_or_report(change) {
            return;
        }
    }
    workspace.conflicts.remove(i);
    if let Err(error) = workspace.save_conflicts() {
        workspace.error = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use squiggle_core::{Event, EventStatus, LabelId, TraceSamples};
    use squiggle_io::sync::{Hub, serve};

    use super::*;

    fn server() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let hub = Arc::new(Hub::open(dir.path()).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || serve(listener, hub, Arc::new(AtomicBool::new(false))));
        (dir, endpoint)
    }

    fn sync_now(workspace: &mut Workspace) -> crate::workspace::Adopted {
        let remote = workspace.remote().unwrap().clone();
        let sent = workspace.pending().len();
        let (synced, _) = sync(&workspace.analyst.clone(), &remote, workspace.log(), &workspace.project, workspace.cache_dir().as_deref()).unwrap();
        workspace.adopt(sent, synced).unwrap()
    }

    fn reviewed(event: &Event, label: &str) -> Change {
        Change::SetEvent(Event { label: Some(LabelId::new(label)), status: EventStatus::Reviewed, ..event.clone() })
    }

    #[test]
    fn two_analysts_merge_offline_edits_and_see_conflicts() {
        let (_server, endpoint) = server();
        let dir = tempfile::tempdir().unwrap();

        // Ana records a trace and an automatic event, and shares them.
        let files = dir.path().join("ana-files");
        std::fs::create_dir_all(&files).unwrap();
        let csv = files.join("s1.csv");
        let values: Vec<f32> = (0..500).map(|i| (i as f32 * 0.1).sin()).collect();
        std::fs::write(&csv, format!("x\n{}\n", values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n"))).unwrap();
        let read = squiggle_io::read_file(&csv).unwrap();
        let mut ana = Workspace::new(Analyst("ana".into()));
        let (sensor, trace) = (Id::generate(), Id::generate());
        let start = squiggle_core::Timestamp::new(1_000, 0.0);
        ana.commit(Change::AddSensor { id: sensor, code: "S1".into() }).unwrap();
        ana.commit(Change::AssignPhenomenology { sensor, phenomenology: squiggle_core::PhenomenologyId::new("seismic") }).unwrap();
        ana.commit(Change::AddTrace(squiggle_core::Trace {
            id: trace,
            sensor,
            start,
            sample_rate: 100.0,
            len: values.len(),
            kind: Default::default(),
            source: read.source(0),
            processing: Default::default(),
            alignment: None,
        }))
        .unwrap();
        let event = Event {
            id: Id::generate(),
            start: start.add_seconds(1.0),
            end: start.add_seconds(2.0),
            members: vec![squiggle_core::Member::new(trace)],
            origin: None,
            status: EventStatus::Auto,
            provenance: squiggle_core::sensor::Provenance::Auto { stage: "detect".into(), config_hash: "x".into() },
            label: None,
            predictions: Vec::new(),
        };
        ana.commit(Change::AddEvent(event.clone())).unwrap();
        let a_path = dir.path().join("ana.squiggle");
        ana.save_as(&a_path).unwrap();
        let remote = share(&endpoint, &ana.analyst, ana.log(), &ana.project, ana.cache_dir().as_deref()).unwrap();
        ana.link(remote).unwrap();
        std::fs::remove_dir_all(&files).unwrap();

        // Bo opens it on "another machine": the trace file comes from the server.
        let b_path = dir.path().join("bo").join("bo.squiggle");
        std::fs::create_dir_all(b_path.parent().unwrap()).unwrap();
        let bo_analyst = Analyst("bo".into());
        let id = ana.project.id.unwrap();
        assert_eq!(list(&endpoint, &bo_analyst).unwrap()[0].id, id);
        join(&endpoint, &bo_analyst, id, &b_path).unwrap();
        let mut bo = Workspace::open(&b_path, bo_analyst).unwrap();
        assert_eq!(bo.project, ana.project);
        let TraceSamples::Real(loaded) = squiggle_io::sync::load_trace_from(&bo.project.traces()[0], bo.cache_dir().as_deref()).unwrap() else {
            panic!()
        };
        assert_eq!(loaded.iter().collect::<Vec<_>>(), values);

        // Both edit offline: different fields, and the same reviewed event.
        ana.commit(Change::RenameProject { name: "Swarm review".into() }).unwrap();
        ana.commit(reviewed(&event, "earthquake")).unwrap();
        let clock = squiggle_core::sensor::ClockCorrection { offset_s: 0.25, drift_ppm: 0.0 };
        bo.commit(Change::SetClockCorrection { sensor, clock }).unwrap();
        bo.commit(reviewed(&event, "explosion")).unwrap();

        let a = sync_now(&mut ana);
        assert_eq!((a.accepted, a.conflicts), (2, 0));
        let b = sync_now(&mut bo);
        assert_eq!((b.accepted, b.conflicts, b.remote_ops), (1, 1, 2));
        let a = sync_now(&mut ana);
        assert_eq!(a.remote_ops, 1);

        // Merged identically; Ana's label stands, Bo's is held back.
        assert_eq!(ana.project, bo.project);
        assert_eq!(ana.log(), bo.log());
        assert_eq!(bo.project.name, "Swarm review");
        assert_eq!(bo.project.sensor(sensor).unwrap().clock, clock);
        assert_eq!(bo.project.event(event.id).unwrap().label, Some(LabelId::new("earthquake")));
        assert_eq!(bo.conflicts.len(), 1);
        assert!(bo.conflicts[0].reason.contains("ana"), "{}", bo.conflicts[0].reason);

        // The conflict survives a reopen, then Bo settles it by applying his label.
        drop(bo);
        let mut bo = Workspace::open(&b_path, Analyst("bo".into())).unwrap();
        assert_eq!(bo.conflicts.len(), 1);
        settle(&mut bo, 0, true);
        assert!(bo.conflicts.is_empty());
        assert_eq!(sync_now(&mut bo).accepted, 1);
        sync_now(&mut ana);
        assert_eq!(ana.project.event(event.id).unwrap().label, Some(LabelId::new("explosion")));
        assert_eq!(ana.project, bo.project);
    }

    #[test]
    fn edits_made_during_a_sync_are_kept() {
        let (_server, endpoint) = server();
        let dir = tempfile::tempdir().unwrap();
        let mut ana = Workspace::new(Analyst("ana".into()));
        ana.save_as(&dir.path().join("a.squiggle")).unwrap();
        let remote = share(&endpoint, &ana.analyst, ana.log(), &ana.project, None).unwrap();
        ana.link(remote).unwrap();
        ana.commit(Change::RenameProject { name: "one".into() }).unwrap();
        // The sync is computed, then another edit lands before it is adopted.
        let remote = ana.remote().unwrap().clone();
        let sent = ana.pending().len();
        let (synced, _) = sync(&ana.analyst.clone(), &remote, ana.log(), &ana.project, None).unwrap();
        ana.commit(Change::RenameProject { name: "two".into() }).unwrap();
        ana.adopt(sent, synced).unwrap();
        assert_eq!(ana.project.name, "two");
        assert_eq!(ana.pending().len(), 1);
        sync_now(&mut ana);
        assert!(ana.pending().is_empty());
        let reopened = Workspace::open(&dir.path().join("a.squiggle"), Analyst("ana".into())).unwrap();
        assert_eq!(reopened.project.name, "two");
        assert_eq!(reopened.remote(), ana.remote());
    }
}
