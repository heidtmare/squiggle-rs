//! The open project: its op log, its store, loaded samples, and the squiggles
//! that plot its traces.
//!
//! File dialogs, file reads and sample loads run as tasks; systems here poll
//! them and fold results into [`Workspace`] and [`ImportQueue`].

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task, futures::check_ready};
use bevy::window::FileDragAndDrop;
use squiggle_core::{
    Analyst, Change, ChunkedSamples, Id, Op, PhenomenologyId, Point, Project, Rgba, Squiggle, Style, Suggestion,
    Timestamp, Trace, TraceSource,
};
use squiggle_io::{FileRead, PROJECT_EXTENSION, ProjectStore, SqliteStore};

use crate::plot::{AddSquiggle, FitToData, MeshOutOfDate, Plot, RemoveSquiggle, SquiggleModel};

/// Colour of traces whose sensor has no phenomenology yet.
pub const UNASSIGNED_COLOR: Rgba = Rgba::new(0.6, 0.6, 0.6);
const TRACE_LINE_WIDTH: f32 = 1.5;

pub struct WorkspacePlugin;

impl Plugin for WorkspacePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Workspace::new(Analyst::local()))
            .init_resource::<ImportQueue>()
            .init_resource::<Tasks>()
            .add_message::<WorkspaceRequest>()
            .add_systems(
                Update,
                (handle_drops, handle_requests, poll_dialog, poll_reads, poll_loads, sync_trace_squiggles).chain(),
            );
    }
}

/// What the UI can ask of the workspace.
#[derive(Message, Clone, Debug, PartialEq)]
pub enum WorkspaceRequest {
    New,
    OpenDialog,
    Open(PathBuf),
    SaveAsDialog,
    SaveAs(PathBuf),
    ImportDialog,
    Import(Vec<PathBuf>),
    /// Re-issues a request that was held back by the unsaved-project prompt.
    Discard(Box<WorkspaceRequest>),
}

impl WorkspaceRequest {
    /// Requests that replace the current project.
    fn replaces_project(&self) -> bool {
        matches!(self, Self::New | Self::OpenDialog | Self::Open(_))
    }
}

pub enum SampleState {
    Loading,
    Loaded(ChunkedSamples),
    Failed(String),
}

#[derive(Resource)]
pub struct Workspace {
    pub project: Project,
    pub analyst: Analyst,
    pub path: Option<PathBuf>,
    pub samples: HashMap<Id, SampleState>,
    /// Shown until dismissed.
    pub error: Option<String>,
    /// Held while the analyst decides whether to discard an unsaved project.
    pub confirm_discard: Option<WorkspaceRequest>,
    log: Vec<Op>,
    /// Ops a fresh project starts with; more than this and an unsaved project has edits.
    created_ops: usize,
    store: Option<Mutex<SqliteStore>>,
    /// Bumped on every change the plot must reflect.
    revision: u64,
}

impl Workspace {
    pub fn new(analyst: Analyst) -> Self {
        let (project, log) = Project::create("Untitled", &analyst);
        Self {
            project,
            analyst,
            path: None,
            samples: HashMap::new(),
            error: None,
            confirm_discard: None,
            created_ops: log.len(),
            log,
            store: None,
            revision: 1,
        }
    }

    /// Commits a change, persisting it when the project has a file.
    pub fn commit(&mut self, change: Change) -> Result<(), String> {
        let op = self.project.commit(&self.analyst, change).map_err(|e| e.to_string())?;
        if let Some(store) = &self.store {
            let mut store = store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            store.append(std::slice::from_ref(&op)).map_err(|e| format!("Not saved: {e}"))?;
        }
        self.log.push(op);
        self.revision += 1;
        Ok(())
    }

    /// Commits, routing failure to the error window.
    pub fn commit_or_report(&mut self, change: Change) -> bool {
        match self.commit(change) {
            Ok(()) => true,
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }

    pub fn is_unsaved(&self) -> bool {
        self.path.is_none() && self.log.len() > self.created_ops
    }

    pub fn title(&self) -> String {
        match &self.path {
            Some(path) => {
                let file = path.file_name().map_or_else(|| path.display().to_string(), |f| f.to_string_lossy().into_owned());
                format!("{} — {file}", self.project.name)
            }
            None if self.is_unsaved() => format!("{} (unsaved)", self.project.name),
            None => self.project.name.clone(),
        }
    }

    fn save_as(&mut self, path: &Path) -> Result<(), String> {
        let mut store = SqliteStore::create(path).map_err(|e| e.to_string())?;
        store.append(&self.log).map_err(|e| e.to_string())?;
        self.store = Some(Mutex::new(store));
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    fn open(path: &Path, analyst: Analyst) -> Result<Self, String> {
        let store = SqliteStore::open(path).map_err(|e| e.to_string())?;
        let log = store.load().map_err(|e| e.to_string())?;
        let project = Project::replay(&log).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            project,
            analyst,
            path: Some(path.to_path_buf()),
            samples: HashMap::new(),
            error: None,
            confirm_discard: None,
            created_ops: log.len(),
            log,
            store: Some(Mutex::new(store)),
            revision: 1,
        })
    }

    /// Replaces the project, keeping the analyst and bumping the revision so stale squiggles go.
    fn replace(&mut self, mut next: Self) {
        next.revision = self.revision + 1;
        *self = next;
    }
}

/// One channel awaiting the analyst's confirmation in the import dialog.
pub struct ImportRow {
    pub include: bool,
    pub file: String,
    pub code: String,
    pub source: TraceSource,
    pub samples: Vec<f32>,
    /// Editable; parsed on confirm.
    pub start: String,
    /// Where the start text came from.
    pub start_origin: &'static str,
    pub rate: String,
    pub phenomenology: Option<PhenomenologyId>,
    pub suggestion: Option<Suggestion>,
    /// Validation failure from the last confirm.
    pub problem: Option<String>,
}

#[derive(Resource, Default)]
pub struct ImportQueue {
    pub rows: Vec<ImportRow>,
    /// Files that could not be read.
    pub failures: Vec<String>,
}

impl ImportQueue {
    pub fn is_open(&self) -> bool {
        !self.rows.is_empty() || !self.failures.is_empty()
    }

    fn push_file(&mut self, read: FileRead, modified: Option<Timestamp>, project: &Project) {
        let file = read.path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        for recording in &read.recordings {
            // A known sensor keeps its project assignment; otherwise the suggestion is pre-selected.
            let known = project.sensor_by_code(&recording.code).and_then(|s| s.assignment.as_ref());
            let phenomenology = known
                .map(|a| a.phenomenology.clone())
                .or_else(|| recording.suggestion.as_ref().map(|s| s.phenomenology.clone()))
                .filter(|id| project.phenomenology(id).is_some());
            let (start, start_origin) = match (recording.start, modified) {
                (Some(start), _) => (start.to_string(), "file"),
                (None, Some(modified)) => (modified.to_string(), "file modified time"),
                (None, None) => (String::new(), "required"),
            };
            self.rows.push(ImportRow {
                include: true,
                file: file.clone(),
                code: recording.code.clone(),
                source: read.source(recording.channel),
                samples: recording.samples.clone(),
                start,
                start_origin,
                rate: recording.sample_rate.map_or_else(String::new, |r| r.to_string()),
                phenomenology,
                suggestion: recording.suggestion.clone(),
                problem: None,
            });
        }
    }

    /// Validates every included row, then commits sensors, assignments and traces.
    /// On any problem nothing is committed and the problems are shown per row.
    pub fn confirm(&mut self, workspace: &mut Workspace) -> bool {
        let mut parsed = Vec::new();
        let mut chosen: HashMap<String, Option<PhenomenologyId>> = HashMap::new();
        for row in self.rows.iter_mut().filter(|r| r.include) {
            row.problem = None;
            let start = row.start.parse::<Timestamp>();
            let rate = row.rate.trim().parse::<f64>().ok().filter(|r| r.is_finite() && *r > 0.0);
            let code = row.code.trim().to_string();
            let earlier = chosen.insert(code.clone(), row.phenomenology.clone());
            row.problem = if code.is_empty() {
                Some("sensor code required".to_string())
            } else if start.is_err() {
                Some("start must be ISO 8601 UTC".to_string())
            } else if rate.is_none() {
                Some("sample rate must be positive".to_string())
            } else if earlier.is_some_and(|e| e != row.phenomenology) {
                Some("same sensor code with a different phenomenology".to_string())
            } else {
                None
            };
            if row.problem.is_none() {
                parsed.push((code, start.expect("checked"), rate.expect("checked")));
            }
        }
        if self.rows.iter().any(|r| r.problem.is_some()) {
            return false;
        }

        let rows = std::mem::take(&mut self.rows).into_iter().filter(|r| r.include);
        let mut earliest: Option<Timestamp> = None;
        for (row, (code, start, sample_rate)) in rows.zip(parsed) {
            let sensor = match workspace.project.sensor_by_code(&code) {
                Some(sensor) => sensor.id,
                None => {
                    let id = Id::generate();
                    if !workspace.commit_or_report(Change::AddSensor { id, code }) {
                        break;
                    }
                    id
                }
            };
            let unassigned = workspace.project.sensor(sensor).is_some_and(|s| s.assignment.is_none());
            if let (true, Some(phenomenology)) = (unassigned, row.phenomenology) {
                workspace.commit_or_report(Change::AssignPhenomenology { sensor, phenomenology });
            }

            let trace = Trace { id: Id::generate(), sensor, start, sample_rate, len: row.samples.len(), source: row.source };
            let id = trace.id;
            if !workspace.commit_or_report(Change::AddTrace(trace)) {
                break;
            }
            workspace.samples.insert(id, SampleState::Loaded(ChunkedSamples::from_vec(row.samples)));
            earliest = Some(earliest.map_or(start, |e| e.min(start)));
        }

        if let (None, Some(reference)) = (workspace.project.reference, earliest) {
            workspace.commit_or_report(Change::SetReference { reference });
        }
        self.failures.clear();
        true
    }

    pub fn cancel(&mut self) {
        self.rows.clear();
        self.failures.clear();
    }
}

enum DialogOutcome {
    Import(Vec<PathBuf>),
    Open(PathBuf),
    SaveAs(PathBuf),
}

type FileOutcome = Result<(FileRead, Option<Timestamp>), String>;
type SampleLoad = Vec<(Id, Result<Vec<f32>, String>)>;

#[derive(Resource, Default)]
struct Tasks {
    dialog: Option<Task<Option<DialogOutcome>>>,
    reads: Vec<Task<Vec<FileOutcome>>>,
    /// Tagged with the project they were started for; results for a replaced project are dropped.
    loads: Vec<(Option<Id>, Task<SampleLoad>)>,
}

fn project_dialog() -> rfd::AsyncFileDialog {
    rfd::AsyncFileDialog::new().add_filter("Squiggle project", &[PROJECT_EXTENSION])
}

fn read_files(paths: Vec<PathBuf>) -> Task<Vec<FileOutcome>> {
    AsyncComputeTaskPool::get().spawn(async move {
        paths
            .into_iter()
            .map(|path| {
                let modified = std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| Timestamp::new(d.as_secs() as i64, f64::from(d.subsec_nanos()) * 1e-9));
                squiggle_io::read_file(&path).map(|read| (read, modified)).map_err(|e| e.to_string())
            })
            .collect()
    })
}

fn load_traces(workspace: &mut Workspace) -> Task<SampleLoad> {
    let sources: Vec<(Id, TraceSource)> =
        workspace.project.traces().iter().map(|t| (t.id, t.source.clone())).collect();
    for (id, _) in &sources {
        workspace.samples.insert(*id, SampleState::Loading);
    }
    AsyncComputeTaskPool::get().spawn(async move {
        sources
            .into_iter()
            .map(|(id, source)| (id, squiggle_io::load_samples(&source).map_err(|e| e.to_string())))
            .collect()
    })
}

fn handle_drops(mut drops: MessageReader<FileDragAndDrop>, mut requests: MessageWriter<WorkspaceRequest>) {
    let mut imports = Vec::new();
    for drop in drops.read() {
        if let FileDragAndDrop::DroppedFile { path_buf, .. } = drop {
            if path_buf.extension().is_some_and(|e| e == PROJECT_EXTENSION) {
                requests.write(WorkspaceRequest::Open(path_buf.clone()));
            } else {
                imports.push(path_buf.clone());
            }
        }
    }
    if !imports.is_empty() {
        requests.write(WorkspaceRequest::Import(imports));
    }
}

fn handle_requests(
    mut requests: MessageReader<WorkspaceRequest>,
    mut workspace: ResMut<Workspace>,
    mut queue: ResMut<ImportQueue>,
    mut tasks: ResMut<Tasks>,
) {
    for request in requests.read() {
        let request = match request {
            WorkspaceRequest::Discard(inner) => inner.as_ref(),
            request if request.replaces_project() && workspace.is_unsaved() => {
                workspace.confirm_discard = Some(request.clone());
                continue;
            }
            request => request,
        };

        match request {
            WorkspaceRequest::New => {
                let next = Workspace::new(workspace.analyst.clone());
                workspace.replace(next);
                queue.cancel();
            }
            WorkspaceRequest::OpenDialog => {
                tasks.dialog = Some(IoTaskPool::get().spawn(async {
                    let file = project_dialog().set_title("Open project").pick_file().await?;
                    Some(DialogOutcome::Open(file.path().to_path_buf()))
                }));
            }
            WorkspaceRequest::Open(path) => match Workspace::open(path, workspace.analyst.clone()) {
                Ok(next) => {
                    workspace.replace(next);
                    queue.cancel();
                    let project = workspace.project.id;
                    let task = load_traces(&mut workspace);
                    tasks.loads.push((project, task));
                }
                Err(error) => workspace.error = Some(error),
            },
            WorkspaceRequest::SaveAsDialog => {
                let name = format!("{}.{PROJECT_EXTENSION}", workspace.project.name);
                tasks.dialog = Some(IoTaskPool::get().spawn(async move {
                    let file = project_dialog().set_title("Save project as").set_file_name(name).save_file().await?;
                    Some(DialogOutcome::SaveAs(file.path().to_path_buf()))
                }));
            }
            WorkspaceRequest::SaveAs(path) => {
                let path = path.with_extension(PROJECT_EXTENSION);
                if let Err(error) = workspace.save_as(&path) {
                    workspace.error = Some(error);
                }
            }
            WorkspaceRequest::ImportDialog => {
                tasks.dialog = Some(IoTaskPool::get().spawn(async {
                    let files = rfd::AsyncFileDialog::new()
                        .set_title("Import waveforms")
                        .add_filter("Waveforms", &["wav", "csv"])
                        .pick_files()
                        .await?;
                    Some(DialogOutcome::Import(files.iter().map(|f| f.path().to_path_buf()).collect()))
                }));
            }
            WorkspaceRequest::Import(paths) => tasks.reads.push(read_files(paths.clone())),
            WorkspaceRequest::Discard(_) => {}
        }
    }
}

fn poll_dialog(mut tasks: ResMut<Tasks>, mut requests: MessageWriter<WorkspaceRequest>) {
    let Some(task) = &mut tasks.dialog else {
        return;
    };
    let Some(outcome) = check_ready(task) else {
        return;
    };
    tasks.dialog = None;
    match outcome {
        Some(DialogOutcome::Import(paths)) => requests.write(WorkspaceRequest::Import(paths)),
        Some(DialogOutcome::Open(path)) => requests.write(WorkspaceRequest::Open(path)),
        Some(DialogOutcome::SaveAs(path)) => requests.write(WorkspaceRequest::SaveAs(path)),
        None => return,
    };
}

fn poll_reads(mut tasks: ResMut<Tasks>, mut queue: ResMut<ImportQueue>, workspace: Res<Workspace>) {
    let mut finished = Vec::new();
    tasks.reads.retain_mut(|task| match check_ready(task) {
        Some(outcomes) => {
            finished.extend(outcomes);
            false
        }
        None => true,
    });
    for outcome in finished {
        match outcome {
            Ok((read, modified)) => queue.push_file(read, modified, &workspace.project),
            Err(error) => queue.failures.push(error),
        }
    }
}

fn poll_loads(mut tasks: ResMut<Tasks>, mut workspace: ResMut<Workspace>) {
    let mut finished = Vec::new();
    tasks.loads.retain_mut(|(project, task)| match check_ready(task) {
        Some(results) => {
            finished.push((*project, results));
            false
        }
        None => true,
    });
    for (project, results) in finished {
        if project != workspace.project.id {
            continue;
        }
        for (id, result) in results {
            if workspace.project.trace(id).is_none() {
                continue;
            }
            let state = match result {
                Ok(samples) => SampleState::Loaded(ChunkedSamples::from_vec(samples)),
                Err(error) => SampleState::Failed(error),
            };
            workspace.samples.insert(id, state);
        }
        workspace.revision += 1;
    }
}

/// The squiggle that plots a trace: x in seconds from the project reference.
fn trace_squiggle(project: &Project, trace: &Trace, samples: &ChunkedSamples) -> Squiggle {
    let reference = project.reference.unwrap_or(trace.start);
    let offset = project.corrected_start(trace).seconds_since(reference);
    let interval = 1.0 / trace.sample_rate;
    // NaN marks a missing sample; gaps are drawn joined until tracks land.
    let points = samples
        .iter()
        .enumerate()
        .filter(|(_, y)| y.is_finite())
        .map(|(i, y)| Point::new(offset + i as f64 * interval, f64::from(y)));

    let code = project.sensor(trace.sensor).map_or("?", |s| s.code.as_str());
    let color = project.trace_phenomenology(trace).map_or(UNASSIGNED_COLOR, |p| p.color);
    let mut squiggle =
        Squiggle::with_points(code.to_string(), points).with_style(Style::new(color, TRACE_LINE_WIDTH));
    squiggle.trace = Some(trace.id);
    squiggle
}

/// Reconciles trace squiggles with the workspace whenever it changes.
#[allow(clippy::too_many_arguments)]
fn sync_trace_squiggles(
    mut commands: Commands,
    workspace: Res<Workspace>,
    mut plot: ResMut<Plot>,
    mut seen: Local<u64>,
    mut squiggles: Query<(Entity, &mut SquiggleModel)>,
    mut add: MessageWriter<AddSquiggle>,
    mut remove: MessageWriter<RemoveSquiggle>,
    mut fit: MessageWriter<FitToData>,
) {
    if *seen == workspace.revision {
        return;
    }
    *seen = workspace.revision;
    plot.name = workspace.title();

    let project = &workspace.project;
    let mut plotted = HashSet::new();
    for (entity, mut model) in &mut squiggles {
        let Some(id) = model.trace else {
            continue;
        };
        let wanted = match (project.trace(id), workspace.samples.get(&id)) {
            (Some(trace), Some(SampleState::Loaded(samples))) => trace_squiggle(project, trace, samples),
            _ => {
                remove.write(RemoveSquiggle(entity));
                continue;
            }
        };
        plotted.insert(id);
        if model.name != wanted.name || model.style != wanted.style {
            model.name = wanted.name.clone();
            model.style = wanted.style;
        }
        if model.points().first() != wanted.points().first() {
            model.clear();
            model.extend(wanted.points().iter().copied());
            commands.entity(entity).insert(MeshOutOfDate);
        }
    }

    let mut added = false;
    for trace in project.traces() {
        if let (false, Some(SampleState::Loaded(samples))) = (plotted.contains(&trace.id), workspace.samples.get(&trace.id)) {
            add.write(AddSquiggle(trace_squiggle(project, trace, samples)));
            added = true;
        }
    }
    if added {
        fit.write(FitToData);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(code: &str, start: &str, rate: &str, phenomenology: Option<&str>) -> ImportRow {
        ImportRow {
            include: true,
            file: "f.csv".to_string(),
            code: code.to_string(),
            source: TraceSource::File {
                path: "f.csv".to_string(),
                format: "csv".to_string(),
                channel: 0,
                content_hash: "00".to_string(),
            },
            samples: vec![0.0; 4],
            start: start.to_string(),
            start_origin: "file",
            rate: rate.to_string(),
            phenomenology: phenomenology.map(PhenomenologyId::new),
            suggestion: None,
            problem: None,
        }
    }

    fn workspace() -> Workspace {
        Workspace::new(Analyst("ana".to_string()))
    }

    #[test]
    fn confirming_commits_sensors_assignments_traces_and_reference() {
        let mut workspace = workspace();
        let mut queue = ImportQueue {
            rows: vec![
                row("BHZ", "2026-10-11T00:00:01Z", "100", Some("seismic")),
                row("mic", "2026-10-11T00:00:00Z", "48000", None),
            ],
            failures: vec![],
        };
        assert!(queue.confirm(&mut workspace));
        assert!(queue.rows.is_empty());

        let project = &workspace.project;
        assert_eq!(project.traces().len(), 2);
        assert_eq!(project.reference, Some("2026-10-11T00:00:00Z".parse().unwrap()));
        let bhz = project.sensor_by_code("BHZ").unwrap();
        assert_eq!(bhz.assignment.as_ref().map(|a| a.phenomenology.as_str()), Some("seismic"));
        assert!(project.sensor_by_code("mic").unwrap().assignment.is_none());
        assert!(workspace.is_unsaved());
    }

    #[test]
    fn invalid_rows_block_the_whole_import() {
        let mut workspace = workspace();
        let mut queue = ImportQueue {
            rows: vec![row("a", "2026-10-11T00:00:00Z", "100", None), row("b", "yesterday", "0", None)],
            failures: vec![],
        };
        assert!(!queue.confirm(&mut workspace));
        assert!(workspace.project.traces().is_empty());
        assert!(queue.rows[1].problem.is_some());
        assert!(queue.rows[0].problem.is_none());
    }

    #[test]
    fn conflicting_phenomenologies_for_one_code_are_rejected() {
        let mut workspace = workspace();
        let mut queue = ImportQueue {
            rows: vec![
                row("s", "2026-10-11T00:00:00Z", "100", Some("seismic")),
                row("s", "2026-10-11T00:00:00Z", "100", Some("acoustic")),
            ],
            failures: vec![],
        };
        assert!(!queue.confirm(&mut workspace));
    }

    #[test]
    fn a_second_trace_for_a_known_sensor_reuses_it() {
        let mut workspace = workspace();
        let mut queue = ImportQueue { rows: vec![row("s", "2026-10-11T00:00:00Z", "100", Some("seismic"))], failures: vec![] };
        assert!(queue.confirm(&mut workspace));
        queue.rows.push(row("s", "2026-10-11T00:01:00Z", "100", Some("seismic")));
        assert!(queue.confirm(&mut workspace));
        assert_eq!(workspace.project.sensors().len(), 1);
        assert_eq!(workspace.project.traces().len(), 2);
    }

    #[test]
    fn saving_then_opening_restores_the_project() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.squiggle");
        let mut workspace = workspace();
        let mut queue = ImportQueue { rows: vec![row("s", "2026-10-11T00:00:00Z", "100", Some("rf"))], failures: vec![] };
        queue.confirm(&mut workspace);
        workspace.save_as(&path).unwrap();
        assert!(!workspace.is_unsaved());

        // Edits after saving go straight to the file.
        workspace.commit(Change::RenameProject { name: "renamed".to_string() }).unwrap();
        let reopened = Workspace::open(&path, workspace.analyst.clone()).unwrap();
        assert_eq!(reopened.project, workspace.project);
    }
}
