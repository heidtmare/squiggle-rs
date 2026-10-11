//! The open project: its op log, its store and its loaded samples.
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
    Analyst, Change, Id, Op, PhenomenologyId, Project, RawLayout, Rgba, SampleBuf, Suggestion, Timestamp, Trace,
    TraceSamples, TraceSource,
};
use squiggle_io::{FileRead, PROJECT_EXTENSION, ProjectStore, ReadError, SqliteStore};

use crate::plot::Plot;
use crate::synthetic;

/// Colour of traces whose sensor has no phenomenology yet.
pub const UNASSIGNED_COLOR: Rgba = Rgba::new(0.6, 0.6, 0.6);

pub struct WorkspacePlugin;

impl Plugin for WorkspacePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Workspace::new(Analyst::local()))
            .init_resource::<ImportQueue>()
            .init_resource::<Tasks>()
            .add_message::<WorkspaceRequest>()
            .add_systems(
                Update,
                (handle_drops, handle_requests, poll_dialog, poll_reads, poll_loads, sync_title).chain(),
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
    /// Decodes headerless files with an analyst-supplied layout.
    ImportRaw(Vec<PathBuf>, RawLayout),
    /// Replaces the project with the synthetic alignment event.
    Synthetic,
    /// Re-issues a request that was held back by the unsaved-project prompt.
    Discard(Box<WorkspaceRequest>),
}

impl WorkspaceRequest {
    /// Requests that replace the current project.
    fn replaces_project(&self) -> bool {
        matches!(self, Self::New | Self::OpenDialog | Self::Open(_) | Self::Synthetic)
    }
}

pub enum SampleState {
    Loading,
    Loaded(TraceSamples),
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
    /// Known event delay per trace, seconds; synthetic data only.
    pub truth: HashMap<Id, f64>,
    log: Vec<Op>,
    /// Ops a fresh project starts with; more than this and an unsaved project has edits.
    created_ops: usize,
    store: Option<Mutex<SqliteStore>>,
    /// Bumped on every change the plot must reflect.
    revision: u64,
    undo: Vec<Command>,
    redo: Vec<Command>,
}

/// One undoable analyst action. Undo and redo append ops; the log is never rewritten.
#[derive(Clone, Debug)]
struct Command {
    label: String,
    forward: Vec<Change>,
    inverse: Vec<Change>,
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
            truth: HashMap::new(),
            created_ops: log.len(),
            log,
            store: None,
            revision: 1,
            undo: Vec::new(),
            redo: Vec::new(),
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

    /// Commits `changes` as one undoable command. All are checked first, so
    /// either every change lands or none does.
    pub fn execute(&mut self, label: impl Into<String>, changes: Vec<Change>) -> Result<(), String> {
        let mut trial = self.project.clone();
        let mut inverse = Vec::with_capacity(changes.len());
        for change in &changes {
            inverse.push(trial.inverse(change).ok_or("this change cannot be undone")?);
            trial.commit(&self.analyst, change.clone()).map_err(|e| e.to_string())?;
        }
        for change in &changes {
            self.commit(change.clone())?;
        }
        inverse.reverse();
        self.undo.push(Command { label: label.into(), forward: changes, inverse });
        self.redo.clear();
        Ok(())
    }

    /// [`Self::execute`], routing failure to the error window.
    pub fn execute_or_report(&mut self, label: impl Into<String>, changes: Vec<Change>) -> bool {
        match self.execute(label, changes) {
            Ok(()) => true,
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|c| c.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|c| c.label.as_str())
    }

    pub fn undo(&mut self) {
        if let Some(command) = self.undo.pop() {
            match self.replay(&command.inverse) {
                Ok(()) => self.redo.push(command),
                Err(error) => self.error = Some(format!("Undo {}: {error}", command.label)),
            }
        }
    }

    pub fn redo(&mut self) {
        if let Some(command) = self.redo.pop() {
            match self.replay(&command.forward) {
                Ok(()) => self.undo.push(command),
                Err(error) => self.error = Some(format!("Redo {}: {error}", command.label)),
            }
        }
    }

    fn replay(&mut self, changes: &[Change]) -> Result<(), String> {
        let mut trial = self.project.clone();
        for change in changes {
            trial.commit(&self.analyst, change.clone()).map_err(|e| e.to_string())?;
        }
        changes.iter().try_for_each(|change| self.commit(change.clone()))
    }

    pub fn is_unsaved(&self) -> bool {
        self.path.is_none() && self.log.len() > self.created_ops
    }

    /// Bumped on every change the plot must reflect.
    pub fn revision(&self) -> u64 {
        self.revision
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
            truth: HashMap::new(),
            created_ops: log.len(),
            log,
            store: Some(Mutex::new(store)),
            revision: 1,
            undo: Vec::new(),
            redo: Vec::new(),
        })
    }

    /// A project holding the synthetic event, with its true delays.
    fn synthetic(analyst: Analyst) -> Result<Self, String> {
        let mut next = Self::new(analyst);
        next.commit(Change::RenameProject { name: "Synthetic event".to_string() })?;
        let paths = synthetic::write(&synthetic::directory())?;
        let mut queue = ImportQueue::default();
        for (path, spec) in paths.iter().zip(&synthetic::SPECS) {
            let read = squiggle_io::read_file(path).map_err(|e| e.to_string())?;
            queue.push_file(read, None, &next.project);
            if let Some(row) = queue.rows.last_mut() {
                row.phenomenology = Some(PhenomenologyId::new(spec.phenomenology));
            }
        }
        if !queue.confirm(&mut next) {
            return Err("synthetic import failed".to_string());
        }
        let origin: Timestamp = synthetic::ORIGIN.parse().expect("valid constant");
        next.commit(Change::SetReference { reference: origin })?;
        for spec in &synthetic::SPECS {
            let sensor = next.project.sensor_by_code(&spec.code()).map(|s| s.id);
            if let Some(trace) = next.project.traces().iter().find(|t| Some(t.sensor) == sensor) {
                next.truth.insert(trace.id, spec.delay);
            }
        }
        Ok(next)
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
    pub samples: SampleBuf,
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
    /// Files no reader recognised, awaiting a raw layout.
    pub raw_pending: Vec<PathBuf>,
    pub raw_layout: RawLayout,
}

impl ImportQueue {
    pub fn is_open(&self) -> bool {
        !self.rows.is_empty() || !self.failures.is_empty() || !self.raw_pending.is_empty()
    }

    fn push_raw(&mut self, path: PathBuf, project: &Project) {
        if self.raw_pending.is_empty() {
            // Most recent layout used in this project, else the last one entered.
            let used = project.traces().iter().rev().find_map(|t| match &t.source {
                TraceSource::File { layout, .. } => *layout,
            });
            if let Some(layout) = used {
                self.raw_layout = layout;
            }
        }
        if !self.raw_pending.contains(&path) {
            self.raw_pending.push(path);
        }
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

            let trace = Trace {
                id: Id::generate(),
                sensor,
                start,
                sample_rate,
                len: row.samples.len(),
                kind: row.samples.kind(),
                source: row.source,
                processing: Default::default(),
                alignment: None,
            };
            let id = trace.id;
            if !workspace.commit_or_report(Change::AddTrace(trace)) {
                break;
            }
            workspace.samples.insert(id, SampleState::Loaded(row.samples.into()));
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
        self.raw_pending.clear();
    }
}

enum DialogOutcome {
    Import(Vec<PathBuf>),
    Open(PathBuf),
    SaveAs(PathBuf),
}

enum ReadFailure {
    /// No reader recognised the file; ask for a raw layout.
    Unrecognised(PathBuf),
    Failed(String),
}

type FileOutcome = Result<(FileRead, Option<Timestamp>), ReadFailure>;
type SampleLoad = Vec<(Id, Result<TraceSamples, String>)>;

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

fn modified_time(path: &Path) -> Option<Timestamp> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| Timestamp::new(d.as_secs() as i64, f64::from(d.subsec_nanos()) * 1e-9))
}

/// Reads files, or with `layout` decodes them as raw binary. A SigMF pair
/// dropped together is read once.
fn read_files(paths: Vec<PathBuf>, layout: Option<RawLayout>) -> Task<Vec<FileOutcome>> {
    AsyncComputeTaskPool::get().spawn(async move {
        let mut seen = HashSet::new();
        paths
            .into_iter()
            .map(|path| squiggle_io::primary_path(&path))
            .filter(|path| seen.insert(path.clone()))
            .map(|path| {
                let read = match layout {
                    Some(layout) => squiggle_io::read_raw(&path, layout),
                    None => squiggle_io::read_file(&path),
                };
                match read {
                    Ok(read) => Ok((read, modified_time(&path))),
                    Err(ReadError::UnknownFormat(path)) => Err(ReadFailure::Unrecognised(path)),
                    Err(error) => Err(ReadFailure::Failed(error.to_string())),
                }
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
                        .add_filter("Waveforms", &squiggle_io::EXTENSIONS)
                        .add_filter("All files (raw binary)", &["*"])
                        .pick_files()
                        .await?;
                    Some(DialogOutcome::Import(files.iter().map(|f| f.path().to_path_buf()).collect()))
                }));
            }
            WorkspaceRequest::Import(paths) => tasks.reads.push(read_files(paths.clone(), None)),
            WorkspaceRequest::ImportRaw(paths, layout) => {
                queue.raw_pending.retain(|p| !paths.contains(p));
                queue.raw_layout = *layout;
                tasks.reads.push(read_files(paths.clone(), Some(*layout)));
            }
            WorkspaceRequest::Synthetic => match Workspace::synthetic(workspace.analyst.clone()) {
                Ok(next) => {
                    workspace.replace(next);
                    queue.cancel();
                }
                Err(error) => workspace.error = Some(error),
            },
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
            Err(ReadFailure::Unrecognised(path)) => queue.push_raw(path, &workspace.project),
            Err(ReadFailure::Failed(error)) => queue.failures.push(error),
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
                Ok(samples) => SampleState::Loaded(samples),
                Err(error) => SampleState::Failed(error),
            };
            workspace.samples.insert(id, state);
        }
        workspace.revision += 1;
    }
}

fn sync_title(workspace: Res<Workspace>, mut plot: ResMut<Plot>, mut seen: Local<u64>) {
    if *seen != workspace.revision {
        *seen = workspace.revision;
        plot.name = workspace.title();
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
                layout: None,
            },
            samples: SampleBuf::Real(vec![0.0; 4]),
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
            ..ImportQueue::default()
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
            ..ImportQueue::default()
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
            ..ImportQueue::default()
        };
        assert!(!queue.confirm(&mut workspace));
    }

    #[test]
    fn a_second_trace_for_a_known_sensor_reuses_it() {
        let mut workspace = workspace();
        let mut queue = ImportQueue { rows: vec![row("s", "2026-10-11T00:00:00Z", "100", Some("seismic"))], ..ImportQueue::default() };
        assert!(queue.confirm(&mut workspace));
        queue.rows.push(row("s", "2026-10-11T00:01:00Z", "100", Some("seismic")));
        assert!(queue.confirm(&mut workspace));
        assert_eq!(workspace.project.sensors().len(), 1);
        assert_eq!(workspace.project.traces().len(), 2);
    }

    #[test]
    fn complex_rows_become_complex_traces() {
        let mut workspace = workspace();
        let mut complex = row("rx", "2026-10-11T00:00:00Z", "1e6", Some("rf"));
        complex.samples = SampleBuf::Complex(vec![[1.0, 0.0]; 3]);
        let mut queue = ImportQueue { rows: vec![complex], ..ImportQueue::default() };
        assert!(queue.confirm(&mut workspace));
        let trace = &workspace.project.traces()[0];
        assert_eq!(trace.kind, squiggle_core::SampleKind::Complex);
        assert!(matches!(workspace.samples.get(&trace.id), Some(SampleState::Loaded(TraceSamples::Complex(_)))));
    }

    #[test]
    fn raw_files_reuse_the_projects_last_layout() {
        let mut workspace = workspace();
        let layout = RawLayout { channels: 4, big_endian: true, ..RawLayout::default() };
        let mut raw = row("dump", "2026-10-11T00:00:00Z", "1e9", Some("emp"));
        let TraceSource::File { layout: slot, .. } = &mut raw.source;
        *slot = Some(layout);
        let mut queue = ImportQueue { rows: vec![raw], ..ImportQueue::default() };
        assert!(queue.confirm(&mut workspace));

        queue.push_raw("a.bin".into(), &workspace.project);
        queue.push_raw("a.bin".into(), &workspace.project);
        assert_eq!(queue.raw_pending.len(), 1);
        assert_eq!(queue.raw_layout, layout);
        assert!(queue.is_open());
    }

    #[test]
    fn commands_undo_and_redo_through_the_log() {
        use squiggle_core::sensor::Provenance;
        let mut workspace = workspace();
        let mut queue = ImportQueue { rows: vec![row("s", "2026-10-11T00:00:00Z", "100", Some("seismic"))], ..ImportQueue::default() };
        queue.confirm(&mut workspace);
        let trace = workspace.project.traces()[0].id;
        let shift = |offset_s| Change::SetAlignment {
            trace,
            alignment: Some(squiggle_core::Alignment {
                offset_s,
                provenance: Provenance::Manual { analyst: Analyst("ana".into()), at: Timestamp::now() },
                locked: true,
            }),
            revert: false,
        };
        let ops = workspace.log.len();
        workspace.execute("Shift", vec![shift(1.0), shift(2.5)]).unwrap();
        assert_eq!(workspace.project.trace(trace).unwrap().offset(), 2.5);
        assert_eq!(workspace.undo_label(), Some("Shift"));

        workspace.undo();
        assert_eq!(workspace.project.trace(trace).unwrap().alignment, None);
        assert_eq!(workspace.redo_label(), Some("Shift"));
        workspace.redo();
        assert_eq!(workspace.project.trace(trace).unwrap().offset(), 2.5);
        // Two forward, two inverse, two redone: the log only grows.
        assert_eq!(workspace.log.len(), ops + 6);
        assert_eq!(Project::replay(&workspace.log).unwrap(), workspace.project);

        // A failing change leaves nothing behind.
        let before = workspace.log.len();
        let ghost = Change::SetProcessing { trace: Id::generate(), processing: Default::default() };
        assert!(workspace.execute("Bad", vec![shift(9.0), ghost]).is_err());
        assert_eq!(workspace.log.len(), before);
        assert_eq!(workspace.project.trace(trace).unwrap().offset(), 2.5);
        assert_eq!(workspace.redo_label(), None);
    }

    #[test]
    fn the_synthetic_event_loads_with_its_truth() {
        let workspace = Workspace::synthetic(Analyst("ana".into())).unwrap();
        assert_eq!(workspace.project.traces().len(), synthetic::SPECS.len());
        assert_eq!(workspace.truth.len(), synthetic::SPECS.len());
        assert!(workspace.project.traces().iter().all(|t| workspace.project.trace_phenomenology(t).is_some()));
        assert_eq!(workspace.project.reference, Some(synthetic::ORIGIN.parse().unwrap()));
    }

    #[test]
    fn saving_then_opening_restores_the_project() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.squiggle");
        let mut workspace = workspace();
        let mut queue = ImportQueue { rows: vec![row("s", "2026-10-11T00:00:00Z", "100", Some("rf"))], ..ImportQueue::default() };
        queue.confirm(&mut workspace);
        workspace.save_as(&path).unwrap();
        assert!(!workspace.is_unsaved());

        // Edits after saving go straight to the file.
        workspace.commit(Change::RenameProject { name: "renamed".to_string() }).unwrap();
        let reopened = Workspace::open(&path, workspace.analyst.clone()).unwrap();
        assert_eq!(reopened.project, workspace.project);
    }
}
