//! Shared projects over TCP (`heidtmare.squiggle.sync/1`).
//!
//! The server holds each shared project's log ([`Shared`]) in a project file
//! of its own, plus a blob store of the files behind traces and models, keyed
//! by the content hash the project records. Clients keep working offline in
//! their own project file and sync by pushing the ops made since their last
//! pull; the reply carries each op's [`Outcome`] and the shared ops they had
//! not seen, in the shared order, which the client adopts.
//!
//! Both sides greet with `SQSY` and a little-endian `u16` version, then
//! exchange messages: a JSON frame (type 1) followed by the byte frames
//! (type 2) it declares. A frame is a type byte, a little-endian `u32`
//! payload length, the payload.

use core::fmt;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use squiggle_core::sync::{Conflict, Outcome, Shared};
use squiggle_core::{Analyst, Change, Id, Op, Trace, TraceSamples, TraceSource};

use crate::store::{ProjectStore, SqliteStore, StoreError};
use crate::{PROJECT_EXTENSION, ReadError};

pub const PROTOCOL: &str = "heidtmare.squiggle.sync/1";
const MAGIC: &[u8; 4] = b"SQSY";
const VERSION: u16 = 1;
const JSON: u8 = 1;
const BYTES: u8 = 2;
/// Largest frame accepted: 1 GiB.
const MAX_FRAME: u32 = 1 << 30;
/// Project-file metadata keys of a synced project.
const REMOTE_KEY: &str = "remote";
const CONFLICTS_KEY: &str = "conflicts";

#[derive(Debug)]
pub enum SyncError {
    Io(std::io::Error),
    Store(StoreError),
    Protocol(String),
    /// Refused by the server.
    Server(String),
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Store(error) => write!(f, "{error}"),
            Self::Protocol(why) => write!(f, "{PROTOCOL}: {why}"),
            Self::Server(why) => write!(f, "server: {why}"),
        }
    }
}

impl std::error::Error for SyncError {}

impl From<std::io::Error> for SyncError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<StoreError> for SyncError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

impl From<ReadError> for SyncError {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::Io(error) => Self::Io(error),
            other => Self::Protocol(other.to_string()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Request {
    Hello { analyst: Analyst },
    List,
    Create { ops: Vec<Op> },
    Sync { project: Id, base: usize, ops: Vec<Op> },
    Missing { hashes: Vec<String> },
    /// Followed by one byte frame per name.
    PutBlob { hash: String, names: Vec<String> },
    GetBlob { hash: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Response {
    Hello,
    Projects { projects: Vec<ProjectInfo> },
    Created { head: usize },
    Synced(Synced),
    Missing { hashes: Vec<String> },
    Stored,
    /// Followed by one byte frame per name.
    Blob { names: Vec<String> },
    Error { message: String },
}

/// A project the server shares.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub id: Id,
    pub name: String,
    pub ops: usize,
    /// Everyone who has edited it, first editor first.
    pub analysts: Vec<String>,
}

/// The server's reply to a push.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Synced {
    /// One per pushed op.
    pub outcomes: Vec<Outcome>,
    /// The shared log from the pushed `base` to `head`.
    pub ops: Vec<Op>,
    pub head: usize,
}

/// Files of one blob, in hash order.
pub type Files = Vec<(String, Vec<u8>)>;

fn greeting() -> [u8; 6] {
    let mut out = [0u8; 6];
    out[..4].copy_from_slice(MAGIC);
    out[4..].copy_from_slice(&VERSION.to_le_bytes());
    out
}

fn check_greeting(reader: &mut impl Read) -> Result<(), SyncError> {
    let mut hello = [0u8; 6];
    reader.read_exact(&mut hello)?;
    if &hello[..4] != MAGIC {
        return Err(SyncError::Protocol("not a sync peer".to_string()));
    }
    let version = u16::from_le_bytes([hello[4], hello[5]]);
    if version != VERSION {
        return Err(SyncError::Protocol(format!("version {version} is not {VERSION}")));
    }
    Ok(())
}

fn write_frame(writer: &mut impl Write, kind: u8, payload: &[u8]) -> Result<(), SyncError> {
    let len = u32::try_from(payload.len()).ok().filter(|&n| n <= MAX_FRAME).ok_or_else(|| SyncError::Protocol("frame too large".to_string()))?;
    writer.write_all(&[kind])?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(payload)?;
    Ok(())
}

/// `None` when the peer closed cleanly between frames.
fn read_frame(reader: &mut impl Read, expected: u8) -> Result<Option<Vec<u8>>, SyncError> {
    let mut head = [0u8; 5];
    match reader.read_exact(&mut head[..1]) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    reader.read_exact(&mut head[1..])?;
    if head[0] != expected {
        return Err(SyncError::Protocol(format!("frame type {} where {expected} was expected", head[0])));
    }
    let len = u32::from_le_bytes([head[1], head[2], head[3], head[4]]);
    if len > MAX_FRAME {
        return Err(SyncError::Protocol(format!("frame of {len} bytes")));
    }
    let mut payload = vec![0u8; len as usize];
    reader.read_exact(&mut payload)?;
    Ok(Some(payload))
}

fn send<T: Serialize>(writer: &mut impl Write, message: &T, attachments: &[&[u8]]) -> Result<(), SyncError> {
    let json = serde_json::to_vec(message).map_err(|e| SyncError::Protocol(e.to_string()))?;
    write_frame(writer, JSON, &json)?;
    for bytes in attachments {
        write_frame(writer, BYTES, bytes)?;
    }
    writer.flush()?;
    Ok(())
}

fn receive<T: for<'de> Deserialize<'de>>(reader: &mut impl Read) -> Result<Option<T>, SyncError> {
    let Some(json) = read_frame(reader, JSON)? else { return Ok(None) };
    serde_json::from_slice(&json).map(Some).map_err(|e| SyncError::Protocol(e.to_string()))
}

fn attachments(reader: &mut impl Read, names: Vec<String>) -> Result<Files, SyncError> {
    names
        .into_iter()
        .map(|name| Ok((name, read_frame(reader, BYTES)?.ok_or_else(|| SyncError::Protocol("closed mid-message".to_string()))?)))
        .collect()
}

/// A plain file name: no directories, not hidden.
fn plain_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\']) && !name.starts_with('.')
}

fn plain_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}

/// BLAKE3 of the files concatenated in order: the hash a project records.
pub fn files_hash(files: &[(String, Vec<u8>)]) -> String {
    let mut hasher = blake3::Hasher::new();
    for (_, bytes) in files {
        hasher.update(bytes);
    }
    hasher.finalize().to_hex().to_string()
}

/// A blob's file names in hash order; hidden, so no file can take its name.
const MANIFEST: &str = ".files";

/// Where a blob's files sit under a blob directory.
pub fn blob_dir(root: &Path, hash: &str) -> PathBuf {
    root.join(hash)
}

/// Writes a blob's files under `root` once their hash checks.
pub fn store_blob(root: &Path, hash: &str, files: &[(String, Vec<u8>)]) -> Result<(), SyncError> {
    if !plain_hash(hash) || files.is_empty() || files.iter().any(|(n, _)| !plain_name(n)) {
        return Err(SyncError::Protocol("bad blob name".to_string()));
    }
    if files_hash(files) != hash {
        return Err(SyncError::Protocol(format!("blob {hash} does not match its files")));
    }
    let dir = blob_dir(root, hash);
    if dir.is_dir() {
        return Ok(());
    }
    let staging = root.join(format!(".{hash}.{}", Id::generate()));
    std::fs::create_dir_all(&staging)?;
    for (name, bytes) in files {
        std::fs::write(staging.join(name), bytes)?;
    }
    let order: Vec<&str> = files.iter().map(|f| f.0.as_str()).collect();
    std::fs::write(staging.join(MANIFEST), order.join("\n"))?;
    // Another writer may have won; its files are the same.
    if std::fs::rename(&staging, &dir).is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    Ok(())
}

/// The content hash a trace's source records.
pub fn source_hash(source: &TraceSource) -> &str {
    match source {
        TraceSource::File { content_hash, .. } | TraceSource::Stream { content_hash, .. } => content_hash,
    }
}

/// The files behind a trace, in hash order: a file and its companions, or a
/// stream recording's checkpointed bytes.
pub fn trace_files(source: &TraceSource) -> Result<Files, ReadError> {
    let name = |path: &Path| path.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or_else(|| ReadError::Format(format!("{}: no file name", path.display())));
    match source {
        TraceSource::File { path, format, .. } => {
            let path = Path::new(path);
            let mut paths = vec![path.to_path_buf()];
            paths.extend(crate::reader(format).map(|r| r.companions(path)).unwrap_or_default());
            paths.iter().map(|p| Ok((name(p)?, std::fs::read(p)?))).collect()
        }
        TraceSource::Stream { path, bytes, .. } => {
            let path = Path::new(path);
            let mut data = std::fs::read(path)?;
            if (data.len() as u64) < *bytes {
                return Err(ReadError::Changed(path.to_path_buf()));
            }
            data.truncate(*bytes as usize);
            Ok(vec![(name(path)?, data)])
        }
    }
}

/// `source` with its path moved into its blob under `root`.
pub fn cached_source(source: &TraceSource, root: &Path) -> TraceSource {
    let mut source = source.clone();
    let hash = source_hash(&source).to_string();
    match &mut source {
        TraceSource::File { path, .. } | TraceSource::Stream { path, .. } => {
            if let Some(name) = Path::new(path).file_name() {
                *path = blob_dir(root, &hash).join(name).to_string_lossy().into_owned();
            }
        }
    }
    source
}

/// Loads a trace from its own path, else from its blob under `cache`.
pub fn load_trace_from(trace: &Trace, cache: Option<&Path>) -> Result<TraceSamples, ReadError> {
    match (crate::load_trace(trace), cache) {
        (Err(error @ (ReadError::Io(_) | ReadError::Changed(_))), Some(cache)) => {
            let cached = Trace { source: cached_source(&trace.source, cache), ..trace.clone() };
            if Path::new(match &cached.source {
                TraceSource::File { path, .. } | TraceSource::Stream { path, .. } => path,
            })
            .exists()
            {
                crate::load_trace(&cached)
            } else {
                Err(error)
            }
        }
        (result, _) => result,
    }
}

/// How a project file is linked to a shared project.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Remote {
    pub endpoint: String,
    pub project: Id,
    /// The first `confirmed` ops of the local log are the shared log's; the rest are pending.
    pub confirmed: usize,
}

impl Remote {
    pub fn load(store: &SqliteStore) -> Result<Option<Self>, SyncError> {
        store.meta(REMOTE_KEY)?.map(|json| serde_json::from_str(&json).map_err(|e| SyncError::Store(StoreError::Json(e)))).transpose()
    }

    pub fn save(store: &mut SqliteStore, remote: Option<&Self>) -> Result<(), SyncError> {
        let json = remote.map(|r| serde_json::to_string(r).expect("plain data"));
        Ok(store.set_meta(REMOTE_KEY, json.as_deref())?)
    }
}

/// Conflicts awaiting an analyst, kept in the project file.
pub fn load_conflicts(store: &SqliteStore) -> Result<Vec<Conflict>, SyncError> {
    match store.meta(CONFLICTS_KEY)? {
        Some(json) => serde_json::from_str(&json).map_err(|e| SyncError::Store(StoreError::Json(e))),
        None => Ok(Vec::new()),
    }
}

pub fn save_conflicts(store: &mut SqliteStore, conflicts: &[Conflict]) -> Result<(), SyncError> {
    let json = (!conflicts.is_empty()).then(|| serde_json::to_string(conflicts).expect("plain data"));
    Ok(store.set_meta(CONFLICTS_KEY, json.as_deref())?)
}

/// A connection to a sync server.
pub struct SyncClient {
    stream: TcpStream,
}

impl SyncClient {
    /// Connects as `analyst`; every op pushed must be theirs.
    pub fn connect(endpoint: &str, analyst: &Analyst, timeout: Duration) -> Result<Self, SyncError> {
        let address = endpoint.to_socket_addrs()?.next().ok_or_else(|| SyncError::Protocol(format!("{endpoint}: no address")))?;
        let mut stream = TcpStream::connect_timeout(&address, timeout)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        stream.write_all(&greeting())?;
        check_greeting(&mut stream)?;
        let mut client = Self { stream };
        match client.call(&Request::Hello { analyst: analyst.clone() }, &[])? {
            Response::Hello => Ok(client),
            other => Err(unexpected(other)),
        }
    }

    fn call(&mut self, request: &Request, attached: &[&[u8]]) -> Result<Response, SyncError> {
        send(&mut self.stream, request, attached)?;
        match receive(&mut self.stream)? {
            Some(Response::Error { message }) => Err(SyncError::Server(message)),
            Some(response) => Ok(response),
            None => Err(SyncError::Protocol("the server closed the connection".to_string())),
        }
    }

    pub fn list(&mut self) -> Result<Vec<ProjectInfo>, SyncError> {
        match self.call(&Request::List, &[])? {
            Response::Projects { projects } => Ok(projects),
            other => Err(unexpected(other)),
        }
    }

    /// Shares a project: its whole log. Returns the shared head.
    pub fn create(&mut self, ops: &[Op]) -> Result<usize, SyncError> {
        match self.call(&Request::Create { ops: ops.to_vec() }, &[])? {
            Response::Created { head } => Ok(head),
            other => Err(unexpected(other)),
        }
    }

    /// Pushes ops made after the first `base` shared ops, and pulls the rest.
    pub fn sync(&mut self, project: Id, base: usize, ops: &[Op]) -> Result<Synced, SyncError> {
        match self.call(&Request::Sync { project, base, ops: ops.to_vec() }, &[])? {
            Response::Synced(synced) => Ok(synced),
            other => Err(unexpected(other)),
        }
    }

    /// Which of `hashes` the server lacks.
    pub fn missing(&mut self, hashes: &[String]) -> Result<Vec<String>, SyncError> {
        match self.call(&Request::Missing { hashes: hashes.to_vec() }, &[])? {
            Response::Missing { hashes } => Ok(hashes),
            other => Err(unexpected(other)),
        }
    }

    pub fn put_blob(&mut self, hash: &str, files: &[(String, Vec<u8>)]) -> Result<(), SyncError> {
        let names = files.iter().map(|f| f.0.clone()).collect();
        let bytes: Vec<&[u8]> = files.iter().map(|f| f.1.as_slice()).collect();
        match self.call(&Request::PutBlob { hash: hash.to_string(), names }, &bytes)? {
            Response::Stored => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    pub fn get_blob(&mut self, hash: &str) -> Result<Files, SyncError> {
        match self.call(&Request::GetBlob { hash: hash.to_string() }, &[])? {
            Response::Blob { names } => {
                let files = attachments(&mut self.stream, names)?;
                if files_hash(&files) != hash {
                    return Err(SyncError::Protocol(format!("blob {hash} does not match its files")));
                }
                Ok(files)
            }
            other => Err(unexpected(other)),
        }
    }
}

fn unexpected(response: Response) -> SyncError {
    SyncError::Protocol(format!("unexpected reply {response:?}").chars().take(200).collect())
}

struct Hosted {
    shared: Shared,
    store: SqliteStore,
}

/// The server's projects and blobs, in one directory.
pub struct Hub {
    dir: PathBuf,
    projects: Mutex<HashMap<Id, Arc<Mutex<Hosted>>>>,
}

impl Hub {
    /// Opens or creates a server directory: `<project id>.squiggle` files and `blobs/`.
    pub fn open(dir: &Path) -> Result<Self, SyncError> {
        std::fs::create_dir_all(dir.join("blobs"))?;
        let mut projects = HashMap::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != PROJECT_EXTENSION) {
                continue;
            }
            let store = SqliteStore::open(&path)?;
            let shared = Shared::new(store.load()?).map_err(|e| SyncError::Protocol(format!("{}: {e}", path.display())))?;
            let id = shared.project().id.ok_or_else(|| SyncError::Protocol(format!("{}: empty", path.display())))?;
            projects.insert(id, Arc::new(Mutex::new(Hosted { shared, store })));
        }
        Ok(Self { dir: dir.to_path_buf(), projects: Mutex::new(projects) })
    }

    fn blobs(&self) -> PathBuf {
        self.dir.join("blobs")
    }

    fn project(&self, id: Id) -> Result<Arc<Mutex<Hosted>>, String> {
        let projects = self.projects.lock().unwrap_or_else(|p| p.into_inner());
        projects.get(&id).cloned().ok_or_else(|| format!("no shared project {id}"))
    }

    pub fn list(&self) -> Vec<ProjectInfo> {
        let projects = self.projects.lock().unwrap_or_else(|p| p.into_inner());
        let mut out: Vec<ProjectInfo> = projects
            .iter()
            .map(|(id, hosted)| {
                let hosted = hosted.lock().unwrap_or_else(|p| p.into_inner());
                let mut analysts: Vec<String> = Vec::new();
                for op in hosted.shared.log() {
                    if !analysts.contains(&op.analyst.0) {
                        analysts.push(op.analyst.0.clone());
                    }
                }
                ProjectInfo { id: *id, name: hosted.shared.project().name.clone(), ops: hosted.shared.head(), analysts }
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name).then(a.id.cmp(&b.id)));
        out
    }

    fn create(&self, ops: Vec<Op>) -> Result<usize, String> {
        let Some(Op { change: Change::CreateProject { id, .. }, .. }) = ops.first() else {
            return Err("a project log starts with CreateProject".to_string());
        };
        let id = *id;
        let mut projects = self.projects.lock().unwrap_or_else(|p| p.into_inner());
        if projects.contains_key(&id) {
            return Err(format!("project {id} is already shared"));
        }
        let shared = Shared::new(ops).map_err(|e| e.to_string())?;
        let path = self.dir.join(format!("{id}.{PROJECT_EXTENSION}"));
        let mut store = SqliteStore::create(&path).map_err(|e| e.to_string())?;
        store.append(shared.log()).map_err(|e| e.to_string())?;
        let head = shared.head();
        projects.insert(id, Arc::new(Mutex::new(Hosted { shared, store })));
        Ok(head)
    }

    fn sync(&self, analyst: &Analyst, project: Id, base: usize, ops: &[Op]) -> Result<Synced, String> {
        if let Some(op) = ops.iter().find(|op| &op.analyst != analyst) {
            return Err(format!("op {} is by {}, not {analyst}", op.id, op.analyst));
        }
        let hosted = self.project(project)?;
        let mut hosted = hosted.lock().unwrap_or_else(|p| p.into_inner());
        let head = hosted.shared.head();
        if base > head {
            return Err(format!("base {base} is past the shared head {head}"));
        }
        let outcomes = hosted.shared.push(base, ops);
        let appended = hosted.shared.log()[head..].to_vec();
        if let Err(error) = hosted.store.append(&appended) {
            // Keep memory and disk equal: reload what was saved.
            if let Ok(shared) = hosted.store.load().map_err(|e| e.to_string()).and_then(|ops| Shared::new(ops).map_err(|e| e.to_string())) {
                hosted.shared = shared;
            }
            return Err(format!("not saved: {error}"));
        }
        let head = hosted.shared.head();
        Ok(Synced { outcomes, ops: hosted.shared.log()[base..head].to_vec(), head })
    }

    /// Answers one request; `analyst` is who said hello.
    fn handle(&self, analyst: &Option<Analyst>, request: Request, files: Files) -> (Response, Files) {
        let reply = |result: Result<Response, String>| result.unwrap_or_else(|message| Response::Error { message });
        let who = || analyst.clone().ok_or_else(|| "say hello first".to_string());
        match request {
            Request::Hello { .. } => (Response::Hello, Vec::new()),
            Request::List => (Response::Projects { projects: self.list() }, Vec::new()),
            Request::Create { ops } => (reply(who().and_then(|_| self.create(ops)).map(|head| Response::Created { head })), Vec::new()),
            Request::Sync { project, base, ops } => {
                (reply(who().and_then(|a| self.sync(&a, project, base, &ops)).map(Response::Synced)), Vec::new())
            }
            Request::Missing { hashes } => {
                let hashes = hashes.into_iter().filter(|h| !plain_hash(h) || !blob_dir(&self.blobs(), h).is_dir()).collect();
                (Response::Missing { hashes }, Vec::new())
            }
            Request::PutBlob { hash, .. } => {
                (reply(who().and_then(|_| store_blob(&self.blobs(), &hash, &files).map_err(|e| e.to_string())).map(|()| Response::Stored)), Vec::new())
            }
            Request::GetBlob { hash } => match read_blob(&self.blobs(), &hash) {
                Ok(files) => (Response::Blob { names: files.iter().map(|f| f.0.clone()).collect() }, files),
                Err(message) => (Response::Error { message }, Vec::new()),
            },
        }
    }
}

/// A stored blob's files, in hash order.
fn read_blob(root: &Path, hash: &str) -> Result<Files, String> {
    if !plain_hash(hash) {
        return Err("bad blob name".to_string());
    }
    let dir = blob_dir(root, hash);
    let order = std::fs::read_to_string(dir.join(MANIFEST)).map_err(|_| format!("no blob {hash}"))?;
    order.lines().map(|n| Ok((n.to_string(), std::fs::read(dir.join(n)).map_err(|e| e.to_string())?))).collect()
}

/// Serves `hub` to each client of `listener` until `stop` is set.
pub fn serve(listener: TcpListener, hub: Arc<Hub>, stop: Arc<AtomicBool>) -> std::io::Result<()> {
    listener.set_nonblocking(true)?;
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let (hub, stop) = (hub.clone(), stop.clone());
                std::thread::spawn(move || {
                    let _ = serve_client(stream, &hub, &stop);
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn serve_client(mut stream: TcpStream, hub: &Hub, stop: &AtomicBool) -> Result<(), SyncError> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.write_all(&greeting())?;
    check_greeting(&mut stream)?;
    let mut analyst = None;
    while !stop.load(Ordering::Relaxed) {
        let Some(request) = receive::<Request>(&mut stream)? else { return Ok(()) };
        let files = match &request {
            Request::PutBlob { names, .. } => attachments(&mut stream, names.clone())?,
            _ => Vec::new(),
        };
        if let Request::Hello { analyst: who } = &request {
            analyst = Some(who.clone());
        }
        let (response, files) = hub.handle(&analyst, request, files);
        let bytes: Vec<&[u8]> = files.iter().map(|f| f.1.as_slice()).collect();
        send(&mut stream, &response, &bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::{Project, SampleBuf};

    fn server() -> (tempfile::TempDir, String, Arc<AtomicBool>) {
        let dir = tempfile::tempdir().unwrap();
        let hub = Arc::new(Hub::open(dir.path()).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        std::thread::spawn(move || serve(listener, hub, flag));
        (dir, endpoint, stop)
    }

    fn client(endpoint: &str, name: &str) -> SyncClient {
        SyncClient::connect(endpoint, &Analyst(name.into()), Duration::from_secs(5)).unwrap()
    }

    #[test]
    fn two_clients_share_a_project_through_the_server() {
        let (dir, endpoint, stop) = server();
        let (ana, bo) = (Analyst("ana".into()), Analyst("bo".into()));
        let (mut project, ops) = Project::create("shared", &ana);
        let id = project.id.unwrap();
        let mut a = client(&endpoint, "ana");
        assert_eq!(a.create(&ops).unwrap(), ops.len());
        assert!(matches!(a.create(&ops), Err(SyncError::Server(_))));

        let mut b = client(&endpoint, "bo");
        let listed = b.list().unwrap();
        assert_eq!(listed, [ProjectInfo { id, name: "shared".into(), ops: ops.len(), analysts: vec!["ana".into()] }]);
        let pulled = b.sync(id, 0, &[]).unwrap();
        assert_eq!(pulled.ops, ops);

        let rename = project.commit(&ana, Change::RenameProject { name: "renamed".into() }).unwrap();
        let synced = a.sync(id, ops.len(), std::slice::from_ref(&rename)).unwrap();
        assert_eq!(synced.outcomes, [Outcome::Accepted]);
        // Pushing someone else's op is refused.
        let mut theirs = Project::replay(&ops).unwrap();
        let forged = theirs.commit(&bo, Change::RenameProject { name: "x".into() }).unwrap();
        assert!(matches!(a.sync(id, ops.len(), &[forged]), Err(SyncError::Server(_))));

        // A restarted server still has it.
        stop.store(true, Ordering::Relaxed);
        let hub = Hub::open(dir.path()).unwrap();
        assert_eq!(hub.list()[0].name, "renamed");
        assert_eq!(hub.list()[0].analysts, ["ana"]);
    }

    #[test]
    fn trace_files_round_trip_through_the_blob_store() {
        let (_dir, endpoint, _stop) = server();
        let local = tempfile::tempdir().unwrap();
        let meta = local.path().join("a.sigmf-meta");
        std::fs::write(&meta, r#"{"global":{"core:datatype":"rf32_le","core:sample_rate":10}}"#).unwrap();
        std::fs::write(local.path().join("a.sigmf-data"), [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let read = crate::read_file(&meta).unwrap();
        let source = read.source(0);
        let files = trace_files(&source).unwrap();
        assert_eq!(files_hash(&files), source_hash(&source));

        let mut a = client(&endpoint, "ana");
        let hash = source_hash(&source).to_string();
        assert_eq!(a.missing(std::slice::from_ref(&hash)).unwrap(), std::slice::from_ref(&hash));
        a.put_blob(&hash, &files).unwrap();
        assert!(a.missing(std::slice::from_ref(&hash)).unwrap().is_empty());
        let mut wrong = files.clone();
        wrong[1].1.push(0);
        assert!(a.put_blob(&hash, &wrong).is_err());

        // Another machine: the original path is gone, the cache has it.
        let cache = tempfile::tempdir().unwrap();
        let fetched = client(&endpoint, "bo").get_blob(&hash).unwrap();
        assert_eq!(fetched, files);
        store_blob(cache.path(), &hash, &fetched).unwrap();
        let trace = Trace {
            id: Id::generate(),
            sensor: Id::generate(),
            start: squiggle_core::Timestamp::new(0, 0.0),
            sample_rate: 10.0,
            len: 2,
            kind: Default::default(),
            source: match source.clone() {
                TraceSource::File { format, channel, content_hash, layout, .. } => {
                    TraceSource::File { path: "/nowhere/a.sigmf-meta".into(), format, channel, content_hash, layout }
                }
                stream => stream,
            },
            processing: Default::default(),
            alignment: None,
        };
        assert!(crate::load_trace(&trace).is_err());
        let TraceSamples::Real(samples) = load_trace_from(&trace, Some(cache.path())).unwrap() else { panic!() };
        assert_eq!(samples.iter().collect::<Vec<_>>(), [1.0, 2.0]);
    }

    #[test]
    fn stream_recordings_share_their_checkpointed_bytes() {
        let local = tempfile::tempdir().unwrap();
        let path = local.path().join("c.f32");
        let mut recorder = crate::stream::Recorder::create(&path, Default::default()).unwrap();
        recorder.append(&SampleBuf::Real(vec![1.0, 2.0, 3.0])).unwrap();
        let (bytes, content_hash) = recorder.checkpoint().unwrap();
        recorder.append(&SampleBuf::Real(vec![4.0])).unwrap();
        let source = TraceSource::Stream { endpoint: "e".into(), code: "c".into(), path: path.to_string_lossy().into(), bytes, content_hash };
        let files = trace_files(&source).unwrap();
        assert_eq!(files[0].1.len(), 12);
        assert_eq!(files_hash(&files), source_hash(&source));
    }
}
