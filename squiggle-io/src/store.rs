//! Project persistence. A project file stores only the op log; state is
//! rebuilt by [`Project::replay`](squiggle_core::Project::replay).

use core::fmt;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use squiggle_core::Op;
use squiggle_core::project::FORMAT;

pub trait ProjectStore {
    /// Appends ops atomically, in order.
    fn append(&mut self, ops: &[Op]) -> Result<(), StoreError>;
    /// Every op, in append order.
    fn load(&self) -> Result<Vec<Op>, StoreError>;
}

#[derive(Debug)]
pub enum StoreError {
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    Io(std::io::Error),
    /// Not a project file, or a format this build cannot read.
    Format(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "corrupt op: {error}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Format(found) => write!(f, "unsupported project format: {found}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// SQLite file. `ops.seq` is local append order only; `ops.id` is the global identity.
pub struct SqliteStore {
    connection: Connection,
}

impl SqliteStore {
    /// Creates a project file, replacing any file at `path`.
    pub fn create(path: &Path) -> Result<Self, StoreError> {
        if path.exists() {
            std::fs::remove_file(path).map_err(StoreError::Io)?;
        }
        let connection = Connection::open(path)?;
        connection.execute_batch(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE ops (seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, op TEXT NOT NULL);",
        )?;
        connection.execute("INSERT INTO meta (key, value) VALUES ('format', ?1)", [FORMAT])?;
        Ok(Self { connection })
    }

    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if !path.is_file() {
            return Err(StoreError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, path.display().to_string())));
        }
        let connection = Connection::open(path)?;
        let format: Option<String> = connection
            .query_row("SELECT value FROM meta WHERE key = 'format'", [], |row| row.get(0))
            .optional()
            .unwrap_or(None);
        match format {
            Some(format) if format == FORMAT => Ok(Self { connection }),
            other => Err(StoreError::Format(other.unwrap_or_else(|| "not a project file".to_string()))),
        }
    }
}

impl SqliteStore {
    pub fn meta(&self, key: &str) -> Result<Option<String>, StoreError> {
        Ok(self.connection.query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| row.get(0)).optional()?)
    }

    /// Sets or, with `None`, clears a metadata value.
    pub fn set_meta(&mut self, key: &str, value: Option<&str>) -> Result<(), StoreError> {
        match value {
            Some(value) => self.connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![key, value])?,
            None => self.connection.execute("DELETE FROM meta WHERE key = ?1", [key])?,
        };
        Ok(())
    }

    /// Replaces the whole log atomically: a synced project adopting the shared order.
    pub fn replace(&mut self, ops: &[Op]) -> Result<(), StoreError> {
        let transaction = self.connection.transaction()?;
        transaction.execute("DELETE FROM ops", [])?;
        {
            let mut insert = transaction.prepare("INSERT INTO ops (id, op) VALUES (?1, ?2)")?;
            for op in ops {
                insert.execute(params![op.id.to_string(), serde_json::to_string(op)?])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }
}

impl ProjectStore for SqliteStore {
    fn append(&mut self, ops: &[Op]) -> Result<(), StoreError> {
        let transaction = self.connection.transaction()?;
        {
            let mut insert = transaction.prepare("INSERT INTO ops (id, op) VALUES (?1, ?2)")?;
            for op in ops {
                insert.execute(params![op.id.to_string(), serde_json::to_string(op)?])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    fn load(&self) -> Result<Vec<Op>, StoreError> {
        let mut select = self.connection.prepare("SELECT op FROM ops ORDER BY seq")?;
        let rows = select.query_map([], |row| row.get::<_, String>(0))?;
        rows.map(|json| Ok(serde_json::from_str(&json?)?)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::{Analyst, Change, Id, PhenomenologyId, Project};

    #[test]
    fn a_saved_project_replays_identically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.squiggle");
        let analyst = Analyst("ana".to_string());

        let (mut project, mut ops) = Project::create("p", &analyst);
        let sensor = Id::generate();
        ops.push(project.commit(&analyst, Change::AddSensor { id: sensor, code: "BHZ".to_string() }).unwrap());
        let mut store = SqliteStore::create(&path).unwrap();
        store.append(&ops).unwrap();

        // Appending later, after reopening, extends the same log.
        let assign = project
            .commit(&analyst, Change::AssignPhenomenology { sensor, phenomenology: PhenomenologyId::new("seismic") })
            .unwrap();
        drop(store);
        let mut store = SqliteStore::open(&path).unwrap();
        store.append(&[assign]).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(Project::replay(&loaded).unwrap(), project);
    }

    #[test]
    fn logs_can_be_replaced_and_meta_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.squiggle");
        let analyst = Analyst("ana".to_string());
        let (project, ops) = Project::create("p", &analyst);
        let mut store = SqliteStore::create(&path).unwrap();
        store.append(&ops).unwrap();
        store.set_meta("remote", Some("x")).unwrap();
        store.replace(&ops[..1]).unwrap();
        assert_eq!(store.load().unwrap(), ops[..1]);
        store.replace(&ops).unwrap();
        assert_eq!(Project::replay(&store.load().unwrap()).unwrap(), project);
        assert_eq!(store.meta("remote").unwrap().as_deref(), Some("x"));
        store.set_meta("remote", None).unwrap();
        assert_eq!(store.meta("remote").unwrap(), None);
    }

    #[test]
    fn foreign_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("other.db");
        Connection::open(&path).unwrap().execute_batch("CREATE TABLE x (y);").unwrap();
        assert!(matches!(SqliteStore::open(&path), Err(StoreError::Format(_))));
    }
}
