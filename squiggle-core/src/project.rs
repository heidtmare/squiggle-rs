//! A project is the replay of an append-only operation log.
//!
//! Every edit is an [`Op`]; [`Project`] state is a projection of the log.
//! Persisting only ops keeps stores mergeable later: sync exchanges ops.

use core::fmt;

use crate::id::Id;
use crate::phenomenology::{self, Phenomenology, PhenomenologyId};
use crate::sensor::{Analyst, Assignment, ClockCorrection, Provenance, Sensor, Trace};
use crate::time::Timestamp;

/// Project file format identifier.
pub const FORMAT: &str = "heidtmare.squiggle.project/1";

/// One logged edit.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Op {
    pub id: Id,
    pub analyst: Analyst,
    pub at: Timestamp,
    pub change: Change,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Change {
    /// Must be the first op.
    CreateProject { id: Id, name: String },
    RenameProject { name: String },
    /// Zero of the plot's time axis.
    SetReference { reference: Timestamp },
    /// Adds a phenomenology, or replaces one with a higher version.
    DefinePhenomenology(Phenomenology),
    AddSensor { id: Id, code: String },
    /// Recorded as manual, by the op's analyst.
    AssignPhenomenology { sensor: Id, phenomenology: PhenomenologyId },
    SetClockCorrection { sensor: Id, clock: ClockCorrection },
    AddTrace(Trace),
    RemoveTrace { trace: Id },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ApplyError {
    NotCreated,
    AlreadyCreated,
    DuplicateId(Id),
    DuplicateCode(String),
    UnknownSensor(Id),
    UnknownTrace(Id),
    UnknownPhenomenology(PhenomenologyId),
    StaleVersion { phenomenology: PhenomenologyId, version: u32, current: u32 },
    InvalidSampleRate(f64),
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCreated => write!(f, "the log does not start with CreateProject"),
            Self::AlreadyCreated => write!(f, "the project already exists"),
            Self::DuplicateId(id) => write!(f, "id {id} is already in use"),
            Self::DuplicateCode(code) => write!(f, "sensor code {code} is already in use"),
            Self::UnknownSensor(id) => write!(f, "no sensor {id}"),
            Self::UnknownTrace(id) => write!(f, "no trace {id}"),
            Self::UnknownPhenomenology(id) => write!(f, "no phenomenology {id}"),
            Self::StaleVersion { phenomenology, version, current } => {
                write!(f, "{phenomenology} v{version} does not supersede v{current}")
            }
            Self::InvalidSampleRate(rate) => write!(f, "invalid sample rate {rate}"),
        }
    }
}

impl std::error::Error for ApplyError {}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Project {
    pub id: Option<Id>,
    pub name: String,
    pub reference: Option<Timestamp>,
    phenomenologies: Vec<Phenomenology>,
    sensors: Vec<Sensor>,
    traces: Vec<Trace>,
}

impl Project {
    /// A new project with the built-in phenomenologies, and the ops that created it.
    pub fn create(name: impl Into<String>, analyst: &Analyst) -> (Self, Vec<Op>) {
        let mut project = Self::default();
        let mut changes = vec![Change::CreateProject { id: Id::generate(), name: name.into() }];
        changes.extend(phenomenology::builtins().into_iter().map(Change::DefinePhenomenology));
        let ops = changes
            .into_iter()
            .map(|change| project.commit(analyst, change).expect("built-ins are consistent"))
            .collect();
        (project, ops)
    }

    /// Rebuilds a project from its log.
    pub fn replay<'a>(ops: impl IntoIterator<Item = &'a Op>) -> Result<Self, ApplyError> {
        let mut project = Self::default();
        for op in ops {
            project.apply(op)?;
        }
        Ok(project)
    }

    /// Stamps `change` as a new op by `analyst`, applies it, and returns it for persisting.
    pub fn commit(&mut self, analyst: &Analyst, change: Change) -> Result<Op, ApplyError> {
        let op = Op { id: Id::generate(), analyst: analyst.clone(), at: Timestamp::now(), change };
        self.apply(&op)?;
        Ok(op)
    }

    /// Applies an op. On error the project is unchanged.
    pub fn apply(&mut self, op: &Op) -> Result<(), ApplyError> {
        if self.id.is_none() && !matches!(op.change, Change::CreateProject { .. }) {
            return Err(ApplyError::NotCreated);
        }

        match &op.change {
            Change::CreateProject { id, name } => {
                if self.id.is_some() {
                    return Err(ApplyError::AlreadyCreated);
                }
                self.id = Some(*id);
                self.name.clone_from(name);
            }
            Change::RenameProject { name } => self.name.clone_from(name),
            Change::SetReference { reference } => self.reference = Some(*reference),
            Change::DefinePhenomenology(definition) => {
                match self.phenomenologies.iter_mut().find(|p| p.id == definition.id) {
                    Some(existing) if definition.version <= existing.version => {
                        return Err(ApplyError::StaleVersion {
                            phenomenology: definition.id.clone(),
                            version: definition.version,
                            current: existing.version,
                        });
                    }
                    Some(existing) => *existing = definition.clone(),
                    None => self.phenomenologies.push(definition.clone()),
                }
            }
            Change::AddSensor { id, code } => {
                if self.sensor(*id).is_some() || self.trace(*id).is_some() {
                    return Err(ApplyError::DuplicateId(*id));
                }
                if self.sensor_by_code(code).is_some() {
                    return Err(ApplyError::DuplicateCode(code.clone()));
                }
                self.sensors.push(Sensor {
                    id: *id,
                    code: code.clone(),
                    assignment: None,
                    location: None,
                    clock: ClockCorrection::default(),
                });
            }
            Change::AssignPhenomenology { sensor, phenomenology } => {
                let version = self
                    .phenomenology(phenomenology)
                    .ok_or_else(|| ApplyError::UnknownPhenomenology(phenomenology.clone()))?
                    .version;
                let sensor = self.sensor_mut(*sensor)?;
                sensor.assignment = Some(Assignment {
                    phenomenology: phenomenology.clone(),
                    version,
                    provenance: Provenance::Manual { analyst: op.analyst.clone(), at: op.at },
                });
            }
            Change::SetClockCorrection { sensor, clock } => self.sensor_mut(*sensor)?.clock = *clock,
            Change::AddTrace(trace) => {
                if self.sensor(trace.sensor).is_none() {
                    return Err(ApplyError::UnknownSensor(trace.sensor));
                }
                if self.trace(trace.id).is_some() || self.sensor(trace.id).is_some() {
                    return Err(ApplyError::DuplicateId(trace.id));
                }
                if !(trace.sample_rate.is_finite() && trace.sample_rate > 0.0) {
                    return Err(ApplyError::InvalidSampleRate(trace.sample_rate));
                }
                self.traces.push(trace.clone());
            }
            Change::RemoveTrace { trace } => {
                let index = self
                    .traces
                    .iter()
                    .position(|t| t.id == *trace)
                    .ok_or(ApplyError::UnknownTrace(*trace))?;
                self.traces.remove(index);
            }
        }
        Ok(())
    }

    pub fn phenomenologies(&self) -> &[Phenomenology] {
        &self.phenomenologies
    }

    pub fn phenomenology(&self, id: &PhenomenologyId) -> Option<&Phenomenology> {
        self.phenomenologies.iter().find(|p| &p.id == id)
    }

    pub fn sensors(&self) -> &[Sensor] {
        &self.sensors
    }

    pub fn sensor(&self, id: Id) -> Option<&Sensor> {
        self.sensors.iter().find(|s| s.id == id)
    }

    pub fn sensor_by_code(&self, code: &str) -> Option<&Sensor> {
        self.sensors.iter().find(|s| s.code == code)
    }

    fn sensor_mut(&mut self, id: Id) -> Result<&mut Sensor, ApplyError> {
        self.sensors.iter_mut().find(|s| s.id == id).ok_or(ApplyError::UnknownSensor(id))
    }

    pub fn traces(&self) -> &[Trace] {
        &self.traces
    }

    pub fn trace(&self, id: Id) -> Option<&Trace> {
        self.traces.iter().find(|t| t.id == id)
    }

    /// The trace's phenomenology, or `None` while its sensor is unassigned.
    /// Processing runs only on traces where this is `Some`.
    pub fn trace_phenomenology(&self, trace: &Trace) -> Option<&Phenomenology> {
        let assignment = self.sensor(trace.sensor)?.assignment.as_ref()?;
        self.phenomenology(&assignment.phenomenology)
    }

    /// Clock-corrected start of a trace.
    pub fn corrected_start(&self, trace: &Trace) -> Timestamp {
        match (self.sensor(trace.sensor), self.reference) {
            (Some(sensor), Some(reference)) => sensor.clock.apply(trace.start, reference),
            (Some(sensor), None) => sensor.clock.apply(trace.start, trace.start),
            (None, _) => trace.start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::TraceSource;

    fn analyst() -> Analyst {
        Analyst("ana".to_string())
    }

    fn trace(sensor: Id) -> Trace {
        Trace {
            id: Id::generate(),
            sensor,
            start: Timestamp::new(1_000, 0.0),
            sample_rate: 100.0,
            len: 10,
            source: TraceSource::File {
                path: "a.wav".to_string(),
                format: "wav".to_string(),
                channel: 0,
                content_hash: "00".to_string(),
            },
        }
    }

    #[test]
    fn a_new_project_carries_the_builtins() {
        let (project, ops) = Project::create("p", &analyst());
        assert_eq!(project.phenomenologies().len(), phenomenology::builtins().len());
        assert_eq!(ops.len(), 1 + phenomenology::builtins().len());
    }

    #[test]
    fn replaying_the_log_reproduces_the_project() {
        let (mut project, mut ops) = Project::create("p", &analyst());
        let sensor = Id::generate();
        for change in [
            Change::AddSensor { id: sensor, code: "IU.ANMO.00.BHZ".to_string() },
            Change::AssignPhenomenology { sensor, phenomenology: PhenomenologyId::new("seismic") },
            Change::AddTrace(trace(sensor)),
        ] {
            ops.push(project.commit(&analyst(), change).unwrap());
        }
        assert_eq!(Project::replay(&ops), Ok(project));
    }

    #[test]
    fn unassigned_sensors_block_processing() {
        let (mut project, _) = Project::create("p", &analyst());
        let sensor = Id::generate();
        project.commit(&analyst(), Change::AddSensor { id: sensor, code: "mic".to_string() }).unwrap();
        let t = trace(sensor);
        project.commit(&analyst(), Change::AddTrace(t.clone())).unwrap();
        assert!(project.trace_phenomenology(&t).is_none());

        project
            .commit(&analyst(), Change::AssignPhenomenology { sensor, phenomenology: PhenomenologyId::new("acoustic") })
            .unwrap();
        assert_eq!(project.trace_phenomenology(&t).map(|p| p.id.as_str()), Some("acoustic"));
    }

    #[test]
    fn assignments_record_analyst_and_version() {
        let (mut project, _) = Project::create("p", &analyst());
        let sensor = Id::generate();
        project.commit(&analyst(), Change::AddSensor { id: sensor, code: "s".to_string() }).unwrap();
        let op = project
            .commit(&analyst(), Change::AssignPhenomenology { sensor, phenomenology: PhenomenologyId::new("rf") })
            .unwrap();
        let assignment = project.sensor(sensor).unwrap().assignment.clone().unwrap();
        assert_eq!(assignment.version, 1);
        assert_eq!(assignment.provenance, Provenance::Manual { analyst: analyst(), at: op.at });
    }

    #[test]
    fn invalid_changes_are_rejected_and_leave_the_project_unchanged() {
        let (mut project, _) = Project::create("p", &analyst());
        let before = project.clone();
        let ghost = Id::generate();
        assert_eq!(project.commit(&analyst(), Change::AddTrace(trace(ghost))), Err(ApplyError::UnknownSensor(ghost)));
        assert!(matches!(
            project.commit(&analyst(), Change::AssignPhenomenology { sensor: ghost, phenomenology: PhenomenologyId::new("x") }),
            Err(ApplyError::UnknownPhenomenology(_))
        ));
        assert!(matches!(
            project.commit(&analyst(), Change::DefinePhenomenology(phenomenology::builtins().remove(0))),
            Err(ApplyError::StaleVersion { .. })
        ));
        assert_eq!(project, before);
    }

    #[test]
    fn duplicate_sensor_codes_are_rejected() {
        let (mut project, _) = Project::create("p", &analyst());
        let add = |project: &mut Project| {
            project.commit(&analyst(), Change::AddSensor { id: Id::generate(), code: "s".to_string() })
        };
        add(&mut project).unwrap();
        assert_eq!(add(&mut project), Err(ApplyError::DuplicateCode("s".to_string())));
    }

    #[test]
    fn a_log_must_start_with_create() {
        let op = Op {
            id: Id::generate(),
            analyst: analyst(),
            at: Timestamp::now(),
            change: Change::RenameProject { name: "x".to_string() },
        };
        assert_eq!(Project::replay([&op]), Err(ApplyError::NotCreated));
    }
}
