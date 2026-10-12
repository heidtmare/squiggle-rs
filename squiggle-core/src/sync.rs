//! Merging op logs from several analysts.
//!
//! A shared log has one order, the server's. A client pushes the ops it made
//! since it last pulled, with `base`: how much of the shared log it had seen.
//! Ops appended after `base` are concurrent with its ops. An op whose
//! [`Field`]s overlap a concurrent op's is resolved per field:
//!
//! - a protected field (an analyst's judgement: a phenomenology assignment,
//!   a locked alignment, a manual pick or origin, a reviewed, rejected,
//!   labelled or manual event) raises a [`Conflict`] for a human to settle;
//! - otherwise the later edit (by `at`, then id) wins: an older incoming op
//!   is dropped as superseded.
//!
//! An op that no longer applies (its event was removed, say) is a conflict too.

use std::collections::HashMap;

use crate::id::Id;
use crate::phenomenology::PhenomenologyId;
use crate::project::{Change, Op, Project};
use crate::taxonomy::LabelId;

/// One editable value of the project.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    ProjectName,
    Reference,
    /// The project-level origin; an event's is [`Field::Origin`].
    ProjectOrigin,
    Phenomenology(PhenomenologyId),
    /// Creation and code.
    Sensor(Id),
    Assignment(Id),
    Clock(Id),
    Location(Id),
    /// Creation and removal.
    Trace(Id),
    Recording(Id),
    Processing(Id),
    Alignment { event: Option<Id>, trace: Id },
    Pick(Id),
    PickSlot { trace: Id, phase: String, event: Option<Id> },
    Origin(Id),
    /// Window, members, status, label.
    Event(Id),
    Predictions(Id),
    Label(LabelId),
    Model(String),
    Template(Id),
}

/// The fields `change` writes, read against the state it applies to.
pub fn fields(change: &Change, project: &Project) -> Vec<Field> {
    let slot = |p: &crate::sensor::Pick| Field::PickSlot { trace: p.trace, phase: p.phase.clone(), event: p.event };
    match change {
        Change::CreateProject { .. } | Change::RenameProject { .. } => vec![Field::ProjectName],
        Change::SetReference { .. } => vec![Field::Reference],
        Change::DefinePhenomenology(p) => vec![Field::Phenomenology(p.id.clone())],
        Change::AddSensor { id, .. } => vec![Field::Sensor(*id)],
        Change::AssignPhenomenology { sensor, .. } => vec![Field::Assignment(*sensor)],
        Change::SetClockCorrection { sensor, .. } => vec![Field::Clock(*sensor)],
        Change::SetSensorLocation { sensor, .. } => vec![Field::Location(*sensor)],
        Change::AddTrace(trace) => vec![Field::Trace(trace.id)],
        Change::RemoveTrace { trace } => vec![Field::Trace(*trace), Field::Recording(*trace), Field::Processing(*trace)],
        Change::ExtendTrace { trace, .. } => vec![Field::Recording(*trace)],
        Change::SetProcessing { trace, .. } => vec![Field::Processing(*trace)],
        Change::SetAlignment { trace, event, .. } => vec![Field::Alignment { event: *event, trace: *trace }],
        Change::SetPick { pick, .. } => vec![Field::Pick(pick.id), slot(pick)],
        Change::RemovePick { pick } => {
            let mut out = vec![Field::Pick(*pick)];
            out.extend(project.pick(*pick).map(slot));
            out
        }
        Change::SetOrigin { event: None, .. } => vec![Field::ProjectOrigin],
        Change::SetOrigin { event: Some(event), .. } => vec![Field::Origin(*event)],
        Change::AddEvent(event) | Change::SetEvent(event) => vec![Field::Event(event.id), Field::Predictions(event.id)],
        Change::RemoveEvent { event } => vec![Field::Event(*event), Field::Predictions(*event), Field::Origin(*event)],
        Change::SetPredictions { event, .. } => vec![Field::Predictions(*event)],
        Change::DefineLabel(label) => vec![Field::Label(label.id.clone())],
        Change::RemoveLabel { label } => vec![Field::Label(label.clone())],
        Change::SetModel(model) => vec![Field::Model(model.id.clone())],
        Change::RemoveModel { model } => vec![Field::Model(model.clone())],
        Change::SetTemplate(template) => vec![Field::Template(template.id)],
        Change::RemoveTemplate { template } => vec![Field::Template(*template)],
    }
}

/// Whether `field` holds an analyst's judgement in `project`.
pub fn is_protected(field: &Field, project: &Project) -> bool {
    match field {
        Field::Assignment(sensor) => project.sensor(*sensor).is_some_and(|s| s.assignment.is_some()),
        Field::Alignment { event, trace } => project.is_locked(*event, *trace),
        Field::Pick(id) => project.pick(*id).is_some_and(|p| p.provenance.is_manual()),
        Field::PickSlot { trace, phase, event } => {
            project.event_picks(*event, *trace).any(|p| &p.phase == phase && p.provenance.is_manual())
        }
        Field::ProjectOrigin => project.origin.as_ref().is_some_and(|o| o.provenance.is_manual()),
        Field::Origin(event) => project.origin_of(Some(*event)).is_some_and(|o| o.provenance.is_manual()),
        Field::Event(id) => project.event(*id).is_some_and(|e| !Project::replaceable(e) || e.label.is_some()),
        _ => false,
    }
}

/// Plain words for a field.
pub fn describe(field: &Field, project: &Project) -> String {
    let sensor = |id: &Id| project.sensor(*id).map_or_else(|| id.to_string(), |s| s.code.clone());
    let trace = |id: &Id| project.trace(*id).and_then(|t| project.sensor(t.sensor)).map_or_else(|| format!("trace {id}"), |s| s.code.clone());
    let event = |id: &Id| project.event(*id).map_or_else(|| format!("event {id}"), |e| format!("event at {}", e.start));
    match field {
        Field::ProjectName => "the project name".to_string(),
        Field::Reference => "the reference time".to_string(),
        Field::ProjectOrigin => "the project origin".to_string(),
        Field::Phenomenology(id) => format!("phenomenology {id}"),
        Field::Sensor(id) => format!("sensor {}", sensor(id)),
        Field::Assignment(id) => format!("the phenomenology of {}", sensor(id)),
        Field::Clock(id) => format!("the clock correction of {}", sensor(id)),
        Field::Location(id) => format!("the location of {}", sensor(id)),
        Field::Trace(id) => format!("the trace of {}", trace(id)),
        Field::Recording(id) => format!("the recording of {}", trace(id)),
        Field::Processing(id) => format!("the processing of {}", trace(id)),
        Field::Alignment { event: Some(e), trace: t } => format!("the alignment of {} in the {}", trace(t), event(e)),
        Field::Alignment { event: None, trace: t } => format!("the alignment of {}", trace(t)),
        Field::Pick(id) => match project.pick(*id) {
            Some(p) => format!("the {} pick on {}", p.phase_label(), trace(&p.trace)),
            None => format!("pick {id}"),
        },
        Field::PickSlot { trace: t, phase, .. } => {
            format!("the {} pick on {}", if phase.is_empty() { "onset" } else { phase }, trace(t))
        }
        Field::Origin(id) => format!("the origin of the {}", event(id)),
        Field::Event(id) => format!("the {}", event(id)),
        Field::Predictions(id) => format!("the predictions for the {}", event(id)),
        Field::Label(id) => format!("label {id}"),
        Field::Model(id) => format!("model {id}"),
        Field::Template(id) => project.template(*id).map_or_else(|| format!("template {id}"), |t| format!("template {}", t.name)),
    }
}

/// What became of one pushed op.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Outcome {
    /// Appended to the shared log.
    Accepted,
    /// Already in the shared log (a retried push).
    Duplicate,
    /// A later concurrent edit of the same fields won.
    Superseded { by: Id },
    /// Not applied; an analyst must settle it.
    Conflict(Box<Conflict>),
}

/// An op held back from the shared log, and why.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conflict {
    pub mine: Op,
    /// Concurrent ops it clashed with; empty when it simply no longer applies.
    pub theirs: Vec<Op>,
    pub reason: String,
}

/// A shared log, its project, and each op's fields: the server's side of a merge.
#[derive(Clone, Debug)]
pub struct Shared {
    project: Project,
    log: Vec<Op>,
    touched: Vec<Vec<Field>>,
    index: HashMap<Id, usize>,
}

impl Shared {
    pub fn new(ops: Vec<Op>) -> Result<Self, crate::project::ApplyError> {
        let mut shared = Self { project: Project::default(), log: Vec::new(), touched: Vec::new(), index: HashMap::new() };
        for op in ops {
            let touched = fields(&op.change, &shared.project);
            shared.project.apply(&op)?;
            shared.append(op, touched);
        }
        Ok(shared)
    }

    fn append(&mut self, op: Op, touched: Vec<Field>) {
        self.index.insert(op.id, self.log.len());
        self.log.push(op);
        self.touched.push(touched);
    }

    pub fn head(&self) -> usize {
        self.log.len()
    }

    pub fn log(&self) -> &[Op] {
        &self.log
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    /// Merges ops made after seeing the first `base` shared ops. Accepted ops
    /// are appended in order; the result has one outcome per op.
    pub fn push(&mut self, base: usize, ops: &[Op]) -> Vec<Outcome> {
        let head = self.log.len();
        let base = base.min(head);
        ops.iter().map(|op| self.merge(base..head, op)).collect()
    }

    fn merge(&mut self, concurrent: core::ops::Range<usize>, op: &Op) -> Outcome {
        if self.index.contains_key(&op.id) {
            return Outcome::Duplicate;
        }
        let mine = fields(&op.change, &self.project);
        let mut theirs = Vec::new();
        let mut clashing = Vec::new();
        for i in concurrent {
            let shared: Vec<&Field> = mine.iter().filter(|f| self.touched[i].contains(f)).collect();
            if !shared.is_empty() {
                theirs.push(i);
                clashing.extend(shared);
            }
        }
        if !theirs.is_empty() {
            let mut trial = self.project.clone();
            let applies = trial.apply(op).is_ok();
            let protected = clashing.iter().find(|f| is_protected(f, &self.project) || (applies && is_protected(f, &trial)));
            if let Some(field) = protected {
                let mut by: Vec<String> = theirs.iter().map(|&i| self.log[i].analyst.to_string()).collect();
                by.dedup();
                return Outcome::Conflict(Box::new(Conflict {
                    mine: op.clone(),
                    theirs: theirs.iter().map(|&i| self.log[i].clone()).collect(),
                    reason: format!("{} was also changed by {}", describe(field, &self.project), by.join(", ")),
                }));
            }
            let latest = theirs.iter().map(|&i| &self.log[i]).max_by(|a, b| newer(a, b)).expect("non-empty");
            if newer(latest, op).is_gt() {
                return Outcome::Superseded { by: latest.id };
            }
        }
        match self.project.apply(op) {
            Ok(()) => {
                self.append(op.clone(), mine);
                Outcome::Accepted
            }
            Err(error) => Outcome::Conflict(Box::new(Conflict { mine: op.clone(), theirs: Vec::new(), reason: error.to_string() })),
        }
    }
}

/// Orders ops by `at`, then id.
fn newer(a: &Op, b: &Op) -> core::cmp::Ordering {
    a.at.partial_cmp(&b.at).unwrap_or(core::cmp::Ordering::Equal).then(a.id.cmp(&b.id))
}

/// Replays `confirmed` then reapplies `pending` on top: a client's state after
/// a pull. Pending ops that no longer apply are returned with the error.
pub fn rebase(confirmed: &[Op], pending: &[Op]) -> Result<(Project, Vec<Op>, Vec<Conflict>), crate::project::ApplyError> {
    let mut project = Project::replay(confirmed)?;
    let (mut kept, mut failed) = (Vec::new(), Vec::new());
    for op in pending {
        match project.apply(op) {
            Ok(()) => kept.push(op.clone()),
            Err(error) => failed.push(Conflict { mine: op.clone(), theirs: Vec::new(), reason: error.to_string() }),
        }
    }
    Ok((project, kept, failed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Processing;
    use crate::event::{Event, EventStatus, Member};
    use crate::sensor::{Analyst, Provenance, Trace, TraceSource};
    use crate::time::Timestamp;

    fn ana() -> Analyst {
        Analyst("ana".into())
    }

    fn bo() -> Analyst {
        Analyst("bo".into())
    }

    /// A shared project with one assigned trace and one automatic event.
    fn setup() -> (Shared, Id, Event) {
        let (mut project, mut ops) = Project::create("p", &ana());
        let sensor = Id::generate();
        let trace = Trace {
            id: Id::generate(),
            sensor,
            start: Timestamp::new(1_000, 0.0),
            sample_rate: 100.0,
            len: 1000,
            kind: Default::default(),
            source: TraceSource::File { path: "a.wav".into(), format: "wav".into(), channel: 0, content_hash: "00".into(), layout: None },
            processing: Processing::default(),
            alignment: None,
        };
        let event = Event {
            id: Id::generate(),
            start: Timestamp::new(1_001, 0.0),
            end: Timestamp::new(1_005, 0.0),
            members: vec![Member::new(trace.id)],
            origin: None,
            status: EventStatus::Auto,
            provenance: Provenance::Auto { stage: "detect".into(), config_hash: "x".into() },
            label: None,
            predictions: Vec::new(),
        };
        for change in [
            Change::AddSensor { id: sensor, code: "S".into() },
            Change::AssignPhenomenology { sensor, phenomenology: PhenomenologyId::new("seismic") },
            Change::AddTrace(trace.clone()),
            Change::AddEvent(event.clone()),
        ] {
            ops.push(project.commit(&ana(), change).unwrap());
        }
        (Shared::new(ops).unwrap(), trace.id, event)
    }

    /// `changes` made offline by `analyst` on a copy of the first `base` shared ops.
    fn offline(shared: &Shared, base: usize, analyst: &Analyst, changes: Vec<Change>) -> Vec<Op> {
        let mut project = Project::replay(&shared.log()[..base]).unwrap();
        changes.into_iter().map(|c| project.commit(analyst, c).unwrap()).collect()
    }

    #[test]
    fn edits_of_different_fields_merge() {
        let (mut shared, _, event) = setup();
        let base = shared.head();
        let a = offline(&shared, base, &ana(), vec![Change::RenameProject { name: "quarry study".into() }]);
        let label = Event { label: Some(LabelId::new("explosion")), status: EventStatus::Reviewed, ..event.clone() };
        let b = offline(&shared, base, &bo(), vec![Change::SetEvent(label.clone())]);
        assert_eq!(shared.push(base, &a), [Outcome::Accepted]);
        assert_eq!(shared.push(base, &b), [Outcome::Accepted]);
        assert_eq!(shared.project().name, "quarry study");
        assert_eq!(shared.project().event(event.id), Some(&label));
        // A retried push changes nothing.
        assert_eq!(shared.push(base, &b), [Outcome::Duplicate]);
        assert_eq!(shared.head(), base + 2);
    }

    #[test]
    fn plain_fields_keep_the_later_edit() {
        let (mut shared, _, _) = setup();
        let base = shared.head();
        let early = offline(&shared, base, &ana(), vec![Change::RenameProject { name: "early".into() }]);
        let mut late = offline(&shared, base, &bo(), vec![Change::RenameProject { name: "late".into() }]);
        late[0].at = early[0].at.add_seconds(1.0);
        // The later edit wins whichever arrives first.
        assert_eq!(shared.push(base, &late), [Outcome::Accepted]);
        assert_eq!(shared.push(base, &early), [Outcome::Superseded { by: late[0].id }]);
        assert_eq!(shared.project().name, "late");

        let (mut shared, _, _) = setup();
        assert_eq!(shared.push(base, &early), [Outcome::Accepted]);
        assert_eq!(shared.push(base, &late), [Outcome::Accepted]);
        assert_eq!(shared.project().name, "late");
    }

    #[test]
    fn reviewed_events_raise_conflicts() {
        let (mut shared, _, event) = setup();
        let base = shared.head();
        let quake = Event { label: Some(LabelId::new("earthquake")), status: EventStatus::Reviewed, ..event.clone() };
        let blast = Event { label: Some(LabelId::new("explosion")), status: EventStatus::Reviewed, ..event.clone() };
        let a = offline(&shared, base, &ana(), vec![Change::SetEvent(quake.clone())]);
        let b = offline(&shared, base, &bo(), vec![Change::SetEvent(blast)]);
        assert_eq!(shared.push(base, &a), [Outcome::Accepted]);
        let Ok::<[Outcome; 1], _>([Outcome::Conflict(conflict)]) = shared.push(base, &b).try_into() else { panic!() };
        assert_eq!(conflict.theirs, a);
        assert!(conflict.reason.contains("ana"), "{}", conflict.reason);
        assert_eq!(shared.project().event(event.id), Some(&quake));
        // Settled by re-pushing on the merged state: no longer concurrent.
        let retry = offline(&shared, shared.head(), &bo(), vec![conflict.mine.change.clone()]);
        assert_eq!(shared.push(shared.head(), &retry), [Outcome::Accepted]);
    }

    #[test]
    fn ops_that_no_longer_apply_are_conflicts() {
        let (mut shared, trace, event) = setup();
        let base = shared.head();
        let a = offline(&shared, base, &ana(), vec![Change::RemoveEvent { event: event.id }]);
        let pick = crate::sensor::Pick {
            id: Id::generate(),
            trace,
            time: Timestamp::new(1_002, 0.0),
            phase: "P".into(),
            uncertainty_s: 0.1,
            provenance: Provenance::Manual { analyst: bo(), at: Timestamp::now() },
            event: Some(event.id),
        };
        let b = offline(&shared, base, &bo(), vec![Change::SetPick { pick, revert: false }]);
        shared.push(base, &a);
        let Ok::<[Outcome; 1], _>([Outcome::Conflict(conflict)]) = shared.push(base, &b).try_into() else { panic!() };
        assert!(conflict.theirs.is_empty());
    }

    #[test]
    fn rebase_keeps_what_still_applies() {
        let (shared, _, event) = setup();
        let mut project = shared.project().clone();
        let pending = vec![
            project.commit(&bo(), Change::RenameProject { name: "x".into() }).unwrap(),
            project.commit(&bo(), Change::RemoveEvent { event: event.id }).unwrap(),
        ];
        let mut confirmed = shared.log().to_vec();
        let mut remote = shared.project().clone();
        confirmed.push(remote.commit(&ana(), Change::RemoveEvent { event: event.id }).unwrap());
        let (project, kept, failed) = rebase(&confirmed, &pending).unwrap();
        assert_eq!(project.name, "x");
        assert_eq!(kept, pending[..1]);
        assert_eq!(failed.len(), 1);
    }
}
