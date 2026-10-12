//! A project is the replay of an append-only operation log.
//!
//! Every edit is an [`Op`]; [`Project`] state is a projection of the log.
//! Persisting only ops keeps stores mergeable later: sync exchanges ops.

use core::fmt;

use crate::dsp::Processing;
use crate::event::{Event, EventStatus, Prediction};
use crate::id::Id;
use crate::locate::Origin;
use crate::model::ModelRef;
use crate::phenomenology::{self, Phenomenology, PhenomenologyId};
use crate::sensor::{Alignment, Analyst, Assignment, ClockCorrection, Location, Pick, Provenance, Sensor, Trace};
use crate::taxonomy::{self, Label, LabelId, Taxonomy};
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
    SetSensorLocation { sensor: Id, location: Option<Location> },
    AddTrace(Trace),
    /// Also removes the trace's picks and event memberships.
    RemoveTrace { trace: Id },
    /// Requires an assigned phenomenology and a filter valid at the trace's rate.
    SetProcessing { trace: Id, processing: Processing },
    /// The trace's alignment in `event` (it must be a member), or its global
    /// one. An automatic alignment is refused on a locked one unless `revert`
    /// (an undo restoring an earlier value).
    SetAlignment {
        trace: Id,
        alignment: Option<Alignment>,
        #[cfg_attr(feature = "serde", serde(default))]
        revert: bool,
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        event: Option<Id>,
    },
    /// Adds a pick, replacing any with its id or its trace, phase and event.
    /// An automatic pick needs an assigned phenomenology and never replaces a
    /// manual one unless `revert`.
    SetPick {
        pick: Pick,
        #[cfg_attr(feature = "serde", serde(default))]
        revert: bool,
    },
    RemovePick { pick: Id },
    /// The located source of `event`, or the project's outside any event.
    SetOrigin {
        origin: Option<Origin>,
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        event: Option<Id>,
    },
    /// Needs a window that ends after it starts, and known member traces.
    AddEvent(Event),
    /// Replaces the event with this id.
    SetEvent(Event),
    /// Also removes the event's picks.
    RemoveEvent { event: Id },
    /// Adds or replaces a label; bumps the taxonomy version.
    DefineLabel(Label),
    /// Refused while an event or a subtype uses it.
    RemoveLabel { label: LabelId },
    /// Replaces the event's predictions; leaves its status alone.
    SetPredictions { event: Id, predictions: Vec<Prediction> },
    /// Adds or replaces an imported model with this id.
    SetModel(ModelRef),
    RemoveModel { model: String },
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
    /// Processing needs an assigned phenomenology.
    Unassigned(Id),
    InvalidProcessing(String),
    Locked(Id),
    InvalidLocation,
    UnknownPick(Id),
    InvalidPick(String),
    /// An automatic pick would replace this manual one.
    LockedPick(Id),
    UnknownEvent(Id),
    NotMember { event: Id, trace: Id },
    InvalidEvent(String),
    UnknownLabel(LabelId),
    InvalidLabel(String),
    UnknownModel(String),
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
            Self::Unassigned(id) => write!(f, "trace {id} has no phenomenology assigned"),
            Self::InvalidProcessing(why) => write!(f, "invalid processing: {why}"),
            Self::Locked(id) => write!(f, "trace {id} has a locked alignment"),
            Self::InvalidLocation => write!(f, "latitude, longitude or elevation out of range"),
            Self::UnknownPick(id) => write!(f, "no pick {id}"),
            Self::InvalidPick(why) => write!(f, "invalid pick: {why}"),
            Self::LockedPick(id) => write!(f, "pick {id} is manual and locked"),
            Self::UnknownEvent(id) => write!(f, "no event {id}"),
            Self::NotMember { event, trace } => write!(f, "trace {trace} is not a member of event {event}"),
            Self::InvalidEvent(why) => write!(f, "invalid event: {why}"),
            Self::UnknownLabel(id) => write!(f, "no label {id}"),
            Self::InvalidLabel(why) => write!(f, "invalid label: {why}"),
            Self::UnknownModel(id) => write!(f, "no model {id}"),
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
    picks: Vec<Pick>,
    /// Located source outside any event.
    pub origin: Option<Origin>,
    events: Vec<Event>,
    taxonomy: Taxonomy,
    models: Vec<ModelRef>,
}

impl Project {
    /// A new project with the built-in phenomenologies and seed labels, and the ops that created it.
    pub fn create(name: impl Into<String>, analyst: &Analyst) -> (Self, Vec<Op>) {
        let mut project = Self::default();
        let mut changes = vec![Change::CreateProject { id: Id::generate(), name: name.into() }];
        changes.extend(phenomenology::builtins().into_iter().map(Change::DefinePhenomenology));
        changes.extend(taxonomy::seed().into_iter().map(Change::DefineLabel));
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
            Change::SetSensorLocation { sensor, location } => {
                if location.is_some_and(|l| !l.is_valid()) {
                    return Err(ApplyError::InvalidLocation);
                }
                self.sensor_mut(*sensor)?.location = *location;
            }
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
                self.picks.retain(|p| p.trace != *trace);
                for event in &mut self.events {
                    event.members.retain(|m| m.trace != *trace);
                }
            }
            Change::SetProcessing { trace, processing } => {
                let current = self.trace(*trace).ok_or(ApplyError::UnknownTrace(*trace))?;
                if !processing.is_identity() {
                    if self.trace_phenomenology(current).is_none() {
                        return Err(ApplyError::Unassigned(*trace));
                    }
                    processing
                        .validate(current.sample_rate)
                        .map_err(|e| ApplyError::InvalidProcessing(e.to_string()))?;
                }
                self.trace_mut(*trace)?.processing = *processing;
            }
            Change::SetAlignment { trace, alignment, revert, event } => {
                let current = self.trace(*trace).ok_or(ApplyError::UnknownTrace(*trace))?;
                if let Some(event) = event {
                    let e = self.event(*event).ok_or(ApplyError::UnknownEvent(*event))?;
                    if e.member(*trace).is_none() {
                        return Err(ApplyError::NotMember { event: *event, trace: *trace });
                    }
                }
                let automatic = alignment.as_ref().is_some_and(|a| !a.provenance.is_manual());
                if automatic && !revert {
                    if self.is_locked(*event, *trace) {
                        return Err(ApplyError::Locked(*trace));
                    }
                    if self.trace_phenomenology(current).is_none() {
                        return Err(ApplyError::Unassigned(*trace));
                    }
                }
                match event {
                    Some(event) => {
                        let member = self.event_mut(*event)?.member_mut(*trace).expect("checked above");
                        member.alignment = alignment.clone();
                    }
                    None => self.trace_mut(*trace)?.alignment = alignment.clone(),
                }
            }
            Change::SetPick { pick, revert } => {
                let trace = self.trace(pick.trace).ok_or(ApplyError::UnknownTrace(pick.trace))?;
                if !(pick.uncertainty_s.is_finite() && pick.uncertainty_s > 0.0) {
                    return Err(ApplyError::InvalidPick("uncertainty must be positive".to_string()));
                }
                if let Some(event) = pick.event
                    && self.event(event).is_none()
                {
                    return Err(ApplyError::UnknownEvent(event));
                }
                if !pick.provenance.is_manual() && !revert {
                    if self.trace_phenomenology(trace).is_none() {
                        return Err(ApplyError::Unassigned(pick.trace));
                    }
                    if let Some(manual) = self.replaced_picks(pick).find(|p| p.provenance.is_manual()) {
                        return Err(ApplyError::LockedPick(manual.id));
                    }
                }
                self.picks.retain(|p| !Self::replaces(pick, p));
                self.picks.push(pick.clone());
            }
            Change::RemovePick { pick } => {
                let index = self.picks.iter().position(|p| p.id == *pick).ok_or(ApplyError::UnknownPick(*pick))?;
                self.picks.remove(index);
            }
            Change::SetOrigin { origin, event } => match event {
                Some(event) => self.event_mut(*event)?.origin = origin.clone(),
                None => self.origin = origin.clone(),
            },
            Change::AddEvent(event) => {
                if self.event(event.id).is_some() {
                    return Err(ApplyError::DuplicateId(event.id));
                }
                self.validate_event(event)?;
                self.events.push(event.clone());
            }
            Change::SetEvent(event) => {
                self.validate_event(event)?;
                *self.event_mut(event.id)? = event.clone();
            }
            Change::RemoveEvent { event } => {
                let index = self.events.iter().position(|e| e.id == *event).ok_or(ApplyError::UnknownEvent(*event))?;
                self.events.remove(index);
                self.picks.retain(|p| p.event != Some(*event));
            }
            Change::DefineLabel(label) => {
                self.taxonomy.check(label).map_err(ApplyError::InvalidLabel)?;
                match self.taxonomy.labels.iter_mut().find(|l| l.id == label.id) {
                    Some(existing) => *existing = label.clone(),
                    None => self.taxonomy.labels.push(label.clone()),
                }
                self.taxonomy.version += 1;
            }
            Change::RemoveLabel { label } => {
                let index = self.taxonomy.labels.iter().position(|l| &l.id == label).ok_or_else(|| ApplyError::UnknownLabel(label.clone()))?;
                if self.events.iter().any(|e| e.label.as_ref() == Some(label)) {
                    return Err(ApplyError::InvalidLabel(format!("events are labelled {label}")));
                }
                if self.taxonomy.labels.iter().any(|l| l.parent.as_ref() == Some(label)) {
                    return Err(ApplyError::InvalidLabel(format!("{label} has subtypes")));
                }
                self.taxonomy.labels.remove(index);
                self.taxonomy.version += 1;
            }
            Change::SetPredictions { event, predictions } => self.event_mut(*event)?.predictions = predictions.clone(),
            Change::SetModel(model) => match self.models.iter_mut().find(|m| m.id == model.id) {
                Some(existing) => *existing = model.clone(),
                None => self.models.push(model.clone()),
            },
            Change::RemoveModel { model } => {
                let index = self.models.iter().position(|m| &m.id == model).ok_or_else(|| ApplyError::UnknownModel(model.clone()))?;
                self.models.remove(index);
            }
        }
        Ok(())
    }

    fn validate_event(&self, event: &Event) -> Result<(), ApplyError> {
        if event.end.seconds_since(event.start) <= 0.0 {
            return Err(ApplyError::InvalidEvent("the window must end after it starts".to_string()));
        }
        for (i, member) in event.members.iter().enumerate() {
            if self.trace(member.trace).is_none() {
                return Err(ApplyError::UnknownTrace(member.trace));
            }
            if event.members[..i].iter().any(|m| m.trace == member.trace) {
                return Err(ApplyError::InvalidEvent("a trace is listed twice".to_string()));
            }
        }
        if let Some(label) = &event.label
            && self.taxonomy.get(label).is_none()
        {
            return Err(ApplyError::UnknownLabel(label.clone()));
        }
        Ok(())
    }

    /// Whether setting `new` removes `old`: same id, or same trace, phase and event.
    fn replaces(new: &Pick, old: &Pick) -> bool {
        old.id == new.id || (old.trace == new.trace && old.phase == new.phase && old.event == new.event)
    }

    /// The changes that undo `change`, from the current state, applied in
    /// order. `None` for changes that cannot be undone yet.
    pub fn inverse(&self, change: &Change) -> Option<Vec<Change>> {
        let one = |c: Option<Change>| c.map(|c| vec![c]);
        match change {
            Change::RenameProject { .. } => one(Some(Change::RenameProject { name: self.name.clone() })),
            Change::SetReference { .. } => one(self.reference.map(|reference| Change::SetReference { reference })),
            Change::SetClockCorrection { sensor, .. } => {
                one(self.sensor(*sensor).map(|s| Change::SetClockCorrection { sensor: *sensor, clock: s.clock }))
            }
            Change::SetProcessing { trace, .. } => {
                one(self.trace(*trace).map(|t| Change::SetProcessing { trace: *trace, processing: t.processing }))
            }
            Change::SetAlignment { trace, event, .. } => {
                self.trace(*trace)?;
                one(Some(Change::SetAlignment {
                    trace: *trace,
                    alignment: self.alignment(*event, *trace).cloned(),
                    revert: true,
                    event: *event,
                }))
            }
            Change::SetSensorLocation { sensor, .. } => {
                one(self.sensor(*sensor).map(|s| Change::SetSensorLocation { sensor: *sensor, location: s.location }))
            }
            Change::SetPick { pick, .. } => {
                let replaced: Vec<&Pick> = self.replaced_picks(pick).collect();
                match replaced.as_slice() {
                    [] => one(Some(Change::RemovePick { pick: pick.id })),
                    // Matches the new pick by id or slot, so it replaces it back.
                    [old] => one(Some(Change::SetPick { pick: (*old).clone(), revert: true })),
                    _ => None,
                }
            }
            Change::RemovePick { pick } => one(self.pick(*pick).map(|p| Change::SetPick { pick: p.clone(), revert: true })),
            Change::SetOrigin { event, .. } => {
                let origin = match event {
                    Some(event) => self.event(*event)?.origin.clone(),
                    None => self.origin.clone(),
                };
                one(Some(Change::SetOrigin { origin, event: *event }))
            }
            Change::AddEvent(event) => one(Some(Change::RemoveEvent { event: event.id })),
            Change::SetEvent(event) => one(self.event(event.id).map(|e| Change::SetEvent(e.clone()))),
            Change::RemoveEvent { event } => {
                let mut changes = vec![Change::AddEvent(self.event(*event)?.clone())];
                changes.extend(
                    self.picks.iter().filter(|p| p.event == Some(*event)).map(|p| Change::SetPick { pick: p.clone(), revert: true }),
                );
                Some(changes)
            }
            Change::DefineLabel(label) => one(Some(match self.taxonomy.get(&label.id) {
                Some(old) => Change::DefineLabel(old.clone()),
                None => Change::RemoveLabel { label: label.id.clone() },
            })),
            Change::RemoveLabel { label } => one(self.taxonomy.get(label).map(|l| Change::DefineLabel(l.clone()))),
            Change::SetPredictions { event, .. } => {
                one(self.event(*event).map(|e| Change::SetPredictions { event: *event, predictions: e.predictions.clone() }))
            }
            Change::SetModel(model) => one(Some(match self.model(&model.id) {
                Some(old) => Change::SetModel(old.clone()),
                None => Change::RemoveModel { model: model.id.clone() },
            })),
            Change::RemoveModel { model } => one(self.model(model).map(|m| Change::SetModel(m.clone()))),
            _ => None,
        }
    }

    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }

    pub fn models(&self) -> &[ModelRef] {
        &self.models
    }

    pub fn model(&self, id: &str) -> Option<&ModelRef> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Seed labels for a project saved before labels existed.
    pub fn seed_taxonomy(&self) -> Vec<Change> {
        if self.taxonomy.version > 0 {
            return Vec::new();
        }
        taxonomy::seed().into_iter().map(Change::DefineLabel).collect()
    }

    /// Picks a [`Change::SetPick`] of `pick` would replace.
    fn replaced_picks<'a>(&'a self, pick: &'a Pick) -> impl Iterator<Item = &'a Pick> + 'a {
        self.picks.iter().filter(|p| Self::replaces(pick, p))
    }

    pub fn picks(&self) -> &[Pick] {
        &self.picks
    }

    pub fn pick(&self, id: Id) -> Option<&Pick> {
        self.picks.iter().find(|p| p.id == id)
    }

    /// Every pick on `trace`, in any event.
    pub fn trace_picks(&self, trace: Id) -> impl Iterator<Item = &Pick> {
        self.picks.iter().filter(move |p| p.trace == trace)
    }

    /// Picks on `trace` in `event` (`None`: outside any event).
    pub fn event_picks(&self, event: Option<Id>, trace: Id) -> impl Iterator<Item = &Pick> {
        self.picks.iter().filter(move |p| p.trace == trace && p.event == event)
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn event(&self, id: Id) -> Option<&Event> {
        self.events.iter().find(|e| e.id == id)
    }

    fn event_mut(&mut self, id: Id) -> Result<&mut Event, ApplyError> {
        self.events.iter_mut().find(|e| e.id == id).ok_or(ApplyError::UnknownEvent(id))
    }

    /// A trace's alignment in `event`, or its global one.
    pub fn alignment(&self, event: Option<Id>, trace: Id) -> Option<&Alignment> {
        match event {
            Some(event) => self.event(event)?.member(trace)?.alignment.as_ref(),
            None => self.trace(trace)?.alignment.as_ref(),
        }
    }

    pub fn offset(&self, event: Option<Id>, trace: Id) -> f64 {
        self.alignment(event, trace).map_or(0.0, |a| a.offset_s)
    }

    pub fn is_locked(&self, event: Option<Id>, trace: Id) -> bool {
        self.alignment(event, trace).is_some_and(|a| a.locked)
    }

    /// The located source of `event`, or the project's.
    pub fn origin_of(&self, event: Option<Id>) -> Option<&Origin> {
        match event {
            Some(event) => self.event(event)?.origin.as_ref(),
            None => self.origin.as_ref(),
        }
    }

    /// Statuses that automatic detection may replace.
    pub fn replaceable(event: &Event) -> bool {
        event.status == EventStatus::Auto && !event.provenance.is_manual()
    }

    /// A recorded time on `trace`'s clock, corrected.
    pub fn correct(&self, trace: &Trace, recorded: Timestamp) -> Timestamp {
        match (self.sensor(trace.sensor), self.reference) {
            (Some(sensor), Some(reference)) => sensor.clock.apply(recorded, reference),
            (Some(sensor), None) => sensor.clock.apply(recorded, trace.start),
            (None, _) => recorded,
        }
    }

    /// Clock-corrected arrival time of a pick: what location uses.
    pub fn pick_time(&self, pick: &Pick) -> Option<Timestamp> {
        self.trace(pick.trace).map(|t| self.correct(t, pick.time))
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

    fn trace_mut(&mut self, id: Id) -> Result<&mut Trace, ApplyError> {
        self.traces.iter_mut().find(|t| t.id == id).ok_or(ApplyError::UnknownTrace(id))
    }

    /// The trace's phenomenology, or `None` while its sensor is unassigned.
    /// Processing runs only on traces where this is `Some`.
    pub fn trace_phenomenology(&self, trace: &Trace) -> Option<&Phenomenology> {
        let assignment = self.sensor(trace.sensor)?.assignment.as_ref()?;
        self.phenomenology(&assignment.phenomenology)
    }

    /// Clock-corrected start of a trace.
    pub fn corrected_start(&self, trace: &Trace) -> Timestamp {
        self.correct(trace, trace.start)
    }

    /// Clock-corrected start plus the alignment offset in `event`: where the
    /// trace is drawn and analysed.
    pub fn aligned_start(&self, event: Option<Id>, trace: &Trace) -> Timestamp {
        self.corrected_start(trace).add_seconds(self.offset(event, trace.id))
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
            kind: Default::default(),
            source: TraceSource::File {
                path: "a.wav".to_string(),
                format: "wav".to_string(),
                channel: 0,
                content_hash: "00".to_string(),
                layout: None,
            },
            processing: Processing::default(),
            alignment: None,
        }
    }

    #[test]
    fn a_new_project_carries_the_builtins() {
        let (project, ops) = Project::create("p", &analyst());
        assert_eq!(project.phenomenologies().len(), phenomenology::builtins().len());
        assert_eq!(ops.len(), 1 + phenomenology::builtins().len() + taxonomy::seed().len());
        assert_eq!(project.taxonomy().labels, taxonomy::seed());
        assert!(project.seed_taxonomy().is_empty());
    }

    #[test]
    fn labels_validate_and_undo() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let quarry = Label { id: LabelId::new("quarry"), name: "Quarry".into(), parent: Some(LabelId::new("explosion")), expects: vec![] };
        let version = project.taxonomy().version;
        let undo = project.inverse(&Change::DefineLabel(quarry.clone())).unwrap();
        project.commit(&analyst(), Change::DefineLabel(quarry.clone())).unwrap();
        assert_eq!(project.taxonomy().version, version + 1);

        let labelled = Event { label: Some(quarry.id.clone()), ..event(&[t.id]) };
        project.commit(&analyst(), Change::AddEvent(labelled.clone())).unwrap();
        let remove = Change::RemoveLabel { label: quarry.id.clone() };
        assert!(matches!(project.commit(&analyst(), remove.clone()), Err(ApplyError::InvalidLabel(_))));
        let parent = Change::RemoveLabel { label: LabelId::new("explosion") };
        assert!(matches!(project.commit(&analyst(), parent), Err(ApplyError::InvalidLabel(_))));
        let unknown = Event { label: Some(LabelId::new("nope")), ..event(&[t.id]) };
        assert!(matches!(project.commit(&analyst(), Change::AddEvent(unknown)), Err(ApplyError::UnknownLabel(_))));

        let prediction = Prediction {
            model: "m@1".into(),
            config_hash: "x".into(),
            taxonomy_version: 1,
            labels: vec![(LabelId::new("earthquake"), 0.7), (LabelId::new("explosion"), 0.3)],
        };
        let set = Change::SetPredictions { event: labelled.id, predictions: vec![prediction] };
        let undo_prediction = project.inverse(&set).unwrap();
        project.commit(&analyst(), set).unwrap();
        assert_eq!(project.event(labelled.id).unwrap().predictions.len(), 1);
        for change in undo_prediction {
            project.commit(&analyst(), change).unwrap();
        }
        assert!(project.event(labelled.id).unwrap().predictions.is_empty());

        project.commit(&analyst(), Change::RemoveEvent { event: labelled.id }).unwrap();
        for change in undo {
            project.commit(&analyst(), change).unwrap();
        }
        assert!(project.taxonomy().get(&quarry.id).is_none());
    }

    #[test]
    fn models_are_referenced_and_undo() {
        let (mut project, _) = Project::create("p", &analyst());
        let model = ModelRef { id: "m@1".into(), path: "/m".into(), hash: "h".into(), enabled: true };
        let undo_add = project.inverse(&Change::SetModel(model.clone())).unwrap();
        project.commit(&analyst(), Change::SetModel(model.clone())).unwrap();
        let disabled = ModelRef { enabled: false, ..model.clone() };
        let undo_disable = project.inverse(&Change::SetModel(disabled.clone())).unwrap();
        project.commit(&analyst(), Change::SetModel(disabled)).unwrap();
        assert_eq!(project.models().len(), 1);
        assert!(!project.models()[0].enabled);
        for change in undo_disable {
            project.commit(&analyst(), change).unwrap();
        }
        assert_eq!(project.model("m@1"), Some(&model));
        for change in undo_add {
            project.commit(&analyst(), change).unwrap();
        }
        assert!(project.models().is_empty());
        assert!(matches!(project.commit(&analyst(), Change::RemoveModel { model: "m@1".into() }), Err(ApplyError::UnknownModel(_))));
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

    fn assigned(project: &mut Project) -> Trace {
        let sensor = Id::generate();
        project.commit(&analyst(), Change::AddSensor { id: sensor, code: "s".to_string() }).unwrap();
        project
            .commit(&analyst(), Change::AssignPhenomenology { sensor, phenomenology: PhenomenologyId::new("seismic") })
            .unwrap();
        let t = trace(sensor);
        project.commit(&analyst(), Change::AddTrace(t.clone())).unwrap();
        t
    }

    fn alignment(offset_s: f64, provenance: Provenance, locked: bool) -> Option<Alignment> {
        Some(Alignment { offset_s, provenance, locked })
    }

    #[test]
    fn locked_alignments_refuse_automatic_changes_but_not_reverts() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let auto = Provenance::Auto { stage: "align".into(), config_hash: "x".into() };
        let manual = Provenance::Manual { analyst: analyst(), at: Timestamp::now() };
        let set = |alignment, revert| Change::SetAlignment { trace: t.id, alignment, revert, event: None };

        project.commit(&analyst(), set(alignment(0.5, auto.clone(), false), false)).unwrap();
        let undo = project.inverse(&set(alignment(1.0, manual.clone(), true), false)).unwrap().remove(0);
        project.commit(&analyst(), set(alignment(1.0, manual, true), false)).unwrap();
        assert_eq!(project.aligned_start(None, project.trace(t.id).unwrap()), Timestamp::new(1_001, 0.0));
        assert_eq!(
            project.commit(&analyst(), set(alignment(2.0, auto, false), false)),
            Err(ApplyError::Locked(t.id))
        );
        project.commit(&analyst(), undo).unwrap();
        assert_eq!(project.trace(t.id).unwrap().offset(), 0.5);
        assert!(!project.trace(t.id).unwrap().is_locked());
    }

    #[test]
    fn processing_needs_an_assignment_and_a_valid_filter() {
        use crate::dsp::{Band, FilterSpec};
        let (mut project, _) = Project::create("p", &analyst());
        let sensor = Id::generate();
        project.commit(&analyst(), Change::AddSensor { id: sensor, code: "u".to_string() }).unwrap();
        let unassigned = trace(sensor);
        project.commit(&analyst(), Change::AddTrace(unassigned.clone())).unwrap();
        let filter = |hz| Processing {
            filter: Some(FilterSpec { band: Band::Lowpass { hz }, order: 2, zero_phase: true }),
            ..Processing::default()
        };
        assert_eq!(
            project.commit(&analyst(), Change::SetProcessing { trace: unassigned.id, processing: filter(10.0) }),
            Err(ApplyError::Unassigned(unassigned.id))
        );
        let t = assigned(&mut project);
        // 100 Hz trace: 60 Hz is past Nyquist.
        assert!(matches!(
            project.commit(&analyst(), Change::SetProcessing { trace: t.id, processing: filter(60.0) }),
            Err(ApplyError::InvalidProcessing(_))
        ));
        let change = Change::SetProcessing { trace: t.id, processing: filter(10.0) };
        let undo = project.inverse(&change).unwrap().remove(0);
        project.commit(&analyst(), change).unwrap();
        assert!(project.trace(t.id).unwrap().processing.filter.is_some());
        project.commit(&analyst(), undo).unwrap();
        assert!(project.trace(t.id).unwrap().processing.is_identity());
    }

    fn pick(trace: Id, phase: &str, offset: f64, provenance: Provenance) -> Pick {
        Pick {
            id: Id::generate(),
            trace,
            time: Timestamp::new(1_000, offset),
            phase: phase.to_string(),
            uncertainty_s: 0.01,
            provenance,
            event: None,
        }
    }

    #[test]
    fn picks_replace_per_phase_and_manual_ones_are_locked() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let auto = Provenance::Auto { stage: "pick".into(), config_hash: "x".into() };
        let manual = Provenance::Manual { analyst: analyst(), at: Timestamp::now() };
        let set = |pick: Pick| Change::SetPick { pick, revert: false };

        project.commit(&analyst(), set(pick(t.id, "P", 0.1, auto.clone()))).unwrap();
        project.commit(&analyst(), set(pick(t.id, "S", 0.3, auto.clone()))).unwrap();
        let replacement = set(pick(t.id, "P", 0.2, manual));
        let undo = project.inverse(&replacement).unwrap().remove(0);
        project.commit(&analyst(), replacement).unwrap();
        assert_eq!(project.trace_picks(t.id).count(), 2);
        let manual_id = project.trace_picks(t.id).find(|p| p.phase == "P").unwrap().id;
        assert_eq!(
            project.commit(&analyst(), set(pick(t.id, "P", 0.4, auto))),
            Err(ApplyError::LockedPick(manual_id))
        );
        project.commit(&analyst(), undo).unwrap();
        let p = project.trace_picks(t.id).find(|p| p.phase == "P").unwrap();
        assert_eq!((p.time, p.provenance.is_manual()), (Timestamp::new(1_000, 0.1), false));

        project.commit(&analyst(), Change::RemoveTrace { trace: t.id }).unwrap();
        assert!(project.picks().is_empty());
    }

    #[test]
    fn pick_times_are_clock_corrected() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let manual = Provenance::Manual { analyst: analyst(), at: Timestamp::now() };
        let p = pick(t.id, "", 0.5, manual);
        project.commit(&analyst(), Change::SetPick { pick: p.clone(), revert: false }).unwrap();
        let clock = ClockCorrection { offset_s: 0.25, drift_ppm: 0.0 };
        project.commit(&analyst(), Change::SetClockCorrection { sensor: t.sensor, clock }).unwrap();
        assert_eq!(project.pick_time(&p), Some(Timestamp::new(1_000, 0.75)));
        assert_eq!(p.phase_label(), "onset");
    }

    #[test]
    fn sensor_locations_validate_and_undo() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let bad = Location { latitude: 91.0, longitude: 0.0, elevation_m: 0.0 };
        assert_eq!(
            project.commit(&analyst(), Change::SetSensorLocation { sensor: t.sensor, location: Some(bad) }),
            Err(ApplyError::InvalidLocation)
        );
        let good = Location { latitude: 35.0, longitude: -106.0, elevation_m: 1500.0 };
        let change = Change::SetSensorLocation { sensor: t.sensor, location: Some(good) };
        let undo = project.inverse(&change).unwrap().remove(0);
        project.commit(&analyst(), change).unwrap();
        assert_eq!(project.sensor(t.sensor).unwrap().location, Some(good));
        project.commit(&analyst(), undo).unwrap();
        assert_eq!(project.sensor(t.sensor).unwrap().location, None);
    }

    fn event(members: &[Id]) -> Event {
        Event {
            id: Id::generate(),
            start: Timestamp::new(1_000, 0.0),
            end: Timestamp::new(1_005, 0.0),
            members: members.iter().map(|&t| crate::event::Member::new(t)).collect(),
            origin: None,
            status: EventStatus::Auto,
            provenance: Provenance::Auto { stage: "detect".into(), config_hash: "x".into() },
            label: None,
            predictions: Vec::new(),
        }
    }

    #[test]
    fn event_alignment_is_separate_from_the_global_one() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let e = event(&[t.id]);
        project.commit(&analyst(), Change::AddEvent(e.clone())).unwrap();
        let manual = Provenance::Manual { analyst: analyst(), at: Timestamp::now() };
        let set = |event| Change::SetAlignment { trace: t.id, alignment: alignment(0.5, manual.clone(), true), revert: false, event };
        project.commit(&analyst(), set(Some(e.id))).unwrap();
        assert_eq!((project.offset(Some(e.id), t.id), project.offset(None, t.id)), (0.5, 0.0));
        assert!(project.is_locked(Some(e.id), t.id) && !project.is_locked(None, t.id));

        // Not a member of a second event.
        let other = event(&[]);
        project.commit(&analyst(), Change::AddEvent(other.clone())).unwrap();
        assert_eq!(project.commit(&analyst(), set(Some(other.id))), Err(ApplyError::NotMember { event: other.id, trace: t.id }));
    }

    #[test]
    fn picks_are_unique_per_event() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let (a, b) = (event(&[t.id]), event(&[t.id]));
        project.commit(&analyst(), Change::AddEvent(a.clone())).unwrap();
        project.commit(&analyst(), Change::AddEvent(b.clone())).unwrap();
        let auto = Provenance::Auto { stage: "pick".into(), config_hash: "x".into() };
        for e in [a.id, b.id] {
            let p = Pick { event: Some(e), ..pick(t.id, "P", 0.1, auto.clone()) };
            project.commit(&analyst(), Change::SetPick { pick: p, revert: false }).unwrap();
        }
        assert_eq!(project.trace_picks(t.id).count(), 2);
        assert_eq!(project.event_picks(Some(a.id), t.id).count(), 1);

        // Removing an event removes its picks; undo restores both.
        let remove = Change::RemoveEvent { event: a.id };
        let undo = project.inverse(&remove).unwrap();
        project.commit(&analyst(), remove).unwrap();
        assert_eq!((project.events().len(), project.trace_picks(t.id).count()), (1, 1));
        for change in undo {
            project.commit(&analyst(), change).unwrap();
        }
        assert_eq!((project.events().len(), project.event_picks(Some(a.id), t.id).count()), (2, 1));
    }

    #[test]
    fn events_validate_and_lose_removed_traces() {
        let (mut project, _) = Project::create("p", &analyst());
        let t = assigned(&mut project);
        let backwards = Event { end: Timestamp::new(999, 0.0), ..event(&[t.id]) };
        assert!(matches!(project.commit(&analyst(), Change::AddEvent(backwards)), Err(ApplyError::InvalidEvent(_))));
        let ghost = Id::generate();
        assert_eq!(project.commit(&analyst(), Change::AddEvent(event(&[ghost]))), Err(ApplyError::UnknownTrace(ghost)));
        let e = event(&[t.id]);
        project.commit(&analyst(), Change::AddEvent(e.clone())).unwrap();
        let origin = Origin {
            latitude: 1.0,
            longitude: 2.0,
            depth_m: 0.0,
            time: Timestamp::new(1_000, 0.5),
            rms_s: 0.0,
            provenance: Provenance::Auto { stage: "locate".into(), config_hash: "x".into() },
        };
        project.commit(&analyst(), Change::SetOrigin { origin: Some(origin.clone()), event: Some(e.id) }).unwrap();
        assert_eq!(project.origin_of(Some(e.id)), Some(&origin));
        assert_eq!(project.origin_of(None), None);
        project.commit(&analyst(), Change::RemoveTrace { trace: t.id }).unwrap();
        assert!(project.event(e.id).unwrap().members.is_empty());
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
