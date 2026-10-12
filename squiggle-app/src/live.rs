//! Detection and association on a live feed. Each trace streams through its
//! detector (and matched filter) as samples arrive; onsets are refined once
//! the samples AIC reads exist. Association runs over a sliding window of
//! open arrivals: a candidate is final, and becomes an event, once the feed
//! has settled a horizon past its first arrival on every trace, so no later
//! arrival could still join it. The result equals detecting the whole
//! recording at once.

use std::collections::VecDeque;

use squiggle_core::associate::{AssociateSettings, Candidate, associate};
use squiggle_core::dsp::{Detector, MatchedFilter, Trigger};
use squiggle_core::template::TemplateTrace;

use crate::events_ui::{DetectInput, DetectOutcome, Detection, arrivals, merge_matched};

/// A trace whose newest sample lags the newest feed time by more than this
/// does not hold back closing events.
pub const STALL_S: f64 = 30.0;

struct LiveInput {
    input: DetectInput,
    detector: Option<Box<dyn Detector>>,
    matched: Option<MatchedFilter>,
    /// Detector triggers waiting for the samples refinement reads.
    pending: VecDeque<Trigger>,
    floor: usize,
    own: Vec<Detection>,
    matched_onsets: Vec<Detection>,
}

impl LiveInput {
    /// Seconds after the reference before which this trace reports nothing new.
    fn settled(&self) -> f64 {
        let len = self.input.values.len();
        let mut index = len;
        if let Some(detector) = &self.detector {
            index = index.min(detector.settled().saturating_sub(self.input.lookback()));
        }
        if let Some(front) = self.pending.front() {
            index = index.min(front.on.saturating_sub(self.input.lookback()));
        }
        if let Some(filter) = &self.matched {
            index = index.min(filter.settled());
        }
        self.input.start + index as f64 / self.input.rate
    }

    fn newest(&self) -> f64 {
        self.input.start + self.input.values.len() as f64 / self.input.rate
    }

    fn refine_ready(&mut self, input: usize, all: bool) {
        while let Some(front) = self.pending.front() {
            if !all && front.on + self.input.lookahead() > self.input.values.len() {
                break;
            }
            let trigger = self.pending.pop_front().expect("checked");
            self.own.extend(self.input.onset(input, &trigger, self.floor));
            self.floor = trigger.off;
        }
    }
}

pub struct LiveDetection {
    inputs: Vec<LiveInput>,
    settings: AssociateSettings,
    /// Longest an arrival can follow the first one of its event.
    pub horizon: f64,
    /// Arrivals already in events: (input, onset).
    committed: Vec<(usize, f64)>,
    pub errors: Vec<(squiggle_core::Id, String)>,
    /// Final events so far.
    pub events: usize,
    /// Closing line of the last association: arrivals before it were all considered.
    checked: f64,
}

impl LiveDetection {
    /// `inputs` with whatever samples they hold so far (usually none).
    pub fn new(inputs: Vec<DetectInput>, settings: AssociateSettings) -> Self {
        let mut live = Self { inputs: Vec::new(), settings, horizon: 0.0, committed: Vec::new(), errors: Vec::new(), events: 0, checked: f64::NEG_INFINITY };
        for input in inputs {
            live.add(input);
        }
        live
    }

    /// Adds a trace (e.g. a channel that joined the feed); returns its index.
    pub fn add(&mut self, mut input: DetectInput) -> usize {
        let reach = match (input.speed_m_s, input.location) {
            (Some(v), Some(_)) if v > 0.0 => self.settings.radius_m / v,
            _ => self.settings.max_window_s,
        };
        self.horizon = self.horizon.max(reach + input.coda_s + 5.0);
        let values = std::mem::take(&mut input.values);
        let detector = input.spec.build(input.rate).map_err(|e| self.errors.push((input.trace, e))).ok();
        let matched = (!input.templates.is_empty())
            .then(|| {
                let templates: Vec<&TemplateTrace> = input.templates.iter().collect();
                MatchedFilter::new(&templates, input.rate, &input.matched).map_err(|e| self.errors.push((input.trace, e))).ok()
            })
            .flatten();
        let mut entry = LiveInput { input, detector, matched, pending: VecDeque::new(), floor: 0, own: Vec::new(), matched_onsets: Vec::new() };
        let index = self.inputs.len();
        push_into(&mut entry, index, &values);
        self.inputs.push(entry);
        index
    }

    /// Input index of `trace`.
    pub fn input_of(&self, trace: squiggle_core::Id) -> Option<usize> {
        self.inputs.iter().position(|i| i.input.trace == trace)
    }

    /// Samples fed to input `index` so far.
    pub fn fed(&self, index: usize) -> usize {
        self.inputs[index].input.values.len()
    }

    pub fn inputs(&self) -> Vec<&DetectInput> {
        self.inputs.iter().map(|i| &i.input).collect()
    }

    /// Appends samples to input `index` and runs its detectors over them.
    pub fn push(&mut self, index: usize, values: &[f32]) {
        push_into(&mut self.inputs[index], index, values);
    }

    /// Feed time up to which every live trace has settled.
    pub fn settled(&self) -> f64 {
        let newest = self.inputs.iter().map(LiveInput::newest).fold(f64::NEG_INFINITY, f64::max);
        self.inputs.iter().filter(|i| i.newest() >= newest - STALL_S).map(LiveInput::settled).fold(f64::INFINITY, f64::min)
    }

    /// Events that became final: candidates whose first arrival is a horizon
    /// before the settled time. Associates only when an arrival has crossed
    /// that line since the last call.
    pub fn close(&mut self) -> DetectOutcome {
        let until = self.settled() - self.horizon;
        let crossed = self.inputs.iter().any(|e| e.own.iter().chain(&e.matched_onsets).any(|d| d.at >= self.checked && d.at < until));
        if !crossed {
            return DetectOutcome { detections: Vec::new(), candidates: Vec::new(), errors: std::mem::take(&mut self.errors) };
        }
        self.checked = until;
        self.outcome(Some(until))
    }

    /// Arrivals not yet in a final event.
    pub fn open(&self) -> usize {
        self.inputs.iter().flat_map(|e| e.own.iter().chain(&e.matched_onsets)).filter(|d| !self.committed.contains(&(d.input, d.at))).count()
    }

    /// Ends the feed: flushes every detector and closes every candidate.
    pub fn finish(&mut self) -> DetectOutcome {
        for (index, entry) in self.inputs.iter_mut().enumerate() {
            if let Some(detector) = &mut entry.detector {
                entry.pending.extend(detector.finish());
            }
            entry.refine_ready(index, true);
            if let Some(filter) = &mut entry.matched {
                let triggers = filter.finish();
                entry.matched_onsets.extend(triggers.iter().map(|t| entry.input.matched_onset(index, t)));
            }
        }
        self.outcome(None)
    }

    fn outcome(&mut self, until: Option<f64>) -> DetectOutcome {
        let detections: Vec<Detection> = self
            .inputs
            .iter()
            .flat_map(|entry| {
                let own = if entry.matched_onsets.is_empty() {
                    entry.own.clone()
                } else {
                    merge_matched(entry.own.clone(), &entry.matched_onsets, entry.input.matched.length_s)
                };
                own.into_iter()
            })
            .filter(|d| !self.committed.contains(&(d.input, d.at)))
            .filter(|d| until.is_none_or(|u| d.at >= u - 2.0 * self.horizon))
            .collect();
        let inputs: Vec<&DetectInput> = self.inputs.iter().map(|i| &i.input).collect();
        let candidates: Vec<Candidate> = associate(&arrivals(&inputs, &detections), &self.settings)
            .into_iter()
            .filter(|c| until.is_none_or(|u| c.arrivals.first().is_some_and(|&a| detections[a].at < u)))
            .collect();
        for c in &candidates {
            self.committed.extend(c.arrivals.iter().chain(&c.absorbed).map(|&a| (detections[a].input, detections[a].at)));
        }
        self.events += candidates.len();
        DetectOutcome { detections, candidates, errors: std::mem::take(&mut self.errors) }
    }
}

fn push_into(entry: &mut LiveInput, index: usize, values: &[f32]) {
    entry.input.values.extend_from_slice(values);
    if let Some(detector) = &mut entry.detector {
        entry.pending.extend(detector.process(values));
    }
    entry.refine_ready(index, false);
    if let Some(filter) = &mut entry.matched {
        let triggers = filter.process(values);
        entry.matched_onsets.extend(triggers.iter().map(|t| entry.input.matched_onset(index, t)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify_ui::{event_data, trace_data};
    use crate::events_ui::{CHUNK, DetectMeta, default_detector, event_changes, run_detection};
    use crate::templates_ui::template_change;
    use crate::workspace::{SampleState, SyntheticEvent, Workspace};
    use squiggle_core::dsp::MatchedSpec;
    use squiggle_core::{Analyst, Change, Id, IqComponent, Trace};

    fn loaded(workspace: &Workspace, trace: &Trace) -> Option<Vec<f32>> {
        match workspace.samples.get(&trace.id) {
            Some(SampleState::Loaded(samples)) => Some(samples.series(IqComponent::Magnitude).iter().collect()),
            _ => None,
        }
    }

    fn inputs(workspace: &Workspace, templates: bool) -> Vec<DetectInput> {
        let project = &workspace.project;
        let radius = AssociateSettings::default().radius_m;
        project
            .traces()
            .iter()
            .filter_map(|trace| {
                let id = project.trace_phenomenology(trace)?.id.clone();
                let input = DetectInput::new(project, trace, loaded(workspace, trace)?, default_detector(id.as_str(), trace.sample_rate), radius)?;
                Some(if templates { input.with_templates(project, MatchedSpec::default()) } else { input })
            })
            .collect()
    }

    /// Each new event as (origin time, members, picks), from detection changes.
    fn events(changes: &[Change]) -> Vec<(i64, Vec<Id>, usize)> {
        let mut out: Vec<(i64, Vec<Id>, usize)> = changes
            .iter()
            .filter_map(|c| match c {
                Change::AddEvent(e) => {
                    let mut members: Vec<Id> = e.members.iter().map(|m| m.trace).collect();
                    members.sort();
                    let picks = changes.iter().filter(|c| matches!(c, Change::SetPick { pick, .. } if pick.event == Some(e.id))).count();
                    Some(((e.time().unix_millis() as f64 * 10.0).round() as i64, members, picks))
                }
                _ => None,
            })
            .collect();
        out.sort();
        out
    }

    /// Detects `workspace` whole, then fed in half-second chunks with events
    /// closed as they become final; both must agree exactly.
    fn live_equals_whole(workspace: &Workspace, templates: bool) -> usize {
        let project = &workspace.project;
        let reference = project.reference.unwrap();
        let settings = AssociateSettings::default();
        let whole = inputs(workspace, templates);
        let meta: Vec<DetectMeta> = whole.iter().map(DetectMeta::from).collect();
        let outcome = run_detection(&whole, &settings, CHUNK);
        let expected = events(&event_changes(project, &meta, &outcome, reference, true, "h"));

        let mut live = LiveDetection::new(inputs(workspace, templates).into_iter().map(|i| DetectInput { values: Vec::new(), ..i }).collect(), settings);
        let mut changes = Vec::new();
        let mut early = 0;
        let longest = whole.iter().map(|i| i.values.len() as f64 / i.rate).fold(0.0, f64::max);
        let steps = (longest / 0.5).ceil() as usize;
        for step in 0..steps {
            for (index, input) in whole.iter().enumerate() {
                let n = (0.5 * input.rate).round() as usize;
                let (lo, hi) = ((step * n).min(input.values.len()), ((step + 1) * n).min(input.values.len()));
                live.push(index, &input.values[lo..hi]);
            }
            let closed = live.close();
            early += closed.candidates.len();
            changes.extend(event_changes(project, &meta, &closed, reference, true, "h").into_iter().filter(|c| !matches!(c, Change::RemoveEvent { .. })));
        }
        let rest = live.finish();
        changes.extend(event_changes(project, &meta, &rest, reference, true, "h").into_iter().filter(|c| !matches!(c, Change::RemoveEvent { .. })));
        assert_eq!(events(&changes), expected);
        // Events within a horizon of the end close only when the feed ends.
        let end = reference.add_seconds(longest - live.horizon - 15.0);
        let due = expected.iter().filter(|e| (e.0 as f64) / 10.0 < end.unix_millis() as f64).count();
        assert!(early >= due && due > 0, "{early} closed before the end; {due} due");
        expected.len()
    }

    /// Phase 10 self-check: a replayed stream gives the same events as the file.
    #[test]
    fn a_streamed_swarm_gives_the_whole_files_events() {
        let workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Swarm).unwrap();
        assert_eq!(live_equals_whole(&workspace, false), crate::synthetic::SWARM_DETECTABLE);
    }

    /// With infrasound (slow arrivals held for located events) and matched filters.
    #[test]
    fn streamed_families_with_templates_give_the_whole_files_events() {
        let mut workspace = Workspace::synthetic(Analyst("ana".into()), SyntheticEvent::Families).unwrap();
        let settings = AssociateSettings::default();
        let whole = inputs(&workspace, false);
        let meta: Vec<DetectMeta> = whole.iter().map(DetectMeta::from).collect();
        let outcome = run_detection(&whole, &settings, CHUNK);
        let reference = workspace.project.reference.unwrap();
        let changes = event_changes(&workspace.project, &meta, &outcome, reference, true, "h");
        workspace.execute("detect", changes).unwrap();
        let project = &workspace.project;
        let traces = trace_data(project, |t| loaded(&workspace, t));
        let first: Vec<Id> = project.events().iter().take(4).map(|e| e.id).collect();
        let mut made = Vec::new();
        for id in first {
            let data = event_data(project, project.event(id).unwrap(), &traces).unwrap();
            made.push(template_change(project, &traces, &data, id, "t").unwrap());
        }
        workspace.execute("templates", made).unwrap();
        // Drop the events, keep the templates: the stream finds them again.
        let remove: Vec<Change> = workspace.project.events().iter().map(|e| Change::RemoveEvent { event: e.id }).collect();
        workspace.execute("clear", remove).unwrap();
        assert!(live_equals_whole(&workspace, true) >= 24);
    }
}
