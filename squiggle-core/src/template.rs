//! Template library: reference events cut per sensor, curated by analysts.
//! An event matches a template by normalised cross-correlation on each
//! sensor they share, searched over a lag, averaged over those sensors (a
//! network correlation). Matches vote for the templates' labels, and the
//! same templates drive the matched-filter detector ([`crate::dsp::matched`]).

use crate::Id;
use crate::features::{EventData, TraceData};
use crate::phenomenology::PhenomenologyId;
use crate::taxonomy::LabelId;
use crate::windows::{WindowSpec, cut, onset, windows};

/// Model id that template predictions are recorded under.
pub const TEMPLATES_MODEL: &str = "heidtmare.squiggle.model/templates@1";

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Template {
    pub id: Id,
    pub name: String,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub label: Option<LabelId>,
    /// Event it was cut from.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub event: Option<Id>,
    pub traces: Vec<TemplateTrace>,
}

/// One sensor's window, from `pre_s` before its onset.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TemplateTrace {
    pub sensor: Id,
    pub phenomenology: PhenomenologyId,
    pub rate: f64,
    pub pre_s: f64,
    /// Demeaned, unit peak.
    pub samples: Vec<f32>,
}

impl Template {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("a template needs a name".to_string());
        }
        if self.traces.is_empty() {
            return Err("a template needs at least one trace".to_string());
        }
        for (i, t) in self.traces.iter().enumerate() {
            if !(t.rate.is_finite() && t.rate > 0.0) || t.samples.len() < 8 || t.pre_s.is_nan() || t.pre_s < 0.0 {
                return Err("each template trace needs a rate, an onset offset and at least 8 samples".to_string());
            }
            if self.traces[..i].iter().any(|o| o.sensor == t.sensor) {
                return Err("a sensor is listed twice".to_string());
            }
        }
        Ok(())
    }
}

/// Member windows of `event` for `specs`, one per sensor: a template's traces.
pub fn template_traces(traces: &[TraceData], event: &EventData, specs: &[WindowSpec]) -> Vec<TemplateTrace> {
    let mut out: Vec<TemplateTrace> = Vec::new();
    for (spec, cut) in specs.iter().zip(windows(traces, event, specs)) {
        for (samples, &i) in cut.traces.into_iter().zip(&cut.sources) {
            let trace = &traces[i];
            if event.members.contains(&i) && !out.iter().any(|t| t.sensor == trace.sensor) && samples.iter().any(|v| *v != 0.0) {
                out.push(TemplateTrace { sensor: trace.sensor, phenomenology: spec.phenomenology.clone(), rate: spec.rate, pre_s: spec.pre_s, samples });
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchSettings {
    /// Lag searched either side of each onset.
    pub max_lag_s: f64,
    /// Fewest shared sensors for a match.
    pub min_sensors: usize,
    /// Network correlation below which templates do not vote.
    pub min_cc: f64,
    /// Softmax temperature turning per-label correlations into probabilities.
    pub sharpness: f64,
}

impl Default for MatchSettings {
    fn default() -> Self {
        Self { max_lag_s: 1.0, min_sensors: 2, min_cc: 0.5, sharpness: 0.05 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    pub template: Id,
    /// Mean over shared sensors of the best normalised correlation.
    pub cc: f64,
    pub sensors: usize,
    /// Mean lag of the best correlations; positive when the event is late.
    pub lag_s: f64,
}

/// `event`'s network correlation with `template`; `None` when fewer than
/// `settings.min_sensors` sensors have both a template trace and an onset.
pub fn match_template(traces: &[TraceData], event: &EventData, template: &Template, settings: &MatchSettings) -> Option<Match> {
    let mut sum = (0.0, 0.0);
    let mut n = 0;
    for t in &template.traces {
        let Some(i) = traces.iter().position(|d| d.sensor == t.sensor && d.phenomenology == t.phenomenology) else { continue };
        let Some(at) = onset(traces, event, i) else { continue };
        let lag = (settings.max_lag_s * t.rate).round().max(0.0) as usize;
        let spec = WindowSpec { phenomenology: t.phenomenology.clone(), rate: t.rate, pre_s: t.pre_s, samples: t.samples.len() + 2 * lag, max_traces: 1 };
        let segment = cut(&traces[i], at - t.pre_s - lag as f64 / t.rate, &spec);
        if let Some((k, cc)) = best_lag(&t.samples, &segment) {
            sum.0 += cc;
            sum.1 += (k as f64 - lag as f64) / t.rate;
            n += 1;
        }
    }
    (n >= settings.min_sensors.max(1)).then(|| Match { template: template.id, cc: sum.0 / n as f64, sensors: n, lag_s: sum.1 / n as f64 })
}

/// Offset into `x` and value of the largest normalised correlation of `t`
/// with each `t.len()`-long stretch of `x`; `None` when either is flat.
pub fn best_lag(t: &[f32], x: &[f32]) -> Option<(usize, f64)> {
    let n = t.len();
    if n == 0 || x.len() < n {
        return None;
    }
    let mean = t.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let t: Vec<f64> = t.iter().map(|&v| f64::from(v) - mean).collect();
    let norm = t.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return None;
    }
    let x: Vec<f64> = x.iter().map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 }).collect();
    let mut best: Option<(usize, f64)> = None;
    for k in 0..=x.len() - n {
        let w = &x[k..k + n];
        let (s, s2) = w.iter().fold((0.0, 0.0), |(a, b), v| (a + v, b + v * v));
        let energy = s2 - s * s / n as f64;
        if energy <= 1e-12 * n as f64 {
            continue;
        }
        let cc = t.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / (norm * energy.sqrt());
        if best.is_none_or(|b| cc > b.1) {
            best = Some((k, cc));
        }
    }
    best
}

/// Matches of `event` with every template, best first.
pub fn match_all(traces: &[TraceData], event: &EventData, templates: &[Template], settings: &MatchSettings) -> Vec<Match> {
    let mut matches: Vec<Match> = templates.iter().filter_map(|t| match_template(traces, event, t, settings)).collect();
    matches.sort_by(|a, b| b.cc.total_cmp(&a.cc));
    matches
}

/// Label probabilities from `matches`: a softmax over each labelled
/// template class's best correlation; `None` when no labelled match reaches
/// `settings.min_cc`.
pub fn vote(matches: &[Match], templates: &[Template], settings: &MatchSettings) -> Option<Vec<(LabelId, f32)>> {
    let mut best: Vec<(LabelId, f64)> = Vec::new();
    for template in templates {
        let Some(label) = &template.label else { continue };
        let cc = matches.iter().find(|m| m.template == template.id).map_or(-1.0, |m| m.cc);
        match best.iter_mut().find(|b| &b.0 == label) {
            Some(b) => b.1 = b.1.max(cc),
            None => best.push((label.clone(), cc)),
        }
    }
    let top = best.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
    if top.is_nan() || top < settings.min_cc {
        return None;
    }
    let weights: Vec<f64> = best.iter().map(|b| ((b.1 - top) / settings.sharpness.max(1e-6)).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut out: Vec<(LabelId, f32)> = best.into_iter().zip(weights).map(|(b, w)| (b.0, (w / total) as f32)).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenomenology::{Phase, Propagation};

    fn wiggle(i: usize, shape: u32) -> f32 {
        let t = i as f64 / 40.0;
        let f = 2.0 + f64::from(shape);
        if t < 1.0 { 0.0 } else { ((std::f64::consts::TAU * f * (t - 1.0)).sin() * (-(t - 1.0) / 0.8).exp()) as f32 }
    }

    /// Two sensors recording `shape` with an onset at `onset` s (40 Hz, 20 s), plus noise.
    fn traces(onset: f64, shape: u32, sensors: [Id; 2], seed: u32) -> Vec<TraceData> {
        let mut state = seed | 1;
        sensors
            .iter()
            .map(|&sensor| TraceData {
                trace: Id::generate(),
                sensor,
                phenomenology: PhenomenologyId::new("seismic"),
                propagation: Propagation::Phases(vec![Phase { name: "P".into(), speed_m_s: 6000.0 }, Phase { name: "S".into(), speed_m_s: 3500.0 }]),
                values: (0..800)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        let k = i as f64 - (onset - 1.0) * 40.0;
                        let signal = if k >= 0.0 { wiggle(k as usize, shape) } else { 0.0 };
                        signal + 0.05 * ((state % 2001) as f32 / 1000.0 - 1.0)
                    })
                    .collect(),
                rate: 40.0,
                start: 0.0,
                location: None,
            })
            .collect()
    }

    fn event(onset: f64) -> EventData {
        EventData { start: onset - 1.0, end: onset + 5.0, origin: None, members: vec![0, 1], picks: vec![(0, "P".into(), onset), (1, "P".into(), onset)] }
    }

    #[test]
    fn repeats_match_their_template_and_vote_its_label() {
        let sensors = [Id::generate(), Id::generate()];
        let spec = WindowSpec { phenomenology: PhenomenologyId::new("seismic"), rate: 40.0, pre_s: 0.5, samples: 160, max_traces: 4 };
        let make = |name: &str, shape: u32, label: &str| {
            let traces = template_traces(&traces(5.0, shape, sensors, 3), &event(5.0), std::slice::from_ref(&spec));
            Template { id: Id::generate(), name: name.into(), label: Some(LabelId::new(label)), event: None, traces }
        };
        let library = [make("a", 0, "quarry"), make("b", 3, "swarm")];
        assert_eq!(library[0].traces.len(), 2);
        assert!(library[0].validate().is_ok());

        // A repeat of shape 0, picked 0.2 s late, with different noise.
        let repeat = traces(9.0, 0, sensors, 7);
        let settings = MatchSettings::default();
        let matches = match_all(&repeat, &event(8.8), &library, &settings);
        assert_eq!(matches[0].template, library[0].id);
        assert!(matches[0].cc > 0.9 && matches[1].cc < 0.6, "{matches:?}");
        assert!((matches[0].lag_s - 0.2).abs() < 0.05, "{matches:?}");
        let p = vote(&matches, &library, &settings).unwrap();
        assert_eq!(p[0].0.as_str(), "quarry");
        assert!(p[0].1 > 0.99);

        // One sensor is too few; an unrelated event does not vote.
        let one = MatchSettings { min_sensors: 3, ..settings };
        assert!(match_template(&repeat, &event(8.8), &library[0], &one).is_none());
        let flat: Vec<TraceData> = repeat.iter().map(|t| TraceData { values: t.values.iter().map(|v| v * 0.0 + (v * 1e3).sin() * 0.05).collect(), ..t.clone() }).collect();
        assert!(vote(&match_all(&flat, &event(8.8), &library, &settings), &library, &settings).is_none());
    }

    #[test]
    fn best_lag_finds_a_shifted_copy() {
        let t: Vec<f32> = (0..50).map(|i| wiggle(i + 40, 1)).collect();
        let mut x = vec![0.0f32; 30];
        x.extend(&t);
        x.extend(vec![0.0f32; 20]);
        let (k, cc) = best_lag(&t, &x).unwrap();
        assert_eq!(k, 30);
        assert!((cc - 1.0).abs() < 1e-9);
        assert!(best_lag(&[0.0; 4], &x).is_none());
    }
}
