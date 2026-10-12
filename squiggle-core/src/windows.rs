//! Aligned waveform windows, the input of deep models: per phenomenology, a
//! fixed-length window of each trace from its onset, resampled to one rate,
//! demeaned and scaled to unit peak. Exported with datasets and computed again
//! at inference by this same code.
//!
//! Onsets are the trace's first pick in the event, else the first phase
//! predicted from the origin. Members come first; when the event is located,
//! every other located trace of the phenomenology joins at its predicted
//! onset, so a missing arrival is seen as noise rather than as absence.

use crate::features::{EventData, Kind, TraceData};
use crate::locate::travel_time;
use crate::phenomenology::PhenomenologyId;

/// Version of the window definitions; recorded with datasets and models.
pub const WINDOW_SET: &str = "heidtmare.squiggle.windows/1";

/// How one phenomenology's windows are cut.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowSpec {
    pub phenomenology: PhenomenologyId,
    /// Samples per second after resampling.
    pub rate: f64,
    /// Seconds before the onset.
    pub pre_s: f64,
    pub samples: usize,
    /// Most traces per event; members first, then by onset.
    pub max_traces: usize,
}

impl WindowSpec {
    /// Defaults by propagation class: 12.8 s at 40 Hz for seismic-like
    /// traces, 25.6 s at 10 Hz through air, 256 samples at the trace rate otherwise.
    pub fn default_for(phenomenology: PhenomenologyId, kind: Kind, rate: f64) -> Self {
        let (rate, pre_s, samples) = match kind {
            Kind::Seismic => (40.0, 1.0, 512),
            Kind::Air => (10.0, 2.0, 256),
            _ => (rate, 32.0 / rate, 256),
        };
        Self { phenomenology, rate, pre_s, samples, max_traces: 16 }
    }

    pub fn duration(&self) -> f64 {
        self.samples as f64 / self.rate
    }
}

/// One phenomenology's windows for an event.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Windows {
    /// Each `samples` long.
    pub traces: Vec<Vec<f32>>,
    /// Index into the traces of each window.
    pub sources: Vec<usize>,
}

/// Windows of `event` for each spec, in spec order.
pub fn windows(traces: &[TraceData], event: &EventData, specs: &[WindowSpec]) -> Vec<Windows> {
    specs.iter().map(|spec| branch(traces, event, spec)).collect()
}

/// Onset of trace `i` in `event`: its first pick, else the first phase
/// predicted from the origin, else the window start for a member.
pub fn onset(traces: &[TraceData], event: &EventData, i: usize) -> Option<f64> {
    let first_pick = event.picks.iter().filter(|p| p.0 == i).map(|p| p.2).min_by(f64::total_cmp);
    let predicted = || {
        let origin = event.origin?;
        let trace = traces.get(i)?;
        Some(origin.time + travel_time(origin.source, trace.location.as_ref()?, trace.propagation.speed("")?))
    };
    first_pick.or_else(predicted).or_else(|| event.members.contains(&i).then_some(event.start))
}

fn branch(traces: &[TraceData], event: &EventData, spec: &WindowSpec) -> Windows {
    // (member first, onset, trace)
    let mut chosen: Vec<(bool, f64, usize)> = traces
        .iter()
        .enumerate()
        .filter(|(_, t)| t.phenomenology == spec.phenomenology)
        .filter_map(|(i, _)| Some((event.members.contains(&i), onset(traces, event, i)?, i)))
        .collect();
    chosen.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    chosen.truncate(spec.max_traces);
    Windows {
        traces: chosen.iter().map(|&(_, onset, i)| cut(&traces[i], onset - spec.pre_s, spec)).collect(),
        sources: chosen.iter().map(|c| c.2).collect(),
    }
}

/// `spec.samples` at `spec.rate` from `from`: box-averaged when decimating,
/// linearly interpolated otherwise; zero outside the trace; demeaned, unit peak.
pub fn cut(trace: &TraceData, from: f64, spec: &WindowSpec) -> Vec<f32> {
    let ratio = trace.rate / spec.rate;
    let at = |t: f64| -> Option<f64> {
        let x = (t - trace.start) * trace.rate;
        if x < 0.0 || x > (trace.values.len() - 1) as f64 || trace.values.is_empty() {
            return None;
        }
        let i = x.floor() as usize;
        let f = x - i as f64;
        let a = f64::from(trace.values[i]);
        let b = f64::from(*trace.values.get(i + 1).unwrap_or(&trace.values[i]));
        Some(a + (b - a) * f)
    };
    let mut out: Vec<f64> = (0..spec.samples)
        .map(|k| {
            let t = from + k as f64 / spec.rate;
            if ratio > 1.0 {
                let n = ratio.round() as usize;
                let values: Vec<f64> = (0..n).filter_map(|j| at(t + (j as f64 - (n as f64 - 1.0) / 2.0) / trace.rate)).collect();
                if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
            } else {
                at(t).unwrap_or(0.0)
            }
        })
        .collect();
    let mean = out.iter().sum::<f64>() / out.len().max(1) as f64;
    for v in &mut out {
        *v -= mean;
    }
    let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    out.into_iter().map(|v| if peak > 0.0 { (v / peak) as f32 } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::OriginData;
    use crate::locate::Source;
    use crate::phenomenology::{Phase, Propagation};
    use crate::sensor::Location;
    use crate::Id;

    fn trace(phenomenology: &str, rate: f64, values: Vec<f32>, location: Option<Location>) -> TraceData {
        TraceData {
            trace: Id::generate(),
            sensor: Id::generate(),
            phenomenology: PhenomenologyId::new(phenomenology),
            propagation: Propagation::Phases(vec![Phase { name: "P".into(), speed_m_s: 6000.0 }, Phase { name: "S".into(), speed_m_s: 3500.0 }]),
            values,
            rate,
            start: 0.0,
            location,
        }
    }

    #[test]
    fn windows_start_at_the_onset_resampled_with_unit_peak() {
        // A 2 Hz sine from t = 5 s at 100 Hz; window at 20 Hz from 1 s before a pick at 5 s.
        let values: Vec<f32> = (0..1000).map(|i| { let t = i as f64 / 100.0; if t >= 5.0 { (std::f64::consts::TAU * 2.0 * (t - 5.0)).sin() as f32 * 3.0 } else { 0.0 } }).collect();
        let spec = WindowSpec { phenomenology: PhenomenologyId::new("seismic"), rate: 20.0, pre_s: 1.0, samples: 80, max_traces: 4 };
        let t = trace("seismic", 100.0, values, None);
        let event = EventData { start: 4.0, end: 9.0, origin: None, members: vec![0], picks: vec![(0, "P".into(), 5.0)] };
        let w = windows(std::slice::from_ref(&t), &event, std::slice::from_ref(&spec));
        assert_eq!(w[0].traces.len(), 1);
        let x = &w[0].traces[0];
        assert_eq!(x.len(), 80);
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 1e-6);
        // Quiet before the onset (only the demeaning offset), oscillating after.
        assert!(x[..18].windows(2).all(|p| (p[0] - p[1]).abs() < 1e-6));
        assert!(x[20..].iter().any(|v| *v > 0.5) && x[20..].iter().any(|v| *v < -0.5));
        // A window past the end is zero-padded.
        let late = cut(&t, 9.5, &spec);
        assert!(late[20..].iter().all(|v| *v == late[79]));
    }

    #[test]
    fn located_events_add_other_traces_at_their_predicted_onset() {
        let here = Location { latitude: 35.0, longitude: -106.0, elevation_m: 0.0 };
        let there = Location { latitude: 35.1, longitude: -106.0, elevation_m: 0.0 };
        let traces = vec![trace("seismic", 100.0, vec![0.0; 3000], Some(here)), trace("seismic", 100.0, vec![0.0; 3000], Some(there)), trace("infrasound", 100.0, vec![0.0; 3000], Some(here))];
        let spec = WindowSpec { phenomenology: PhenomenologyId::new("seismic"), rate: 40.0, pre_s: 1.0, samples: 64, max_traces: 8 };
        let unlocated = EventData { start: 1.0, end: 5.0, origin: None, members: vec![1], picks: vec![] };
        assert_eq!(windows(&traces, &unlocated, std::slice::from_ref(&spec))[0].sources, [1]);
        let origin = OriginData { source: Source { latitude: 35.0, longitude: -106.0, depth_m: 1000.0 }, time: 2.0, rms_s: 0.0 };
        let located = EventData { origin: Some(origin), ..unlocated };
        let w = windows(&traces, &located, &[spec.clone(), WindowSpec { phenomenology: PhenomenologyId::new("emp"), ..spec }]);
        assert_eq!(w[0].sources, [1, 0], "member first");
        assert!(w[1].traces.is_empty(), "no such sensors");
    }
}
