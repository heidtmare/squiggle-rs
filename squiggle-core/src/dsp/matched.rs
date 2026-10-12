//! Matched filter: streams a trace against templates of its own sensor by
//! normalised cross-correlation, after box-averaging to near the template
//! rate. Each correlation is computed afresh from the buffer, so chunked and
//! whole runs agree exactly. A trigger's `on` is the onset implied by the
//! best-correlating alignment, not the threshold crossing; triggers within
//! half a template of each other (correlation sidelobes) merge.

use std::collections::VecDeque;

use super::detect::Detector;
use super::pick::Trigger;
use crate::template::TemplateTrace;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchedSpec {
    /// Template length used from its start, seconds.
    pub length_s: f64,
    /// Correlation that opens a trigger.
    pub on: f64,
    /// Correlation that closes it.
    pub off: f64,
}

impl Default for MatchedSpec {
    fn default() -> Self {
        Self { length_s: 6.0, on: 0.5, off: 0.3 }
    }
}

struct Kernel {
    /// Demeaned.
    values: Vec<f64>,
    norm: f64,
    /// Onset within the kernel, decimated samples.
    onset: usize,
}

pub struct MatchedFilter {
    decimate: usize,
    kernels: Vec<Kernel>,
    ring: VecDeque<f64>,
    longest: usize,
    partial: (f64, usize),
    /// Decimated samples seen.
    seen: usize,
    on: f64,
    off: f64,
    /// Open trigger and its best onset (input samples).
    active: Option<(Trigger, usize)>,
    /// Closed trigger that a trigger within `hold` input samples reopens.
    pending: Option<(Trigger, usize)>,
    hold: usize,
}

impl MatchedFilter {
    /// Correlates input at `rate` with `templates` (each resampled to the
    /// detector rate and cut to `spec.length_s` after its start).
    pub fn new(templates: &[&TemplateTrace], rate: f64, spec: &MatchedSpec) -> Result<Self, String> {
        let top = templates.iter().map(|t| t.rate).fold(0.0, f64::max);
        if templates.is_empty() || top <= 0.0 {
            return Err("no templates for this sensor".to_string());
        }
        if !(spec.on > spec.off && spec.on <= 1.0) {
            return Err("need trigger on > off, at most 1".to_string());
        }
        let decimate = ((rate / top).floor() as usize).max(1);
        let at = rate / decimate as f64;
        let kernels: Vec<Kernel> = templates
            .iter()
            .filter_map(|t| {
                let span = (t.samples.len() as f64 / t.rate).min(spec.length_s.max(t.pre_s + 1.0 / t.rate));
                let n = (span * at).floor() as usize;
                let values: Vec<f64> = (0..n)
                    .map(|k| {
                        let x = k as f64 / at * t.rate;
                        let i = (x.floor() as usize).min(t.samples.len() - 1);
                        let f = x - i as f64;
                        let b = t.samples.get(i + 1).copied().unwrap_or(t.samples[i]);
                        f64::from(t.samples[i]) * (1.0 - f) + f64::from(b) * f
                    })
                    .collect();
                let mean = values.iter().sum::<f64>() / n.max(1) as f64;
                let values: Vec<f64> = values.into_iter().map(|v| v - mean).collect();
                let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
                (n >= 8 && norm > 0.0).then(|| Kernel { values, norm, onset: (t.pre_s * at).round() as usize })
            })
            .collect();
        if kernels.is_empty() {
            return Err("templates shorter than 8 samples at this rate".to_string());
        }
        let longest = kernels.iter().map(|k| k.values.len()).max().expect("non-empty");
        let hold = kernels.iter().map(|k| k.values.len()).min().expect("non-empty") / 2 * decimate;
        Ok(Self {
            decimate,
            kernels,
            ring: VecDeque::with_capacity(longest + 1),
            longest,
            partial: (0.0, 0),
            seen: 0,
            on: spec.on,
            off: spec.off,
            active: None,
            pending: None,
            hold,
        })
    }

    /// Input samples per correlation step.
    pub fn decimation(&self) -> usize {
        self.decimate
    }

    /// Best correlation over the kernels ending at the newest sample, with the onset it implies (input samples).
    fn correlate(&self) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for kernel in &self.kernels {
            let n = kernel.values.len();
            if self.ring.len() < n {
                continue;
            }
            let window = self.ring.range(self.ring.len() - n..);
            let (mut s, mut s2, mut dot) = (0.0, 0.0, 0.0);
            for (x, k) in window.zip(&kernel.values) {
                s += x;
                s2 += x * x;
                dot += x * k;
            }
            let energy = s2 - s * s / n as f64;
            if energy <= 1e-12 * n as f64 {
                continue;
            }
            let cc = dot / (kernel.norm * energy.sqrt());
            let onset = (self.seen - n + kernel.onset) * self.decimate;
            if best.is_none_or(|b| cc > b.0) {
                best = Some((cc, onset));
            }
        }
        best
    }

    fn step(&mut self, x: f64) -> Option<Trigger> {
        self.partial.0 += x;
        self.partial.1 += 1;
        if self.partial.1 < self.decimate {
            return None;
        }
        let value = self.partial.0 / self.decimate as f64;
        self.partial = (0.0, 0);
        self.ring.push_back(value);
        if self.ring.len() > self.longest {
            self.ring.pop_front();
        }
        self.seen += 1;
        let index = self.seen * self.decimate;
        let expired = self.pending.as_ref().is_some_and(|(t, _)| index > t.off + self.hold);
        let done = if expired { self.pending.take().map(|(t, best)| Trigger { on: best, ..t }) } else { None };
        let Some((cc, onset)) = self.correlate() else { return done };
        match &mut self.active {
            None if cc >= self.on => {
                self.active = Some(match self.pending.take() {
                    Some((t, best)) if cc <= t.peak => (t, best),
                    Some((t, _)) => (Trigger { peak: cc, ..t }, onset),
                    None => (Trigger { on: onset, off: index, peak: cc }, onset),
                });
                done
            }
            None => done,
            Some((t, _)) if cc < self.off => {
                t.off = index;
                self.pending = self.active.take();
                done
            }
            Some((t, best)) => {
                if cc > t.peak {
                    t.peak = cc;
                    *best = onset;
                }
                None
            }
        }
    }
}

impl Detector for MatchedFilter {
    fn process(&mut self, chunk: &[f32]) -> Vec<Trigger> {
        chunk.iter().filter_map(|&v| self.step(if v.is_finite() { f64::from(v) } else { 0.0 })).collect()
    }

    fn finish(&mut self) -> Vec<Trigger> {
        let end = self.seen * self.decimate;
        let open = self.active.take().map(|(t, best)| Trigger { on: best, off: end.max(best + 1), ..t });
        self.pending.take().map(|(t, best)| Trigger { on: best, ..t }).into_iter().chain(open).collect()
    }
}

/// Runs a matched filter over `values` in chunks of `chunk` samples.
pub fn matched(templates: &[&TemplateTrace], rate: f64, spec: &MatchedSpec, values: &[f32], chunk: usize) -> Result<Vec<Trigger>, String> {
    let mut filter = MatchedFilter::new(templates, rate, spec)?;
    let mut out: Vec<Trigger> = values.chunks(chunk.max(1)).flat_map(|c| filter.process(c)).collect();
    out.extend(filter.finish());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Id;
    use crate::phenomenology::PhenomenologyId;

    /// A 1 s pre-onset quiet, then a decaying two-tone burst, at `rate`.
    fn burst(rate: f64, len_s: f64) -> Vec<f32> {
        (0..(len_s * rate) as usize)
            .map(|i| {
                let t = i as f64 / rate - 1.0;
                if t < 0.0 { 0.0 } else { ((std::f64::consts::TAU * 4.0 * t).sin() + 0.5 * (std::f64::consts::TAU * 9.0 * t).sin()) as f32 * (-t / 1.0).exp() as f32 }
            })
            .collect()
    }

    /// Noise at 100 Hz with bursts of `amplitudes` at the onsets (seconds).
    fn record(onsets: &[(f64, f32)], len_s: f64) -> Vec<f32> {
        let shape = burst(100.0, 6.0);
        let mut state = 0x51ed_270b_u32;
        let mut x: Vec<f32> = (0..(len_s * 100.0) as usize)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                0.1 * ((state % 2001) as f32 / 1000.0 - 1.0)
            })
            .collect();
        for &(at, a) in onsets {
            let start = ((at - 1.0) * 100.0) as usize;
            for (j, v) in shape.iter().enumerate() {
                if let Some(slot) = x.get_mut(start + j) {
                    *slot += a * v;
                }
            }
        }
        x
    }

    fn template() -> TemplateTrace {
        TemplateTrace { sensor: Id::generate(), phenomenology: PhenomenologyId::new("seismic"), rate: 40.0, pre_s: 1.0, samples: burst(40.0, 5.0) }
    }

    #[test]
    fn weak_repeats_are_found_at_their_onset() {
        // The second repeat peaks near the noise level.
        let onsets = [(20.0, 1.0), (55.0, 0.15), (90.0, 0.3)];
        let x = record(&onsets, 120.0);
        let t = template();
        let triggers = matched(&[&t], 100.0, &MatchedSpec::default(), &x, usize::MAX).unwrap();
        assert_eq!(triggers.len(), 3, "{triggers:?}");
        for (trigger, (at, _)) in triggers.iter().zip(onsets) {
            let onset = trigger.on as f64 / 100.0;
            assert!((onset - at).abs() <= 0.03, "{trigger:?} for {at}");
            assert!(trigger.off > trigger.on && trigger.peak > 0.5);
        }
    }

    #[test]
    fn chunked_runs_match_whole_runs_exactly() {
        let x = record(&[(20.0, 1.0), (61.0, 0.2)], 90.0);
        let t = template();
        let spec = MatchedSpec::default();
        let whole = matched(&[&t], 100.0, &spec, &x, usize::MAX).unwrap();
        for chunk in [1, 3, 997, 4096] {
            assert_eq!(matched(&[&t], 100.0, &spec, &x, chunk).unwrap(), whole, "chunk {chunk}");
        }
        assert!(matched(&[], 100.0, &spec, &x, 10).is_err());
    }
}
