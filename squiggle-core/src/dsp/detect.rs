//! Streaming detectors. Each keeps its state between [`Detector::process`]
//! calls, so a file processed whole and a stream processed in chunks give
//! identical triggers, bit for bit: every running sum is updated in the same
//! order, and resynchronised from its buffer at fixed absolute sample counts.

use std::collections::VecDeque;

use super::pick::{StaLta, Trigger};

/// Running sums are recomputed from their buffer every this many samples.
const RESYNC: usize = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DetectorSpec {
    /// Energy ratio of a short window over the long window before it
    /// (classic), or of exponential averages (recursive).
    StaLta { sta_s: f64, lta_s: f64, on: f64, off: f64, recursive: bool },
    /// RMS over a window above an absolute level, in the trace's units.
    Energy { window_s: f64, on: f64, off: f64 },
    /// Excess kurtosis over a window: impulsive onsets in Gaussian noise.
    Kurtosis { window_s: f64, on: f64, off: f64 },
}

impl DetectorSpec {
    pub fn label(&self) -> &'static str {
        match self {
            Self::StaLta { recursive: false, .. } => "STA/LTA",
            Self::StaLta { recursive: true, .. } => "Recursive STA/LTA",
            Self::Energy { .. } => "Energy",
            Self::Kurtosis { .. } => "Kurtosis",
        }
    }

    /// Stage name for provenance.
    pub fn stage(&self) -> &'static str {
        match self {
            Self::StaLta { .. } => "detect.sta-lta",
            Self::Energy { .. } => "detect.energy",
            Self::Kurtosis { .. } => "detect.kurtosis",
        }
    }

    /// Averaging window in samples: the span an onset refinement looks back over.
    pub fn window(&self, rate: f64) -> usize {
        let seconds = match self {
            Self::StaLta { sta_s, .. } => *sta_s,
            Self::Energy { window_s, .. } | Self::Kurtosis { window_s, .. } => *window_s,
        };
        ((seconds * rate).round() as usize).max(1)
    }

    pub fn build(&self, rate: f64) -> Result<Box<dyn Detector>, String> {
        let samples = |s: f64| {
            let n = (s * rate).round();
            if n >= 1.0 && n.is_finite() { Ok(n as usize) } else { Err(format!("{s} s is under a sample at {rate} Hz")) }
        };
        let thresholds = |on: f64, off: f64| {
            if on > off && off.is_finite() { Ok(Triggering::new(on, off)) } else { Err("need trigger on > off".to_string()) }
        };
        Ok(match *self {
            Self::StaLta { sta_s, lta_s, on, off, recursive } => {
                let (sta, lta) = (samples(sta_s)?, samples(lta_s)?);
                if lta <= sta {
                    return Err("LTA must be longer than STA".to_string());
                }
                let function = if recursive { Function::Recursive(StaLta::new(sta, lta)) } else { Function::Classic(Classic::new(sta, lta)) };
                Box::new(Streaming { function, trigger: thresholds(on, off)? })
            }
            Self::Energy { window_s, on, off } => {
                Box::new(Streaming { function: Function::Energy(Moments::new(samples(window_s)?)), trigger: thresholds(on, off)? })
            }
            Self::Kurtosis { window_s, on, off } => {
                let n = samples(window_s)?;
                if n < 4 {
                    return Err("kurtosis needs a window of at least 4 samples".to_string());
                }
                Box::new(Streaming { function: Function::Kurtosis(Moments::new(n)), trigger: thresholds(on, off)? })
            }
        })
    }
}

pub trait Detector: Send {
    /// Triggers that closed within `chunk`; indices count from the stream start.
    fn process(&mut self, chunk: &[f32]) -> Vec<Trigger>;
    /// Closes a trigger still open at the end of the stream.
    fn finish(&mut self) -> Vec<Trigger>;
    /// No trigger reported later has its `on` before this sample index.
    fn settled(&self) -> usize;
}

/// Runs `spec` over `values` in chunks of `chunk` samples.
pub fn detect(spec: &DetectorSpec, rate: f64, values: &[f32], chunk: usize) -> Result<Vec<Trigger>, String> {
    let mut detector = spec.build(rate)?;
    let mut out: Vec<Trigger> = values.chunks(chunk.max(1)).flat_map(|c| detector.process(c)).collect();
    out.extend(detector.finish());
    Ok(out)
}

/// Gaps count as silence.
fn clean(v: f32) -> f64 {
    if v.is_finite() { f64::from(v) } else { 0.0 }
}

/// On/off hysteresis over a characteristic function.
struct Triggering {
    on: f64,
    off: f64,
    index: usize,
    active: Option<Trigger>,
}

impl Triggering {
    fn new(on: f64, off: f64) -> Self {
        Self { on, off, index: 0, active: None }
    }

    /// `None` while the function is warming up.
    fn step(&mut self, value: Option<f64>) -> Option<Trigger> {
        let i = self.index;
        self.index += 1;
        let value = value?;
        match &mut self.active {
            None if value >= self.on => {
                self.active = Some(Trigger { on: i, off: i + 1, peak: value });
                None
            }
            None => None,
            Some(t) if value < self.off => {
                t.off = i;
                self.active.take()
            }
            Some(t) => {
                t.peak = t.peak.max(value);
                None
            }
        }
    }

    fn finish(&mut self) -> Option<Trigger> {
        let mut t = self.active.take()?;
        t.off = self.index;
        Some(t)
    }
}

/// Classic STA/LTA over a ring of energies.
struct Classic {
    sta: usize,
    lta: usize,
    ring: VecDeque<f64>,
    short: f64,
    long: f64,
    seen: usize,
}

impl Classic {
    fn new(sta: usize, lta: usize) -> Self {
        Self { sta, lta, ring: VecDeque::with_capacity(sta + lta + 1), short: 0.0, long: 0.0, seen: 0 }
    }

    fn step(&mut self, x: f64) -> Option<f64> {
        let e = x * x;
        self.ring.push_back(e);
        self.short += e;
        if self.ring.len() > self.sta {
            let moved = self.ring[self.ring.len() - 1 - self.sta];
            self.short -= moved;
            self.long += moved;
        }
        if self.ring.len() > self.sta + self.lta {
            self.long -= self.ring.pop_front().expect("non-empty");
        }
        self.seen += 1;
        if self.seen.is_multiple_of(RESYNC) {
            let split = self.ring.len().saturating_sub(self.sta);
            self.long = self.ring.range(..split).sum();
            self.short = self.ring.range(split..).sum();
        }
        if self.ring.len() < self.sta + self.lta {
            return None;
        }
        let long = self.long / self.lta as f64;
        Some(if long > 0.0 { (self.short / self.sta as f64) / long } else { 0.0 })
    }
}

/// Running power sums over a window.
struct Moments {
    n: usize,
    ring: VecDeque<f64>,
    sums: [f64; 4],
    seen: usize,
}

impl Moments {
    fn new(n: usize) -> Self {
        Self { n, ring: VecDeque::with_capacity(n + 1), sums: [0.0; 4], seen: 0 }
    }

    fn powers(x: f64) -> [f64; 4] {
        let x2 = x * x;
        [x, x2, x2 * x, x2 * x2]
    }

    /// `None` until the window is full.
    fn step(&mut self, x: f64) -> Option<&[f64; 4]> {
        self.ring.push_back(x);
        for (s, p) in self.sums.iter_mut().zip(Self::powers(x)) {
            *s += p;
        }
        if self.ring.len() > self.n {
            let old = self.ring.pop_front().expect("non-empty");
            for (s, p) in self.sums.iter_mut().zip(Self::powers(old)) {
                *s -= p;
            }
        }
        self.seen += 1;
        if self.seen.is_multiple_of(RESYNC) {
            self.sums = [0.0; 4];
            for &v in &self.ring {
                for (s, p) in self.sums.iter_mut().zip(Self::powers(v)) {
                    *s += p;
                }
            }
        }
        (self.ring.len() == self.n).then_some(&self.sums)
    }

    fn rms(sums: &[f64; 4], n: usize) -> f64 {
        (sums[1] / n as f64).max(0.0).sqrt()
    }

    fn excess_kurtosis(sums: &[f64; 4], n: usize) -> f64 {
        let n = n as f64;
        let [m1, m2, m3, m4] = sums.map(|s| s / n);
        let var = m2 - m1 * m1;
        if var <= 0.0 {
            return 0.0;
        }
        let central4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.powi(4);
        central4 / (var * var) - 3.0
    }
}

enum Function {
    Classic(Classic),
    Recursive(StaLta),
    Energy(Moments),
    Kurtosis(Moments),
}

impl Function {
    fn step(&mut self, v: f32) -> Option<f64> {
        let x = clean(v);
        match self {
            Self::Classic(c) => c.step(x),
            Self::Recursive(r) => {
                let ratio = r.process(&[v])[0];
                (r.warm()).then_some(ratio)
            }
            Self::Energy(m) => {
                let n = m.n;
                m.step(x).map(|s| Moments::rms(s, n))
            }
            Self::Kurtosis(m) => {
                let n = m.n;
                m.step(x).map(|s| Moments::excess_kurtosis(s, n))
            }
        }
    }
}

struct Streaming {
    function: Function,
    trigger: Triggering,
}

impl Detector for Streaming {
    fn process(&mut self, chunk: &[f32]) -> Vec<Trigger> {
        chunk.iter().filter_map(|&v| {
            let value = self.function.step(v);
            self.trigger.step(value)
        }).collect()
    }

    fn finish(&mut self) -> Vec<Trigger> {
        self.trigger.finish().into_iter().collect()
    }

    fn settled(&self) -> usize {
        self.trigger.active.as_ref().map_or(self.trigger.index, |t| t.on)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise, with impulsive bursts at the given samples.
    fn signal(len: usize, bursts: &[usize], rate: f64) -> Vec<f32> {
        let mut state = 0x1234_5679_u32;
        (0..len)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = (f64::from(state % 2001) / 1000.0 - 1.0) * 0.02;
                let burst: f64 = bursts
                    .iter()
                    .map(|&b| {
                        let t = (i as f64 - b as f64) / rate;
                        if t >= 0.0 { (std::f64::consts::TAU * 8.0 * t).sin() * (-t / 0.3).exp() } else { 0.0 }
                    })
                    .sum();
                (noise + burst) as f32
            })
            .collect()
    }

    fn specs() -> [DetectorSpec; 4] {
        [
            DetectorSpec::StaLta { sta_s: 0.3, lta_s: 5.0, on: 4.0, off: 1.5, recursive: false },
            DetectorSpec::StaLta { sta_s: 0.3, lta_s: 5.0, on: 4.0, off: 1.5, recursive: true },
            DetectorSpec::Energy { window_s: 0.2, on: 0.1, off: 0.05 },
            DetectorSpec::Kurtosis { window_s: 1.0, on: 3.0, off: 1.0 },
        ]
    }

    #[test]
    fn every_detector_finds_each_burst_once() {
        let bursts = [1500, 4000, 7000];
        let x = signal(9000, &bursts, 100.0);
        for spec in specs() {
            let triggers = detect(&spec, 100.0, &x, usize::MAX).unwrap();
            assert_eq!(triggers.len(), 3, "{}: {triggers:?}", spec.label());
            for (t, &b) in triggers.iter().zip(&bursts) {
                assert!(t.on >= b && t.on < b + 60, "{}: {t:?} for {b}", spec.label());
                assert!(t.off > t.on);
            }
        }
    }

    #[test]
    fn chunked_runs_match_whole_runs_exactly() {
        let x = signal(200_000, &[30_000, 90_000, 150_000], 100.0);
        for spec in specs() {
            let whole = detect(&spec, 100.0, &x, usize::MAX).unwrap();
            for chunk in [1, 7, 4096, 65_537] {
                assert_eq!(detect(&spec, 100.0, &x, chunk).unwrap(), whole, "{} chunk {chunk}", spec.label());
            }
        }
    }

    #[test]
    fn a_trigger_open_at_the_end_is_closed() {
        let x = signal(3000, &[2985], 100.0);
        let spec = DetectorSpec::StaLta { sta_s: 0.3, lta_s: 5.0, on: 4.0, off: 1.5, recursive: false };
        let triggers = detect(&spec, 100.0, &x, 100).unwrap();
        assert_eq!(triggers.last().map(|t| t.off), Some(3000));
    }

    #[test]
    fn later_triggers_start_after_the_settled_index() {
        let x = signal(9000, &[1500, 4000, 7000], 100.0);
        for spec in specs() {
            let mut detector = spec.build(100.0).unwrap();
            let mut settled = 0;
            for chunk in x.chunks(50) {
                for t in detector.process(chunk) {
                    assert!(t.on >= settled, "{}: {t:?} before {settled}", spec.label());
                }
                settled = detector.settled();
            }
        }
    }

    #[test]
    fn invalid_specs_are_refused() {
        assert!(DetectorSpec::StaLta { sta_s: 1.0, lta_s: 0.5, on: 4.0, off: 1.0, recursive: false }.build(100.0).is_err());
        assert!(DetectorSpec::Energy { window_s: 0.001, on: 1.0, off: 0.5 }.build(100.0).is_err());
        assert!(DetectorSpec::Kurtosis { window_s: 1.0, on: 1.0, off: 2.0 }.build(100.0).is_err());
    }
}
