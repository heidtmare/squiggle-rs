//! Onset picking: an STA/LTA trigger on signal energy, refined by the
//! Akaike information criterion (Maeda 1985) over a window around the trigger.
//!
//! The classic ratio compares a short window with the long window just before
//! it; the recursive form uses exponential averages and keeps state, so a
//! stream gives the same ratio as one buffer.

/// STA/LTA and trigger thresholds, in seconds and ratio units.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PickSettings {
    pub sta_s: f64,
    pub lta_s: f64,
    /// Ratio that starts a trigger.
    pub on: f64,
    /// Ratio that ends it.
    pub off: f64,
    pub recursive: bool,
}

impl Default for PickSettings {
    fn default() -> Self {
        Self { sta_s: 0.5, lta_s: 10.0, on: 4.0, off: 1.5, recursive: false }
    }
}

impl PickSettings {
    /// Window lengths in samples at `rate`.
    pub fn lengths(&self, rate: f64) -> Result<(usize, usize), String> {
        let sta = (self.sta_s * rate).round();
        let lta = (self.lta_s * rate).round();
        if !(sta >= 1.0 && lta > sta && lta.is_finite()) {
            return Err(format!("need 1 ≤ STA < LTA samples (got {sta} and {lta})"));
        }
        if !(self.on > self.off && self.off > 0.0) {
            return Err("need trigger on > off > 0".to_string());
        }
        Ok((sta as usize, lta as usize))
    }
}

/// Energy characteristic function; gaps count as silence.
fn energy(values: &[f32]) -> impl Iterator<Item = f64> + '_ {
    values.iter().map(|&v| if v.is_finite() { f64::from(v).powi(2) } else { 0.0 })
}

/// Classic ratio: mean energy over the last `sta` samples over the mean of the
/// `lta` samples before them. Zero until both windows are full.
pub fn sta_lta(values: &[f32], sta: usize, lta: usize) -> Vec<f64> {
    let mut sums = Vec::with_capacity(values.len() + 1);
    sums.push(0.0);
    for e in energy(values) {
        sums.push(sums.last().copied().unwrap_or(0.0) + e);
    }
    (0..values.len())
        .map(|i| {
            if i + 1 < sta + lta {
                return 0.0;
            }
            let short = (sums[i + 1] - sums[i + 1 - sta]) / sta as f64;
            let long = (sums[i + 1 - sta] - sums[i + 1 - sta - lta]) / lta as f64;
            if long > 0.0 { short / long } else { 0.0 }
        })
        .collect()
}

/// Recursive STA/LTA (exponential averages). Zero for the first `lta` samples.
#[derive(Clone, Debug)]
pub struct StaLta {
    sta: f64,
    lta: f64,
    short: f64,
    long: f64,
    seen: usize,
    warmup: usize,
}

impl StaLta {
    pub fn new(sta: usize, lta: usize) -> Self {
        Self { sta: sta as f64, lta: lta as f64, short: 0.0, long: 0.0, seen: 0, warmup: lta }
    }

    pub fn process(&mut self, chunk: &[f32]) -> Vec<f64> {
        energy(chunk)
            .map(|e| {
                self.short += (e - self.short) / self.sta;
                self.long += (e - self.long) / self.lta;
                self.seen += 1;
                if self.seen > self.warmup && self.long > 0.0 { self.short / self.long } else { 0.0 }
            })
            .collect()
    }
}

/// A span where the ratio stayed above `off` after reaching `on`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trigger {
    pub on: usize,
    /// Exclusive.
    pub off: usize,
    pub peak: f64,
}

pub fn triggers(ratio: &[f64], on: f64, off: f64) -> Vec<Trigger> {
    let mut out = Vec::new();
    let mut active: Option<Trigger> = None;
    for (i, &r) in ratio.iter().enumerate() {
        match &mut active {
            None if r >= on => active = Some(Trigger { on: i, off: i + 1, peak: r }),
            None => {}
            Some(t) if r < off => {
                t.off = i;
                out.push(*t);
                active = None;
            }
            Some(t) => t.peak = t.peak.max(r),
        }
    }
    if let Some(mut t) = active {
        t.off = ratio.len();
        out.push(t);
    }
    out
}

/// Maeda AIC over `values`: entry `k` is `k·ln σ²(x[..k]) + (n−k−1)·ln σ²(x[k..])`;
/// the minimum marks the change from noise to signal. Splits leaving fewer than
/// two samples on a side (whose variance is degenerate) are infinite.
pub fn aic(values: &[f32]) -> Vec<f64> {
    let n = values.len();
    if n < 5 {
        return vec![f64::INFINITY; n];
    }
    let x: Vec<f64> = values.iter().map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 }).collect();
    let mut s1 = vec![0.0; n + 1];
    let mut s2 = vec![0.0; n + 1];
    for i in 0..n {
        s1[i + 1] = s1[i] + x[i];
        s2[i + 1] = s2[i] + x[i] * x[i];
    }
    let raw = |a: usize, b: usize| {
        let m = (b - a) as f64;
        let mean = (s1[b] - s1[a]) / m;
        (s2[b] - s2[a]) / m - mean * mean
    };
    // Constant runs (digitised silence) would give ln 0.
    let floor = raw(0, n).max(0.0) * 1e-12 + f64::MIN_POSITIVE;
    let var = |a: usize, b: usize| raw(a, b).max(floor);
    (0..n)
        .map(|k| {
            if k < 2 || k + 2 > n {
                return f64::INFINITY;
            }
            k as f64 * var(0, k).ln() + (n - k - 1) as f64 * var(k, n).ln()
        })
        .collect()
}

/// A picked onset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Onset {
    /// First sample of the signal.
    pub index: usize,
    /// Peak STA/LTA of its trigger.
    pub ratio: f64,
    /// Half-width of the AIC basin within 2 of its minimum; at least half a sample.
    pub uncertainty_s: f64,
}

/// Onsets in time order, one per trigger.
pub fn pick(values: &[f32], rate: f64, settings: &PickSettings) -> Result<Vec<Onset>, String> {
    let (sta, lta) = settings.lengths(rate)?;
    let ratio = if settings.recursive { StaLta::new(sta, lta).process(values) } else { sta_lta(values, sta, lta) };
    let mut previous_end = 0;
    let mut onsets = Vec::new();
    for trigger in triggers(&ratio, settings.on, settings.off) {
        // The trigger lags the onset by up to an STA; look well before it.
        let lo = trigger.on.saturating_sub(4 * sta + 8).max(previous_end);
        let hi = (trigger.on + sta + 8).min(values.len());
        previous_end = trigger.off;
        let curve = aic(&values[lo..hi]);
        let Some((k, &min)) = curve.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)) else {
            continue;
        };
        let within = |j: &usize| curve[*j] <= min + 2.0;
        let left = (0..=k).rev().take_while(within).last().unwrap_or(k);
        let right = (k..curve.len()).take_while(within).last().unwrap_or(k);
        if !min.is_finite() {
            continue;
        }
        let half_width = ((right - left) as f64 / 2.0).max(0.5);
        onsets.push(Onset { index: lo + k, ratio: trigger.peak, uncertainty_s: half_width / rate });
    }
    Ok(onsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Low noise, then an impulsive 5 Hz arrival at `onset`.
    fn arrival(len: usize, onset: usize, rate: f64) -> Vec<f32> {
        let mut state = 0x2545_f491_u32;
        (0..len)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = (f64::from(state % 2001) / 1000.0 - 1.0) * 0.02;
                let t = (i as f64 - onset as f64) / rate;
                let signal = if t >= 0.0 { (std::f64::consts::TAU * 5.0 * t).sin() * (-t / 0.5).exp() } else { 0.0 };
                (noise + signal) as f32
            })
            .collect()
    }

    #[test]
    fn classic_and_recursive_ratios_rise_at_the_onset() {
        let x = arrival(3000, 2000, 100.0);
        let classic = sta_lta(&x, 50, 1000);
        let recursive = StaLta::new(50, 1000).process(&x);
        for ratio in [classic, recursive] {
            assert!(ratio[1990] < 3.0, "{}", ratio[1990]);
            assert!(ratio[2030] > 10.0, "{}", ratio[2030]);
        }
    }

    #[test]
    fn recursive_ratio_streams_identically() {
        let x = arrival(3000, 2000, 100.0);
        let whole = StaLta::new(50, 1000).process(&x);
        let mut streamed = StaLta::new(50, 1000);
        let chunks: Vec<f64> = x.chunks(77).flat_map(|c| streamed.process(c)).collect();
        assert_eq!(whole, chunks);
    }

    #[test]
    fn triggers_span_on_to_off() {
        let ratio = [0.0, 5.0, 6.0, 2.0, 1.0, 0.0, 9.0];
        assert_eq!(
            triggers(&ratio, 4.0, 1.5),
            vec![Trigger { on: 1, off: 4, peak: 6.0 }, Trigger { on: 6, off: 7, peak: 9.0 }]
        );
    }

    #[test]
    fn aic_finds_a_variance_change() {
        let x = arrival(400, 250, 100.0);
        let curve = aic(&x);
        let k = curve.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert!((k as i64 - 250).abs() <= 2, "{k}");
        // Silence then a step: no ln 0, and the step is found.
        let step: Vec<f32> = (0..100).map(|i| if i < 60 { 0.0 } else { (i % 3 + 1) as f32 }).collect();
        let curve = aic(&step);
        assert_eq!(curve.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap().0, 60);
    }

    #[test]
    fn picks_land_within_two_samples() {
        for recursive in [false, true] {
            let settings = PickSettings { sta_s: 0.3, lta_s: 5.0, on: 4.0, off: 1.5, recursive };
            let onsets = pick(&arrival(3000, 2000, 100.0), 100.0, &settings).unwrap();
            assert_eq!(onsets.len(), 1, "{onsets:?}");
            assert!((onsets[0].index as i64 - 2000).abs() <= 2, "{onsets:?}");
            assert!(onsets[0].uncertainty_s >= 0.005 && onsets[0].uncertainty_s < 0.1, "{onsets:?}");
        }
    }

    #[test]
    fn noise_alone_does_not_trigger() {
        let settings = PickSettings { sta_s: 0.3, lta_s: 5.0, ..PickSettings::default() };
        assert!(pick(&arrival(3000, 99_999, 100.0), 100.0, &settings).unwrap().is_empty());
    }

    #[test]
    fn invalid_windows_are_rejected() {
        let settings = PickSettings { sta_s: 1.0, lta_s: 0.5, ..PickSettings::default() };
        assert!(pick(&[0.0; 10], 100.0, &settings).is_err());
    }
}
