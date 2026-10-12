//! Association: grouping detections on many traces into candidate events.
//!
//! A source within `radius_m` of a sensor emits at most `radius_m / speed`
//! before an arrival there, so each arrival bounds the origin time to
//! `[t − radius/speed, t]` (or `[t − max_window, t]` without a speed or
//! location). Arrivals join a cluster while those intervals still intersect,
//! so the coincidence window follows each phenomenology's physics: EMP
//! arrivals must agree to microseconds, infrasound within minutes.
//!
//! Clusters are seeded from the earliest unused arrival. One arrival per
//! trace; later arrivals on a member trace within its coda (the spread of its
//! phases) are absorbed, so S waves do not seed events of their own. A
//! cluster needs `min_sensors` distinct sensors; with enough located members
//! it is located, and arrivals whose residual exceeds `outlier_sigma` are
//! dropped (worst first) and the event relocated.
//!
//! Slow arrivals, whose origin window is wider than `max_window_s` (e.g.
//! infrasound at a 50 km radius: 147 s), would join any cluster and pull its
//! location. They are held back: once an event is located from the others,
//! each such trace contributes its arrival nearest the predicted time, if
//! within `outlier_sigma`. The rest cluster among themselves afterwards.

use crate::id::Id;
use crate::locate::{Fit, LocateSettings, Observation, locate, travel_time};
use crate::sensor::Location;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arrival {
    pub trace: Id,
    pub sensor: Id,
    /// Onset, seconds after a common reference (clock-corrected).
    pub time: f64,
    /// First-phase speed; `None` for unknown propagation.
    pub speed_m_s: Option<f64>,
    pub location: Option<Location>,
    pub sigma_s: f64,
    /// Later arrivals on this trace within this many seconds belong to the same event.
    pub coda_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AssociateSettings {
    /// Farthest a source may be from a sensor that records it, metres.
    pub radius_m: f64,
    /// Origin window for arrivals without a speed or location, seconds.
    pub max_window_s: f64,
    pub min_sensors: usize,
    /// Residual, in pick standard deviations, beyond which an arrival is dropped.
    pub outlier_sigma: f64,
    /// Deepest source the locator tries.
    pub max_depth_m: f64,
}

impl Default for AssociateSettings {
    fn default() -> Self {
        Self { radius_m: 50_000.0, max_window_s: 10.0, min_sensors: 3, outlier_sigma: 20.0, max_depth_m: 20_000.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// Indices into the input, in time order.
    pub arrivals: Vec<usize>,
    /// Arrivals absorbed as later phases of members.
    pub absorbed: Vec<usize>,
    /// Located when at least four members have a location and speed.
    pub fit: Option<Fit>,
}

fn origin_bounds(a: &Arrival, settings: &AssociateSettings) -> (f64, f64) {
    let reach = match (a.speed_m_s, a.location) {
        (Some(speed), Some(_)) if speed > 0.0 => settings.radius_m / speed,
        _ => settings.max_window_s,
    };
    (a.time - reach, a.time)
}

/// Whether `a`'s origin window is too wide to seed or join clusters of faster arrivals.
fn is_wide(a: &Arrival, settings: &AssociateSettings) -> bool {
    let (lo, hi) = origin_bounds(a, settings);
    hi - lo > settings.max_window_s
}

fn distinct_sensors(arrivals: &[Arrival], members: &[usize]) -> usize {
    let mut sensors: Vec<Id> = members.iter().map(|&i| arrivals[i].sensor).collect();
    sensors.sort_unstable();
    sensors.dedup();
    sensors.len()
}

/// Locates `members`; `None` without four located arrivals.
fn fit(arrivals: &[Arrival], members: &[usize], settings: &AssociateSettings) -> Option<(Vec<usize>, Fit)> {
    let located: Vec<usize> = members.iter().copied().filter(|&i| arrivals[i].speed_m_s.is_some() && arrivals[i].location.is_some()).collect();
    if located.len() < 4 {
        return None;
    }
    let observations: Vec<Observation> = located
        .iter()
        .map(|&i| {
            let a = &arrivals[i];
            Observation { location: a.location.expect("filtered"), speed_m_s: a.speed_m_s.expect("filtered"), time: a.time, sigma_s: a.sigma_s }
        })
        .collect();
    let locate_settings = LocateSettings { max_depth_m: settings.max_depth_m, ..LocateSettings::default() };
    Some((located, locate(&observations, &locate_settings)?))
}

pub fn associate(arrivals: &[Arrival], settings: &AssociateSettings) -> Vec<Candidate> {
    let mut order: Vec<usize> = (0..arrivals.len()).filter(|&i| arrivals[i].time.is_finite()).collect();
    order.sort_by(|&a, &b| arrivals[a].time.total_cmp(&arrivals[b].time));
    let (wide, narrow): (Vec<usize>, Vec<usize>) = order.iter().partition(|&&i| is_wide(&arrivals[i], settings));
    let mut used = vec![false; arrivals.len()];
    let mut candidates = cluster(arrivals, &narrow, &mut used, settings);

    for candidate in &mut candidates {
        let Some(fit) = &candidate.fit else { continue };
        let mut traces: Vec<Id> = candidate.arrivals.iter().map(|&i| arrivals[i].trace).collect();
        for &i in &wide {
            let a = &arrivals[i];
            if used[i] || traces.contains(&a.trace) {
                continue;
            }
            let (Some(location), Some(speed)) = (a.location, a.speed_m_s) else { continue };
            let predicted = fit.origin + travel_time(fit.source, &location, speed);
            let nearest = wide
                .iter()
                .copied()
                .filter(|&j| !used[j] && arrivals[j].trace == a.trace)
                .min_by(|&x, &y| (arrivals[x].time - predicted).abs().total_cmp(&(arrivals[y].time - predicted).abs()))
                .expect("i qualifies");
            let b = &arrivals[nearest];
            if (b.time - predicted).abs() <= settings.outlier_sigma * b.sigma_s {
                used[nearest] = true;
                traces.push(b.trace);
                candidate.arrivals.push(nearest);
            }
        }
        candidate.arrivals.sort_by(|&a, &b| arrivals[a].time.total_cmp(&arrivals[b].time));
    }
    candidates.extend(cluster(arrivals, &wide, &mut used, settings));
    candidates.sort_by(|a, b| arrivals[a.arrivals[0]].time.total_cmp(&arrivals[b.arrivals[0]].time));
    candidates
}

/// Clusters `order` (time-sorted indices), skipping and marking `used`.
fn cluster(arrivals: &[Arrival], order: &[usize], used: &mut [bool], settings: &AssociateSettings) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for (position, &seed) in order.iter().enumerate() {
        if used[seed] {
            continue;
        }
        let (mut lo, mut hi) = origin_bounds(&arrivals[seed], settings);
        let mut members = vec![seed];
        for &next in &order[position + 1..] {
            let a = &arrivals[next];
            if used[next] || members.iter().any(|&m| arrivals[m].trace == a.trace) {
                continue;
            }
            let (a_lo, a_hi) = origin_bounds(a, settings);
            if a_lo <= hi && a_hi >= lo {
                (lo, hi) = (lo.max(a_lo), hi.min(a_hi));
                members.push(next);
            }
        }

        // Drop outliers by location residual, worst first.
        let mut located = None;
        while distinct_sensors(arrivals, &members) >= settings.min_sensors {
            let Some((observed, fit)) = fit(arrivals, &members, settings) else {
                break;
            };
            let worst = observed
                .iter()
                .zip(&fit.residuals)
                .map(|(&i, r)| (i, (r / arrivals[i].sigma_s).abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match worst {
                Some((i, score)) if score > settings.outlier_sigma => members.retain(|&m| m != i),
                _ => {
                    located = Some(fit);
                    break;
                }
            }
        }
        if distinct_sensors(arrivals, &members) < settings.min_sensors {
            // Not an event; only the seed is spent, so its partners can seed or join others.
            used[seed] = true;
            continue;
        }

        let mut absorbed = Vec::new();
        for &m in &members {
            used[m] = true;
            let (trace, from, coda) = (arrivals[m].trace, arrivals[m].time, arrivals[m].coda_s);
            for &i in order {
                if !used[i] && arrivals[i].trace == trace && arrivals[i].time > from && arrivals[i].time <= from + coda {
                    used[i] = true;
                    absorbed.push(i);
                }
            }
        }
        members.sort_by(|&a, &b| arrivals[a].time.total_cmp(&arrivals[b].time));
        candidates.push(Candidate { arrivals: members, absorbed, fit: located });
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locate::{Source, travel_time};

    const C: f64 = 299_792_458.0;

    fn at(east_km: f64, north_km: f64) -> Location {
        let latitude = 35.0 + (north_km * 1e3 / crate::locate::EARTH_RADIUS_M).to_degrees();
        let longitude = -106.0 + (east_km * 1e3 / (crate::locate::EARTH_RADIUS_M * 35f64.to_radians().cos())).to_degrees();
        Location { latitude, longitude, elevation_m: 0.0 }
    }

    struct Net {
        sensors: Vec<(Id, Id, Location, f64, f64)>,
    }

    impl Net {
        /// Six seismic stations (P 6 km/s, coda 7 s) and three EMP sensors.
        fn new() -> Self {
            let mut sensors = Vec::new();
            for (e, n) in [(12.0, 5.0), (-8.0, 14.0), (-15.0, -6.0), (5.0, -18.0), (22.0, -9.0), (-3.0, 30.0)] {
                sensors.push((Id::generate(), Id::generate(), at(e, n), 6000.0, 7.0));
            }
            for (e, n) in [(4.0, 3.0), (-6.0, -2.0), (1.0, -9.0)] {
                sensors.push((Id::generate(), Id::generate(), at(e, n), C, 0.0));
            }
            Self { sensors }
        }

        fn arrivals(&self, source: Source, origin: f64, which: std::ops::Range<usize>) -> Vec<Arrival> {
            self.sensors[which]
                .iter()
                .map(|&(trace, sensor, location, speed, coda)| Arrival {
                    trace,
                    sensor,
                    time: origin + travel_time(source, &location, speed),
                    speed_m_s: Some(speed),
                    location: Some(location),
                    sigma_s: if speed > 1e6 { 1e-6 } else { 0.01 },
                    coda_s: coda,
                })
                .collect()
        }
    }

    fn source(e: f64, n: f64) -> Source {
        let l = at(e, n);
        Source { latitude: l.latitude, longitude: l.longitude, depth_m: 2_000.0 }
    }

    #[test]
    fn separate_events_and_noise_are_told_apart() {
        let net = Net::new();
        let mut arrivals = net.arrivals(source(0.0, 0.0), 10.0, 0..9);
        arrivals.extend(net.arrivals(source(5.0, 5.0), 30.0, 0..9));
        // S phases on the first event's seismic stations, and lone noise triggers.
        let s_waves: Vec<Arrival> = arrivals[..6].iter().map(|a| Arrival { time: a.time + 3.0, ..*a }).collect();
        arrivals.extend(s_waves);
        for (i, t) in [(0, 55.0), (2, 70.0), (7, 80.0)] {
            arrivals.push(Arrival { time: t, ..arrivals[i] });
        }
        let candidates = associate(&arrivals, &AssociateSettings::default());
        assert_eq!(candidates.len(), 2, "{candidates:?}");
        for (c, origin) in candidates.iter().zip([10.0, 30.0]) {
            assert_eq!(c.arrivals.len(), 9);
            let fit = c.fit.as_ref().unwrap();
            assert!((fit.origin - origin).abs() < 1e-4, "{}", fit.origin);
        }
        assert_eq!(candidates[0].absorbed.len(), 6);
    }

    #[test]
    fn a_wrong_arrival_is_dropped_by_its_residual() {
        let net = Net::new();
        let mut arrivals = net.arrivals(source(0.0, 0.0), 10.0, 0..9);
        arrivals[2].time += 1.5;
        let candidates = associate(&arrivals, &AssociateSettings::default());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].arrivals.len(), 8);
        assert!(!candidates[0].arrivals.contains(&2));
    }

    #[test]
    fn slow_arrivals_attach_only_where_predicted() {
        let net = Net::new();
        let mut arrivals = net.arrivals(source(0.0, 0.0), 10.0, 0..6);
        arrivals.extend(net.arrivals(source(4.0, -3.0), 40.0, 0..6));
        // Two microphones hear the second source only, plus a stray arrival.
        let mics: Vec<(Id, Id, Location)> = [(1.0, 0.5), (-1.5, -1.0)].iter().map(|&(e, n)| (Id::generate(), Id::generate(), at(e, n))).collect();
        for &(trace, sensor, location) in &mics {
            let time = 40.0 + travel_time(source(4.0, -3.0), &location, 340.0);
            arrivals.push(Arrival { trace, sensor, time, speed_m_s: Some(340.0), location: Some(location), sigma_s: 0.05, coda_s: 0.0 });
        }
        arrivals.push(Arrival { time: 25.0, ..arrivals[12] });
        let candidates = associate(&arrivals, &AssociateSettings::default());
        assert_eq!(candidates.len(), 2, "{candidates:?}");
        assert_eq!(candidates[0].arrivals.len(), 6);
        assert_eq!(candidates[1].arrivals.len(), 8);
        assert!(candidates[1].arrivals.contains(&12) && candidates[1].arrivals.contains(&13));
        let fit = candidates[1].fit.as_ref().unwrap();
        assert!((fit.origin - 40.0).abs() < 1e-4);
    }

    #[test]
    fn too_few_sensors_make_no_event() {
        let net = Net::new();
        let arrivals = net.arrivals(source(0.0, 0.0), 10.0, 0..2);
        assert!(associate(&arrivals, &AssociateSettings::default()).is_empty());
    }
}
