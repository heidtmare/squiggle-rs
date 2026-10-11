//! Source location from arrival times.
//!
//! Travel time is a straight ray at a constant speed per phase: slant distance
//! (great-circle surface distance on a spherical Earth, combined with source
//! depth plus sensor elevation) over speed. Valid for regional distances,
//! where curvature and layering are second-order.
//!
//! A grid search over latitude, longitude and depth solves the origin time in
//! closed form at each node (weighted mean of arrival minus travel time) and
//! refines around the best node. Weights are `1/σ²`, so precise arrivals
//! (EMP, RF) anchor the origin time and slower phases resolve position.

use crate::sensor::{Location, Provenance};
use crate::time::Timestamp;

pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// A located source, as saved in the project.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Origin {
    pub latitude: f64,
    pub longitude: f64,
    pub depth_m: f64,
    pub time: Timestamp,
    /// Unweighted RMS of the residuals, seconds.
    pub rms_s: f64,
    pub provenance: Provenance,
}

impl Origin {
    pub fn source(&self) -> Source {
        Source { latitude: self.latitude, longitude: self.longitude, depth_m: self.depth_m }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Source {
    pub latitude: f64,
    pub longitude: f64,
    /// Below the surface, metres.
    pub depth_m: f64,
}

/// Great-circle distance, metres (haversine).
pub fn surface_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().min(1.0).asin()
}

/// Straight-ray distance from source to sensor, metres.
pub fn slant_distance(source: Source, sensor: &Location) -> f64 {
    let surface = surface_distance(source.latitude, source.longitude, sensor.latitude, sensor.longitude);
    surface.hypot(source.depth_m + sensor.elevation_m)
}

pub fn travel_time(source: Source, sensor: &Location, speed_m_s: f64) -> f64 {
    slant_distance(source, sensor) / speed_m_s
}

/// One arrival for the locator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub location: Location,
    pub speed_m_s: f64,
    /// Arrival, seconds after an arbitrary common reference.
    pub time: f64,
    /// One standard deviation, seconds.
    pub sigma_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocateSettings {
    /// Grid centre; the sensors' centroid when `None`.
    pub centre: Option<(f64, f64)>,
    /// Half-width of the first grid, metres; from the sensor spread when `None`.
    pub half_extent_m: Option<f64>,
    /// Deepest source tried; zero fixes the source at the surface.
    pub max_depth_m: f64,
    /// Nodes per side; odd.
    pub nodes: usize,
    pub refinements: usize,
}

impl Default for LocateSettings {
    fn default() -> Self {
        Self { centre: None, half_extent_m: None, max_depth_m: 20_000.0, nodes: 41, refinements: 8 }
    }
}

/// χ² misfit of the first grid, best over depth, row-major from the south-west.
#[derive(Clone, Debug, PartialEq)]
pub struct MisfitMap {
    pub centre: (f64, f64),
    pub half_extent_m: f64,
    pub nodes: usize,
    pub chi2: Vec<f64>,
}

impl MisfitMap {
    /// Latitude and longitude of node `(east, north)`.
    pub fn node(&self, east: usize, north: usize) -> (f64, f64) {
        let step = 2.0 * self.half_extent_m / (self.nodes - 1) as f64;
        let offset = |i: usize| -self.half_extent_m + i as f64 * step;
        displace(self.centre, offset(east), offset(north))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub source: Source,
    /// Seconds after the observations' reference.
    pub origin: f64,
    /// Observed minus predicted, seconds, in observation order.
    pub residuals: Vec<f64>,
    pub rms_s: f64,
    pub chi2: f64,
    /// Observations minus the four solved parameters (at least one).
    pub dof: usize,
    pub map: MisfitMap,
}

/// Moves `(lat, lon)` by metres east and north on a local tangent plane.
fn displace((lat, lon): (f64, f64), east: f64, north: f64) -> (f64, f64) {
    let lat2 = lat + (north / EARTH_RADIUS_M).to_degrees();
    let lon2 = lon + (east / (EARTH_RADIUS_M * lat.to_radians().cos().max(1e-6))).to_degrees();
    (lat2.clamp(-90.0, 90.0), (lon2 + 540.0).rem_euclid(360.0) - 180.0)
}

/// Origin time and χ² for a trial source.
fn evaluate(observations: &[Observation], source: Source) -> (f64, f64) {
    let mut sum_w = 0.0;
    let mut sum = 0.0;
    let travel: Vec<f64> = observations.iter().map(|o| travel_time(source, &o.location, o.speed_m_s)).collect();
    for (o, t) in observations.iter().zip(&travel) {
        let w = o.sigma_s.powi(-2);
        sum_w += w;
        sum += w * (o.time - t);
    }
    let origin = sum / sum_w;
    let chi2 = observations.iter().zip(&travel).map(|(o, t)| ((o.time - t - origin) / o.sigma_s).powi(2)).sum();
    (origin, chi2)
}

fn depths(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    if hi <= lo {
        return vec![lo];
    }
    (0..count).map(|i| lo + (hi - lo) * i as f64 / (count - 1) as f64).collect()
}

/// Grid-search location. `None` with fewer than two usable observations.
pub fn locate(observations: &[Observation], settings: &LocateSettings) -> Option<Fit> {
    let observations: Vec<Observation> = observations
        .iter()
        .copied()
        .filter(|o| o.speed_m_s > 0.0 && o.sigma_s > 0.0 && o.time.is_finite())
        .collect();
    if observations.len() < 2 {
        return None;
    }
    let count = observations.len() as f64;
    let centre = settings.centre.unwrap_or_else(|| {
        let lat = observations.iter().map(|o| o.location.latitude).sum::<f64>() / count;
        let lon = observations.iter().map(|o| o.location.longitude).sum::<f64>() / count;
        (lat, lon)
    });
    let spread = observations
        .iter()
        .map(|o| surface_distance(centre.0, centre.1, o.location.latitude, o.location.longitude))
        .fold(0.0, f64::max);
    let nodes = settings.nodes.max(5) | 1;
    let depth_nodes = 11;
    let mut half = settings.half_extent_m.unwrap_or((2.0 * spread).max(10_000.0));
    let mut depth_range = (0.0, settings.max_depth_m.max(0.0));
    let mut best = (Source { latitude: centre.0, longitude: centre.1, depth_m: 0.0 }, f64::INFINITY);
    let mut search_centre = centre;
    let mut map = None;

    for level in 0..=settings.refinements {
        let step = 2.0 * half / (nodes - 1) as f64;
        let mut chi2_map = vec![f64::INFINITY; nodes * nodes];
        let levels = depths(depth_range.0, depth_range.1, depth_nodes);
        for north in 0..nodes {
            for east in 0..nodes {
                let (lat, lon) = displace(search_centre, -half + east as f64 * step, -half + north as f64 * step);
                for &depth_m in &levels {
                    let source = Source { latitude: lat, longitude: lon, depth_m };
                    let (_, chi2) = evaluate(&observations, source);
                    let cell = &mut chi2_map[north * nodes + east];
                    *cell = cell.min(chi2);
                    if chi2 < best.1 {
                        best = (source, chi2);
                    }
                }
            }
        }
        if level == 0 {
            map = Some(MisfitMap { centre, half_extent_m: half, nodes, chi2: chi2_map });
        }
        // Next grid: three cells around the best node.
        search_centre = (best.0.latitude, best.0.longitude);
        half = 3.0 * step;
        let depth_step = if levels.len() > 1 { levels[1] - levels[0] } else { 0.0 };
        depth_range = (
            (best.0.depth_m - 3.0 * depth_step).max(0.0),
            (best.0.depth_m + 3.0 * depth_step).min(settings.max_depth_m.max(0.0)),
        );
    }

    let source = best.0;
    let (origin, chi2) = evaluate(&observations, source);
    let residuals: Vec<f64> = observations
        .iter()
        .map(|o| o.time - origin - travel_time(source, &o.location, o.speed_m_s))
        .collect();
    let rms_s = (residuals.iter().map(|r| r * r).sum::<f64>() / count).sqrt();
    Some(Fit {
        source,
        origin,
        residuals,
        rms_s,
        chi2,
        dof: observations.len().saturating_sub(4).max(1),
        map: map.expect("level 0 ran"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: f64 = 299_792_458.0;

    fn at(source: Source, east_km: f64, north_km: f64) -> Location {
        let (latitude, longitude) = displace((source.latitude, source.longitude), east_km * 1e3, north_km * 1e3);
        Location { latitude, longitude, elevation_m: 0.0 }
    }

    fn observe(source: Source, origin: f64, location: Location, speed_m_s: f64, sigma_s: f64) -> Observation {
        Observation { location, speed_m_s, time: origin + travel_time(source, &location, speed_m_s), sigma_s }
    }

    #[test]
    fn haversine_matches_a_degree_of_latitude() {
        let d = surface_distance(10.0, 20.0, 11.0, 20.0);
        assert!((d - EARTH_RADIUS_M * 1f64.to_radians()).abs() < 1e-6, "{d}");
        assert!(surface_distance(0.0, 179.5, 0.0, -179.5) < 112_000.0);
    }

    #[test]
    fn exact_arrivals_locate_the_source() {
        let source = Source { latitude: 35.0, longitude: -106.0, depth_m: 4_000.0 };
        let origin = 10.0;
        let observations: Vec<Observation> = [(12.0, 5.0), (-8.0, 14.0), (-15.0, -6.0), (5.0, -18.0), (22.0, -9.0), (-3.0, 30.0)]
            .iter()
            .flat_map(|&(e, n)| {
                let location = at(source, e, n);
                [observe(source, origin, location, 6000.0, 0.01), observe(source, origin, location, 3500.0, 0.02)]
            })
            .collect();
        let fit = locate(&observations, &LocateSettings::default()).unwrap();
        let error = surface_distance(fit.source.latitude, fit.source.longitude, source.latitude, source.longitude);
        assert!(error < 10.0, "{error} m");
        assert!((fit.source.depth_m - source.depth_m).abs() < 50.0, "{:?}", fit.source);
        assert!((fit.origin - origin).abs() < 1e-3, "{}", fit.origin);
        assert!(fit.rms_s < 1e-3);
    }

    #[test]
    fn mixed_phenomenologies_anchor_on_the_fastest() {
        let source = Source { latitude: -20.0, longitude: 140.0, depth_m: 0.0 };
        let origin = 3.0;
        let mut observations = Vec::new();
        for (e, n) in [(4.0, 3.0), (-6.0, -2.0), (1.0, -9.0)] {
            observations.push(observe(source, origin, at(source, e, n), C, 1e-6));
        }
        for (e, n) in [(1.5, 0.8), (-2.0, 1.5), (0.5, -3.0)] {
            observations.push(observe(source, origin, at(source, e, n), 340.0, 0.02));
        }
        // A 30 ms bias on one infrasound pick moves position, not the origin time.
        observations[4].time += 0.03;
        let settings = LocateSettings { max_depth_m: 0.0, ..LocateSettings::default() };
        let fit = locate(&observations, &settings).unwrap();
        assert!((fit.origin - origin).abs() < 1e-5, "{}", fit.origin - origin);
        let error = surface_distance(fit.source.latitude, fit.source.longitude, source.latitude, source.longitude);
        assert!(error < 100.0, "{error} m");
        assert!(fit.residuals[4] > 0.005, "{:?}", fit.residuals);
    }

    #[test]
    fn the_misfit_map_is_lowest_near_the_source() {
        let source = Source { latitude: 0.0, longitude: 0.0, depth_m: 0.0 };
        let observations: Vec<Observation> =
            [(10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0)].iter().map(|&(e, n)| observe(source, 0.0, at(source, e, n), 1000.0, 0.01)).collect();
        let settings = LocateSettings { max_depth_m: 0.0, centre: Some((0.0, 0.0)), half_extent_m: Some(20_000.0), ..LocateSettings::default() };
        let fit = locate(&observations, &settings).unwrap();
        let map = &fit.map;
        let (i, _) = map.chi2.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        let (lat, lon) = map.node(i % map.nodes, i / map.nodes);
        assert!(surface_distance(lat, lon, 0.0, 0.0) < 1_000.0);
    }

    #[test]
    fn one_observation_is_not_enough() {
        let o = Observation { location: Location { latitude: 0.0, longitude: 0.0, elevation_m: 0.0 }, speed_m_s: 1.0, time: 0.0, sigma_s: 1.0 };
        assert!(locate(&[o], &LocateSettings::default()).is_none());
    }
}
