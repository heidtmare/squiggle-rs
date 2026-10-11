//! Phenomenologies: the physics a sensor measures. Data, not code — projects
//! carry their own registry, seeded with built-ins and extended by analysts.

use core::fmt;

use crate::color::Rgba;

/// Stable slug, e.g. `seismic`. Unique within a project.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct PhenomenologyId(pub String);

impl PhenomenologyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PhenomenologyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SampleKind {
    #[default]
    Real,
    /// I/Q pairs.
    Complex,
}

impl SampleKind {
    pub const ALL: [Self; 2] = [Self::Real, Self::Complex];

    pub fn label(self) -> &'static str {
        match self {
            Self::Real => "Real",
            Self::Complex => "Complex (IQ)",
        }
    }
}

/// A named arrival travelling at a fixed speed, e.g. seismic P.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Phase {
    pub name: String,
    pub speed_m_s: f64,
}

/// How a signal travels from source to sensor; drives arrival prediction.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Propagation {
    Constant { speed_m_s: f64 },
    Phases(Vec<Phase>),
    /// Alignment by picks or manual offsets only.
    Unknown,
}

impl Propagation {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "Constant speed",
            Self::Phases(_) => "Phases",
            Self::Unknown => "Unknown",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Phenomenology {
    pub id: PhenomenologyId,
    pub name: String,
    /// Physical quantity, e.g. "ground velocity".
    pub quantity: String,
    pub units: String,
    pub sample_kind: SampleKind,
    pub propagation: Propagation,
    pub color: Rgba,
    /// Bumped on every change; sensor assignments record the version they used.
    pub version: u32,
}

const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Definitions shipped with the app, at version 1.
pub fn builtins() -> Vec<Phenomenology> {
    let make = |id: &str, name: &str, quantity: &str, units: &str, kind, propagation, color| Phenomenology {
        id: PhenomenologyId::new(id),
        name: name.to_string(),
        quantity: quantity.to_string(),
        units: units.to_string(),
        sample_kind: kind,
        propagation,
        color,
        version: 1,
    };
    let phase = |name: &str, speed_m_s| Phase { name: name.to_string(), speed_m_s };

    vec![
        make(
            "seismic",
            "Seismic",
            "ground velocity",
            "m/s",
            SampleKind::Real,
            // Average crustal speeds; refined per region by analysts.
            Propagation::Phases(vec![phase("P", 6000.0), phase("S", 3500.0), phase("Surface", 3000.0)]),
            Rgba::new(0.95, 0.65, 0.25),
        ),
        make(
            "acoustic",
            "Acoustic",
            "pressure",
            "Pa",
            SampleKind::Real,
            Propagation::Constant { speed_m_s: 343.0 },
            Rgba::new(0.45, 0.85, 0.45),
        ),
        make(
            "infrasound",
            "Infrasound",
            "pressure",
            "Pa",
            SampleKind::Real,
            Propagation::Constant { speed_m_s: 340.0 },
            Rgba::new(0.35, 0.75, 0.95),
        ),
        make(
            "rf",
            "RF",
            "IQ voltage",
            "V",
            SampleKind::Complex,
            Propagation::Constant { speed_m_s: SPEED_OF_LIGHT },
            Rgba::new(0.85, 0.5, 0.95),
        ),
        make(
            "emp",
            "EMP",
            "electric field",
            "V/m",
            SampleKind::Real,
            Propagation::Constant { speed_m_s: SPEED_OF_LIGHT },
            Rgba::new(1.0, 0.4, 0.4),
        ),
    ]
}

/// A reader's guess at a phenomenology. Shown to the analyst, never applied unconfirmed.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    pub phenomenology: PhenomenologyId,
    pub reason: String,
}

/// Guesses from a SEED channel code (`BHZ`, or `NET.STA.LOC.CHA`).
/// Instrument code `H/L/G/N` is seismic; `D` (pressure) with orientation `F` is
/// infrasound, `H` is hydroacoustic.
pub fn suggest_from_seed_code(code: &str) -> Option<Suggestion> {
    let channel = code.rsplit('.').next()?.as_bytes();
    if channel.len() != 3 || !channel.iter().all(u8::is_ascii_alphanumeric) {
        return None;
    }
    let id = match (channel[1].to_ascii_uppercase(), channel[2].to_ascii_uppercase()) {
        (b'H' | b'L' | b'G' | b'N', _) => "seismic",
        (b'D', b'F') => "infrasound",
        (b'D', b'H') => "acoustic",
        _ => return None,
    };
    Some(Suggestion {
        phenomenology: PhenomenologyId::new(id),
        reason: format!("SEED channel code {}", String::from_utf8_lossy(channel)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_ids_are_unique() {
        let mut ids: Vec<_> = builtins().into_iter().map(|p| p.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), builtins().len());
    }

    #[test]
    fn seed_codes_suggest_phenomenologies() {
        let id = |code| suggest_from_seed_code(code).map(|s| s.phenomenology.0);
        assert_eq!(id("IU.ANMO.00.BHZ").as_deref(), Some("seismic"));
        assert_eq!(id("HNE").as_deref(), Some("seismic"));
        assert_eq!(id("IM.I53H1..BDF").as_deref(), Some("infrasound"));
        assert_eq!(id("EDH").as_deref(), Some("acoustic"));
    }

    #[test]
    fn non_seed_codes_suggest_nothing() {
        for code in ["mic1", "rx-2", "BXZ", "B Z", ""] {
            assert_eq!(suggest_from_seed_code(code), None, "{code}");
        }
    }
}
