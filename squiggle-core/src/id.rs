//! UUIDv7 identifiers: time-ordered, globally unique, mergeable across stores.

use core::fmt;
use core::hash::{BuildHasher, Hasher};
use core::str::FromStr;
use std::collections::hash_map::RandomState;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::time::Timestamp;

/// A UUIDv7 (RFC 9562). Serialised as the canonical hyphenated string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "String", try_from = "String"))]
pub struct Id(u128);

impl Id {
    /// Builds a v7 ID from a Unix-millisecond timestamp and 74 random bits.
    pub fn v7(unix_millis: u64, random: u128) -> Self {
        let mut value = u128::from(unix_millis & 0xFFFF_FFFF_FFFF) << 80;
        value |= 0x7 << 76; // version
        value |= ((random >> 62) & 0xFFF) << 64; // rand_a
        value |= 0b10 << 62; // variant
        value |= random & ((1 << 62) - 1); // rand_b
        Self(value)
    }

    /// A fresh ID. Entropy comes from std's randomly keyed SipHash, seeded per process,
    /// mixed with a counter, so the crate needs no RNG dependency.
    pub fn generate() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let state = RandomState::new();
        let mut random = 0u128;
        for half in 0..2 {
            let mut hasher = state.build_hasher();
            hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
            hasher.write_u8(half);
            random = (random << 64) | u128::from(hasher.finish());
        }
        Self::v7(Timestamp::now().unix_millis().max(0) as u64, random)
    }

    pub const fn as_u128(self) -> u128 {
        self.0
    }

    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = format!("{:032x}", self.0);
        write!(f, "{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseIdError(pub String);

impl fmt::Display for ParseIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid id: {}", self.0)
    }
}

impl std::error::Error for ParseIdError {}

impl FromStr for Id {
    type Err = ParseIdError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let hex: String = text.chars().filter(|c| *c != '-').collect();
        if hex.len() != 32 {
            return Err(ParseIdError(text.to_string()));
        }
        u128::from_str_radix(&hex, 16).map(Self).map_err(|_| ParseIdError(text.to_string()))
    }
}

impl From<Id> for String {
    fn from(id: Id) -> Self {
        id.to_string()
    }
}

impl TryFrom<String> for Id {
    type Error = ParseIdError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_and_variant_bits_are_set() {
        let id = Id::generate().as_u128();
        assert_eq!((id >> 76) & 0xF, 7);
        assert_eq!((id >> 62) & 0b11, 0b10);
    }

    #[test]
    fn ids_are_unique_and_ordered_by_time() {
        let a = Id::v7(1_000, u128::MAX);
        let b = Id::v7(1_001, 0);
        assert!(a < b);
        assert_ne!(Id::generate(), Id::generate());
    }

    #[test]
    fn string_round_trip() {
        let id = Id::generate();
        let text = id.to_string();
        assert_eq!(text.len(), 36);
        assert_eq!(text.parse::<Id>(), Ok(id));
    }
}
