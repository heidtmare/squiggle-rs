//! Absolute time with sub-nanosecond resolution.
//!
//! `f64` epoch seconds resolve only ~240 ns in 2026, too coarse for RF/EMP.
//! [`Timestamp`] keeps whole seconds as `i64` and the fraction as `f64`
//! (resolution ~1e-16 s). Plots work in `f64` seconds relative to a reference.

use core::fmt;
use core::str::FromStr;

/// UTC instant: `secs` since the Unix epoch plus `frac` in `[0, 1)`. Leap seconds are ignored.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Timestamp {
    secs: i64,
    frac: f64,
}

impl Timestamp {
    pub const UNIX_EPOCH: Self = Self { secs: 0, frac: 0.0 };

    /// Normalises any `frac` into `[0, 1)`.
    pub fn new(secs: i64, frac: f64) -> Self {
        let carry = frac.floor();
        let mut frac = frac - carry;
        let mut secs = secs + carry as i64;
        // `frac - floor(frac)` can round up to exactly 1.0.
        if frac >= 1.0 {
            frac -= 1.0;
            secs += 1;
        }
        Self { secs, frac }
    }

    pub fn from_unix_seconds(seconds: f64) -> Self {
        Self::new(0, seconds)
    }

    pub fn now() -> Self {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self::new(elapsed.as_secs() as i64, elapsed.subsec_nanos() as f64 * 1e-9)
    }

    pub const fn secs(self) -> i64 {
        self.secs
    }

    pub const fn frac(self) -> f64 {
        self.frac
    }

    /// `self + seconds`, exact in the whole part.
    pub fn add_seconds(self, seconds: f64) -> Self {
        let whole = seconds.trunc();
        Self::new(self.secs + whole as i64, self.frac + (seconds - whole))
    }

    /// `self - earlier` in seconds. Exact while the difference fits an `f64` mantissa.
    pub fn seconds_since(self, earlier: Self) -> f64 {
        (self.secs - earlier.secs) as f64 + (self.frac - earlier.frac)
    }

    /// Unix milliseconds, for IDs.
    pub fn unix_millis(self) -> i64 {
        self.secs * 1000 + (self.frac * 1000.0) as i64
    }

    pub fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }
}

/// Days since 1970-01-01 to (year, month, day). Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// (year, month, day) to days since 1970-01-01. Hinnant's `days_from_civil`.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year.rem_euclid(400);
    let month = i64::from(month);
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// ISO 8601 UTC with nanoseconds: `2026-10-11T08:30:00.123456789Z`.
impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut secs = self.secs;
        let mut nanos = (self.frac * 1e9).round() as u64;
        if nanos >= 1_000_000_000 {
            nanos -= 1_000_000_000;
            secs += 1;
        }
        let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
        let of_day = secs.rem_euclid(86_400);
        write!(
            f,
            "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{nanos:09}Z",
            of_day / 3600,
            of_day / 60 % 60,
            of_day % 60
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseTimestampError(pub String);

impl fmt::Display for ParseTimestampError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid timestamp: {}", self.0)
    }
}

impl std::error::Error for ParseTimestampError {}

/// Parses `YYYY-MM-DD[T| ]HH:MM:SS[.fraction][Z]`, UTC only, or bare Unix seconds.
impl FromStr for Timestamp {
    type Err = ParseTimestampError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let error = || ParseTimestampError(text.to_string());
        let trimmed = text.trim();
        if let Ok(seconds) = trimmed.parse::<f64>() {
            return seconds.is_finite().then(|| Self::from_unix_seconds(seconds)).ok_or_else(error);
        }

        let body = trimmed.strip_suffix('Z').unwrap_or(trimmed);
        let (date, time) = body.split_once(['T', ' ']).ok_or_else(error)?;

        let mut date_parts = date.splitn(3, '-');
        let mut next_date = || date_parts.next().ok_or_else(error);
        let year: i64 = next_date()?.parse().map_err(|_| error())?;
        let month: u32 = next_date()?.parse().map_err(|_| error())?;
        let day: u32 = next_date()?.parse().map_err(|_| error())?;

        let mut time_parts = time.splitn(3, ':');
        let mut next_time = || time_parts.next().ok_or_else(error);
        let hour: i64 = next_time()?.parse().map_err(|_| error())?;
        let minute: i64 = next_time()?.parse().map_err(|_| error())?;
        let seconds_text = next_time()?;
        let (whole, fraction) = seconds_text.split_once('.').unwrap_or((seconds_text, ""));
        let second: i64 = whole.parse().map_err(|_| error())?;
        let frac = if fraction.is_empty() {
            0.0
        } else if fraction.bytes().all(|b| b.is_ascii_digit()) {
            format!("0.{fraction}").parse::<f64>().map_err(|_| error())?
        } else {
            return Err(error());
        };

        if !(1..=12).contains(&month)
            || !(1..=31).contains(&day)
            || !(0..24).contains(&hour)
            || !(0..60).contains(&minute)
            || !(0..61).contains(&second)
        {
            return Err(error());
        }

        let days = days_from_civil(year, month, day);
        Ok(Self::new(days * 86_400 + hour * 3600 + minute * 60 + second, frac))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_is_normalised() {
        let t = Timestamp::new(10, 2.25);
        assert_eq!((t.secs(), t.frac()), (12, 0.25));
        let t = Timestamp::new(10, -0.25);
        assert_eq!((t.secs(), t.frac()), (9, 0.75));
    }

    #[test]
    fn sub_nanosecond_differences_survive_at_modern_epochs() {
        let t0: Timestamp = "2026-10-11T00:00:00Z".parse().unwrap();
        let t1 = t0.add_seconds(1e-10);
        let dt = t1.seconds_since(t0);
        assert!((dt - 1e-10).abs() < 1e-18, "{dt}");
    }

    #[test]
    fn display_and_parse_round_trip() {
        let text = "2026-10-11T08:30:05.123456789Z";
        let t: Timestamp = text.parse().unwrap();
        assert_eq!(t.to_string(), text);
    }

    #[test]
    fn dates_before_the_epoch_format_correctly() {
        let t: Timestamp = "1969-12-31T23:59:59Z".parse().unwrap();
        assert_eq!(t.secs(), -1);
        assert_eq!(t.to_string(), "1969-12-31T23:59:59.000000000Z");
    }

    #[test]
    fn leap_day_and_space_separator_parse() {
        let t: Timestamp = "2024-02-29 12:00:00".parse().unwrap();
        assert_eq!(t.to_string(), "2024-02-29T12:00:00.000000000Z");
    }

    #[test]
    fn bare_unix_seconds_parse() {
        let t: Timestamp = "86400.5".parse().unwrap();
        assert_eq!(t.to_string(), "1970-01-02T00:00:00.500000000Z");
    }

    #[test]
    fn malformed_text_is_rejected() {
        for text in ["", "2026-13-01T00:00:00Z", "2026-10-11", "2026-10-11T00:00:00.1x"] {
            assert!(text.parse::<Timestamp>().is_err(), "{text}");
        }
    }
}
