//! Signal processing on sample buffers: filters, resampling, delay
//! estimation, detection, onset picking and spectrograms. Everything that can keep state does, so
//! chunked and whole-buffer runs agree.

pub mod align;
pub mod correlate;
pub mod detect;
pub mod filter;
pub mod matched;
pub mod pick;
pub mod resample;
pub mod spectrogram;

pub use align::{AlignSettings, Channel, Measured, Solution, align, candidates, measure};
pub use correlate::{LagEstimate, LagMethod, PairLag, correlation, estimate_lag, solve_delays};
pub use filter::{Band, Detrend, FilterSpec, Processing, SosFilter, butterworth};
pub use detect::{Detector, DetectorSpec, detect};
pub use matched::{MatchedFilter, MatchedSpec, matched};
pub use pick::{Onset, PickSettings, StaLta, Trigger, pick, refine};
pub use resample::{Resampler, resample, resample_at};
pub use spectrogram::{Spectrogram, spectrogram};
