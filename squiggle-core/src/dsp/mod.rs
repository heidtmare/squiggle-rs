//! Signal processing on sample buffers: filters, resampling, delay
//! estimation, onset picking and spectrograms. Everything that can keep state does, so
//! chunked and whole-buffer runs agree.

pub mod align;
pub mod correlate;
pub mod filter;
pub mod pick;
pub mod resample;
pub mod spectrogram;

pub use align::{AlignSettings, Channel, Measured, Solution, align, candidates, measure};
pub use correlate::{LagEstimate, LagMethod, PairLag, correlation, estimate_lag, solve_delays};
pub use filter::{Band, Detrend, FilterSpec, Processing, SosFilter, butterworth};
pub use pick::{Onset, PickSettings, StaLta, pick};
pub use resample::{Resampler, resample, resample_at};
pub use spectrogram::{Spectrogram, spectrogram};
