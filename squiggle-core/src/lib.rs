//! Core model and coordinate math for Squiggle.
//!
//! This crate is a port of the `squiggle-core` Java module. Unlike the original
//! it has no rendering dependencies: it owns the plot *model* (squiggles,
//! points, styles, axes) and the *math* that maps between cartesian data space
//! and plot-screen pixels, plus the signal transforms (Fourier, wavelet) that
//! derive new squiggles from old. Rendering lives in `squiggle-app`.
//!
//! The project model ([`project`], [`sensor`], [`phenomenology`]) is plain
//! data; the optional `serde` feature derives its serialisation.

pub mod axis;
pub mod color;
pub mod id;
pub mod phenomenology;
pub mod project;
pub mod sensor;
pub mod squiggle;
pub mod time;
pub mod transform;
pub mod view;

pub use axis::{Anchor, Axis, AxisKind, Orientation, Tick};
pub use color::Rgba;
pub use id::Id;
pub use phenomenology::{Phenomenology, PhenomenologyId, Propagation, SampleKind, Suggestion};
pub use project::{Change, Op, Project};
pub use sensor::{Analyst, ChunkedSamples, Sensor, Trace, TraceSource};
pub use squiggle::{Bounds, Point, Squiggle, Style};
pub use transform::{Component, Transform, TransformError, TransformOptions, Wavelet, WaveletOutput, Window};
pub use time::Timestamp;
pub use view::{Coord, Insets, PlotView};
