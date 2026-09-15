//! Core model and coordinate math for Squiggle.
//!
//! This crate is a port of the `squiggle-core` Java module. Unlike the original
//! it has no rendering dependencies: it owns the plot *model* (squiggles,
//! points, styles, axes) and the *math* that maps between cartesian data space
//! and plot-screen pixels. Rendering lives in `squiggle-app`.

pub mod axis;
pub mod color;
pub mod squiggle;
pub mod view;

pub use axis::{Anchor, Axis, AxisKind, Orientation, Tick};
pub use color::Rgba;
pub use squiggle::{Bounds, Point, Squiggle, Style};
pub use view::{Coord, Insets, PlotView};
