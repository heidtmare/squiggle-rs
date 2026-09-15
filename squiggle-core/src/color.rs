//! A minimal RGBA color, so that the model layer stays free of rendering
//! dependencies. `squiggle-app` converts these to `bevy::Color`.

/// A color with sRGB components and straight alpha, each in `0.0..=1.0`.
///
/// Ported from the JavaFX `Paint` values held by `RenderingAttributes`, which
/// were likewise sRGB. Bevy renders in linear space, so `squiggle-app`
/// converts on the way to the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub alpha: f32,
}

impl Rgba {
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0);
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0);
    pub const CYAN: Self = Self::new(0.0, 1.0, 1.0);
    /// JavaFX `Color.GOLD`, used for the cursor crosshair.
    pub const GOLD: Self = Self::new(1.0, 0.843_137_3, 0.0);

    pub const fn new(red: f32, green: f32, blue: f32) -> Self {
        Self { red, green, blue, alpha: 1.0 }
    }

    pub const fn with_alpha(self, alpha: f32) -> Self {
        Self { alpha, ..self }
    }
}

impl Default for Rgba {
    fn default() -> Self {
        Self::WHITE
    }
}
