//! Just enough complex arithmetic for the transforms, so the crate keeps its
//! promise of having no dependencies.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);

    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `e^(iθ)`: the unit phasor at angle `theta`.
    pub fn cis(theta: f64) -> Self {
        let (sin, cos) = theta.sin_cos();
        Self::new(cos, sin)
    }

    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Magnitude.
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Phase in radians, in `-π..=π`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }

    /// `|z|²`.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Principal square root.
    pub fn sqrt(self) -> Self {
        let r = self.norm();
        let re = ((r + self.re) / 2.0).max(0.0).sqrt();
        let im = ((r - self.re) / 2.0).max(0.0).sqrt();
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
}

impl From<f64> for Complex {
    fn from(re: f64) -> Self {
        Self::new(re, 0.0)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.re * rhs.re - self.im * rhs.im, self.re * rhs.im + self.im * rhs.re)
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sqr();
        Self::new((self.re * rhs.re + self.im * rhs.im) / d, (self.im * rhs.re - self.re * rhs.im) / d)
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
