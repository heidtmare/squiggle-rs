//! The discrete Fourier transform, computed directly and fast.
//!
//! Both use the unnormalised convention `X[k] = Σ x[n]·e^(-2πikn/N)`, with the
//! `1/N` applied on the way back.

use core::f64::consts::{PI, TAU};

use super::complex::Complex;

/// The first `bins` coefficients of the DFT of `input`, summed term by term.
///
/// This is the O(N·bins) definition, kept because "the DFT" and "the FFT" are
/// different algorithms for the same result and the plot offers both.
pub fn dft(input: &[Complex], bins: usize) -> Vec<Complex> {
    let n = input.len();
    // Every phase the sum needs is one of N roots of unity; indexing a table by
    // `(j·k) mod N` keeps the error from growing with the index.
    let roots: Vec<Complex> = (0..n).map(|j| Complex::cis(-TAU * j as f64 / n as f64)).collect();

    (0..bins.min(n))
        .map(|k| {
            let mut index = 0;
            let mut sum = Complex::ZERO;
            for sample in input {
                sum += *sample * roots[index];
                index = (index + k) % n;
            }
            sum
        })
        .collect()
}

/// Forward FFT in place, for any length.
pub fn fft(data: &mut [Complex]) {
    transform(data, -1.0);
}

/// Inverse FFT in place, including the `1/N`.
pub fn ifft(data: &mut [Complex]) {
    transform(data, 1.0);
    let n = data.len() as f64;
    for value in data {
        *value = value.scale(1.0 / n);
    }
}

/// Unnormalised transform; `sign` is -1 forward and +1 inverse.
fn transform(data: &mut [Complex], sign: f64) {
    match data.len() {
        0 | 1 => {}
        n if n.is_power_of_two() => radix2(data, sign),
        _ => bluestein(data, sign),
    }
}

/// Iterative Cooley–Tukey for power-of-two lengths.
fn radix2(data: &mut [Complex], sign: f64) {
    let n = data.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            data.swap(i, j);
        }
    }

    // One table serves every stage: stage `len` uses every `n/len`-th root.
    let roots: Vec<Complex> =
        (0..n / 2).map(|j| Complex::cis(sign * TAU * j as f64 / n as f64)).collect();

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len;
        for start in (0..n).step_by(len) {
            for j in 0..half {
                let a = data[start + j];
                let b = data[start + j + half] * roots[j * stride];
                data[start + j] = a + b;
                data[start + j + half] = a - b;
            }
        }
        len <<= 1;
    }
}

/// Bluestein's chirp-z: rewrites a length-N transform as a convolution, which
/// a power-of-two FFT can do. This is what keeps the FFT fast for sample
/// counts like 1000 that are not powers of two.
fn bluestein(data: &mut [Complex], sign: f64) {
    let n = data.len();
    let m = (2 * n - 1).next_power_of_two();

    // kn = (k² + n² - (k-n)²) / 2. Reducing k² mod 2N keeps the angle small.
    let chirp: Vec<Complex> = (0..n)
        .map(|k| {
            let square = (k * k) % (2 * n);
            Complex::cis(sign * PI * square as f64 / n as f64)
        })
        .collect();

    let mut a = vec![Complex::ZERO; m];
    for k in 0..n {
        a[k] = data[k] * chirp[k];
    }

    let mut b = vec![Complex::ZERO; m];
    b[0] = chirp[0].conj();
    for k in 1..n {
        b[k] = chirp[k].conj();
        b[m - k] = chirp[k].conj();
    }

    radix2(&mut a, -1.0);
    radix2(&mut b, -1.0);
    for (x, y) in a.iter_mut().zip(&b) {
        *x *= *y;
    }
    radix2(&mut a, 1.0);

    let scale = 1.0 / m as f64;
    for k in 0..n {
        data[k] = (a[k] * chirp[k]).scale(scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<Complex> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                Complex::from((t * 0.3).sin() + 0.5 * (t * 1.7).cos() + 0.01 * t)
            })
            .collect()
    }

    fn assert_close(a: &[Complex], b: &[Complex]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((*x - *y).norm() < 1e-8, "{x:?} vs {y:?}");
        }
    }

    #[test]
    fn the_fft_agrees_with_the_dft_at_power_of_two_lengths() {
        let input = signal(64);
        let mut fast = input.clone();
        fft(&mut fast);
        assert_close(&fast, &dft(&input, 64));
    }

    #[test]
    fn the_fft_agrees_with_the_dft_at_awkward_lengths() {
        for n in [3, 7, 100, 1000] {
            let input = signal(n);
            let mut fast = input.clone();
            fft(&mut fast);
            assert_close(&fast, &dft(&input, n));
        }
    }

    #[test]
    fn the_inverse_undoes_the_forward() {
        for n in [16, 999] {
            let input = signal(n);
            let mut round_trip = input.clone();
            fft(&mut round_trip);
            ifft(&mut round_trip);
            assert_close(&round_trip, &input);
        }
    }

    #[test]
    fn a_pure_tone_lands_in_one_bin() {
        let n = 32;
        let input: Vec<Complex> =
            (0..n).map(|i| Complex::from((TAU * 4.0 * i as f64 / n as f64).cos())).collect();
        let spectrum = dft(&input, n / 2 + 1);
        for (k, bin) in spectrum.iter().enumerate() {
            let expected = if k == 4 { n as f64 / 2.0 } else { 0.0 };
            assert!((bin.norm() - expected).abs() < 1e-9, "bin {k}: {}", bin.norm());
        }
    }
}
