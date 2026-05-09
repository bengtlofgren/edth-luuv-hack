// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Sampling helpers — Box–Muller standard normal, generic over the float
// type. Only compiled with `feature = "sample"`. Uses `rand_core` for
// `RngCore` (no_std-compatible).

use num_traits::{Float, FloatConst, NumCast};
use rand_core::RngCore;

/// Draw u ~ Uniform[0, 1) at full f64 mantissa precision (53 bits),
/// then cast to the target float type. The conversion to a smaller
/// type (e.g. f32) is the standard truncation.
#[inline]
pub fn uniform_01<F: Float>(rng: &mut impl RngCore) -> F {
    // Top 53 bits of a u64 give a uniform integer in [0, 2^53);
    // dividing by 2^53 yields uniform [0, 1) at full f64 precision.
    let bits = rng.next_u64() >> 11;
    let denom = (1u64 << 53) as f64;
    let u = (bits as f64) / denom;
    F::from(u).unwrap_or_else(F::zero)
}

/// Draw z ~ N(0, 1) via Box–Muller transform.
///
/// Source: Box, G. E. P.; Muller, M. E. "A Note on the Generation of
/// Random Normal Deviates," *Annals of Mathematical Statistics*,
/// vol. 29, no. 2, pp. 610–611, 1958.
///
/// Returns one of the two independent standard-normal variates produced
/// by a single Box–Muller pair (u₁, u₂):
///
///   z = √(−2 · ln u₁) · cos(2π · u₂)
///
/// We discard the sin-companion variate to keep the API stateless.
/// Callers needing extreme RNG efficiency can substitute a Ziggurat
/// implementation later — for an IMU drift Monte Carlo, Box–Muller is
/// fine.
#[inline]
pub fn standard_normal<F, R>(rng: &mut R) -> F
where
    F: Float + FloatConst,
    R: RngCore,
{
    let two = F::one() + F::one();
    // u₁ ∈ (0, 1) — must avoid exactly 0 because of ln(0).
    let mut u1: F = uniform_01::<F>(rng);
    while u1 <= F::zero() {
        u1 = uniform_01::<F>(rng);
    }
    let u2: F = uniform_01::<F>(rng);
    let r = (-two * u1.ln()).sqrt();
    let theta = two * F::PI() * u2;
    r * theta.cos()
}

// Suppress the unused-import warning when no test or downstream code in
// this crate uses NumCast directly.
#[allow(dead_code)]
fn _force_use_numcast<F: NumCast>(_: F) {}
