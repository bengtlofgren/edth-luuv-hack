// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Concrete drift processes. One struct per equation from the spec
// (Part C of the design document). Each implements [`DriftProcess`].
//
// All times in seconds. Units listed per struct.

use crate::process::DriftProcess;
use num_traits::{Float, FloatConst};

#[cfg(feature = "sample")]
use rand_core::RngCore;

// -----------------------------------------------------------------------------
// Orientation drift
// -----------------------------------------------------------------------------

/// Orientation error from white-noise on the gyro signal — Brownian motion.
///
/// Parameter `arw` is the **angle random walk** N in [rad/√s].
///
/// - μ(t) = 0
/// - σ²(t) = N² · t          (eq. C1)
///
/// Source: Woodman 2007 §6.1.1, eq. (6.1) [W07].
#[derive(Copy, Clone, Debug)]
pub struct AngleRandomWalk<F> {
    pub arw: F,
}

impl<F: Float> DriftProcess<F> for AngleRandomWalk<F> {
    #[inline]
    fn mean(&self, _t: F) -> F {
        F::zero()
    }
    #[inline]
    fn variance(&self, t: F) -> F {
        let n = self.arw;
        n * n * t
    }

    /// Coherent Brownian path: out[k] = out[k-1] + √(N²·Δt) · z_k,
    /// with out[0] = √(N²·t₀) · z_0 (anchored to X(0)=0).
    ///
    /// Source: Brownian-motion increment property; e.g. Øksendal,
    /// *Stochastic Differential Equations*, 6th ed., §2.2.
    #[cfg(feature = "sample")]
    fn sample_path<R: RngCore>(&self, ts: &[F], out: &mut [F], rng: &mut R)
    where
        F: FloatConst,
    {
        brownian_increment_path(self.arw, ts, out, rng);
    }
}

/// Orientation error from a constant gyro bias — deterministic linear ramp.
///
/// Parameter `b_g` in [rad/s].
///
/// - μ(t) = b_g · t          (eq. C2)
/// - σ²(t) = 0
///
/// Source: Woodman 2007 §6.1.2 [W07]; Titterton & Weston §12.2.1 [TW04].
#[derive(Copy, Clone, Debug)]
pub struct ConstantGyroBias<F> {
    pub b_g: F,
}

impl<F: Float> DriftProcess<F> for ConstantGyroBias<F> {
    #[inline]
    fn mean(&self, t: F) -> F {
        self.b_g * t
    }
    #[inline]
    fn variance(&self, _t: F) -> F {
        F::zero()
    }
}

/// Orientation error from gyro bias-instability (flicker noise).
///
/// Parameter `b` is bias instability B in [rad/s] (the floor of the
/// Allan-deviation curve on the gyro rate signal).
///
/// For t much larger than the correlation time of the flicker process,
/// the integrated angle error variance approximates:
///
/// - μ(t) = 0
/// - σ²(t) = (2 · ln 2 / π) · B² · t²    (eq. C7)
///
/// Source: El-Sheimy, Hou, Niu 2008, §III.B and Table I [ESN08].
#[derive(Copy, Clone, Debug)]
pub struct BiasInstabilityFlicker<F> {
    pub b: F,
}

impl<F: Float + FloatConst> DriftProcess<F> for BiasInstabilityFlicker<F> {
    #[inline]
    fn mean(&self, _t: F) -> F {
        F::zero()
    }
    #[inline]
    fn variance(&self, t: F) -> F {
        // (2 · ln 2 / π) · B² · t²
        let two = F::one() + F::one();
        let coef = two * F::LN_2() / F::PI();
        let b = self.b;
        coef * b * b * t * t
    }
}

/// First-order Gauss–Markov (Ornstein–Uhlenbeck) bias process.
///
/// Parameters:
/// - `sigma`: steady-state std-dev σ_b
/// - `tau_c`: correlation time τ_c
///
/// - μ(t) = 0
/// - σ²(t) = σ_b² · (1 − exp(−2t/τ_c))   (eq. B3 marginal variance)
///
/// Source: Farrell 2008, §4.6.2, eq. (4.118) [F08]; Groves 2013,
/// §14.2.6, eq. (14.81) [G13].
#[derive(Copy, Clone, Debug)]
pub struct GaussMarkov<F> {
    pub sigma: F,
    pub tau_c: F,
}

impl<F: Float> DriftProcess<F> for GaussMarkov<F> {
    #[inline]
    fn mean(&self, _t: F) -> F {
        F::zero()
    }
    #[inline]
    fn variance(&self, t: F) -> F {
        let two = F::one() + F::one();
        let s = self.sigma;
        s * s * (F::one() - (-two * t / self.tau_c).exp())
    }
}

// -----------------------------------------------------------------------------
// Velocity drift
// -----------------------------------------------------------------------------

/// Velocity error from white-noise on the accelerometer signal — Brownian motion.
///
/// Parameter `vrw` is the **velocity random walk** N_a in [m/s/√s].
///
/// - μ(t) = 0
/// - σ²(t) = N_a² · t        (eq. C3)
///
/// Source: Woodman 2007 §6.2.1 [W07].
#[derive(Copy, Clone, Debug)]
pub struct VelocityRandomWalk<F> {
    pub vrw: F,
}

impl<F: Float> DriftProcess<F> for VelocityRandomWalk<F> {
    #[inline]
    fn mean(&self, _t: F) -> F {
        F::zero()
    }
    #[inline]
    fn variance(&self, t: F) -> F {
        let n = self.vrw;
        n * n * t
    }

    /// Coherent Brownian path of velocity error.
    #[cfg(feature = "sample")]
    fn sample_path<R: RngCore>(&self, ts: &[F], out: &mut [F], rng: &mut R)
    where
        F: FloatConst,
    {
        brownian_increment_path(self.vrw, ts, out, rng);
    }
}

/// Velocity error from a constant accelerometer bias.
///
/// Parameter `b_a` in [m/s²].
///
/// - μ(t) = b_a · t
/// - σ²(t) = 0
///
/// Source: Woodman 2007 §6.2.2 [W07].
#[derive(Copy, Clone, Debug)]
pub struct ConstantAccelBiasOnVelocity<F> {
    pub b_a: F,
}

impl<F: Float> DriftProcess<F> for ConstantAccelBiasOnVelocity<F> {
    #[inline]
    fn mean(&self, t: F) -> F {
        self.b_a * t
    }
    #[inline]
    fn variance(&self, _t: F) -> F {
        F::zero()
    }
}

// -----------------------------------------------------------------------------
// Position drift
// -----------------------------------------------------------------------------

/// Position error from accelerometer white noise — integrated Brownian motion.
///
/// Parameter `vrw` is the **velocity random walk** N_a in [m/s/√s] (the
/// same parameter that governs [`VelocityRandomWalk`]; here it is
/// integrated once more to give position).
///
/// - μ(t) = 0
/// - σ²(t) = N_a² · t³ / 3        (eq. C4)
///
/// Source: Woodman 2007, §6.2.1, eq. (6.4) [W07].
#[derive(Copy, Clone, Debug)]
pub struct IntegratedVelocityRandomWalk<F> {
    pub vrw: F,
}

impl<F: Float> DriftProcess<F> for IntegratedVelocityRandomWalk<F> {
    #[inline]
    fn mean(&self, _t: F) -> F {
        F::zero()
    }
    #[inline]
    fn variance(&self, t: F) -> F {
        let three = F::one() + F::one() + F::one();
        let n = self.vrw;
        n * n * t * t * t / three
    }
}

/// Position error from a constant accelerometer bias.
///
/// Parameter `b_a` in [m/s²].
///
/// - μ(t) = ½ · b_a · t²            (eq. C5)
/// - σ²(t) = 0
///
/// Source: Woodman 2007 §6.2.2 [W07]; Titterton & Weston §12.4.1,
/// eq. (12.27) [TW04].
#[derive(Copy, Clone, Debug)]
pub struct ConstantAccelBiasOnPosition<F> {
    pub b_a: F,
}

impl<F: Float> DriftProcess<F> for ConstantAccelBiasOnPosition<F> {
    #[inline]
    fn mean(&self, t: F) -> F {
        let half = F::one() / (F::one() + F::one());
        half * self.b_a * t * t
    }
    #[inline]
    fn variance(&self, _t: F) -> F {
        F::zero()
    }
}

/// Position error from coupled gyro-bias × gravity (the cubic killer).
///
/// A constant gyro bias produces a tilt error θ(t) = b_g · t which
/// mis-projects gravity onto the horizontal axes. To leading order in
/// small bias:
///
/// - μ(t) ≈ ½ · g · b_g · t³        (eq. C6)
/// - σ²(t) = 0
///
/// Parameters:
/// - `g`:   local gravity magnitude in [m/s²] (typically 9.81)
/// - `b_g`: constant gyro bias in [rad/s]
///
/// Source: Woodman 2007 §6.1.2, eq. (6.2) [W07]; same cubic horizontal
/// growth in Titterton & Weston §12.4.2, eq. (12.41) [TW04] and
/// Groves §5.7.2, eq. (5.118) [G13].
#[derive(Copy, Clone, Debug)]
pub struct GyroBiasGravityCoupling<F> {
    pub g: F,
    pub b_g: F,
}

impl<F: Float> DriftProcess<F> for GyroBiasGravityCoupling<F> {
    #[inline]
    fn mean(&self, t: F) -> F {
        let half = F::one() / (F::one() + F::one());
        half * self.g * self.b_g * t * t * t
    }
    #[inline]
    fn variance(&self, _t: F) -> F {
        F::zero()
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Generate a coherent Brownian-motion path X(t) with diffusion coefficient
/// `sigma` (so X(t) ~ N(0, σ²·t)) on the time grid `ts`, into `out`.
///
/// Anchored at X(0) = 0 (i.e. the marginal at ts[0] is N(0, σ²·ts[0])).
/// Increments are independent Gaussians ΔW_k ~ N(0, σ²·(ts[k]-ts[k-1])).
#[cfg(feature = "sample")]
fn brownian_increment_path<F, R>(sigma: F, ts: &[F], out: &mut [F], rng: &mut R)
where
    F: Float + FloatConst,
    R: RngCore,
{
    debug_assert_eq!(ts.len(), out.len());
    if ts.is_empty() {
        return;
    }
    let zero = F::zero();
    // X(t_0) = √(σ²·t_0) · z_0 — assumes X(0) = 0 anchor.
    let z0 = crate::sampling::standard_normal::<F, R>(rng);
    out[0] = sigma * ts[0].max(zero).sqrt() * z0;
    for k in 1..ts.len() {
        let dt = ts[k] - ts[k - 1];
        debug_assert!(dt >= zero, "ts must be non-decreasing");
        let z = crate::sampling::standard_normal::<F, R>(rng);
        let increment = sigma * dt.max(zero).sqrt() * z;
        out[k] = out[k - 1] + increment;
    }
}
