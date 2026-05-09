// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Core abstraction: a `DriftProcess<F>` is a stochastic process X(t) with
// closed-form mean μ(t) and variance σ²(t). Independent processes compose
// via [`Sum`] or the `+` operator.

use num_traits::Float;

#[cfg(feature = "sample")]
use num_traits::FloatConst;
#[cfg(feature = "sample")]
use rand_core::RngCore;

/// A stochastic drift process X(t) with closed-form mean and variance.
///
/// All times `t` are in seconds. Implementations report:
/// - μ(t) = `mean(t)`         — deterministic component (e.g. b·t for a constant bias)
/// - σ²(t) = `variance(t)`    — variance of the stochastic component
/// - σ(t) = `std(t)`          — convenience wrapper, defaults to `sqrt(variance(t))`
///
/// Independent processes compose via [`Sum`] / `+`. The composition law
/// (`Var[X+Y] = Var[X] + Var[Y]`, `E[X+Y] = E[X] + E[Y]` for independent
/// X, Y) holds termwise.
pub trait DriftProcess<F: Float> {
    /// μ(t) — deterministic mean of X(t).
    fn mean(&self, t: F) -> F;

    /// σ²(t) — variance of the stochastic component of X(t).
    fn variance(&self, t: F) -> F;

    /// σ(t) — standard deviation of the stochastic component.
    #[inline]
    fn std(&self, t: F) -> F {
        self.variance(t).sqrt()
    }

    /// Marginal sample of X(t) ~ N(μ(t), σ²(t)) at a single time `t`.
    ///
    /// Default implementation uses Box–Muller via [`crate::sampling::standard_normal`].
    #[cfg(feature = "sample")]
    fn sample<R: RngCore>(&self, t: F, rng: &mut R) -> F
    where
        F: FloatConst,
    {
        let z = crate::sampling::standard_normal::<F, R>(rng);
        self.mean(t) + self.std(t) * z
    }

    /// Path sample of X on the time grid `ts`, written into `out`
    /// (must satisfy `out.len() == ts.len()`).
    ///
    /// **Default implementation:** independent marginal samples at each
    /// grid point. Each `out[i]` is correctly distributed as
    /// N(μ(ts[i]), σ²(ts[i])), but the resulting sequence is **not** a
    /// coherent realization of a single trajectory for processes with
    /// correlated increments.
    ///
    /// Brownian-class processes ([`crate::AngleRandomWalk`],
    /// [`crate::VelocityRandomWalk`]) override this with a proper
    /// increment-based sampler that produces a coherent path.
    #[cfg(feature = "sample")]
    fn sample_path<R: RngCore>(&self, ts: &[F], out: &mut [F], rng: &mut R)
    where
        F: FloatConst,
    {
        debug_assert_eq!(ts.len(), out.len());
        for (i, &t) in ts.iter().enumerate() {
            out[i] = self.sample(t, rng);
        }
    }
}

/// Independent sum of two drift processes — the algebraic core of
/// composition.
///
/// `Sum<P, Q>` represents X(t) + Y(t) where X, Y are independent.
/// Mean adds linearly; variance adds linearly (independence). Sampling
/// a single time draws independent samples from each child and sums.
///
/// ## Path sampling
/// `sample_path` of a `Sum` falls back to marginal sampling per grid point
/// (the default-impl behavior). For a coherent joint trajectory of a sum,
/// allocate a scratch buffer and call `sample_path` on each child
/// separately, then add elementwise.
#[derive(Copy, Clone, Debug)]
pub struct Sum<P, Q>(pub P, pub Q);

impl<F, P, Q> DriftProcess<F> for Sum<P, Q>
where
    F: Float,
    P: DriftProcess<F>,
    Q: DriftProcess<F>,
{
    #[inline]
    fn mean(&self, t: F) -> F {
        self.0.mean(t) + self.1.mean(t)
    }

    #[inline]
    fn variance(&self, t: F) -> F {
        self.0.variance(t) + self.1.variance(t)
    }

    #[cfg(feature = "sample")]
    fn sample<R: RngCore>(&self, t: F, rng: &mut R) -> F
    where
        F: FloatConst,
    {
        self.0.sample(t, rng) + self.1.sample(t, rng)
    }
}

/// Three independent per-axis drift processes (x, y, z).
///
/// Returns `[F; 3]` from each query. Useful for IMUs with anisotropic
/// noise (e.g. different z-axis ARW from xy on a typical MEMS die).
#[derive(Copy, Clone, Debug)]
pub struct Vec3<P>(pub [P; 3]);

impl<P> Vec3<P> {
    /// μ(t) per axis.
    #[inline]
    pub fn mean<F: Float>(&self, t: F) -> [F; 3]
    where
        P: DriftProcess<F>,
    {
        [self.0[0].mean(t), self.0[1].mean(t), self.0[2].mean(t)]
    }

    /// σ²(t) per axis.
    #[inline]
    pub fn variance<F: Float>(&self, t: F) -> [F; 3]
    where
        P: DriftProcess<F>,
    {
        [
            self.0[0].variance(t),
            self.0[1].variance(t),
            self.0[2].variance(t),
        ]
    }

    /// σ(t) per axis.
    #[inline]
    pub fn std<F: Float>(&self, t: F) -> [F; 3]
    where
        P: DriftProcess<F>,
    {
        [self.0[0].std(t), self.0[1].std(t), self.0[2].std(t)]
    }

    /// Marginal sample per axis at time `t`.
    #[cfg(feature = "sample")]
    pub fn sample<F, R: RngCore>(&self, t: F, rng: &mut R) -> [F; 3]
    where
        F: Float + FloatConst,
        P: DriftProcess<F>,
    {
        [
            self.0[0].sample(t, rng),
            self.0[1].sample(t, rng),
            self.0[2].sample(t, rng),
        ]
    }
}

// `+` operator on drift processes — produces an independent `Sum`.
//
// We restrict the impl to the concrete process types in the crate by
// providing it through a marker trait `IsDriftProcess` that is
// implemented for each. (A blanket `impl<P, Q> Add for P where P:
// DriftProcess<F>` would clash with core's blanket impls.)
//
// Users compose via `process_a + process_b`.
mod add_impls {
    use super::Sum;
    use core::ops::Add;

    macro_rules! impl_add_for {
        ($($t:ty),* $(,)?) => {
            $(
                impl<Rhs> Add<Rhs> for $t {
                    type Output = Sum<$t, Rhs>;
                    #[inline]
                    fn add(self, rhs: Rhs) -> Sum<$t, Rhs> { Sum(self, rhs) }
                }
            )*
        }
    }

    use crate::processes::*;

    impl_add_for!(
        AngleRandomWalk<f32>, AngleRandomWalk<f64>,
        ConstantGyroBias<f32>, ConstantGyroBias<f64>,
        VelocityRandomWalk<f32>, VelocityRandomWalk<f64>,
        IntegratedVelocityRandomWalk<f32>, IntegratedVelocityRandomWalk<f64>,
        ConstantAccelBiasOnVelocity<f32>, ConstantAccelBiasOnVelocity<f64>,
        ConstantAccelBiasOnPosition<f32>, ConstantAccelBiasOnPosition<f64>,
        GyroBiasGravityCoupling<f32>, GyroBiasGravityCoupling<f64>,
        BiasInstabilityFlicker<f32>, BiasInstabilityFlicker<f64>,
        GaussMarkov<f32>, GaussMarkov<f64>,
    );

    // Sums compose with anything on the right.
    impl<P, Q, Rhs> Add<Rhs> for Sum<P, Q> {
        type Output = Sum<Sum<P, Q>, Rhs>;
        #[inline]
        fn add(self, rhs: Rhs) -> Sum<Sum<P, Q>, Rhs> {
            Sum(self, rhs)
        }
    }
}
