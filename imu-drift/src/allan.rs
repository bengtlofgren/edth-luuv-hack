// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Allan variance estimation per IEEE 952-2020 (Annex C) and El-Sheimy,
// Hou, Niu 2008 [ESN08]. Pure no_std, caller-provides all buffers, zero
// allocation.

use num_traits::{Float, FloatConst};

/// Errors returned by [`allan_variance`] and [`fit_arw_bias_instab_rrw`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AllanError {
    /// Sample buffer too short to form even one cluster pair at the
    /// requested averaging time.
    InsufficientSamples { needed: usize, got: usize },
    /// `dt` was zero or negative.
    NonPositiveDt,
    /// A requested averaging time τ exceeds half the dataset length.
    TauTooLarge,
    /// `out.len() != taus.len()` or `sigma_a_sq.len() != taus.len()`.
    OutputLengthMismatch,
    /// Fewer than two τ values supplied to the fitter.
    InsufficientTaus,
}

/// Compute the **non-overlapping** Allan variance σ_A²(τ) for each τ in
/// `taus`, using the rate-domain definition:
///
/// ```text
///                        1                M-1
///     σ_A²(τ)  =  ──────────────────  ·   Σ   ( Ω̄_{k+1}(τ) − Ω̄_k(τ) )²
///                  2·(M − 1)               k=1
/// ```
///
/// where `Ω̄_k(τ)` is the average of `n = round(τ / dt)` consecutive
/// samples and `M = ⌊N / n⌋` is the number of full clusters.
///
/// Source: IEEE Std 952-2020, Annex C, eq. (C.4) [IEEE952]; equivalent
/// formulation in El-Sheimy et al. 2008, eq. (4) [ESN08].
///
/// # Arguments
///
/// - `samples`: rate samples (rad/s for gyro, m/s² for accel)
/// - `dt`: sample period τ₀ in seconds (must be > 0)
/// - `taus`: averaging times in seconds at which to evaluate σ_A²
/// - `out`: output buffer for σ_A² values, **same length as `taus`**
///
/// # Errors
/// Returns `AllanError` if buffers are mismatched or τ exceeds the
/// largest τ supportable by the dataset (≈ N · dt / 2).
pub fn allan_variance<F: Float>(
    samples: &[F],
    dt: F,
    taus: &[F],
    out: &mut [F],
) -> Result<(), AllanError> {
    if dt <= F::zero() {
        return Err(AllanError::NonPositiveDt);
    }
    if out.len() != taus.len() {
        return Err(AllanError::OutputLengthMismatch);
    }
    let n_total = samples.len();
    if n_total < 4 {
        return Err(AllanError::InsufficientSamples {
            needed: 4,
            got: n_total,
        });
    }

    for (i, &tau) in taus.iter().enumerate() {
        if tau <= F::zero() {
            return Err(AllanError::NonPositiveDt);
        }
        // n = round(tau / dt) — number of base samples per cluster.
        let n_f = (tau / dt).round();
        let n = n_f.to_usize().ok_or(AllanError::TauTooLarge)?;
        if n == 0 {
            return Err(AllanError::TauTooLarge);
        }
        let m = n_total / n; // number of full clusters
        if m < 2 {
            return Err(AllanError::InsufficientSamples {
                needed: 2 * n,
                got: n_total,
            });
        }

        // Compute cluster means inline; track previous to form
        // (Ω̄_{k+1} − Ω̄_k)² without storing the full sequence.
        let mut sum_diff_sq = F::zero();
        let n_inv = F::one() / F::from(n).ok_or(AllanError::TauTooLarge)?;

        // First cluster mean
        let mut s = F::zero();
        for &x in &samples[0..n] {
            s = s + x;
        }
        let mut prev_mean = s * n_inv;

        for k in 1..m {
            let start = k * n;
            let end = start + n;
            let mut s = F::zero();
            for &x in &samples[start..end] {
                s = s + x;
            }
            let cur_mean = s * n_inv;
            let diff = cur_mean - prev_mean;
            sum_diff_sq = sum_diff_sq + diff * diff;
            prev_mean = cur_mean;
        }

        // σ_A²(τ) = sum_diff_sq / (2 · (M − 1))
        let two = F::one() + F::one();
        let denom = two * F::from(m - 1).ok_or(AllanError::TauTooLarge)?;
        out[i] = sum_diff_sq / denom;
    }

    Ok(())
}

/// Recovered noise parameters from a slope-read of an Allan-deviation curve.
#[derive(Copy, Clone, Debug)]
pub struct NoiseFit<F> {
    /// Angle/Velocity Random Walk N — read from σ_A(τ=1 s) on the −½
    /// slope segment. Units: rad/√s for gyro, m/s/√s for accel.
    pub arw: F,
    /// Bias instability B — read from the floor (slope-0 minimum) of
    /// the Allan-deviation curve. Conversion: B = σ_A_min · √(π / (2 ln 2)).
    /// Units: rad/s for gyro, m/s² for accel.
    pub bias_instab: F,
    /// Rate Random Walk K — read from σ_A(τ=3 s) on the +½ slope.
    /// Units: rad/s^{3/2} for gyro, m/s²/√s for accel.
    pub rrw: F,
}

/// Read off (N, B, K) from a precomputed Allan curve via the canonical
/// slope-segment lookups (IEEE 952-2020 Annex C).
///
/// - **N** = σ_A(τ = 1 s)                        (slope −½ identity)
/// - **B** = min_τ σ_A(τ) · √(π / (2 ln 2))      (floor identity, [ESN08] eq. 12)
/// - **K** = σ_A(τ = 3 s) · √3                   (slope +½ identity at τ = 3 s)
///
/// Inputs:
/// - `taus`: must be sorted ascending and contain points near 1 s and 3 s
/// - `sigma_a_sq`: the σ_A²(τ) values from [`allan_variance`]
///
/// If τ = 1 s or τ = 3 s are not present exactly, the closest τ is used.
/// For higher-precision fitting, perform log-log least squares over each
/// slope segment (deferred to v0.2).
pub fn fit_arw_bias_instab_rrw<F: Float + FloatConst>(
    taus: &[F],
    sigma_a_sq: &[F],
) -> Result<NoiseFit<F>, AllanError> {
    if taus.len() != sigma_a_sq.len() {
        return Err(AllanError::OutputLengthMismatch);
    }
    if taus.len() < 2 {
        return Err(AllanError::InsufficientTaus);
    }

    let one = F::one();
    let three = one + one + one;

    // σ_A(τ) = √σ_A²(τ).
    // 1) Closest τ to 1.0 s → N = σ_A(τ).
    // 2) Closest τ to 3.0 s → K = σ_A(τ) · √3.
    // 3) Min of σ_A(τ) over all τ → B = min · √(π / (2 · ln 2)).
    let idx_near = |target: F| -> usize {
        let mut best = 0usize;
        let mut best_dist = (taus[0] - target).abs();
        for (i, &t) in taus.iter().enumerate().skip(1) {
            let d = (t - target).abs();
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        best
    };

    let i_one = idx_near(one);
    let i_three = idx_near(three);

    let arw = sigma_a_sq[i_one].max(F::zero()).sqrt();
    let rrw = sigma_a_sq[i_three].max(F::zero()).sqrt() * three.sqrt();

    let two = one + one;
    let mut min_sigma_sq = sigma_a_sq[0];
    for &v in &sigma_a_sq[1..] {
        if v < min_sigma_sq {
            min_sigma_sq = v;
        }
    }
    let min_sigma = min_sigma_sq.max(F::zero()).sqrt();
    let b_factor = (F::PI() / (two * F::LN_2())).sqrt();
    let bias_instab = min_sigma * b_factor;

    Ok(NoiseFit {
        arw,
        bias_instab,
        rrw,
    })
}
