// SPDX-License-Identifier: AGPL-3.0-or-later

use imu_drift::{
    AngleRandomWalk, BiasInstabilityFlicker, ConstantAccelBiasOnPosition,
    ConstantGyroBias, DriftProcess, GaussMarkov, GyroBiasGravityCoupling,
    IntegratedVelocityRandomWalk, Sum,
};

const EPS: f64 = 1e-12;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

// -----------------------------------------------------------------------------
// Closed-form math properties (no RNG needed)
// -----------------------------------------------------------------------------

#[test]
fn arw_variance_matches_n_squared_t() {
    let p = AngleRandomWalk { arw: 0.01_f64 };
    assert!(approx_eq(p.variance(0.0), 0.0, EPS));
    assert!(approx_eq(p.variance(1.0), 1e-4, EPS));
    assert!(approx_eq(p.variance(100.0), 1e-2, 1e-14));
}

#[test]
fn arw_std_scales_as_sqrt_t() {
    let p = AngleRandomWalk { arw: 0.01_f64 };
    let s1 = p.std(1.0);
    assert!(approx_eq(p.std(4.0), 2.0 * s1, 1e-14));
    assert!(approx_eq(p.std(9.0), 3.0 * s1, 1e-14));
    assert!(approx_eq(p.std(100.0), 10.0 * s1, 1e-14));
}

#[test]
fn integrated_vrw_scales_as_t_three_halves() {
    let p = IntegratedVelocityRandomWalk { vrw: 0.05_f64 };
    let s1 = p.std(1.0);
    // σ ∝ t^{3/2} ⇒ σ(4)/σ(1) = 4^{3/2} = 8.
    assert!(approx_eq(p.std(4.0) / s1, 8.0, 1e-12));
    assert!(approx_eq(p.std(9.0) / s1, 27.0, 1e-12));
}

#[test]
fn constant_gyro_bias_is_linear_and_deterministic() {
    let p = ConstantGyroBias { b_g: 1e-3_f64 }; // 1 mrad/s
    assert!(approx_eq(p.mean(60.0), 0.06, EPS));
    assert!(approx_eq(p.variance(60.0), 0.0, EPS));
}

#[test]
fn const_accel_bias_on_position_is_quadratic() {
    let p = ConstantAccelBiasOnPosition { b_a: 0.01_f64 };
    let expected = 0.5 * 0.01 * 60.0_f64.powi(2);
    assert!(approx_eq(p.mean(60.0), expected, EPS));
}

#[test]
fn gyro_bias_gravity_coupling_is_cubic() {
    let p = GyroBiasGravityCoupling {
        g: 9.81_f64,
        b_g: 1e-4,
    };
    let expected = 0.5 * 9.81 * 1e-4 * 60.0_f64.powi(3);
    assert!(approx_eq(p.mean(60.0), expected, 1e-9));
}

#[test]
fn flicker_variance_law_matches_constant_times_t_squared() {
    let b = 1e-5_f64;
    let p = BiasInstabilityFlicker { b };
    let coef = 2.0 * 2.0_f64.ln() / core::f64::consts::PI;
    let t = 60.0_f64;
    let expected = coef * b * b * t * t;
    assert!(approx_eq(p.variance(t), expected, 1e-20));
}

#[test]
fn gauss_markov_variance_saturates() {
    let p = GaussMarkov {
        sigma: 1e-3_f64,
        tau_c: 100.0,
    };
    assert!(approx_eq(p.variance(0.0), 0.0, EPS));
    let var_inf = p.variance(1e9);
    assert!(approx_eq(var_inf, 1e-6, 1e-15));
    // At t = τ_c/2, variance = σ²(1 - e^{-1}).
    let v_half = p.variance(50.0);
    let expected_half = 1e-6 * (1.0 - (-1.0_f64).exp());
    assert!(approx_eq(v_half, expected_half, 1e-15));
}

// -----------------------------------------------------------------------------
// Composition: Sum + Add operator
// -----------------------------------------------------------------------------

#[test]
fn sum_adds_mean_and_variance() {
    let arw = AngleRandomWalk { arw: 0.01_f64 };
    let bias = ConstantGyroBias { b_g: 1e-3_f64 };
    let composed = Sum(arw, bias);
    let t = 60.0_f64;
    assert!(approx_eq(
        composed.mean(t),
        arw.mean(t) + bias.mean(t),
        EPS
    ));
    assert!(approx_eq(
        composed.variance(t),
        arw.variance(t) + bias.variance(t),
        EPS
    ));
}

#[test]
fn add_operator_produces_sum() {
    let arw = AngleRandomWalk { arw: 0.01_f64 };
    let bias = ConstantGyroBias { b_g: 1e-3_f64 };
    let composed = arw + bias;
    let t = 30.0_f64;
    let expected_var = arw.variance(t) + bias.variance(t);
    let expected_mean = arw.mean(t) + bias.mean(t);
    assert!(approx_eq(composed.variance(t), expected_var, EPS));
    assert!(approx_eq(composed.mean(t), expected_mean, EPS));
}

#[test]
fn three_way_sum_via_add_operator_chains() {
    let arw = AngleRandomWalk { arw: 0.01_f64 };
    let flicker = BiasInstabilityFlicker { b: 1e-5_f64 };
    let bias = ConstantGyroBias { b_g: 1e-3_f64 };
    let composed = arw + flicker + bias;
    let t = 60.0_f64;
    let expected_var = arw.variance(t) + flicker.variance(t) + bias.variance(t);
    let expected_mean = arw.mean(t) + flicker.mean(t) + bias.mean(t);
    assert!(approx_eq(composed.variance(t), expected_var, 1e-20));
    assert!(approx_eq(composed.mean(t), expected_mean, EPS));
}

// -----------------------------------------------------------------------------
// Numeric sanity from the literature
// -----------------------------------------------------------------------------

#[test]
fn woodman_xsens_60s_sanity() {
    // Woodman 2007 reports ~150 m position drift after 60 s of operation
    // with a representative MEMS IMU (Xsens MTx). The dominant term is
    // the cubic gyro-bias × gravity coupling: ε_p ≈ ½·g·b_g·t³.
    //
    // Inverting for b_g given ε_p(60s) ≈ 150 m, g = 9.81:
    //   b_g = 2·150 / (9.81 · 60³) ≈ 1.41e-4 rad/s ≈ 0.0081 deg/s
    // This is consistent with order-of-magnitude MEMS gyro biases of the era.
    let b_g = 1.41e-4_f64;
    let p = GyroBiasGravityCoupling { g: 9.81, b_g };
    let drift = p.mean(60.0);
    assert!(
        drift > 140.0 && drift < 160.0,
        "expected ~150 m, got {} m",
        drift
    );
}

// -----------------------------------------------------------------------------
// Allan variance — synthetic white noise should give σ_A(τ) = N/√τ
// -----------------------------------------------------------------------------

#[cfg(feature = "sample")]
mod allan_tests {
    use imu_drift::allan::{allan_variance, fit_arw_bias_instab_rrw};
    use imu_drift::sampling::standard_normal;
    use rand_core::SeedableRng;

    #[test]
    fn white_noise_allan_deviation_recovers_arw() {
        // For continuous white noise with PSD N², discrete samples at
        // period dt have variance N²/dt. The Allan variance of the
        // integrated angle satisfies σ_A²(τ) = N²/τ at long τ.
        let n: f64 = 0.01;     // ARW = 0.01 rad/√s
        let dt: f64 = 0.01;    // 100 Hz
        let n_samples = 200_000usize;
        let std_per_sample = n / dt.sqrt();

        let mut rng = rand_pcg::Pcg64::seed_from_u64(0xABBA_5EED);
        let mut samples = vec![0.0_f64; n_samples];
        for s in samples.iter_mut() {
            *s = std_per_sample * standard_normal::<f64, _>(&mut rng);
        }

        // Pick τ values well within [dt, N·dt/4] so we have enough clusters.
        let taus = [0.1_f64, 0.5, 1.0, 5.0];
        let mut out = [0.0_f64; 4];
        allan_variance(&samples, dt, &taus, &mut out).unwrap();

        for (i, &tau) in taus.iter().enumerate() {
            let sigma_a = out[i].sqrt();
            let expected = n / tau.sqrt();
            let rel_err = (sigma_a - expected).abs() / expected;
            assert!(
                rel_err < 0.10,
                "τ={}: σ_A got {}, expected {}, rel_err {:.3}",
                tau,
                sigma_a,
                expected,
                rel_err
            );
        }
    }

    #[test]
    fn fit_recovers_arw_from_white_noise_curve() {
        // Same white-noise dataset; verify that fit_arw_bias_instab_rrw
        // recovers N within 10% from the τ ≈ 1 s point.
        let n: f64 = 0.01;
        let dt: f64 = 0.01;
        let n_samples = 200_000usize;
        let std_per_sample = n / dt.sqrt();

        let mut rng = rand_pcg::Pcg64::seed_from_u64(0xC0FFEE);
        let mut samples = vec![0.0_f64; n_samples];
        for s in samples.iter_mut() {
            *s = std_per_sample * standard_normal::<f64, _>(&mut rng);
        }

        let taus = [0.1_f64, 0.5, 1.0, 3.0, 5.0];
        let mut sigma_a_sq = [0.0_f64; 5];
        allan_variance(&samples, dt, &taus, &mut sigma_a_sq).unwrap();
        let fit = fit_arw_bias_instab_rrw(&taus, &sigma_a_sq).unwrap();

        let rel_err = (fit.arw - n).abs() / n;
        assert!(
            rel_err < 0.10,
            "fit ARW = {}, expected {}, rel_err {:.3}",
            fit.arw,
            n,
            rel_err
        );
    }
}

// -----------------------------------------------------------------------------
// Brownian sample path (sanity: marginal variance growth holds in expectation)
// -----------------------------------------------------------------------------

#[cfg(feature = "sample")]
#[test]
fn arw_sample_path_is_anchored_brownian() {
    use rand_core::SeedableRng;
    let p = AngleRandomWalk { arw: 0.1_f64 };
    let mut rng = rand_pcg::Pcg64::seed_from_u64(42);

    // Average sample variance at the largest grid time over many trials
    // should approach σ²(t_max) = N²·t_max.
    let ts: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let n_trials = 5_000;
    let mut sum_sq_at_4 = 0.0_f64;
    let mut out = [0.0_f64; 4];
    for _ in 0..n_trials {
        p.sample_path(&ts, &mut out, &mut rng);
        sum_sq_at_4 += out[3] * out[3];
    }
    let estimated_var = sum_sq_at_4 / (n_trials as f64);
    let expected_var = 0.1_f64 * 0.1 * 4.0; // N² · t = 0.04
    let rel_err = (estimated_var - expected_var).abs() / expected_var;
    assert!(
        rel_err < 0.07,
        "estimated var {}, expected {}, rel_err {:.3}",
        estimated_var,
        expected_var,
        rel_err
    );
}
