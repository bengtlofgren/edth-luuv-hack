"""Tests for eval.bridges.hybrid_v2 (chi-square gate + covariance intersection)."""
import numpy as np

from eval.bridges import hybrid_v2


def test_fused_var_never_below_min_input_var():
    rng = np.random.default_rng(2)
    n = 500
    gp_mean = rng.normal(0.0, 1.0, n)
    gp_var = rng.uniform(0.005, 1.0, n)
    imu_mean = gp_mean + rng.normal(0.0, 2.0, n)     # some huge disagreements too
    imu_var = rng.uniform(0.005, 1.0, n)

    mean, var = hybrid_v2(gp_mean, gp_var, imu_mean, imu_var)
    assert np.all(var >= np.minimum(gp_var, imu_var) - 1e-9)


def test_gate_fires_on_large_discrepancy():
    gp_mean = np.array([0.0])
    gp_var = np.array([0.01])
    imu_mean = np.array([5.0])                        # d=5, way outside combined sigma
    imu_var = np.array([0.01])

    mean, var = hybrid_v2(gp_mean, gp_var, imu_mean, imu_var)
    # gate must have failed -> both variances inflated by d^2=25 before CI,
    # so the fused (min-picking) variance should reflect that inflation.
    assert var[0] > 20.0


def test_gate_does_not_fire_on_small_discrepancy():
    gp_mean = np.array([1.0])
    gp_var = np.array([0.05])
    imu_mean = np.array([1.05])                        # small, consistent disagreement
    imu_var = np.array([0.05])

    mean, var = hybrid_v2(gp_mean, gp_var, imu_mean, imu_var)
    # gate passes -> CI over the RAW variances -> fused var == min(raw vars)
    assert np.isclose(var[0], min(gp_var[0], imu_var[0]))


def test_symmetric_under_input_swap():
    gp_mean = np.array([1.0, -2.0, 0.3])
    gp_var = np.array([0.05, 0.2, 0.01])
    imu_mean = np.array([1.3, -1.0, 4.0])
    imu_var = np.array([0.1, 0.05, 0.02])

    m1, v1 = hybrid_v2(gp_mean, gp_var, imu_mean, imu_var)
    m2, v2 = hybrid_v2(imu_mean, imu_var, gp_mean, gp_var)
    assert np.allclose(v1, v2)
    assert np.allclose(m1, m2)
