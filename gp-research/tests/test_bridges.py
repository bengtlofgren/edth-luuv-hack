"""Tests for eval.bridges: Procrustes IMU calibration and the IMU DR bridge."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from eval.bridges import Calib, imu_bridge, imu_calibrate


def _synthetic_v_a(t, amps, omegas, phases):
    """Smooth synthetic 3-axis velocity and its analytic derivative (accel)."""
    v = np.column_stack([a * np.sin(w * t + p) for a, w, p in zip(amps, omegas, phases)])
    a = np.column_stack([a * w * np.cos(w * t + p) for a, w, p in zip(amps, omegas, phases)])
    return v, a


def test_procrustes_recovers_rotation_and_scale():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 60.0, 601)             # 10 Hz, 60 s
    v, a = _synthetic_v_a(t, amps=[0.5, 0.4, 0.3],
                           omegas=[0.3, 0.5, 0.7], phases=[0.0, 1.0, 2.5])

    R_true = Rotation.from_euler("xyz", [12.0, -20.0, 35.0], degrees=True).as_matrix()
    s_true = 0.8
    b_true = np.array([0.05, -0.03, 9.81])

    # f = R_true^T (a - b) / s  <=>  a = s * R_true @ f + b  (the model
    # imu_calibrate fits), plus small IMU measurement noise.
    f = (a - b_true) @ R_true / s_true
    f += rng.normal(0.0, 0.01, f.shape)
    w = np.zeros_like(v)                          # no rotation rate: isolates Procrustes

    calib = imu_calibrate(t, v, t, f, w, band_s=0.3)

    cos_ang = (np.trace(R_true.T @ calib.R) - 1.0) / 2.0
    angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
    assert angle_deg < 5.0, f"rotation off by {angle_deg:.2f} deg"
    assert abs(calib.s - s_true) / s_true < 0.10, f"scale off: {calib.s} vs {s_true}"


def test_procrustes_valid_under_gravity_dominated_attenuated_input():
    """Heavy noise on the IMU (predictor) side should collapse the
    UNCONSTRAINED ridge map's singular values well below the true rotation's
    (~s_true), while the Procrustes-constrained R stays a valid rotation
    regardless -- the whole point of review finding 3."""
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 60.0, 601)
    # tiny dynamic content...
    v, a = _synthetic_v_a(t, amps=[0.01, 0.008, 0.006],
                           omegas=[0.3, 0.5, 0.7], phases=[0.0, 1.0, 2.5])
    R_true = Rotation.from_euler("xyz", [5.0, 10.0, -15.0], degrees=True).as_matrix()
    s_true = 1.0
    b_true = np.array([0.0, 0.0, 9.81])           # ...dominated by a constant gravity offset

    f = (a - b_true) @ R_true / s_true
    f += rng.normal(0.0, 5.0, f.shape)              # heavy noise, >>> dynamic signal amplitude
    w = np.zeros_like(v)

    calib = imu_calibrate(t, v, t, f, w, band_s=0.3)

    # R stays a valid rotation no matter how attenuated the raw regression is.
    assert abs(np.linalg.det(calib.R) - 1.0) < 1e-6
    assert np.allclose(calib.R @ calib.R.T, np.eye(3), atol=1e-6)

    # The unconstrained (ridge) map's singular values are flagged as attenuated.
    assert np.all(calib.sv_unconstrained < 0.3), calib.sv_unconstrained


def test_imu_bridge_variance_increases_and_t2_term_dominates():
    calib = Calib(R=np.eye(3), s=1.0, b=np.zeros(3),
                   sigma_r=np.full(3, 0.01), tau=np.full(3, 1.0),
                   sigma_b=np.full(3, 0.5), sv_unconstrained=np.ones(3), band_s=1.0)
    t_imu = np.arange(0.0, 40.0, 0.1)
    f = np.zeros((t_imu.size, 3))
    w = np.zeros((t_imu.size, 3))

    t, mean, std = imu_bridge(calib, v0=np.zeros(3), t0=0.0, t_imu=t_imu, f=f, w=w,
                               t_end=30.0, var0=0.0)
    var = std ** 2
    assert np.all(np.diff(var[:, 0]) > 0), "variance must strictly increase with T"

    iT = np.argmin(np.abs(t - 10.0))
    i2T = np.argmin(np.abs(t - 20.0))
    var_T, var_2T = var[iT, 0], var[i2T, 0]
    # sigma_b dominates here (0.5^2*T^2 >> sigma_r^2*tau*T for these T) so the
    # T^2 term should make var(2T) more than double var(T).
    assert var_2T > 2.0 * var_T


def test_imu_bridge_tracks_constant_accel_truth():
    R_true = Rotation.from_euler("xyz", [8.0, -4.0, 20.0], degrees=True).as_matrix()
    s_true = 1.2
    a_const = np.array([0.3, -0.1, 0.05])
    f_const = R_true.T @ a_const / s_true          # constant specific force

    calib = Calib(R=R_true, s=s_true, b=np.zeros(3),
                   sigma_r=np.full(3, 1e-6), tau=np.full(3, 0.2),
                   sigma_b=np.full(3, 1e-6), sv_unconstrained=np.ones(3), band_s=1.0)
    t_imu = np.arange(0.0, 20.0, 0.05)
    f = np.tile(f_const, (t_imu.size, 1))
    w = np.zeros((t_imu.size, 3))
    v0 = np.array([1.0, 0.0, -0.5])

    t, mean, std = imu_bridge(calib, v0=v0, t0=0.0, t_imu=t_imu, f=f, w=w, t_end=15.0)
    truth = v0 + a_const * t[:, None]
    assert np.allclose(mean, truth, atol=1e-9)
