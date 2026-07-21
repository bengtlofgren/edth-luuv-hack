"""Dynamics-GP bridge on a known synthetic control system.

Ground truth: first-order linear vehicle dynamics per axis,
    dv/dt = -lam * v + B @ u,
driven by smooth random thrust commands. The bridge trains on a noisy
"healthy DVL" segment and must roll a maneuver-containing gap forward from
commands alone -- the exact setting where the time-indexed GP was shown to
be overconfident.
"""
import numpy as np
import pytest

from eval.bridges import dynamics_bridge, dynamics_calibrate


LAM = np.array([0.5, 0.6, 0.8])                # damping per axis
B = np.array([[0.8, 0.0, 0.1, 0.0],            # thrust gains (3 axes x 4 ch)
              [0.0, 0.7, 0.0, 0.2],
              [0.0, 0.0, 0.0, 0.6]])
NOISE = 0.01                                    # DVL measurement noise [m/s]


def _simulate(rng, dur=120.0, dt=0.05):
    """Integrate the true system under smooth random commands."""
    n = int(dur / dt)
    t = np.arange(n) * dt
    # smooth commands: random steps low-pass filtered, in [-1, 1]
    u = np.zeros((n, 4))
    step = rng.uniform(-1, 1, (n // 40 + 2, 4))
    for k in range(n):
        u[k] = step[k // 40]
    for i in range(4):
        u[:, i] = np.convolve(u[:, i], np.ones(30) / 30, mode="same")
    v = np.zeros((n, 3))
    for k in range(1, n):
        a = -LAM * v[k - 1] + B @ u[k - 1]
        v[k] = v[k - 1] + dt * a
    return t, v, u


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(7)
    t, v, u = _simulate(rng)
    # DVL observes at 10 Hz with noise; commands at 20 Hz
    obs = slice(None, None, 2)
    t_v = t[obs]
    v_meas = v[obs] + rng.normal(0, NOISE, (t_v.size, 3))
    return dict(t=t, v=v, u=u, t_v=t_v, v_meas=v_meas)


def test_dynamics_bridge_beats_zoh_on_maneuver_gap(system):
    t, v, u, t_v, v_meas = (system[k] for k in ("t", "v", "u", "t_v", "v_meas"))
    gap_start, gap_len = 80.0, 15.0
    train = t_v < gap_start
    cal = dynamics_calibrate(t_v[train], v_meas[train], t, u)

    i0 = np.flatnonzero(t_v < gap_start)[-1]
    tb, mean, std = dynamics_bridge(cal, v_meas[i0], t_v[i0], t, u,
                                     gap_start + gap_len, var0=NOISE ** 2)
    in_gap = (t >= gap_start) & (t < gap_start + gap_len)
    v_true = np.column_stack([np.interp(tb, t[in_gap], v[in_gap, ax])
                               for ax in range(3)])
    rmse_dyn = np.sqrt(((mean - v_true) ** 2).mean(axis=0))
    rmse_zoh = np.sqrt(((v_meas[i0][None, :] - v_true) ** 2).mean(axis=0))
    # commands change inside the gap, so ZOH must lose badly on every axis
    assert (rmse_dyn < 0.5 * rmse_zoh).all(), (rmse_dyn, rmse_zoh)
    assert (rmse_dyn < 0.05).all()


def test_dynamics_bridge_variance_covers_error(system):
    t, v, u, t_v, v_meas = (system[k] for k in ("t", "v", "u", "t_v", "v_meas"))
    gap_start, gap_len = 80.0, 15.0
    train = t_v < gap_start
    cal = dynamics_calibrate(t_v[train], v_meas[train], t, u)
    i0 = np.flatnonzero(t_v < gap_start)[-1]
    tb, mean, std = dynamics_bridge(cal, v_meas[i0], t_v[i0], t, u,
                                     gap_start + gap_len, var0=NOISE ** 2)
    in_gap = (t >= gap_start) & (t < gap_start + gap_len)
    v_true = np.column_stack([np.interp(tb, t[in_gap], v[in_gap, ax])
                               for ax in range(3)])
    cover = (np.abs(mean - v_true) <= 2 * std).mean()
    assert cover >= 0.85, cover
    # variance must grow through the rollout
    assert (std[-1] > std[0]).all()


def test_linear_mean_explains_linear_system(system):
    """On a truly linear system the ridge mean should do most of the work."""
    t, v, u, t_v, v_meas = (system[k] for k in ("t", "v", "u", "t_v", "v_meas"))
    cal = dynamics_calibrate(t_v, v_meas, t, u)
    assert (cal.lin_r2 > 0.8).all(), cal.lin_r2


def test_constant_command_channel_is_dropped():
    """A dead command channel (constant PWM) must not break standardization."""
    rng = np.random.default_rng(3)
    t, v, u = _simulate(rng, dur=60.0)
    u[:, 1] = 0.0                                    # channel never driven
    obs = slice(None, None, 2)
    t_v, v_meas = t[obs], v[obs] + rng.normal(0, NOISE, (t[obs].size, 3))
    cal = dynamics_calibrate(t_v, v_meas, t, u)
    assert not cal.keep[4]                           # feature 3 (v) + ch1 -> idx 4
    tb, mean, std = dynamics_bridge(cal, v_meas[-1], t_v[-1], t, u, t_v[-1] + 5.0)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
