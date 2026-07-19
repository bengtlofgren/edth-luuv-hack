"""The shared 2-state (position, velocity) constant-velocity Kalman filter.

Identical seam for every supplier (POC 3 / POC 6 design): the only thing that
changes between "naive GP", "adaptive GP", "IMU-DR", "hybrid" etc. is what
(mean, variance) stream feeds the velocity measurement update. Generalized
from the POCs' fixed-dt loop to handle irregular sample spacing.
"""
from __future__ import annotations

import numpy as np


def run_kf(t, meas, meas_var, has_meas, q, v0, var0):
    """2-state constant-velocity KF: state x = [position, velocity].

    t: (N,) time [s], possibly irregular -- F, Q are rebuilt each step from
        dt = t[k] - t[k-1] (dt = 0, i.e. no predict, at k=0).
    meas, meas_var, has_meas: (N,) velocity measurement [m/s], its variance,
        and whether a measurement is actually available at that step (bridge
        samples inside a gap are sparse; has_meas is False elsewhere in the
        gap, in which case meas[k]/meas_var[k] are ignored).
    q: accel-noise PSD [(m/s^2)^2 / s], i.e. the process noise density behind
        the standard constant-velocity Q = q * [[dt^3/3, dt^2/2],[dt^2/2, dt]].
    v0, var0: initial velocity [m/s] and its variance; initial position is 0
        with a tight prior (1e-4 m^2), matching the POCs.

    Returns xs (N,2) = [position, velocity] and Ps (N,2,2) covariance.
    """
    t = np.asarray(t, float)
    meas = np.asarray(meas, float)
    meas_var = np.asarray(meas_var, float)
    has_meas = np.asarray(has_meas, bool)
    n = t.size
    H = np.array([[0.0, 1.0]])
    x = np.array([0.0, v0], float)
    P = np.diag([1e-4, var0])
    xs = np.empty((n, 2))
    Ps = np.empty((n, 2, 2))
    t_prev = t[0] if n else 0.0
    for k in range(n):
        dt = float(t[k] - t_prev) if k > 0 else 0.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x, P = F @ x, F @ P @ F.T + Q
        if has_meas[k]:
            S = (H @ P @ H.T)[0, 0] + meas_var[k]
            K = (P @ H.T / S).ravel()
            x = x + K * (meas[k] - x[1])
            P = P - np.outer(K, H @ P)
        xs[k] = x
        Ps[k] = P
        t_prev = t[k]
    return xs, Ps
