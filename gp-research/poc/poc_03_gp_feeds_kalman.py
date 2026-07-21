# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""POC 3 — The key integration seam: GP output as a Kalman measurement with
adaptive R.

The implementation plan's central claim (implementation/00) is that a GP needs
ZERO changes to the existing EKF, because the EKF already accepts a
per-measurement covariance.  The GP simply supplies (mean, variance) pairs:

    measurement value   = GP posterior mean
    measurement noise R = GP posterior variance   <- "adaptive R"

This POC shows why the variance matters.  A toy 1-D vehicle integrates
velocity into position.  The DVL drops out for 15 s and -- worst case -- the
vehicle DECELERATES during the blackout, so the GP's bridged velocity is
smoothly wrong (it cannot know about the manoeuvre).  We fuse the identical
bridge values two ways:

  (a) naive   : fixed R, as if bridged values were as good as real DVL
  (b) adaptive: R = GP variance, so trust decays through the gap

Both filters end up off course -- that is unavoidable without a sensor.  The
difference is HONESTY: the adaptive filter's reported uncertainty covers its
actual error, while the naive filter claims centimetre confidence while being
metres wrong.  Downstream consumers (guidance, operator display) live or die
by that difference.

Run:  uv run poc_03_gp_feeds_kalman.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

rng = np.random.default_rng(3)

# --- 1. Truth: cruise at 1.0 m/s, then decelerate to 0.65 m/s at t = 30 s --
#     in the MIDDLE of the DVL blackout (25..40 s), where no sensor sees it.
#     (A manoeuvre of this size is consistent with the GP's prior of 0.3 m/s
#     signal std; a wilder one would defeat the GP's own uncertainty too.)
dt = 1.0
t = np.arange(0.0, 60.0, dt)
v_true = np.where(t < 30.0, 1.0, 0.65)
x_true = np.cumsum(v_true) * dt

# --- 2. DVL stream with the dropout; GP fitted on the surviving samples
#     (fixed, physically-sensible hyperparameters -- see POC 2 for why).
noise_std = 0.08
v_dvl = v_true + rng.normal(0.0, noise_std, t.shape)
have_dvl = ~((t > 25.0) & (t < 40.0))
v_cruise = np.mean(v_dvl[have_dvl])
kernel = (ConstantKernel(0.3**2, constant_value_bounds="fixed")
          * Matern(length_scale=5.0, nu=1.5, length_scale_bounds="fixed")
          + WhiteKernel(noise_level=noise_std**2, noise_level_bounds="fixed"))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(t[have_dvl].reshape(-1, 1), v_dvl[have_dvl] - v_cruise)
gp_mean, gp_std = gp.predict(t.reshape(-1, 1), return_std=True)
gp_mean += v_cruise

# --- 3. A deliberately tiny Kalman filter.  State: [position, velocity].
#     Prediction = constant velocity + white acceleration noise; measurement
#     = velocity only (exactly the DVL's role in the real stack).
F = np.array([[1.0, dt], [0.0, 1.0]])
q = 0.05**2                                   # accel noise density [(m/s^2)^2]
Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
H = np.array([[0.0, 1.0]])


def run_kf(meas, meas_var, has_meas):
    """Standard KF; meas_var[k] is the (possibly adaptive) R at step k;
    steps with has_meas[k] == False just propagate (no update)."""
    x, P = np.array([0.0, 1.0]), np.diag([1e-4, noise_std**2])
    xs, Ps = [], []
    for k in range(len(meas)):
        x, P = F @ x, F @ P @ F.T + Q                       # predict
        if has_meas[k]:
            S = (H @ P @ H.T)[0, 0] + meas_var[k]           # innovation var
            K = (P @ H.T / S).ravel()                       # gain
            x = x + K * (meas[k] - x[1])                    # update
            P = P - np.outer(K, H @ P)
        xs.append(x.copy()); Ps.append(P.copy())
    return np.array(xs), np.array(Ps)


# --- 4. Both filters consume the SAME values: real DVL every second when
#     present, and inside the gap a GP bridge sample every 5 s (roughly the
#     GP's lengthscale).  Injecting bridges MORE often would feed the filter
#     the same smoothly-wrong value again and again -- time-correlated errors
#     that per-step white R cannot express (the reason implementation/00 caps
#     bridging with `max_bridge_s`).  Only the trust differs between filters:
bridge = ~have_dvl & (t.astype(int) % 5 == 2)               # t = 27, 32, 37 s
has_meas = have_dvl | bridge
meas = np.where(have_dvl, v_dvl, gp_mean)
R_naive = np.full_like(t, noise_std**2)                     # (a) constant trust
R_adapt = np.where(have_dvl, noise_std**2, gp_std**2)       # (b) GP-supplied trust

xs_naive, Ps_naive = run_kf(meas, R_naive, has_meas)
xs_adapt, Ps_adapt = run_kf(meas, R_adapt, has_meas)

# --- 5. Honesty check on the VELOCITY state (the quantity the GP measures):
#     is the actual error inside the filter's own claimed 2-sigma?  Report the
#     worst error/claim ratio over the run (<= 1 means the filter never lied).
#     Position honesty is a harder ask for BOTH filters: position error is the
#     INTEGRAL of the correlated velocity errors, so it is reported for
#     context only.
err_naive = np.abs(xs_naive[:, 1] - v_true)
err_adapt = np.abs(xs_adapt[:, 1] - v_true)
sig_naive = 2 * np.sqrt(Ps_naive[:, 1, 1])
sig_adapt = 2 * np.sqrt(Ps_adapt[:, 1, 1])
print(f"peak position error       naive: {np.abs(xs_naive[:,0]-x_true).max():5.2f} m | "
      f"adaptive: {np.abs(xs_adapt[:,0]-x_true).max():5.2f} m  (drift: unavoidable without a sensor)")
print(f"worst velocity error/2s   naive: {np.max(err_naive/sig_naive):5.1f}x | "
      f"adaptive: {np.max(err_adapt/sig_adapt):5.1f}x   (<=1 = honest)")

# --- 6. Plot: each filter's claimed velocity 2-sigma vs its actual error.
fig, ax = plt.subplots(figsize=(10, 4))
ax.axvspan(25, 40, color="0.9", label="DVL dropout (GP bridging; vehicle decelerates at 30 s)")
ax.plot(t, err_naive, "C3", label="|velocity error|, naive fixed R")
ax.plot(t, sig_naive, "C3--", lw=1, label="naive claimed 2$\\sigma$ (overconfident)")
ax.plot(t, err_adapt, "C0", label="|velocity error|, adaptive R = GP variance")
ax.plot(t, sig_adapt, "C0--", lw=1, label="adaptive claimed 2$\\sigma$ (covers the error)")
ax.set(xlabel="time [s]", ylabel="velocity error [m/s]",
       title="POC 3: GP variance as adaptive measurement noise keeps the Kalman filter honest")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("poc_03_gp_feeds_kalman.png", dpi=120)
print("saved poc_03_gp_feeds_kalman.png")
