# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""POC 2 — DVL dropout bridging + outlier rejection with a GP.

Demonstrates the two operational modes from references/04 and the
implementation plan that go beyond plain denoising:

  1. BRIDGE: when the DVL loses bottom-lock (here: a 15 s gap), the GP still
     predicts a velocity inside the gap, and its variance GROWS with distance
     from the last good sample.  Graceful degradation, not a cliff -- the
     downstream filter is told exactly how little to trust the bridged values.

  2. OUTLIER REJECTION: an incoming sample far outside the GP's predictive
     band (large z-score) is flagged and dropped BEFORE it can pollute fusion.

Run:  uv run poc_02_dropout_bridge.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

rng = np.random.default_rng(7)

# --- 1. Truth and a 1 Hz DVL stream, as in POC 1.
t_grid = np.linspace(0.0, 60.0, 600)
v_true = 1.0 + 0.3 * np.sin(2 * np.pi * t_grid / 25.0)
t_dvl = np.arange(0.5, 60.0, 1.0)
v_dvl = np.interp(t_dvl, t_grid, v_true) + rng.normal(0.0, 0.08, t_dvl.shape)

# --- 2. Inject the two failure modes the DVL actually has:
#     a dropout (loss of bottom-lock, t = 25..40 s) and one gross outlier.
dropout = (t_dvl > 25.0) & (t_dvl < 40.0)
outlier_idx = 15                       # sample at t = 15.5 s
v_dvl[outlier_idx] += 1.2              # a 1.2 m/s spike -- clearly wrong

# --- 3. Fit the GP only on samples OUTSIDE the dropout (they never arrived).
#     The outlier is left IN on purpose -- step 5 shows the GP catching it.
#     NOTE: hyperparameters are FIXED here, not learned as in POC 1.  A single
#     gross outlier can wreck marginal-likelihood fitting (it drags the
#     lengthscale to zero so the GP can "explain" the spike); a real system
#     learns hyperparameters only from vetted data.  Small lesson, for free.
#     Everything is in physical units: signal std 0.3 m/s around the cruise
#     speed (subtracted below), lengthscale 5 s, sensor noise 0.08 m/s.
t_fit, v_fit = t_dvl[~dropout], v_dvl[~dropout]
v_cruise = np.mean(v_fit)                       # GP models deviations from this
kernel = (ConstantKernel(0.3**2, constant_value_bounds="fixed")
          * Matern(length_scale=5.0, nu=1.5, length_scale_bounds="fixed")
          + WhiteKernel(noise_level=0.08**2, noise_level_bounds="fixed"))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(t_fit.reshape(-1, 1), v_fit - v_cruise)
v_mean, v_std = gp.predict(t_grid.reshape(-1, 1), return_std=True)
v_mean += v_cruise

# --- 4. MODE 1 (bridge): show that uncertainty grows inside the gap.  These
#     (mean, std) pairs are what would be injected into the EKF as synthetic
#     'CorrectedDvlMeasurement's during the silence.
in_gap = (t_grid > 25.0) & (t_grid < 40.0)
print(f"std at gap edge   : {v_std[np.argmin(np.abs(t_grid-25.5))]*100:5.1f} cm/s")
print(f"std mid-gap (32 s): {v_std[np.argmin(np.abs(t_grid-32.5))]*100:5.1f} cm/s  <- grows: trust decays")
gap_rmse = np.sqrt(np.mean((v_mean[in_gap] - v_true[in_gap]) ** 2))
print(f"bridged-velocity RMSE inside the 15 s gap: {gap_rmse*100:.1f} cm/s")

# --- 5. MODE 2 (outlier gate): z-score each sample against the GP's
#     PREDICTIVE band for observations -- latent std plus the sensor noise
#     (sklearn's return_std excludes the noise, so we add it back).  A
#     leave-self-out prediction would be ideal; the plain posterior already
#     exposes the spike because 40+ neighbours pin the mean down while one
#     spike cannot drag it far.
mu_i, std_i = gp.predict(t_fit.reshape(-1, 1), return_std=True)
z = np.abs(v_fit - v_cruise - mu_i) / np.sqrt(std_i**2 + 0.08**2)
flagged = z > 3.0                      # the classic 3-sigma gate
print(f"samples flagged as outliers (z > 3): {np.sum(flagged)} "
      f"(true outlier at t={t_dvl[outlier_idx]:.1f}s flagged: "
      f"{bool(flagged[np.argmin(np.abs(t_fit - t_dvl[outlier_idx]))])})")

# --- 6. Plot.
fig, ax = plt.subplots(figsize=(10, 4))
ax.axvspan(25, 40, color="0.9", label="DVL dropout (no bottom-lock)")
ax.plot(t_grid, v_true, "k--", lw=1, label="true velocity")
ax.plot(t_fit[~flagged], v_fit[~flagged], "r.", ms=5, label="DVL samples")
ax.plot(t_fit[flagged], v_fit[flagged], "rx", ms=10, mew=2, label="flagged outlier (z>3)")
ax.plot(t_grid, v_mean, "C0", lw=2, label="GP mean (bridges the gap)")
ax.fill_between(t_grid, v_mean - 2 * v_std, v_mean + 2 * v_std, color="C0", alpha=0.2,
                label="GP 95% band (widens in gap)")
ax.set(xlabel="time [s]", ylabel="surge velocity [m/s]",
       title="POC 2: GP bridges a DVL dropout with growing uncertainty and flags outliers")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("poc_02_dropout_bridge.png", dpi=120)
print("saved poc_02_dropout_bridge.png")
