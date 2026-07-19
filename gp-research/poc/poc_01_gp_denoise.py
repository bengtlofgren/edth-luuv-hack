# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""POC 1 — GP regression basics: denoising a DVL-like velocity signal.

Demonstrates the core claim of references/01-03: a GP fit to noisy samples of a
smooth function returns, at every query time, a predicted MEAN (a denoised
velocity) and a predicted VARIANCE (how much to trust it). Near data the
variance is small; away from data it grows. That mean+variance pair is exactly
what a Kalman filter wants as a measurement.

Run:  uv run poc_01_gp_denoise.py   ->  prints RMSE, saves poc_01_gp_denoise.png
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: save the figure instead of opening a window
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

rng = np.random.default_rng(42)

# --- 1. Synthetic truth: a smooth surge velocity, like an AUV gently manoeuvring.
t_true = np.linspace(0.0, 60.0, 600)  # seconds
v_true = 1.0 + 0.3 * np.sin(2 * np.pi * t_true / 25.0) + 0.1 * np.sin(2 * np.pi * t_true / 7.0)

# --- 2. DVL samples: ~1 Hz, corrupted by Gaussian noise (sigma = 8 cm/s).
t_dvl = np.arange(0.5, 60.0, 1.0)
noise_std = 0.08
v_dvl = np.interp(t_dvl, t_true, v_true) + rng.normal(0.0, noise_std, t_dvl.shape)

# --- 3. Fit a GP.  Matern(nu=1.5) is the research docs' recommended kernel for
#     vehicle velocity: smooth but not infinitely so (an RBF would over-smooth
#     real manoeuvres).  WhiteKernel lets the GP LEARN the sensor noise level
#     instead of us hard-coding it.  All hyperparameters (lengthscale, signal
#     variance, noise) are tuned automatically by marginal-likelihood
#     maximisation inside .fit() -- no hand-tuning.
kernel = Matern(length_scale=5.0, nu=1.5) + WhiteKernel(noise_level=noise_std**2)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(t_dvl.reshape(-1, 1), v_dvl)

# --- 4. Query the posterior everywhere: mean = denoised velocity, std = trust.
v_mean, v_std = gp.predict(t_true.reshape(-1, 1), return_std=True)

# --- 5. The takeaway, quantified: the GP mean is much closer to truth than the
#     raw samples are, i.e. the GP denoises the DVL.
rmse_raw = np.sqrt(np.mean((v_dvl - np.interp(t_dvl, t_true, v_true)) ** 2))
rmse_gp = np.sqrt(np.mean((v_mean - v_true) ** 2))
print(f"raw DVL RMSE : {rmse_raw*100:.1f} cm/s")
print(f"GP mean RMSE : {rmse_gp*100:.1f} cm/s   (denoised)")
print(f"learned kernel: {gp.kernel_}")

# --- 6. Plot: truth, noisy samples, GP mean with a 2-sigma band.
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_true, v_true, "k--", lw=1, label="true velocity")
ax.plot(t_dvl, v_dvl, "r.", ms=5, label="noisy DVL samples (1 Hz)")
ax.plot(t_true, v_mean, "C0", lw=2, label="GP posterior mean (denoised)")
ax.fill_between(t_true, v_mean - 2 * v_std, v_mean + 2 * v_std, color="C0", alpha=0.2,
                label="GP 95% band (the 'trust' signal)")
ax.set(xlabel="time [s]", ylabel="surge velocity [m/s]",
       title="POC 1: GP regression denoises DVL and reports its own uncertainty")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("poc_01_gp_denoise.png", dpi=120)
print("saved poc_01_gp_denoise.png")
