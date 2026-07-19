# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""POC 3b — the GP->Kalman adaptive-R seam of POC 3, re-run on REAL DVL data.

Same integration seam as poc_03_gp_feeds_kalman.py (GP posterior mean as the
measurement, GP posterior variance as adaptive R), but the synthetic sine
truth is replaced by the real-data harness: forward velocity from the Snapir
AUV (ECA A18-D, Teledyne RDI Workhorse Navigator DVL @ 1 Hz), test mission of
the ANSFL BeamsNet dataset (CC-BY-4.0, github.com/ansfl/BeamsNet).

Differences from the toy forced by real data:

  * No ground truth.  The velocity reference inside a simulated dropout is
    the HIDDEN DVL samples themselves (noisy); the position reference is
    dead-reckoning on the full DVL stream.  The honesty check therefore
    compares |error| against 2*sqrt(P + sigma_dvl^2), not 2*sqrt(P) -- the
    reference's own noise is added to the claim.
  * Hyperparameters are LEARNED per scenario (the toy fixed them), with the
    lengthscale floored above the sample spacing so unstructured variance
    lands in the WhiteKernel; sigma_dvl^2 is read back from the fitted
    WhiteKernel instead of being assumed.
  * Two segments (steady cruise / maneuver) x two outage lengths (30 s /
    60 s).  The maneuver segments are the cases where the plain-GP harness
    showed the +/-2sigma band failing to cover the error.

Run:  uv run poc_03_real_data.py
"""
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# --- 0. Locate the data (reuse a local copy if present, else download). -----
URL = "https://raw.githubusercontent.com/ansfl/BeamsNet/main/dataset/Test/V_test.npy"
CANDIDATES = [Path(__file__).with_name("data") / "V_test.npy",
              Path.home() / "dvl-gp" / "data" / "V_test.npy"]
data_path = next((p for p in CANDIDATES if p.exists()), None)
if data_path is None:
    data_path = CANDIDATES[0]
    data_path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, data_path)

AXIS = 0                                     # forward velocity: drives along-track drift
WINDOW = slice(600, 1200)                    # 10-min segment; 1 Hz -> index == seconds
SCENARIOS = {"steady cruise": 720, "maneuver": 950}
GAP_LENGTHS = [30, 60]
PLOT_GAP = 60                                # the stress case shown in the figure

v_all = np.load(data_path)[AXIS, WINDOW]
t = np.arange(WINDOW.start, WINDOW.stop, dtype=float)    # mission time [s]
dt = 1.0

# --- 1. The tiny KF from POC 3, unchanged: state [position, velocity],
#     constant-velocity prediction, velocity-only measurement.
F = np.array([[1.0, dt], [0.0, 1.0]])
q = 0.05**2                                   # accel noise density; identical for both
Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
H = np.array([[0.0, 1.0]])


def run_kf(meas, meas_var, has_meas, v0, p0_vel):
    x, P = np.array([0.0, v0]), np.diag([1e-4, p0_vel])
    xs, Ps = [], []
    for k in range(len(meas)):
        x, P = F @ x, F @ P @ F.T + Q
        if has_meas[k]:
            S = (H @ P @ H.T)[0, 0] + meas_var[k]
            K = (P @ H.T / S).ravel()
            x = x + K * (meas[k] - x[1])
            P = P - np.outer(K, H @ P)
        xs.append(x.copy()); Ps.append(P.copy())
    return np.array(xs), np.array(Ps)


def run_scenario(gap_start, gap_len):
    gap = (t >= gap_start) & (t < gap_start + gap_len)
    have_dvl = ~gap

    # --- 2. GP on the surviving samples, hyperparameters learned (harness
    #     settings: Matern-5/2, lengthscale floor above the 1 s spacing).
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=10.0, length_scale_bounds=(3.0, 200.0), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=2)
    gp.fit(t[have_dvl][:, None], v_all[have_dvl])
    gp_mean, gp_std = gp.predict(t[:, None], return_std=True)

    # Effective DVL noise, read back from the fitted WhiteKernel.  normalize_y
    # means noise_level is in normalized-variance units -> rescale by var(y).
    # (This is sensor noise PLUS sub-lengthscale vehicle motion -- the right R
    # for a filter that cannot resolve that motion either.)
    ls_fit = gp.kernel_.k1.k2.length_scale
    var_dvl = gp.kernel_.k2.noise_level * v_all[have_dvl].var()

    # --- 3. Bridge samples inside the gap, spaced ~lengthscale/3 (>= 3 s) to
    #     limit the time-correlated-error problem flagged in POC 3 step 4.
    spacing = int(np.clip(round(ls_fit / 3), 3, 10))
    bridge = gap & (t.astype(int) % spacing == 0)
    has_meas = have_dvl | bridge
    meas = np.where(have_dvl, v_all, gp_mean)
    R_naive = np.full_like(t, var_dvl)
    R_adapt = np.where(have_dvl, var_dvl, gp_std**2)

    xs_n, Ps_n = run_kf(meas, R_naive, has_meas, v_all[0], var_dvl)
    xs_a, Ps_a = run_kf(meas, R_adapt, has_meas, v_all[0], var_dvl)

    # --- 4. Metrics against the noisy references (see module docstring):
    #     velocity ref = hidden DVL inside the gap; position ref = full-DVL
    #     dead-reckoning.  Honesty ratio <= 1 means "never lied".
    pos_ref = np.cumsum(v_all) * dt
    out = {"ls": ls_fit, "sigma_dvl": np.sqrt(var_dvl), "spacing": spacing,
           "gap": gap, "gp_mean": gp_mean, "gp_std": gp_std, "pos_ref": pos_ref}
    for tag, xs, Ps in [("naive", xs_n, Ps_n), ("adaptive", xs_a, Ps_a)]:
        v_err = np.abs(xs[:, 1] - v_all)                       # noisy ref, all t
        v_claim = 2 * np.sqrt(Ps[:, 1, 1] + var_dvl)
        p_err = np.abs(xs[:, 0] - pos_ref)
        p_claim = 2 * np.sqrt(Ps[:, 0, 0])
        out[tag] = {
            "v_rmse_gap": float(np.sqrt(np.mean((xs[gap, 1] - v_all[gap]) ** 2))),
            "v_honesty": float(np.max(v_err[gap] / v_claim[gap])),
            "p_end_err": float(p_err[-1]), "p_peak_err": float(p_err.max()),
            "v_err": v_err, "v_claim": v_claim, "p_err": p_err, "p_claim": p_claim,
        }
    return out


print(f"{'segment':<15} {'gap':>4}  {'vRMSE n/a':>12}  {'honesty n/a':>12}  "
      f"{'peak pos err n/a':>17}   (honesty <=1 = never lied)")
results = {}
for seg, gap_start in SCENARIOS.items():
    for gap_len in GAP_LENGTHS:
        r = run_scenario(gap_start, gap_len)
        results[(seg, gap_len)] = r
        n, a = r["naive"], r["adaptive"]
        print(f"{seg:<15} {gap_len:>3}s  "
              f"{n['v_rmse_gap']*100:5.1f}/{a['v_rmse_gap']*100:5.1f} cm/s  "
              f"{n['v_honesty']:5.1f}/{a['v_honesty']:5.1f} x  "
              f"{n['p_peak_err']:7.2f}/{a['p_peak_err']:7.2f} m")

# --- 5. Figure: the PLOT_GAP stress case.  Row 1 = velocity honesty (as in
#     POC 3), row 2 = position drift, one column per segment.
C_NAIVE, C_ADAPT = "#e34948", "#2a78d6"
fig, axgrid = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
for col, (seg, gap_start) in enumerate(SCENARIOS.items()):
    r = results[(seg, PLOT_GAP)]
    view = (t >= gap_start - 60) & (t < gap_start + PLOT_GAP + 90)
    for row, key in enumerate(["v", "p"]):
        ax = axgrid[row, col]
        ax.axvspan(gap_start, gap_start + PLOT_GAP, color="0.9")
        for tag, c in [("naive", C_NAIVE), ("adaptive", C_ADAPT)]:
            ax.plot(t[view], r[tag][f"{key}_err"][view], color=c,
                    label=f"|error|, {tag} R")
            ax.plot(t[view], r[tag][f"{key}_claim"][view], color=c, ls="--",
                    lw=1, label=f"{tag} claimed 2$\\sigma$")
        ax.grid(alpha=0.25, lw=0.5)
        if row == 0:
            ax.set_title(f"{seg} — {PLOT_GAP} s outage "
                         f"(fitted $\\ell$={r['ls']:.0f} s, bridge every {r['spacing']} s)",
                         fontsize=10)
        else:
            ax.set_xlabel("mission time [s]")
axgrid[0, 0].set_ylabel("velocity error [m/s]")
axgrid[1, 0].set_ylabel("position error [m]")
axgrid[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("POC 3b: adaptive R = GP variance on real Snapir DVL data "
             "(vs. naive fixed R)", fontsize=12)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("poc_03_real_data.png"), dpi=120)
print("saved poc_03_real_data.png")
