# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "rosbags"]
# ///
"""POC 7 -- control-conditioned dynamics GP (PILCO-style) as an outage bridge.

The ensemble re-evaluation (../eval/RESULTS.md) showed the time-indexed
causal GP is structurally overconfident during maneuvers: regressing v(t)
on time alone, it cannot see an acceleration that starts inside the gap.
The robotics answer (Deisenroth & Rasmussen, PILCO) is to regress the
*dynamics* on state and action instead:

    a = dv/dt ~ f(v, u_thrust),

because the commanded thrust u is still known during a DVL outage, and a
process that is nonstationary in time is close to stationary in (v, u).
Semi-parametric: a ridge-fit linear mean (thrust gains + damping) carries
the physics; a GP learns the residual; the outage is bridged by rolling
the model forward from the last good fix with the logged commands,
accumulating the model's own predictive variance each step.

This demo runs both bridges on one maneuver-containing gap of a SOLAQUA
bag (auto-downloaded) and plots them against the hidden DVL truth,
alongside the thrust commands that make the maneuver visible to the
dynamics model. The ensemble verdict on this supplier across all bags/gaps
lives in ../eval/RESULTS.md ("dynamics-GP" section); this script is the
single-gap illustration, not the evidence.

Run from gp-research/:  uv run poc/poc_07_dynamics_gp.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.bridges import (  # noqa: E402
    GP_LS_BOUNDS_10HZ,
    dynamics_bridge,
    dynamics_calibrate,
    gp_bridge,
    select_train,
)
from eval.data import load_solaqua_bag  # noqa: E402
from eval.metrics import rmse  # noqa: E402

warnings.filterwarnings("ignore")

BAG = "2024-08-20_15-09-34_data.bag"     # dynamic net-following run, thrust logged
GAP_START, GAP_LEN = 45.0, 20.0
CAL_WINDOW_S = 60.0                      # causal GP trailing window (eval protocol)

C_GP, C_DYN, C_MEAS = "#2a78d6", "#d9962e", "#8a897f"

d = load_solaqua_bag(BAG)
a50, thr = d["a50"], d["thrust"]
t, v, fom = a50["t"], a50["v"], a50["fom"]
gap_end = GAP_START + GAP_LEN
in_gap = (t >= GAP_START) & (t < gap_end)
t_hid, v_hid = t[in_gap], v[in_gap]

i0 = np.flatnonzero(t < GAP_START)[-1]
t0, v0 = t[i0], v[i0]
var0 = max(fom[i0] ** 2, 1e-4)

# time-indexed causal GP (trailing window), surge axis
mask_c = select_train(t, (GAP_START, GAP_LEN), True, CAL_WINDOW_S)
gp_mean, gp_std = gp_bridge(t[mask_c], v[mask_c, 0], t_hid, True,
                             length_scale_bounds=GP_LS_BOUNDS_10HZ)

# control-conditioned dynamics GP (all pre-gap data -- causal)
pre = t < GAP_START
cal = dynamics_calibrate(t[pre], v[pre], thr["t"], thr["u"])
t_dyn, dyn_mean3, dyn_std3 = dynamics_bridge(cal, v0, t0, thr["t"], thr["u"],
                                              gap_end, var0=var0)
dyn_mean = np.interp(t_hid, t_dyn, dyn_mean3[:, 0])
dyn_std = np.interp(t_hid, t_dyn, dyn_std3[:, 0])

r_gp, r_dyn = rmse(gp_mean, v_hid[:, 0]), rmse(dyn_mean, v_hid[:, 0])
r_zoh = rmse(np.full(t_hid.size, v0[0]), v_hid[:, 0])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                height_ratios=[2.2, 1])

ax1.plot(t[~in_gap], v[~in_gap, 0], ".", ms=3, color=C_MEAS, label="DVL surge (valid)")
ax1.plot(t_hid, v_hid[:, 0], ".", ms=4, color="k", label="hidden truth")
ax1.plot(t_hid, gp_mean, color=C_GP, lw=1.6, label=f"time GP ({r_gp*100:.1f} cm/s)")
ax1.fill_between(t_hid, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std,
                  color=C_GP, alpha=0.18)
ax1.plot(t_hid, dyn_mean, color=C_DYN, lw=1.6, label=f"dynamics GP ({r_dyn*100:.1f} cm/s)")
ax1.fill_between(t_hid, dyn_mean - 2 * dyn_std, dyn_mean + 2 * dyn_std,
                  color=C_DYN, alpha=0.18)
ax1.axvspan(GAP_START, gap_end, color="0.5", alpha=0.08)
ax1.set_xlim(GAP_START - 25, gap_end + 10)
ax1.set_ylabel("surge velocity [m/s]")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.25, lw=0.5)
ax1.set_title(f"{BAG}: {GAP_LEN:.0f} s outage @ t={GAP_START:.0f} s -- "
              "time-indexed GP vs control-conditioned dynamics GP")

for i, lbl in enumerate(["ch2", "ch3", "ch4", "ch5"]):
    ax2.plot(thr["t"], thr["u"][:, i], lw=1, label=lbl)
ax2.axvspan(GAP_START, gap_end, color="0.5", alpha=0.08)
ax2.set_xlabel("time [s]")
ax2.set_ylabel("commanded thrust\n(normalized PWM)")
ax2.legend(fontsize=7, ncol=4)
ax2.grid(alpha=0.25, lw=0.5)

fig.tight_layout()
out = Path(__file__).resolve().parent / "poc_07_dynamics_gp.png"
fig.savefig(out, dpi=130)

print(f"gap RMSE [cm/s]: time-GP {r_gp*100:.2f} | dynamics-GP {r_dyn*100:.2f} | "
      f"ZOH {r_zoh*100:.2f}; linear-mean R^2 per axis "
      f"{np.round(cal.lin_r2, 3).tolist()} -- the commands are visible to the "
      "dynamics model during the gap, but how much they explain on this "
      "tethered ROV is an empirical question; see ../eval/RESULTS.md.")
print(f"wrote {out}")
