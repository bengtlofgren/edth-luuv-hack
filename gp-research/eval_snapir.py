# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""Phase 2a: Snapir sweep -- re-tests POC 3 claims C1/C2 (gp-research/eval/PLAN.md).

Protocol (PLAN.md "Snapir" section): 2001 s @ 1 Hz, 3 axes, gap durations
{10, 30, 60} s sliding every 30 s from t=180 to t=1900 (~57 starts x 3
durations). Bridge suppliers per gap: causal GP (trailing 180 s window),
acausal GP (+-300 s window excluding the gap), ZOH (last pre-gap sample),
linear (endpoint interpolation). KF (surge axis only): causal-naive-R,
causal-adaptive-R, acausal-adaptive-R, ZOH-fixed-R, q = 0.04^2, bridge
injected every 5 s (poc_03_real_data.py's scheme).

Uses gp-research/eval/{data,bridges,seam,metrics}.py as-is (not modified).
`evaluate.sweep_gaps` is not used directly: this sweep needs bridge-level
metrics (mean/var vs the hidden 1 Hz samples) *and* 4 different KF R-configs
sharing the same causal/acausal GP fit, which the generic engine does not
expose without redoing the GP fit per KF config. So GP fits are done once per
(gap, axis) here and reused for both the bridge metrics and (axis 0) the KF
sweep -- this is the "your own loop" option PLAN.md allows.

Run: ~/dvl-gp/.venv/bin/python gp-research/eval_snapir.py   (from repo root)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from math import floor, log10
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

# The GP kernel's bound-hugging optimum warnings (bridges.py's Matern/White
# bounds are tuned per PLAN.md, not left to auto-widen) are expected and
# noisy at ~1000 fits; silence them here rather than in the library.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.data import load_snapir  # noqa: E402
from eval.bridges import GP_LS_BOUNDS_1HZ, gp_bridge, select_train  # noqa: E402
from eval.metrics import coverage_2sigma, drift, honesty, rmse  # noqa: E402
from eval.seam import run_kf  # noqa: E402

OUT_DIR = ROOT / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

AXES = ["x", "y", "z"]          # axis 0 ("x") is the surge/along-track axis
DURATIONS = [10, 30, 60]
GAP_STARTS = list(range(180, 1901, 30))
CAUSAL_WINDOW_S = 180.0
ACAUSAL_WINDOW_S = 300.0
Q_KF = 0.04 ** 2
BRIDGE_SPACING_S = 5.0
QUIET_S = 180.0                 # t < QUIET_S used for the sensor-noise estimate

BRIDGE_SUPPLIERS = ["causal_gp", "acausal_gp", "zoh", "linear"]
KF_CONFIGS = ["causal_naive", "causal_adaptive", "acausal_adaptive", "zoh_fixed"]

COLORS = {
    "causal_gp": "#2a78d6",
    "acausal_gp": "#8a89d0",
    "zoh": "#eb6834",
    "linear": "#1baf7a",
}


# --- small helpers -----------------------------------------------------------

def boxcar_smooth(y, dt, win_s):
    """Centered boxcar of ~win_s seconds (odd length, >=3 samples). Same
    formula as eval.bridges._smooth, reimplemented locally so this script
    does not depend on that module's private helper."""
    n = max(3, int(round(win_s / dt)) | 1)
    return np.convolve(y, np.ones(n) / n, mode="same")


def round_sig(x, sig=6):
    """Round a float to `sig` significant figures (for the JSON dump)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    xf = float(x)
    if xf == 0.0 or not np.isfinite(xf):
        return xf
    d = sig - int(floor(log10(abs(xf)))) - 1
    return round(xf, d)


def round_record(rec):
    return {k: (round_sig(v) if isinstance(v, (int, float, np.floating, np.integer))
                else v) for k, v in rec.items()}


def med_iqr(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    return med, float(q1), float(q3)


def fmt_mi(vals, nd=4):
    m, q1, q3 = med_iqr(vals)
    return f"{m:.{nd}f} [{q1:.{nd}f}, {q3:.{nd}f}]"


# --- 1. Load data, estimate sensor noise --------------------------------------

print("Loading Snapir V_test.npy ...")
data = load_snapir()
t, v = data["t"], data["v"]              # t: (2001,), v: (3, 2001)
N = t.size
assert np.allclose(t, np.arange(N)), "expected t == arange(N) (1 Hz, index==second)"
print(f"  N={N} samples, t in [{t[0]:.0f}, {t[-1]:.0f}] s, axes={v.shape[0]}")

quiet = t < QUIET_S
sigma_hat = np.zeros(3)
for a in range(3):
    y_q = v[a, quiet]
    resid = y_q - boxcar_smooth(y_q, dt=1.0, win_s=5.0)
    sigma_hat[a] = resid.std()
print("Sensor-noise estimate (independent, quiet-segment high-frequency "
      "residual std, t<180s):")
for a in range(3):
    print(f"  axis {AXES[a]}: sigma_hat = {sigma_hat[a]:.5f} m/s "
          f"(var = {sigma_hat[a]**2:.6e})")

# --- 2. Gap grid --------------------------------------------------------------

gaps = []
for start in GAP_STARTS:
    for dur in DURATIONS:
        if start >= t[0] and start + dur <= t[-1]:
            gaps.append((int(start), int(dur)))
print(f"\n{len(gaps)} gaps ({len(GAP_STARTS)} starts x {len(DURATIONS)} durations)")

# full-stream trapezoid dead-reckoning position reference, surge axis
y0 = v[0]
pos_ref_x = np.concatenate([[0.0], np.cumsum(0.5 * (y0[1:] + y0[:-1]) * np.diff(t))])
var_meas_x = np.full(N, sigma_hat[0] ** 2)   # R for real surge measurements

records = []
gap_dyn_score = {}       # (start, dur) -> dynamic-ness score (surge axis)
t0_wall = time.time()

for gi, (start, dur) in enumerate(gaps):
    i0, n_gap = start, dur
    in_gap_slice = slice(i0, i0 + n_gap)
    t_query = t[in_gap_slice]

    # dynamic-ness label (evaluation-only; uses hidden surge data in the gap)
    v_gap_x = v[0, in_gap_slice]
    smoothed_x = boxcar_smooth(v_gap_x, dt=1.0, win_s=5.0)
    dvdt_x = np.gradient(smoothed_x, t_query)
    dyn_score = float(np.std(dvdt_x))
    gap_dyn_score[(start, dur)] = dyn_score

    causal_mask = select_train(t, (start, dur), True, CAUSAL_WINDOW_S)
    acausal_mask = select_train(t, (start, dur), False, ACAUSAL_WINDOW_S)

    for axis in range(3):
        y = v[axis]
        v_true = y[in_gap_slice]

        mean_c, std_c = gp_bridge(t[causal_mask], y[causal_mask], t_query,
                                   True, GP_LS_BOUNDS_1HZ)
        mean_a, std_a = gp_bridge(t[acausal_mask], y[acausal_mask], t_query,
                                   False, GP_LS_BOUNDS_1HZ)

        v_pre, v_post = y[i0 - 1], y[i0 + n_gap]
        mean_zoh = np.full(n_gap, v_pre)
        std_zoh = np.full(n_gap, sigma_hat[axis])
        mean_lin = np.interp(t_query, [t[i0 - 1], t[i0 + n_gap]], [v_pre, v_post])
        std_lin = np.full(n_gap, sigma_hat[axis])

        bridge_out = {
            "causal_gp": (mean_c, std_c),
            "acausal_gp": (mean_a, std_a),
            "zoh": (mean_zoh, std_zoh),
            "linear": (mean_lin, std_lin),
        }
        for name, (mean_s, std_s) in bridge_out.items():
            records.append(round_record(dict(
                kind="bridge", supplier=name, axis=AXES[axis],
                gap_start=start, gap_len=dur,
                bridge_rmse=rmse(mean_s, v_true),
                bridge_coverage=coverage_2sigma(mean_s - v_true, std_s),
                dyn_score=dyn_score,
            )))

        if axis == 0:
            have = np.ones(N, bool)
            have[in_gap_slice] = False
            bridge_times = np.arange(start + BRIDGE_SPACING_S / 2, start + dur,
                                      BRIDGE_SPACING_S)
            is_bridge = np.zeros(N, bool)
            for bt in bridge_times:
                is_bridge[np.argmin(np.abs(t - bt))] = True
            has_meas = have | is_bridge

            def full(mean_gap, var_gap):
                mean_full = np.zeros(N)
                var_full = np.zeros(N)
                mean_full[in_gap_slice] = mean_gap
                var_full[in_gap_slice] = var_gap
                return mean_full, var_full

            mean_c_full, var_c_full = full(mean_c, std_c ** 2)
            mean_a_full, var_a_full = full(mean_a, std_a ** 2)
            mean_zoh_full, _ = full(mean_zoh, np.zeros(n_gap))
            fixed_r_full = np.full(N, sigma_hat[0] ** 2)

            kf_out = {
                "causal_naive": (mean_c_full, fixed_r_full),
                "causal_adaptive": (mean_c_full, var_c_full),
                "acausal_adaptive": (mean_a_full, var_a_full),
                "zoh_fixed": (mean_zoh_full, fixed_r_full),
            }
            for name, (mean_full, var_full) in kf_out.items():
                meas = np.where(have, y, mean_full)
                meas_var = np.where(have, var_meas_x, var_full)
                xs, Ps = run_kf(t, meas, meas_var, has_meas, Q_KF,
                                 float(y[0]), float(sigma_hat[0] ** 2))
                v_err = np.abs(xs[:, 1] - y)
                v_claim = 2 * np.sqrt(Ps[:, 1, 1] + var_meas_x)
                p_err = drift(xs[:, 0], pos_ref_x)
                records.append(round_record(dict(
                    kind="kf", supplier=name, axis="x",
                    gap_start=start, gap_len=dur,
                    kf_vrmse=rmse(xs[in_gap_slice, 1], y[in_gap_slice]),
                    kf_honesty=honesty(v_err[in_gap_slice], v_claim[in_gap_slice]),
                    kf_peak_drift=float(p_err[in_gap_slice].max()),
                    dyn_score=dyn_score,
                )))

    if (gi + 1) % 20 == 0 or gi == len(gaps) - 1:
        elapsed = time.time() - t0_wall
        rate = (gi + 1) / elapsed
        eta = (len(gaps) - gi - 1) / rate if rate > 0 else float("nan")
        print(f"  gap {gi + 1}/{len(gaps)}  (start={start}, dur={dur})  "
              f"elapsed={elapsed / 60:.1f} min  eta={eta / 60:.1f} min")

print(f"\nDone: {len(records)} records, {time.time() - t0_wall:.0f} s total.")

records_path = OUT_DIR / "snapir_records.json"
with open(records_path, "w") as fh:
    json.dump(records, fh, indent=1)
print(f"wrote {records_path}")

# --- 3. Aggregate tables -------------------------------------------------------

median_dyn = float(np.median(list(gap_dyn_score.values())))
print(f"\nMedian dynamic-ness score (surge, quiet/dynamic split threshold): "
      f"{median_dyn:.6g}")


def is_dynamic(rec):
    return gap_dyn_score[(int(rec["gap_start"]), int(rec["gap_len"]))] > median_dyn


bridge_recs = [r for r in records if r["kind"] == "bridge"]
kf_recs = [r for r in records if r["kind"] == "kf"]


def sel(recs, **kw):
    out = recs
    for k, val in kw.items():
        out = [r for r in out if r[k] == val]
    return out


lines = []


def emit(s=""):
    print(s)
    lines.append(s)


emit("# Snapir sweep -- summary\n")
emit(f"N gaps: {len(gaps)} ({len(GAP_STARTS)} starts x {len(DURATIONS)} durations); "
     f"N records: {len(records)}\n")
emit("## Sensor-noise estimate (independent, t<180s high-frequency residual)\n")
emit("| axis | sigma_hat [m/s] | var [(m/s)^2] |")
emit("|---|---|---|")
for a in range(3):
    emit(f"| {AXES[a]} | {sigma_hat[a]:.5f} | {sigma_hat[a] ** 2:.6e} |")
emit(f"\nMedian dynamic-ness score (quiet/dynamic split threshold): {median_dyn:.6g}\n")

# --- Table 1: bridge RMSE median [IQR] by supplier x duration (surge) ----------
emit("## Table 1a: Bridge RMSE median [IQR] by supplier x duration (surge, axis x)\n")
emit("| supplier | " + " | ".join(f"{d}s" for d in DURATIONS) + " | all |")
emit("|---|" + "---|" * (len(DURATIONS) + 1))
for sup in BRIDGE_SUPPLIERS:
    row = [sup]
    for d in DURATIONS:
        vals = [r["bridge_rmse"] for r in sel(bridge_recs, supplier=sup, axis="x", gap_len=d)]
        row.append(fmt_mi(vals))
    vals_all = [r["bridge_rmse"] for r in sel(bridge_recs, supplier=sup, axis="x")]
    row.append(fmt_mi(vals_all))
    emit("| " + " | ".join(row) + " |")

emit("\n## Table 1b: Bridge RMSE median (all durations pooled), compact all-axes\n")
emit("| supplier | " + " | ".join(AXES) + " |")
emit("|---|" + "---|" * len(AXES))
for sup in BRIDGE_SUPPLIERS:
    row = [sup]
    for ax in AXES:
        vals = [r["bridge_rmse"] for r in sel(bridge_recs, supplier=sup, axis=ax)]
        m, _, _ = med_iqr(vals)
        row.append(f"{m:.4f}")
    emit("| " + " | ".join(row) + " |")

# --- Table 2: 2-sigma coverage fraction by supplier x duration, overall + split -
emit("\n## Table 2: Bridge 2-sigma coverage fraction by supplier x duration (surge)\n")
emit("### Overall\n")
emit("| supplier | " + " | ".join(f"{d}s" for d in DURATIONS) + " | all |")
emit("|---|" + "---|" * (len(DURATIONS) + 1))
for sup in BRIDGE_SUPPLIERS:
    row = [sup]
    for d in DURATIONS:
        vals = [r["bridge_coverage"] for r in sel(bridge_recs, supplier=sup, axis="x", gap_len=d)]
        row.append(f"{np.mean(vals):.3f}" if vals else "n/a")
    vals_all = [r["bridge_coverage"] for r in sel(bridge_recs, supplier=sup, axis="x")]
    row.append(f"{np.mean(vals_all):.3f}")
    emit("| " + " | ".join(row) + " |")

emit("\n### Quiet vs dynamic split (median split on gap dynamic-ness score, C2)\n")
emit("| supplier | quiet (n=" +
     str(len(set((r["gap_start"], r["gap_len"]) for r in sel(bridge_recs, axis="x") if not is_dynamic(r)))) +
     ") | dynamic (n=" +
     str(len(set((r["gap_start"], r["gap_len"]) for r in sel(bridge_recs, axis="x") if is_dynamic(r)))) +
     ") |")
emit("|---|---|---|")
for sup in BRIDGE_SUPPLIERS:
    recs_x = sel(bridge_recs, supplier=sup, axis="x")
    quiet_vals = [r["bridge_coverage"] for r in recs_x if not is_dynamic(r)]
    dyn_vals = [r["bridge_coverage"] for r in recs_x if is_dynamic(r)]
    emit(f"| {sup} | {np.mean(quiet_vals):.3f} | {np.mean(dyn_vals):.3f} |")

# --- Table 3: KF median drift + honesty by config x duration, overall + split --
emit("\n## Table 3a: KF median peak drift [IQR] (m) + median honesty by config x duration (surge)\n")
emit("| config | " + " | ".join(f"{d}s drift" for d in DURATIONS) +
     " | " + " | ".join(f"{d}s honesty" for d in DURATIONS) + " |")
emit("|---|" + "---|" * (2 * len(DURATIONS)))
for cfg in KF_CONFIGS:
    drift_cells = []
    hon_cells = []
    for d in DURATIONS:
        rs = sel(kf_recs, supplier=cfg, gap_len=d)
        drift_cells.append(fmt_mi([r["kf_peak_drift"] for r in rs]))
        hon_vals = [r["kf_honesty"] for r in rs]
        m, _, _ = med_iqr(hon_vals)
        hon_cells.append(f"{m:.3f}")
    emit(f"| {cfg} | " + " | ".join(drift_cells) + " | " + " | ".join(hon_cells) + " |")

emit("\n## Table 3b: KF median peak drift + honesty, quiet vs dynamic split\n")
emit("| config | quiet drift | dynamic drift | quiet honesty | dynamic honesty |")
emit("|---|---|---|---|---|")
for cfg in KF_CONFIGS:
    rs = sel(kf_recs, supplier=cfg)
    quiet_d = [r["kf_peak_drift"] for r in rs if not is_dynamic(r)]
    dyn_d = [r["kf_peak_drift"] for r in rs if is_dynamic(r)]
    quiet_h = [r["kf_honesty"] for r in rs if not is_dynamic(r)]
    dyn_h = [r["kf_honesty"] for r in rs if is_dynamic(r)]
    mqd, _, _ = med_iqr(quiet_d); mdd, _, _ = med_iqr(dyn_d)
    mqh, _, _ = med_iqr(quiet_h); mdh, _, _ = med_iqr(dyn_h)
    emit(f"| {cfg} | {mqd:.4f} | {mdd:.4f} | {mqh:.3f} | {mdh:.3f} |")

# C1: paired per-gap drift deltas, causal_adaptive vs causal_naive (causal setting)
emit("\n## Table 3c: C1 -- paired peak-drift deltas, causal_naive - causal_adaptive "
     "(positive = adaptive wins) (m)\n")


def paired_deltas(cfg_a, cfg_b, subset_pred=None):
    """delta = drift(cfg_a) - drift(cfg_b) per (gap_start, gap_len); positive
    means cfg_b (usually adaptive) has lower drift, i.e. cfg_b wins."""
    by_key_a = {(r["gap_start"], r["gap_len"]): r["kf_peak_drift"]
                for r in sel(kf_recs, supplier=cfg_a) if subset_pred is None or subset_pred(r)}
    by_key_b = {(r["gap_start"], r["gap_len"]): r["kf_peak_drift"]
                for r in sel(kf_recs, supplier=cfg_b) if subset_pred is None or subset_pred(r)}
    keys = sorted(set(by_key_a) & set(by_key_b))
    return np.array([by_key_a[k] - by_key_b[k] for k in keys])


emit("| split | " + " | ".join(f"{d}s median delta" for d in DURATIONS) +
     " | all median delta | all win rate (adaptive<naive) |")
emit("|---|" + "---|" * (len(DURATIONS) + 2))
for label, pred in [("overall", None), ("quiet", lambda r: not is_dynamic(r)),
                     ("dynamic", lambda r: is_dynamic(r))]:
    row = [label]
    for d in DURATIONS:
        dd = paired_deltas("causal_naive", "causal_adaptive",
                            lambda r, d=d, pred=pred: r["gap_len"] == d and (pred is None or pred(r)))
        row.append(f"{np.median(dd):.4f}" if dd.size else "n/a")
    dd_all = paired_deltas("causal_naive", "causal_adaptive", pred)
    row.append(f"{np.median(dd_all):.4f}" if dd_all.size else "n/a")
    win_rate = float(np.mean(dd_all > 0)) if dd_all.size else float("nan")
    row.append(f"{win_rate:.3f}")
    emit("| " + " | ".join(row) + " |")

# --- Table 4: win-rate matrix, bridge RMSE vs causal_gp, paired ----------------
emit("\n## Table 4: Win-rate matrix -- fraction of gaps where supplier beats "
     "causal_gp on bridge RMSE (paired, surge axis)\n")


def rmse_win_rate(sup, d=None):
    by_key_c = {(r["gap_start"], r["gap_len"]): r["bridge_rmse"]
                for r in sel(bridge_recs, supplier="causal_gp", axis="x")
                if d is None or r["gap_len"] == d}
    by_key_s = {(r["gap_start"], r["gap_len"]): r["bridge_rmse"]
                for r in sel(bridge_recs, supplier=sup, axis="x")
                if d is None or r["gap_len"] == d}
    keys = sorted(set(by_key_c) & set(by_key_s))
    wins = [by_key_s[k] < by_key_c[k] for k in keys]
    return float(np.mean(wins)) if wins else float("nan")


emit("| supplier | " + " | ".join(f"{d}s" for d in DURATIONS) + " | all |")
emit("|---|" + "---|" * (len(DURATIONS) + 1))
for sup in ["acausal_gp", "zoh", "linear"]:
    row = [sup]
    for d in DURATIONS:
        row.append(f"{rmse_win_rate(sup, d):.3f}")
    row.append(f"{rmse_win_rate(sup):.3f}")
    emit("| " + " | ".join(row) + " |")

summary_path = OUT_DIR / "snapir_summary.md"
with open(summary_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nwrote {summary_path}")

# --- 4. Figure -----------------------------------------------------------------

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) bridge RMSE distributions by supplier, 30 s gaps, surge axis
box_data = [[r["bridge_rmse"] for r in sel(bridge_recs, supplier=sup, axis="x", gap_len=30)]
            for sup in BRIDGE_SUPPLIERS]
bp = ax[0].boxplot(box_data, tick_labels=BRIDGE_SUPPLIERS, patch_artist=True, showfliers=False)
for patch, sup in zip(bp["boxes"], BRIDGE_SUPPLIERS):
    patch.set_facecolor(COLORS[sup])
    patch.set_alpha(0.7)
ax[0].set_ylabel("bridge RMSE [m/s]")
ax[0].set_title("(a) Bridge RMSE by supplier, 30 s gaps (surge)")
ax[0].tick_params(axis="x", rotation=20)
ax[0].grid(alpha=0.25, lw=0.5)

# (b) coverage vs dynamic-ness scatter for causal GP (surge, all durations)
causal_x = sel(bridge_recs, supplier="causal_gp", axis="x")
dscore = [r["dyn_score"] for r in causal_x]
cov = [r["bridge_coverage"] for r in causal_x]
ax[1].scatter(dscore, cov, s=14, color=COLORS["causal_gp"], alpha=0.6)
ax[1].axhline(0.954, color="0.4", ls=":", lw=1, label="nominal 95.4%")
ax[1].axvline(median_dyn, color="0.4", ls="--", lw=1, label="median split")
ax[1].set_xlabel("gap dynamic-ness score [m/s^2]")
ax[1].set_ylabel("bridge 2-sigma coverage fraction")
ax[1].set_title("(b) Causal-GP coverage vs dynamic-ness")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.25, lw=0.5)

# (c) paired adaptive-vs-naive drift deltas histogram (causal setting)
deltas = paired_deltas("causal_naive", "causal_adaptive")
ax[2].hist(deltas, bins=30, color=COLORS["causal_gp"], alpha=0.8)
ax[2].axvline(0.0, color="k", lw=1)
ax[2].axvline(float(np.median(deltas)), color="0.2", ls="--", lw=1.2,
              label=f"median={np.median(deltas):.3f} m")
ax[2].set_xlabel("peak-drift delta: naive - adaptive [m]  (>0 = adaptive wins)")
ax[2].set_ylabel("count (gaps)")
ax[2].set_title("(c) Paired adaptive-vs-naive drift deltas (causal GP)")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25, lw=0.5)

fig.suptitle("Snapir sweep: bridge accuracy, causal-GP honesty, adaptive-R drift "
             f"({len(gaps)} gaps x {len(DURATIONS)} durations)", fontsize=12)
fig.tight_layout()
fig_path = OUT_DIR / "snapir_eval.png"
fig.savefig(fig_path, dpi=130)
print(f"wrote {fig_path}")
