# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""Dagon basin sweep -- does the control-conditioned dynamics GP win when
commanded thrust genuinely explains the motion?

The SOLAQUA sweep refuted the dynamics-GP supplier on a tethered ROV where
commands explained only ~6% of surge acceleration (see eval/RESULTS.md).
The DFKI Dagon dataset is the other side of that natural experiment: an
untethered AUV in a saltwater basin, driven by sinusoidal thruster
system-ID excitation (Wehbe et al. ICRA'19 configuration 1; see
eval/data.py::load_dagon for provenance and the empirical 4 Hz sample-rate
determination). If control-conditioned bridging is ever going to beat the
time-indexed GP, it is here.

Protocol (mirrors eval_solaqua.py): sliding-gap ensemble, durations
{10, 20} s, starts every 60 s from t=120 s (the dynamics GP trains causally
on ALL pre-gap data -- it assumes stationarity in (v, u), not time -- while
the time-GP trains on a trailing 60 s window per the established protocol).
Suppliers: causal time-GP (gp_bridge), dyn_gp (dynamics_calibrate +
dynamics_bridge rolled forward with the commanded thrust), ZOH. Axes are
surge/sway [m/s] and yaw RATE [rad/s] -- NEVER pooled across units; every
aggregate below is per-axis.

Extra diagnostic beyond the SOLAQUA sweep: per gap, a one-step acceleration
prediction test on the hidden in-gap samples quantifies what the GP residual
adds on top of the ridge linear mean (residual RMS with vs without the GP
correction, at the true in-gap states -- evaluation-only, no leakage into
any supplier).

Run: ~/dvl-gp/.venv/bin/python gp-research/eval_dagon.py   (from repo root)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.bridges import (  # noqa: E402
    GP_LS_BOUNDS_10HZ,
    dynamics_bridge,
    dynamics_calibrate,
    gp_bridge,
    select_train,
)
from eval.data import DAGON_DT_S, load_dagon  # noqa: E402
from eval.metrics import coverage_2sigma, rmse  # noqa: E402

OUT_DIR = ROOT / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

AXIS_LABELS = ["surge", "sway", "yaw_rate"]
AXIS_UNITS = ["m/s", "m/s", "rad/s"]
# display scaling per axis: cm/s for the linear axes, mrad/s for yaw rate
AXIS_SCALE = [100.0, 100.0, 1000.0]
AXIS_DISP = ["cm/s", "cm/s", "mrad/s"]

GAP_LENGTHS = (10.0, 20.0)
CAL_WINDOW_S = 60.0        # trailing window for the causal time-GP
FIRST_START = 120.0        # earliest gap start (dyn_gp gets >=120 s history)
START_STRIDE = 60.0
TAIL_MARGIN = 5.0
BAND_S = 1.0               # smoothing band for the one-step diagnostic
                           # (matches dynamics_calibrate's default band_s)

C_GP, C_DYN, C_ZOH, C_MEAS = "#2a78d6", "#d9962e", "#eb6834", "#8a897f"
SUPPLIER_COLORS = {"causal_gp": C_GP, "dyn_gp": C_DYN, "zoh": C_ZOH}
SUPPLIERS_ORDER = ["causal_gp", "dyn_gp", "zoh"]


def _smooth(y, dt, win_s):
    n = max(3, int(round(win_s / dt)) | 1)
    return np.convolve(y, np.ones(n) / n, mode="same")


def dynamic_ness(t_hid, v_surge_hid):
    """std of 2s-smoothed surge dv/dt inside the gap (evaluation label only,
    not used by any supplier) -- same recipe as eval_solaqua.py."""
    if t_hid.size < 4:
        return 0.0
    dt = float(np.median(np.diff(t_hid)))
    v_s = _smooth(v_surge_hid, dt, 2.0)
    dv_dt = np.gradient(v_s, t_hid)
    return float(np.std(dv_dt))


def median_iqr(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(arr)), float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)))


def fmt_mi(vals, ax):
    m, lo, hi = median_iqr(vals)
    if np.isnan(m):
        return "n/a"
    s = AXIS_SCALE[ax]
    return f"{m*s:6.2f} [{lo*s:5.2f}, {hi*s:5.2f}] {AXIS_DISP[ax]} (n={len(vals)})"


def dyn_lin_gp_accel(dyncal, v_states, u_states):
    """Evaluate the dynamics model's acceleration prediction at given
    (v, u) states: returns (a_lin, a_lin_plus_gp), each (k, 3). Uses only
    public DynCalib fields; mirrors dynamics_bridge's feature construction."""
    X = np.column_stack([v_states, u_states])
    Phi = np.column_stack([X, np.ones(X.shape[0])])
    a_lin = Phi @ dyncal.W.T
    Xs = ((X - dyncal.mu) / dyncal.sd)[:, dyncal.keep]
    a_gp = a_lin.copy()
    for ax in range(3):
        a_gp[:, ax] += dyncal.gps[ax].predict(Xs)
    return a_lin, a_gp


def process_gap(d, gap_start, gap_len, sigma_hat, v_smooth, u_smooth, a_ref):
    t, v, u_cmd = d["t"], d["v"], d["u_cmd"]
    gap_end = gap_start + gap_len
    in_gap = (t >= gap_start) & (t < gap_end)
    if not in_gap.any():
        return []
    t_hid, v_hid = t[in_gap], v[in_gap]

    i_last_pre = np.flatnonzero(t < gap_start)[-1]
    t0, v0 = t[i_last_pre], v[i_last_pre]
    var0 = sigma_hat ** 2

    dyn = dynamic_ness(t_hid, v_hid[:, 0])

    # --- causal time-GP, trailing CAL_WINDOW_S window, per axis.
    mask_c = select_train(t, (gap_start, gap_len), True, CAL_WINDOW_S)
    gp_mean = np.zeros((t_hid.size, 3)); gp_var = np.zeros_like(gp_mean)
    for ax in range(3):
        m, s = gp_bridge(t[mask_c], v[mask_c, ax], t_hid, True,
                          length_scale_bounds=GP_LS_BOUNDS_10HZ)
        gp_mean[:, ax], gp_var[:, ax] = m, s ** 2

    # --- dynamics GP: trains causally on ALL pre-gap data, rolls the gap
    # forward with the commanded thrust (known during a real outage).
    pre = t < gap_start
    dyncal = dynamics_calibrate(t[pre], v[pre], t[pre], u_cmd[pre])
    t_dyn, mean_dyn, std_dyn = dynamics_bridge(
        dyncal, v0, t0, t, u_cmd, gap_end, var0=var0)
    dyn_mean = np.column_stack(
        [np.interp(t_hid, t_dyn, mean_dyn[:, ax]) for ax in range(3)])
    dyn_var = np.column_stack(
        [np.interp(t_hid, t_dyn, std_dyn[:, ax]) for ax in range(3)]) ** 2

    # --- ZOH: hold last pre-gap sample, sensor-noise-floor variance.
    zoh_mean = np.tile(v0, (t_hid.size, 1))
    zoh_var = np.tile(sigma_hat ** 2, (t_hid.size, 1))

    # --- one-step diagnostic: at the TRUE hidden states, does the GP
    # residual improve the ridge mean's acceleration prediction?
    # (evaluation-only: uses in-gap truth, mirrors calibration's smoothing.)
    a_lin, a_gp = dyn_lin_gp_accel(dyncal, v_smooth[in_gap], u_smooth[in_gap])
    onestep_lin = np.sqrt(np.mean((a_ref[in_gap] - a_lin) ** 2, axis=0))
    onestep_gp = np.sqrt(np.mean((a_ref[in_gap] - a_gp) ** 2, axis=0))

    suppliers = {
        "causal_gp": (gp_mean, gp_var),
        "dyn_gp": (dyn_mean, dyn_var),
        "zoh": (zoh_mean, zoh_var),
    }
    recs = []
    for sup, (mean, var) in suppliers.items():
        for ax in range(3):
            err = np.abs(mean[:, ax] - v_hid[:, ax])
            rec = dict(
                supplier=sup, axis=AXIS_LABELS[ax],
                gap_start=float(gap_start), gap_len=float(gap_len),
                bridge_rmse=rmse(mean[:, ax], v_hid[:, ax]),
                bridge_coverage=coverage_2sigma(
                    err, np.sqrt(np.maximum(var[:, ax], 1e-12))),
                dynamic_ness=dyn, n_hidden=int(t_hid.size),
                dyn_lin_r2=None, onestep_rms_lin=None, onestep_rms_gp=None,
            )
            if sup == "dyn_gp":
                rec.update(dyn_lin_r2=float(dyncal.lin_r2[ax]),
                           onestep_rms_lin=float(onestep_lin[ax]),
                           onestep_rms_gp=float(onestep_gp[ax]))
            recs.append(rec)
    return recs


def main():
    t_run0 = time.time()
    d = load_dagon()
    t, v, u_cmd = d["t"], d["v"], d["u_cmd"]
    t_end = float(t[-1])
    dt = DAGON_DT_S
    print(f"Dagon: n={t.size}, duration={t_end:.1f}s, dt={dt}s")

    # sensor-noise floor per axis: high-frequency residual around a 5 s
    # boxcar over the full record (same recipe as eval_snapir.py's
    # sigma_hat; there is no quiet segment here -- excitation is continuous,
    # so this slightly overestimates the floor. Used for ZOH's claimed
    # variance and the rollout seed variance only).
    sigma_hat = np.array([np.std(v[:, ax] - _smooth(v[:, ax], dt, 5.0))
                          for ax in range(3)])
    print("sigma_hat per axis:", np.round(sigma_hat, 4).tolist())

    # smoothed states + reference acceleration for the one-step diagnostic
    # (band-matched to dynamics_calibrate's band_s=1.0 preprocessing).
    v_smooth = np.column_stack([_smooth(v[:, ax], dt, BAND_S) for ax in range(3)])
    u_smooth = np.column_stack([_smooth(u_cmd[:, ax], dt, BAND_S) for ax in range(3)])
    a_ref = np.gradient(v_smooth, t, axis=0)

    starts_by_len = {}
    for gl in GAP_LENGTHS:
        starts, s = [], FIRST_START
        while s + gl <= t_end - TAIL_MARGIN:
            starts.append(s)
            s += START_STRIDE
        starts_by_len[gl] = starts
    n_gaps = sum(len(s) for s in starts_by_len.values())
    print(f"gaps: {' '.join(f'{int(gl)}s x {len(s)}' for gl, s in starts_by_len.items())}"
          f" = {n_gaps} total")

    records = []
    done = 0
    for gl in GAP_LENGTHS:
        for gs in starts_by_len[gl]:
            t_g0 = time.time()
            recs = process_gap(d, gs, gl, sigma_hat, v_smooth, u_smooth, a_ref)
            records.extend(recs)
            done += 1
            if done % 10 == 0 or done == n_gaps:
                print(f"  gap {done}/{n_gaps} (start={gs:.0f} len={gl:.0f}) "
                      f"{time.time()-t_g0:.1f}s/gap, elapsed "
                      f"{(time.time()-t_run0)/60:.1f} min", flush=True)

    print(f"\nTotal records: {len(records)} "
          f"(runtime {(time.time()-t_run0)/60:.1f} min)")

    meta = dict(n_samples=int(t.size), duration_s=t_end, dt_s=dt,
                sigma_hat=sigma_hat.tolist(),
                starts={f"{int(gl)}": s for gl, s in starts_by_len.items()},
                cal_window_s=CAL_WINDOW_S, first_start=FIRST_START,
                start_stride=START_STRIDE)
    with open(OUT_DIR / "dagon_records.json", "w") as fh:
        json.dump(dict(records=records, meta=meta), fh, indent=1)
    print(f"wrote {OUT_DIR / 'dagon_records.json'}")

    summary = build_summary(records, meta)
    (OUT_DIR / "dagon_summary.md").write_text(summary)
    print(f"wrote {OUT_DIR / 'dagon_summary.md'}")
    print("\n" + summary)

    make_figure(records, d)
    print(f"wrote {OUT_DIR / 'dagon_eval.png'}")


# --- Aggregation --------------------------------------------------------------

def filt(records, **kw):
    out = records
    for k, val in kw.items():
        out = [r for r in out if r[k] == val]
    return out


def gap_key(r):
    return (r["gap_start"], r["gap_len"])


def build_summary(records, meta):
    lines = []
    lines.append("# Dagon basin sweep summary\n")
    lines.append(f"DFKI Dagon AUV, saltwater basin, sinusoidal thruster "
                 f"system-ID excitation. {meta['n_samples']} samples, "
                 f"{meta['duration_s']:.0f} s at dt={meta['dt_s']} s (4 Hz, "
                 "empirically determined -- see eval/data.py::load_dagon). "
                 "Axes: surge/sway [m/s], yaw RATE [rad/s]; axes are never "
                 "pooled.\n")
    n10, n20 = len(meta["starts"]["10"]), len(meta["starts"]["20"])
    lines.append(f"Gaps: {n10} x 10 s + {n20} x 20 s, starts every "
                 f"{meta['start_stride']:.0f} s from t={meta['first_start']:.0f} s. "
                 f"Causal time-GP trains on a trailing {meta['cal_window_s']:.0f} s "
                 "window; dyn_gp trains on ALL pre-gap data; ZOH holds the "
                 "last pre-gap sample.\n")

    lines.append("## 1. Bridge RMSE median [IQR] by supplier x duration, per axis\n")
    for ax, label in enumerate(AXIS_LABELS):
        lines.append(f"### {label} [{AXIS_UNITS[ax]}]\n")
        for gl in GAP_LENGTHS:
            lines.append(f"- {gl:.0f}s gaps:")
            for sup in SUPPLIERS_ORDER:
                vals = [r["bridge_rmse"] for r in
                        filt(records, supplier=sup, axis=label, gap_len=gl)]
                lines.append(f"    - {sup:10s}: {fmt_mi(vals, ax)}")
        lines.append("")

    lines.append("## 2. Paired deltas -- dyn_gp minus causal time-GP and minus "
                 "ZOH, per axis\n")
    lines.append("Negative delta = dyn_gp wins. Win rate = fraction of gaps "
                 "where dyn_gp's RMSE is lower (paired per gap).\n")
    surge_recs = filt(records, axis="surge")
    dyn_vals = sorted({gap_key(r): r["dynamic_ness"] for r in surge_recs}.values())
    dyn_med = float(np.median(dyn_vals)) if dyn_vals else float("nan")
    lines.append(f"Median gap dynamic-ness (std of 2s-smoothed surge dv/dt in "
                 f"gap): {dyn_med:.4f} m/s^2 -- split point below.\n")

    for ax, label in enumerate(AXIS_LABELS):
        by_gap = {}
        for r in filt(records, axis=label):
            by_gap.setdefault(gap_key(r), {})[r["supplier"]] = r
        lines.append(f"### {label} [{AXIS_UNITS[ax]}]\n")
        for split, cond in [("all gaps", lambda dd: True),
                            ("dynamic half (>= median)", lambda dd: dd >= dyn_med),
                            ("calm half (< median)", lambda dd: dd < dyn_med)]:
            pairs_dg, pairs_dz = [], []
            for k, sups in by_gap.items():
                if not all(s in sups for s in SUPPLIERS_ORDER):
                    continue
                if not cond(sups["dyn_gp"]["dynamic_ness"]):
                    continue
                pairs_dg.append(sups["dyn_gp"]["bridge_rmse"]
                                - sups["causal_gp"]["bridge_rmse"])
                pairs_dz.append(sups["dyn_gp"]["bridge_rmse"]
                                - sups["zoh"]["bridge_rmse"])
            if not pairs_dg:
                lines.append(f"- {split}: n=0")
                continue
            w_g = float(np.mean(np.array(pairs_dg) < 0))
            w_z = float(np.mean(np.array(pairs_dz) < 0))
            lines.append(f"- {split} (n={len(pairs_dg)}):")
            lines.append(f"    - dyn_gp minus time-GP: {fmt_mi(pairs_dg, ax)} "
                         f"(dyn_gp wins {w_g*100:.0f}%)")
            lines.append(f"    - dyn_gp minus ZOH:     {fmt_mi(pairs_dz, ax)} "
                         f"(dyn_gp wins {w_z*100:.0f}%)")
        lines.append("")

    lines.append("## 3. Bridge 2-sigma coverage (mean over gaps), per axis\n")
    lines.append("| supplier | " + " | ".join(AXIS_LABELS) + " |")
    lines.append("|---|---|---|---|")
    for sup in SUPPLIERS_ORDER:
        row = [sup]
        for label in AXIS_LABELS:
            vals = [r["bridge_coverage"] for r in filt(records, supplier=sup,
                                                        axis=label)]
            row.append(f"{np.mean(vals)*100:.0f}%" if vals else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 4. Observability diagnostic -- linear-mean R^2 per axis\n")
    lines.append("R^2 of the ridge thrust-gain/damping mean alone on pre-gap "
                 "acceleration (dyn_gp's DynCalib.lin_r2; median [IQR] over "
                 "gaps -- each gap refits on its own causal history):\n")
    for label in AXIS_LABELS:
        vals = [r["dyn_lin_r2"] for r in filt(records, supplier="dyn_gp",
                                               axis=label)]
        m, lo, hi = median_iqr(vals)
        lines.append(f"- {label:9s}: R^2 = {m:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"(n={len(vals)})")
    lines.append("")

    lines.append("## 5. What the GP residual adds on top of the ridge mean\n")
    lines.append("One-step acceleration prediction at the TRUE hidden in-gap "
                 "states (evaluation-only diagnostic): residual RMS of the "
                 "ridge mean alone vs ridge+GP, median over gaps. Ratio < 1 "
                 "means the GP correction helps out-of-sample.\n")
    for ax, label in enumerate(AXIS_LABELS):
        recs = filt(records, supplier="dyn_gp", axis=label)
        lin = [r["onestep_rms_lin"] for r in recs]
        gpv = [r["onestep_rms_gp"] for r in recs]
        ratios = [g / l for g, l in zip(gpv, lin) if l > 0]
        ml, _, _ = median_iqr(lin)
        mg, _, _ = median_iqr(gpv)
        mr, lor, hir = median_iqr(ratios)
        unit = "m/s^2" if ax < 2 else "rad/s^2"
        lines.append(f"- {label:9s}: ridge-only {ml:.4f} {unit}, ridge+GP "
                     f"{mg:.4f} {unit}; ratio {mr:.3f} [{lor:.3f}, {hir:.3f}] "
                     f"(n={len(ratios)})")
    lines.append("")

    return "\n".join(lines)


def make_figure(records, d):
    t, v = d["t"], d["v"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (a-c) RMSE distributions by supplier, 20 s gaps, per axis.
    for ax_i, label in enumerate(AXIS_LABELS):
        ax = axes[0, ax_i]
        data, labels, colors = [], [], []
        for sup in SUPPLIERS_ORDER:
            vals = [r["bridge_rmse"] * AXIS_SCALE[ax_i] for r in
                    filt(records, supplier=sup, axis=label, gap_len=20.0)]
            if vals:
                data.append(vals); labels.append(sup)
                colors.append(SUPPLIER_COLORS[sup])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        showfliers=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel(f"bridge RMSE [{AXIS_DISP[ax_i]}]")
        ax.set_title(f"({'abc'[ax_i]}) {label} RMSE, 20 s gaps")
        ax.grid(alpha=0.25, lw=0.5)

    # (d) example gap trace, surge: pick the 20 s gap whose dynamic-ness is
    # closest to the ensemble median.
    surge20 = filt(records, supplier="dyn_gp", axis="surge", gap_len=20.0)
    med_dyn = float(np.median([r["dynamic_ness"] for r in surge20]))
    ex = min(surge20, key=lambda r: abs(r["dynamic_ness"] - med_dyn))
    gs, gl = ex["gap_start"], ex["gap_len"]
    ax = axes[1, 0]
    dd = d
    sigma_hat = np.array([np.std(v[:, k] - _smooth(v[:, k], DAGON_DT_S, 5.0))
                          for k in range(3)])
    # recompute supplier tracks for plotting (cheap: one gap)
    in_gap = (t >= gs) & (t < gs + gl)
    t_hid = t[in_gap]
    view = (t >= gs - 30) & (t < gs + gl + 10)
    ax.plot(t[view], v[view, 0], ".", ms=3, color=C_MEAS, label="measured")
    mask_c = select_train(t, (gs, gl), True, CAL_WINDOW_S)
    m_gp, s_gp = gp_bridge(t[mask_c], v[mask_c, 0], t_hid, True,
                            length_scale_bounds=GP_LS_BOUNDS_10HZ)
    pre = t < gs
    i0 = np.flatnonzero(pre)[-1]
    dyncal = dynamics_calibrate(t[pre], v[pre], t[pre], dd["u_cmd"][pre])
    t_dyn, m_dyn, s_dyn = dynamics_bridge(dyncal, v[i0], t[i0], t,
                                           dd["u_cmd"], gs + gl,
                                           var0=sigma_hat ** 2)
    ax.plot(t_hid, m_gp, color=C_GP, lw=1.5, label="time-GP")
    ax.fill_between(t_hid, m_gp - 2 * s_gp, m_gp + 2 * s_gp, color=C_GP,
                    alpha=0.15)
    ax.plot(t_dyn, m_dyn[:, 0], color=C_DYN, lw=1.5, label="dyn-GP")
    ax.fill_between(t_dyn, m_dyn[:, 0] - 2 * s_dyn[:, 0],
                    m_dyn[:, 0] + 2 * s_dyn[:, 0], color=C_DYN, alpha=0.15)
    ax.axhline(v[i0, 0], color=C_ZOH, lw=1.2, ls="--", label="ZOH")
    ax.axvspan(gs, gs + gl, color="0.5", alpha=0.08)
    ax.set_xlabel("t [s]"); ax.set_ylabel("surge [m/s]")
    ax.set_title(f"(d) example 20 s gap at t={gs:.0f} s (median dynamic-ness)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, lw=0.5)

    # (e) paired delta time-GP minus dyn-GP vs dynamic-ness, surge.
    ax = axes[1, 1]
    by_gap = {}
    for r in filt(records, axis="surge"):
        by_gap.setdefault(gap_key(r), {})[r["supplier"]] = r
    xs, ys = [], []
    for sups in by_gap.values():
        if "causal_gp" in sups and "dyn_gp" in sups:
            xs.append(sups["dyn_gp"]["dynamic_ness"])
            ys.append((sups["causal_gp"]["bridge_rmse"]
                       - sups["dyn_gp"]["bridge_rmse"]) * 100)
    ax.scatter(xs, ys, color=C_DYN, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("gap dynamic-ness [m/s^2]")
    ax.set_ylabel("time-GP minus dyn-GP RMSE [cm/s]  (>0: dyn-GP better)")
    ax.set_title("(e) paired delta vs dynamic-ness (surge)")
    ax.grid(alpha=0.25, lw=0.5)

    # (f) one-step acceleration residual: ridge-only vs ridge+GP, per axis.
    ax = axes[1, 2]
    for ax_i, label in enumerate(AXIS_LABELS):
        recs = filt(records, supplier="dyn_gp", axis=label)
        lin = np.array([r["onestep_rms_lin"] for r in recs]) * AXIS_SCALE[ax_i]
        gpv = np.array([r["onestep_rms_gp"] for r in recs]) * AXIS_SCALE[ax_i]
        ax.scatter(lin, gpv, s=14, alpha=0.6,
                   color=[C_GP, C_DYN, C_ZOH][ax_i],
                   label=f"{label} [{AXIS_DISP[ax_i]}/s]")
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, hi], [0, hi], color="k", lw=1, ls="--")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("one-step accel residual RMS, ridge only")
    ax.set_ylabel("ridge + GP residual")
    ax.set_title("(f) GP-residual contribution (below line = GP helps)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Dagon basin sweep: control-conditioned dynamics GP vs "
                 "time-GP vs ZOH", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dagon_eval.png", dpi=120)


if __name__ == "__main__":
    main()
