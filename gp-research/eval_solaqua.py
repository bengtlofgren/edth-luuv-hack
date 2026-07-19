# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "rosbags"]
# ///
"""SOLAQUA sweep -- re-tests poc_06_imu_bridge.py's claims ("IMU wins when
dynamic"; "hybrid near-best in both regimes")
with the causal-by-construction, multi-gap-ensemble methodology of
gp-research/eval/PLAN.md.

Uses the eval/ package (data.py, bridges.py, seam.py, metrics.py) as a
library; this script owns the sweep loop, the SOLAQUA-specific supplier
wiring (all 3 axes for bridging, surge-only for the KF), and the
aggregation/plots. See PLAN.md's "Protocol" and "SOLAQUA" sections for the
exact per-gap design this implements.

Axis convention (WaterLinked A50 body frame, verified empirically against
the "fast 0.3 m/s" bag from poc_06's docstring -- axis 0 carries the mean
~0.28 m/s traverse speed): x=surge, y=sway, z=heave.

Run:  ~/dvl-gp/.venv/bin/python gp-research/eval_solaqua.py   (from repo root)
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # gp-research/ -> eval package

from eval.bridges import (
    GP_LS_BOUNDS_10HZ,
    gp_bridge,
    hybrid_v1,
    hybrid_v2,
    imu_bridge,
    imu_calibrate,
    select_train,
)
from eval.data import SOLAQUA_BAGS, load_solaqua_bag
from eval.metrics import coverage_2sigma, drift, honesty, rmse
from eval.seam import run_kf

OUT_DIR = Path(__file__).resolve().parent / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

AXIS_LABELS = ["surge", "sway", "heave"]
KF_SUPPLIERS = {"causal_gp", "imu_dr", "hybrid_v2", "zoh"}
GAP_LENGTHS = (10.0, 20.0)
CAL_WINDOW_S = 60.0          # "trailing window up to 60 s back" for causal GP + IMU-DR calib
FIRST_START = 25.0
START_STRIDE = 10.0
TAIL_MARGIN = 5.0
BRIDGE_SPACING_S = 5.0
KF_Q = 0.04 ** 2
FOM_VAR_FLOOR = 1e-4

C_GP, C_IMU, C_HYB2, C_HYB1, C_ZOH = "#2a78d6", "#1baf7a", "#4a3aa7", "#e87ba4", "#eb6834"
SUPPLIER_COLORS = {
    "causal_gp": C_GP, "acausal_gp": "#7fb2e8", "imu_dr": C_IMU,
    "hybrid_v1": C_HYB1, "hybrid_v2": C_HYB2, "zoh": C_ZOH,
}


def valid_gap_starts(t_end, gap_len):
    """Sliding starts every START_STRIDE s, first >= FIRST_START, last gap
    end >= TAIL_MARGIN s before t_end (PLAN.md SOLAQUA protocol)."""
    starts = []
    s = FIRST_START
    while s + gap_len <= t_end - TAIL_MARGIN:
        starts.append(s)
        s += START_STRIDE
    return starts


def _smooth_boxcar(y, dt, win_s):
    n = max(3, int(round(win_s / dt)) | 1)
    return np.convolve(y, np.ones(n) / n, mode="same")


def dynamic_ness(t_hid, v_surge_hid):
    """std of 2s-smoothed surge dv/dt inside the gap (evaluation label only,
    not used by any supplier)."""
    if t_hid.size < 4:
        return 0.0
    dt = float(np.median(np.diff(t_hid)))
    v_s = _smooth_boxcar(v_surge_hid, dt, 2.0)
    dv_dt = np.gradient(v_s, t_hid)
    return float(np.std(dv_dt))


def median_iqr(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(arr)), float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)))


def process_gap(d, filename, group, gap_start, gap_len, pos_ref, var_meas_kf):
    a50, imu = d["a50"], d["imu"]
    t, v, fom = a50["t"], a50["v"], a50["fom"]
    t_imu, f, w = imu["t"], imu["f"], imu["w"]
    n = t.size
    gap_end = gap_start + gap_len
    in_gap = (t >= gap_start) & (t < gap_end)
    have = ~in_gap
    if not in_gap.any():
        return []
    t_hid, v_hid, fom_hid = t[in_gap], v[in_gap], fom[in_gap]

    pre_mask_a50 = (t >= gap_start - CAL_WINDOW_S) & (t < gap_start)
    pre_mask_imu = (t_imu >= gap_start - CAL_WINDOW_S) & (t_imu < gap_start)
    if pre_mask_a50.sum() < 10 or pre_mask_imu.sum() < 10:
        return []  # should not happen given FIRST_START=25, defensive only

    i_last_pre = np.flatnonzero(t < gap_start)[-1]
    t0, v0, fom0 = t[i_last_pre], v[i_last_pre], fom[i_last_pre]
    var0 = np.clip(fom0 ** 2, FOM_VAR_FLOOR, None)
    noise_floor_gap = float(np.clip(np.median(fom_hid ** 2), FOM_VAR_FLOOR, None))

    dyn = dynamic_ness(t_hid, v_hid[:, 0])

    # --- causal + acausal GP, all 3 axes.
    gp_c_mean = np.zeros((t_hid.size, 3)); gp_c_var = np.zeros_like(gp_c_mean)
    gp_a_mean = np.zeros_like(gp_c_mean); gp_a_var = np.zeros_like(gp_c_mean)
    mask_c = select_train(t, (gap_start, gap_len), True, CAL_WINDOW_S)
    mask_a = select_train(t, (gap_start, gap_len), False, 1e6)  # -> all non-gap samples
    for ax in range(3):
        m, s = gp_bridge(t[mask_c], v[mask_c, ax], t_hid, True,
                          length_scale_bounds=GP_LS_BOUNDS_10HZ)
        gp_c_mean[:, ax], gp_c_var[:, ax] = m, s ** 2
        m, s = gp_bridge(t[mask_a], v[mask_a, ax], t_hid, False,
                          length_scale_bounds=GP_LS_BOUNDS_10HZ)
        gp_a_mean[:, ax], gp_a_var[:, ax] = m, s ** 2

    # --- IMU calibration + dead-reckoning bridge.
    calib = imu_calibrate(t[pre_mask_a50], v[pre_mask_a50],
                           t_imu[pre_mask_imu], f[pre_mask_imu], w[pre_mask_imu],
                           band_s=1.0)
    t_dr, mean_dr, std_dr = imu_bridge(calib, v0, t0, t_imu, f, w, gap_end, var0=var0)
    if t_dr.size < 2:
        return []  # no IMU samples spanning the gap -- defensive, shouldn't occur
    imu_mean = np.column_stack([np.interp(t_hid, t_dr, mean_dr[:, ax]) for ax in range(3)])
    imu_std = np.column_stack([np.interp(t_hid, t_dr, std_dr[:, ax]) for ax in range(3)])
    imu_var = imu_std ** 2

    # --- hybrids (causal GP + IMU-DR, the deployable pairing).
    hyb1_mean = np.zeros_like(gp_c_mean); hyb1_var = np.zeros_like(gp_c_mean)
    hyb2_mean = np.zeros_like(gp_c_mean); hyb2_var = np.zeros_like(gp_c_mean)
    for ax in range(3):
        gv = np.maximum(gp_c_var[:, ax], 1e-8)
        iv = np.maximum(imu_var[:, ax], 1e-8)
        hyb1_mean[:, ax], hyb1_var[:, ax] = hybrid_v1(gp_c_mean[:, ax], gv, imu_mean[:, ax], iv)
        hyb2_mean[:, ax], hyb2_var[:, ax] = hybrid_v2(gp_c_mean[:, ax], gv, imu_mean[:, ax], iv)

    # --- ZOH: hold last pre-gap fix, fixed (bag-median-fom) R.
    zoh_mean = np.tile(v0, (t_hid.size, 1))
    zoh_var_scalar = np.clip(np.median(fom) ** 2, FOM_VAR_FLOOR, None)
    zoh_var = np.full_like(zoh_mean, zoh_var_scalar)

    suppliers = {
        "causal_gp": (gp_c_mean, gp_c_var),
        "acausal_gp": (gp_a_mean, gp_a_var),
        "imu_dr": (imu_mean, imu_var),
        "hybrid_v1": (hyb1_mean, hyb1_var),
        "hybrid_v2": (hyb2_mean, hyb2_var),
        "zoh": (zoh_mean, zoh_var),
    }

    bridge_times = np.arange(gap_start + BRIDGE_SPACING_S / 2, gap_end, BRIDGE_SPACING_S)
    is_bridge = np.zeros(n, bool)
    for bt in bridge_times:
        is_bridge[np.argmin(np.abs(t - bt))] = True
    has_meas = have | is_bridge

    recs = []
    for sup, (mean, var) in suppliers.items():
        for ax in range(3):
            err = np.abs(mean[:, ax] - v_hid[:, ax])
            rec = dict(
                bag=filename, bag_group=group, supplier=sup, axis=AXIS_LABELS[ax],
                gap_start=float(gap_start), gap_len=float(gap_len),
                bridge_rmse=rmse(mean[:, ax], v_hid[:, ax]),
                bridge_coverage=coverage_2sigma(err, np.sqrt(np.maximum(var[:, ax], 1e-12))),
                dynamic_ness=dyn, n_hidden=int(t_hid.size),
                kf_vrmse=None, kf_honesty=None, kf_drift=None,
                calib_s=None, calib_sigma_r=None, calib_sv_unconstrained=None,
            )
            if sup in ("imu_dr", "hybrid_v1", "hybrid_v2"):
                rec.update(calib_s=float(calib.s),
                           calib_sigma_r=calib.sigma_r.tolist(),
                           calib_sv_unconstrained=calib.sv_unconstrained.tolist())
            if sup in KF_SUPPLIERS and ax == 0:
                full_mean = np.zeros(n); full_var = np.zeros(n)
                full_mean[in_gap] = mean[:, 0]; full_var[in_gap] = var[:, 0]
                meas = np.where(have, v[:, 0], full_mean)
                meas_var = np.where(have, var_meas_kf, full_var)
                xs, Ps = run_kf(t, meas, meas_var, has_meas, KF_Q,
                                 float(v[0, 0]), float(var_meas_kf[0]))
                v_err = np.abs(xs[:, 1] - v[:, 0])
                v_claim_gap = 2 * np.sqrt(Ps[in_gap, 1, 1] + noise_floor_gap)
                p_err = drift(xs[:, 0], pos_ref)
                rec.update(
                    kf_vrmse=rmse(xs[in_gap, 1], v[in_gap, 0]),
                    kf_honesty=honesty(v_err[in_gap], v_claim_gap),
                    kf_drift=float(p_err[in_gap].max()),
                )
            recs.append(rec)
    return recs


def process_bag(filename, cfg):
    group = "nucleus" if cfg["imu_style"] == "vector3" else "pixhawk"
    t_dl = time.time()
    d = load_solaqua_bag(filename)
    a50 = d["a50"]
    t_end = float(a50["t"][-1])
    var_meas_kf = np.clip(a50["fom"] ** 2, FOM_VAR_FLOOR, None)
    v = a50["v"][:, 0]
    t = a50["t"]
    pos_ref = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])

    records, skipped_starts = [], []
    starts_by_len = {gl: valid_gap_starts(t_end, gl) for gl in GAP_LENGTHS}
    n_starts_total = sum(len(s) for s in starts_by_len.values())
    print(f"[{filename}] dur={t_end:.1f}s group={group} "
          f"starts: 10s={starts_by_len[10.0]} 20s={starts_by_len[20.0]} "
          f"(loaded in {time.time()-t_dl:.1f}s)", flush=True)
    if n_starts_total == 0:
        print(f"  SKIP: bag too short for protocol (need >= "
              f"{FIRST_START+min(GAP_LENGTHS)+TAIL_MARGIN:.0f}s, have {t_end:.1f}s)")
        return [], dict(bag=filename, group=group, dur=t_end, starts_10=[], starts_20=[])

    for gl in GAP_LENGTHS:
        for gs in starts_by_len[gl]:
            t_g0 = time.time()
            try:
                recs = process_gap(d, filename, group, gs, gl, pos_ref, var_meas_kf)
            except Exception as e:
                print(f"  gap start={gs:.0f} len={gl:.0f}: ERROR {e!r}")
                traceback.print_exc()
                skipped_starts.append((gl, gs, repr(e)))
                continue
            if not recs:
                skipped_starts.append((gl, gs, "no records (defensive skip)"))
                continue
            records.extend(recs)
            print(f"  gap start={gs:.0f} len={gl:.0f}: {len(recs)} records "
                  f"({time.time()-t_g0:.1f}s)", flush=True)
    return records, dict(bag=filename, group=group, dur=t_end,
                          starts_10=starts_by_len[10.0], starts_20=starts_by_len[20.0],
                          skipped=skipped_starts)


def main():
    warnings.filterwarnings("ignore")
    t_run0 = time.time()
    all_records = []
    bag_reports = []
    load_failures = []
    for filename, cfg in SOLAQUA_BAGS.items():
        try:
            recs, report = process_bag(filename, cfg)
        except Exception as e:
            print(f"[{filename}] LOAD/PROCESS FAILURE: {e!r}")
            traceback.print_exc()
            load_failures.append((filename, repr(e)))
            continue
        all_records.extend(recs)
        bag_reports.append(report)

    print(f"\nTotal records: {len(all_records)}  (total runtime "
          f"{time.time()-t_run0:.1f}s)")

    with open(OUT_DIR / "solaqua_records.json", "w") as fh:
        json.dump(dict(records=all_records, bag_reports=bag_reports,
                        load_failures=load_failures), fh, indent=1)
    print(f"wrote {OUT_DIR / 'solaqua_records.json'}")

    summary = build_summary(all_records, bag_reports, load_failures)
    (OUT_DIR / "solaqua_summary.md").write_text(summary)
    print(f"wrote {OUT_DIR / 'solaqua_summary.md'}")
    print("\n" + summary)

    make_figure(all_records)
    print(f"wrote {OUT_DIR / 'solaqua_eval.png'}")


# --- Aggregation --------------------------------------------------------

def filt(records, **kw):
    out = records
    for k, v in kw.items():
        if callable(v):
            out = [r for r in out if v(r[k])]
        else:
            out = [r for r in out if r[k] == v]
    return out


def fmt_mi(vals):
    m, lo, hi = median_iqr(vals)
    if np.isnan(m):
        return "n/a"
    return f"{m*100:6.2f} [{lo*100:5.2f}, {hi*100:5.2f}] cm/s (n={len(vals)})"


SUPPLIERS_ORDER = ["causal_gp", "acausal_gp", "imu_dr", "hybrid_v1", "hybrid_v2", "zoh"]


def build_summary(records, bag_reports, load_failures):
    lines = []
    lines.append("# SOLAQUA sweep summary\n")

    lines.append("## Bags used / skipped\n")
    for r in bag_reports:
        n10, n20 = len(r["starts_10"]), len(r["starts_20"])
        status = "OK" if (n10 or n20) else "SKIPPED (too short)"
        lines.append(f"- `{r['bag']}` ({r['group']}, dur={r['dur']:.1f}s): "
                      f"{n10} x 10s gaps, {n20} x 20s gaps -- {status}")
        if r.get("skipped"):
            for gl, gs, reason in r["skipped"]:
                lines.append(f"    - skipped gap len={gl:.0f} start={gs:.0f}: {reason}")
    for fn, err in load_failures:
        lines.append(f"- `{fn}`: LOAD FAILURE -- {err}")
    lines.append("")

    lines.append("## 1. Bridge RMSE median [IQR], surge axis, by supplier x duration\n")
    for group_name, group_filter in [("nucleus bags", "nucleus"),
                                      ("pixhawk bags", "pixhawk"), ("overall", None)]:
        lines.append(f"### {group_name}\n")
        for gl in GAP_LENGTHS:
            lines.append(f"- {gl:.0f}s gaps:")
            for sup in SUPPLIERS_ORDER:
                kw = dict(supplier=sup, axis="surge", gap_len=gl)
                if group_filter:
                    kw["bag_group"] = group_filter
                vals = [r["bridge_rmse"] for r in filt(records, **kw)]
                lines.append(f"    - {sup:12s}: {fmt_mi(vals)}")
        lines.append("")

    lines.append("## 1b. Bridge RMSE median [IQR], compact all-axes (overall, both durations)\n")
    for ax in AXIS_LABELS:
        lines.append(f"- {ax}:")
        for sup in SUPPLIERS_ORDER:
            vals = [r["bridge_rmse"] for r in filt(records, supplier=sup, axis=ax)]
            lines.append(f"    - {sup:12s}: {fmt_mi(vals)}")
    lines.append("")

    lines.append("## 2. IMU-when-dynamic claim -- IMU-DR vs causal GP vs ZOH on dynamic vs calm gaps (surge)\n")
    surge_recs = filt(records, axis="surge")
    dyn_vals = sorted({(r["bag"], r["gap_start"], r["gap_len"]): r["dynamic_ness"]
                        for r in surge_recs}.values())
    dyn_med = float(np.median(dyn_vals)) if dyn_vals else float("nan")
    lines.append(f"Median dynamic-ness across {len(dyn_vals)} gaps: {dyn_med:.4f} m/s^2 "
                 "(split point for dynamic/calm halves below)\n")

    def gap_key(r):
        return (r["bag"], r["gap_start"], r["gap_len"])

    by_gap = {}
    for r in surge_recs:
        by_gap.setdefault(gap_key(r), {})[r["supplier"]] = r

    for label, cond in [("dynamic half (dynamic_ness >= median)", lambda d: d >= dyn_med),
                         ("calm half (dynamic_ness < median)", lambda d: d < dyn_med)]:
        lines.append(f"### {label}\n")
        pairs_gi, pairs_gz, pairs_iz = [], [], []
        for k, sups in by_gap.items():
            if "causal_gp" not in sups or "imu_dr" not in sups or "zoh" not in sups:
                continue
            d = sups["causal_gp"]["dynamic_ness"]
            if not cond(d):
                continue
            g, i, z = (sups["causal_gp"]["bridge_rmse"], sups["imu_dr"]["bridge_rmse"],
                       sups["zoh"]["bridge_rmse"])
            pairs_gi.append(i - g)   # IMU - GP: negative = IMU wins
            pairs_gz.append(i - z)   # IMU - ZOH
            pairs_iz.append(z - g)   # ZOH - GP (for context)
        n = len(pairs_gi)
        if n:
            win_imu_vs_gp = float(np.mean(np.array(pairs_gi) < 0))
            win_imu_vs_zoh = float(np.mean(np.array(pairs_gz) < 0))
            m_gi, lo_gi, hi_gi = median_iqr(pairs_gi)
            m_gz, lo_gz, hi_gz = median_iqr(pairs_gz)
            lines.append(f"- n={n} gaps")
            lines.append(f"- IMU-DR minus causal-GP RMSE: median {m_gi*100:.2f} "
                         f"[{lo_gi*100:.2f}, {hi_gi*100:.2f}] cm/s "
                         f"(IMU wins {win_imu_vs_gp*100:.0f}% of gaps)")
            lines.append(f"- IMU-DR minus ZOH RMSE: median {m_gz*100:.2f} "
                         f"[{lo_gz*100:.2f}, {hi_gz*100:.2f}] cm/s "
                         f"(IMU wins {win_imu_vs_zoh*100:.0f}% of gaps)")
        else:
            lines.append("- n=0 gaps (no complete causal_gp/imu_dr/zoh triples)")
        lines.append("")

    lines.append("### IMU-DR vs ZOH overall (all gaps, surge)\n")
    all_imu = {gap_key(r): r["bridge_rmse"] for r in filt(records, supplier="imu_dr")}
    all_zoh = {gap_key(r): r["bridge_rmse"] for r in filt(records, supplier="zoh")}
    common = sorted(set(all_imu) & set(all_zoh))
    deltas = [all_zoh[k] - all_imu[k] for k in common]  # positive = IMU better
    within_1cm = float(np.mean(np.abs(deltas) <= 0.01)) if deltas else float("nan")
    m, lo, hi = median_iqr(deltas)
    lines.append(f"- n={len(common)} gaps; ZOH-minus-IMU RMSE median {m*100:.2f} "
                 f"[{lo*100:.2f}, {hi*100:.2f}] cm/s")
    lines.append(f"- fraction of gaps where |ZOH - IMU| RMSE <= 1 cm/s: {within_1cm*100:.0f}%\n")

    lines.append("## 3. Hybrid claim -- hybrid v2 vs best single supplier (regret), hybrid v1 vs v2\n")
    for ax in AXIS_LABELS:
        ax_recs = filt(records, axis=ax)
        by_gap_ax = {}
        for r in ax_recs:
            by_gap_ax.setdefault(gap_key(r), {})[r["supplier"]] = r
        regret2, regret1, cov1, cov2, hyb_vs_hyb = [], [], [], [], []
        for k, sups in by_gap_ax.items():
            need = ("causal_gp", "imu_dr", "hybrid_v1", "hybrid_v2")
            if not all(s in sups for s in need):
                continue
            best = min(sups["causal_gp"]["bridge_rmse"], sups["imu_dr"]["bridge_rmse"])
            regret2.append(sups["hybrid_v2"]["bridge_rmse"] - best)
            regret1.append(sups["hybrid_v1"]["bridge_rmse"] - best)
            cov1.append(sups["hybrid_v1"]["bridge_coverage"])
            cov2.append(sups["hybrid_v2"]["bridge_coverage"])
            hyb_vs_hyb.append(sups["hybrid_v2"]["bridge_rmse"] - sups["hybrid_v1"]["bridge_rmse"])
        if not regret2:
            lines.append(f"- {ax}: n=0 complete gaps")
            continue
        m2, lo2, hi2 = median_iqr(regret2)
        m1, lo1, hi1 = median_iqr(regret1)
        mh, loh, hih = median_iqr(hyb_vs_hyb)
        lines.append(f"- {ax} (n={len(regret2)}):")
        lines.append(f"    - hybrid_v2 regret vs min(GP,IMU): median {m2*100:.2f} "
                     f"[{lo2*100:.2f}, {hi2*100:.2f}] cm/s")
        lines.append(f"    - hybrid_v1 regret vs min(GP,IMU): median {m1*100:.2f} "
                     f"[{lo1*100:.2f}, {hi1*100:.2f}] cm/s")
        lines.append(f"    - hybrid_v2 minus hybrid_v1 RMSE: median {mh*100:.2f} "
                     f"[{loh*100:.2f}, {hih*100:.2f}] cm/s "
                     f"({'v2 better' if mh < 0 else 'v1 better'})")
        lines.append(f"    - hybrid_v1 bridge coverage: {np.mean(cov1)*100:.0f}%, "
                     f"hybrid_v2 bridge coverage: {np.mean(cov2)*100:.0f}%")
    lines.append("")

    lines.append("## 4. Coverage table (bridge-level 2-sigma coverage, surge, all gaps)\n")
    for sup in SUPPLIERS_ORDER:
        vals = [r["bridge_coverage"] for r in filt(records, supplier=sup, axis="surge")]
        if vals:
            lines.append(f"- {sup:12s}: {np.mean(vals)*100:5.1f}% (n={len(vals)})")
    lines.append("")

    lines.append("## 5. KF metrics (surge only, causal_gp/imu_dr/hybrid_v2/zoh)\n")
    for sup in ["causal_gp", "imu_dr", "hybrid_v2", "zoh"]:
        recs = filt(records, supplier=sup, axis="surge")
        vrmse = [r["kf_vrmse"] for r in recs if r["kf_vrmse"] is not None]
        hon = [r["kf_honesty"] for r in recs if r["kf_honesty"] is not None]
        dft = [r["kf_drift"] for r in recs if r["kf_drift"] is not None]
        lines.append(f"- {sup:12s}: vRMSE {fmt_mi(vrmse)}; "
                     f"honesty median {np.median(hon):.2f}x (n={len(hon)}); "
                     f"peak drift median {np.median(dft):.3f} m (n={len(dft)})"
                     if vrmse else f"- {sup:12s}: n=0")
    lines.append("")

    lines.append("## 6. IMU calibration diagnostics by bag group\n")
    for group_name in ("nucleus", "pixhawk"):
        recs = [r for r in records if r["bag_group"] == group_name and r["calib_s"] is not None]
        seen = {}
        for r in recs:
            seen[gap_key(r)] = r
        recs_u = list(seen.values())
        if not recs_u:
            lines.append(f"- {group_name}: n=0")
            continue
        s_vals = [r["calib_s"] for r in recs_u]
        sv = np.array([r["calib_sv_unconstrained"] for r in recs_u])
        lines.append(f"- {group_name} (n={len(recs_u)} gaps): Procrustes scale s median "
                     f"{np.median(s_vals):.3f} [{np.percentile(s_vals,25):.3f}, "
                     f"{np.percentile(s_vals,75):.3f}]; "
                     f"unconstrained-map singular values median "
                     f"{np.median(sv, axis=0).round(3).tolist()}")
    lines.append("")

    return "\n".join(lines)


def make_figure(records):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) RMSE distributions by supplier, 20s gaps, surge.
    ax = axes[0]
    data, labels, colors = [], [], []
    for sup in SUPPLIERS_ORDER:
        vals = [r["bridge_rmse"] * 100 for r in filt(records, supplier=sup, axis="surge",
                                                       gap_len=20.0)]
        if vals:
            data.append(vals); labels.append(sup); colors.append(SUPPLIER_COLORS[sup])
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("bridge RMSE [cm/s]")
    ax.set_title("(a) RMSE by supplier, 20s gaps, surge")
    ax.grid(alpha=0.25, lw=0.5)

    # (b) hybrid v2 vs v1 regret histogram.
    ax = axes[1]
    by_gap = {}
    for r in filt(records, axis="surge"):
        by_gap.setdefault((r["bag"], r["gap_start"], r["gap_len"]), {})[r["supplier"]] = r
    reg1, reg2 = [], []
    for sups in by_gap.values():
        need = ("causal_gp", "imu_dr", "hybrid_v1", "hybrid_v2")
        if not all(s in sups for s in need):
            continue
        best = min(sups["causal_gp"]["bridge_rmse"], sups["imu_dr"]["bridge_rmse"])
        reg1.append((sups["hybrid_v1"]["bridge_rmse"] - best) * 100)
        reg2.append((sups["hybrid_v2"]["bridge_rmse"] - best) * 100)
    if reg1:
        bins = np.linspace(min(reg1 + reg2), max(reg1 + reg2), 20)
        ax.hist(reg1, bins=bins, alpha=0.5, color=C_HYB1, label="hybrid v1")
        ax.hist(reg2, bins=bins, alpha=0.5, color=C_HYB2, label="hybrid v2")
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.legend(fontsize=8)
    ax.set_xlabel("regret vs min(GP, IMU) [cm/s]")
    ax.set_ylabel("count")
    ax.set_title("(b) hybrid regret, surge, all gaps")
    ax.grid(alpha=0.25, lw=0.5)

    # (c) IMU-vs-ZOH paired delta vs dynamic-ness.
    ax = axes[2]
    dyns, deltas = [], []
    for sups in by_gap.values():
        if "imu_dr" not in sups or "zoh" not in sups:
            continue
        dyns.append(sups["imu_dr"]["dynamic_ness"])
        deltas.append((sups["zoh"]["bridge_rmse"] - sups["imu_dr"]["bridge_rmse"]) * 100)
    if dyns:
        ax.scatter(dyns, deltas, color=C_IMU, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("dynamic-ness [m/s^2] (std of smoothed surge dv/dt)")
    ax.set_ylabel("ZOH minus IMU-DR RMSE [cm/s]  (>0: IMU better)")
    ax.set_title("(c) IMU vs ZOH paired delta vs dynamic-ness")
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("SOLAQUA sweep: bridge RMSE, hybrid regret, IMU-vs-ZOH dynamics",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "solaqua_eval.png", dpi=120)


if __name__ == "__main__":
    main()
