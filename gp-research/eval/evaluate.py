"""Generic sliding multi-gap sweep engine (review finding 2: n=1 gap per
scenario in the POCs -> many outage start times x durations x suppliers,
distributions not single numbers).

This module is deliberately policy-free about *what* fills a gap: the
Phase-2 scripts (eval_snapir.py, eval_solaqua.py) wire up the actual
suppliers (causal GP, acausal GP, IMU-DR, hybrids, ZOH, linear) via
`suppliers_fn`. sweep_gaps only owns the masking, the shared KF seam, and the
per-gap metrics.
"""
from __future__ import annotations

import numpy as np

from .metrics import coverage_2sigma, drift, honesty, rmse
from .seam import run_kf


def sweep_gaps(t, v, suppliers_fn, gap_lengths, gap_starts,
               q=None, var_meas=None, axis=0, bridge_spacing_s=5.0):
    """Sweep suppliers over every (gap_start, gap_len) combination and score
    each with the shared KF seam.

    t: (N,) time [s]. v: (N,) or (N,3) velocity [m/s] (axis picks the column
        if 2-D). gap_lengths, gap_starts: iterables combined as a full
        cartesian product; a (start, length) pair is skipped if it runs off
        either end of t.
    suppliers_fn(t, v, gap_start, gap_len, have_mask) -> dict[name] = (mean, var)
        called once per gap; each (mean, var) is a full-length (N,) series
        (suppliers are free to use whatever training window/method they like
        -- causal vs. acausal is a decision the supplier makes internally,
        e.g. via bridges.select_train, not something this engine enforces).
    q: KF accel-noise PSD; if None, uses max(0.02, std(diff(v)/diff(t)))^2
        (the POCs' data-driven estimate) computed once over the whole series.
    var_meas: (N,) measurement variance for the real ("have") samples, e.g.
        per-fix fom^2 clipped to a floor. Defaults to a constant 0.025^2
        (the POCs' VAR_A50) if omitted.
    bridge_spacing_s: spacing of bridge samples injected into the gap for the
        KF (unchanged crutch from the POCs, acknowledged in PLAN.md).

    Returns list[dict] with keys: supplier, gap_start, gap_len, rmse,
    coverage, honesty, peak_drift -- one record per (gap, supplier).
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if v.ndim == 2:
        v = v[:, axis]
    n = t.size

    if var_meas is None:
        var_meas = np.full(n, 0.025 ** 2)
    else:
        var_meas = np.asarray(var_meas, float)

    if q is None:
        dv_dt = np.diff(v) / np.diff(t)
        q = max(0.02, float(np.std(dv_dt))) ** 2

    pos_ref = np.concatenate(
        [[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])

    records = []
    for gap_len in gap_lengths:
        for gap_start in gap_starts:
            gap_end = gap_start + gap_len
            in_gap = (t >= gap_start) & (t < gap_end)
            if not in_gap.any() or gap_start < t[0] or gap_end > t[-1]:
                continue
            have = ~in_gap

            bridge_times = np.arange(gap_start + bridge_spacing_s / 2, gap_end,
                                      bridge_spacing_s)
            is_bridge = np.zeros(n, bool)
            for bt in bridge_times:
                is_bridge[np.argmin(np.abs(t - bt))] = True
            has_meas = have | is_bridge

            suppliers = suppliers_fn(t, v, gap_start, gap_len, have)
            for name, (mean_s, var_s) in suppliers.items():
                mean_s = np.asarray(mean_s, float)
                var_s = np.asarray(var_s, float)
                meas = np.where(have, v, mean_s)
                meas_var = np.where(have, var_meas, var_s)
                xs, Ps = run_kf(t, meas, meas_var, has_meas, q,
                                 float(v[0]), float(var_meas[0]))

                v_err = np.abs(xs[:, 1] - v)
                v_claim = 2 * np.sqrt(Ps[:, 1, 1] + var_meas)
                p_err = drift(xs[:, 0], pos_ref)

                records.append(dict(
                    supplier=name,
                    gap_start=float(gap_start),
                    gap_len=float(gap_len),
                    rmse=rmse(xs[in_gap, 1], v[in_gap]),
                    coverage=coverage_2sigma(v_err[in_gap], v_claim[in_gap] / 2.0),
                    honesty=honesty(v_err[in_gap], v_claim[in_gap]),
                    peak_drift=float(p_err[in_gap].max()),
                ))
    return records
