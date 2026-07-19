"""Tests for eval.seam.run_kf: metric sanity under correct vs. wrong R."""
import numpy as np

from eval.metrics import coverage_2sigma, honesty
from eval.seam import run_kf

BURN_IN = 10   # drop the initial transient before scoring honesty/coverage

# n=40 matches the size of an actual bridged gap (PLAN.md gap durations are
# 10-60 s); honesty is a MAX ratio, so a long stationary run has an
# increasingly heavy tail (E[max|z|] grows with n) even for a perfectly
# calibrated filter -- scoring over a gap-sized window is what the honesty
# check is actually used for downstream (evaluate.py scores per-gap).


def _white_noise_case(r_true, r_claimed, seed=1, n=40, dt=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * dt
    v_true = np.zeros(n)                     # stationary; measurements are pure white noise
    meas = v_true + rng.normal(0.0, np.sqrt(r_true), n)
    meas_var = np.full(n, r_claimed)
    has_meas = np.ones(n, bool)
    xs, Ps = run_kf(t, meas, meas_var, has_meas, q=(0.01) ** 2, v0=0.0, var0=r_claimed)
    err = np.abs(xs[:, 1] - v_true)[BURN_IN:]
    claim = 2 * np.sqrt(Ps[:, 1, 1])[BURN_IN:]
    return err, claim


def test_honest_with_correct_R():
    err, claim = _white_noise_case(r_true=0.02 ** 2, r_claimed=0.02 ** 2)
    assert honesty(err, claim) <= 1.2
    assert coverage_2sigma(err, claim / 2.0) >= 0.9


def test_overconfident_with_R_100x_too_small():
    r_true = 0.02 ** 2
    err, claim = _white_noise_case(r_true=r_true, r_claimed=r_true / 100.0)
    assert honesty(err, claim) > 3.0
