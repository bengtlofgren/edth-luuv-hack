"""Evaluation metrics: accuracy, honesty (calibration), and drift.

Review finding 6: honesty must be checked against an independently estimated
noise floor, not the fitted WhiteKernel evaluating itself -- these functions
are noise-source-agnostic (the caller supplies `claim`), so that fix lives in
how eval_snapir.py / eval_solaqua.py build `claim`, not here.
"""
from __future__ import annotations

import numpy as np


def rmse(pred, truth):
    """Root-mean-square error between pred and truth (same shape)."""
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, float)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def coverage_2sigma(err, sigma):
    """Fraction of samples where |err| <= 2*sigma (nominal 95.4% under
    Gaussian-honest uncertainty)."""
    err = np.asarray(err, float)
    sigma = np.asarray(sigma, float)
    return float(np.mean(np.abs(err) <= 2.0 * sigma))


def honesty(err, claim):
    """Max ratio |err| / claim over all samples. claim is the filter's
    already-computed uncertainty bound (e.g. 2*sqrt(P_vv + sigma_ref^2), with
    the reference's own noise added per the POCs' honesty check). <= 1 means
    the filter never claimed tighter uncertainty than it actually achieved;
    >> 1 flags overconfidence."""
    err = np.asarray(err, float)
    claim = np.asarray(claim, float)
    return float(np.max(np.abs(err) / claim))


def drift(pos_est, pos_ref):
    """Position-drift error time series |pos_est - pos_ref| [m]."""
    pos_est = np.asarray(pos_est, float)
    pos_ref = np.asarray(pos_ref, float)
    return np.abs(pos_est - pos_ref)
