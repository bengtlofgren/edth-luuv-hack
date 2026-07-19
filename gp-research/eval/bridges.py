"""Bridge suppliers: causal/acausal GP, IMU calibration + dead-reckoning, hybrids.

See gp-research/eval/PLAN.md review findings 1, 3, 4 for the methodology this
fixes relative to poc_03_real_data.py / poc_06_imu_bridge.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge

# --- GP bridge ---------------------------------------------------------------

# Matern lengthscale bounds [s]: tight for the ~10 Hz SOLAQUA A50 stream,
# wide for the 1 Hz Snapir stream (poc_06 vs poc_03 settings respectively).
GP_LS_BOUNDS_10HZ = (1.0, 60.0)
GP_LS_BOUNDS_1HZ = (3.0, 200.0)


def select_train(t, gap, causal, window_s):
    """Boolean mask selecting GP training samples for one gap (review finding 1).

    gap = (start, length) in the same units as t.
    causal=True:  trailing window of `window_s` strictly before the gap
                  (deployment-realistic -- the only data actually available
                  when the bridge must be computed).
    causal=False: symmetric +-window_s window around the gap, excluding the
                  gap itself (the POCs' original setting, kept for the
                  acausal-baseline comparison in PLAN.md's protocol).
    """
    start, length = gap
    end = start + length
    t = np.asarray(t, float)
    in_gap = (t >= start) & (t < end)
    if causal:
        mask = (t >= start - window_s) & (t < start)
    else:
        mask = (t >= start - window_s) & (t < end + window_s) & ~in_gap
    return mask & ~in_gap


def gp_bridge(t_train, y_train, t_query, causal, length_scale_bounds=GP_LS_BOUNDS_10HZ):
    """GP posterior mean/std at t_query, fit on (t_train, y_train).

    Matern-2.5 * Constant + WhiteKernel, normalize_y=True, 2 restarts (same
    settings as the POCs). `causal` is a caller-facing label only: whether
    t_train is a trailing pre-gap window or straddles the gap is a decision
    made by the caller (see select_train) -- this function fits identically
    either way, since GP regression itself has no notion of causality (review
    finding 1: causality has to live in what data is selected, not in the
    regressor). length_scale_bounds: GP_LS_BOUNDS_10HZ=(1.0, 60.0) s for
    ~10 Hz SOLAQUA A50 data (default), GP_LS_BOUNDS_1HZ=(3.0, 200.0) s for
    1 Hz Snapir data -- pass explicitly for the latter.

    Returns (mean, std), each shape t_query.shape.
    """
    assert isinstance(causal, bool)
    t_train = np.asarray(t_train, float)[:, None]
    y_train = np.asarray(y_train, float)
    t_query = np.asarray(t_query, float)[:, None]
    lo, hi = length_scale_bounds
    ls0 = float(np.sqrt(lo * hi))
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=ls0, length_scale_bounds=length_scale_bounds, nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(t_train, y_train)
    mean, std = gp.predict(t_query, return_std=True)
    return mean, std


# --- IMU calibration + dead-reckoning bridge ---------------------------------

def _smooth(y, dt, win_s):
    """Centered boxcar of ~win_s seconds (odd length, at least 3 samples)."""
    n = max(3, int(round(win_s / dt)) | 1)
    return np.convolve(y, np.ones(n) / n, mode="same")


def _acf_tau(x, dt):
    """Autocorrelation time: lag (in seconds) at which the sample ACF first
    crosses zero, i.e. the residual decorrelation time."""
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return dt
    c = np.correlate(x, x, "full")[x.size - 1:] / denom
    below = np.flatnonzero(c < 0)
    return (below[0] if below.size else x.size) * dt


def _umeyama(X, Y):
    """Scaled orthogonal Procrustes (Umeyama 1991): find rotation R (proper,
    det=+1), scalar s>0 and offset b minimizing sum_i ||y_i - (s R x_i + b)||^2
    over rows x_i of X, y_i of Y (both (n,3))."""
    n = X.shape[0]
    mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mu_x, Y - mu_y
    var_x = (Xc ** 2).sum(axis=1).mean()
    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_x) if var_x > 0 else 1.0
    b = mu_y - s * (R @ mu_x)
    return R, s, b


@dataclass
class Calib:
    """IMU->body-frame calibration: a_body = s * R @ f_imu + b, with
    w_body = R @ w_imu (same rotation applied to the gyro)."""
    R: np.ndarray            # (3,3) proper rotation, det=+1
    s: float                 # scalar gain (~1 for a healthy, well-scaled IMU)
    b: np.ndarray            # (3,) m/s^2, bias + unmodeled-gravity offset
    sigma_r: np.ndarray      # (3,) residual std [m/s^2]
    tau: np.ndarray          # (3,) residual autocorrelation time [s], clipped [0.2, 5]
    sigma_b: np.ndarray      # (3,) bias-mean uncertainty [m/s^2] = sigma_r/sqrt(n_eff)
    sv_unconstrained: np.ndarray   # (3,) singular values of the unconstrained ridge map
    band_s: float


def imu_calibrate(t_v, v, t_imu, f, w, band_s=1.0):
    """Calibrate IMU specific-force/rate against DVL-derived acceleration
    (review finding 3: replaces the POC's plain ridge regression, whose
    singular values collapsed to 0.005-0.087 under errors-in-variables
    attenuation, with a constrained orthogonal Procrustes fit).

    Model: a_k = dv/dt|_k + w_body x v_k  ~  s * R @ f_k + b, with
    w_body = R @ w_imu (one refinement pass: R is first fit without the
    w x v term, then w_body is formed with that R and the fit is redone --
    "iterate once" per PLAN.md). R is constrained to a proper rotation
    (det=+1) and s is a single scalar gain, found via scaled orthogonal
    Procrustes (_umeyama) rather than an unconstrained 3x3 regression, so
    sensor noise on the IMU side cannot shrink the map towards zero the way
    ordinary least squares does under errors-in-variables.

    Both sides are boxcar-smoothed with the SAME `band_s` window before
    differentiating (v) / resampling onto the DVL clock (f, w) -- the POC
    used mismatched windows (1.0 s on v, 0.3 s on f/w), which is itself a
    source of the apparent attenuation this function is built to avoid.

    t_v, v: (n,), (n,3) DVL fix times [s] and body-frame velocity [m/s].
    t_imu, f, w: (m,), (m,3), (m,3) IMU times [s], specific force [m/s^2] and
        angular rate [rad/s], already SI (apply data.py's unit scales first).

    Returns a Calib. sv_unconstrained is the SVD of an ordinary Ridge(f->a)
    fit on the same (smoothed, band-matched) data: values close to `s`
    indicate a well-conditioned mapping; values << s flag the
    errors-in-variables attenuation the Procrustes fit is designed to avoid
    (report this as a diagnostic, not as the calibration used downstream).
    """
    t_v = np.asarray(t_v, float); v = np.asarray(v, float)
    t_imu = np.asarray(t_imu, float); f = np.asarray(f, float); w = np.asarray(w, float)
    dt_v = float(np.median(np.diff(t_v)))
    dt_imu = float(np.median(np.diff(t_imu)))

    v_s = np.column_stack([_smooth(v[:, i], dt_v, band_s) for i in range(3)])
    dvdt = np.gradient(v_s, t_v, axis=0)
    f_s = np.column_stack([np.interp(t_v, t_imu, _smooth(f[:, i], dt_imu, band_s))
                            for i in range(3)])
    w_s = np.column_stack([np.interp(t_v, t_imu, _smooth(w[:, i], dt_imu, band_s))
                            for i in range(3)])

    trim = slice(5, -5) if t_v.size > 12 else slice(None)
    F, A = f_s[trim], dvdt[trim]
    V = v_s[trim]
    R, s, b = _umeyama(F, A)                 # pass 1: fit ignoring w x v
    w_body = w_s[trim] @ R.T                 # pass 2: one refinement adding it
    A = dvdt[trim] + np.cross(w_body, V)
    R, s, b = _umeyama(F, A)

    resid = A - (s * F @ R.T + b)
    sigma_r = resid.std(axis=0)
    tau = np.clip([_acf_tau(resid[:, i], dt_v) for i in range(3)], 0.2, 5.0)
    n_eff = F.shape[0] * dt_v / tau
    sigma_b = sigma_r / np.sqrt(n_eff)

    ridge = Ridge(alpha=1.0).fit(F, A)
    sv_unconstrained = np.linalg.svd(ridge.coef_, compute_uv=False)

    return Calib(R=R, s=float(s), b=b, sigma_r=sigma_r, tau=np.asarray(tau, float),
                 sigma_b=sigma_b, sv_unconstrained=sv_unconstrained, band_s=float(band_s))


def imu_bridge(calib, v0, t0, t_imu, f, w, t_end, var0=0.0):
    """Causal IMU dead-reckoning bridge: integrate the calibrated specific
    force from (t0, v0) forward through t_end, at the native IMU rate.

    v' = s * R @ f + b - w_body x v,  w_body = R @ w   (calib.R, calib.s, calib.b)

    Variance grows as
        var(T) = var0 + sigma_r^2 * tau * T + (sigma_b * T)^2
    -- a residual-driven random walk (as in the POC) PLUS the sigma_b^2*T^2
    systematic-bias term added per review finding 3 (a calibration bias that
    is merely uncertain, not zero-mean-and-forgotten, degrades the DR
    estimate quadratically in time, not just diffusively).

    t_imu, f, w: full IMU stream (only samples in (t0, t_end] are used).
    var0: variance of the seed v0, per axis (scalar or (3,)); default 0.

    Returns t (k,) IMU sample times in (t0, t_end], mean (k,3), std (k,3).
    """
    t_imu = np.asarray(t_imu, float); f = np.asarray(f, float); w = np.asarray(w, float)
    in_range = (t_imu > t0) & (t_imu <= t_end)
    t = t_imu[in_range]
    fi, wi = f[in_range], w[in_range]
    mean = np.empty((t.size, 3))
    v = np.array(v0, dtype=float).copy()
    t_prev = float(t0)
    for k in range(t.size):
        dt_k = t[k] - t_prev
        w_body = calib.R @ wi[k]
        v = v + dt_k * (calib.s * (calib.R @ fi[k]) + calib.b - np.cross(w_body, v))
        mean[k] = v
        t_prev = t[k]

    T = t - t0
    var0 = np.broadcast_to(np.asarray(var0, float), (3,))
    var = (var0[None, :]
           + (calib.sigma_r ** 2 * calib.tau)[None, :] * T[:, None]
           + (calib.sigma_b[None, :] * T[:, None]) ** 2)
    std = np.sqrt(var)
    return t, mean, std


# --- Hybrid fusion -----------------------------------------------------------

def hybrid_v1(gp_mean, gp_var, imu_mean, imu_var):
    """POC discrepancy-inflation hybrid (kept for direct comparison; review
    finding 4 flags it as asymmetric and as double-using v_imu -- it both
    inflates the GP variance by (v_imu - v_gp)^2 AND then fuses against
    v_imu, so the IMU estimate influences the fused result twice).

    var_gp' = var_gp + (imu_mean - gp_mean)^2, then inverse-variance fuse
    (gp_mean, var_gp') with (imu_mean, imu_var).
    """
    gp_mean = np.asarray(gp_mean, float); gp_var = np.asarray(gp_var, float)
    imu_mean = np.asarray(imu_mean, float); imu_var = np.asarray(imu_var, float)
    d2 = (imu_mean - gp_mean) ** 2
    gp_var_infl = gp_var + d2
    var = 1.0 / (1.0 / gp_var_infl + 1.0 / imu_var)
    mean = var * (gp_mean / gp_var_infl + imu_mean / imu_var)
    return mean, var


def _ci_scalar(mean1, var1, mean2, var2, n_omega=101):
    """Scalar covariance intersection: search omega in [0,1] for the fusion
    weight that minimizes the fused variance
        var_ci(omega) = 1 / (omega/var1 + (1-omega)/var2).
    For scalars, omega/var1 + (1-omega)/var2 is linear in omega, so the
    minimum of var_ci is always at an endpoint -- i.e. this always ends up
    picking whichever input has the smaller variance whole. That is exactly
    covariance intersection's conservatism: with unknown cross-correlation
    between the two inputs it refuses to claim a variance reduction from
    "fusing" and instead falls back to the more certain input. Implemented as
    an explicit grid search (rather than hard-coding the endpoint) per
    PLAN.md's "implement omega search over [0,1] for clarity", so this
    generalizes cleanly if a less-conservative blend rule replaces it later.
    """
    mean1 = np.asarray(mean1, float); var1 = np.asarray(var1, float)
    mean2 = np.asarray(mean2, float); var2 = np.asarray(var2, float)
    omega = np.linspace(0.0, 1.0, n_omega).reshape((n_omega,) + (1,) * mean1.ndim)
    inv_fused = omega / var1 + (1.0 - omega) / var2
    var_all = 1.0 / inv_fused
    idx = np.argmin(var_all, axis=0, keepdims=True)
    var_fused = np.take_along_axis(var_all, idx, axis=0)[0]
    mean_all = var_all * (omega * mean1 / var1 + (1.0 - omega) * mean2 / var2)
    mean_fused = np.take_along_axis(mean_all, idx, axis=0)[0]
    return mean_fused, var_fused


def hybrid_v2(gp_mean, gp_var, imu_mean, imu_var, chi2_thresh=3.84):
    """Symmetric chi-square-gated covariance-intersection hybrid (review
    finding 4 fix): a chi-square gate decides agreement/disagreement, and
    covariance intersection (not inverse-variance fusion) combines the two
    estimates -- CI does not assume independence, so it is safe for the
    correlated-inputs case the POC's inverse-variance fusion got wrong.

    Gate statistic stat = (imu_mean - gp_mean)^2 / (gp_var + imu_var), which
    is ~chi2_1 distributed if both suppliers are honest and independent;
    chi2_thresh=3.84 is the 95% chi2_1 critical value.
      * gate passes (stat <= thresh, suppliers agree): CI-fuse (gp_mean,
        gp_var) with (imu_mean, imu_var) directly.
      * gate fails (stat > thresh, suppliers disagree -- e.g. GP is smoothly
        wrong through an unobserved maneuver): inflate BOTH variances by d^2
        before CI-fusing, so the surprised supplier becomes less certain
        rather than either one over-trusting itself the way v1's
        one-sided inflation did.

    Symmetric in (gp_mean, gp_var) <-> (imu_mean, imu_var): swapping the two
    inputs leaves the fused variance and mean unchanged.
    """
    gp_mean = np.asarray(gp_mean, float); gp_var = np.asarray(gp_var, float)
    imu_mean = np.asarray(imu_mean, float); imu_var = np.asarray(imu_var, float)
    d2 = (imu_mean - gp_mean) ** 2
    stat = d2 / (gp_var + imu_var)
    gate_pass = stat <= chi2_thresh
    v1 = np.where(gate_pass, gp_var, gp_var + d2)
    v2 = np.where(gate_pass, imu_var, imu_var + d2)
    mean, var = _ci_scalar(gp_mean, v1, imu_mean, v2)
    return mean, var
