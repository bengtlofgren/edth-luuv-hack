# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""POC 4 — The GP <-> Kalman duality: exact GP regression in O(n).

references/05 and 08 make the cheap-edge-compute claim: a temporal GP with a
Matern kernel is MATHEMATICALLY IDENTICAL to a small Kalman filter + smoother.
Not an approximation -- the same posterior, at O(n) cost instead of O(n^3),
with a constant memory footprint.  That is what makes the GP deployable on a
vehicle without a GPU.

This POC proves the identity numerically for the Matern-3/2 kernel:

    k(tau) = sigma^2 (1 + lam*tau) exp(-lam*tau),   lam = sqrt(3)/lengthscale

which is exactly the covariance of the 2-state stochastic differential
equation (state = [f, f']):

    dx = F x dt + L dw,   F = [[0, 1], [-lam^2, -2*lam]]

We run (a) textbook O(n^3) GP regression with dense matrices and (b) a 2-state
Kalman filter + RTS smoother, on the same data, and show the posteriors match
to machine precision -- then time both to show the O(n) vs O(n^3) scaling.

Run:  uv run poc_04_state_space_duality.py
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

rng = np.random.default_rng(11)

# Shared hyperparameters (fixed, not learned -- this POC is about the duality,
# not about fitting; POC 1 already showed hyperparameter learning).
SIGMA2 = 0.3**2        # signal variance sigma^2
ELL = 5.0              # lengthscale [s]
LAM = np.sqrt(3.0) / ELL
NOISE2 = 0.08**2       # measurement noise variance


def matern32(tau):
    """The Matern-3/2 kernel as a plain formula (the 'function-space' view)."""
    a = LAM * np.abs(tau)
    return SIGMA2 * (1.0 + a) * np.exp(-a)


def gp_exact(t_obs, y, t_query):
    """Textbook O(n^3) GP regression: build K, factorise, condition."""
    K = matern32(t_obs[:, None] - t_obs[None, :]) + NOISE2 * np.eye(len(t_obs))
    Ks = matern32(t_query[:, None] - t_obs[None, :])       # cross-covariance
    alpha = np.linalg.solve(K, y)
    mean = Ks @ alpha
    var = matern32(0.0) - np.sum(Ks * np.linalg.solve(K, Ks.T).T, axis=1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def gp_state_space(t_all, y_all, observed):
    """The SAME posterior via Kalman filter + RTS smoother: O(n), 2x2 algebra.

    t_all must be sorted; observed[k] says whether y_all[k] exists (False =
    a pure prediction point, e.g. inside a dropout).
    """
    F = np.array([[0.0, 1.0], [-LAM**2, -2.0 * LAM]])
    Pinf = np.diag([SIGMA2, LAM**2 * SIGMA2])   # stationary state covariance
    H = np.array([[1.0, 0.0]])                  # we observe f, not f'

    n = len(t_all)
    x, P = np.zeros(2), Pinf.copy()             # start from the GP prior
    xf, Pf, xp, Pp, As = [], [], [], [], []     # store for the smoother
    for k in range(n):
        if k > 0:
            # Discretise the SDE over this (possibly irregular) time step.
            A = expm(F * (t_all[k] - t_all[k - 1]))
            Q = Pinf - A @ Pinf @ A.T           # exact, thanks to stationarity
            x, P = A @ x, A @ P @ A.T + Q
        else:
            A = np.eye(2)
        xp.append(x.copy()); Pp.append(P.copy()); As.append(A)
        if observed[k]:                          # standard Kalman update
            S = (H @ P @ H.T)[0, 0] + NOISE2
            K = (P @ H.T / S).ravel()
            x = x + K * (y_all[k] - x[0])
            P = P - np.outer(K, H @ P)
        xf.append(x.copy()); Pf.append(P.copy())

    # RTS smoother: run backwards so every estimate uses ALL data, exactly
    # like the GP posterior (a forward-only filter would only use the past).
    xs, Ps = xf[-1].copy(), Pf[-1].copy()
    mean, var = np.empty(n), np.empty(n)
    mean[-1], var[-1] = xs[0], Ps[0, 0]
    for k in range(n - 2, -1, -1):
        G = Pf[k] @ As[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs = xf[k] + G @ (xs - xp[k + 1])
        Ps = Pf[k] + G @ (Ps - Pp[k + 1]) @ G.T
        mean[k], var[k] = xs[0], Ps[0, 0]
    return mean, np.sqrt(np.maximum(var, 0.0))


# --- 1. Data: noisy samples of a smooth velocity, with a gap (as in POC 2).
t_obs = np.sort(rng.uniform(0.0, 60.0, 80))
t_obs = t_obs[~((t_obs > 25.0) & (t_obs < 40.0))]
y_obs = 0.3 * np.sin(2 * np.pi * t_obs / 25.0) + rng.normal(0.0, 0.08, t_obs.shape)

# Query grid = observation times + prediction-only times, merged and sorted
# (the state-space form handles both in one sweep: no update at silent steps).
t_query = np.linspace(0.0, 60.0, 200)
t_all = np.concatenate([t_obs, t_query])
y_all = np.concatenate([y_obs, np.zeros_like(t_query)])
observed = np.concatenate([np.ones_like(t_obs, bool), np.zeros_like(t_query, bool)])
order = np.argsort(t_all, kind="stable")
t_all, y_all, observed = t_all[order], y_all[order], observed[order]

# --- 2. Run both. Same prior, same data => the duality says: same posterior.
m_gp, s_gp = gp_exact(t_obs, y_obs, t_all)
m_ss, s_ss = gp_state_space(t_all, y_all, observed)
print(f"max |mean difference| : {np.max(np.abs(m_gp - m_ss)):.2e}")
print(f"max |std  difference| : {np.max(np.abs(s_gp - s_ss)):.2e}   (machine precision => identical)")

# --- 3. Cost: the reason this matters for the vehicle.  Dense GP work grows
#     with n^3; the Kalman form stays a fixed 2x2 recursion per sample.
print("\n   n   dense GP   state-space")
for n in (250, 500, 1000, 2000):
    tt = np.sort(rng.uniform(0, 600, n))
    yy = rng.normal(0, 1, n)
    obs = np.ones(n, bool)
    t0 = time.perf_counter(); gp_exact(tt, yy, tt);            t1 = time.perf_counter()
    gp_state_space(tt, yy, obs);                               t2 = time.perf_counter()
    print(f"{n:5d}   {t1-t0:7.3f}s   {t2-t1:9.3f}s")

# --- 4. Plot the two posteriors on top of each other.
fig, ax = plt.subplots(figsize=(10, 4))
ax.axvspan(25, 40, color="0.9", label="gap (prediction only)")
ax.plot(t_obs, y_obs, "r.", ms=5, label="observations")
ax.plot(t_all, m_gp, "C0", lw=3, alpha=0.5, label="dense GP posterior  O(n$^3$)")
ax.plot(t_all, m_ss, "k--", lw=1, label="Kalman smoother  O(n) -- identical")
ax.fill_between(t_all, m_ss - 2 * s_ss, m_ss + 2 * s_ss, color="C0", alpha=0.15)
ax.set(xlabel="time [s]", ylabel="f(t)",
       title="POC 4: a Matern-3/2 GP IS a 2-state Kalman smoother (exact, O(n))")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("poc_04_state_space_duality.png", dpi=120)
print("saved poc_04_state_space_duality.png")
