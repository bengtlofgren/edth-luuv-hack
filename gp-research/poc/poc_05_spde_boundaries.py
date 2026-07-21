# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""POC 5 — The GP <-> SPDE duality: physics and boundary conditions for free.

references/06 and 07 explain the deep idea: a Matern-kernel GP is not just
*like* a physical field -- it IS the stationary solution of a stochastic PDE:

    (kappa^2 - Laplacian) f(s) = white noise           (1-D, Matern nu=3/2)

That buys two things you cannot get from a kernel formula alone:

  1. PHYSICS: discretise the differential operator instead of building a dense
     covariance matrix.  The resulting precision (inverse covariance) matrix
     is SPARSE and banded => O(n) inference over a spatial field (e.g. an
     ocean-current map).  Swap the operator (add advection, vary kappa in
     space) and the GP's assumed physics changes with it.

  2. BOUNDARY CONDITIONS: the PDE needs boundary conditions anyway, so you
     state them directly -- e.g. Dirichlet "current is zero at the harbour
     wall".  A plain kernel GP has no vocabulary for this at all.

This POC discretises the operator with finite differences on a 1-D transect,
then verifies both claims numerically.

Run:  uv run poc_05_spde_boundaries.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# --- 1. The GP we want: Matern-3/2, lengthscale ELL, on a 1-D transect.
#     The SPDE parameter kappa maps to the kernel lengthscale as
#     kappa = sqrt(2*nu)/ell = sqrt(3)/ell, and the marginal variance of the
#     unit-noise SPDE in 1-D is sigma^2 = 1/(4*kappa^3)  (standard identities
#     from the Lindgren/Rue SPDE literature, see references/06).
ELL = 30.0                       # metres
KAPPA = np.sqrt(3.0) / ELL
SIGMA2 = 1.0 / (4.0 * KAPPA**3)  # implied marginal variance (we'll verify it)

# --- 2. Discretise the operator (kappa^2 - Laplacian) on a grid.
#     D2 is the standard second-difference Laplacian.  Truncating it at the
#     grid ends implicitly imposes DIRICHLET f=0 outside the domain -- that
#     truncation IS the boundary condition (claim 2 uses this on purpose).
L_DOM, N = 600.0, 601            # 600 m transect, 1 m spacing
x = np.linspace(0.0, L_DOM, N)
h = x[1] - x[0]
D2 = (np.diag(np.full(N - 1, 1.0), -1) - 2.0 * np.eye(N)
      + np.diag(np.full(N - 1, 1.0), 1)) / h**2
A = KAPPA**2 * np.eye(N) - D2    # the discretised differential operator

# The SPDE says A f = w with w discrete white noise, Cov(w) = (1/h) I.
# So   Cov(f) = (1/h) A^-1 A^-T   and   Precision(f) = h A^T A.
# A is tridiagonal => the precision is PENTADIAGONAL (sparse!).  Here we use
# dense numpy for clarity; a real implementation uses a sparse Cholesky.
Q = h * (A.T @ A)
print(f"precision sparsity: {np.mean(np.abs(Q) > 1e-12)*100:.1f}% non-zero "
      f"(banded) vs 100% for a dense kernel covariance")

# --- 3. CLAIM 1: the covariance implied by the PDE equals the Matern kernel.
#     Take one column of Cov(f) at the domain centre (far from boundaries)
#     and overlay the analytic Matern-3/2 formula. No kernel was ever coded --
#     the physics (the operator) generated it.
mid = N // 2
cov_mid = np.linalg.solve(Q, np.eye(N)[:, mid])        # one covariance column
r = np.abs(x - x[mid])
matern = SIGMA2 * (1.0 + KAPPA * r) * np.exp(-KAPPA * r)
interior = r < L_DOM / 2 - 3 * ELL                     # avoid boundary zone
err = np.max(np.abs(cov_mid[interior] - matern[interior])) / SIGMA2
print(f"max relative mismatch, SPDE covariance vs analytic Matern-3/2: {err:.1%}")

# --- 4. CLAIM 2: boundary conditions shape the GP.  With Dirichlet walls the
#     marginal std must fall to 0 at the ends -- "the current is pinned at the
#     harbour wall" -- and every prior sample must respect that.
marg_std = np.sqrt(np.diag(np.linalg.inv(Q)))
Lc = np.linalg.cholesky(Q)                              # sample: solve L^T f = z
samples = np.linalg.solve(Lc.T, rng.normal(size=(N, 3)))
print(f"marginal std: centre {marg_std[mid]:.2f} (analytic {np.sqrt(SIGMA2):.2f}), "
      f"at wall {marg_std[1]:.2f}  -> pinned by the boundary condition")

# --- 5. Plot both claims side by side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(r[interior], cov_mid[interior], "C0", lw=3, alpha=0.5,
         label="covariance implied by the PDE")
ax1.plot(r[interior], matern[interior], "k--", lw=1,
         label="analytic Matern-3/2 kernel")
ax1.set(xlabel="distance [m]", ylabel="covariance",
        title="Claim 1: the PDE *is* the Matern kernel")
ax1.legend(fontsize=8)

ax2.plot(x, samples, lw=0.8, alpha=0.8)
ax2.fill_between(x, -2 * marg_std, 2 * marg_std, color="C0", alpha=0.15,
                 label="prior 95% band")
ax2.axvline(0, color="k", lw=2); ax2.axvline(L_DOM, color="k", lw=2)
ax2.set(xlabel="position along transect [m]", ylabel="current anomaly",
        title="Claim 2: Dirichlet walls pin samples & variance to 0")
ax2.legend(fontsize=8)
fig.suptitle("POC 5: a Matern GP as the solution of a stochastic PDE", y=1.0)
fig.tight_layout()
fig.savefig("poc_05_spde_boundaries.png", dpi=120)
print("saved poc_05_spde_boundaries.png")
