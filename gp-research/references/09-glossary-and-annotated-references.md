# 09 — Glossary and Annotated References

> **What you'll learn:** quick definitions of the recurring terms, and an annotated, linked
> bibliography of every source with a one-line "read this for X."
> **Prereqs:** none (reference doc).
> **Why it matters here:** a single place to look things up and to find the primary sources.

## Glossary

- **Gaussian Process (GP)** — a distribution over functions; any finite set of outputs is jointly
  Gaussian, defined by a mean function and a kernel. [doc 01](01-gaussian-processes-fundamentals.md)
- **Kernel / covariance function `k(x,x′)`** — encodes how correlated two outputs are; sets
  smoothness, amplitude, reach. [doc 02](02-kernels-and-covariance-functions.md)
- **Matérn kernel** — kernel family with tunable smoothness `ν`; ν=3/2 (once-differentiable) is our
  default. Exactly representable as an SDE/Kalman model and the SPDE Laplacian model. [02](02-kernels-and-covariance-functions.md), [05](05-state-space-gps-and-the-kalman-duality.md), [06](06-spde-approach-the-pde-duality.md)
- **Lengthscale `ℓ`** — input distance over which outputs stay correlated; the main smoothness knob.
- **Signal variance `σ²` / noise variance `σₙ²`** — amplitude of the function / of observation noise.
- **ARD** — automatic relevance determination; per-input lengthscales that let the model ignore
  irrelevant inputs. [02](02-kernels-and-covariance-functions.md)
- **Multitask / multi-output GP (LMC/ICM)** — joint GP over correlated outputs via a coregionalization
  matrix `B`; used for correlated `(vx,vy,vz)`. [02 §8](02-kernels-and-covariance-functions.md)
- **Marginal likelihood** — probability of the data under the GP prior; maximized to learn
  hyperparameters; auto-balances fit vs complexity. [01 §5](01-gaussian-processes-fundamentals.md)
- **Predictive mean / covariance** — the GP's outputs at a query point; the covariance is what we feed
  the EKF as adaptive `R`. [03 §6](03-gp-regression-and-inference.md)
- **Cholesky factorization** — stable `LLᵀ` decomposition used to solve GP equations; the O(n³) step.
  [03](03-gp-regression-and-inference.md)
- **Jitter** — tiny diagonal added for numerical stability. [03 §5](03-gp-regression-and-inference.md)
- **Error-state (indirect) EKF** — filters the small error between truth and an IMU-propagated nominal
  state; this repo's filter. [04 §1](04-gps-for-navigation-and-sensor-fusion.md)
- **Adaptive measurement noise `Rₖ`** — using a per-measurement, GP-predicted covariance instead of a
  fixed `R`; the core fusion idea. [04 §2–3](04-gps-for-navigation-and-sensor-fusion.md)
- **Mahalanobis distance** — covariance-aware distance for outlier gating. [04 §2](04-gps-for-navigation-and-sensor-fusion.md)
- **DVL (Doppler Velocity Log)** — acoustic sensor giving body-frame velocity (here, resolved 3-D);
  loses **bottom-lock** over soft/steep terrain → dropouts.
- **Dead-reckoning** — integrating velocity/acceleration to get position; drifts without absolute fixes.
- **SDE (stochastic differential equation)** — differential equation driven by white noise; a temporal
  GP's state-space form. [05](05-state-space-gps-and-the-kalman-duality.md)
- **State-space GP / Kalman duality** — Markovian temporal GP ≡ Kalman filter/RTS smoother; O(n).
  [05](05-state-space-gps-and-the-kalman-duality.md)
- **Precision matrix `Q = Σ⁻¹`** — inverse covariance; sparse for Markov processes (block-tridiagonal
  in time, neighbour-sparse in space). [05 §3](05-state-space-gps-and-the-kalman-duality.md), [06 §3](06-spde-approach-the-pde-duality.md)
- **SPDE (stochastic PDE)** — PDE with random forcing; `(κ²−Δ)^{α/2}X=𝒲` has a Matérn solution — the
  PDE↔GP duality. [06](06-spde-approach-the-pde-duality.md)
- **Laplacian `Δ`** — the second-order operator `Σ∂²/∂s²`; the "second-order PDE" in the duality.
- **GMRF (Gaussian Markov Random Field)** — Gaussian field with sparse precision; the FEM solution of
  the SPDE. [06](06-spde-approach-the-pde-duality.md)
- **FEM (finite element method)** — mesh-and-basis discretization that turns the SPDE into a GMRF.
- **Advection / diffusion** — transport by a velocity field / spreading by mixing; the physics encoded
  in the current-field SPDE. [07 §2](07-physics-informed-gps-and-boundary-conditions.md)
- **Dirichlet / Neumann / Robin boundary conditions** — fix value / normal-derivative / a mix on the
  domain boundary. [07 §3](07-physics-informed-gps-and-boundary-conditions.md)
- **Inducing points (FITC/SVGP)** — `m≪n` summary points for approximate large-scale GPs.
  [08 §2d](08-cheap-edge-compute-for-gps.md)
- **GPyTorch** — PyTorch-based GP library (autodiff hyperparameters, multitask kernels, SVGP); our
  demo-build tool.

## Annotated references

### GPs for underwater / AUV navigation
- **Cohen & Klein (2025), *Gaussian Process Regression for Improved Underwater Navigation*** —
  arXiv:2502.16510. *Read for:* the exact pattern we follow — MOGPR on DVL feeding adaptive `Rₖ` into
  an error-state EKF; ~20–42% gains; composite ARD-SE+Matérn3/2+RQ kernel; real-time cost caveat.
  <https://arxiv.org/abs/2502.16510>
- **Position Estimation for UVs using UKF with GP Prediction (GP-UKF)** — *Read for:* learning the
  motion/observation model with a GP inside a UKF; graceful degradation.
- **Ko & Fox, *GP-BayesFilters*** — *Read for:* the foundational GP-in-Bayes-filter framework.

### Continuous-time trajectory estimation / state-space GPs
- **Barfoot, Tong, Anderson (2014), *Batch Continuous-Time Trajectory Estimation as Exactly Sparse GP
  Regression*** — arXiv:1412.0630. *Read for:* the block-tridiagonal precision, white-noise-on-
  acceleration prior, O(n) trajectory estimation, query/interpolation. <https://arxiv.org/abs/1412.0630>
- **Särkkä, Solin, Hartikainen (2013), *Spatiotemporal learning via infinite-dimensional Bayesian
  filtering and smoothing*** — *Read for:* the kernel↔SDE↔Kalman equivalence and O(n) temporal GPs.
- **Särkkä & Solin, *Applied Stochastic Differential Equations* (2019, free)** — *Read for:* the math
  of turning kernels into SDEs. <https://users.aalto.fi/~ssarkka/>
- **Hartikainen & Särkkä (2010), *Kalman filtering and smoothing solutions to temporal GP regression*.*

### The SPDE / PDE duality
- **Lindgren, Rue, Lindström (2011), *An explicit link between Gaussian fields and Gaussian Markov
  random fields: the SPDE approach*, JRSS-B 73(4)** — *Read for:* the core duality and the FEM→GMRF
  sparse-precision construction.
  <https://portal.research.lu.se/en/publications/an-explicit-link-between-gaussian-fields-and-gaussian-markov-rand/>
- **Whittle (1963)** — *Read for:* the original Matérn↔SPDE observation.
- **Bakka et al. (2018), *Spatial modelling with INLA: a review*** — *Read for:* an accessible tour of
  SPDE/GMRF modelling and the R-INLA tooling.
- **Lindgren, Bolin, Rue (2022), *The SPDE approach ... 10 years and still running*, Spatial
  Statistics** — *Read for:* the modern state of the field.

### Physics-informed GPs and boundary conditions
- **Clarotto, Allard, Romary, Desassis (2022/2024), *The SPDE approach for spatio-temporal datasets
  with advection and diffusion*, Spatial Statistics** — arXiv:2208.14015. *Read for:* the
  advection–diffusion SPDE (currents), nonseparability, streamline-diffusion stabilization.
  <https://arxiv.org/abs/2208.14015>
- **Gulian / Tan et al. (2024), *Boundary-constrained Gaussian processes for robust physics-informed
  ML of linear PDEs*, JMLR 25** — *Read for:* exact Dirichlet/Neumann/Robin-constrained mean+kernel
  design. <https://www.jmlr.org/papers/v25/23-1508.html>
- **Jidling, Wahlström, Wills, Schön, *Linearly constrained Gaussian processes*** — *Read for:*
  divergence-free / curl-free vector-field kernels (incompressible currents).
- **Physics-informed, boundary-constrained GPR for fluid flow fields (2025), arXiv:2507.17582** —
  *Read for:* a worked fluid-reconstruction example with incompressibility + boundaries.

### Scalable / sparse GPs (edge compute)
- **Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning* (free)** — *Read for:*
  everything foundational (chs. 2–5). <https://gaussianprocess.org/gpml/>
- **Quiñonero-Candela & Rasmussen (2005)** — *Read for:* a unifying view of sparse approximate GPs
  (SoR/FITC/DTC).
- **Hensman, Fusi, Lawrence (2013), *Gaussian Processes for Big Data*** — *Read for:* SVGP / variational
  sparse GPs.
- **Wilson & Nickisch (2015), *KISS-GP*** — *Read for:* structured-kernel near-linear inference on grids.
- **GPyTorch (Gardner et al., 2018) + docs** — *Read for:* the library we'll use; multitask GPs, SVGP,
  exact GP tutorials. <https://gpytorch.ai/>

## Further reading

- Duvenaud, *The Kernel Cookbook* — kernel-composition intuition. <https://www.cs.toronto.edu/~duvenaud/cookbook/>
- A Visual Exploration of Gaussian Processes (distill.pub) — interactive intuition.
- Back to the [overview](00-overview-and-reading-map.md) or on to the
  [implementation plan](../implementation/00-implementation-plan.md).
