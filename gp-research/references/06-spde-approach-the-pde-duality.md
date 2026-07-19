# 06 — The SPDE Approach (the PDE Duality)

> **What you'll learn:** the exact sense in which a Matérn-covariance GP **is** the solution of a
> second-order stochastic PDE, why that operator is the place to inject physics, and how the
> finite-element/GMRF construction turns a dense covariance into a **sparse precision** matrix.
> **Prereqs:** [01](01-gaussian-processes-fundamentals.md), [02](02-kernels-and-covariance-functions.md);
> [05](05-state-space-gps-and-the-kalman-duality.md) gives the temporal analogue.
> **Why it matters here:** this is the duality you asked about — the route to physical processes,
> boundary conditions, and cheap compute, all from one equation.

## 1. The duality in one equation

Whittle (1963) and Lindgren–Rue–Lindström (2011) showed: a Gaussian field with **Matérn covariance**
is the **stationary solution** of the linear **stochastic partial differential equation (SPDE)**

```
(κ² − Δ)^{α/2}  X(s)  =  𝒲(s)
```

- `Δ` is the **Laplacian** (`Σ ∂²/∂s_d²`) — the second-order differential operator. The name "duality
  of the second-order PDE" points here.
- `κ > 0` sets the **spatial range** (`range ≈ √(8ν)/κ`); larger `κ` → shorter correlations.
- `α = ν + d/2` ties the SPDE order to the Matérn smoothness `ν` and the spatial dimension `d`.
- `𝒲` is **Gaussian white noise** (the random forcing).

Read it as: *"a Matérn GP is what you get when you pass white noise through the operator
`(κ²−Δ)^{−α/2}`."* The covariance function and the differential operator are **two descriptions of the
same object** — the duality. (This is the spatial sibling of [doc 05](05-state-space-gps-and-the-kalman-duality.md):
there a Matérn GP in *time* equaled an *ordinary* SDE; here a Matérn GP in *space* equals a *partial*
SDE.)

## 2. Why this is powerful — three doors from one equation

Because the GP is now defined by an **operator equation** rather than a fixed covariance formula, you
can edit the operator to change the model's assumptions:

1. **Inject physical processes.** Replace/augment `(κ²−Δ)` with operators encoding real dynamics —
   **advection** (transport by currents) and **anisotropic diffusion** (mixing). The resulting field
   automatically has the spatial/temporal correlation structure those physics imply. (Detailed in
   [doc 07](07-physics-informed-gps-and-boundary-conditions.md).)
2. **Impose boundary conditions.** A PDE is only well-posed *with* boundary conditions. Specifying
   **Dirichlet/Neumann/Robin** conditions on the domain boundary makes the GP respect walls, the
   seabed, the surface, or known values — exactly, not approximately. (Also [doc 07](07-physics-informed-gps-and-boundary-conditions.md).)
3. **Compute cheaply.** Solving the SPDE by the **finite element method (FEM)** yields a **Gaussian
   Markov Random Field (GMRF)** with a **sparse precision matrix** — the compute win, below.

## 3. From SPDE to GMRF: the sparse-precision win

The construction (LRL 2011):

1. **Mesh the domain.** Triangulate the region of interest (e.g. the operating area / water column)
   into nodes and elements.
2. **Basis expansion.** Approximate the field as `X(s) ≈ Σ_k w_k φ_k(s)` with local "tent" basis
   functions `φ_k` (each nonzero only near node `k`) and random weights `w`.
3. **Weak form / FEM assembly.** Projecting the SPDE onto this basis gives the weights a Gaussian
   distribution `w ~ N(0, Q⁻¹)` whose **precision matrix `Q`** is built from sparse FEM matrices
   (mass `C` and stiffness `G`), e.g. for `α=2`: `Q = (κ²C + G) C⁻¹ (κ²C + G)`.

Because each basis function overlaps only its neighbours, **`Q` is sparse** (mostly zeros). This is
the **Markov property in space**: a node is conditionally independent of the rest given its
neighbours — the spatial analogue of the block-tridiagonal precision in
[doc 05 §3](05-state-space-gps-and-the-kalman-duality.md).

### Dense covariance vs sparse precision — the cost table

| | Covariance `Σ = K` (classic GP) | Precision `Q = Σ⁻¹` (GMRF/SPDE) |
|---|---|---|
| Structure | **dense** (every point covaries with every point) | **sparse** (only neighbours) |
| Storage | O(n²) | O(n) |
| Factorize / solve | O(n³) | ≈ O(n^{3/2}) in 2-D, O(n) in 1-D, using sparse Cholesky |
| Interpretation | "how outputs correlate" | "conditional dependence graph" |

This is why the SPDE/GMRF approach scales to large spatial fields where a classic dense GP cannot — and
why it is a serious **edge-compute** option ([doc 08](08-cheap-edge-compute-for-gps.md)).

## 4. Non-stationarity for free

Let `κ = κ(s)` and the diffusion vary over space, and the SPDE produces a **non-stationary** field —
different correlation length/strength in different regions — while *keeping* the sparse precision.
Classic stationary kernels can't do this; encoding it in covariance form is hard, but in operator form
it's just spatially-varying coefficients. Underwater this matters: correlation structure near a
seamount differs from open water.

## 5. Where it does and doesn't fit our project

- **Not the first build.** Our first GP is a *temporal* DVL-velocity model
  ([doc 04](04-gps-for-navigation-and-sensor-fusion.md)); it doesn't need a spatial mesh.
- **The natural next step.** Modelling the **ocean-current field** the vehicle moves through is
  inherently spatial (and spatio-temporal). The SPDE/GMRF approach is the right tool: it can encode
  current physics (advection–diffusion), respect boundaries (seabed/surface), stay non-stationary, and
  compute on a sparse precision suitable for on-board use. This is item (2) of the edge/SPDE roadmap in
  [`implementation/04`](../implementation/04-edge-and-spde-roadmap.md).
- **Spatio-temporal = combine docs 05 + 06.** Keep Markov-in-time (Kalman) and GMRF-in-space → an
  efficient current-field filter (Särkkä/Solin infinite-dimensional Kalman; Clarotto et al. for the
  advection–diffusion SPDE specifically).

## 6. Tooling note

The classic implementations live in the R-INLA ecosystem (`R-INLA`, `inlabru`) and `fmesher`;
Python options include `findiff`/FEM libraries plus sparse solvers in `scipy.sparse`. For our roadmap
this would be a *separate* component from the GPyTorch velocity model, justified only when we tackle
the current field. Adding it implies a `scipy`-class dependency for sparse linear algebra and a meshing
step.

## Key takeaways

- **The duality:** a Matérn GP **is** the solution of `(κ²−Δ)^{α/2} X = white noise`. Covariance and
  second-order PDE operator are the same object described two ways.
- Editing the **operator** is how you inject **physics** and impose **boundary conditions**
  (see [doc 07](07-physics-informed-gps-and-boundary-conditions.md)).
- Solving by FEM yields a **GMRF with a sparse precision matrix** → O(n)-ish compute instead of O(n³),
  plus easy **non-stationarity**.
- For us: not the first build, but the principled, scalable way to model the **ocean-current field**
  later.

## Further reading

- Lindgren, Rue, Lindström (2011), *An explicit link between Gaussian fields and Gaussian Markov random
  fields: the SPDE approach*, JRSS-B — **the** paper.
- Whittle (1963) — the original Matérn↔SPDE observation.
- Bakka et al. (2018), *Spatial modelling with INLA: a review* — accessible overview.
- Next: [07 — Physics-informed GPs and boundary conditions](07-physics-informed-gps-and-boundary-conditions.md).
