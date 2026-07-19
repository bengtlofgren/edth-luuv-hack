# 07 — Physics-Informed GPs and Boundary Conditions

> **What you'll learn:** how to put real physics (advection–diffusion / ocean currents) and exact
> boundary conditions (Dirichlet/Neumann/Robin) into a GP, via both the SPDE operator and
> linear-PDE-constrained kernels.
> **Prereqs:** [06 — the SPDE approach](06-spde-approach-the-pde-duality.md).
> **Why it matters here:** this is the concrete "incorporate physical processes and boundary
> conditions" ask; it underpins the future ocean-current-field model.

## 1. Two ways to make a GP obey physics

There are two complementary routes, both reducing to "constrain the GP with a **linear** differential
operator":

- **A. The SPDE-operator route** ([doc 06](06-spde-approach-the-pde-duality.md)): change the operator
  on the left of `L X = 𝒲` to encode the dynamics. The field's covariance then *inherits* the physics.
- **B. The constrained-kernel route:** start from a standard GP and **transform its kernel** so every
  sample exactly satisfies a linear PDE and/or boundary condition. If `f ~ GP(0, k)` and `g = 𝒟f` for a
  linear operator `𝒟`, then `g` is also a GP with kernel `𝒟 𝒟′ k`. So differential constraints map to
  kernel transformations — physics enters analytically.

Both work because **Gaussianity is preserved under linear operations** (differentiation, integration,
linear PDE operators). Nonlinear physics needs approximation, but a lot of ocean transport is well
captured by *linear* advection–diffusion.

## 2. The advection–diffusion SPDE (ocean currents)

The physical process most relevant to an AUV's drift is **advection–diffusion**: a quantity is
**transported** by the current velocity field **and** **diffused** by mixing. The spatio-temporal SPDE
(Clarotto, Allard, Romary, Desassis, 2022/2024) is:

```
∂X/∂t  +  (γ·∇)X  −  ∇·(D ∇X)  +  κ² X   =  𝒲(s, t)
        └─ advection ┘  └─ diffusion ┘   └ damping ┘   └ random forcing ┘
```

- **`γ`** — the **advection (drift) velocity** vector. Set it to the local current and the field's
  correlations **tilt and travel downstream** — features are carried along the flow, exactly like
  dye in a current. This is how the *current itself* enters the prior.
- **`D`** — the **diffusion tensor** (can be anisotropic): how fast structure smears out and in which
  directions (e.g. stronger horizontal than vertical mixing).
- **`κ²`** — damping/range term (as in [doc 06](06-spde-approach-the-pde-duality.md)).
- First-order `∂/∂t` makes it a genuinely **nonseparable spatio-temporal** model — space and time
  correlations are coupled, which separable kernels cannot represent.

### Why this beats a plain space×time kernel

A naive spatio-temporal GP often uses a *separable* kernel `k_space · k_time`, which assumes the
spatial pattern is frozen and just fades. Advection–diffusion produces **transported, evolving**
correlations — physically correct for currents. That structure is the payoff of going physics-informed.

### A numerical wrinkle: streamline diffusion

When advection dominates diffusion (fast current, little mixing), naive FEM discretization becomes
unstable/oscillatory. The standard fix is **streamline-diffusion (SUPG) stabilization** — add a small
amount of diffusion along the flow direction. Worth knowing because it's a required ingredient, not an
optional tweak, in advection-dominated regimes (Clarotto et al. discuss exactly this).

## 3. Boundary conditions — making the GP respect the domain

A PDE needs boundary conditions to be well-posed, and they are how the GP learns about **walls, the
seabed, the surface, and known values**. The three classical types:

| Type | Constrains | Underwater example |
|---|---|---|
| **Dirichlet** | the **value** on the boundary: `X = g` | known current at an inflow boundary; zero anomaly at a reference |
| **Neumann** | the **normal derivative**: `∂X/∂n = h` | **no flow through the seabed** (zero normal velocity) — a hard physical fact |
| **Robin** | a mix: `aX + b ∂X/∂n = c` | partially absorbing / leaky boundaries |

Two ways to enforce them:

- **In the SPDE/FEM mesh:** impose the condition on boundary nodes during assembly — exact and natural
  once you have a mesh. The choice of boundary condition genuinely changes the field near edges
  (a Neumann "no-flow" seabed makes the modelled current go *along* the bottom, not through it).
- **In the kernel (boundary-constrained GPs):** design mean and covariance so **every** sample
  satisfies the boundary condition by construction. Gulian et al. / Tan (JMLR v25, 23-1508) give
  closed-form **boundary-constrained kernels for Dirichlet, Neumann, Robin and mixed** conditions —
  e.g. expanding the kernel in the **Dirichlet eigenfunctions of the Laplacian** enforces zero-value
  boundaries globally and noiselessly. This route needs no mesh and is attractive on simple domains.

**Why bother:** enforcing physics-correct boundaries (a) removes nonsensical predictions near edges,
(b) sharply reduces uncertainty there (the model "knows" the boundary), and (c) improves
extrapolation into sparsely-observed regions — all valuable when bridging long DVL outages near the
seabed.

## 4. Divergence-free / incompressibility constraints

Sea water is effectively **incompressible**: the current field should be **divergence-free**
(`∇·u = 0`). Using the constrained-kernel route (§1B) you can build a **vector-field kernel** whose
samples are divergence-free by construction (curl-based / stream-function kernels). Physics-informed,
boundary-constrained GP regression for fluid flow fields does exactly this (Jidling et al.; recent
fluid-reconstruction work). For us this would make a reconstructed current field physically consistent,
which improves drift prediction.

## 5. How this would serve the AUV (the payoff)

Put together, a physics-informed current model gives the navigator a **prior over the water it is
moving through**:

- During **DVL bottom-lock loss**, the vehicle still measures **water-relative** velocity (or nothing);
  a current-field GP predicts the **current**, so `ground velocity ≈ water-relative velocity + current`
  can be reconstructed → drift is bounded even without bottom-lock.
- Boundary conditions keep predictions sane near the seabed/surface.
- The SPDE/GMRF form ([doc 06](06-spde-approach-the-pde-duality.md)) keeps it computable, and combined
  with Markov-in-time ([doc 05](05-state-space-gps-and-the-kalman-duality.md)) it can run as an
  efficient spatio-temporal filter.

## 6. Scope and honesty

This is the **most ambitious** thread and is **explicitly future work**
([`implementation/04`](../implementation/04-edge-and-spde-roadmap.md)). It needs: a meshing/FEM or
constrained-kernel implementation, sparse linear algebra (`scipy`-class), an estimate of the current
field (`γ`) from data or a model, and validation data. The first build deliberately stays with the
simpler, high-value temporal velocity GP.

## Key takeaways

- Physics enters a GP through a **linear operator** — either by editing the **SPDE operator** (doc 06)
  or by **transforming the kernel** (`g=𝒟f ⇒ k_g = 𝒟𝒟′k`); Gaussianity survives linear operations.
- **Advection–diffusion** encodes ocean currents: `γ` (drift) tilts/transports correlations, `D`
  (diffusion) smears them — a true nonseparable spatio-temporal model (needs streamline-diffusion
  stabilization when advection dominates).
- **Boundary conditions** (Dirichlet/Neumann/Robin) make the GP respect seabed/surface/walls exactly,
  via the FEM mesh or **boundary-constrained kernels**; **divergence-free** kernels enforce
  incompressibility.
- Payoff: a physics-correct **current-field prior** that bounds drift during bottom-lock loss — future
  work, but the principled end-state.

## Further reading

- Clarotto et al. (2022/2024), *The SPDE approach for spatio-temporal datasets with advection and
  diffusion*, arXiv:2208.14015.
- Gulian/Tan et al. (2024), *Boundary-constrained Gaussian processes for ... linear PDEs*, JMLR
  v25/23-1508.
- Jidling et al., *Linearly constrained Gaussian processes* (divergence-free / curl-free kernels).
- Next: [08 — Cheap edge compute for GPs](08-cheap-edge-compute-for-gps.md).
