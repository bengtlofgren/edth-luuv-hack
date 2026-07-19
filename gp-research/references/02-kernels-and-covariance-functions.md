# 02 — Kernels and Covariance Functions

> **What you'll learn:** the common kernels (RBF, Matérn, Rational Quadratic, periodic), what their
> hyperparameters do, how to combine kernels, and multi-output/multitask kernels for correlated
> x/y/z velocity.
> **Prereqs:** [01 — GP fundamentals](01-gaussian-processes-fundamentals.md).
> **Why it matters here:** the kernel encodes *every assumption* the GP makes; our velocity model
> uses a specific composite, multitask kernel.

## 1. What a kernel is and must be

The kernel `k(x, x′)` defines the covariance between outputs at two inputs. It controls smoothness,
amplitude, and how far correlations reach. To be valid, a kernel must be **positive semi-definite**
(every Gram matrix `K` it produces must be PSD) — this is what guarantees the joint distribution is a
legal Gaussian. The good news: you rarely build kernels from scratch; you compose known-valid ones.

Two near-universal hyperparameters:

- **Variance / signal amplitude `σ²`** — vertical scale of the function (how big the wiggles are).
- **Lengthscale `ℓ`** — horizontal scale (how far apart inputs must be before outputs decorrelate).
  Small `ℓ` → wiggly, fast-changing; large `ℓ` → smooth, slowly-changing. This is usually the most
  important knob.

## 2. The squared-exponential (RBF) kernel

```
k_SE(x,x′) = σ² · exp( − (x−x′)² / (2ℓ²) )
```

The default smooth kernel. Samples are **infinitely differentiable** — extremely smooth. That is
often *too* smooth for physical signals (real motion is not infinitely smooth), which is why Matérn
is usually preferred for navigation.

## 3. The Matérn family (the workhorse for physical signals)

```
k_Matérn,ν(r) = σ² · (2^{1−ν}/Γ(ν)) · (√(2ν) r/ℓ)^ν · K_ν(√(2ν) r/ℓ),   r = |x−x′|
```

The smoothness parameter **ν** controls how many times sample paths are differentiable. Three values
have closed forms and dominate practice:

- **Matérn-1/2** (`ν=1/2`): `k = σ² exp(−r/ℓ)`. This is the **exponential / Ornstein–Uhlenbeck**
  kernel — continuous but non-differentiable (rough). Equivalent to a 1st-order Markov process.
- **Matérn-3/2** (`ν=3/2`): `k = σ² (1 + √3 r/ℓ) exp(−√3 r/ℓ)`. **Once differentiable.** A very
  common, robust default for real-world signals — smooth enough to look physical, rough enough not to
  over-smooth. **This is the backbone of our velocity kernel.**
- **Matérn-5/2** (`ν=5/2`): `k = σ² (1 + √5 r/ℓ + 5r²/3ℓ²) exp(−√5 r/ℓ)`. **Twice differentiable.**
  Good when you believe the signal has a meaningful acceleration.

As `ν → ∞`, Matérn → RBF. **Why this matters for us:** vehicle velocity is a physical quantity with
finite smoothness; Matérn-3/2 or -5/2 models it far better than the unrealistically smooth RBF.
Matérn also has a deep second role — it is the kernel that connects GPs to **SPDEs** and to
**state-space/Kalman** forms (docs 05 and 06). Choosing Matérn now keeps both efficiency doors open.

## 4. Rational Quadratic (RQ)

```
k_RQ(x,x′) = σ² · (1 + (x−x′)²/(2αℓ²))^{−α}
```

An infinite mixture of RBF kernels with different lengthscales (parameter `α` sets the spread). It
captures variation happening at **several scales at once** — useful when the signal has both
slow trends and faster fluctuations. As `α→∞` it reduces to RBF.

## 5. Periodic and other structural kernels

```
k_per(x,x′) = σ² · exp( −2 sin²(π|x−x′|/p) / ℓ² )
```

Encodes repetition with period `p` — e.g. a lawnmower survey pattern or a periodic disturbance.
Probably not needed for the first velocity model but worth knowing for richer trajectory models.

## 6. ARD — automatic relevance determination

When the input is multi-dimensional (`x ∈ ℝ^d`), give each dimension **its own lengthscale**:

```
k(x,x′) = σ² · exp( −½ Σ_d (x_d − x′_d)² / ℓ_d² )
```

If dimension `d` is irrelevant, training drives `ℓ_d → ∞` and that input is effectively ignored —
the model *learns which inputs matter*. Relevant if we later feed the GP extra features (attitude,
specific force) alongside time.

## 7. Combining kernels (the composable algebra)

Valid kernels are closed under **addition** and **multiplication**:

- **Sum** `k₁ + k₂`: the function is a sum of independent components (e.g. a smooth trend **plus**
  multi-scale fluctuation). Intuitively "OR".
- **Product** `k₁ · k₂`: components must *both* be active (e.g. periodic **and** slowly decaying).
  Intuitively "AND".

**Our velocity kernel is a sum** `Matérn-3/2 + RQ + RBF` over time: Matérn-3/2 supplies physically
plausible local smoothness, RQ captures multi-scale variation, RBF a smooth long trend. This mirrors
the composite-ARD choice reported in the underwater-navigation GP literature (Cohen & Klein 2025,
see [doc 09](09-glossary-and-annotated-references.md)). It is intentionally expressive because this
round prioritizes accuracy over compute.

## 8. Multi-output / multitask kernels (correlated x, y, z)

The DVL gives a **3-vector** velocity `(vx, vy, vz)` in the body frame, and the axes are correlated
(the vehicle moves as one rigid body). Modelling them as three independent GPs throws that away.
**Multitask / multi-output GPs** model all outputs jointly.

The standard construction is the **Linear Model of Coregionalization (LMC)** and its special case the
**Intrinsic Coregionalization Model (ICM)**:

```
k((x,i),(x′,j)) = B[i,j] · k_time(x, x′)
```

- `k_time` is the shared temporal kernel from §7.
- `B` is a small (here `3×3`) PSD **coregionalization matrix** of *learned* inter-axis correlations.

So "how correlated is vx at time t with vz at time t′" = `B[x,z]` × (temporal similarity). ICM uses
one shared `k_time`; LMC sums several such terms with different kernels for richer cross-axis
structure. GPyTorch implements these as `MultitaskKernel` (ICM) / `LCMKernel` (LMC). Our model uses a
multitask kernel so the GP exploits the rigid-body coupling between velocity axes — the same idea
that lets multi-output GP regression beat per-axis fits in the published results.

## 9. Picking and reading hyperparameters

After training, the learned values are interpretable diagnostics:

- **Lengthscale** ≈ the timescale over which velocity stays correlated. If it learns "0.5 s", the GP
  can only safely bridge gaps much shorter than that before its variance balloons — directly informs
  the dropout-bridge thresholds in [`implementation/02`](../implementation/02-gp-velocity-module-design.md).
- **Signal variance** ≈ how much the velocity actually varies — sanity-check against physics.
- **Noise `σₙ²`** ≈ the DVL's effective measurement noise — should be in the ballpark of the sensor
  spec / configured `default_dvl_velocity_std_m_s`.

## Key takeaways

- The kernel encodes all assumptions: smoothness (Matérn ν), amplitude (σ²), reach (ℓ), multi-scale
  (RQ), periodicity, per-input relevance (ARD).
- **Matérn-3/2** is our smoothness backbone — physically realistic *and* the gateway to the SPDE and
  Kalman dualities.
- Kernels **add (OR)** and **multiply (AND)**; our model is a **sum** of Matérn-3/2 + RQ + RBF.
- **Multitask (LMC/ICM)** kernels model correlated `(vx,vy,vz)` jointly via a learned `3×3` matrix —
  better than three independent GPs.

## Further reading

- Rasmussen & Williams, *GPML*, ch. 4 (covariance functions).
- Duvenaud, *The Kernel Cookbook* (online) — excellent intuition for composing kernels.
- Álvarez, Rosasco, Lawrence, *Kernels for Vector-Valued Functions* — multi-output GPs.
- Next: [03 — GP regression and inference](03-gp-regression-and-inference.md).
