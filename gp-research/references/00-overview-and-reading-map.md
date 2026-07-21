# 00 — Overview and Reading Map

> **What you'll learn:** what a Gaussian Process is in one paragraph, why it suits underwater
> positioning, and how the rest of these docs fit together.
> **Prereqs:** none.
> **Why it matters here:** orientation before the detail.

## A Gaussian Process in one paragraph

A **Gaussian Process (GP)** is a probability distribution over *functions*. Instead of fitting a
single curve `f(x)`, a GP says: "any finite set of function values
`f(x₁), …, f(xₙ)` is jointly Gaussian," fully specified by a **mean function** `m(x)` (often 0) and
a **covariance/kernel function** `k(x, x′)` that says how correlated two outputs are based on how
close their inputs are. Given some noisy observations, conditioning this joint Gaussian on the data
yields a **posterior**: at any new input `x*` you get a predicted **mean** *and* a predicted
**variance**. The variance is small near data and grows as you move away — built-in, honest
uncertainty. That uncertainty is the property that makes GPs valuable for sensor fusion.

## Why this is the right tool underwater

Underwater navigation has three painful facts:

1. **No GPS.** Position is obtained by integrating velocity/acceleration → unbounded drift.
2. **The DVL drops out and lies.** The Doppler Velocity Log is the workhorse velocity sensor, but it
   loses bottom-lock (soft mud, steep slopes, altitude limits, bubbles) and emits outliers.
3. **Everything is uncertain and asynchronous.** IMU at ~100 Hz, DVL at ~1 Hz, magnetometer
   sporadically; each with its own noise.

A GP addresses all three: it produces a **continuous, uncertainty-aware** estimate of velocity (or
position) over time, so it can **denoise** good DVL, **reject** outliers, and **bridge** gaps with a
variance that grows — telling the downstream Kalman filter exactly how much to trust the bridged
value. And through the SPDE duality (doc 06) it can be taught **physics** and **boundary conditions**;
through the Kalman duality (doc 05) it can run **cheaply on the vehicle** (doc 08).

## The three themes the user asked about

- **GPs for AUV navigation** — the practical "how does it plug in and help" — doc 04, plus all of
  `implementation/`.
- **The second-order-PDE / SPDE duality** — "incorporate physical processes and boundary
  conditions" — docs 06 and 07. This is the deep idea: a Matérn-covariance GP *is the solution of a
  stochastic PDE*, so changing the PDE changes the physics the GP assumes.
- **Cheap edge compute** — docs 05 and 08. The key trick: a temporal GP with the right kernel is
  *mathematically identical* to a Kalman filter, which is O(n) and tiny.

## Dependency-ordered reading map

```
        ┌────────────────────────── 01 GP fundamentals ──────────────────────────┐
        │                                                                          │
        ▼                                                                          ▼
  02 Kernels & covariance                                              03 GP regression & inference
        │                                                                          │
        └───────────────────────────────┬──────────────────────────────┬─────────┘
                                         ▼                              ▼
                          04 GPs for navigation & fusion        05 State-space GP ↔ Kalman
                                         │                              │
                  ┌──────────────────────┼───────────────┐            │
                  ▼                      ▼                ▼            ▼
        06 SPDE / PDE duality   07 Physics-informed   (impl 00–03)   08 Cheap edge compute
                  │                & boundary conds                    │
                  └───────────────────────┬───────────────────────────┘
                                          ▼
                                09 Glossary & references
```

- **Just want the build?** Read 01 → 03 → 04, then `implementation/`.
- **Care most about the PDE duality?** Read 01 → 02 → 06 → 07.
- **Care most about edge compute?** Read 01 → 02 → 05 → 08.

## How the math is written here

Equations are plain text/Markdown with backticks, e.g. `k(x,x′) = σ² exp(-(x-x′)²/(2ℓ²))`. Vectors
are bold in prose ("**x**"), matrices are capitals (`K`). Where a symbol matters, it is defined
inline and collected in [doc 09's glossary](09-glossary-and-annotated-references.md).

## Key takeaways

- A GP = distribution over functions = mean + kernel; its output is a **mean and a variance**.
- The variance (uncertainty) is *why* it helps sensor fusion: it tells the Kalman filter how much to
  trust each value, including bridged values during DVL dropouts.
- Two dualities unlock the advanced asks: **GP ↔ Kalman** (cheap edge) and **GP ↔ SPDE** (physics +
  boundary conditions).

## Further reading

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (free online) — the canonical text.
- Continue to [01 — GP fundamentals](01-gaussian-processes-fundamentals.md).
