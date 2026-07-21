# 04 — GPs for Navigation and Sensor Fusion

> **What you'll learn:** how dead-reckoning and the error-state EKF work in brief, the concrete ways a
> GP improves DVL/IMU fusion, and a walkthrough of the published underwater-GP results and what
> transfers to *our* sensor setup.
> **Prereqs:** [01–03](01-gaussian-processes-fundamentals.md).
> **Why it matters here:** this is the bridge from "what is a GP" to "what we actually build."

## 1. The underwater navigation problem, briefly

With no GPS, an AUV estimates pose by **dead-reckoning**: integrate IMU specific force/angular rate
(at ~100 Hz) for a high-rate but **drifting** estimate, and correct it with **DVL** body-frame
velocity (~1 Hz) and a **magnetometer** heading. This repo does exactly that with an **error-state
(indirect) Extended Kalman Filter** in `dvl_correction/src/dvl_imu_kalman.py`.

### Error-state EKF in one paragraph

Rather than filtering the full nonlinear state, the error-state EKF tracks a small **error** between
the true state and the IMU-propagated nominal state. IMU samples **propagate** the nominal state and
grow the error covariance; DVL/mag measurements **update** (shrink) it. Our filter's 16-element error
state is `[position(3), velocity(3), attitude(3), gyro-bias(3), accel-bias(3), declination(1)]`. Each
measurement update needs three things: a **residual** (measurement − prediction), a measurement model
**H**, and a **measurement-noise covariance R**. *The GP's job is to supply better DVL velocity
values and, crucially, a better, time-varying R.* (Code anchors in
[`implementation/01`](../implementation/01-codebase-architecture-and-seams.md).)

## 2. Four ways a GP improves DVL/IMU fusion

1. **Denoising.** Replace each raw DVL velocity with the GP posterior **mean** over a recent window.
   The GP smooths sensor noise using temporal (and cross-axis) correlation, so the EKF ingests a
   cleaner measurement.
2. **Adaptive measurement noise `Rₖ`.** Feed the GP's predictive **covariance** as the EKF's `R` for
   that update — *per measurement*, instead of a fixed constant. When the GP is confident, `R` is
   small and the filter leans on the DVL; when not (sparse/odd data), `R` grows and the filter
   protects itself. **This is the single highest-value idea** and the core of the published gains.
3. **Outlier rejection.** Score each incoming DVL sample by its **Mahalanobis distance** to the GP's
   predictive band; reject the gross ones before they corrupt the filter. (Our EKF already has a
   Mahalanobis gate; the GP adds a second, smarter pre-filter.)
4. **Dropout bridging.** During DVL silence, **predict** velocity from the GP at the expected DVL
   cadence, with covariance that **grows with the gap length**. The filter keeps getting velocity
   updates but trusts them less and less — a smooth fallback to IMU-only, rather than a hard switch.

All four use only the GP's two outputs (mean, covariance) from [doc 03](03-gp-regression-and-inference.md).

## 3. Why "predicted covariance as Rₖ" is principled, not a hack

A Kalman update is optimal **only if `R` reflects the true measurement uncertainty.** A fixed `R`
is wrong whenever conditions change (which underwater they always do). The GP's predictive variance
*is* a calibrated estimate of how uncertain this particular velocity value is, given the recent data
and learned dynamics. Plugging it in makes the filter's uncertainty bookkeeping honest — exactly what
the error-state EKF assumes. This is why the approach improves not just velocity but downstream states
like orientation: better-weighted velocity updates propagate into a better attitude estimate.

## 4. What the literature reports

- **Cohen & Klein, *GPR for Improved Underwater Navigation* (2025).** A **multi-output GP regression
  (MOGPR)** estimates DVL velocity and its covariance, fed into an error-state EKF as **velocity
  measurement + adaptive `Rₖ`** (loosely-coupled). Reported ≈ **20% velocity-RMSE** reduction and up
  to **42% orientation** improvement on field trajectories. Kernel: a **sum of ARD squared-exponential
  + Matérn-3/2 + rational-quadratic**, trained by gradient descent (Adam). Honest caveat: MOGPR
  stores the full training set and inverts a large matrix — **expensive in real time** (motivates our
  windowing + edge plan).
- **GP-UKF (Ko & Fox lineage).** Use a GP to *learn the motion/observation model* and run it inside
  an Unscented Kalman Filter. Better tracking and graceful degradation than a fixed parametric model
  when dynamics are hard to model analytically.
- **GP for outlier-robust filtering.** GPR regresses "pseudo-measurements" to replace detected
  outliers in changeable marine environments (robustness under disturbance).

Full citations: [doc 09](09-glossary-and-annotated-references.md).

## 5. What transfers to *our* setup (and one key difference)

| Aspect | Cohen & Klein 2025 | This repo |
|---|---|---|
| GP **inputs** | raw **4-beam** radial velocities | **already-resolved 3-D velocity** (beams are only validity flags) |
| GP **outputs** | 3-D DVL velocity + covariance | same (3-D body-frame velocity + 3×3 covariance) |
| **Fusion** | error-state EKF, adaptive `Rₖ` | identical pattern; our EKF already accepts a per-measurement `covariance_body` |
| Kernel | ARD-SE + Matérn-3/2 + RQ | same composite, made **multitask** for cross-axis correlation |

The one real difference: **our DVL exposes resolved velocity, not raw beams.** So our GP is a
**temporal** model `time → velocity` (denoise/bridge/outlier), rather than a beam→velocity map. This
is *simpler* and still delivers denoising, adaptive `R`, outlier rejection, and—uniquely—**gap
bridging**, which the resolved-velocity temporal form is especially good at. If raw beams are exposed
later, the beam→velocity MOGPR can be added as an input-space variant.

## 6. Loosely- vs tightly-coupled (our choice)

- **Loosely coupled** (our plan): the GP produces a velocity measurement + `R`; the EKF treats it as
  it already treats DVL. Minimal, low-risk, **no EKF-core changes**.
- **Tightly coupled:** fold GP states into the filter (e.g. GP latent force). More powerful, much more
  invasive. Out of scope for the first build.

## 7. Failure modes to respect

- **Garbage in, garbage out:** if the window is full of outliers, the GP learns them. Mitigate with
  the EKF gate + GP outlier pre-filter + robust hyperparameter bounds.
- **Over-smoothing:** too-large lengthscale erases real manoeuvres. The learned lengthscale and the
  Matérn-3/2 choice guard against this; validate on the benchmark.
- **Over-confident bridging:** if the predictive variance grows too slowly, the filter over-trusts a
  long-bridged value. Cap bridging duration (`max_bridge_s`) and verify variance growth on the
  benchmark ([`implementation/03`](../implementation/03-benchmark-and-frontend-demo.md)).

## Key takeaways

- A GP improves DVL/IMU fusion four ways: **denoise, adaptive `Rₖ`, outlier rejection, gap bridging**.
- **Adaptive `Rₖ`** (predicted covariance as measurement noise) is the principled core and the source
  of published gains; it even improves orientation downstream.
- Published results (~20–42% gains) use the same EKF + adaptive-`R` pattern we plan; the only
  difference is we model **resolved** velocity over time (our DVL gives no raw beams), which is
  simpler and excels at **bridging**.
- We use the **loosely-coupled** form: **no EKF-core changes**.

## Further reading

- Cohen & Klein 2025 (arXiv:2502.16510); Ko & Fox, *GP-BayesFilters*; see [doc 09](09-glossary-and-annotated-references.md).
- Next: [05 — State-space GPs and the Kalman duality](05-state-space-gps-and-the-kalman-duality.md).
