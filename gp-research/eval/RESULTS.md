# Ensemble re-evaluation — verdicts on the POC claims

Companion to [PLAN.md](PLAN.md). Raw per-gap records, aggregate tables, and
figures live in [`../eval_results/`](../eval_results/) (`snapir_*`,
`solaqua_*`), produced by `../eval_snapir.py` (174 gaps: 58 starts x
{10,30,60} s, ANSFL Snapir 1 Hz) and `../eval_solaqua.py` (31 gaps over 7
SOLAQUA bags, {10,20} s, per-fix `fom` as R).

These sweeps supersede the single-gap numbers quoted in
`../poc/poc_03_real_data.py` and `../poc/poc_06_imu_bridge.py`. The POC
scripts are untouched historical artifacts; where a POC docstring and this
file disagree, this file wins.

## "Adaptive R halves cruise drift" — REFUTED as stated, revised

Causal setting, paired per-gap peak-drift deltas (naive R minus adaptive R):
overall median **+1.4 cm** with a 55% win rate — a marginal drift gain, not a
halving (quiet gaps: +6.3 cm, 60%). The original "halves drift" number came
from one acausal gap.

What adaptive R actually buys is **honesty**: median KF overclaim drops from
1.29x / 2.75x / 4.04x (naive, 10/30/60 s gaps) to 1.11x / 1.70x / 2.31x.
Revised claim: *adaptive R roughly halves the filter's overconfidence during
outages; its drift benefit is marginal.*

## "GP + adaptive R overconfident in maneuvers" — CONFIRMED

Causal-GP 2-sigma coverage: **0.949 on quiet gaps vs 0.766 on dynamic gaps**
(median split on gap dynamic-ness; nominal 0.954). Downstream, even the
adaptive-R KF overclaims 2.32x on dynamic gaps (1.34x quiet). Adaptive R is
the least-bad config but does not fix maneuver overconfidence — the GP cannot
see accelerations that start inside the gap.

## "IMU wins when dynamic" — REFUTED

On the SOLAQUA ensemble the IMU dead-reckoning bridge loses to the causal GP
**even on the dynamic half** of gaps: median +0.77 cm/s worse RMSE, winning
only 19% of dynamic gaps (7% of calm ones). It does not reliably beat ZOH
either (wins 38% dynamic / 33% calm).

Root cause (diagnosed, not a code bug — unit tests pass on synthetic data):
the DVL-frame calibration is unobservable on these bags. The Procrustes scale
collapses to 0.02–0.07 (should be ~1) because the vehicle's DVL-observable
accelerations are small relative to IMU noise, so the learned map attenuates
the IMU signal to near-zero and the "IMU" bridge degenerates toward ZOH with
extra noise. The poc_06 single-gap "IMU wins the fast run" result does not
survive the ensemble. A bag with genuinely aggressive maneuvers (or a factory
frame calibration) could still rescue the concept; these bags cannot show it.

## "Hybrid near-best in both regimes" — SURVIVES (v2)

Hybrid v2 (chi-square gate + covariance intersection) tracks the best single
supplier with **~zero regret** on surge and sway (median 0.00 cm/s vs
min(GP, IMU); v1's regret 0.06–0.09 cm/s), with 99–100% bridge coverage.
Caveat: on heave, v1 edges v2 (v2 regret 0.18 cm/s). Through the KF, hybrid
v2 matches the causal GP as best overall (vRMSE 2.35 cm/s, peak drift
0.16 m, honesty 0.44x — honest). Since the IMU bridge itself adds nothing
here (see the refuted IMU claim above), the hybrid's current value is graceful robustness, not a win over
plain causal GP.

## Control-conditioned dynamics GP (poc_07) — concept validated, blocked by actuation observability on these bags

PILCO-style reformulation attacking the maneuver-overconfidence result at
its root: regress acceleration on state and commanded thrust, a = f(v, u)
(semi-parametric: ridge thrust-gain/damping mean + GP residual), and bridge
an outage by rolling forward with the commands the vehicle still knows.
On a synthetic control system this crushes ZOH by >2x with calibrated
variance (tests/test_dynamics.py) — the machinery is correct.

On the SOLAQUA ensemble it **loses**: median +1.65 cm/s vs the causal GP
(wins 19% of gaps), also behind ZOH, though honest (96% 2-sigma coverage)
and least-bad on the dynamic half (+1.48 cm/s, 25% wins). The diagnostic
says why: the linear thrust->acceleration fit explains a median of only
6% of surge acceleration variance (heave: 32%). On this tethered,
net-following ROV, tether drag, currents and the net's wake dominate the
DVL-observable acceleration; the commands barely do. Same observability
wall as the IMU bridge, one level up: the *inputs* are known perfectly,
but their effect on the vehicle is too weak to identify from these bags.

Implication: control-conditioned bridging is the right structure (the
maneuver becomes an observed input, uncertainty compounds through model
ignorance instead of elapsed time), but it needs a vehicle whose commands
actually explain its acceleration — an untethered AUV, or bags with
authoritative maneuvers. Keep the supplier; gate it on the calibration
R^2 the same way the IMU leg is gated on Procrustes scale.

## New finding — causal GP vs ZOH on Snapir

Causal GP extrapolation barely beats zero-order hold on RMSE (median 0.239 vs
0.227 m/s; ZOH wins 49% of gaps paired). Its real value is **calibrated
uncertainty**: 86% 2-sigma coverage vs ZOH's 45%, which is what makes the
adaptive-R seam honest. Acausal (post-mission smoothing) GP is a different
story: it beats the causal GP on 76% of gaps and linear interpolation remains
a strong cheap baseline (beats causal GP 68%).

## Practical takeaways for `gp_velocity` (implementation/02)

1. Ship the causal GP for its variance, not its mean — the mean is
   ZOH-with-decay; the honest, growing sigma is the product.
2. Adaptive R: keep. Sell it as an honesty feature, not a drift feature.
3. IMU bridging: blocked on frame-calibration observability. Do not fuse an
   IMU through a data-learned map whose scale is unidentifiable; require a
   known extrinsic or an observability check (Procrustes scale near 1) before
   trusting the IMU leg.
4. Hybrid v2 is the right fusion shape (gate + covariance intersection) and
   costs nothing when the IMU leg is useless; keep it as the integration
   point for a future, properly calibrated IMU.
5. Every "smarter" bridge tried so far (IMU-DR, dynamics GP) fails on the
   same axis: the extra information channel is not observable on this data
   (frame scale, thrust authority). The general rule for gp_velocity: any
   auxiliary leg needs an online observability check (Procrustes scale ~ 1,
   dynamics R^2 above a floor) and must degrade to the causal GP when the
   check fails.

## Dagon basin data — does the dynamics GP win when thrust explains motion?

The SOLAQUA verdict left the dynamics GP with an untested alibi: it lost
because commanded thrust explained only ~6% of that tethered ROV's surge
acceleration, not because the supplier is wrong. The DFKI Dagon dataset
(eval_dagon.py; loader provenance and the empirical 4 Hz sample-rate
determination in eval/data.py::load_dagon) is the other side of the natural
experiment: an untethered AUV in a saltwater basin, driven in the horizontal
plane by sinusoidal thruster system-ID excitation — a vehicle whose commands
should genuinely explain its motion. 92 sliding gaps (46 x 10 s + 46 x 20 s
over the 2892 s record); suppliers: causal time-GP (trailing 60 s window),
dyn_gp (trained causally on all pre-gap data, rolled forward with the
commanded thrust), ZOH. Axes surge/sway [m/s] and yaw rate [rad/s], never
pooled.

The alibi holds and the supplier delivers. The observability diagnostic
flips exactly as predicted: the ridge thrust-gain/damping mean alone
explains a median 49% of surge acceleration (SOLAQUA: 6%), 26% of sway, 67%
of yaw — and with that observability the dynamics GP **wins every axis,
decisively**. Median bridge RMSE (20 s gaps): surge 4.0 cm/s vs 15.0 (time-GP)
and 20.7 (ZOH); sway 3.4 vs 7.1 and 9.4 cm/s; yaw rate 62 vs 297 and
359 mrad/s. Paired per gap it beats the time-GP on 86% (surge), 85% (sway),
90% (yaw) of gaps and ZOH on 85-90%, with the margin largest on the dynamic
half (surge: median -22.2 cm/s vs ZOH, 98% wins) but still overwhelming on
the calm half (78-89% wins) — through command-driven maneuvers there is no
regime where the time-GP is preferable here. The failure mode RESULTS.md
opened with — maneuvers structurally unobservable during an outage — is
gone: the maneuver arrives as an observed input (see the example gap in
eval_results/dagon_eval.png, where dyn_gp tracks a full sinusoidal reversal
mid-gap that the time-GP flattens through).

The semi-parametric split also pulls its weight: at the true hidden in-gap
states, adding the GP residual to the ridge mean cuts one-step acceleration
residual RMS roughly in half on every axis (median ratio 0.45 surge, 0.46
sway, 0.50 yaw) — the GP is learning real unmodeled dynamics (quadratic
drag, thruster asymmetries), not noise, even where the linear model is
already good. Uncertainty is the one soft spot: dyn_gp's 2-sigma bridge
coverage is 80/76/86% per axis against the time-GP's 93/93/87% — the
first-order variance rollout (no input-uncertainty propagation through the
state feedback) is mildly overconfident once its mean is this accurate;
acceptable, but short of the nominal 95%.

Verdict: the SOLAQUA loss was an observability failure of that platform, not
of the method — on a vehicle where commands explain motion, the
control-conditioned dynamics GP is the best bridge supplier tested in this
whole evaluation, and the R^2 gate proposed in the SOLAQUA section is
confirmed as the right deployment switch (R^2 ~ 0.5 here vs 0.06 there
cleanly separates the win from the loss). Caveats: this is a basin, not
open water (no currents or waves — the disturbance term that sank SOLAQUA
is absent by construction, so this is the method's best case); 3-DOF
horizontal motion only (yaw axis is a rate, heave untested); velocities are
the vehicle's navigation estimates (DVL + fiber-optic gyro), not raw DVL
bottom-track; and the 4 Hz sample rate is empirically determined (three
independent checks in the loader docstring), not stated by the source.
