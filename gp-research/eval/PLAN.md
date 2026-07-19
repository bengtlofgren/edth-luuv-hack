# Evaluation plan: re-testing the POC claims rigorously

Response to critical review of `poc/poc_03_real_data.py` and `poc/poc_06_imu_bridge.py`.
The POC scripts stay untouched (historical artifacts). This package re-tests their
claims with defensible methodology. Update the claims where results differ.

## Review findings driving this plan

1. GP bridges were acausal (conditioned on post-gap data); the deployment design
   (implementation/02 §5) is causal. Must evaluate causal GP extrapolation.
2. n=1 gap per scenario. Field standard (ST-AidedEKF, arXiv 2504.07697): many
   outage start times x durations x runs, distributions not single numbers.
3. IMU calibration via ridge collapsed (singular values 0.005-0.087 vs ~1 for a
   rotation; errors-in-variables attenuation). Use constrained orthogonal
   Procrustes (rotation + single scale), band-matched filtering of both sides,
   and add the sigma_b^2*T^2 calibration-bias term to bridge variance.
4. Hybrid discrepancy-inflation is asymmetric, double-uses v_imu, and does
   inverse-variance fusion on correlated inputs. Add hybrid v2: chi-square gate
   on d^2/(var_gp+var_imu), covariance intersection for fusion; keep v1 for
   comparison.
5. VAR_A50 hardcoded; per-fix `fom` field must be used as R (gentle bag median
   fom 0.025, fast bag 0.074).
6. Honesty yardstick was the fitted WhiteKernel evaluating itself. Estimate
   sensor noise independently (quiet-segment high-pass estimate) and report
   ensemble coverage fractions, not single-gap max ratios.
7. Missing baselines: ZOH always; linear interpolation in acausal comparisons.

## Package layout

    gp-research/eval/
      __init__.py
      data.py       # loaders + downloads (Snapir npy, SOLAQUA bags via public API)
      bridges.py    # gp_bridge (causal/acausal), imu_bridge + calibrate, hybrids
      seam.py       # 2-state KF (position, velocity), supplier injection
      metrics.py    # rmse, coverage_2sigma, honesty, position drift
      evaluate.py   # sliding multi-gap sweep -> list[dict] records
    gp-research/tests/          # pytest
    gp-research/eval_snapir.py  # sweep on ANSFL V_test (2001 s @ 1 Hz)
    gp-research/eval_solaqua.py # sweep over >=6 SOLAQUA bags

Run everything with the uv-managed env: `uv run --with <deps>` or the PEP 723
headers on the two eval scripts. Tests: `uv run --with pytest,numpy,scikit-learn,rosbags,matplotlib pytest gp-research/tests/`.

## Protocol

- Snapir (data/V_test.npy from BeamsNet repo; 1 Hz, 3 axes): gap durations
  {10, 30, 60} s, gap starts sliding every 30 s from t=180 to t=1900. Causal GP
  trains on trailing 180 s window before the gap; acausal GP on all non-gap data
  within a +-300 s window. Baselines: ZOH, linear (acausal). Report per-gap
  records; aggregate median/IQR RMSE, 2sigma coverage fraction, KF honesty +
  drift, paired win rates vs causal GP.
- SOLAQUA: >=6 data bags (existing three + newly downloaded), gap durations
  {10, 20} s sliding every 10 s (skip first 20 s for calibration). Suppliers:
  causal GP, acausal GP, IMU-DR (new calibration), hybrid v1, hybrid v2, ZOH.
  R from per-fix fom (clipped to >=1e-4 variance). IMU topic/units per bag as
  in poc_06 (Nucleus vector3 m/s^2, rad/s; Pixhawk flat mg, mrad/s).
- KF seam: identical 2-state filter for all suppliers; bridge sample every 5 s
  during gap (unchanged from POCs, acknowledged crutch); honesty measured with
  independent noise estimate added to the claim.

## Success criteria / claims under test

- "adaptive R halves cruise drift" -> does adaptive R reduce median drift
  across the gap ensemble vs naive fixed R? Causal setting.
- "GP+adaptive R overconfident in maneuvers" (the negative result) ->
  coverage fraction by gap dynamic-ness quantile.
- "IMU wins when dynamic" -> does the rebuilt IMU bridge beat causal GP and
  ZOH on dynamic gaps? Must also report IMU-vs-ZOH explicitly.
- "hybrid near-best both regimes" -> hybrid v2 vs best single supplier,
  paired deltas; hybrid v1 sway-degradation case re-measured.
