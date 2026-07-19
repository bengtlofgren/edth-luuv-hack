# SOLAQUA sweep summary (Phase 2b)

## Bags used / skipped

- `2024-08-22_14-29-05_data.bag` (nucleus, dur=61.0s): 3 x 10s gaps, 2 x 20s gaps -- OK
- `2024-08-20_15-18-27_data.bag` (pixhawk, dur=56.0s): 2 x 10s gaps, 1 x 20s gaps -- OK
- `2024-08-20_15-09-34_data.bag` (pixhawk, dur=75.6s): 4 x 10s gaps, 3 x 20s gaps -- OK
- `2024-08-20_15-20-29_data.bag` (pixhawk, dur=57.2s): 2 x 10s gaps, 1 x 20s gaps -- OK
- `2024-08-20_14-57-38_data.bag` (pixhawk, dur=91.4s): 6 x 10s gaps, 5 x 20s gaps -- OK
- `2024-08-20_17-14-36_data.bag` (pixhawk, dur=47.8s): 1 x 10s gaps, 0 x 20s gaps -- OK
- `2024-08-22_14-47-39_data.bag` (nucleus, dur=45.4s): 1 x 10s gaps, 0 x 20s gaps -- OK
- `2024-08-22_14-06-43_data.bag` (nucleus, dur=33.7s): 0 x 10s gaps, 0 x 20s gaps -- SKIPPED (too short)

## 1. Bridge RMSE median [IQR], surge axis, by supplier x duration

### nucleus bags

- 10s gaps:
    - causal_gp   :   2.29 [ 2.11,  2.39] cm/s (n=4)
    - acausal_gp  :   2.43 [ 1.95,  2.89] cm/s (n=4)
    - imu_dr      :   2.95 [ 2.51,  3.25] cm/s (n=4)
    - hybrid_v1   :   2.42 [ 2.18,  2.53] cm/s (n=4)
    - hybrid_v2   :   2.32 [ 2.03,  2.56] cm/s (n=4)
    - zoh         :   2.67 [ 2.49,  2.90] cm/s (n=4)
- 20s gaps:
    - causal_gp   :   2.29 [ 2.25,  2.33] cm/s (n=2)
    - acausal_gp  :   2.45 [ 2.36,  2.53] cm/s (n=2)
    - imu_dr      :   3.19 [ 2.89,  3.50] cm/s (n=2)
    - hybrid_v1   :   2.25 [ 2.22,  2.28] cm/s (n=2)
    - hybrid_v2   :   2.32 [ 2.32,  2.33] cm/s (n=2)
    - zoh         :   2.80 [ 2.59,  3.02] cm/s (n=2)

### pixhawk bags

- 10s gaps:
    - causal_gp   :   2.33 [ 1.89,  3.14] cm/s (n=15)
    - acausal_gp  :   3.17 [ 2.24,  3.43] cm/s (n=15)
    - imu_dr      :   3.11 [ 2.31,  3.98] cm/s (n=15)
    - hybrid_v1   :   2.30 [ 1.83,  3.30] cm/s (n=15)
    - hybrid_v2   :   2.29 [ 1.81,  3.11] cm/s (n=15)
    - zoh         :   2.80 [ 2.31,  3.68] cm/s (n=15)
- 20s gaps:
    - causal_gp   :   2.50 [ 2.09,  2.77] cm/s (n=10)
    - acausal_gp  :   2.93 [ 2.40,  3.40] cm/s (n=10)
    - imu_dr      :   3.55 [ 2.59,  4.47] cm/s (n=10)
    - hybrid_v1   :   2.46 [ 2.11,  2.96] cm/s (n=10)
    - hybrid_v2   :   2.45 [ 2.05,  2.76] cm/s (n=10)
    - zoh         :   3.05 [ 2.25,  3.26] cm/s (n=10)

### overall

- 10s gaps:
    - causal_gp   :   2.33 [ 1.89,  2.88] cm/s (n=19)
    - acausal_gp  :   2.87 [ 2.10,  3.36] cm/s (n=19)
    - imu_dr      :   3.11 [ 2.31,  3.61] cm/s (n=19)
    - hybrid_v1   :   2.37 [ 1.83,  3.02] cm/s (n=19)
    - hybrid_v2   :   2.29 [ 1.81,  2.90] cm/s (n=19)
    - zoh         :   2.70 [ 2.31,  3.52] cm/s (n=19)
- 20s gaps:
    - causal_gp   :   2.33 [ 2.16,  2.75] cm/s (n=12)
    - acausal_gp  :   2.91 [ 2.27,  3.26] cm/s (n=12)
    - imu_dr      :   3.55 [ 2.58,  4.45] cm/s (n=12)
    - hybrid_v1   :   2.33 [ 2.15,  2.93] cm/s (n=12)
    - hybrid_v2   :   2.32 [ 2.09,  2.72] cm/s (n=12)
    - zoh         :   3.05 [ 2.25,  3.24] cm/s (n=12)

## 1b. Bridge RMSE median [IQR], compact all-axes (overall, both durations)

- surge:
    - causal_gp   :   2.33 [ 2.03,  2.76] cm/s (n=31)
    - acausal_gp  :   2.91 [ 2.22,  3.36] cm/s (n=31)
    - imu_dr      :   3.14 [ 2.52,  4.00] cm/s (n=31)
    - hybrid_v1   :   2.34 [ 2.05,  2.94] cm/s (n=31)
    - hybrid_v2   :   2.32 [ 1.98,  2.74] cm/s (n=31)
    - zoh         :   2.80 [ 2.25,  3.41] cm/s (n=31)
- sway:
    - causal_gp   :   2.29 [ 2.01,  2.87] cm/s (n=31)
    - acausal_gp  :   2.52 [ 2.13,  2.96] cm/s (n=31)
    - imu_dr      :   3.90 [ 3.03,  5.42] cm/s (n=31)
    - hybrid_v1   :   2.57 [ 2.17,  3.66] cm/s (n=31)
    - hybrid_v2   :   2.37 [ 2.02,  2.89] cm/s (n=31)
    - zoh         :   2.50 [ 2.29,  3.42] cm/s (n=31)
- heave:
    - causal_gp   :   5.65 [ 4.20,  8.73] cm/s (n=31)
    - acausal_gp  :   5.61 [ 4.20,  8.59] cm/s (n=31)
    - imu_dr      :   6.72 [ 4.37, 10.60] cm/s (n=31)
    - hybrid_v1   :   5.45 [ 4.13,  8.61] cm/s (n=31)
    - hybrid_v2   :   5.65 [ 4.50,  9.00] cm/s (n=31)
    - zoh         :   6.61 [ 4.55,  7.77] cm/s (n=31)

## 2. C3 -- IMU-DR vs causal GP vs ZOH on dynamic vs calm gaps (surge)

Median dynamic-ness across 31 gaps: 0.0435 m/s^2 (split point for dynamic/calm halves below)

### dynamic half (dynamic_ness >= median)

- n=16 gaps
- IMU-DR minus causal-GP RMSE: median 0.77 [0.15, 1.11] cm/s (IMU wins 19% of gaps)
- IMU-DR minus ZOH RMSE: median 0.17 [-0.62, 0.74] cm/s (IMU wins 38% of gaps)

### calm half (dynamic_ness < median)

- n=15 gaps
- IMU-DR minus causal-GP RMSE: median 0.69 [0.25, 1.52] cm/s (IMU wins 7% of gaps)
- IMU-DR minus ZOH RMSE: median 0.20 [-0.38, 1.43] cm/s (IMU wins 33% of gaps)

### IMU-DR vs ZOH overall (all gaps, surge)

- n=31 gaps; ZOH-minus-IMU RMSE median -1.25 [-2.43, 0.12] cm/s
- fraction of gaps where |ZOH - IMU| RMSE <= 1 cm/s: 35%

## 3. C4 -- hybrid v2 vs best single supplier (regret), hybrid v1 vs v2

- surge (n=31):
    - hybrid_v2 regret vs min(GP,IMU): median 0.00 [-0.02, 0.00] cm/s
    - hybrid_v1 regret vs min(GP,IMU): median 0.06 [-0.03, 0.16] cm/s
    - hybrid_v2 minus hybrid_v1 RMSE: median -0.06 [-0.17, 0.02] cm/s (v2 better)
    - hybrid_v1 bridge coverage: 100%, hybrid_v2 bridge coverage: 100%
- sway (n=31):
    - hybrid_v2 regret vs min(GP,IMU): median 0.00 [0.00, 0.02] cm/s
    - hybrid_v1 regret vs min(GP,IMU): median 0.09 [-0.09, 0.23] cm/s
    - hybrid_v2 minus hybrid_v1 RMSE: median -0.07 [-0.12, 0.09] cm/s (v2 better)
    - hybrid_v1 bridge coverage: 99%, hybrid_v2 bridge coverage: 99%
- heave (n=31):
    - hybrid_v2 regret vs min(GP,IMU): median 0.18 [0.00, 1.37] cm/s
    - hybrid_v1 regret vs min(GP,IMU): median 0.25 [-0.12, 1.01] cm/s
    - hybrid_v2 minus hybrid_v1 RMSE: median 0.18 [-0.29, 0.53] cm/s (v1 better)
    - hybrid_v1 bridge coverage: 97%, hybrid_v2 bridge coverage: 98%

## 4. Coverage table (bridge-level 2-sigma coverage, surge, all gaps)

- causal_gp   :  99.8% (n=31)
- acausal_gp  :  99.8% (n=31)
- imu_dr      : 100.0% (n=31)
- hybrid_v1   :  99.6% (n=31)
- hybrid_v2   :  99.8% (n=31)
- zoh         :  98.6% (n=31)

## 5. KF metrics (surge only, causal_gp/imu_dr/hybrid_v2/zoh)

- causal_gp   : vRMSE   2.35 [ 2.10,  2.74] cm/s (n=31); honesty median 0.44x (n=31); peak drift median 0.159 m (n=31)
- imu_dr      : vRMSE   2.68 [ 2.37,  3.26] cm/s (n=31); honesty median 0.42x (n=31); peak drift median 0.219 m (n=31)
- hybrid_v2   : vRMSE   2.35 [ 2.09,  2.83] cm/s (n=31); honesty median 0.44x (n=31); peak drift median 0.161 m (n=31)
- zoh         : vRMSE   2.81 [ 2.28,  3.27] cm/s (n=31); honesty median 0.49x (n=31); peak drift median 0.198 m (n=31)

## 6. IMU calibration diagnostics by bag group

- nucleus (n=6 gaps): Procrustes scale s median 0.060 [0.059, 0.071]; unconstrained-map singular values median [0.084, 0.047, 0.004]
- pixhawk (n=25 gaps): Procrustes scale s median 0.025 [0.022, 0.040]; unconstrained-map singular values median [0.163, 0.046, 0.006]
