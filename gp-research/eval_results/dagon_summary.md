# Dagon basin sweep summary

DFKI Dagon AUV, saltwater basin, sinusoidal thruster system-ID excitation. 11567 samples, 2892 s at dt=0.25 s (4 Hz, empirically determined -- see eval/data.py::load_dagon). Axes: surge/sway [m/s], yaw RATE [rad/s]; axes are never pooled.

Gaps: 46 x 10 s + 46 x 20 s, starts every 60 s from t=120 s. Causal time-GP trains on a trailing 60 s window; dyn_gp trains on ALL pre-gap data; ZOH holds the last pre-gap sample.

## 1. Bridge RMSE median [IQR] by supplier x duration, per axis

### surge [m/s]

- 10s gaps:
    - causal_gp :   9.37 [ 4.97, 15.03] cm/s (n=46)
    - dyn_gp    :   2.29 [ 0.96,  4.57] cm/s (n=46)
    - zoh       :  15.32 [ 8.91, 24.53] cm/s (n=46)
- 20s gaps:
    - causal_gp :  14.95 [11.63, 18.77] cm/s (n=46)
    - dyn_gp    :   4.02 [ 1.76,  7.36] cm/s (n=46)
    - zoh       :  20.72 [15.53, 28.76] cm/s (n=46)

### sway [m/s]

- 10s gaps:
    - causal_gp :   6.22 [ 3.88,  7.97] cm/s (n=46)
    - dyn_gp    :   1.81 [ 1.05,  3.07] cm/s (n=46)
    - zoh       :   7.42 [ 5.64,  9.99] cm/s (n=46)
- 20s gaps:
    - causal_gp :   7.07 [ 5.70, 11.19] cm/s (n=46)
    - dyn_gp    :   3.40 [ 1.87,  5.27] cm/s (n=46)
    - zoh       :   9.39 [ 6.53, 12.62] cm/s (n=46)

### yaw_rate [rad/s]

- 10s gaps:
    - causal_gp : 168.03 [98.36, 301.50] mrad/s (n=46)
    - dyn_gp    :  37.84 [19.05, 82.00] mrad/s (n=46)
    - zoh       : 230.41 [89.80, 401.21] mrad/s (n=46)
- 20s gaps:
    - causal_gp : 297.11 [160.37, 378.81] mrad/s (n=46)
    - dyn_gp    :  62.41 [22.62, 97.63] mrad/s (n=46)
    - zoh       : 358.86 [235.57, 442.15] mrad/s (n=46)

## 2. Paired deltas -- dyn_gp minus causal time-GP and minus ZOH, per axis

Negative delta = dyn_gp wins. Win rate = fraction of gaps where dyn_gp's RMSE is lower (paired per gap).

Median gap dynamic-ness (std of 2s-smoothed surge dv/dt in gap): 0.0481 m/s^2 -- split point below.

### surge [m/s]

- all gaps (n=92):
    - dyn_gp minus time-GP:  -7.76 [-13.90, -2.06] cm/s (n=92) (dyn_gp wins 86%)
    - dyn_gp minus ZOH:     -14.12 [-24.11, -5.86] cm/s (n=92) (dyn_gp wins 89%)
- dynamic half (>= median) (n=46):
    - dyn_gp minus time-GP: -11.45 [-15.94, -4.53] cm/s (n=46) (dyn_gp wins 89%)
    - dyn_gp minus ZOH:     -22.15 [-27.03, -13.19] cm/s (n=46) (dyn_gp wins 98%)
- calm half (< median) (n=46):
    - dyn_gp minus time-GP:  -5.15 [-10.93, -0.44] cm/s (n=46) (dyn_gp wins 83%)
    - dyn_gp minus ZOH:      -8.71 [-14.65, -1.93] cm/s (n=46) (dyn_gp wins 80%)

### sway [m/s]

- all gaps (n=92):
    - dyn_gp minus time-GP:  -3.97 [-6.39, -1.45] cm/s (n=92) (dyn_gp wins 85%)
    - dyn_gp minus ZOH:      -5.43 [-8.17, -2.72] cm/s (n=92) (dyn_gp wins 90%)
- dynamic half (>= median) (n=46):
    - dyn_gp minus time-GP:  -4.09 [-5.83, -2.30] cm/s (n=46) (dyn_gp wins 91%)
    - dyn_gp minus ZOH:      -6.24 [-8.59, -3.68] cm/s (n=46) (dyn_gp wins 96%)
- calm half (< median) (n=46):
    - dyn_gp minus time-GP:  -3.49 [-6.61, -0.21] cm/s (n=46) (dyn_gp wins 78%)
    - dyn_gp minus ZOH:      -4.68 [-7.28, -1.78] cm/s (n=46) (dyn_gp wins 85%)

### yaw_rate [rad/s]

- all gaps (n=92):
    - dyn_gp minus time-GP: -182.87 [-295.94, -48.01] mrad/s (n=92) (dyn_gp wins 90%)
    - dyn_gp minus ZOH:     -264.46 [-383.01, -66.58] mrad/s (n=92) (dyn_gp wins 85%)
- dynamic half (>= median) (n=46):
    - dyn_gp minus time-GP: -213.77 [-296.34, -52.76] mrad/s (n=46) (dyn_gp wins 91%)
    - dyn_gp minus ZOH:     -237.75 [-356.45, -49.41] mrad/s (n=46) (dyn_gp wins 85%)
- calm half (< median) (n=46):
    - dyn_gp minus time-GP: -173.19 [-267.30, -51.65] mrad/s (n=46) (dyn_gp wins 89%)
    - dyn_gp minus ZOH:     -300.62 [-409.93, -76.20] mrad/s (n=46) (dyn_gp wins 85%)

## 3. Bridge 2-sigma coverage (mean over gaps), per axis

| supplier | surge | sway | yaw_rate |
|---|---|---|---|
| causal_gp | 93% | 93% | 87% |
| dyn_gp | 80% | 76% | 86% |
| zoh | 27% | 27% | 35% |

## 4. Observability diagnostic -- linear-mean R^2 per axis

R^2 of the ridge thrust-gain/damping mean alone on pre-gap acceleration (dyn_gp's DynCalib.lin_r2; median [IQR] over gaps -- each gap refits on its own causal history):

- surge    : R^2 = 0.485 [0.352, 0.549] (n=92)
- sway     : R^2 = 0.261 [0.218, 0.279] (n=92)
- yaw_rate : R^2 = 0.672 [0.566, 0.695] (n=92)

## 5. What the GP residual adds on top of the ridge mean

One-step acceleration prediction at the TRUE hidden in-gap states (evaluation-only diagnostic): residual RMS of the ridge mean alone vs ridge+GP, median over gaps. Ratio < 1 means the GP correction helps out-of-sample.

- surge    : ridge-only 0.0238 m/s^2, ridge+GP 0.0106 m/s^2; ratio 0.445 [0.268, 0.829] (n=92)
- sway     : ridge-only 0.0189 m/s^2, ridge+GP 0.0074 m/s^2; ratio 0.464 [0.227, 0.652] (n=92)
- yaw_rate : ridge-only 0.0397 rad/s^2, ridge+GP 0.0206 rad/s^2; ratio 0.500 [0.220, 0.755] (n=92)
