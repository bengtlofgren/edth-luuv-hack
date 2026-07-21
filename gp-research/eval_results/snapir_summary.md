# Snapir sweep -- summary

N gaps: 174 (58 starts x 3 durations); N records: 2784

## Sensor-noise estimate (independent, t<180s high-frequency residual)

| axis | sigma_hat [m/s] | var [(m/s)^2] |
|---|---|---|
| x | 0.07108 | 5.051985e-03 |
| y | 0.06582 | 4.332671e-03 |
| z | 0.01712 | 2.930712e-04 |

Median dynamic-ness score (quiet/dynamic split threshold): 0.0416775

## Table 1a: Bridge RMSE median [IQR] by supplier x duration (surge, axis x)

| supplier | 10s | 30s | 60s | all |
|---|---|---|---|---|
| causal_gp | 0.1609 [0.1105, 0.2072] | 0.2706 [0.1660, 0.4046] | 0.4088 [0.2383, 0.5057] | 0.2394 [0.1498, 0.4129] |
| acausal_gp | 0.1349 [0.0929, 0.1774] | 0.1648 [0.1223, 0.2612] | 0.2389 [0.1517, 0.4047] | 0.1692 [0.1210, 0.2643] |
| zoh | 0.1691 [0.1041, 0.2337] | 0.2458 [0.1443, 0.4257] | 0.3962 [0.1841, 0.5997] | 0.2266 [0.1395, 0.4323] |
| linear | 0.1453 [0.1026, 0.1830] | 0.1724 [0.1290, 0.2729] | 0.2632 [0.1487, 0.4359] | 0.1738 [0.1290, 0.2832] |

## Table 1b: Bridge RMSE median (all durations pooled), compact all-axes

| supplier | x | y | z |
|---|---|---|---|
| causal_gp | 0.2394 | 0.2622 | 0.2016 |
| acausal_gp | 0.1692 | 0.1825 | 0.2012 |
| zoh | 0.2266 | 0.2861 | 0.2608 |
| linear | 0.1738 | 0.2118 | 0.2584 |

## Table 2: Bridge 2-sigma coverage fraction by supplier x duration (surge)

### Overall

| supplier | 10s | 30s | 60s | all |
|---|---|---|---|---|
| causal_gp | 0.900 | 0.852 | 0.821 | 0.857 |
| acausal_gp | 0.914 | 0.910 | 0.897 | 0.907 |
| zoh | 0.572 | 0.438 | 0.352 | 0.454 |
| linear | 0.659 | 0.542 | 0.436 | 0.546 |

### Quiet vs dynamic split (median split on gap dynamic-ness score, maneuver-overconfidence claim)

| supplier | quiet (n=87) | dynamic (n=87) |
|---|---|---|
| causal_gp | 0.949 | 0.766 |
| acausal_gp | 0.965 | 0.849 |
| zoh | 0.562 | 0.346 |
| linear | 0.638 | 0.454 |

## Table 3a: KF median peak drift [IQR] (m) + median honesty by config x duration (surge)

| config | 10s drift | 30s drift | 60s drift | 10s honesty | 30s honesty | 60s honesty |
|---|---|---|---|---|---|---|
| causal_naive | 0.6191 [0.2856, 1.2252] | 4.7377 [2.5455, 9.4489] | 16.2161 [8.0729, 23.4872] | 1.285 | 2.753 | 4.036 |
| causal_adaptive | 0.4799 [0.3261, 0.9563] | 4.7095 [1.6765, 7.7028] | 14.8004 [7.0376, 23.1647] | 1.110 | 1.698 | 2.305 |
| acausal_adaptive | 0.4440 [0.3030, 0.7182] | 1.8547 [1.1127, 5.8990] | 8.0100 [3.2497, 16.2852] | 1.120 | 1.563 | 2.175 |
| zoh_fixed | 0.7162 [0.3917, 1.4600] | 4.7682 [2.0867, 9.9384] | 16.4651 [5.6169, 26.8887] | 1.370 | 2.510 | 3.974 |

## Table 3b: KF median peak drift + honesty, quiet vs dynamic split

| config | quiet drift | dynamic drift | quiet honesty | dynamic honesty |
|---|---|---|---|---|
| causal_naive | 2.7893 | 5.1063 | 1.927 | 3.513 |
| causal_adaptive | 2.4573 | 5.5540 | 1.340 | 2.320 |
| acausal_adaptive | 1.3611 | 3.6652 | 1.226 | 2.178 |
| zoh_fixed | 2.1041 | 6.8006 | 1.809 | 3.305 |

## Table 3c: Adaptive-R drift claim -- paired peak-drift deltas, causal_naive - causal_adaptive (positive = adaptive wins) (m)

| split | 10s median delta | 30s median delta | 60s median delta | all median delta | all win rate (adaptive<naive) |
|---|---|---|---|---|---|
| overall | 0.0058 | 0.1044 | 0.0355 | 0.0140 | 0.552 |
| quiet | -0.0051 | 0.2071 | 0.3236 | 0.0629 | 0.598 |
| dynamic | 0.0074 | 0.0257 | -0.3526 | 0.0043 | 0.506 |

## Table 4: Win-rate matrix -- fraction of gaps where supplier beats causal_gp on bridge RMSE (paired, surge axis)

| supplier | 10s | 30s | 60s | all |
|---|---|---|---|---|
| acausal_gp | 0.724 | 0.828 | 0.741 | 0.764 |
| zoh | 0.431 | 0.517 | 0.517 | 0.489 |
| linear | 0.534 | 0.724 | 0.776 | 0.678 |
