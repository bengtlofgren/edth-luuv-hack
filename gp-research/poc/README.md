# `gp-research/poc/` — Toy proof-of-concept scripts

Minimal, self-contained Python demos — one per researched idea from
[`../references/`](../references/). Each script is deliberately tiny, leans on
library imports, is commented throughout, prints a one-line quantitative
takeaway, and saves a PNG next to itself.

No install needed — each script declares its own dependencies (PEP 723):

```bash
uv run poc_01_gp_denoise.py
```

| POC | Idea demonstrated | Research docs |
|---|---|---|
| `poc_01_gp_denoise.py` | GP regression basics: denoise a noisy 1 Hz DVL velocity, get a calibrated uncertainty band | refs 01–03 |
| `poc_02_dropout_bridge.py` | Bridge a 15 s DVL dropout with variance that grows through the gap; flag outliers via the predictive band | ref 04 |
| `poc_03_gp_feeds_kalman.py` | The integration seam: GP (mean, variance) as a Kalman measurement with adaptive R — honest vs overconfident fusion | ref 04, impl 00 |
| `poc_03_real_data.py` | POC 3 on real Snapir AUV DVL data (ANSFL BeamsNet). Single-gap demo; the [ensemble re-evaluation](../eval/RESULTS.md) revised its claims: adaptive R's drift gain is marginal — what it halves is the filter's *overconfidence* — and the GP stays overconfident during unobserved maneuvers (confirmed) | ref 04, impl 00 |
| `poc_06_imu_bridge.py` | IMU-informed bridging on real SOLAQUA BlueROV2 data (Nucleus and Pixhawk IMUs): causal IMU dead-reckoning vs GP bridge through the POC 3 seam, plus a **hybrid supplier**. Single-gap demo suggested "GP when quiet, IMU when dynamic"; the [ensemble re-evaluation](../eval/RESULTS.md) **refuted** the IMU leg (frame calibration unobservable on these bags — IMU-DR loses to causal GP even on dynamic gaps) while the hybrid v2 (χ² gate + covariance intersection, in [`../eval/bridges.py`](../eval/bridges.py)) survives with ~zero regret vs the best single supplier | refs 04, impl 00, impl 02 |
| `poc_04_state_space_duality.py` | GP ↔ Kalman duality: a Matérn-3/2 GP *is* a 2-state Kalman smoother — identical posterior, O(n) not O(n³) (the edge-compute story) | refs 05, 08 |
| `poc_05_spde_boundaries.py` | GP ↔ SPDE duality: the Matérn kernel emerges from a discretised PDE; Dirichlet boundary conditions come for free | refs 06, 07 |

Reading order = numeric order; each builds on the previous one's setup.

> These are teaching toys, not production code: dense linear algebra, fixed
> seeds, 1-D signals. The production design lives in
> [`../implementation/`](../implementation/).

> **Claims audit:** the single-gap numbers printed by `poc_03_real_data.py`
> and `poc_06_imu_bridge.py` were re-tested on multi-gap ensembles (174
> Snapir gaps, 31 SOLAQUA gaps over 7 bags) with causal bridging and ZOH
> baselines. Verdicts and revised claims: [`../eval/RESULTS.md`](../eval/RESULTS.md).
> The scripts themselves are kept as-is (historical artifacts).
