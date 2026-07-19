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
| `poc_03_real_data.py` | POC 3 on real Snapir AUV DVL data (ANSFL BeamsNet): adaptive R stays honest in cruise but inherits the GP's overconfidence during unobserved maneuvers | ref 04, impl 00 |
| `poc_06_imu_bridge.py` | IMU-informed bridging on real SOLAQUA BlueROV2 data, two regimes (gentle 0.1 m/s w/ Nucleus IMU + second DVL; fast 0.3 m/s w/ Pixhawk IMU): causal IMU dead-reckoning vs acausal GP bridge through the POC 3 seam (GP wins when quiet, IMU when dynamic), plus the **hybrid supplier** — discrepancy-inflated inverse-variance fusion that tracks the better bridge in each regime without being told which | refs 04, impl 00, impl 02 |
| `poc_04_state_space_duality.py` | GP ↔ Kalman duality: a Matérn-3/2 GP *is* a 2-state Kalman smoother — identical posterior, O(n) not O(n³) (the edge-compute story) | refs 05, 08 |
| `poc_05_spde_boundaries.py` | GP ↔ SPDE duality: the Matérn kernel emerges from a discretised PDE; Dirichlet boundary conditions come for free | refs 06, 07 |

Reading order = numeric order; each builds on the previous one's setup.

> These are teaching toys, not production code: dense linear algebra, fixed
> seeds, 1-D signals. The production design lives in
> [`../implementation/`](../implementation/).
