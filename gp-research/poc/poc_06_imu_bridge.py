# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "rosbags"]
# ///
"""POC 6 — IMU-informed bridging of a DVL dropout, on real SOLAQUA ROV data.

POC 3b showed the failure mode of GP-only bridging: during an unobserved
maneuver the GP (and thus adaptive R) is smoothly wrong AND overconfident.
The fix explored here: during the outage, bridge with a sensor that still
observes the maneuver — the IMU.

Data: SOLAQUA dataset (SINTEF Ocean, CC-BY-SA-4.0, data.sintef.no), BlueROV2
net-following runs at a full-scale fish farm.  Two regimes:
  * 2024-08-22_14-29-05 — gentle (0.1 m/s).  Nortek Nucleus IMU @96 Hz plus a
    SECOND real DVL (Nortek bottomtrack) as an independent check.
  * 2024-08-20_15-18-27 — fast (0.3 m/s, larger surge swings).  Only the
    vehicle's Pixhawk-style IMU @15 Hz (MAVLink RAW_IMU units: mg, mrad/s).

IMU bridge design (causal — uses only pre-gap data, unlike the GP which
conditions on both sides):
  1. CALIBRATE on the pre-gap window: ridge-regress A50 acceleration dv/dt
     onto IMU specific force:  dv/dt + w x v ~ R f + b.   R absorbs the
     unknown IMU->body mounting rotation (and any scale), b absorbs
     gravity+bias (valid while attitude is steady, roll/pitch std ~0.01 rad
     here; a full INS would use attitude instead).  One refinement pass adds
     the w x v term using w_body = R_orth w_imu, R_orth = nearest rotation
     to R (SVD).
  2. INTEGRATE from the last pre-gap A50 fix:  v' = R f + b - w_body x v,
     at the IMU rate, through the gap.
  3. UNCERTAINTY: residual-driven random walk, var(T) = sigma_r^2 * tau * T
     (sigma_r, tau = std and autocorrelation time of calibration residuals),
     plus the last fix's variance.  Grows with gap length, like a real INS.

Both bridges then feed the unchanged POC 3 Kalman seam as (mean, variance).

HYBRID supplier (the design target of implementation/02): the GP's variance is
exactly what is overconfident during a maneuver, so naive inverse-variance
fusion would over-trust it.  But the disagreement between the inertially
observed estimate and the prior-based guess IS the maneuver evidence the GP
lacks.  So:  var_gp' = var_gp + (v_imu - v_gp)^2  (discrepancy inflation),
then inverse-variance fuse (v_gp, var_gp') with (v_imu, var_imu).  Quiet =>
discrepancy ~ 0 => fusion leans on the (tight, correct) GP.  Maneuver =>
discrepancy inflates the GP away => fusion follows the IMU.

Run:  uv run poc_06_imu_bridge.py
"""
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge

BAGS = [
    dict(tag="gentle 0.1 m/s (Nucleus IMU @96 Hz)",
         file="2024-08-22_14-29-05_data.bag",
         data_id="5026fa35-2df8-44eb-aa06-2ed1df6c53ae",
         imu_topic="/nucleus1000dvl/imu", imu_style="vector3",
         acc_scale=1.0, gyr_scale=1.0,            # already m/s^2, rad/s
         nortek=True, gap=(30.0, 50.0)),
    dict(tag="fast 0.3 m/s (Pixhawk IMU @15 Hz)",
         file="2024-08-20_15-18-27_data.bag",
         data_id="ab14ebc0-a2d2-4c7f-89e2-b7c1fcd11b9b",
         imu_topic="/sensor/imu", imu_style="flat",
         acc_scale=9.80665e-3, gyr_scale=1e-3,    # MAVLink RAW_IMU: mg, mrad/s
         nortek=False, gap=(25.0, 45.0)),
]
VAR_A50 = 0.025**2                       # A50 fom-scale variance of a fix


def locate(cfg):
    cands = [Path(__file__).with_name("data") / cfg["file"],
             Path.home() / "dvl-gp" / "data" / "solaqua" / cfg["file"]]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        p = cands[0]
        p.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(
            f"https://data.sintef.no/api/public/data/{cfg['data_id']}/file", p)
    return p


def smooth_t(y, dt, win_s):
    n = max(3, int(round(win_s / dt)) | 1)
    return np.convolve(y, np.ones(n) / n, mode="same")


def acf_tau(x, dt):
    x = x - x.mean()
    c = np.correlate(x, x, "full")[x.size - 1:] / (x**2).sum()
    below = np.flatnonzero(c < 0)
    return (below[0] if below.size else x.size) * dt


def run_kf(meas, meas_var, has_meas, dt, q, v0):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[0.0, 1.0]])
    x, P = np.array([0.0, v0]), np.diag([1e-4, VAR_A50])
    xs, Ps = [], []
    for k in range(len(meas)):
        x, P = F @ x, F @ P @ F.T + Q
        if has_meas[k]:
            S = (H @ P @ H.T)[0, 0] + meas_var[k]
            K = (P @ H.T / S).ravel()
            x = x + K * (meas[k] - x[1])
            P = P - np.outer(K, H @ P)
        xs.append(x.copy()); Ps.append(P.copy())
    return np.array(xs), np.array(Ps)


def run_bag(cfg):
    a50, imu, nortek = [], [], []
    with AnyReader([locate(cfg)]) as r:
        topics = ("/sensor/dvl_velocity", cfg["imu_topic"], "/nucleus1000dvl/bottomtrack")
        for conn, ts, raw in r.messages(connections=[c for c in r.connections
                                                     if c.topic in topics]):
            m = r.deserialize(raw, conn.msgtype)
            t = ts / 1e9
            if conn.topic == "/sensor/dvl_velocity":
                if m.velocity_valid:
                    a50.append((t, m.velocity.x, m.velocity.y, m.velocity.z))
            elif conn.topic == cfg["imu_topic"]:
                if cfg["imu_style"] == "vector3":
                    imu.append((t, m.accelerometer.x, m.accelerometer.y, m.accelerometer.z,
                                m.gyroscope.x, m.gyroscope.y, m.gyroscope.z))
                else:
                    imu.append((t, m.acc_x, m.acc_y, m.acc_z, m.gyro_x, m.gyro_y, m.gyro_z))
            else:
                v = m.dvl_velocity_xyz
                if m.data_valid and abs(v.x) < 10 and abs(v.y) < 10:   # -32.77 = invalid
                    nortek.append((t, v.x, v.y, v.z))
    a50, imu = np.array(a50), np.array(imu)
    nortek = np.array(nortek) if nortek else np.empty((0, 4))
    imu[:, 1:4] *= cfg["acc_scale"]; imu[:, 4:7] *= cfg["gyr_scale"]
    t0 = a50[0, 0]
    a50[:, 0] -= t0; imu[:, 0] -= t0
    if nortek.size:
        nortek[:, 0] -= t0

    gap0, gap1 = cfg["gap"]
    gap_mask = (a50[:, 0] >= gap0) & (a50[:, 0] < gap1)
    have = ~gap_mask
    CAL = a50[:, 0] < gap0

    # --- IMU bridge: causal calibration, then integration through the gap.
    dt_a50 = float(np.median(np.diff(a50[:, 0])))
    dt_imu = float(np.median(np.diff(imu[:, 0])))
    t_cal = a50[CAL, 0]
    v_cal = np.column_stack([smooth_t(a50[CAL, i], dt_a50, 1.0) for i in (1, 2, 3)])
    dvdt = np.gradient(v_cal, t_cal, axis=0)
    f_cal = np.column_stack([np.interp(t_cal, imu[:, 0], smooth_t(imu[:, i], dt_imu, 0.3))
                             for i in (1, 2, 3)])
    w_cal = np.column_stack([np.interp(t_cal, imu[:, 0], smooth_t(imu[:, i], dt_imu, 0.3))
                             for i in (4, 5, 6)])
    trim = slice(5, -5)
    ridge = Ridge(alpha=1.0).fit(f_cal[trim], dvdt[trim])
    for _ in range(2):
        U, _, Vt = np.linalg.svd(ridge.coef_)
        R_orth = U @ Vt
        wxv = np.cross(w_cal @ R_orth.T, v_cal)
        ridge = Ridge(alpha=1.0).fit(f_cal[trim], (dvdt + wxv)[trim])
    R_map, b_map = ridge.coef_, ridge.intercept_
    resid = (dvdt + wxv)[trim] - ridge.predict(f_cal[trim])
    sigma_r = resid.std(axis=0)
    tau = np.clip([acf_tau(resid[:, i], dt_a50) for i in range(3)], 0.2, 5.0)

    i_last = np.flatnonzero(CAL)[-1]
    in_gap = (imu[:, 0] > a50[i_last, 0]) & (imu[:, 0] < gap1 + 1.0)
    t_int = imu[in_gap, 0]
    v_imu = np.empty((t_int.size, 3))
    v = a50[i_last, 1:4].copy()
    for k in range(t_int.size):
        dt_k = t_int[k] - (t_int[k - 1] if k else a50[i_last, 0])
        v = v + dt_k * (R_map @ imu[in_gap][k, 1:4] + b_map
                        - np.cross(R_orth @ imu[in_gap][k, 4:7], v))
        v_imu[k] = v
    sig_imu = np.sqrt(VAR_A50 + sigma_r**2 * tau * (t_int[:, None] - t_int[0]))

    # --- GP bridge (harness settings; acausal).
    gp_mean = np.empty((a50.shape[0], 3)); gp_sig = np.empty_like(gp_mean)
    for i in (1, 2, 3):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=5.0, length_scale_bounds=(1.0, 60.0), nu=2.5)
                  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2)
        gp.fit(a50[have, 0][:, None], a50[have, i])
        gp_mean[:, i - 1], gp_sig[:, i - 1] = gp.predict(a50[:, 0][:, None],
                                                         return_std=True)

    # --- Hybrid supplier: discrepancy-inflated inverse-variance fusion.
    v_imu_all = np.column_stack([np.interp(a50[:, 0], t_int, v_imu[:, i])
                                 for i in range(3)])
    var_imu_all = np.column_stack([np.interp(a50[:, 0], t_int, sig_imu[:, i])
                                   for i in range(3)]) ** 2
    var_gp_infl = gp_sig**2 + (v_imu_all - gp_mean) ** 2
    hyb_var = 1.0 / (1.0 / var_gp_infl + 1.0 / var_imu_all)
    hyb_mean = hyb_var * (gp_mean / var_gp_infl + v_imu_all / var_imu_all)

    # --- Bridge accuracy vs the hidden A50.
    t_hid, v_hid = a50[gap_mask, 0], a50[gap_mask, 1:4]
    v_imu_at = np.column_stack([np.interp(t_hid, t_int, v_imu[:, i]) for i in range(3)])
    zoh = np.tile(a50[i_last, 1:4], (t_hid.size, 1))
    print(f"\n=== {cfg['tag']} — {cfg['file']}, {gap1-gap0:.0f} s outage ===")
    print(f"bridge RMSE vs hidden A50 [cm/s]:   {'axis':>4} {'ZOH':>6} {'GP':>6} "
          f"{'IMU-DR':>7} {'HYBRID':>7}")
    for i, ax in enumerate("xyz"):
        rm = lambda p: np.sqrt(np.mean((p - v_hid[:, i]) ** 2)) * 100
        print(f"{'':35}{ax:>4} {rm(zoh[:, i]):6.1f} {rm(gp_mean[gap_mask, i]):6.1f} "
              f"{rm(v_imu_at[:, i]):7.1f} {rm(hyb_mean[gap_mask, i]):7.1f}")
    if nortek.size:
        ng = (nortek[:, 0] >= gap0) & (nortek[:, 0] < gap1)
        if ng.any():
            d = np.column_stack([np.interp(nortek[ng, 0], t_hid, v_hid[:, i])
                                 for i in range(3)]) - nortek[ng, 1:4]
            print(f"   (two-real-DVL noise floor in gap: {np.sqrt((d**2).mean(0)).round(3)*100} cm/s)")

    # --- The unchanged POC 3 KF seam, surge axis, three suppliers.
    AXIS = 0
    q = max(0.02, float(dvdt[trim, AXIS].std())) ** 2   # data-driven accel PSD
    bridge_times = np.arange(gap0 + 2.5, gap1, 5.0)
    is_bridge = np.zeros(a50.shape[0], bool)
    for bt in bridge_times:
        is_bridge[np.argmin(np.abs(a50[:, 0] - bt))] = True
    has_meas = have | is_bridge
    suppliers = {
        "naive GP (fixed R)": (gp_mean[:, AXIS], np.full(a50.shape[0], VAR_A50)),
        "adaptive GP": (gp_mean[:, AXIS], gp_sig[:, AXIS] ** 2),
        "adaptive IMU-DR": (v_imu_all[:, AXIS], var_imu_all[:, AXIS]),
        "hybrid GP+IMU": (hyb_mean[:, AXIS], hyb_var[:, AXIS]),
    }
    pos_ref = np.concatenate([[0.0], np.cumsum(0.5 * (a50[1:, 1 + AXIS] + a50[:-1, 1 + AXIS])
                                               * np.diff(a50[:, 0]))])
    print(f"KF on surge (q_std {np.sqrt(q):.3f} m/s^2): "
          f"{'supplier':<20} {'vRMSE gap':>10} {'honesty':>8} {'peak pos err':>13}")
    kf_out = {}
    for name, (mean_s, var_s) in suppliers.items():
        meas = np.where(have, a50[:, 1 + AXIS], mean_s)
        var = np.where(have, VAR_A50, var_s)
        xs, Ps = run_kf(meas, var, has_meas, dt_a50, q, a50[0, 1 + AXIS])
        v_err = np.abs(xs[:, 1] - a50[:, 1 + AXIS])
        v_claim = 2 * np.sqrt(Ps[:, 1, 1] + VAR_A50)
        p_err = np.abs(xs[:, 0] - pos_ref)
        kf_out[name] = (p_err, 2 * np.sqrt(Ps[:, 0, 0]))
        print(f"{'':29}{name:<20} "
              f"{np.sqrt(np.mean((xs[gap_mask,1]-a50[gap_mask,1+AXIS])**2))*100:8.1f} cm/s "
              f"{np.max(v_err[gap_mask]/v_claim[gap_mask]):7.1f}x {p_err.max():11.2f} m")
    return dict(cfg=cfg, a50=a50, nortek=nortek, have=have, gap_mask=gap_mask,
                t_hid=t_hid, v_hid=v_hid, gp_mean=gp_mean, gp_sig=gp_sig,
                t_int=t_int, v_imu=v_imu, sig_imu=sig_imu,
                hyb_mean=hyb_mean, hyb_var=hyb_var, kf_out=kf_out)


results = [run_bag(cfg) for cfg in BAGS]

# --- Figure: one column per bag; surge bridge (top), KF position drift (bottom).
C_GP, C_IMU, C_NAIVE, C_MEAS, C_HYB = "#2a78d6", "#1baf7a", "#e34948", "#8a897f", "#4a3aa7"
fig, axgrid = plt.subplots(2, len(results), figsize=(13, 7), sharex="col")
axgrid = np.atleast_2d(axgrid.T).T if len(results) == 1 else axgrid
for col, res in enumerate(results):
    cfg, a50 = res["cfg"], res["a50"]
    ax1, ax2 = axgrid[0, col], axgrid[1, col]
    ax1.axvspan(*cfg["gap"], color="0.92")
    ax1.plot(a50[res["have"], 0], a50[res["have"], 1], ".", color="#33322e", ms=3,
             label="A50 DVL (available)")
    ax1.plot(res["t_hid"], res["v_hid"][:, 0], "o", mfc="none", color=C_MEAS, ms=4,
             label="hidden A50 (truth proxy)")
    if res["nortek"].size:
        ax1.plot(res["nortek"][:, 0], res["nortek"][:, 1], "x", color="#33322e", ms=6,
                 label="Nortek DVL (independent)")
    ax1.plot(a50[:, 0], res["gp_mean"][:, 0], color=C_GP, lw=2, label="GP bridge (acausal)")
    ax1.fill_between(a50[:, 0], res["gp_mean"][:, 0] - 2 * res["gp_sig"][:, 0],
                     res["gp_mean"][:, 0] + 2 * res["gp_sig"][:, 0],
                     color=C_GP, alpha=0.15, lw=0)
    ax1.plot(res["t_int"], res["v_imu"][:, 0], color=C_IMU, lw=2,
             label="IMU dead-reckoning (causal)")
    ax1.fill_between(res["t_int"], res["v_imu"][:, 0] - 2 * res["sig_imu"][:, 0],
                     res["v_imu"][:, 0] + 2 * res["sig_imu"][:, 0],
                     color=C_IMU, alpha=0.15, lw=0)
    gm = res["gap_mask"]
    ax1.plot(a50[gm, 0], res["hyb_mean"][gm, 0], color=C_HYB, lw=1.8, ls=":",
             label="hybrid GP+IMU")
    ax1.set_title(cfg["tag"], fontsize=10)
    ax1.grid(alpha=0.25, lw=0.5)
    ax2.axvspan(*cfg["gap"], color="0.92")
    for name, c in [("naive GP (fixed R)", C_NAIVE), ("adaptive GP", C_GP),
                    ("adaptive IMU-DR", C_IMU), ("hybrid GP+IMU", C_HYB)]:
        p_err, p_claim = res["kf_out"][name]
        ax2.plot(a50[:, 0], p_err, color=c, label=f"|pos error|, {name}")
        ax2.plot(a50[:, 0], p_claim, color=c, ls="--", lw=1)
    ax2.set_xlabel("time since start [s]")
    ax2.grid(alpha=0.25, lw=0.5)
axgrid[0, 0].set_ylabel("surge velocity [m/s]")
axgrid[1, 0].set_ylabel("position error [m]  (dashed: claimed 2$\\sigma$)")
axgrid[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
axgrid[1, 0].legend(loc="upper left", fontsize=7)
fig.suptitle("POC 6: IMU-informed bridging of a 20 s A50 DVL outage — SOLAQUA BlueROV2, real data",
             fontsize=12)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("poc_06_imu_bridge.png"), dpi=120)
print("\nsaved poc_06_imu_bridge.png")
