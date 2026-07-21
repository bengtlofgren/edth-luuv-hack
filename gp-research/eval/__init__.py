"""Evaluation harness for the GP/IMU DVL-bridging POCs (see eval/PLAN.md).

Re-tests the claims of poc_03_real_data.py and poc_06_imu_bridge.py with a
causal-by-construction, multi-gap-ensemble methodology. The POC scripts
themselves are left untouched; this package extracts and fixes their working
pieces (bag/npy loading, GP kernel, IMU calibration + integration, the 2-state
KF seam, the hybrid fusion schemes) into reusable, tested modules.

Submodules:
    data      -- Snapir V_test.npy and SOLAQUA rosbag loaders (+ downloads).
    bridges   -- GP bridge, IMU calibration + dead-reckoning bridge, hybrids.
    seam      -- the shared 2-state constant-velocity Kalman filter.
    metrics   -- rmse, 2-sigma coverage, honesty ratio, position drift.
    evaluate  -- generic sliding multi-gap sweep engine.
"""
