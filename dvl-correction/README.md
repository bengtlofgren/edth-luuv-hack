# dvl-correction

A navigation fusion layer for corrected DVL measurements and IMU data.

The main entry point is `DvlImuKalmanLayer` in
`dvl_correction.dvl_imu_kalman`. It performs:

1. IMU strapdown propagation from angular velocity and linear acceleration.
2. Corrected-DVL velocity and optional position measurement updates with an
   error-state Kalman filter.
3. Output of corrected position, velocity, attitude, gyroscope bias, and
   accelerometer bias estimates.

The higher-level `NavigationFusionPipeline` adds the pieces needed around the
filter for real logs:

- CSV and JSONL adapters for provided IMU/DVL data.
- DVL frame calibration, scale, and lever-arm compensation.
- DVL quality gates for mode/status/beams/altitude/velocity.
- Buffered timestamp ordering for high-rate IMU and lower-rate DVL streams.
- JSON configuration for initial state, covariance, noise, calibration, and
  synchronization settings.

The IMU path predicts position by integrating acceleration and attitude over
time. The DVL input is expected to be corrected before entering this layer, for
example after beam quality checks, frame alignment, scale correction, and any
lever-arm compensation needed by the sensor stack. If the DVL stack also
produces an integrated/corrected position or displacement track, pass it as
`position_nav_m` so the filter can directly correct the IMU-predicted position.

```python
from dvl_correction import (
    CorrectedDvlMeasurement,
    DvlImuKalmanLayer,
    ImuSample,
)

layer = DvlImuKalmanLayer()

output = layer.process(
    ImuSample(
        timestamp_s=0.0,
        angular_velocity_rad_s=[0.0, 0.0, 0.0],
        linear_acceleration_m_s2=[0.0, 0.0, 0.0],
    ),
    CorrectedDvlMeasurement(
        timestamp_s=0.0,
        velocity_body_m_s=[0.2, 0.0, 0.0],
        position_nav_m=[1.5, 0.0, -0.2],
    ),
)

print(output.position_m)
print(output.velocity_m_s)
print(output.attitude_quat_wxyz)
print(output.gyro_bias_rad_s)
print(output.accel_bias_m_s2)
```

For raw DVL streams, use `NavigationFusionPipeline`:

```python
from dvl_correction import (
    DvlCorrectionLayer,
    FrameCalibration,
    ImuSample,
    NavigationFusionPipeline,
    RawDvlMeasurement,
    SynchronizerConfig,
)

pipeline = NavigationFusionPipeline(
    dvl_correction=DvlCorrectionLayer(calibration=FrameCalibration()),
    config=SynchronizerConfig(max_delay_s=0.1),
)

pipeline.add_imu_sample(
    ImuSample(
        timestamp_s=0.0,
        angular_velocity_rad_s=[0.0, 0.0, 0.0],
        linear_acceleration_m_s2=[0.0, 0.0, 0.0],
    )
)
outputs = pipeline.add_raw_dvl_measurement(
    RawDvlMeasurement(
        timestamp_s=0.0,
        velocity_dvl_m_s=[0.2, 0.0, 0.0],
        position_nav_m=[1.5, 0.0, -0.2],
        valid_beams=[True, True, True, True],
        mode="bottom",
        status="valid",
    ),
)
outputs.extend(pipeline.flush())

for output in outputs:
    print(output.position_m)
    print(output.velocity_m_s)
    print(output.attitude_quat_wxyz)
    print(output.gyro_bias_rad_s)
    print(output.accel_bias_m_s2)
```

Run the tests with:

```sh
python3 -m unittest discover -s tests
```

Example config file shape:

```json
{
  "initial_state": {
    "position_m": [0.0, 0.0, 0.0],
    "velocity_m_s": [0.0, 0.0, 0.0],
    "attitude_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "gyro_bias_rad_s": [0.0, 0.0, 0.0],
    "accel_bias_m_s2": [0.0, 0.0, 0.0]
  },
  "calibration": {
    "dvl_to_body_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "dvl_lever_arm_body_m": [0.0, 0.0, 0.0],
    "dvl_velocity_scale": 1.0
  },
  "kalman": {
    "gravity_nav_m_s2": [0.0, 0.0, -9.80665],
    "default_dvl_velocity_std_m_s": 0.05,
    "default_dvl_position_std_m": 0.25
  },
  "dvl_quality": {
    "allowed_modes": ["bottom"],
    "max_velocity_m_s": 5.0
  },
  "synchronizer": {
    "max_delay_s": 0.25
  }
}
```
