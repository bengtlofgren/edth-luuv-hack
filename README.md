# edth-luuv-hack

This branch adds a navigation fusion layer for corrected DVL measurements and
IMU data.

The main entry point is `DvlImuKalmanLayer` in
`edth_luuv_hack.dvl_imu_kalman`. It performs:

1. IMU strapdown propagation from angular velocity and linear acceleration.
2. Corrected-DVL velocity and optional position measurement updates with an
   error-state Kalman filter.
3. Output of corrected position, velocity, attitude, gyroscope bias, and
   accelerometer bias estimates.

The IMU path predicts position by integrating acceleration and attitude over
time. The DVL input is expected to be corrected before entering this layer, for
example after beam quality checks, frame alignment, scale correction, and any
lever-arm compensation needed by the sensor stack. If the DVL stack also
produces an integrated/corrected position or displacement track, pass it as
`position_nav_m` so the filter can directly correct the IMU-predicted position.

```python
from edth_luuv_hack import (
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

Run the tests with:

```sh
python3 -m unittest discover -s tests
```
