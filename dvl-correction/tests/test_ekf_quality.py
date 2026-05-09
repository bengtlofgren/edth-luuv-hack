"""Quality tests for the DVL/IMU error-state Kalman filter.

These exercise the EKF beyond happy-path API: drift reduction over a
trajectory, covariance behavior on predict/update, accel-bias observability
through DVL position fixes, and the Mahalanobis outlier gate.
"""

import unittest

import numpy as np

from dvl_correction import (
    CorrectedDvlMeasurement,
    DvlImuKalmanLayer,
    ImuSample,
    KalmanConfig,
)


def _run_stationary_with_accel_bias(
    *,
    duration_s: float,
    imu_rate_hz: float,
    dvl_rate_hz: float | None,
    accel_bias_true: np.ndarray,
    accel_noise_std: float,
    dvl_position_noise_std: float,
    seed: int,
) -> DvlImuKalmanLayer:
    """Run a stationary, identity-attitude scenario with a constant accel bias.

    Ground truth is `position = velocity = 0`. The IMU reports
    `accel_bias_true + N(0, accel_noise_std)` so the filter must learn the
    bias from DVL position fixes (when enabled). Gravity and the Mahalanobis
    gate are disabled so the test isolates fusion behaviour.
    """

    rng = np.random.default_rng(seed)
    layer = DvlImuKalmanLayer(
        config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0), mahalanobis_gate=None),
    )

    dt_imu = 1.0 / imu_rate_hz
    dt_dvl = None if dvl_rate_hz is None else 1.0 / dvl_rate_hz
    next_dvl_t = float("inf") if dt_dvl is None else dt_dvl

    n_steps = int(round(duration_s / dt_imu)) + 1
    for i in range(n_steps):
        t = i * dt_imu
        measured_accel = accel_bias_true + accel_noise_std * rng.standard_normal(3)
        imu = ImuSample(
            timestamp_s=t,
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
            linear_acceleration_m_s2=measured_accel,
        )
        dvl = None
        if t >= next_dvl_t:
            assert dt_dvl is not None
            next_dvl_t = t + dt_dvl
            measured_pos = dvl_position_noise_std * rng.standard_normal(3)
            dvl = CorrectedDvlMeasurement(
                timestamp_s=t,
                position_nav_m=measured_pos,
                position_covariance_nav=np.eye(3) * dvl_position_noise_std**2,
            )
        layer.process(imu, dvl)

    return layer


class EkfDriftReductionTest(unittest.TestCase):
    def test_dvl_position_fusion_bounds_drift_when_imu_alone_diverges(self) -> None:
        accel_bias_true = np.array([0.05, 0.0, 0.0])
        common = dict(
            duration_s=30.0,
            imu_rate_hz=100.0,
            accel_bias_true=accel_bias_true,
            accel_noise_std=0.005,
            dvl_position_noise_std=0.1,
            seed=0,
        )

        imu_only = _run_stationary_with_accel_bias(dvl_rate_hz=None, **common)
        fused = _run_stationary_with_accel_bias(dvl_rate_hz=1.0, **common)

        imu_only_error = float(np.linalg.norm(imu_only.state.position_m))
        fused_error = float(np.linalg.norm(fused.state.position_m))

        # Without DVL, a 0.05 m/s² bias accumulates to ~0.5*0.05*30² = 22.5 m.
        self.assertGreater(imu_only_error, 10.0)
        # With 1 Hz DVL position fixes, the position error stays sub-metre and
        # is at least 10x smaller than dead-reckoning alone.
        self.assertLess(fused_error, 1.0)
        self.assertLess(fused_error, 0.1 * imu_only_error)


class EkfCovarianceBehaviorTest(unittest.TestCase):
    def test_position_covariance_grows_in_prediction_and_shrinks_after_update(self) -> None:
        layer = DvlImuKalmanLayer(
            config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0)),
        )

        # Seed the timestamp with a quiescent sample.
        layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            )
        )
        trace_pos_initial = float(np.trace(layer.covariance[0:3, 0:3]))

        # 1 second of pure IMU integration must grow position uncertainty
        # because position variance accrues from velocity variance via dt.
        layer.process(
            ImuSample(
                timestamp_s=1.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            )
        )
        trace_pos_after_predict = float(np.trace(layer.covariance[0:3, 0:3]))
        self.assertGreater(trace_pos_after_predict, trace_pos_initial)

        # A position measurement must shrink position uncertainty.
        output = layer.process(
            ImuSample(
                timestamp_s=1.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            ),
            CorrectedDvlMeasurement(
                timestamp_s=1.0,
                position_nav_m=(0.0, 0.0, 0.0),
                position_covariance_nav=np.eye(3) * 0.01,
            ),
        )
        self.assertTrue(output.dvl_update_applied)
        trace_pos_after_update = float(np.trace(layer.covariance[0:3, 0:3]))
        self.assertLess(trace_pos_after_update, trace_pos_after_predict)


class EkfBiasLearningTest(unittest.TestCase):
    def test_accel_bias_estimate_converges_to_injected_bias(self) -> None:
        accel_bias_true = np.array([0.05, 0.0, 0.0])
        layer = _run_stationary_with_accel_bias(
            duration_s=60.0,
            imu_rate_hz=100.0,
            dvl_rate_hz=2.0,
            accel_bias_true=accel_bias_true,
            accel_noise_std=0.005,
            dvl_position_noise_std=0.05,
            seed=0,
        )

        np.testing.assert_allclose(
            layer.state.accel_bias_m_s2,
            accel_bias_true,
            atol=0.025,
        )


class EkfMahalanobisGateTest(unittest.TestCase):
    def test_extreme_outlier_is_rejected_and_normal_measurement_is_accepted(self) -> None:
        layer = DvlImuKalmanLayer(
            config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0)),
        )

        layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            )
        )

        velocity_before = layer.state.velocity_m_s.copy()
        outlier_output = layer.process(
            ImuSample(
                timestamp_s=0.1,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            ),
            CorrectedDvlMeasurement(
                timestamp_s=0.1,
                velocity_body_m_s=(1000.0, 0.0, 0.0),
                covariance_body=np.eye(3) * 0.01,
            ),
        )

        self.assertFalse(outlier_output.dvl_update_applied)
        np.testing.assert_allclose(outlier_output.velocity_m_s, velocity_before)

        normal_output = layer.process(
            ImuSample(
                timestamp_s=0.2,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            ),
            CorrectedDvlMeasurement(
                timestamp_s=0.2,
                velocity_body_m_s=(0.5, 0.0, 0.0),
                covariance_body=np.eye(3) * 0.01,
            ),
        )

        self.assertTrue(normal_output.dvl_update_applied)


if __name__ == "__main__":
    unittest.main()
