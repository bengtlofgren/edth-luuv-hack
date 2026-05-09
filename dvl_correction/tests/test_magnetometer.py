"""Quality tests for the magnetometer correction layer and KF yaw fusion."""

import unittest

import numpy as np

from dvl_correction import (
    CorrectedDvlMeasurement,
    CorrectedMagnetometerMeasurement,
    DvlImuKalmanLayer,
    ImuSample,
    KalmanConfig,
    MagnetometerCalibration,
    MagnetometerCorrectionLayer,
    MagnetometerQualityConfig,
    NavigationState,
    RawMagnetometerMeasurement,
    rotation_matrix_from_euler_rad,
)


def _quat_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a wxyz quaternion for a ZYX Tait-Bryan rotation."""

    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ]
    )


def _synthetic_field_body_uT(
    roll: float, pitch: float, yaw: float, horizontal_uT: float, down_uT: float
) -> np.ndarray:
    """Project a nav-frame magnetic field into the body frame.

    Nav frame is ENU. Magnetic north points along nav-y. ``down_uT`` is
    positive when the field dips into the Earth (so ENU z-component is
    ``-down_uT``).
    """

    rotation_body_to_nav = rotation_matrix_from_euler_rad(roll, pitch, yaw)
    field_nav = np.array([0.0, horizontal_uT, -down_uT])
    return rotation_body_to_nav.T @ field_nav


class MagnetometerCorrectionTest(unittest.TestCase):
    def test_yaw_round_trips_for_level_attitude(self) -> None:
        layer = MagnetometerCorrectionLayer(
            calibration=MagnetometerCalibration(expected_field_magnitude_uT=50.0),
        )
        for true_yaw in (0.0, np.pi / 4, -np.pi / 3, np.pi - 0.1):
            quat = _quat_from_euler_zyx(0.0, 0.0, true_yaw)
            field_body = _synthetic_field_body_uT(0.0, 0.0, true_yaw, 25.0, 43.3)
            corrected = layer.correct(
                RawMagnetometerMeasurement(timestamp_s=0.0, magnetic_field_body_uT=field_body),
                attitude_quat_wxyz=quat,
            )
            assert corrected is not None
            self.assertAlmostEqual(corrected.yaw_magnetic_rad, true_yaw, places=6)

    def test_yaw_round_trips_with_roll_and_pitch(self) -> None:
        layer = MagnetometerCorrectionLayer(
            calibration=MagnetometerCalibration(expected_field_magnitude_uT=50.0),
        )
        roll, pitch, true_yaw = 0.2, -0.15, 0.7
        quat = _quat_from_euler_zyx(roll, pitch, true_yaw)
        field_body = _synthetic_field_body_uT(roll, pitch, true_yaw, 25.0, 43.3)
        corrected = layer.correct(
            RawMagnetometerMeasurement(timestamp_s=0.0, magnetic_field_body_uT=field_body),
            attitude_quat_wxyz=quat,
        )
        assert corrected is not None
        self.assertAlmostEqual(corrected.yaw_magnetic_rad, true_yaw, places=6)

    def test_magnitude_anomaly_rejected(self) -> None:
        layer = MagnetometerCorrectionLayer(
            calibration=MagnetometerCalibration(expected_field_magnitude_uT=50.0),
            quality=MagnetometerQualityConfig(magnitude_tolerance_fraction=0.2),
        )
        quat = _quat_from_euler_zyx(0.0, 0.0, 0.0)
        anomaly_field_body = _synthetic_field_body_uT(0.0, 0.0, 0.0, 60.0, 60.0)
        result = layer.correct(
            RawMagnetometerMeasurement(timestamp_s=0.0, magnetic_field_body_uT=anomaly_field_body),
            attitude_quat_wxyz=quat,
        )
        self.assertIsNone(result)

    def test_invalid_status_rejected(self) -> None:
        layer = MagnetometerCorrectionLayer(
            calibration=MagnetometerCalibration(expected_field_magnitude_uT=50.0),
        )
        quat = _quat_from_euler_zyx(0.0, 0.0, 0.0)
        field_body = _synthetic_field_body_uT(0.0, 0.0, 0.0, 25.0, 43.3)
        result = layer.correct(
            RawMagnetometerMeasurement(
                timestamp_s=0.0, magnetic_field_body_uT=field_body, status="bad"
            ),
            attitude_quat_wxyz=quat,
        )
        self.assertIsNone(result)

    def test_hard_iron_offset_corrected_before_yaw_extraction(self) -> None:
        offset_uT = np.array([5.0, -3.0, 2.0])
        layer = MagnetometerCorrectionLayer(
            calibration=MagnetometerCalibration(
                hard_iron_offset_uT=offset_uT,
                expected_field_magnitude_uT=50.0,
            ),
        )
        true_yaw = 0.5
        quat = _quat_from_euler_zyx(0.0, 0.0, true_yaw)
        field_body = _synthetic_field_body_uT(0.0, 0.0, true_yaw, 25.0, 43.3)
        raw_with_bias = field_body + offset_uT
        corrected = layer.correct(
            RawMagnetometerMeasurement(timestamp_s=0.0, magnetic_field_body_uT=raw_with_bias),
            attitude_quat_wxyz=quat,
        )
        assert corrected is not None
        self.assertAlmostEqual(corrected.yaw_magnetic_rad, true_yaw, places=6)


class KalmanMagnetometerTest(unittest.TestCase):
    def _stationary_imu_sample(self, t: float, gyro_z_bias: float = 0.0) -> ImuSample:
        return ImuSample(
            timestamp_s=t,
            angular_velocity_rad_s=(0.0, 0.0, gyro_z_bias),
            linear_acceleration_m_s2=(0.0, 0.0, 0.0),
        )

    def test_yaw_observability_bounds_drift_under_gyro_bias(self) -> None:
        gyro_bias_z = 1.0e-3
        config = KalmanConfig(
            gravity_nav_m_s2=(0.0, 0.0, 0.0),
            initial_declination_std_rad=1.0e-6,
            mahalanobis_gate=None,
        )

        imu_only = DvlImuKalmanLayer(config=config)
        with_mag = DvlImuKalmanLayer(config=config)

        duration_s = 60.0
        dt_imu = 0.01
        next_mag_t = 1.0
        for i in range(int(duration_s / dt_imu) + 1):
            t = i * dt_imu
            imu = self._stationary_imu_sample(t, gyro_z_bias=gyro_bias_z)
            imu_only.process(imu)
            mag = None
            if t >= next_mag_t:
                next_mag_t = t + 1.0
                mag = CorrectedMagnetometerMeasurement(
                    timestamp_s=t, yaw_magnetic_rad=0.0, yaw_std_rad=0.05
                )
            with_mag.process(imu, corrected_magnetometer=mag)

        from dvl_correction.dvl_imu_kalman import _yaw_from_quaternion

        yaw_drift_imu_only = abs(_yaw_from_quaternion(imu_only.state.attitude_quat_wxyz))
        yaw_drift_with_mag = abs(_yaw_from_quaternion(with_mag.state.attitude_quat_wxyz))

        self.assertGreater(yaw_drift_imu_only, 0.03)
        self.assertLess(yaw_drift_with_mag, 0.05)
        self.assertLess(yaw_drift_with_mag, 0.5 * yaw_drift_imu_only)

    def test_magnetometer_mahalanobis_gate_rejects_huge_yaw_outlier(self) -> None:
        config = KalmanConfig(
            gravity_nav_m_s2=(0.0, 0.0, 0.0),
            initial_attitude_std_rad=0.01,
            initial_declination_std_rad=0.01,
        )
        layer = DvlImuKalmanLayer(config=config)
        layer.process(self._stationary_imu_sample(0.0))

        attitude_before = layer.state.attitude_quat_wxyz.copy()
        declination_before = layer.state.declination_rad

        outlier = CorrectedMagnetometerMeasurement(
            timestamp_s=0.1, yaw_magnetic_rad=np.pi, yaw_std_rad=0.05
        )
        output = layer.process(self._stationary_imu_sample(0.1), corrected_magnetometer=outlier)

        self.assertFalse(output.magnetometer_update_applied)
        np.testing.assert_allclose(layer.state.attitude_quat_wxyz, attitude_before, atol=1.0e-12)
        self.assertAlmostEqual(layer.state.declination_rad, declination_before, places=12)

    def test_declination_state_converges_to_truth(self) -> None:
        true_declination = 0.175
        config = KalmanConfig(
            gravity_nav_m_s2=(0.0, 0.0, 0.0),
            initial_attitude_std_rad=0.001,
            initial_declination_std_rad=0.5,
            mahalanobis_gate=None,
        )
        layer = DvlImuKalmanLayer(config=config)

        for i in range(120):
            t = i * 0.5
            imu = self._stationary_imu_sample(t)
            mag = CorrectedMagnetometerMeasurement(
                timestamp_s=t,
                yaw_magnetic_rad=-true_declination,
                yaw_std_rad=0.05,
            )
            layer.process(imu, corrected_magnetometer=mag)

        self.assertAlmostEqual(layer.state.declination_rad, true_declination, delta=0.02)


class AnisotropicDvlCovarianceTest(unittest.TestCase):
    def test_filter_trusts_strong_axis_more_than_weak_axis(self) -> None:
        config = KalmanConfig(
            gravity_nav_m_s2=(0.0, 0.0, 0.0),
            mahalanobis_gate=None,
            initial_velocity_std_m_s=1.0,
        )
        layer = DvlImuKalmanLayer(config=config)
        layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            )
        )

        covariance_body = np.diag([1.0e-4, 4.0, 0.0025])
        measurement = CorrectedDvlMeasurement(
            timestamp_s=0.0,
            velocity_body_m_s=(1.0, 1.0, 1.0),
            covariance_body=covariance_body,
        )
        output = layer.process(
            ImuSample(
                timestamp_s=0.001,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            ),
            corrected_dvl=measurement,
        )

        self.assertGreater(output.velocity_m_s[0], 0.95)
        self.assertLess(output.velocity_m_s[1], 0.5)
        self.assertGreater(output.velocity_m_s[2], 0.7)


if __name__ == "__main__":
    unittest.main()
