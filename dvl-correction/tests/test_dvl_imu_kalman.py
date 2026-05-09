import unittest

import numpy as np

from dvl_correction import CorrectedDvlMeasurement, DvlImuKalmanLayer, ImuSample, KalmanConfig


class DvlImuKalmanLayerTest(unittest.TestCase):
    def test_imu_propagation_integrates_acceleration(self) -> None:
        layer = DvlImuKalmanLayer(config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0)))

        layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(1.0, 0.0, 0.0),
            )
        )
        output = layer.process(
            ImuSample(
                timestamp_s=1.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(1.0, 0.0, 0.0),
            )
        )

        np.testing.assert_allclose(output.position_m, np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(output.velocity_m_s, np.array([1.0, 0.0, 0.0]))

    def test_corrected_dvl_update_pulls_velocity_toward_measurement(self) -> None:
        layer = DvlImuKalmanLayer(config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0)))

        output = layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            ),
            CorrectedDvlMeasurement(
                timestamp_s=0.0,
                velocity_body_m_s=(1.0, 0.0, 0.0),
                covariance_body=np.eye(3) * 0.01,
            ),
        )

        self.assertTrue(output.dvl_update_applied)
        self.assertGreater(output.velocity_m_s[0], 0.9)
        self.assertLess(abs(output.velocity_m_s[1]), 1.0e-9)
        self.assertLess(abs(output.velocity_m_s[2]), 1.0e-9)

    def test_corrected_dvl_position_update_pulls_position_toward_measurement(self) -> None:
        layer = DvlImuKalmanLayer(config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0)))

        layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(1.0, 0.0, 0.0),
            )
        )
        output = layer.process(
            ImuSample(
                timestamp_s=1.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(1.0, 0.0, 0.0),
            ),
            CorrectedDvlMeasurement(
                timestamp_s=1.0,
                position_nav_m=(0.2, 0.0, 0.0),
                position_covariance_nav=np.eye(3) * 0.01,
            ),
        )

        self.assertTrue(output.dvl_update_applied)
        self.assertLess(output.position_m[0], 0.5)
        self.assertGreater(output.position_m[0], 0.2)

    def test_corrected_dvl_can_fuse_velocity_and_position_together(self) -> None:
        layer = DvlImuKalmanLayer(config=KalmanConfig(gravity_nav_m_s2=(0.0, 0.0, 0.0)))

        output = layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            ),
            CorrectedDvlMeasurement(
                timestamp_s=0.0,
                velocity_body_m_s=(1.0, 0.0, 0.0),
                covariance_body=np.eye(3) * 0.01,
                position_nav_m=(2.0, 0.0, 0.0),
                position_covariance_nav=np.eye(3) * 0.01,
            ),
        )

        self.assertTrue(output.dvl_update_applied)
        self.assertGreater(output.velocity_m_s[0], 0.9)
        self.assertGreater(output.position_m[0], 1.9)

    def test_output_contains_imu_bias_estimates_and_covariance(self) -> None:
        layer = DvlImuKalmanLayer()
        output = layer.process(
            ImuSample(
                timestamp_s=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                linear_acceleration_m_s2=(0.0, 0.0, 0.0),
            )
        )

        self.assertEqual(output.gyro_bias_rad_s.shape, (3,))
        self.assertEqual(output.accel_bias_m_s2.shape, (3,))
        self.assertEqual(output.covariance.shape, (15, 15))

    def test_rejects_bad_dvl_covariance_shape(self) -> None:
        layer = DvlImuKalmanLayer()

        with self.assertRaises(ValueError):
            layer.process(
                ImuSample(
                    timestamp_s=0.0,
                    angular_velocity_rad_s=(0.0, 0.0, 0.0),
                    linear_acceleration_m_s2=(0.0, 0.0, 0.0),
                ),
                CorrectedDvlMeasurement(
                    timestamp_s=0.0,
                    velocity_body_m_s=(0.0, 0.0, 0.0),
                    covariance_body=np.eye(2),
                ),
            )

    def test_rejects_dvl_measurement_without_velocity_or_position(self) -> None:
        with self.assertRaises(ValueError):
            CorrectedDvlMeasurement(timestamp_s=0.0)


if __name__ == "__main__":
    unittest.main()
