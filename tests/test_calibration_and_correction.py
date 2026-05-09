import unittest

import numpy as np

from edth_luuv_hack import (
    DvlCorrectionLayer,
    DvlQualityConfig,
    FrameCalibration,
    RawDvlMeasurement,
    rotation_matrix_from_euler_rad,
)


class CalibrationAndCorrectionTest(unittest.TestCase):
    def test_rotation_scale_and_lever_arm_correction(self) -> None:
        rotation = rotation_matrix_from_euler_rad(0.0, 0.0, np.pi / 2.0)
        calibration = FrameCalibration(
            dvl_to_body_rotation=rotation,
            dvl_velocity_scale=2.0,
            dvl_lever_arm_body_m=(0.0, 1.0, 0.0),
        )

        corrected = calibration.velocity_dvl_to_body(
            velocity_dvl_m_s=(1.0, 0.0, 0.0),
            angular_velocity_body_rad_s=(0.0, 0.0, 1.0),
        )

        np.testing.assert_allclose(corrected, np.array([1.0, 2.0, 0.0]), atol=1.0e-12)

    def test_dvl_correction_rejects_bad_quality(self) -> None:
        correction = DvlCorrectionLayer(quality=DvlQualityConfig(min_valid_beams=3))

        result = correction.correct(
            RawDvlMeasurement(
                timestamp_s=0.0,
                velocity_dvl_m_s=(0.1, 0.0, 0.0),
                valid_beams=(True, False, True, False),
            )
        )

        self.assertIsNone(result)

    def test_dvl_correction_outputs_body_velocity_and_position(self) -> None:
        correction = DvlCorrectionLayer()

        corrected = correction.correct(
            RawDvlMeasurement(
                timestamp_s=1.0,
                velocity_dvl_m_s=(0.5, 0.0, 0.0),
                velocity_covariance_dvl=np.eye(3) * 1.0e-5,
                position_nav_m=(1.0, 2.0, -0.5),
                position_covariance_nav=np.eye(3) * 1.0e-5,
                valid_beams=(True, True, True, True),
                altitude_m=3.0,
            )
        )

        self.assertIsNotNone(corrected)
        assert corrected is not None
        np.testing.assert_allclose(corrected.velocity_body_m_s, np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(corrected.position_nav_m, np.array([1.0, 2.0, -0.5]))
        self.assertGreaterEqual(corrected.covariance_body[0, 0], 0.0025)
        self.assertGreaterEqual(corrected.position_covariance_nav[0, 0], 0.01)


if __name__ == "__main__":
    unittest.main()
