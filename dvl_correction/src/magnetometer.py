"""Magnetometer correction for the DVL/IMU fusion stack.

The correction layer applies hard-iron and soft-iron calibration to a raw
body-frame magnetic field reading, gates on field magnitude, tilt-compensates
using the filter's current roll/pitch, and produces a magnetic-frame yaw with
an associated standard deviation. The Kalman layer combines the magnetic yaw
with its declination state to update true heading.

Conventions:

* Navigation frame is ENU (x=East, y=North, z=Up). Yaw is the math angle of
  body-x in the nav frame (counter-clockwise positive about z, zero pointing
  East), matching ``rotation_matrix_from_euler_rad`` and the quaternion
  utilities in ``dvl_imu_kalman``.
* Declination is defined so that ``yaw_true = yaw_magnetic + declination``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .dvl_imu_kalman import CorrectedMagnetometerMeasurement


ArrayLike3 = Sequence[float] | np.ndarray
Matrix3Like = Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class RawMagnetometerMeasurement:
    """Raw magnetometer reading in the body frame."""

    timestamp_s: float
    magnetic_field_body_uT: ArrayLike3
    status: str | None = "valid"


@dataclass(frozen=True)
class MagnetometerCalibration:
    """Hard-iron, soft-iron, and expected-field calibration parameters.

    ``hard_iron_offset_uT`` is subtracted from raw readings. ``soft_iron_matrix``
    is then applied. ``expected_field_magnitude_uT`` is the nominal field
    magnitude at the deployment area, used by the quality gate.
    """

    hard_iron_offset_uT: ArrayLike3 = (0.0, 0.0, 0.0)
    soft_iron_matrix: Matrix3Like = field(default_factory=lambda: np.eye(3))
    expected_field_magnitude_uT: float = 50.0

    def __post_init__(self) -> None:
        offset = np.asarray(self.hard_iron_offset_uT, dtype=float)
        if offset.shape != (3,):
            raise ValueError("hard_iron_offset_uT must be a 3 element vector")
        soft_iron = np.asarray(self.soft_iron_matrix, dtype=float)
        if soft_iron.shape != (3, 3):
            raise ValueError("soft_iron_matrix must have shape (3, 3)")
        if not np.isfinite(self.expected_field_magnitude_uT) or self.expected_field_magnitude_uT <= 0.0:
            raise ValueError("expected_field_magnitude_uT must be positive")
        object.__setattr__(self, "hard_iron_offset_uT", offset)
        object.__setattr__(self, "soft_iron_matrix", soft_iron)

    def apply(self, raw_field_uT: ArrayLike3) -> np.ndarray:
        """Return the calibrated body-frame magnetic field vector."""

        raw = np.asarray(raw_field_uT, dtype=float)
        if raw.shape != (3,):
            raise ValueError("magnetic_field_body_uT must be a 3 element vector")
        return self.soft_iron_matrix @ (raw - self.hard_iron_offset_uT)


@dataclass(frozen=True)
class MagnetometerQualityConfig:
    """Quality gates for magnetometer measurements."""

    magnitude_tolerance_fraction: float = 0.2
    invalid_status_values: tuple[str, ...] = ("invalid", "bad", "error")
    yaw_std_floor_rad: float = 0.05


class MagnetometerCorrectionLayer:
    """Validate a raw magnetometer reading and reduce it to a magnetic yaw."""

    def __init__(
        self,
        calibration: MagnetometerCalibration | None = None,
        quality: MagnetometerQualityConfig | None = None,
        default_yaw_std_rad: float = 0.087,
    ) -> None:
        self.calibration = calibration or MagnetometerCalibration()
        self.quality = quality or MagnetometerQualityConfig()
        if default_yaw_std_rad <= 0.0:
            raise ValueError("default_yaw_std_rad must be positive")
        self.default_yaw_std_rad = float(default_yaw_std_rad)

    def correct(
        self,
        raw: RawMagnetometerMeasurement,
        attitude_quat_wxyz: ArrayLike3,
    ) -> CorrectedMagnetometerMeasurement | None:
        """Return a corrected magnetometer yaw, or None when the gate fails."""

        if (raw.status or "unknown").lower() in self.quality.invalid_status_values:
            return None

        field_body = self.calibration.apply(raw.magnetic_field_body_uT)
        magnitude = float(np.linalg.norm(field_body))
        expected = float(self.calibration.expected_field_magnitude_uT)
        if abs(magnitude - expected) > self.quality.magnitude_tolerance_fraction * expected:
            return None
        if magnitude <= 0.0:
            return None

        roll_rad, pitch_rad = _roll_pitch_from_quaternion(attitude_quat_wxyz)
        leveled = _level_body_to_yawed_nav(field_body, roll_rad, pitch_rad)
        if abs(leveled[0]) < 1.0e-12 and abs(leveled[1]) < 1.0e-12:
            return None
        yaw_magnetic = float(np.arctan2(leveled[0], leveled[1]))

        yaw_std = max(self.default_yaw_std_rad, self.quality.yaw_std_floor_rad)
        return CorrectedMagnetometerMeasurement(
            timestamp_s=float(raw.timestamp_s),
            yaw_magnetic_rad=yaw_magnetic,
            yaw_std_rad=yaw_std,
        )


def _roll_pitch_from_quaternion(quaternion: ArrayLike3) -> tuple[float, float]:
    """Extract roll (x-axis) and pitch (y-axis) from a w-x-y-z quaternion.

    Uses the same Tait-Bryan ZYX decomposition as
    ``rotation_matrix_from_euler_rad``.
    """

    quat = np.asarray(quaternion, dtype=float)
    if quat.shape != (4,):
        raise ValueError("attitude_quat_wxyz must be a 4 element quaternion")
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("attitude quaternion cannot have zero norm")
    w, x, y, z = quat / norm

    sin_pitch = -2.0 * (x * z - y * w)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = float(np.arcsin(sin_pitch))
    roll = float(np.arctan2(2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)))
    return roll, pitch


def _level_body_to_yawed_nav(
    field_body: np.ndarray,
    roll_rad: float,
    pitch_rad: float,
) -> np.ndarray:
    """Rotate a body-frame vector into a yaw-only navigation-aligned frame.

    The output frame is the navigation frame after removing yaw, i.e. the
    body frame after un-rolling and un-pitching. ``yaw = 0`` corresponds to
    body-x pointing East.
    """

    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rotation_y @ rotation_x @ field_body
