"""Frame calibration helpers for DVL/IMU fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


ArrayLike3 = Sequence[float] | np.ndarray
CovarianceLike3 = float | Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class FrameCalibration:
    """Calibration between DVL, IMU/body, and navigation frames.

    `dvl_to_body_rotation` maps vectors from the DVL frame into the IMU/body
    frame. `dvl_lever_arm_body_m` is the vector from the IMU origin to the DVL
    transducer origin, expressed in the IMU/body frame.
    """

    dvl_to_body_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    dvl_velocity_scale: float = 1.0
    dvl_lever_arm_body_m: ArrayLike3 = (0.0, 0.0, 0.0)
    position_offset_nav_m: ArrayLike3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        rotation = np.asarray(self.dvl_to_body_rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("dvl_to_body_rotation must have shape (3, 3)")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6):
            raise ValueError("dvl_to_body_rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
            raise ValueError("dvl_to_body_rotation must be right handed")

        object.__setattr__(self, "dvl_to_body_rotation", rotation)
        object.__setattr__(self, "dvl_lever_arm_body_m", _as_vec3(self.dvl_lever_arm_body_m, "dvl_lever_arm_body_m"))
        object.__setattr__(self, "position_offset_nav_m", _as_vec3(self.position_offset_nav_m, "position_offset_nav_m"))

    def velocity_dvl_to_body(
        self,
        velocity_dvl_m_s: ArrayLike3,
        angular_velocity_body_rad_s: ArrayLike3 | None = None,
    ) -> np.ndarray:
        """Convert DVL-frame velocity to IMU/body origin velocity."""

        velocity_body = self.dvl_velocity_scale * (
            self.dvl_to_body_rotation @ _as_vec3(velocity_dvl_m_s, "velocity_dvl_m_s")
        )
        if angular_velocity_body_rad_s is None:
            return velocity_body
        omega_body = _as_vec3(angular_velocity_body_rad_s, "angular_velocity_body_rad_s")
        return velocity_body - np.cross(omega_body, self.dvl_lever_arm_body_m)

    def covariance_dvl_to_body(self, covariance_dvl: CovarianceLike3) -> np.ndarray:
        """Rotate DVL velocity covariance into the IMU/body frame."""

        covariance = _as_covariance(covariance_dvl)
        scale_squared = self.dvl_velocity_scale * self.dvl_velocity_scale
        return scale_squared * self.dvl_to_body_rotation @ covariance @ self.dvl_to_body_rotation.T

    def position_to_nav(self, position_nav_m: ArrayLike3) -> np.ndarray:
        """Apply a static navigation-frame offset to a corrected DVL track."""

        return _as_vec3(position_nav_m, "position_nav_m") + self.position_offset_nav_m


def rotation_matrix_from_euler_rad(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Create a right-handed XYZ roll-pitch-yaw rotation matrix."""

    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _as_vec3(value: ArrayLike3, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3 element vector")
    return array


def _as_covariance(value: CovarianceLike3) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.ndim == 0:
        return np.eye(3) * float(covariance)
    if covariance.shape != (3, 3):
        raise ValueError("covariance must have shape (3, 3)")
    return 0.5 * (covariance + covariance.T)
