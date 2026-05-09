"""Raw DVL validation and correction before Kalman fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .calibration import ArrayLike3, CovarianceLike3, FrameCalibration
from .dvl_imu_kalman import CorrectedDvlMeasurement


@dataclass(frozen=True)
class RawDvlMeasurement:
    """Raw or lightly decoded DVL measurement.

    Velocity is in the DVL sensor frame. Position, when present, is assumed to
    already be an integrated navigation-frame DVL track.
    """

    timestamp_s: float
    velocity_dvl_m_s: ArrayLike3 | None = None
    velocity_covariance_dvl: CovarianceLike3 | None = None
    position_nav_m: ArrayLike3 | None = None
    position_covariance_nav: CovarianceLike3 | None = None
    valid_beams: Sequence[bool] | None = None
    mode: str | None = "bottom"
    status: str | None = "valid"
    altitude_m: float | None = None
    velocity_scale_factor: float = 1.0


@dataclass(frozen=True)
class DvlQualityConfig:
    """Quality gates for corrected DVL measurements."""

    min_valid_beams: int = 3
    allowed_modes: tuple[str, ...] = ("bottom", "unknown")
    invalid_status_values: tuple[str, ...] = ("invalid", "bad", "error", "no_lock")
    min_altitude_m: float | None = 0.05
    max_altitude_m: float | None = 200.0
    max_velocity_m_s: float | None = 5.0
    velocity_variance_floor: float = 0.0025
    position_variance_floor: float = 0.01


class DvlCorrectionLayer:
    """Validate raw DVL data and convert it to Kalman-ready measurements."""

    def __init__(
        self,
        calibration: FrameCalibration | None = None,
        quality: DvlQualityConfig | None = None,
    ) -> None:
        self.calibration = calibration or FrameCalibration()
        self.quality = quality or DvlQualityConfig()

    def correct(
        self,
        raw: RawDvlMeasurement,
        angular_velocity_body_rad_s: ArrayLike3 | None = None,
    ) -> CorrectedDvlMeasurement | None:
        """Return a corrected DVL measurement, or None when quality gates fail."""

        if not self._passes_quality(raw):
            return None

        velocity_body = None
        velocity_covariance_body = None
        if raw.velocity_dvl_m_s is not None:
            velocity_dvl = raw.velocity_scale_factor * _as_vec3(raw.velocity_dvl_m_s, "velocity_dvl_m_s")
            velocity_body = self.calibration.velocity_dvl_to_body(velocity_dvl, angular_velocity_body_rad_s)
            if self.quality.max_velocity_m_s is not None:
                if np.linalg.norm(velocity_body) > self.quality.max_velocity_m_s:
                    return None
            if raw.velocity_covariance_dvl is not None:
                velocity_covariance_body = _floor_covariance(
                    self.calibration.covariance_dvl_to_body(raw.velocity_covariance_dvl),
                    self.quality.velocity_variance_floor,
                )

        position_nav = None
        position_covariance_nav = None
        if raw.position_nav_m is not None:
            position_nav = self.calibration.position_to_nav(raw.position_nav_m)
            if raw.position_covariance_nav is not None:
                position_covariance_nav = _floor_covariance(
                    _as_covariance(raw.position_covariance_nav),
                    self.quality.position_variance_floor,
                )

        if velocity_body is None and position_nav is None:
            return None

        return CorrectedDvlMeasurement(
            timestamp_s=float(raw.timestamp_s),
            velocity_body_m_s=velocity_body,
            covariance_body=velocity_covariance_body,
            position_nav_m=position_nav,
            position_covariance_nav=position_covariance_nav,
        )

    def _passes_quality(self, raw: RawDvlMeasurement) -> bool:
        status = (raw.status or "unknown").lower()
        if status in self.quality.invalid_status_values:
            return False

        mode = (raw.mode or "unknown").lower()
        if mode not in self.quality.allowed_modes:
            return False

        if raw.valid_beams is not None:
            valid_count = sum(bool(beam) for beam in raw.valid_beams)
            if valid_count < self.quality.min_valid_beams:
                return False

        if raw.altitude_m is not None:
            if self.quality.min_altitude_m is not None and raw.altitude_m < self.quality.min_altitude_m:
                return False
            if self.quality.max_altitude_m is not None and raw.altitude_m > self.quality.max_altitude_m:
                return False

        return True


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


def _floor_covariance(covariance: np.ndarray, variance_floor: float) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    for axis in range(3):
        covariance[axis, axis] = max(covariance[axis, axis], variance_floor)
    return covariance
