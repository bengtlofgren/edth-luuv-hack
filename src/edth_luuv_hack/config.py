"""Configuration helpers for building the navigation fusion stack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import FrameCalibration
from .dvl_correction import DvlCorrectionLayer, DvlQualityConfig
from .dvl_imu_kalman import DvlImuKalmanLayer, KalmanConfig, NavigationState
from .synchronization import NavigationFusionPipeline, SynchronizerConfig


@dataclass(frozen=True)
class InitialNavigationConfig:
    """Initial state and optional covariance for the filter."""

    position_m: Sequence[float] = (0.0, 0.0, 0.0)
    velocity_m_s: Sequence[float] = (0.0, 0.0, 0.0)
    attitude_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    gyro_bias_rad_s: Sequence[float] = (0.0, 0.0, 0.0)
    accel_bias_m_s2: Sequence[float] = (0.0, 0.0, 0.0)
    covariance_diag: Sequence[float] | None = None

    def state(self) -> NavigationState:
        return NavigationState(
            position_m=_as_vec(self.position_m, 3, "position_m"),
            velocity_m_s=_as_vec(self.velocity_m_s, 3, "velocity_m_s"),
            attitude_quat_wxyz=_as_vec(self.attitude_quat_wxyz, 4, "attitude_quat_wxyz"),
            gyro_bias_rad_s=_as_vec(self.gyro_bias_rad_s, 3, "gyro_bias_rad_s"),
            accel_bias_m_s2=_as_vec(self.accel_bias_m_s2, 3, "accel_bias_m_s2"),
        )

    def covariance(self) -> np.ndarray | None:
        if self.covariance_diag is None:
            return None
        diagonal = _as_vec(self.covariance_diag, 15, "covariance_diag")
        return np.diag(diagonal)


def load_navigation_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON navigation configuration file."""

    with Path(path).open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("navigation config must be a JSON object")
    return config


def build_navigation_pipeline(config: Mapping[str, Any]) -> NavigationFusionPipeline:
    """Build a complete fusion pipeline from a mapping or loaded JSON config."""

    initial = _initial_config(config.get("initial_state", {}))
    kalman_config = _kalman_config(config.get("kalman", {}))
    calibration = _frame_calibration(config.get("calibration", {}))
    quality = _quality_config(config.get("dvl_quality", {}))
    synchronizer_config = _synchronizer_config(config.get("synchronizer", {}))

    kalman = DvlImuKalmanLayer(
        initial_state=initial.state(),
        initial_covariance=initial.covariance(),
        config=kalman_config,
    )
    return NavigationFusionPipeline(
        kalman=kalman,
        dvl_correction=DvlCorrectionLayer(calibration=calibration, quality=quality),
        config=synchronizer_config,
    )


def build_navigation_pipeline_from_json(path: str | Path) -> NavigationFusionPipeline:
    """Load a JSON config file and build a fusion pipeline."""

    return build_navigation_pipeline(load_navigation_config(path))


def _initial_config(value: Any) -> InitialNavigationConfig:
    mapping = _mapping(value, "initial_state")
    return InitialNavigationConfig(
        position_m=mapping.get("position_m", InitialNavigationConfig.position_m),
        velocity_m_s=mapping.get("velocity_m_s", InitialNavigationConfig.velocity_m_s),
        attitude_quat_wxyz=mapping.get("attitude_quat_wxyz", InitialNavigationConfig.attitude_quat_wxyz),
        gyro_bias_rad_s=mapping.get("gyro_bias_rad_s", InitialNavigationConfig.gyro_bias_rad_s),
        accel_bias_m_s2=mapping.get("accel_bias_m_s2", InitialNavigationConfig.accel_bias_m_s2),
        covariance_diag=mapping.get("covariance_diag"),
    )


def _kalman_config(value: Any) -> KalmanConfig:
    mapping = _mapping(value, "kalman")
    allowed = KalmanConfig.__dataclass_fields__
    return KalmanConfig(**{key: mapping[key] for key in mapping if key in allowed})


def _frame_calibration(value: Any) -> FrameCalibration:
    mapping = _mapping(value, "calibration")
    allowed = FrameCalibration.__dataclass_fields__
    return FrameCalibration(**{key: mapping[key] for key in mapping if key in allowed})


def _quality_config(value: Any) -> DvlQualityConfig:
    mapping = _mapping(value, "dvl_quality")
    allowed = DvlQualityConfig.__dataclass_fields__
    normalized = {key: mapping[key] for key in mapping if key in allowed}
    if "allowed_modes" in normalized:
        normalized["allowed_modes"] = tuple(normalized["allowed_modes"])
    if "invalid_status_values" in normalized:
        normalized["invalid_status_values"] = tuple(normalized["invalid_status_values"])
    return DvlQualityConfig(**normalized)


def _synchronizer_config(value: Any) -> SynchronizerConfig:
    mapping = _mapping(value, "synchronizer")
    allowed = SynchronizerConfig.__dataclass_fields__
    return SynchronizerConfig(**{key: mapping[key] for key in mapping if key in allowed})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _as_vec(value: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have {size} elements")
    return array
