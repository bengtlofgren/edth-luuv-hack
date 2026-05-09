"""DVL velocity dead-reckoning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


ArrayLike3 = Sequence[float] | np.ndarray
CovarianceLike3 = float | Sequence[Sequence[float]] | np.ndarray


@dataclass
class DvlTrackState:
    """Integrated DVL track state in the navigation frame."""

    timestamp_s: float | None = None
    position_nav_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance_nav: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass(frozen=True)
class DvlTrackConfig:
    """Settings for integrating corrected DVL velocity into position."""

    initial_position_nav_m: ArrayLike3 = (0.0, 0.0, 0.0)
    initial_position_covariance_nav: CovarianceLike3 = 1.0
    velocity_process_variance: float = 0.01


class DvlDeadReckoningTrack:
    """Integrate corrected DVL body-frame velocity into a nav-frame track."""

    def __init__(self, config: DvlTrackConfig | None = None) -> None:
        self.config = config or DvlTrackConfig()
        self.state = DvlTrackState(
            position_nav_m=_as_vec3(self.config.initial_position_nav_m, "initial_position_nav_m"),
            covariance_nav=_as_covariance(self.config.initial_position_covariance_nav),
        )

    def update(
        self,
        timestamp_s: float,
        velocity_body_m_s: ArrayLike3,
        attitude_quat_wxyz: Sequence[float] | np.ndarray,
        velocity_covariance_body: CovarianceLike3 | None = None,
    ) -> DvlTrackState:
        """Advance the integrated DVL track to `timestamp_s`."""

        timestamp_s = float(timestamp_s)
        velocity_body = _as_vec3(velocity_body_m_s, "velocity_body_m_s")
        rotation_body_to_nav = _quat_to_rotation_matrix(attitude_quat_wxyz)
        velocity_nav = rotation_body_to_nav @ velocity_body

        if self.state.timestamp_s is None:
            self.state.timestamp_s = timestamp_s
            return self.copy_state()

        dt = timestamp_s - self.state.timestamp_s
        if dt < 0.0:
            raise ValueError("DVL track timestamps must be monotonic")
        self.state.timestamp_s = timestamp_s
        if dt == 0.0:
            return self.copy_state()

        self.state.position_nav_m = self.state.position_nav_m + velocity_nav * dt

        velocity_covariance_nav = (
            np.eye(3) * self.config.velocity_process_variance
            if velocity_covariance_body is None
            else rotation_body_to_nav @ _as_covariance(velocity_covariance_body) @ rotation_body_to_nav.T
        )
        self.state.covariance_nav = (
            self.state.covariance_nav
            + velocity_covariance_nav * dt * dt
            + np.eye(3) * self.config.velocity_process_variance * dt
        )
        self.state.covariance_nav = 0.5 * (self.state.covariance_nav + self.state.covariance_nav.T)
        return self.copy_state()

    def copy_state(self) -> DvlTrackState:
        return DvlTrackState(
            timestamp_s=self.state.timestamp_s,
            position_nav_m=self.state.position_nav_m.copy(),
            covariance_nav=self.state.covariance_nav.copy(),
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


def _quat_to_rotation_matrix(quaternion: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(quaternion, dtype=float)
    if array.shape != (4,):
        raise ValueError("attitude_quat_wxyz must be a 4 element quaternion")
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("attitude quaternion cannot have zero norm")
    w, x, y, z = array / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
