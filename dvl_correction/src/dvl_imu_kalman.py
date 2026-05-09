"""DVL/IMU error-state Kalman filter layer.

The nominal state is position, velocity, body-to-navigation attitude, and IMU
biases. The covariance is maintained for a 15 element error state ordered as:

    position, velocity, attitude error, gyro bias, accelerometer bias

DVL measurements are assumed to be corrected before they are passed to this
layer. Velocity is expressed in the IMU/body frame. Position is expressed in the
navigation frame and can come from an integrated/corrected DVL track estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


ArrayLike3 = Sequence[float] | np.ndarray
CovarianceLike3 = float | Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class ImuSample:
    """Single IMU sample used for state propagation."""

    timestamp_s: float
    angular_velocity_rad_s: ArrayLike3
    linear_acceleration_m_s2: ArrayLike3


@dataclass(frozen=True)
class CorrectedDvlMeasurement:
    """Corrected DVL measurements used to update the IMU-predicted state."""

    timestamp_s: float
    velocity_body_m_s: ArrayLike3 | None = None
    covariance_body: CovarianceLike3 | None = None
    position_nav_m: ArrayLike3 | None = None
    position_covariance_nav: CovarianceLike3 | None = None

    def __post_init__(self) -> None:
        if self.velocity_body_m_s is None and self.position_nav_m is None:
            raise ValueError("CorrectedDvlMeasurement needs velocity_body_m_s, position_nav_m, or both")


@dataclass
class NavigationState:
    """Nominal navigation state."""

    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gyro_bias_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias_m_s2: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "NavigationState":
        return NavigationState(
            position_m=self.position_m.copy(),
            velocity_m_s=self.velocity_m_s.copy(),
            attitude_quat_wxyz=self.attitude_quat_wxyz.copy(),
            gyro_bias_rad_s=self.gyro_bias_rad_s.copy(),
            accel_bias_m_s2=self.accel_bias_m_s2.copy(),
        )


@dataclass(frozen=True)
class NavigationOutput:
    """Filter output after propagation and optional corrected-DVL update."""

    timestamp_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    attitude_quat_wxyz: np.ndarray
    gyro_bias_rad_s: np.ndarray
    accel_bias_m_s2: np.ndarray
    covariance: np.ndarray
    dvl_update_applied: bool


@dataclass(frozen=True)
class KalmanConfig:
    """Noise and initialization settings for the DVL/IMU Kalman layer."""

    gravity_nav_m_s2: ArrayLike3 = (0.0, 0.0, -9.80665)
    gyro_noise_std_rad_s: float = 0.01
    accel_noise_std_m_s2: float = 0.1
    gyro_bias_walk_std_rad_s2: float = 1.0e-5
    accel_bias_walk_std_m_s3: float = 1.0e-4
    default_dvl_velocity_std_m_s: float = 0.05
    default_dvl_position_std_m: float = 0.25
    initial_position_std_m: float = 1.0
    initial_velocity_std_m_s: float = 1.0
    initial_attitude_std_rad: float = 0.1
    initial_gyro_bias_std_rad_s: float = 0.01
    initial_accel_bias_std_m_s2: float = 0.1
    mahalanobis_gate: float | None = 16.27


class DvlImuKalmanLayer:
    """Fusion layer for IMU propagation and corrected-DVL updates."""

    def __init__(
        self,
        initial_state: NavigationState | None = None,
        initial_covariance: np.ndarray | None = None,
        config: KalmanConfig | None = None,
    ) -> None:
        self.config = config or KalmanConfig()
        self.state = (initial_state or NavigationState()).copy()
        self.state.attitude_quat_wxyz = _normalize_quat(self.state.attitude_quat_wxyz)
        self.covariance = (
            _initial_covariance(self.config) if initial_covariance is None else _as_covariance(initial_covariance, 15)
        )
        self._last_timestamp_s: float | None = None

    def process(
        self,
        imu_sample: ImuSample,
        corrected_dvl: CorrectedDvlMeasurement | None = None,
    ) -> NavigationOutput:
        """Propagate with IMU data and update with a corrected DVL measurement."""

        self.propagate_imu(imu_sample)
        dvl_update_applied = False
        if corrected_dvl is not None:
            dvl_update_applied = self.update_corrected_dvl(corrected_dvl)
        return self.output(imu_sample.timestamp_s, dvl_update_applied=dvl_update_applied)

    @property
    def last_timestamp_s(self) -> float | None:
        """Timestamp of the last IMU propagation sample processed by the filter."""

        return self._last_timestamp_s

    def propagate_imu(self, imu_sample: ImuSample) -> None:
        """Advance the nominal state and covariance using one IMU sample."""

        timestamp_s = float(imu_sample.timestamp_s)
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            return

        dt = timestamp_s - self._last_timestamp_s
        if dt < 0.0:
            raise ValueError("IMU timestamps must be monotonic")
        self._last_timestamp_s = timestamp_s
        if dt == 0.0:
            return

        angular_velocity = _as_vec3(imu_sample.angular_velocity_rad_s, "angular_velocity_rad_s")
        linear_acceleration = _as_vec3(imu_sample.linear_acceleration_m_s2, "linear_acceleration_m_s2")
        omega_body = angular_velocity - self.state.gyro_bias_rad_s
        specific_force_body = linear_acceleration - self.state.accel_bias_m_s2

        rotation_body_to_nav = _quat_to_rotation_matrix(self.state.attitude_quat_wxyz)
        acceleration_nav = rotation_body_to_nav @ specific_force_body + _as_vec3(
            self.config.gravity_nav_m_s2, "gravity_nav_m_s2"
        )

        self.state.position_m = self.state.position_m + self.state.velocity_m_s * dt + 0.5 * acceleration_nav * dt * dt
        self.state.velocity_m_s = self.state.velocity_m_s + acceleration_nav * dt
        self.state.attitude_quat_wxyz = _normalize_quat(
            _quat_multiply(self.state.attitude_quat_wxyz, _delta_quat(omega_body * dt))
        )

        self._propagate_covariance(dt, rotation_body_to_nav, specific_force_body)

    def update_corrected_dvl(self, measurement: CorrectedDvlMeasurement) -> bool:
        """Apply corrected DVL velocity and/or position updates.

        Returns True when at least one measurement was fused, or False when all
        provided measurements were rejected by the optional Mahalanobis gate.
        """

        update_applied = False
        if measurement.velocity_body_m_s is not None:
            update_applied = self.update_corrected_dvl_velocity(measurement) or update_applied
        if measurement.position_nav_m is not None:
            update_applied = self.update_corrected_dvl_position(measurement) or update_applied
        return update_applied

    def update_corrected_dvl_velocity(self, measurement: CorrectedDvlMeasurement) -> bool:
        """Apply a corrected DVL body-frame velocity update."""

        if measurement.velocity_body_m_s is None:
            raise ValueError("velocity_body_m_s is required for a DVL velocity update")

        velocity_body = _as_vec3(measurement.velocity_body_m_s, "velocity_body_m_s")
        measurement_covariance = _velocity_measurement_covariance(measurement, self.config)

        rotation_body_to_nav = _quat_to_rotation_matrix(self.state.attitude_quat_wxyz)
        predicted_velocity_body = rotation_body_to_nav.T @ self.state.velocity_m_s
        residual = velocity_body - predicted_velocity_body

        h_matrix = np.zeros((3, 15))
        h_matrix[:, 3:6] = rotation_body_to_nav.T
        h_matrix[:, 6:9] = rotation_body_to_nav.T @ _skew(self.state.velocity_m_s)

        return self._apply_measurement_update(residual, h_matrix, measurement_covariance)

    def update_corrected_dvl_position(self, measurement: CorrectedDvlMeasurement) -> bool:
        """Apply a corrected DVL navigation-frame position update."""

        if measurement.position_nav_m is None:
            raise ValueError("position_nav_m is required for a DVL position update")

        position_nav = _as_vec3(measurement.position_nav_m, "position_nav_m")
        measurement_covariance = _position_measurement_covariance(measurement, self.config)
        residual = position_nav - self.state.position_m

        h_matrix = np.zeros((3, 15))
        h_matrix[:, 0:3] = np.eye(3)

        return self._apply_measurement_update(residual, h_matrix, measurement_covariance)

    def _apply_measurement_update(
        self,
        residual: np.ndarray,
        h_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> bool:
        innovation_covariance = h_matrix @ self.covariance @ h_matrix.T + measurement_covariance
        if self.config.mahalanobis_gate is not None:
            distance = float(residual.T @ np.linalg.solve(innovation_covariance, residual))
            if distance > self.config.mahalanobis_gate:
                return False

        kalman_gain = np.linalg.solve(innovation_covariance, h_matrix @ self.covariance).T
        error_state = kalman_gain @ residual
        self._apply_error_state(error_state)

        identity = np.eye(15)
        residual_projector = identity - kalman_gain @ h_matrix
        self.covariance = (
            residual_projector @ self.covariance @ residual_projector.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self.covariance = _symmetrize(self.covariance)
        return True

    def output(self, timestamp_s: float, dvl_update_applied: bool = False) -> NavigationOutput:
        """Return a copy of the current navigation estimate."""

        return NavigationOutput(
            timestamp_s=float(timestamp_s),
            position_m=self.state.position_m.copy(),
            velocity_m_s=self.state.velocity_m_s.copy(),
            attitude_quat_wxyz=self.state.attitude_quat_wxyz.copy(),
            gyro_bias_rad_s=self.state.gyro_bias_rad_s.copy(),
            accel_bias_m_s2=self.state.accel_bias_m_s2.copy(),
            covariance=self.covariance.copy(),
            dvl_update_applied=dvl_update_applied,
        )

    def reset(
        self,
        state: NavigationState | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Reset the filter state and timestamp history."""

        self.state = (state or NavigationState()).copy()
        self.state.attitude_quat_wxyz = _normalize_quat(self.state.attitude_quat_wxyz)
        self.covariance = _initial_covariance(self.config) if covariance is None else _as_covariance(covariance, 15)
        self._last_timestamp_s = None

    def _propagate_covariance(
        self,
        dt: float,
        rotation_body_to_nav: np.ndarray,
        specific_force_body: np.ndarray,
    ) -> None:
        transition = np.eye(15)
        dynamics = np.zeros((15, 15))

        dynamics[0:3, 3:6] = np.eye(3)
        dynamics[3:6, 6:9] = -_skew(rotation_body_to_nav @ specific_force_body)
        dynamics[3:6, 12:15] = -rotation_body_to_nav
        dynamics[6:9, 9:12] = -rotation_body_to_nav

        transition += dynamics * dt

        process_noise = np.zeros((15, 15))
        process_noise[3:6, 3:6] = (
            self.config.accel_noise_std_m_s2 * self.config.accel_noise_std_m_s2 * dt * np.eye(3)
        )
        process_noise[6:9, 6:9] = (
            self.config.gyro_noise_std_rad_s * self.config.gyro_noise_std_rad_s * dt * np.eye(3)
        )
        process_noise[9:12, 9:12] = (
            self.config.gyro_bias_walk_std_rad_s2
            * self.config.gyro_bias_walk_std_rad_s2
            * dt
            * np.eye(3)
        )
        process_noise[12:15, 12:15] = (
            self.config.accel_bias_walk_std_m_s3
            * self.config.accel_bias_walk_std_m_s3
            * dt
            * np.eye(3)
        )

        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.covariance = _symmetrize(self.covariance)

    def _apply_error_state(self, error_state: np.ndarray) -> None:
        self.state.position_m = self.state.position_m + error_state[0:3]
        self.state.velocity_m_s = self.state.velocity_m_s + error_state[3:6]
        self.state.attitude_quat_wxyz = _normalize_quat(
            _quat_multiply(_delta_quat(error_state[6:9]), self.state.attitude_quat_wxyz)
        )
        self.state.gyro_bias_rad_s = self.state.gyro_bias_rad_s + error_state[9:12]
        self.state.accel_bias_m_s2 = self.state.accel_bias_m_s2 + error_state[12:15]


def _initial_covariance(config: KalmanConfig) -> np.ndarray:
    variances = np.array(
        [
            *(config.initial_position_std_m * config.initial_position_std_m for _ in range(3)),
            *(config.initial_velocity_std_m_s * config.initial_velocity_std_m_s for _ in range(3)),
            *(config.initial_attitude_std_rad * config.initial_attitude_std_rad for _ in range(3)),
            *(config.initial_gyro_bias_std_rad_s * config.initial_gyro_bias_std_rad_s for _ in range(3)),
            *(config.initial_accel_bias_std_m_s2 * config.initial_accel_bias_std_m_s2 for _ in range(3)),
        ]
    )
    return np.diag(variances)


def _velocity_measurement_covariance(measurement: CorrectedDvlMeasurement, config: KalmanConfig) -> np.ndarray:
    variance = config.default_dvl_velocity_std_m_s * config.default_dvl_velocity_std_m_s
    return _measurement_covariance(measurement.covariance_body, variance)


def _position_measurement_covariance(measurement: CorrectedDvlMeasurement, config: KalmanConfig) -> np.ndarray:
    variance = config.default_dvl_position_std_m * config.default_dvl_position_std_m
    return _measurement_covariance(measurement.position_covariance_nav, variance)


def _measurement_covariance(value: CovarianceLike3 | None, default_variance: float) -> np.ndarray:
    if value is None:
        return np.eye(3) * default_variance

    covariance = np.asarray(value, dtype=float)
    if covariance.ndim == 0:
        return np.eye(3) * float(covariance)
    return _as_covariance(covariance, 3)


def _as_vec3(value: ArrayLike3, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3 element vector")
    return array


def _as_covariance(value: np.ndarray, size: int) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError(f"covariance must have shape ({size}, {size})")
    return _symmetrize(covariance.copy())


def _normalize_quat(quaternion: ArrayLike3) -> np.ndarray:
    array = np.asarray(quaternion, dtype=float)
    if array.shape != (4,):
        raise ValueError("attitude_quat_wxyz must be a 4 element quaternion")
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("attitude quaternion cannot have zero norm")
    return array / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _delta_quat(rotation_vector_rad: ArrayLike3) -> np.ndarray:
    rotation_vector = _as_vec3(rotation_vector_rad, "rotation_vector_rad")
    angle = np.linalg.norm(rotation_vector)
    if angle < 1.0e-12:
        return _normalize_quat(np.array([1.0, *(0.5 * rotation_vector)]))
    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    return np.array([np.cos(half_angle), *(np.sin(half_angle) * axis)])


def _quat_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
