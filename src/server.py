"""
Maritime Mission Planner - Demo 2 Backend Server

FastAPI server with REST endpoints for IMU, GPS, and pressure data,
WebSocket broadcast, data buffering, built-in sensor simulator,
track history, state endpoint, and static file serving.
"""

import asyncio
import io
import json
import math
import sys
from pathlib import Path

# When run directly, need project root on path for 'src' package access
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.bathymetry import get_depth, grid_info
except ImportError:
    from bathymetry import get_depth, grid_info  # fallback for pytest from src/

# Teammate estimation tools
from dvl_correction import (
    CorrectedDvlMeasurement,
    DvlDeadReckoningTrack,
    DvlCorrectionLayer,
    DvlImuKalmanLayer,
    ImuSample,
    KalmanConfig,
    MagnetometerCalibration,
    MagnetometerCorrectionLayer,
    NavigationFusionPipeline,
    NavigationOutput,
    RawDvlMeasurement,
    RawMagnetometerMeasurement,
    SynchronizerConfig,
)

from src.replay import SimrisReplay

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUFFER_SIZE = 300
SIMULATOR_RATE = 0.05  # seconds per tick (20 Hz)
PLANNER_TICK_DT = 0.05
PLANNER_SPEED_MS = 2.0
PLANNER_COV_GROWTH_DIAG = 0.5
PLANNER_COV_GROWTH_OFFDIAG = 0.05
PLANNER_LANDMARK_FIX = 0.6
PLANNER_INIT_COV = [[0.25, 0.0], [0.0, 0.25]]

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
SIMULATOR_DISABLED = os.environ.get(
    "DISABLE_SIMULATOR", os.environ.get("SIMULATOR_DISABLED", "")
).lower() in {"1", "true", "yes", "on"}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PLANNER_DIST_DIR = _PROJECT_ROOT / "frontend" / "web" / "dist"
PLANNER_INDEX = PLANNER_DIST_DIR / "index.html"
RUNTIME_DIR = Path(os.environ.get("EDTH_RUNTIME_DIR", _PROJECT_ROOT / "runtime"))
CONFIG_PATH = Path(os.environ.get("EDTH_CONFIG_PATH", RUNTIME_DIR / "command_center_config.json"))
RECORDINGS_DIR = Path(os.environ.get("EDTH_RECORDINGS_DIR", RUNTIME_DIR / "recordings"))

SENSOR_NAMES = {"accelerometer", "magnetometer", "gyroscope", "orientation", "barometer"}
APP_MODES = {"simulator", "replay", "live", "training"}

# Real data location — Simrishamn field test
BASE_LAT = 55.5601
BASE_LON = 14.3626
SURVEY_SPEED_MS = 2.0          # m/s
SURVEY_LEG_LENGTH = 200.0      # metres per long leg
SURVEY_LEG_SPACING = 20.0      # metres between passes
SURVEY_HEADINGS = (90.0, 0.0, 270.0, 180.0)  # east, north, west, south

# Support vessel (vessel2) figure-8 holding pattern
V2_CENTER_LAT = BASE_LAT + 0.004
V2_CENTER_LON = BASE_LON + 0.004
V2_RADIUS_LAT = 0.0009
V2_RADIUS_LON = 0.0016
V2_SPEED_MS = 1.5
V2_OMEGA = 0.015

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SensorValue(BaseModel):
    name: str
    values: dict[str, float]


class DataPayload(BaseModel):
    payload: list[SensorValue]


class GpsData(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    speed_kn: float | None = None
    heading_deg: float | None = None


class PressureData(BaseModel):
    depth_m: float | None = None
    pressure_bar: float | None = None


class GpsDenialIn(BaseModel):
    enabled: bool


class ModeIn(BaseModel):
    mode: str


class RecordingSaveIn(BaseModel):
    name: str | None = None


class ConfigIn(BaseModel):
    config: dict[str, Any]


class EstimatorResetIn(BaseModel):
    clear_buffers: bool = False
    clear_track: bool = False
    clear_events: bool = False


class MagCorrectionIn(BaseModel):
    enabled: bool


class DvlData(BaseModel):
    timestamp_s: float | None = None
    velocity_dvl_m_s: list[float] | None = Field(default=None, min_length=3, max_length=3)
    velocity_covariance_dvl: float | list[list[float]] | None = None
    position_nav_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    position_covariance_nav: float | list[list[float]] | None = None
    valid_beams: list[bool] | None = None
    mode: str | None = "bottom"
    status: str | None = "valid"
    altitude_m: float | None = None
    velocity_scale_factor: float = 1.0


class WaypointIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: str | None = None


class RiskSegment(BaseModel):
    from_id: int
    to_id: int
    from_name: str | None
    to_name: str | None
    distance_m: float
    risk_score: float
    factors: dict[str, float]
    heading_error_deg: float
    avg_depth_m: float | None


class PlannerPathIn(BaseModel):
    waypoints: list[list[float]] = Field(..., min_length=2)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.buffers: defaultdict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=BUFFER_SIZE)
        )
        self.websockets: set[WebSocket] = set()
        self.simulator_task: asyncio.Task[None] | None = None
        self.track_buffer: deque[dict[str, Any]] = deque(maxlen=2000)
        self.mode: str = "live" if SIMULATOR_DISABLED else "simulator"
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.event_next_id: int = 1

        # Waypoints & IMU drift tracking
        self.waypoints: list[dict[str, Any]] = []
        self.waypoint_next_id: int = 1
        self.last_gps_time: float | None = None
        self.gyro_z_samples: deque[float] = deque(maxlen=300)
        self.accel_mags: deque[float] = deque(maxlen=300)
        self.gyro_z_bias: float = 0.0
        self.accel_variance: float = 0.0
        self.drift_rate_dps: float = 0.0

        # Second vessel (support ship)
        self.vessel2_lat: float = V2_CENTER_LAT
        self.vessel2_lon: float = V2_CENTER_LON
        self.vessel2_heading: float = 0.0
        self.vessel2_speed: float = 1.5
        self.vessel2_track: deque[dict[str, Any]] = deque(maxlen=200)

        # Mission recording
        self.recording: bool = False
        self.recorded_frames: list[dict[str, Any]] = []
        self.recording_start_time: float = 0.0
        self.last_record_time: float = 0.0
        self.last_recording_id: str | None = None

        # GPS denial & dead reckoning
        self.gps_denied: bool = False
        self.gps_deny_start_time: float | None = None
        self.dr_lat: float | None = None
        self.dr_lon: float | None = None
        self.dr_heading: float = 0.0

        # Teammate estimation tools
        self.kf: DvlImuKalmanLayer = DvlImuKalmanLayer(config=KalmanConfig())
        self.dvl_correction: DvlCorrectionLayer = DvlCorrectionLayer()
        self.fusion_pipeline: NavigationFusionPipeline = NavigationFusionPipeline(
            kalman=self.kf,
            dvl_correction=self.dvl_correction,
            dvl_track=DvlDeadReckoningTrack(),
            config=SynchronizerConfig(max_delay_s=0.1),
        )
        self.mag_correction: MagnetometerCorrectionLayer = MagnetometerCorrectionLayer()
        self.mag_correction_enabled: bool = True
        self.kf_output: NavigationOutput | None = None
        self.kf_covariance: list[list[float]] = [[1.0, 0.0], [0.0, 1.0]]
        self.nav_origin_lat: float | None = None
        self.nav_origin_lon: float | None = None
        self.latest_accel_m_s2: list[float] | None = None
        self.latest_gyro_rad_s: list[float] | None = None
        self.dvl_raw_count: int = 0
        self.dvl_rejected_count: int = 0
        self.dvl_updates_applied: int = 0
        self.dvl_quality_rejections: int = 0
        self.dvl_gate_rejections: int = 0
        self.dvl_sample_times: deque[float] = deque(maxlen=300)
        self.latest_dvl: dict[str, Any] | None = None

        # Real data replay
        self.replay: SimrisReplay = SimrisReplay()
        self.replay_task: asyncio.Task[None] | None = None
        self.replay_speed: float = 1.0
        self.dr_last_gps_speed_kn: float = 0.0

        # Magnetometer calibration
        self.mag_hard_iron: list[float] = [0.0, 0.0, 0.0]
        self.mag_soft_iron: list[list[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.expected_mag_field_uT: float = 50.0
        self.mag_heading: float = 0.0
        self.dr_heading_source: str = "magnetometer"

        # Operator settings
        self.safety_config: dict[str, float] = {
            "min_depth_m": 2.0,
            "max_route_length_m": 2500.0,
            "max_gps_age_sec": 30.0,
            "max_uncertainty_m": 50.0,
            "max_support_distance_m": 1500.0,
        }

        # Embedded metric planner route. Planner x is east, y is north, metres.
        self.planner_path_m: list[tuple[float, float]] = []
        self.planner_segment_starts_m: list[float] = []
        self.planner_total_length_m: float = 0.0
        self.planner_progress_m: float = 0.0
        self.planner_enabled: bool = False
        self.planner_paused: bool = False
        self.planner_done: bool = False

state = AppState()


# ---------------------------------------------------------------------------
# Configuration, mode, and event helpers
# ---------------------------------------------------------------------------

def _new_fusion_pipeline() -> NavigationFusionPipeline:
    return NavigationFusionPipeline(
        kalman=state.kf,
        dvl_correction=state.dvl_correction,
        dvl_track=DvlDeadReckoningTrack(),
        config=SynchronizerConfig(max_delay_s=0.1),
    )


def _log_event(
    kind: str,
    message: str,
    severity: str = "info",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "id": state.event_next_id,
        "timestamp": time.time(),
        "kind": kind,
        "severity": severity,
        "message": message,
        "data": data or {},
    }
    state.event_next_id += 1
    state.events.append(event)
    return event


async def _emit_event(
    kind: str,
    message: str,
    severity: str = "info",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = _log_event(kind, message, severity, data)
    await _broadcast({"type": "event", **event})
    return event


def _config_snapshot() -> dict[str, Any]:
    return {
        "mode": state.mode,
        "magnetometer": {
            "enabled": state.mag_correction_enabled,
            "hard_iron_offset_uT": state.mag_hard_iron,
            "soft_iron_matrix": state.mag_soft_iron,
            "expected_field_magnitude_uT": state.expected_mag_field_uT,
        },
        "safety": state.safety_config,
        "replay": {"speed": state.replay_speed},
    }


def _apply_config(config: dict[str, Any]) -> None:
    mode = config.get("mode")
    if isinstance(mode, str) and mode in APP_MODES:
        state.mode = mode

    mag = config.get("magnetometer")
    if isinstance(mag, dict):
        state.mag_correction_enabled = bool(mag.get("enabled", state.mag_correction_enabled))
        hard_iron = mag.get("hard_iron_offset_uT", state.mag_hard_iron)
        soft_iron = mag.get("soft_iron_matrix", state.mag_soft_iron)
        expected = float(mag.get("expected_field_magnitude_uT", state.expected_mag_field_uT))
        hard_array = np.asarray(hard_iron, dtype=float)
        soft_array = np.asarray(soft_iron, dtype=float)
        if hard_array.shape != (3,):
            raise ValueError("magnetometer.hard_iron_offset_uT must have 3 values")
        if soft_array.shape != (3, 3):
            raise ValueError("magnetometer.soft_iron_matrix must be 3x3")
        if not math.isfinite(expected) or expected <= 0.0:
            raise ValueError("magnetometer.expected_field_magnitude_uT must be positive")
        state.mag_hard_iron = hard_array.tolist()
        state.mag_soft_iron = soft_array.tolist()
        state.expected_mag_field_uT = expected
        state.mag_correction = MagnetometerCorrectionLayer(
            calibration=MagnetometerCalibration(
                hard_iron_offset_uT=state.mag_hard_iron,
                soft_iron_matrix=state.mag_soft_iron,
                expected_field_magnitude_uT=state.expected_mag_field_uT,
            )
        )

    safety = config.get("safety")
    if isinstance(safety, dict):
        for key in state.safety_config:
            if key in safety:
                value = float(safety[key])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"safety.{key} must be finite and non-negative")
                state.safety_config[key] = value

    replay = config.get("replay")
    if isinstance(replay, dict) and "speed" in replay:
        speed = float(replay["speed"])
        if math.isfinite(speed) and speed > 0.0:
            state.replay_speed = speed
            state.replay.speed = speed


def _load_persisted_config() -> None:
    if not CONFIG_PATH.is_file():
        return
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(config, dict):
            _apply_config(config)
            _log_event("config", f"Loaded configuration from {CONFIG_PATH}")
    except Exception as exc:
        _log_event("config", f"Failed to load configuration: {exc}", severity="warning")


def _save_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(_config_snapshot(), indent=2, sort_keys=True), encoding="utf-8")


def _reset_estimator(clear_buffers: bool = False, clear_track: bool = False, clear_events: bool = False) -> None:
    state.kf = DvlImuKalmanLayer(config=KalmanConfig())
    state.dvl_correction = DvlCorrectionLayer()
    state.fusion_pipeline = _new_fusion_pipeline()
    state.kf_output = None
    state.kf_covariance = [[1.0, 0.0], [0.0, 1.0]]
    state.nav_origin_lat = None
    state.nav_origin_lon = None
    state.latest_accel_m_s2 = None
    state.latest_gyro_rad_s = None
    state.dvl_raw_count = 0
    state.dvl_rejected_count = 0
    state.dvl_updates_applied = 0
    state.dvl_quality_rejections = 0
    state.dvl_gate_rejections = 0
    state.dvl_sample_times.clear()
    state.latest_dvl = None
    state.dr_lat = None
    state.dr_lon = None
    if clear_buffers:
        state.buffers.clear()
    if clear_track:
        state.track_buffer.clear()
        state.vessel2_track.clear()
    if clear_events:
        state.events.clear()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now() -> float:
    """Current Unix timestamp (seconds)."""
    return time.time()


def _smooth_noise(t: float, seed: float) -> float:
    """Deterministic smooth noise via summed sine waves.  Range ~[-1, 1]."""
    return (
        math.sin(t * 137.0 + seed) * 0.5
        + math.sin(t * 73.0 + seed * 2.0) * 0.3
        + math.sin(t * 41.0 + seed * 3.0) * 0.2
    )


def _buffer_sensor(name: str, values: dict[str, float]) -> None:
    """Append a timestamped entry to the named sensor buffer."""
    state.buffers[name].append({"timestamp": _now(), **values})


def _last_ts(name: str) -> float:
    """Timestamp of the most recent entry in a sensor buffer."""
    return state.buffers[name][-1]["timestamp"]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R, p1, p2 = 6371000.0, math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def _ensure_nav_origin(lat: float, lon: float) -> None:
    if state.nav_origin_lat is None or state.nav_origin_lon is None:
        state.nav_origin_lat = lat
        state.nav_origin_lon = lon


def _latlon_to_nav_m(lat: float, lon: float) -> np.ndarray:
    _ensure_nav_origin(lat, lon)
    assert state.nav_origin_lat is not None and state.nav_origin_lon is not None
    north_m = (lat - state.nav_origin_lat) * 111320.0
    east_m = (lon - state.nav_origin_lon) * 111320.0 * math.cos(math.radians(state.nav_origin_lat))
    return np.array([east_m, north_m, 0.0], dtype=float)


def _nav_m_to_latlon(position_m: Any) -> tuple[float, float] | None:
    if state.nav_origin_lat is None or state.nav_origin_lon is None:
        return None
    pos = np.asarray(position_m, dtype=float)
    if pos.shape[0] < 2 or not np.all(np.isfinite(pos[:2])):
        return None
    east_m = float(pos[0])
    north_m = float(pos[1])
    lat = state.nav_origin_lat + north_m / 111320.0
    lon = state.nav_origin_lon + east_m / (111320.0 * math.cos(math.radians(state.nav_origin_lat)))
    return lat, lon


def _planner_m_to_latlon(x_m: float, y_m: float) -> tuple[float, float]:
    """Convert planner metres into local geographic coordinates near Simrishamn."""
    lat = BASE_LAT + y_m / 111320.0
    lon = BASE_LON + x_m / (111320.0 * math.cos(math.radians(BASE_LAT)))
    return lat, lon


def _set_planner_path(points: list[tuple[float, float]]) -> None:
    state.planner_path_m = points
    state.planner_segment_starts_m = []
    state.planner_total_length_m = 0.0
    state.planner_progress_m = 0.0
    state.planner_enabled = True
    state.planner_paused = False
    state.planner_done = False

    acc = 0.0
    for i, point in enumerate(points):
        state.planner_segment_starts_m.append(acc)
        if i + 1 < len(points):
            nxt = points[i + 1]
            acc += math.hypot(nxt[0] - point[0], nxt[1] - point[1])
    state.planner_total_length_m = acc


def _clear_planner_path() -> None:
    state.planner_path_m = []
    state.planner_segment_starts_m = []
    state.planner_total_length_m = 0.0
    state.planner_progress_m = 0.0
    state.planner_enabled = False
    state.planner_paused = False
    state.planner_done = False


def _planner_xy_at_progress(progress_m: float) -> tuple[float, float, float]:
    """Return x, y, heading_deg for the active metric planner route."""
    path = state.planner_path_m
    if not path:
        return 0.0, 0.0, SURVEY_HEADINGS[0]
    if len(path) == 1:
        return path[0][0], path[0][1], SURVEY_HEADINGS[0]

    progress_m = max(0.0, min(progress_m, state.planner_total_length_m))
    segment = max(0, len(path) - 2)
    for i in range(len(path) - 1):
        start = state.planner_segment_starts_m[i]
        end = state.planner_segment_starts_m[i + 1]
        if progress_m <= end or i == len(path) - 2:
            segment = i
            break

    p0, p1 = path[segment], path[segment + 1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len = max(math.hypot(dx, dy), 1e-9)
    alpha = (progress_m - state.planner_segment_starts_m[segment]) / seg_len
    alpha = max(0.0, min(alpha, 1.0))
    heading = math.degrees(math.atan2(dx, dy)) % 360.0
    return p0[0] + alpha * dx, p0[1] + alpha * dy, heading


def _advance_planner_route(step_m: float) -> tuple[float, float, float]:
    """Advance the active planner route by a metric step and return position."""
    if state.planner_enabled and not state.planner_paused and not state.planner_done:
        state.planner_progress_m = min(
            state.planner_total_length_m,
            state.planner_progress_m + max(0.0, step_m),
        )
        if state.planner_progress_m >= state.planner_total_length_m:
            state.planner_done = True
    return _planner_xy_at_progress(state.planner_progress_m)


def _planner_geo_route() -> list[dict[str, float]]:
    return [
        {"x_m": x, "y_m": y, "lat": _planner_m_to_latlon(x, y)[0], "lon": _planner_m_to_latlon(x, y)[1]}
        for x, y in state.planner_path_m
    ]


def _metric_path(points: list[tuple[float, float]]) -> tuple[list[float], float]:
    starts: list[float] = []
    acc = 0.0
    for i, point in enumerate(points):
        starts.append(acc)
        if i + 1 < len(points):
            nxt = points[i + 1]
            acc += math.hypot(nxt[0] - point[0], nxt[1] - point[1])
    return starts, acc


def _xy_on_metric_path(
    points: list[tuple[float, float]],
    starts: list[float],
    total_length_m: float,
    progress_m: float,
) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1 or total_length_m <= 0.0:
        return points[0]

    progress_m = max(0.0, min(progress_m, total_length_m))
    segment = len(points) - 2
    for i in range(len(points) - 1):
        if progress_m <= starts[i + 1] or i == len(points) - 2:
            segment = i
            break

    p0, p1 = points[segment], points[segment + 1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len = max(math.hypot(dx, dy), 1e-9)
    alpha = max(0.0, min(1.0, (progress_m - starts[segment]) / seg_len))
    return p0[0] + alpha * dx, p0[1] + alpha * dy


def _recalc_drift() -> None:
    """Recalculate drift rate from current gyro bias and accelerometer noise."""
    nf = 1.0 + (math.sqrt(state.accel_variance) / 9.81 if state.accel_variance > 0 else 0)
    state.drift_rate_dps = abs(state.gyro_z_bias) * (180.0 / math.pi) * nf


def _remember_kf_output(output: NavigationOutput) -> None:
    state.kf_output = output
    state.kf_covariance = output.covariance[:2, :2].tolist()


def _maybe_process_imu(timestamp_s: float) -> None:
    if state.latest_accel_m_s2 is None or state.latest_gyro_rad_s is None:
        return
    if state.kf.last_timestamp_s is not None and timestamp_s <= state.kf.last_timestamp_s:
        return
    sample = ImuSample(
        timestamp_s=timestamp_s,
        angular_velocity_rad_s=state.latest_gyro_rad_s,
        linear_acceleration_m_s2=state.latest_accel_m_s2,
    )
    try:
        outputs = state.fusion_pipeline.add_imu_sample(sample)
        outputs.extend(state.fusion_pipeline.flush())
    except ValueError:
        output = state.kf.process(sample)
        _remember_kf_output(output)
        return
    for output in outputs:
        _remember_kf_output(output)
    if not outputs:
        _remember_kf_output(
            state.kf.output(
                timestamp_s,
                dvl_update_applied=False,
                magnetometer_update_applied=False,
            )
        )


def _fallback_imu_sample(timestamp_s: float) -> ImuSample:
    return ImuSample(
        timestamp_s=timestamp_s,
        angular_velocity_rad_s=state.latest_gyro_rad_s or [0.0, 0.0, 0.0],
        linear_acceleration_m_s2=state.latest_accel_m_s2 or [0.0, 0.0, 9.80665],
    )


def _format_nav_output(output: NavigationOutput | None) -> dict[str, Any] | None:
    if output is None:
        return None
    latlon = _nav_m_to_latlon(output.position_m)
    return {
        "timestamp_s": output.timestamp_s,
        "position_m": [float(v) for v in np.asarray(output.position_m)],
        "velocity_m_s": [float(v) for v in np.asarray(output.velocity_m_s)],
        "lat": None if latlon is None else latlon[0],
        "lon": None if latlon is None else latlon[1],
        "dvl_update_applied": output.dvl_update_applied,
        "magnetometer_update_applied": output.magnetometer_update_applied,
    }


def _direct_dvl_update(corrected: CorrectedDvlMeasurement) -> NavigationOutput:
    update_applied = state.kf.update_corrected_dvl(corrected)
    if update_applied:
        state.dvl_updates_applied += 1
    else:
        state.dvl_gate_rejections += 1
    return state.kf.output(corrected.timestamp_s, dvl_update_applied=update_applied)


def _process_raw_dvl(raw: RawDvlMeasurement) -> tuple[str, NavigationOutput | None, CorrectedDvlMeasurement | None]:
    try:
        state.fusion_pipeline.add_imu_sample(_fallback_imu_sample(raw.timestamp_s))
        outputs = state.fusion_pipeline.add_raw_dvl_measurement(raw)
        outputs.extend(state.fusion_pipeline.flush())
    except ValueError:
        corrected = state.dvl_correction.correct(
            raw,
            angular_velocity_body_rad_s=state.latest_gyro_rad_s or [0.0, 0.0, 0.0],
        )
        if corrected is None:
            state.dvl_quality_rejections += 1
            return "quality_gate", None, None
        return "accepted", _direct_dvl_update(corrected), corrected

    diag = state.fusion_pipeline.diagnostics
    state.dvl_quality_rejections = diag.dvl_quality_rejections
    state.dvl_gate_rejections = diag.dvl_gate_rejections
    state.dvl_updates_applied = diag.dvl_updates_applied
    if not outputs:
        state.dvl_quality_rejections += 1
        return "quality_gate", None, None

    output = outputs[-1]
    corrected = None
    return "accepted", output, corrected


def _apply_magnetometer(values: dict[str, float], timestamp_s: float) -> None:
    raw = RawMagnetometerMeasurement(
        timestamp_s=timestamp_s,
        magnetic_field_body_uT=[
            values.get("x", 0.0),
            values.get("y", 0.0),
            values.get("z", 0.0),
        ],
    )
    corrected = state.mag_correction.correct(
        raw,
        attitude_quat_wxyz=state.kf.state.attitude_quat_wxyz,
    )
    if corrected is None:
        return
    state.mag_heading = math.degrees(corrected.yaw_magnetic_rad) % 360.0
    if state.mag_correction_enabled:
        state.dr_heading = state.mag_heading
        state.dr_heading_source = "magnetometer"
        applied = state.kf.update_magnetometer_yaw(corrected)
    else:
        applied = False
    _remember_kf_output(state.kf.output(timestamp_s, magnetometer_update_applied=applied))


def _latest_gps() -> dict[str, Any] | None:
    gps_buf = state.buffers.get("gps")
    return gps_buf[-1] if gps_buf else None


def _gps_age_sec(now: float | None = None) -> float | None:
    last_gps = _latest_gps()
    if not last_gps or "timestamp" not in last_gps:
        return None
    return max(0.0, (now or _now()) - last_gps["timestamp"])


def _gps_denial_summary() -> dict[str, Any]:
    age = _gps_age_sec()
    last_gps = _latest_gps()
    return {
        "enabled": state.gps_denied,
        "gps_denied": state.gps_denied,
        "last_gps": None if last_gps is None else {
            "lat": last_gps.get("lat"),
            "lon": last_gps.get("lon"),
            "age_sec": round(age or 0.0, 1),
        },
        "simulator_gps_stopped": state.gps_denied,
    }


def _latest_depth_m() -> float | None:
    pressure_buf = state.buffers.get("pressure")
    if not pressure_buf:
        return None
    return pressure_buf[-1].get("depth_m")


def _record_frame(source: str, force: bool = False) -> None:
    if not state.recording:
        return
    now = _now()
    if not force and now - state.last_record_time < 1.0:
        return

    last_gps = _latest_gps()
    if state.gps_denied:
        lat = state.dr_lat
        lon = state.dr_lon
    else:
        lat = last_gps.get("lat") if last_gps else state.dr_lat
        lon = last_gps.get("lon") if last_gps else state.dr_lon
    if lat is None or lon is None:
        lat, lon = BASE_LAT, BASE_LON

    speed_kn = (
        state.dr_last_gps_speed_kn
        if state.gps_denied
        else (last_gps.get("speed_kn") if last_gps else None)
    )
    heading = (
        state.dr_heading
        if state.gps_denied
        else (last_gps.get("heading_deg") if last_gps else state.dr_heading)
    )

    state.last_record_time = now
    state.recorded_frames.append({
        "timestamp": now,
        "source": source,
        "gps_denied": state.gps_denied,
        "vessel1": {
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "heading": round(float(heading), 1) if heading is not None else None,
            "speed": round(float(speed_kn), 2) if speed_kn is not None else None,
            "speed_unit": "kn",
            "depth": _latest_depth_m(),
            "position_source": "dead_reckon" if state.gps_denied else "gps",
        },
        "vessel2": {
            "lat": state.vessel2_lat,
            "lon": state.vessel2_lon,
            "heading": state.vessel2_heading,
            "speed": round(state.vessel2_speed * 1.94384, 2),
            "speed_unit": "kn",
            "depth": None,
        },
    })


def _enable_gps_denial() -> None:
    if not state.gps_denied:
        state.gps_deny_start_time = _now()
    state.gps_denied = True
    last = _latest_gps()
    if last:
        state.dr_lat = last.get("lat")
        state.dr_lon = last.get("lon")
        state.dr_last_gps_speed_kn = last.get("speed_kn") or (SURVEY_SPEED_MS * 1.94384)
        state.dr_heading = last.get("heading_deg") if last.get("heading_deg") is not None else state.mag_heading
        state.dr_heading_source = "magnetometer" if state.mag_correction_enabled else "last_gps_heading"
        _ensure_nav_origin(state.dr_lat, state.dr_lon)
    else:
        state.dr_lat = state.dr_lat if state.dr_lat is not None else BASE_LAT
        state.dr_lon = state.dr_lon if state.dr_lon is not None else BASE_LON
        state.dr_last_gps_speed_kn = SURVEY_SPEED_MS * 1.94384
        state.dr_heading = state.mag_heading
        state.dr_heading_source = "magnetometer" if state.mag_correction_enabled else "fallback_heading"
        _ensure_nav_origin(state.dr_lat, state.dr_lon)


def _disable_gps_denial() -> None:
    state.gps_denied = False
    state.gps_deny_start_time = None


async def _stop_replay_task() -> None:
    state.replay.stop()
    if state.replay_task is not None:
        state.replay_task.cancel()
        try:
            await state.replay_task
        except asyncio.CancelledError:
            pass
        state.replay_task = None


async def _set_app_mode(mode: str) -> dict[str, Any]:
    mode = mode.lower()
    if mode not in APP_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(APP_MODES)}")
    previous = state.mode
    if mode != "replay":
        await _stop_replay_task()
    if mode == "training":
        _enable_gps_denial()
    elif mode in {"simulator", "live", "replay"}:
        _disable_gps_denial()
    state.mode = mode
    if previous != mode:
        await _emit_event("mode", f"Mode changed from {previous} to {mode}", data={"mode": mode})
    return {"mode": state.mode, "previous_mode": previous, "gps_denied": state.gps_denied}


def _dead_reckon_payload() -> dict[str, Any]:
    now = _now()
    last_gps = _latest_gps()
    secs = (now - state.gps_deny_start_time) if state.gps_denied and state.gps_deny_start_time else 0.0
    speed_kn = state.dr_last_gps_speed_kn or (last_gps.get("speed_kn", 0.0) if last_gps else 0.0)
    heading = state.dr_heading
    drift_uncertainty_m = max(0.0, speed_kn * 0.514444 * secs * state.drift_rate_dps * 0.1)
    filter_uncertainty_m = 0.0
    if state.kf_output is not None:
        filter_uncertainty_m = float(_navigation_uncertainty_payload(heading).get("uncertainty_m", 0.0))
    base_uncertainty_m = 3.0 if state.gps_denied else 0.0
    uncertainty_m = math.hypot(max(base_uncertainty_m, filter_uncertainty_m), drift_uncertainty_m)
    hr = math.radians(heading)
    c, s = math.cos(hr), math.sin(hr)
    lm, ln = (uncertainty_m * 1.5) ** 2, (uncertainty_m * 0.3) ** 2
    cov_ee = lm * c * c + ln * s * s
    cov_nn = lm * s * s + ln * c * c
    cov_en = (lm - ln) * s * c
    lat = state.dr_lat if state.gps_denied else (last_gps.get("lat") if last_gps else state.dr_lat)
    lon = state.dr_lon if state.gps_denied else (last_gps.get("lon") if last_gps else state.dr_lon)
    fix_type = "dead_reckon" if state.gps_denied else "gps"
    if state.gps_denied and state.dvl_updates_applied > 0 and state.kf_output is not None:
        kf_latlon = _nav_m_to_latlon(state.kf_output.position_m)
        if kf_latlon is not None:
            lat, lon = kf_latlon
            state.dr_lat = lat
            state.dr_lon = lon
            fix_type = "dead_reckon_dvl_imu"
    if lat is None or lon is None:
        lat = BASE_LAT
        lon = BASE_LON
        if state.gps_denied:
            state.dr_lat = lat
            state.dr_lon = lon
    ellipse = {
        "semi_major_m": round(uncertainty_m * 1.5, 1),
        "semi_minor_m": round(uncertainty_m * 0.3, 1),
        "orientation_deg": round(heading, 1),
    }
    covariance = [[round(cov_nn, 4), round(cov_en, 4)], [round(cov_en, 4), round(cov_ee, 4)]]
    return {
        "type": "dead_reckon",
        "lat": lat,
        "lon": lon,
        "dr_lat": lat,
        "dr_lon": lon,
        "gps_lat": last_gps.get("lat") if last_gps else None,
        "gps_lon": last_gps.get("lon") if last_gps else None,
        "gps_age_sec": round(secs, 1),
        "seconds_since_gps": round(secs, 1),
        "fix_type": fix_type,
        "heading_source": state.dr_heading_source,
        "mag_correction_enabled": state.mag_correction_enabled,
        "uncertainty_m": round(uncertainty_m, 1),
        "ellipse": ellipse,
        "uncertainty_ellipse": {
            "semi_major": ellipse["semi_major_m"],
            "semi_minor": ellipse["semi_minor_m"],
            "angle_deg": ellipse["orientation_deg"],
        },
        "position_source": fix_type,
        "estimated_position": {"lat": lat, "lon": lon},
        "mean": [lat, lon],
        "cov": covariance,
        "covariance": covariance,
        "uncertainty_reasons": _uncertainty_reasons(),
        "drift_stats": {
            "gyro_bias_dps": round(state.gyro_z_bias * (180.0 / math.pi), 4),
            "arw_deg_per_sqrt_s": 0.0,
            "total_drift_deg": round(state.drift_rate_dps * secs, 4),
            "position_uncertainty_m": round(uncertainty_m, 1),
            "filter_uncertainty_m": round(filter_uncertainty_m, 1),
            "drift_uncertainty_m": round(drift_uncertainty_m, 1),
            "base_uncertainty_m": round(base_uncertainty_m, 1),
        },
    }


def _navigation_uncertainty_payload(heading_deg: float | None = None) -> dict[str, Any]:
    cov = np.asarray(state.kf_covariance, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        cov = np.eye(2)
    cov = (cov + cov.T) * 0.5
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    confidence_scale = math.sqrt(5.991)
    semi_major = confidence_scale * math.sqrt(max(0.0, float(values[0])))
    semi_minor = confidence_scale * math.sqrt(max(0.0, float(values[1])))
    semi_major = max(6.0, semi_major)
    semi_minor = max(2.5, min(semi_major, semi_minor))
    if abs(float(values[0]) - float(values[1])) < 1.0e-6 and heading_deg is not None:
        orientation_deg = float(heading_deg)
    else:
        axis = vectors[:, 0]
        orientation_deg = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 360.0
    ellipse = {
        "semi_major_m": round(semi_major, 1),
        "semi_minor_m": round(semi_minor, 1),
        "orientation_deg": round(orientation_deg, 1),
    }
    return {
        "uncertainty_m": ellipse["semi_major_m"],
        "ellipse": ellipse,
        "uncertainty_ellipse": {
            "semi_major": ellipse["semi_major_m"],
            "semi_minor": ellipse["semi_minor_m"],
            "angle_deg": ellipse["orientation_deg"],
        },
    }


# ---------------------------------------------------------------------------
# Replay runner
# ---------------------------------------------------------------------------


async def _replay_runner(replay: SimrisReplay, speed: float = 1.0) -> None:
    """Run the Simris replay pipeline, feeding data to internal handlers."""

    async def on_gps(event: dict) -> None:
        await post_gps(GpsData(lat=event["lat"], lon=event["lon"]))

    async def on_imu(event: dict) -> None:
        await post_data(DataPayload(payload=[
            SensorValue(name="accelerometer", values=event["accelerometer"]),
            SensorValue(name="gyroscope", values=event["gyroscope"]),
            SensorValue(name="magnetometer", values=event["magnetometer"]),
        ]))

    async for _ in replay.replay(speed=speed, on_gps=on_gps, on_imu=on_imu):
        pass


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    _load_persisted_config()
    if not SIMULATOR_DISABLED:
        state.simulator_task = asyncio.create_task(_simulator_loop())
    yield
    if state.simulator_task is not None:
        state.simulator_task.cancel()
        try:
            await state.simulator_task
        except asyncio.CancelledError:
            pass
    if state.replay_task is not None:
        state.replay_task.cancel()
        try:
            await state.replay_task
        except asyncio.CancelledError:
            pass

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket broadcast helper
# ---------------------------------------------------------------------------

async def _broadcast(message: dict[str, Any]) -> None:
    """Send *message* to every connected WebSocket client."""
    stale: list[WebSocket] = []
    for ws in list(state.websockets):
        try:
            await ws.send_json(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        state.websockets.discard(ws)

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/data")
async def post_data(data: DataPayload):
    """Ingest IMU sensor payload from a phone."""
    for sensor in data.payload:
        if sensor.name not in SENSOR_NAMES:
            continue
        _buffer_sensor(sensor.name, sensor.values)
        await _broadcast({
            "type": "imu",
            "sensor": sensor.name,
            **sensor.values,
            "timestamp": _last_ts(sensor.name),
        })
        timestamp_s = _last_ts(sensor.name)
        if sensor.name == "accelerometer":
            state.latest_accel_m_s2 = [
                sensor.values.get("x", 0.0),
                sensor.values.get("y", 0.0),
                sensor.values.get("z", 0.0),
            ]
            _maybe_process_imu(timestamp_s)
        if sensor.name == "gyroscope":
            state.latest_gyro_rad_s = [
                sensor.values.get("x", 0.0),
                sensor.values.get("y", 0.0),
                sensor.values.get("z", 0.0),
            ]
            _maybe_process_imu(timestamp_s)
        if sensor.name == "gyroscope" and "z" in sensor.values:
            z = sensor.values["z"]
            if math.isfinite(z):
                state.gyro_z_samples.append(z)
                state.gyro_z_bias = sum(state.gyro_z_samples) / len(state.gyro_z_samples)
                _recalc_drift()
        if sensor.name == "accelerometer":
            mag = math.sqrt(sensor.values.get("x", 0)**2 + sensor.values.get("y", 0)**2 + sensor.values.get("z", 0)**2)
            if math.isfinite(mag):
                state.accel_mags.append(mag)
                if len(state.accel_mags) > 1:
                    m = sum(state.accel_mags) / len(state.accel_mags)
                    state.accel_variance = sum((v - m) ** 2 for v in state.accel_mags) / len(state.accel_mags)
                _recalc_drift()
        if sensor.name == "magnetometer":
            _apply_magnetometer(sensor.values, timestamp_s)
    return {"status": "ok"}

@app.post("/gps")
async def post_gps(data: GpsData):
    """Ingest a GPS position fix."""
    if state.gps_denied:
        return {"status": "ok", "gps_denied": True}

    prev_gps = _latest_gps()
    entry: dict[str, Any] = {"lat": data.lat, "lon": data.lon}
    speed_kn = data.speed_kn
    heading_deg = data.heading_deg
    if prev_gps and speed_kn is None:
        dt = max(_now() - prev_gps.get("timestamp", _now()), 1.0e-3)
        dist_m = _haversine(prev_gps["lat"], prev_gps["lon"], data.lat, data.lon)
        speed_kn = (dist_m / dt) * 1.94384
    if prev_gps and heading_deg is None:
        heading_deg = _bearing_deg(prev_gps["lat"], prev_gps["lon"], data.lat, data.lon)
    if speed_kn is not None:
        entry["speed_kn"] = speed_kn
    if heading_deg is not None:
        entry["heading_deg"] = heading_deg
    _buffer_sensor("gps", entry)
    timestamp_s = _last_ts("gps")
    dvl_update_applied = state.kf.update_corrected_dvl(
        CorrectedDvlMeasurement(
            timestamp_s=timestamp_s,
            position_nav_m=_latlon_to_nav_m(data.lat, data.lon),
            position_covariance_nav=np.diag([4.0, 4.0, 0.01]),
        )
    )
    _remember_kf_output(state.kf.output(timestamp_s, dvl_update_applied=dvl_update_applied))
    uncertainty = _navigation_uncertainty_payload(heading_deg)
    state.buffers["gps"][-1].update(uncertainty)

    if not state.gps_denied:
        await _broadcast({
            "type": "gps",
            "lat": data.lat,
            "lon": data.lon,
            "speed_kn": speed_kn,
            "heading_deg": heading_deg,
            **uncertainty,
            "timestamp": _last_ts("gps"),
        })
        state.track_buffer.append({
            "lat": data.lat,
            "lon": data.lon,
            "timestamp": _now(),
            "speed_kn": speed_kn,
            "heading_deg": heading_deg,
        })
        state.last_gps_time = _now()
        _record_frame("gps", force=True)
    return {"status": "ok"}

@app.post("/pressure")
async def post_pressure(data: PressureData):
    """Ingest depth / pressure reading."""
    entry: dict[str, float] = {}
    if data.depth_m is not None:
        entry["depth_m"] = data.depth_m
    if data.pressure_bar is not None:
        entry["pressure_bar"] = data.pressure_bar
    _buffer_sensor("pressure", entry)
    await _broadcast({
        "type": "pressure",
        "depth_m": data.depth_m,
        "pressure_bar": data.pressure_bar,
        "timestamp": _last_ts("pressure"),
    })
    return {"status": "ok"}


async def _handle_dvl(data: DvlData) -> dict[str, Any]:
    """Ingest one raw DVL measurement and fuse it into the navigation filter."""
    timestamp_s = data.timestamp_s if data.timestamp_s is not None else _now()
    if not math.isfinite(timestamp_s):
        raise HTTPException(status_code=400, detail="timestamp_s must be finite")
    if data.velocity_dvl_m_s is None and data.position_nav_m is None:
        raise HTTPException(
            status_code=400,
            detail="DVL payload requires velocity_dvl_m_s, position_nav_m, or both",
        )
    if state.nav_origin_lat is None or state.nav_origin_lon is None:
        last_gps = _latest_gps()
        if last_gps and last_gps.get("lat") is not None and last_gps.get("lon") is not None:
            _ensure_nav_origin(last_gps["lat"], last_gps["lon"])
        else:
            _ensure_nav_origin(state.dr_lat if state.dr_lat is not None else BASE_LAT, state.dr_lon if state.dr_lon is not None else BASE_LON)

    raw = RawDvlMeasurement(
        timestamp_s=timestamp_s,
        velocity_dvl_m_s=data.velocity_dvl_m_s,
        velocity_covariance_dvl=data.velocity_covariance_dvl,
        position_nav_m=data.position_nav_m,
        position_covariance_nav=data.position_covariance_nav,
        valid_beams=data.valid_beams,
        mode=data.mode,
        status=data.status,
        altitude_m=data.altitude_m,
        velocity_scale_factor=data.velocity_scale_factor,
    )
    state.dvl_raw_count += 1
    state.dvl_sample_times.append(_now())
    try:
        status, output, corrected = _process_raw_dvl(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if status != "accepted" or output is None:
        state.dvl_rejected_count += 1
        state.latest_dvl = {
            "timestamp": timestamp_s,
            "received_at": _now(),
            "status": "rejected",
            "reason": status,
            "valid_beams": data.valid_beams,
            "altitude_m": data.altitude_m,
            "mode": data.mode,
            "raw_status": data.status,
        }
        _buffer_sensor("dvl", state.latest_dvl)
        await _broadcast({"type": "dvl", **state.latest_dvl})
        await _emit_event("dvl", f"DVL sample rejected: {status}", "warning", state.latest_dvl)
        return {
            "status": "rejected",
            "reason": status,
            "diagnostics": _dvl_status_payload(),
        }

    _remember_kf_output(output)

    if state.gps_denied and state.kf_output is not None:
        kf_latlon = _nav_m_to_latlon(state.kf_output.position_m)
        if kf_latlon is not None:
            state.dr_lat, state.dr_lon = kf_latlon
            velocity = np.asarray(state.kf_output.velocity_m_s, dtype=float)
            if velocity.shape[0] >= 2 and np.all(np.isfinite(velocity[:2])):
                state.dr_last_gps_speed_kn = float(np.linalg.norm(velocity[:2]) * 1.94384)

    velocity_body = None if corrected is None or corrected.velocity_body_m_s is None else [float(v) for v in np.asarray(corrected.velocity_body_m_s)]
    position_nav = None if corrected is None or corrected.position_nav_m is None else [float(v) for v in np.asarray(corrected.position_nav_m)]
    uncertainty = _navigation_uncertainty_payload(state.dr_heading)
    state.latest_dvl = {
        "timestamp": timestamp_s,
        "received_at": _now(),
        "status": "accepted",
        "dvl_update_applied": output.dvl_update_applied,
        "velocity_body_m_s": velocity_body,
        "position_nav_m": position_nav,
        "valid_beams": data.valid_beams,
        "altitude_m": data.altitude_m,
        "mode": data.mode,
        "raw_status": data.status,
        "estimate": _format_nav_output(output),
        **uncertainty,
    }
    _buffer_sensor("dvl", state.latest_dvl)
    await _broadcast({"type": "dvl", **state.latest_dvl})
    _record_frame("dvl", force=True)
    return {
        "status": "accepted",
        "dvl_update_applied": output.dvl_update_applied,
        "velocity_body_m_s": velocity_body,
        "position_nav_m": position_nav,
        "diagnostics": _dvl_status_payload(),
        **uncertainty,
    }


def _dvl_status_payload() -> dict[str, Any]:
    now = _now()
    recent = [ts for ts in state.dvl_sample_times if now - ts <= 5.0]
    latest_ts = state.latest_dvl.get("received_at") if state.latest_dvl else None
    return {
        "raw_measurements": state.dvl_raw_count,
        "rejected_measurements": state.dvl_rejected_count,
        "updates_applied": state.dvl_updates_applied,
        "quality_rejections": state.dvl_quality_rejections,
        "gate_rejections": state.dvl_gate_rejections,
        "rate_hz": round(len(recent) / 5.0, 2),
        "last_age_sec": None if latest_ts is None else round(max(0.0, now - float(latest_ts)), 2),
        "latest": state.latest_dvl,
    }


def _sensor_status(name: str, stale_after_s: float) -> dict[str, Any]:
    buf = state.buffers.get(name)
    now = _now()
    if not buf:
        return {"ok": False, "count": 0, "age_sec": None, "rate_hz": 0.0, "latest": None}
    latest = buf[-1]
    latest_ts = latest.get("timestamp", latest.get("received_at"))
    age = None if latest_ts is None else max(0.0, now - float(latest_ts))
    recent = [entry for entry in buf if now - float(entry.get("timestamp", entry.get("received_at", 0.0))) <= 5.0]
    return {
        "ok": age is not None and age <= stale_after_s,
        "count": len(buf),
        "age_sec": None if age is None else round(age, 2),
        "rate_hz": round(len(recent) / 5.0, 2),
        "latest": latest,
    }


def _uncertainty_reasons() -> list[str]:
    reasons: list[str] = []
    if state.gps_denied:
        reasons.append("GPS denied; position is dead-reckoned")
    gps_age = _gps_age_sec()
    if gps_age is None:
        reasons.append("No GPS fix has been received")
    elif gps_age > state.safety_config["max_gps_age_sec"]:
        reasons.append(f"GPS fix is stale ({gps_age:.1f}s old)")
    if state.dvl_rejected_count:
        reasons.append(f"{state.dvl_rejected_count} DVL sample(s) rejected")
    if state.dvl_raw_count == 0:
        reasons.append("No live DVL updates")
    if not state.mag_correction_enabled:
        reasons.append("Magnetometer heading correction is disabled")
    if state.drift_rate_dps > 0.05:
        reasons.append("IMU gyro drift estimate is elevated")
    return reasons


def _health_payload() -> dict[str, Any]:
    sensors = {
        "gps": _sensor_status("gps", 5.0 if state.mode == "simulator" else state.safety_config["max_gps_age_sec"]),
        "accelerometer": _sensor_status("accelerometer", 2.0),
        "gyroscope": _sensor_status("gyroscope", 2.0),
        "magnetometer": _sensor_status("magnetometer", 5.0),
        "barometer": _sensor_status("barometer", 5.0),
        "pressure": _sensor_status("pressure", 5.0),
        "dvl": _dvl_status_payload(),
    }
    critical_ok = sensors["accelerometer"]["ok"] and sensors["gyroscope"]["ok"]
    position_ok = sensors["gps"]["ok"] or state.gps_denied or state.dvl_updates_applied > 0
    status = "ok" if critical_ok and position_ok else "degraded"
    if not critical_ok:
        status = "critical"
    return {
        "status": status,
        "mode": state.mode,
        "gps_denied": state.gps_denied,
        "estimator": {
            "initialized": state.kf_output is not None,
            "last_timestamp_s": state.kf.last_timestamp_s,
            "uncertainty_reasons": _uncertainty_reasons(),
        },
        "sensors": sensors,
    }


def _validate_planner_points(points: list[tuple[float, float]]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    total_length = sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )
    min_depth = float("inf")
    shallow_samples = 0
    missing_depth_samples = 0
    samples_checked = 0
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        seg_len = max(1.0, math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        sample_count = max(2, min(20, int(seg_len // 25) + 2))
        for j in range(sample_count):
            alpha = j / max(1, sample_count - 1)
            x = p0[0] + (p1[0] - p0[0]) * alpha
            y = p0[1] + (p1[1] - p0[1]) * alpha
            lat, lon = _planner_m_to_latlon(x, y)
            depth = get_depth(lat, lon)
            samples_checked += 1
            if depth is None:
                missing_depth_samples += 1
            else:
                min_depth = min(min_depth, depth)
                if depth < state.safety_config["min_depth_m"]:
                    shallow_samples += 1

    if total_length > state.safety_config["max_route_length_m"]:
        warnings.append({
            "code": "route_long",
            "message": f"Route length {total_length:.0f}m exceeds configured limit",
            "limit_m": state.safety_config["max_route_length_m"],
        })
    if shallow_samples:
        warnings.append({
            "code": "shallow_depth",
            "message": f"{shallow_samples} route sample(s) are shallower than the configured limit",
            "min_depth_m": None if min_depth == float("inf") else round(min_depth, 1),
            "limit_m": state.safety_config["min_depth_m"],
        })
    if missing_depth_samples:
        warnings.append({
            "code": "missing_depth",
            "message": f"{missing_depth_samples} route sample(s) have no chart depth",
        })
    if state.gps_denied:
        warnings.append({"code": "gps_denied", "message": "GPS denial is active while planning"})
    if state.dvl_raw_count == 0:
        warnings.append({"code": "no_dvl", "message": "No live DVL samples have been received"})

    return {
        "ok": not any(w["code"] == "shallow_depth" for w in warnings),
        "warnings": warnings,
        "total_length_m": round(total_length, 2),
        "samples_checked": samples_checked,
        "min_depth_m": None if min_depth == float("inf") else round(min_depth, 1),
    }


def _recording_path(recording_id: str) -> Path:
    safe_id = "".join(ch for ch in recording_id if ch.isalnum() or ch in ("-", "_"))
    return RECORDINGS_DIR / f"{safe_id}.json"


def _save_recording(name: str | None = None) -> dict[str, Any]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    now = _now()
    recording_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + f"-{int((now % 1.0) * 1000):03d}"
    payload = {
        "id": recording_id,
        "name": name or recording_id,
        "created_at": now,
        "frames": state.recorded_frames,
        "frame_count": len(state.recorded_frames),
        "duration_sec": round(now - state.recording_start_time, 1),
    }
    _recording_path(recording_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state.last_recording_id = recording_id
    return payload


def _list_recordings() -> list[dict[str, Any]]:
    if not RECORDINGS_DIR.is_dir():
        return []
    out = []
    for path in sorted(RECORDINGS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({
                "id": data.get("id", path.stem),
                "name": data.get("name", path.stem),
                "created_at": data.get("created_at"),
                "frame_count": data.get("frame_count", len(data.get("frames", []))),
                "duration_sec": data.get("duration_sec"),
            })
        except Exception:
            continue
    return out


@app.post("/dvl")
async def post_dvl(data: DvlData):
    return await _handle_dvl(data)


@app.post("/api/dvl")
async def post_api_dvl(data: DvlData):
    return await _handle_dvl(data)


@app.get("/api/dvl/status")
async def get_dvl_status():
    return _dvl_status_payload()


@app.get("/api/health")
async def get_health():
    return _health_payload()


@app.get("/api/events")
async def get_events(limit: int = 100):
    limit = max(1, min(500, limit))
    return list(state.events)[-limit:]


@app.delete("/api/events")
async def clear_events():
    state.events.clear()
    return {"status": "cleared"}


@app.get("/api/config")
async def get_config():
    return {
        "config": _config_snapshot(),
        "path": str(CONFIG_PATH),
        "persisted": CONFIG_PATH.is_file(),
    }


@app.post("/api/config")
async def set_config(payload: ConfigIn):
    try:
        _apply_config(payload.config)
        _save_config()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _emit_event("config", "Configuration updated", data={"path": str(CONFIG_PATH)})
    return await get_config()


@app.get("/api/calibration/magnetometer")
async def get_magnetometer_calibration():
    return _config_snapshot()["magnetometer"]


@app.post("/api/calibration/magnetometer")
async def set_magnetometer_calibration(payload: ConfigIn):
    config = {"magnetometer": payload.config}
    try:
        _apply_config(config)
        _save_config()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _emit_event("calibration", "Magnetometer calibration updated")
    return _config_snapshot()["magnetometer"]


@app.post("/api/calibration/magnetometer/capture")
async def capture_magnetometer_calibration():
    mag_buf = state.buffers.get("magnetometer")
    if not mag_buf:
        raise HTTPException(status_code=400, detail="No magnetometer samples available")
    latest = mag_buf[-1]
    field = np.asarray([latest.get("x", 0.0), latest.get("y", 0.0), latest.get("z", 0.0)], dtype=float)
    magnitude = float(np.linalg.norm(field))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise HTTPException(status_code=400, detail="Latest magnetometer sample is invalid")
    state.expected_mag_field_uT = magnitude
    state.mag_correction = MagnetometerCorrectionLayer(
        calibration=MagnetometerCalibration(
            hard_iron_offset_uT=state.mag_hard_iron,
            soft_iron_matrix=state.mag_soft_iron,
            expected_field_magnitude_uT=state.expected_mag_field_uT,
        )
    )
    _save_config()
    await _emit_event("calibration", "Captured magnetometer field magnitude", data={"expected_field_magnitude_uT": magnitude})
    return _config_snapshot()["magnetometer"]


@app.post("/api/estimator/reset")
async def reset_estimator(payload: EstimatorResetIn | None = None):
    payload = payload or EstimatorResetIn()
    _reset_estimator(
        clear_buffers=payload.clear_buffers,
        clear_track=payload.clear_track,
        clear_events=payload.clear_events,
    )
    await _emit_event("estimator", "Estimator reset")
    return {"status": "reset", "health": _health_payload()}

# ---------------------------------------------------------------------------
# State & Track endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_api_state():
    """Return the latest vessel state from GPS, pressure, and vessel2."""
    result: dict[str, Any] = {
        "lat": None, "lon": None, "speed_kn": None, "heading_deg": None,
        "depth_m": None, "timestamp": None,
        "mode": state.mode,
        "gps_denied": state.gps_denied,
        "position_source": "dead_reckon" if state.gps_denied else "gps",
        "vessel2_lat": state.vessel2_lat, "vessel2_lon": state.vessel2_lon,
        "vessel2_heading_deg": state.vessel2_heading,
        "vessel2_speed_kn": round(state.vessel2_speed * 1.94384, 1),
    }
    gps_buf = state.buffers.get("gps")
    if gps_buf:
        last_gps = gps_buf[-1]
        result["lat"] = last_gps.get("lat")
        result["lon"] = last_gps.get("lon")
        result["speed_kn"] = last_gps.get("speed_kn")
        result["heading_deg"] = last_gps.get("heading_deg")
        result["timestamp"] = last_gps.get("timestamp")
    pressure_buf = state.buffers.get("pressure")
    if pressure_buf:
        last_pressure = pressure_buf[-1]
        result["depth_m"] = last_pressure.get("depth_m")
    return result


@app.get("/api/mode")
async def get_mode():
    return {
        "mode": state.mode,
        "available_modes": sorted(APP_MODES),
        "gps_denied": state.gps_denied,
        "replay_active": state.replay_task is not None,
    }


@app.post("/api/mode")
async def set_mode(payload: ModeIn):
    result = await _set_app_mode(payload.mode)
    _save_config()
    return result

@app.get("/api/track")
async def get_api_track():
    """Return track history as a GeoJSON Feature with a LineString."""
    if not state.track_buffer:
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": []},
            "properties": {"count": 0, "start_time": None, "end_time": None},
        }
    coordinates = [[e["lon"], e["lat"]] for e in state.track_buffer]
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {
            "count": len(state.track_buffer),
            "start_time": state.track_buffer[0]["timestamp"],
            "end_time": state.track_buffer[-1]["timestamp"],
        },
    }


@app.get("/api/depth")
async def get_api_depth(lat: float, lon: float):
    """Return seabed depth (meters) at a geographic point.

    Query params: ``lat`` (float), ``lon`` (float).
    Returns ``{"lat": 59.4, "lon": 10.6, "depth_m": 123.4}``
    or ``{"lat": 59.4, "lon": 10.6, "depth_m": null}`` for land / out-of-bounds.
    """
    depth = get_depth(lat, lon)
    return {"lat": lat, "lon": lon, "depth_m": depth}


@app.get("/api/bathymetry/info")
async def get_api_bathymetry_info():
    """Return metadata about the bathymetry grid."""
    return grid_info()


# ---------------------------------------------------------------------------
# Waypoints API
# ---------------------------------------------------------------------------


@app.post("/api/waypoints")
async def create_waypoint(wp: WaypointIn):
    if len(state.waypoints) >= 50:
        raise HTTPException(status_code=400, detail="Maximum 50 waypoints")
    depth = get_depth(wp.lat, wp.lon)
    if depth is None or depth < 2.0:
        raise HTTPException(
            status_code=400,
            detail="Waypoint on land, too shallow, or outside chart area — USV requires water depth ≥ 2 m",
        )
    entry = {
        "id": state.waypoint_next_id,
        "lat": wp.lat,
        "lon": wp.lon,
        "name": wp.name,
        "depth_m": round(depth, 1),
    }
    state.waypoint_next_id += 1
    state.waypoints.append(entry)
    return entry


@app.get("/api/waypoints")
async def list_waypoints():
    return list(state.waypoints)

@app.delete("/api/waypoints/{wp_id}")
async def delete_waypoint(wp_id: int):
    for i, wp in enumerate(state.waypoints):
        if wp["id"] == wp_id:
            state.waypoints.pop(i)
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Waypoint not found")


# ---------------------------------------------------------------------------
# Embedded planner control API
# ---------------------------------------------------------------------------

@app.post("/api/planner/path")
async def set_planner_path(path: PlannerPathIn):
    points: list[tuple[float, float]] = []
    for raw in path.waypoints:
        if len(raw) != 2:
            raise HTTPException(status_code=400, detail="Planner waypoints must be [x_m, y_m] pairs")
        x_m, y_m = float(raw[0]), float(raw[1])
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            raise HTTPException(status_code=400, detail="Planner waypoint coordinates must be finite")
        points.append((x_m, y_m))

    total_length = sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )
    if total_length <= 0.0:
        raise HTTPException(status_code=400, detail="Planner route must have non-zero length")

    validation = _validate_planner_points(points)
    _set_planner_path(points)
    if validation["warnings"]:
        _log_event("planner", "Planner route has safety warnings", "warning", validation)
    return {
        "status": "planner_route_active",
        "waypoints": _planner_geo_route(),
        "total_length_m": round(state.planner_total_length_m, 2),
        "validation": validation,
    }


@app.post("/api/planner/validate")
async def validate_planner_path(path: PlannerPathIn):
    points: list[tuple[float, float]] = []
    for raw in path.waypoints:
        if len(raw) != 2:
            raise HTTPException(status_code=400, detail="Planner waypoints must be [x_m, y_m] pairs")
        points.append((float(raw[0]), float(raw[1])))
    return _validate_planner_points(points)


@app.post("/api/planner/pause")
async def pause_planner_path():
    if state.planner_enabled and not state.planner_done:
        state.planner_paused = True
    return await planner_status()


@app.post("/api/planner/resume")
async def resume_planner_path():
    if state.planner_enabled and not state.planner_done:
        state.planner_paused = False
    return await planner_status()


@app.post("/api/planner/clear")
async def clear_planner_route():
    _clear_planner_path()
    return await planner_status()


@app.get("/api/planner/status")
async def planner_status():
    x_m, y_m, heading = _planner_xy_at_progress(state.planner_progress_m)
    lat, lon = _planner_m_to_latlon(x_m, y_m)
    return {
        "enabled": state.planner_enabled,
        "paused": state.planner_paused,
        "done": state.planner_done,
        "progress_m": round(state.planner_progress_m, 2),
        "total_length_m": round(state.planner_total_length_m, 2),
        "position": {"x_m": x_m, "y_m": y_m, "lat": lat, "lon": lon, "heading_deg": round(heading, 1)},
        "waypoints": _planner_geo_route(),
    }

# ---------------------------------------------------------------------------
# Risk assessment API
# ---------------------------------------------------------------------------

def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(1.0, value))


def _active_risk_route() -> tuple[str, list[dict[str, Any]]]:
    """Return the route that the operator is actually using for risk scoring."""
    if state.planner_enabled and len(state.planner_path_m) >= 2:
        points = []
        for i, (x_m, y_m) in enumerate(state.planner_path_m, start=1):
            lat, lon = _planner_m_to_latlon(x_m, y_m)
            points.append({
                "id": i,
                "name": f"P{i}",
                "lat": lat,
                "lon": lon,
                "x_m": x_m,
                "y_m": y_m,
            })
        return "planner", points

    if len(state.waypoints) >= 2:
        return "waypoints", [
            {
                "id": wp["id"],
                "name": wp.get("name"),
                "lat": wp["lat"],
                "lon": wp["lon"],
            }
            for wp in state.waypoints
        ]

    return "none", []


def _latest_speed_m_s() -> float:
    last_gps = _latest_gps()
    if last_gps and last_gps.get("speed_kn") is not None:
        speed_kn = float(last_gps["speed_kn"])
        if math.isfinite(speed_kn) and speed_kn >= 0.0:
            return speed_kn * 0.514444
    return SURVEY_SPEED_MS


def _segment_depth_stats(lat1: float, lon1: float, lat2: float, lon2: float) -> dict[str, Any]:
    distance_m = _haversine(lat1, lon1, lat2, lon2)
    sample_count = max(2, min(60, int(distance_m // 25.0) + 2))
    depths: list[float] = []
    unknown_count = 0
    shallow_count = 0
    for i in range(sample_count):
        alpha = i / max(1, sample_count - 1)
        lat = lat1 + (lat2 - lat1) * alpha
        lon = lon1 + (lon2 - lon1) * alpha
        depth = get_depth(lat, lon)
        if depth is None:
            unknown_count += 1
            continue
        depths.append(float(depth))
        if depth < state.safety_config["min_depth_m"]:
            shallow_count += 1

    if depths:
        min_depth = min(depths)
        avg_depth = sum(depths) / len(depths)
    else:
        min_depth = None
        avg_depth = None

    return {
        "sample_count": sample_count,
        "known_count": len(depths),
        "unknown_count": unknown_count,
        "shallow_count": shallow_count,
        "min_depth_m": min_depth,
        "avg_depth_m": avg_depth,
    }


def _risk_depth_factor(depth_stats: dict[str, Any]) -> float:
    min_depth = depth_stats["min_depth_m"]
    if min_depth is None:
        return 0.85
    safe_depth = state.safety_config["min_depth_m"]
    if min_depth <= safe_depth:
        return 1.0
    clearance = min_depth - safe_depth
    unknown_penalty = 0.25 * (depth_stats["unknown_count"] / max(1, depth_stats["sample_count"]))
    return _clamp01(math.exp(-clearance / 12.0) + unknown_penalty)


def _risk_dvl_factor(dvl_status: dict[str, Any]) -> float:
    if state.gps_denied:
        if dvl_status["updates_applied"] <= 0:
            return 1.0
        if dvl_status["last_age_sec"] is None:
            return 0.9
        return _clamp01(float(dvl_status["last_age_sec"]) / 10.0)
    if dvl_status["raw_measurements"] <= 0:
        return 0.25
    rejection_ratio = dvl_status["rejected_measurements"] / max(1, dvl_status["raw_measurements"])
    age = 0.0 if dvl_status["last_age_sec"] is None else float(dvl_status["last_age_sec"])
    return _clamp01(0.65 * rejection_ratio + 0.35 * min(age / 30.0, 1.0))


def _risk_sensor_factor(health: dict[str, Any]) -> float:
    sensors = health.get("sensors", {})
    bad = 0
    required = ("accelerometer", "gyroscope", "magnetometer")
    for name in required:
        if not sensors.get(name, {}).get("ok", False):
            bad += 1
    if not sensors.get("gps", {}).get("ok", False) and not state.gps_denied:
        bad += 1
    return _clamp01(bad / 4.0)


def _risk_uncertainty_factor() -> float:
    payload = _dead_reckon_payload() if state.gps_denied else _navigation_uncertainty_payload(
        (_latest_gps() or {}).get("heading_deg")
    )
    uncertainty_m = float(payload.get("uncertainty_m", 0.0) or 0.0)
    limit = max(1.0, state.safety_config["max_uncertainty_m"])
    return _clamp01(uncertainty_m / limit)


@app.get("/api/risk")
async def get_risk():
    now = _now()
    gps_age_value = _gps_age_sec(now)
    gps_age = gps_age_value if gps_age_value is not None else 9999.0
    speed = _latest_speed_m_s()
    route_source, route_points = _active_risk_route()
    dvl_status = _dvl_status_payload()
    health = _health_payload()
    uncertainty_factor = _risk_uncertainty_factor()
    dvl_factor = _risk_dvl_factor(dvl_status)
    sensor_factor = _risk_sensor_factor(health)
    gps_factor = 1.0 if gps_age_value is None else _clamp01(1.0 - math.exp(-gps_age / 120.0))
    if state.gps_denied:
        gps_factor = max(gps_factor, 0.8)

    segments: list[dict[str, Any]] = []
    for i in range(len(route_points) - 1):
        wp1, wp2 = route_points[i], route_points[i + 1]
        d = _haversine(wp1["lat"], wp1["lon"], wp2["lat"], wp2["lon"])
        travel_time = d / max(speed, 0.1)
        heading_error = state.drift_rate_dps * travel_time
        depth_stats = _segment_depth_stats(wp1["lat"], wp1["lon"], wp2["lat"], wp2["lon"])
        risk_drift = _clamp01(max(heading_error / 30.0, uncertainty_factor * 0.75))
        risk_gps = gps_factor
        risk_distance = _clamp01(d / state.safety_config["max_route_length_m"])
        risk_depth = _risk_depth_factor(depth_stats)
        risk_score = (
            0.24 * risk_drift
            + 0.20 * risk_gps
            + 0.14 * risk_distance
            + 0.18 * risk_depth
            + 0.10 * dvl_factor
            + 0.08 * uncertainty_factor
            + 0.06 * sensor_factor
        )
        if depth_stats["shallow_count"] > 0:
            risk_score = max(risk_score, 0.85)
        if depth_stats["min_depth_m"] is None:
            risk_score = max(risk_score, 0.55)
        risk_score = _clamp01(risk_score)
        segments.append({
            "from_id": wp1["id"],
            "to_id": wp2["id"],
            "from_name": wp1.get("name"),
            "to_name": wp2.get("name"),
            "from_lat": wp1["lat"],
            "from_lon": wp1["lon"],
            "to_lat": wp2["lat"],
            "to_lon": wp2["lon"],
            "distance_m": round(d, 1),
            "travel_time_sec": round(travel_time, 1),
            "risk_score": round(risk_score, 3),
            "factors": {
                "drift": round(risk_drift, 3),
                "gps": round(risk_gps, 3),
                "distance": round(risk_distance, 3),
                "depth": round(risk_depth, 3),
                "dvl": round(dvl_factor, 3),
                "uncertainty": round(uncertainty_factor, 3),
                "sensors": round(sensor_factor, 3),
            },
            "heading_error_deg": round(heading_error, 2),
            "avg_depth_m": None if depth_stats["avg_depth_m"] is None else round(depth_stats["avg_depth_m"], 1),
            "min_depth_m": None if depth_stats["min_depth_m"] is None else round(depth_stats["min_depth_m"], 1),
            "unknown_depth_samples": depth_stats["unknown_count"],
            "shallow_depth_samples": depth_stats["shallow_count"],
        })

    mission_risk = max((s["risk_score"] for s in segments), default=0.0)
    return {
        "route_source": route_source,
        "route_points": route_points,
        "segments": segments,
        "mission_risk": mission_risk,
        "imu": {
            "drift_rate_dps": round(state.drift_rate_dps, 4),
            "gyro_bias_dps": round(state.gyro_z_bias * (180.0 / math.pi), 4),
            "accel_variance": round(state.accel_variance, 4),
            "last_gps_sec": round(gps_age, 1),
        },
        "dvl": dvl_status,
        "health": {"status": health["status"], "uncertainty_reasons": health["estimator"]["uncertainty_reasons"]},
    }


@app.get("/api/recording/start")
async def recording_start():
    state.recording = True; state.recorded_frames = []
    state.recording_start_time = state.last_record_time = _now()
    await _emit_event("recording", "Recording started")
    return {"status": "recording"}


@app.get("/api/recording/stop")
async def recording_stop():
    state.recording = False
    n = len(state.recorded_frames)
    dur = _now() - state.recording_start_time
    saved = _save_recording()
    await _emit_event("recording", "Recording stopped", data={"recording_id": saved["id"], "frames": n})
    return {"frames": n, "duration_sec": round(dur, 1), "recording_id": saved["id"], "saved_path": str(_recording_path(saved["id"]))}


@app.get("/api/recording/status")
async def recording_status():
    elapsed = _now() - state.recording_start_time if state.recording else 0.0
    return {
        "recording": state.recording,
        "frames": len(state.recorded_frames),
        "elapsed_sec": round(elapsed, 1),
        "last_recording_id": state.last_recording_id,
    }


@app.get("/api/recording/playback")
async def recording_playback(start: int = 0, end: int | None = None, recording_id: str | None = None):
    if recording_id:
        path = _recording_path(recording_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        return frames[start:end]
    if end is None:
        end = len(state.recorded_frames)
    return state.recorded_frames[start:end]


@app.get("/api/recording/list")
async def recording_list():
    return {"recordings": _list_recordings()}


@app.get("/api/export/geojson")
async def export_geojson():
    features = []
    if state.track_buffer:
        features.append({
            "type": "Feature", "geometry": {"type": "LineString",
                "coordinates": [[e["lon"], e["lat"]] for e in state.track_buffer]},
            "properties": {"vessel": "survey", "count": len(state.track_buffer)},
        })
    if state.vessel2_track:
        features.append({
            "type": "Feature", "geometry": {"type": "LineString",
                "coordinates": [[e["lon"], e["lat"]] for e in state.vessel2_track]},
            "properties": {"vessel": "support", "count": len(state.vessel2_track)},
        })
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/export/csv")
async def export_csv():
    rows = ["timestamp,vessel,lat,lon,heading_deg,speed_kn,depth_m"]
    rows += [f"{e.get('timestamp', '')},survey,{e['lat']},{e['lon']},{e.get('heading_deg', '')},{e.get('speed_kn', '')}," for e in state.track_buffer]
    rows += [f"{e.get('timestamp', '')},support,{e['lat']},{e['lon']},{e.get('heading_deg', '')},{e.get('speed_kn', '')}," for e in state.vessel2_track]
    return Response(content="\n".join(rows), media_type="text/csv")


# ---------------------------------------------------------------------------
# Real-data replay API
# ---------------------------------------------------------------------------



@app.post("/api/replay/start")
async def replay_start(speed: float = 1.0):
    """Start replaying the Simris field dataset at the given speed multiplier."""
    await _stop_replay_task()
    state.replay_speed = max(speed, 0.01)
    state.replay = SimrisReplay()
    state.replay.running = True
    state.replay.speed = state.replay_speed
    state.mode = "replay"
    async def _run():
        await _replay_runner(state.replay, speed=state.replay_speed)
    state.replay_task = asyncio.create_task(_run())
    await _emit_event("replay", "Replay started", data={"speed": state.replay_speed})
    return {"status": "running"}


@app.post("/api/replay/stop")
async def replay_stop():
    """Stop an active replay."""
    await _stop_replay_task()
    if state.mode == "replay":
        state.mode = "live"
    await _emit_event("replay", "Replay stopped")
    return {"status": "stopped"}


@app.get("/api/replay/status")
async def replay_status():
    """Return current replay status."""
    return state.replay.get_status()


@app.post("/api/replay/speed")
async def replay_speed(multiplier: float = 1.0):
    """Adjust the active replay speed multiplier."""
    state.replay_speed = max(multiplier, 0.01)
    state.replay.speed = state.replay_speed
    _save_config()
    return {"status": "ok", "speed": state.replay_speed}

# ---------------------------------------------------------------------------
# GPS Denial & Dead Reckoning
# ---------------------------------------------------------------------------

@app.get("/api/gps-deny/on")
async def gps_deny_on():
    _enable_gps_denial()
    await _emit_event("gps", "GPS denial enabled", "warning")
    return {"status": "gps_denied"}

@app.get("/api/gps-deny/off")
async def gps_deny_off():
    _disable_gps_denial()
    await _emit_event("gps", "GPS restored")
    return {"status": "gps_restored"}

@app.get("/api/gps-deny/status")
async def gps_deny_status():
    secs = (_now() - state.gps_deny_start_time) if state.gps_denied else 0.0
    return {"denied": state.gps_denied, "seconds_without_gps": round(secs, 1)}

@app.post("/api/gps-denial")
async def set_gps_denial(payload: GpsDenialIn):
    if payload.enabled:
        _enable_gps_denial()
        await _emit_event("gps", "GPS denial enabled", "warning")
    else:
        _disable_gps_denial()
        await _emit_event("gps", "GPS restored")
    return _gps_denial_summary()

@app.get("/api/gps-denial")
async def get_gps_denial():
    return _gps_denial_summary()

@app.get("/api/dead-reckon")
async def get_dead_reckon():
    return _dead_reckon_payload()

@app.post("/api/dead-reckon")
async def post_dead_reckon():
    return _dead_reckon_payload()

@app.get("/api/dead-reckon/status")
async def dead_reckon_status():
    payload = _dead_reckon_payload()
    return {
        "gps_denied": state.gps_denied,
        "gps_age_sec": payload["gps_age_sec"],
        "fix_type": payload["fix_type"],
        "estimated_position": payload["estimated_position"],
        "drift": payload["drift_stats"],
        "ellipse": payload["ellipse"],
        "mag_correction_enabled": state.mag_correction_enabled,
    }


@app.post("/api/dead-reckon/mag-correction")
async def set_mag_correction(payload: MagCorrectionIn):
    state.mag_correction_enabled = payload.enabled
    if payload.enabled:
        state.dr_heading = state.mag_heading
        state.dr_heading_source = "magnetometer"
    else:
        last_gps = _latest_gps()
        if last_gps and last_gps.get("heading_deg") is not None:
            state.dr_heading = float(last_gps["heading_deg"])
            state.dr_heading_source = "last_gps_heading"
        else:
            state.dr_heading_source = "fallback_heading"
    return {
        "enabled": state.mag_correction_enabled,
        "mag_heading_deg": round(state.mag_heading, 1),
        "dr_heading_deg": round(state.dr_heading, 1),
        "heading_source": state.dr_heading_source,
    }


@app.get("/api/dead-reckon/mag-correction")
async def get_mag_correction():
    return {
        "enabled": state.mag_correction_enabled,
        "mag_heading_deg": round(state.mag_heading, 1),
        "dr_heading_deg": round(state.dr_heading, 1),
        "heading_source": state.dr_heading_source,
    }


# ---------------------------------------------------------------------------
# React planner WebSocket endpoint
# ---------------------------------------------------------------------------

async def _planner_ws_send_status(ws: WebSocket, sim_state: str) -> None:
    await ws.send_json({"type": "status", "state": sim_state})


def _parse_planner_points(raw_points: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list):
        raise ValueError("waypoints must be a list")
    points: list[tuple[float, float]] = []
    for raw in raw_points:
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            raise ValueError("planner waypoints must be [x_m, y_m] pairs")
        x_m, y_m = float(raw[0]), float(raw[1])
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            raise ValueError("planner waypoint coordinates must be finite")
        points.append((x_m, y_m))
    return points


@app.websocket("/planner/ws")
async def planner_websocket_endpoint(ws: WebSocket):
    await ws.accept()
    path: list[tuple[float, float]] = []
    starts: list[float] = []
    total_length_m = 0.0
    progress_m = 0.0
    sim_time_s = 0.0
    sim_state = "idle"
    cov = [row[:] for row in PLANNER_INIT_COV]
    current_segment = 0

    await _planner_ws_send_status(ws, sim_state)
    last_status_sent = sim_state
    try:
        while True:
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=PLANNER_TICK_DT)
            except asyncio.TimeoutError:
                text = None

            if text is not None:
                try:
                    import json
                    msg = json.loads(text)
                    msg_type = msg.get("type")
                    if msg_type == "set_path":
                        path = _parse_planner_points(msg.get("waypoints"))
                        starts, total_length_m = _metric_path(path)
                        progress_m = 0.0
                        sim_time_s = 0.0
                        cov = [row[:] for row in PLANNER_INIT_COV]
                        current_segment = 0
                        sim_state = "idle"
                        if len(path) >= 2 and total_length_m > 0.0:
                            _set_planner_path(path)
                        else:
                            _clear_planner_path()
                    elif msg_type == "play":
                        if len(path) >= 2 and total_length_m > 0.0 and sim_state != "done":
                            sim_state = "running"
                            if state.planner_enabled and not state.planner_done:
                                state.planner_paused = False
                    elif msg_type == "pause":
                        if sim_state == "running":
                            sim_state = "paused"
                        if state.planner_enabled and not state.planner_done:
                            state.planner_paused = True
                    elif msg_type == "reset":
                        path = []
                        starts = []
                        total_length_m = 0.0
                        progress_m = 0.0
                        sim_time_s = 0.0
                        cov = [row[:] for row in PLANNER_INIT_COV]
                        current_segment = 0
                        sim_state = "idle"
                        _clear_planner_path()
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})

            if sim_state == "running":
                sim_time_s += PLANNER_TICK_DT
                progress_m = min(total_length_m, progress_m + PLANNER_SPEED_MS * PLANNER_TICK_DT)
                cov[0][0] += PLANNER_COV_GROWTH_DIAG * PLANNER_TICK_DT
                cov[1][1] += PLANNER_COV_GROWTH_DIAG * PLANNER_TICK_DT
                cov[0][1] += PLANNER_COV_GROWTH_OFFDIAG * PLANNER_TICK_DT
                cov[1][0] = cov[0][1]
                while current_segment + 1 < len(path) - 1 and progress_m >= starts[current_segment + 1]:
                    current_segment += 1
                    cov[0][0] *= PLANNER_LANDMARK_FIX
                    cov[1][1] *= PLANNER_LANDMARK_FIX
                    cov[0][1] *= PLANNER_LANDMARK_FIX
                    cov[1][0] = cov[0][1]
                if progress_m >= total_length_m:
                    progress_m = total_length_m
                    sim_state = "done"

                mean_x, mean_y = _xy_on_metric_path(path, starts, total_length_m, progress_m)
                await ws.send_json({
                    "type": "tick",
                    "t": sim_time_s,
                    "mean": [mean_x, mean_y],
                    "cov": cov,
                })

            if sim_state != last_status_sent:
                await _planner_ws_send_status(ws, sim_state)
                last_status_sent = sim_state
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.websockets.add(ws)
    history: dict[str, list[dict[str, Any]]] = {}
    for key, buf in state.buffers.items():
        if buf:
            history[key] = list(buf)
    await ws.send_json({"type": "history", "data": history})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.websockets.discard(ws)

# ---------------------------------------------------------------------------
# Embedded React planner
# ---------------------------------------------------------------------------

def _planner_unavailable_response() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>React Planner Not Built</title>
          <style>
            body{margin:0;min-height:100vh;display:grid;place-items:center;background:#08111f;color:#d0d8e8;font-family:Segoe UI,system-ui,sans-serif}
            main{max-width:520px;padding:24px;border:1px solid rgba(0,212,170,.18);background:#0f1f38;border-radius:6px}
            h1{font-size:18px;margin:0 0 10px;color:#00d4aa}
            p{font-size:13px;line-height:1.5;color:#9fb0c8}
            code{color:#d0d8e8}
          </style>
        </head>
        <body>
          <main>
            <h1>React planner is not built</h1>
            <p>Run <code>cd frontend/web && npm run build</code>, then reload the command center planner.</p>
          </main>
        </body>
        </html>
        """,
        status_code=503,
    )


def _serve_planner_file(asset_path: str = ""):
    if not PLANNER_INDEX.is_file():
        return _planner_unavailable_response()
    if not asset_path:
        return FileResponse(PLANNER_INDEX)

    dist_root = PLANNER_DIST_DIR.resolve()
    requested = (PLANNER_DIST_DIR / asset_path).resolve()
    if requested != dist_root and dist_root not in requested.parents:
        raise HTTPException(status_code=404, detail="Planner asset not found")
    if requested.is_file():
        return FileResponse(requested)
    return FileResponse(PLANNER_INDEX)


@app.head("/planner", include_in_schema=False)
@app.head("/planner/", include_in_schema=False)
@app.get("/planner", include_in_schema=False)
@app.get("/planner/", include_in_schema=False)
async def planner_index():
    return _serve_planner_file()


@app.head("/planner/{asset_path:path}", include_in_schema=False)
@app.get("/planner/{asset_path:path}", include_in_schema=False)
async def planner_asset(asset_path: str):
    return _serve_planner_file(asset_path)

# ---------------------------------------------------------------------------
# Sensor simulator (background asyncio task)
# ---------------------------------------------------------------------------

async def _simulator_loop() -> None:
    """Generate realistic maritime sensor data at ~20 Hz."""
    sim_time = 0.0
    phase = 0          # index into SURVEY_HEADINGS
    phase_dist = 0.0   # metres travelled in current phase
    lat_off = 0.0
    lon_off = 0.0
    sim_tick = 0

    while True:
        await asyncio.sleep(SIMULATOR_RATE)
        if state.mode not in {"simulator", "training"}:
            continue
        sim_time += SIMULATOR_RATE
        t = sim_time
        sim_tick += 1

        step_m = SURVEY_SPEED_MS * SIMULATOR_RATE
        if state.planner_enabled:
            _, _, heading = _planner_xy_at_progress(state.planner_progress_m)
        else:
            heading = SURVEY_HEADINGS[phase]

        # ---- IMU sensors ----
        ax = 0.3 * math.sin(0.5 * t) + 0.2 * _smooth_noise(t, 1.0)
        ay = 0.2 * math.cos(0.3 * t) + 0.2 * _smooth_noise(t, 2.0)
        az = 9.81 + 0.1 * math.sin(0.7 * t) + 0.1 * _smooth_noise(t, 3.0)

        roll_rate = 0.02 * math.sin(0.8 * t) + 0.01 * _smooth_noise(t, 4.0)
        pitch_rate = 0.015 * math.cos(0.6 * t) + 0.01 * _smooth_noise(t, 5.0)
        yaw_rate = 0.01 * math.sin(0.4 * t) + 0.005 * _smooth_noise(t, 6.0)

        state.gyro_z_samples.append(yaw_rate)
        state.gyro_z_bias = sum(state.gyro_z_samples) / len(state.gyro_z_samples)
        acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
        state.accel_mags.append(acc_mag)
        if len(state.accel_mags) > 1:
            m = sum(state.accel_mags) / len(state.accel_mags)
            state.accel_variance = sum((v - m) ** 2 for v in state.accel_mags) / len(state.accel_mags)
        _recalc_drift()

        roll = 5.0 * math.sin(0.1 * t)
        pitch = 3.0 * math.cos(0.08 * t)
        yaw = (heading + 10.0 * math.sin(0.05 * t)) % 360.0

        # ---- Magnetometer (compass) ----
        mag_x = 25.0 * math.cos(math.radians(yaw)) + 2.0 * _smooth_noise(t, 7.0)
        mag_y = 25.0 * math.sin(math.radians(yaw)) + 2.0 * _smooth_noise(t, 8.0)
        mag_z = 45.0 + 3.0 * math.sin(0.05 * t)
        mag_heading = yaw  # compass heading = yaw
        state.mag_heading = mag_heading
        state.dr_heading = mag_heading

        # ---- Barometer ----
        atmospheric_pressure = 1013.25 + 2.0 * math.sin(0.01 * t) + 0.5 * _smooth_noise(t, 9.0)

        imu_samples = [
            ("accelerometer", {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)}),
            ("magnetometer", {"x": round(mag_x, 2), "y": round(mag_y, 2), "z": round(mag_z, 2), "heading_deg": round(mag_heading, 1)}),
            ("gyroscope", {"x": round(roll_rate, 6), "y": round(pitch_rate, 6), "z": round(yaw_rate, 6)}),
            ("orientation", {"x": round(roll, 2), "y": round(pitch, 2), "z": round(yaw, 2)}),
        ]

        for sensor_name, values in imu_samples:
            _buffer_sensor(sensor_name, values)
            await _broadcast({
                "type": "imu",
                "sensor": sensor_name,
                **values,
                "timestamp": _last_ts(sensor_name),
            })
            timestamp_s = _last_ts(sensor_name)
            if sensor_name == "accelerometer":
                state.latest_accel_m_s2 = [values["x"], values["y"], values["z"]]
                _maybe_process_imu(timestamp_s)
            elif sensor_name == "gyroscope":
                state.latest_gyro_rad_s = [values["x"], values["y"], values["z"]]
                _maybe_process_imu(timestamp_s)

        # ---- Barometer broadcast (separate from IMU, has its own type) ----
        _buffer_sensor("barometer", {"pressure_hpa": round(atmospheric_pressure, 1)})
        await _broadcast({
            "type": "barometer",
            "pressure_hpa": round(atmospheric_pressure, 1),
            "timestamp": _last_ts("barometer"),
        })

        # ---- GPS: embedded planner route, or fallback slow survey pattern ----
        if state.planner_enabled:
            x_m, y_m, heading = _advance_planner_route(step_m)
            lat, lon = _planner_m_to_latlon(x_m, y_m)
        else:
            if phase == 0:       # east
                lon_off += step_m * 1.8e-5
            elif phase == 1:     # north
                lat_off += step_m * 9.0e-6
            elif phase == 2:     # west
                lon_off -= step_m * 1.8e-5
            else:                # south
                lat_off -= step_m * 9.0e-6

            phase_dist += step_m
            leg_len = SURVEY_LEG_LENGTH if phase in {0, 2} else SURVEY_LEG_SPACING
            if phase_dist >= leg_len:
                phase = (phase + 1) % 4
                phase_dist = 0.0

            lat = BASE_LAT + lat_off + 5e-5 * math.sin(0.1 * t)
            lon = BASE_LON + lon_off + 5e-5 * math.cos(0.08 * t)

        if not state.gps_denied:
            gps_update = {
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "speed_kn": round(SURVEY_SPEED_MS * 1.94384, 1),
                "heading_deg": round(heading, 1),
                **_navigation_uncertainty_payload(round(heading, 1)),
            }
            _buffer_sensor("gps", gps_update)
            await _broadcast({
                "type": "gps",
                **gps_update,
                "timestamp": _last_ts("gps"),
            })
            state.track_buffer.append({
                "lat": gps_update["lat"],
                "lon": gps_update["lon"],
                "timestamp": _last_ts("gps"),
                "speed_kn": gps_update["speed_kn"],
                "heading_deg": gps_update["heading_deg"],
            })

        # Dead reckoning during GPS denial
        if state.gps_denied and state.dr_lat is not None:
            hdg = math.radians(state.dr_heading)
            spd = state.dr_last_gps_speed_kn * 0.514444
            dist = spd * SIMULATOR_RATE
            state.dr_lat += dist * math.cos(hdg) / 111320.0
            state.dr_lon += dist * math.sin(hdg) / (111320.0 * math.cos(math.radians(state.dr_lat)))

        # ---- Dead reckon WS (1 Hz during denial) ----
        if state.gps_denied and state.dr_lat is not None and sim_tick % 20 == 0:
            await _broadcast(_dead_reckon_payload())

        # ---- Vessel 2 (support) figure-8 holding pattern ----
        cos_lat = math.cos(math.radians(BASE_LAT))
        v2_angle = V2_OMEGA * sim_time
        v2_lat = V2_CENTER_LAT + V2_RADIUS_LAT * math.sin(v2_angle)
        v2_lon = V2_CENTER_LON + V2_RADIUS_LON * math.sin(2.0 * v2_angle)
        v2_dlat = V2_RADIUS_LAT * V2_OMEGA * math.cos(v2_angle)
        v2_dlon = V2_RADIUS_LON * 2.0 * V2_OMEGA * math.cos(2.0 * v2_angle)
        v2_hdg = math.degrees(math.atan2(v2_dlon * cos_lat, v2_dlat)) % 360.0

        state.vessel2_lat = round(v2_lat, 6)
        state.vessel2_lon = round(v2_lon, 6)
        state.vessel2_heading = round(v2_hdg, 1)
        state.vessel2_speed = V2_SPEED_MS

        support_timestamp = _now()
        await _broadcast({
            "type": "gps",
            "vessel": "support",
            "lat": round(v2_lat, 6),
            "lon": round(v2_lon, 6),
            "speed_kn": round(V2_SPEED_MS * 1.94384, 1),
            "heading_deg": round(v2_hdg, 1),
            "timestamp": support_timestamp,
        })
        state.vessel2_track.append({
            "lat": round(v2_lat, 6),
            "lon": round(v2_lon, 6),
            "timestamp": _now(),
            "speed_kn": round(V2_SPEED_MS * 1.94384, 1),
            "heading_deg": round(v2_hdg, 1),
        })
        depth = 5.0 + 3.0 * math.sin(0.03 * t)
        pressure_bar = round(depth * 0.0981 + 1.0, 3)
        _buffer_sensor("pressure", {"depth_m": round(depth, 2), "pressure_bar": pressure_bar})
        await _broadcast({
            "type": "pressure",
            "depth_m": round(depth, 2),
            "pressure_bar": pressure_bar,
            "timestamp": _last_ts("pressure"),
        })

        # ---- Recording: save a frame every second ----
        _record_frame("simulator")

# ---------------------------------------------------------------------------
# Static files (mount after API routes so routes take precedence)
# ---------------------------------------------------------------------------
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
