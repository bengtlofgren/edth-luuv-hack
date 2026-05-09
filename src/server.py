"""
Maritime Mission Planner - Demo 2 Backend Server

FastAPI server with REST endpoints for IMU, GPS, and pressure data,
WebSocket broadcast, data buffering, built-in sensor simulator,
track history, state endpoint, and static file serving.
"""

import asyncio
import io
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
from dvl_correction.src.magnetometer import (
    MagnetometerCalibration,
    MagnetometerCorrectionLayer,
    MagnetometerQualityConfig,
    RawMagnetometerMeasurement,
)
from dvl_correction.src.dvl_imu_kalman import (
    DvlImuKalmanLayer,
    ImuSample,
    KalmanConfig,
    NavigationOutput,
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
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUFFER_SIZE = 300
SIMULATOR_RATE = 0.05  # seconds per tick (20 Hz)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
SIMULATOR_DISABLED = os.environ.get(
    "DISABLE_SIMULATOR", os.environ.get("SIMULATOR_DISABLED", "")
).lower() in {"1", "true", "yes", "on"}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PLANNER_DIST_DIR = _PROJECT_ROOT / "frontend" / "web" / "dist"
PLANNER_INDEX = PLANNER_DIST_DIR / "index.html"

SENSOR_NAMES = {"accelerometer", "magnetometer", "gyroscope", "orientation", "barometer"}

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

        # GPS denial & dead reckoning
        self.gps_denied: bool = False
        self.gps_deny_start_time: float | None = None
        self.dr_lat: float | None = None
        self.dr_lon: float | None = None
        self.dr_heading: float = 0.0

        # Teammate estimation tools
        self.kf: DvlImuKalmanLayer = DvlImuKalmanLayer(config=KalmanConfig())
        self.mag_correction: MagnetometerCorrectionLayer = MagnetometerCorrectionLayer()
        self.kf_output: NavigationOutput | None = None
        self.kf_covariance: list[list[float]] = [[1.0, 0.0], [0.0, 1.0]]

        # Real data replay
        self.replay: SimrisReplay = SimrisReplay()
        self.replay_task: asyncio.Task[None] | None = None
        self.replay_speed: float = 1.0
        self.dr_last_gps_speed_kn: float = 0.0

        # Magnetometer calibration
        self.mag_hard_iron: list[float] = [0.0, 0.0, 0.0]
        self.mag_soft_iron: list[list[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.mag_heading: float = 0.0

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


def _planner_geo_route() -> list[dict[str, float]]:
    return [
        {"x_m": x, "y_m": y, "lat": _planner_m_to_latlon(x, y)[0], "lon": _planner_m_to_latlon(x, y)[1]}
        for x, y in state.planner_path_m
    ]


def _recalc_drift() -> None:
    """Recalculate drift rate from current gyro bias and accelerometer noise."""
    nf = 1.0 + (math.sqrt(state.accel_variance) / 9.81 if state.accel_variance > 0 else 0)
    state.drift_rate_dps = abs(state.gyro_z_bias) * (180.0 / math.pi) * nf


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
            # Use teammate MagnetometerCorrectionLayer for proper calibration
            try:
                raw = RawMagnetometerMeasurement(
                    x=sensor.values.get("x", 0.0),
                    y=sensor.values.get("y", 0.0),
                    z=sensor.values.get("z", 0.0),
                )
                corrected = state.mag_correction.correct(raw, roll=0.0, pitch=0.0)
                if corrected is not None:
                    state.mag_heading = math.degrees(corrected.yaw) % 360.0
                    state.dr_heading = state.mag_heading
            except Exception:
                pass
    return {"status": "ok"}

@app.post("/gps")
async def post_gps(data: GpsData):
    """Ingest a GPS position fix."""
    entry: dict[str, Any] = {"lat": data.lat, "lon": data.lon}
    if data.speed_kn is not None:
        entry["speed_kn"] = data.speed_kn
    if data.heading_deg is not None:
        entry["heading_deg"] = data.heading_deg
    _buffer_sensor("gps", entry)
    # Kalman GPS correction using teammate DvlImuKalmanLayer
    try:
        import numpy as np
        from dvl_correction.src.dvl_imu_kalman import CorrectedDvlMeasurement
        dvl = CorrectedDvlMeasurement(
            position=np.array([data.lat, data.lon, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            position_covariance=np.diag([4.0, 4.0, 0.01]),
            velocity_covariance=np.diag([0.5, 0.5, 0.5]),
            beams_valid=4, status=0,
        )
        state.kf.correct(dvl)
        out = state.kf.output()
        if out:
            state.kf_output = out
            state.kf_covariance = out.position_covariance[:2, :2].tolist()
    except Exception:
        pass

    if not state.gps_denied:
        await _broadcast({
            "type": "gps",
            "lat": data.lat,
            "lon": data.lon,
            "speed_kn": data.speed_kn,
            "heading_deg": data.heading_deg,
            "timestamp": _last_ts("gps"),
        })
        state.track_buffer.append({
            "lat": data.lat,
            "lon": data.lon,
            "timestamp": _now(),
            "speed_kn": data.speed_kn,
            "heading_deg": data.heading_deg,
        })
        state.last_gps_time = _now()
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

# ---------------------------------------------------------------------------
# State & Track endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_api_state():
    """Return the latest vessel state from GPS, pressure, and vessel2."""
    result: dict[str, Any] = {
        "lat": None, "lon": None, "speed_kn": None, "heading_deg": None,
        "depth_m": None, "timestamp": None,
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

    _set_planner_path(points)
    return {
        "status": "planner_route_active",
        "waypoints": _planner_geo_route(),
        "total_length_m": round(state.planner_total_length_m, 2),
    }


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

@app.get("/api/risk")
async def get_risk():
    now = _now()
    gps_age = (now - state.last_gps_time) if state.last_gps_time is not None else 9999.0
    speed = SURVEY_SPEED_MS
    gps_buf = state.buffers.get("gps")
    if gps_buf and gps_buf[-1].get("speed_kn"):
        speed = gps_buf[-1]["speed_kn"] * 0.514444

    segments = []
    for i in range(len(state.waypoints) - 1):
        wp1, wp2 = state.waypoints[i], state.waypoints[i + 1]
        d = _haversine(wp1["lat"], wp1["lon"], wp2["lat"], wp2["lon"])
        travel_time = d / max(speed, 0.1)
        heading_error = state.drift_rate_dps * travel_time
        mid_lat = (wp1["lat"] + wp2["lat"]) / 2
        mid_lon = (wp1["lon"] + wp2["lon"]) / 2
        depth = get_depth(mid_lat, mid_lon) or 50.0
        risk_drift = min(heading_error / 30.0, 1.0)
        risk_gps = 1.0 - math.exp(-gps_age / 120.0)
        risk_distance = min(d / 2000.0, 1.0)
        risk_depth = math.exp(-depth / 15.0)
        risk_score = 0.35 * risk_drift + 0.30 * risk_gps + 0.15 * risk_distance + 0.20 * risk_depth
        risk_score = max(0.0, min(1.0, risk_score))
        segments.append({
            "from_id": wp1["id"],
            "to_id": wp2["id"],
            "from_name": wp1.get("name"),
            "to_name": wp2.get("name"),
            "distance_m": round(d, 1),
            "risk_score": round(risk_score, 3),
            "factors": {
                "drift": round(risk_drift, 3),
                "gps": round(risk_gps, 3),
                "distance": round(risk_distance, 3),
                "depth": round(risk_depth, 3),
            },
            "heading_error_deg": round(heading_error, 2),
            "avg_depth_m": round(depth, 1),
        })

    mission_risk = max((s["risk_score"] for s in segments), default=0.0)
    return {
        "segments": segments,
        "mission_risk": mission_risk,
        "imu": {
            "drift_rate_dps": round(state.drift_rate_dps, 4),
            "gyro_bias_dps": round(state.gyro_z_bias * (180.0 / math.pi), 4),
            "accel_variance": round(state.accel_variance, 4),
            "last_gps_sec": round(gps_age, 1),
        },
    }


@app.get("/api/recording/start")
async def recording_start():
    state.recording = True; state.recorded_frames = []
    state.recording_start_time = state.last_record_time = _now()
    return {"status": "recording"}


@app.get("/api/recording/stop")
async def recording_stop():
    state.recording = False
    n = len(state.recorded_frames)
    dur = _now() - state.recording_start_time
    return {"frames": n, "duration_sec": round(dur, 1)}


@app.get("/api/recording/status")
async def recording_status():
    elapsed = _now() - state.recording_start_time if state.recording else 0.0
    return {"recording": state.recording, "frames": len(state.recorded_frames), "elapsed_sec": round(elapsed, 1)}


@app.get("/api/recording/playback")
async def recording_playback(start: int = 0, end: int | None = None):
    if end is None:
        end = len(state.recorded_frames)
    return state.recorded_frames[start:end]


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
    if state.replay_task is not None:
        state.replay.stop()
        state.replay_task.cancel()
        try:
            await state.replay_task
        except asyncio.CancelledError:
            pass
        state.replay_task = None
    state.replay_speed = max(speed, 0.01)
    state.replay = SimrisReplay()
    state.replay.running = True
    state.replay.speed = state.replay_speed
    async def _run():
        await _replay_runner(state.replay, speed=state.replay_speed)
    state.replay_task = asyncio.create_task(_run())
    return {"status": "running"}


@app.post("/api/replay/stop")
async def replay_stop():
    """Stop an active replay."""
    state.replay.stop()
    if state.replay_task is not None:
        state.replay_task.cancel()
        try:
            await state.replay_task
        except asyncio.CancelledError:
            pass
        state.replay_task = None
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
    return {"status": "ok", "speed": state.replay_speed}

# ---------------------------------------------------------------------------
# GPS Denial & Dead Reckoning
# ---------------------------------------------------------------------------

@app.get("/api/gps-deny/on")
async def gps_deny_on():
    state.gps_denied = True
    state.gps_deny_start_time = _now()
    gps_buf = state.buffers.get("gps")
    if gps_buf:
        last = gps_buf[-1]
        state.dr_lat = last.get("lat")
        state.dr_lon = last.get("lon")
        state.dr_last_gps_speed_kn = last.get("speed_kn", 0.0)
    state.dr_heading = state.mag_heading
    return {"status": "gps_denied"}

@app.get("/api/gps-deny/off")
async def gps_deny_off():
    state.gps_denied = False
    state.gps_deny_start_time = None
    return {"status": "gps_restored"}

@app.get("/api/gps-deny/status")
async def gps_deny_status():
    secs = (_now() - state.gps_deny_start_time) if state.gps_denied else 0.0
    return {"denied": state.gps_denied, "seconds_without_gps": round(secs, 1)}

@app.get("/api/dead-reckon")
async def get_dead_reckon():
    now = _now()
    gps_buf = state.buffers.get("gps")
    last_gps = gps_buf[-1] if gps_buf else None
    secs = (now - state.gps_deny_start_time) if state.gps_denied else 0.0
    speed_kn = state.dr_last_gps_speed_kn or (last_gps.get("speed_kn", 0.0) if last_gps else 0.0)
    uncertainty_m = speed_kn * 0.514 * secs * state.drift_rate_dps * 0.1
    heading = state.dr_heading
    hr = math.radians(heading)
    c, s = math.cos(hr), math.sin(hr)
    lm, ln = (uncertainty_m * 1.5) ** 2, (uncertainty_m * 0.3) ** 2
    cov_ee = lm * c * c + ln * s * s
    cov_nn = lm * s * s + ln * c * c
    cov_en = (lm - ln) * s * c
    return {
        "dr_lat": state.dr_lat,
        "dr_lon": state.dr_lon,
        "gps_lat": last_gps.get("lat") if last_gps else None,
        "gps_lon": last_gps.get("lon") if last_gps else None,
        "heading_source": "magnetometer",
        "uncertainty_m": round(uncertainty_m, 1),
        "uncertainty_ellipse": {
            "semi_major": round(uncertainty_m * 1.5, 1),
            "semi_minor": round(uncertainty_m * 0.3, 1),
            "angle_deg": round(heading, 1),
        },
        "seconds_since_gps": round(secs, 1),
        "position_source": "dead_reckon" if state.gps_denied else "gps",
        "mean": [state.dr_lat, state.dr_lon],
        "cov": [[round(cov_nn, 4), round(cov_en, 4)], [round(cov_en, 4), round(cov_ee, 4)]],
    }


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

        # ---- Barometer broadcast (separate from IMU, has its own type) ----
        _buffer_sensor("barometer", {"pressure_hpa": round(atmospheric_pressure, 1)})
        await _broadcast({
            "type": "barometer",
            "pressure_hpa": round(atmospheric_pressure, 1),
            "timestamp": _last_ts("barometer"),
        })

        # ---- GPS: embedded planner route, or fallback slow survey pattern ----
        if state.planner_enabled:
            if not state.planner_paused and not state.planner_done:
                state.planner_progress_m = min(
                    state.planner_total_length_m,
                    state.planner_progress_m + step_m,
                )
                if state.planner_progress_m >= state.planner_total_length_m:
                    state.planner_done = True
            x_m, y_m, heading = _planner_xy_at_progress(state.planner_progress_m)
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
            _buffer_sensor("gps", {"lat": round(lat, 6), "lon": round(lon, 6), "speed_kn": round(SURVEY_SPEED_MS * 1.94384, 1), "heading_deg": round(heading, 1)})
            await _broadcast({
                "type": "gps",
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "speed_kn": round(SURVEY_SPEED_MS * 1.94384, 1),
                "heading_deg": round(heading, 1),
                "timestamp": _last_ts("gps"),
            })
            state.track_buffer.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "timestamp": _last_ts("gps"),
                "speed_kn": round(SURVEY_SPEED_MS * 1.94384, 1),
                "heading_deg": round(heading, 1),
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
            gps_age = _now() - state.gps_deny_start_time if state.gps_deny_start_time else 0.0
            spd = state.dr_last_gps_speed_kn or SURVEY_SPEED_MS * 1.94384
            unc = spd * 0.514 * gps_age * state.drift_rate_dps * 0.1
            hdg, hr = state.dr_heading, math.radians(state.dr_heading)
            lm, ln = (unc * 1.5) ** 2, (unc * 0.3) ** 2
            c, s = math.cos(hr), math.sin(hr)
            cv = [[round(lm * s * s + ln * c * c, 4), round((lm - ln) * s * c, 4)],
                  [round((lm - ln) * s * c, 4), round(lm * c * c + ln * s * s, 4)]]
            await _broadcast({
                "type": "dead_reckon", "lat": state.dr_lat, "lon": state.dr_lon,
                "gps_age_sec": round(gps_age, 1),
                "fix_type": "dead_reckon",
                "mean": [state.dr_lat, state.dr_lon],
                "cov": cv,
            })

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

        await _broadcast({
            "type": "gps",
            "vessel": "support",
            "lat": round(v2_lat, 6),
            "lon": round(v2_lon, 6),
            "speed_kn": round(V2_SPEED_MS * 1.94384, 1),
            "heading_deg": round(v2_hdg, 1),
            "timestamp": _last_ts("gps"),
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
        if state.recording and _now() - state.last_record_time >= 1.0:
            state.last_record_time = _now()
            frame = {
                "timestamp": _now(),
                "vessel1": {
                    "lat": round(lat, 6), "lon": round(lon, 6),
                    "heading": round(heading, 1), "speed": round(SURVEY_SPEED_MS, 2),
                    "depth": round(depth, 2),
                },
                "vessel2": {
                    "lat": round(v2_lat, 6), "lon": round(v2_lon, 6),
                    "heading": round(v2_hdg, 1), "speed": round(V2_SPEED_MS, 2),
                    "depth": None,
                },
            }
            state.recorded_frames.append(frame)

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
