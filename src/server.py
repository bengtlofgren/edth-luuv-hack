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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUFFER_SIZE = 300
SIMULATOR_RATE = 0.05  # seconds per tick (20 Hz)
SIMULATOR_DISABLED = os.environ.get("DISABLE_SIMULATOR") == "1"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

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
        self.mission_logs: dict[str, dict[str, Any]] = {}
        self._replay_active: bool = False
        self._replay_task: asyncio.Task[None] | None = None

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
        self.dr_last_gps_speed_kn: float = 0.0

        # Magnetometer calibration
        self.mag_hard_iron: list[float] = [0.0, 0.0, 0.0]
        self.mag_soft_iron: list[list[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.mag_heading: float = 0.0

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
    # Pre-load simris field data as a selectable mission
    try:
        simris = SimrisReplay()
        frames = []
        rel0 = simris.timeline[0][0] if simris.timeline else 0
        end_rel = simris.timeline[-1][0] if simris.timeline else 0
        gps_idx, imu_idx = 0, 0
        t = 0.0
        while t <= end_rel:
            while gps_idx < len(simris.gps_events) and simris.gps_events[gps_idx]["rel_time"] <= t:
                gps_idx += 1
            while imu_idx < len(simris.imu_events) and simris.imu_events[imu_idx]["rel_time"] <= t:
                imu_idx += 1
            gps = simris.gps_events[gps_idx - 1] if gps_idx > 0 else None
            imu = simris.imu_events[imu_idx - 1] if imu_idx > 0 else None
            mag_hdg = None
            if imu:
                mx, my = imu.get("mag_x", 0), imu.get("mag_y", 0)
                if mx or my:
                    mag_hdg = round(math.degrees(math.atan2(my, mx)) % 360.0, 1)
            base_ts = simris.gps_events[0]["timestamp"] if simris.gps_events else 0.0
            frames.append({
                "timestamp": base_ts + t,
                "rel_time": round(t, 1),
                "lat": gps["lat"] if gps else None,
                "lon": gps["lon"] if gps else None,
                "heading_deg": 0.0, "speed_kn": 0.0,
                "depth_m": None, "mag_heading": mag_hdg, "gps_denied": False,
            })
            t += 1.0
        state.mission_logs["simris-field-test"] = {"frames": frames,
            "duration_sec": round(frames[-1]["rel_time"], 1) if frames else 0, "source": "simris"}
    except Exception:
        pass
    yield
    if state.simulator_task is not None:
        state.simulator_task.cancel()
        try:
            await state.simulator_task
        except asyncio.CancelledError:
            pass
    _cancel_replay()

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
    _maybe_record_frame()
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
            "type": "gps", "lat": data.lat, "lon": data.lon,
            "speed_kn": data.speed_kn, "heading_deg": data.heading_deg,
            "timestamp": _last_ts("gps"),
        })
        state.track_buffer.append({
            "lat": data.lat, "lon": data.lon, "timestamp": _now(),
            "speed_kn": data.speed_kn, "heading_deg": data.heading_deg,
        })
        state.last_gps_time = _now()
    _maybe_record_frame()
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
    _maybe_record_frame()
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


def _maybe_record_frame() -> None:
    """Capture a mission frame every second while recording."""
    if not state.recording:
        return
    now = _now()
    if now - state.last_record_time < 1.0:
        return
    state.last_record_time = now
    gps_buf = state.buffers.get("gps")
    pressure_buf = state.buffers.get("pressure")
    last_gps = gps_buf[-1] if gps_buf else {}
    last_depth = pressure_buf[-1] if pressure_buf else {}
    state.recorded_frames.append({
        "timestamp": now,
        "rel_time": now - state.recording_start_time,
        "lat": last_gps.get("lat"), "lon": last_gps.get("lon"),
        "heading_deg": last_gps.get("heading_deg", 0.0),
        "speed_kn": last_gps.get("speed_kn", 0.0),
        "depth_m": last_depth.get("depth_m"),
        "mag_heading": state.mag_heading,
        "gps_denied": state.gps_denied,
    })

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
    mid = time.strftime("%Y%m%d-%H%M%S", time.localtime(state.recording_start_time))
    state.mission_logs[mid] = {"frames": list(state.recorded_frames), "duration_sec": round(dur, 1), "source": "recording", "timestamp": state.recording_start_time}
    return {"frames": n, "duration_sec": round(dur, 1), "mission_id": mid}


@app.get("/api/recording/status")
async def recording_status():
    elapsed = _now() - state.recording_start_time if state.recording else 0.0
    return {"recording": state.recording, "frames": len(state.recorded_frames), "elapsed_sec": round(elapsed, 1)}


@app.get("/api/recording/playback")
async def recording_playback(start: int = 0, end: int | None = None):
    if end is None:
        end = len(state.recorded_frames)
    return state.recorded_frames[start:end]


@app.get("/api/missions")
async def list_missions():
    """List all saved mission logs."""
    result = []
    for mid, mdata in state.mission_logs.items():
        result.append({
            "id": mid, "frames": len(mdata["frames"]),
            "duration_sec": mdata["duration_sec"], "source": mdata["source"],
        })
    return sorted(result, key=lambda m: m["id"], reverse=True)


@app.get("/api/missions/{mission_id}/load")
async def load_mission(mission_id: str):
    """Load a saved mission into the replay buffer."""
    if mission_id not in state.mission_logs:
        raise HTTPException(status_code=404, detail="Mission not found")
    state.recorded_frames = list(state.mission_logs[mission_id]["frames"])
    return {"mission_id": mission_id, "frames": len(state.recorded_frames), "loaded": True}


@app.post("/api/replay/start")
async def start_replay(req: dict[str, Any] | None = None):
    """Start replay. Body: {"mission_id": "...", "speed": 5.0} or speed query param."""
    _cancel_replay()
    speed = float((req or {}).get("speed", 1.0))
    mission_id = (req or {}).get("mission_id")

    if mission_id:
        if mission_id not in state.mission_logs:
            raise HTTPException(status_code=404, detail="Mission not found")
        state.recorded_frames = list(state.mission_logs[mission_id]["frames"])
        if not state.recorded_frames:
            raise HTTPException(status_code=400, detail="Mission has no frames")
        state._replay_active = True
        state._replay_task = asyncio.create_task(_replay_frames(speed))
        return {"status": "replaying", "mission_id": mission_id, "speed": speed, "total_frames": len(state.recorded_frames)}
    elif state.recorded_frames:
        state._replay_active = True
        state._replay_task = asyncio.create_task(_replay_frames(speed))
        return {"status": "replaying", "speed": speed, "total_frames": len(state.recorded_frames)}
    else:
        state.replay = SimrisReplay()
        state.replay.running = True
        async def _run():
            await _replay_runner(state.replay, speed=speed)
        state.replay_task = asyncio.create_task(_run())
        return {"status": "running", "source": "simris"}


def _cancel_replay():
    """Cancel any active replay task."""
    if state._replay_active:
        state._replay_active = False
    if state._replay_task is not None:
        state._replay_task.cancel()
        try:
            pass
        except asyncio.CancelledError:
            pass
        state._replay_task = None
    if state.replay_task is not None:
        state.replay.stop()
        state.replay_task.cancel()
        try:
            pass
        except asyncio.CancelledError:
            pass
        state.replay_task = None


@app.post("/api/replay/stop")
async def replay_stop():
    """Stop any active replay (Simris or mission)."""
    _cancel_replay()
    return {"status": "stopped"}


@app.get("/api/replay/status")
async def replay_status_endpoint():
    """Get replay progress."""
    simris_status = state.replay.get_status() if state.replay is not None else {}
    return {
        "running": state._replay_active or simris_status.get("running", False),
        "simris_running": simris_status.get("running", False),
        "mission_running": state._replay_active,
        "progress": simris_status.get("progress", 0),
        "total_frames": len(state.recorded_frames),
    }


@app.get("/api/missions/{mission_id}/replay")
async def replay_mission(mission_id: str, speed: float = 5.0):
    """Load and start replaying a stored mission."""
    if mission_id not in state.mission_logs:
        raise HTTPException(status_code=404, detail="Mission not found")
    _cancel_replay()
    state.recorded_frames = list(state.mission_logs[mission_id]["frames"])
    if not state.recorded_frames:
        raise HTTPException(status_code=400, detail="Mission has no frames")
    state._replay_active = True
    state._replay_task = asyncio.create_task(_replay_frames(speed))
    return {"status": "replaying", "mission_id": mission_id, "speed": speed, "total_frames": len(state.recorded_frames)}


async def _replay_frames(speed: float):
    """Background task that replays recorded frames through the pipeline."""
    frames = state.recorded_frames
    if not frames:
        state._replay_active = False; return
    t0 = time.monotonic()
    start_rel = frames[0].get("rel_time", 0)
    for i, frame in enumerate(frames):
        if not state._replay_active:
            break
        target = (frame.get("rel_time", 0) - start_rel) / speed
        elapsed = time.monotonic() - t0
        if target > elapsed:
            await asyncio.sleep(target - elapsed)
        if frame.get("lat") is not None:
            lat, lon = frame["lat"], frame["lon"]
            _buffer_sensor("gps", {"lat": lat, "lon": lon})
            state.last_gps_time = _now()
            await _broadcast({"type": "gps", "lat": lat, "lon": lon,
                "speed_kn": frame.get("speed_kn", 0), "heading_deg": frame.get("heading_deg", 0),
                "timestamp": _now()})
    state._replay_active = False


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

        heading = SURVEY_HEADINGS[phase]
        step_m = SURVEY_SPEED_MS * SIMULATOR_RATE

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

        # ---- GPS: slow survey (lawnmower) pattern ----
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
        _maybe_record_frame()

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
