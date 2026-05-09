"""
Maritime Mission Planner - Demo 2 Backend Server

FastAPI server with REST endpoints for IMU, GPS, and pressure data,
WebSocket broadcast, data buffering, built-in sensor simulator,
track history, state endpoint, and static file serving.
"""

import asyncio
import math
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUFFER_SIZE = 300
SIMULATOR_RATE = 0.05  # seconds per tick (20 Hz)

SIMULATOR_DISABLED = (
    os.environ.get("DISABLE_SIMULATOR", "0").lower() in {"1", "true", "yes"}
)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

SENSOR_NAMES = {"accelerometer", "magnetometer", "gyroscope", "orientation", "barometer"}

# Simulated survey pattern (Oslofjord area)
BASE_LAT = 59.4370
BASE_LON = 10.6550
SURVEY_SPEED_MS = 2.0          # m/s
SURVEY_LEG_LENGTH = 200.0      # metres per long leg
SURVEY_LEG_SPACING = 20.0      # metres between passes
SURVEY_HEADINGS = (90.0, 0.0, 270.0, 180.0)  # east, north, west, south


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


# ---------------------------------------------------------------------------
# Lifespan (start / stop simulator)
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
    for ws in state.websockets:
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
    """Return the latest vessel state from GPS and pressure buffers."""
    result: dict[str, Any] = {
        "lat": None,
        "lon": None,
        "speed_kn": None,
        "heading_deg": None,
        "depth_m": None,
        "timestamp": None,
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


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.websockets.add(ws)

    # Send buffered history to the newly-connected client.
    history: dict[str, list[dict[str, Any]]] = {}
    for key, buf in state.buffers.items():
        if buf:
            history[key] = list(buf)
    await ws.send_json({"type": "history", "data": history})

    try:
        while True:
            await ws.receive_text()  # keep connection alive
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

    while True:
        await asyncio.sleep(SIMULATOR_RATE)
        sim_time += SIMULATOR_RATE
        t = sim_time

        heading = SURVEY_HEADINGS[phase]
        step_m = SURVEY_SPEED_MS * SIMULATOR_RATE

        # ---- IMU sensors ----
        ax = 0.3 * math.sin(0.5 * t) + 0.2 * _smooth_noise(t, 1.0)
        ay = 0.2 * math.cos(0.3 * t) + 0.2 * _smooth_noise(t, 2.0)
        az = 9.81 + 0.1 * math.sin(0.7 * t) + 0.1 * _smooth_noise(t, 3.0)

        roll_rate = 0.02 * math.sin(0.8 * t) + 0.01 * _smooth_noise(t, 4.0)
        pitch_rate = 0.015 * math.cos(0.6 * t) + 0.01 * _smooth_noise(t, 5.0)
        yaw_rate = 0.01 * math.sin(0.4 * t) + 0.005 * _smooth_noise(t, 6.0)

        roll = 5.0 * math.sin(0.1 * t)
        pitch = 3.0 * math.cos(0.08 * t)
        yaw = (heading + 10.0 * math.sin(0.05 * t)) % 360.0

        # ---- Magnetometer (compass) ----
        mag_x = 25.0 * math.cos(math.radians(yaw)) + 2.0 * _smooth_noise(t, 7.0)
        mag_y = 25.0 * math.sin(math.radians(yaw)) + 2.0 * _smooth_noise(t, 8.0)
        mag_z = 45.0 + 3.0 * math.sin(0.05 * t)
        mag_heading = yaw  # compass heading = yaw

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

        for sensor_name, values in imu_samples:
            _buffer_sensor(sensor_name, values)
            await _broadcast({
                "type": "imu",
                "sensor": sensor_name,
                **values,
                "timestamp": _last_ts(sensor_name),
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

        # ---- Pressure / depth ----
        depth = 5.0 + 3.0 * math.sin(0.03 * t)
        pressure_bar = round(depth * 0.0981 + 1.0, 3)

        _buffer_sensor("pressure", {"depth_m": round(depth, 2), "pressure_bar": pressure_bar})
        await _broadcast({
            "type": "pressure",
            "depth_m": round(depth, 2),
            "pressure_bar": pressure_bar,
            "timestamp": _last_ts("pressure"),
        })


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
