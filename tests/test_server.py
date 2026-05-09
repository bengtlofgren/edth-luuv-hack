"""Tests for the maritime mission planner backend (FastAPI).

Run with:
    /home/tyhug/hackathon/venv/bin/python -m pytest tests/ -v
"""

import asyncio
import importlib
import queue
import threading
from contextlib import ExitStack

import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient as StarletteTestClient

import src.server
from src.server import app


# ---------------------------------------------------------------------------
# Test helpers for Demo 2
# ---------------------------------------------------------------------------


def _fresh_app(monkeypatch=None):
    """Reload src.server with DISABLE_SIMULATOR=1 and return a clean app.

    Pass a ``monkeypatch`` fixture when calling from a test so the env var is
    restored automatically.
    """
    if monkeypatch is not None:
        monkeypatch.setenv("DISABLE_SIMULATOR", "1")
    else:
        import os
        os.environ["DISABLE_SIMULATOR"] = "1"
    importlib.reload(src.server)
    return src.server.app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ws_receive(ws, timeout: float = 5.0):
    """Receive one JSON message from a sync WebSocket with a timeout."""
    result_queue: queue.Queue = queue.Queue()

    def _receive():
        try:
            result_queue.put(("ok", ws.receive_json()))
        except Exception as exc:
            result_queue.put(("error", exc))

    t = threading.Thread(target=_receive, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(f"WebSocket did not produce a message within {timeout}s")

    status, value = result_queue.get_nowait()
    if status == "error":
        raise value
    return value


def _count_samples(history_msg, sensor_name: str) -> int:
    """Count buffered samples for a sensor from the history message."""
    data = history_msg.get("data", {})
    sensor_data = data.get(sensor_name)
    if isinstance(sensor_data, list):
        return len(sensor_data)
    return 0


# ===================================================================
# 1-6.  POST /data  --  IMU sensor ingestion
# ===================================================================


class TestPostData:
    @pytest.mark.asyncio
    async def test_valid_imu_payload(self, client):
        body = {
            "payload": [
                {"name": "accelerometer", "values": {"x": 1.0, "y": 2.0, "z": 3.0}},
                {"name": "gyroscope", "values": {"x": 0.1, "y": -0.2, "z": 0.3}},
            ]
        }
        resp = await client.post("/data", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_payload(self, client):
        resp = await client.post("/data", json={"payload": []})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_json(self, client):
        resp = await client.post(
            "/data", content="<not>json</at-all>",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_missing_payload_key(self, client):
        resp = await client.post("/data", json={})
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_unknown_sensor_name(self, client):
        resp = await client.post(
            "/data",
            json={"payload": [{"name": "unknown_sensor", "values": {"x": 9.9, "y": 8.8, "z": 7.7}}]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_magnetometer_replaces_gravity(self, client):
        """Magnetometer should be accepted (gravity was removed)."""
        resp = await client.post(
            "/data",
            json={"payload": [{"name": "magnetometer", "values": {"x": 25.0, "y": 0.0, "z": 45.0, "heading_deg": 90.0}}]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_barometer_accepted(self, client):
        """Barometer is a valid sensor in SENSOR_NAMES."""
        resp = await client.post(
            "/data",
            json={"payload": [{"name": "barometer", "values": {"pressure_hpa": 1013.25}}]},
        )
        assert resp.status_code == 200


# ===================================================================
# 7-9.  POST /gps  --  GPS position data
# ===================================================================


class TestPostGps:
    @pytest.mark.asyncio
    async def test_valid_gps(self, client):
        resp = await client.post(
            "/gps", json={"lat": 59.4, "lon": 10.7, "speed_kn": 5.2, "heading_deg": 45.0},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client):
        resp = await client.post("/gps", json={})
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_invalid_lat(self, client):
        resp = await client.post(
            "/gps", json={"lat": 1000.0, "lon": 10.7},
        )
        assert resp.status_code in (400, 422)


# ===================================================================
# 10-11.  POST /pressure  --  Depth / pressure data
# ===================================================================


class TestPostPressure:
    @pytest.mark.asyncio
    async def test_valid_pressure(self, client):
        resp = await client.post("/pressure", json={"depth_m": 12.5, "pressure_bar": 2.3})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_negative_depth_ok(self, client):
        resp = await client.post("/pressure", json={"depth_m": -5.0, "pressure_bar": 1.0})
        assert resp.status_code == 200


# ===================================================================
# 12-18.  WebSocket /ws
# ===================================================================


class TestWebSocket:
    def test_connect(self, ws_client):
        assert ws_client is not None

    def test_history_on_connect(self, ws_client):
        msg = ws_receive(ws_client)
        assert msg["type"] == "history"
        assert "data" in msg

    @pytest.mark.asyncio
    async def test_live_imu_broadcast(self, client, ws_client):
        ws_receive(ws_client)
        await client.post(
            "/data",
            json={"payload": [{"name": "accelerometer", "values": {"x": 4.5, "y": 6.7, "z": 8.9}}]},
        )
        msg = ws_receive(ws_client)
        assert msg["type"] == "imu"
        assert msg["sensor"] == "accelerometer"
        assert msg["x"] == 4.5

    @pytest.mark.asyncio
    async def test_live_gps_broadcast(self, client, ws_client):
        ws_receive(ws_client)
        await client.post("/gps", json={"lat": 58.0, "lon": 9.5, "speed_kn": 6.0, "heading_deg": 180.0})
        msg = ws_receive(ws_client)
        assert msg["type"] == "gps"
        assert msg["lat"] == 58.0

    @pytest.mark.asyncio
    async def test_live_pressure_broadcast(self, client, ws_client):
        ws_receive(ws_client)
        await client.post("/pressure", json={"depth_m": 30.0, "pressure_bar": 4.0})
        msg = ws_receive(ws_client)
        assert msg["type"] == "pressure"
        assert msg["depth_m"] == 30.0

    @pytest.mark.asyncio
    async def test_live_magnetometer_broadcast(self, client, ws_client):
        ws_receive(ws_client)
        await client.post(
            "/data",
            json={"payload": [{"name": "magnetometer", "values": {"x": 25.0, "y": 0.0, "z": 45.0, "heading_deg": 90.0}}]},
        )
        msg = ws_receive(ws_client)
        assert msg["type"] == "imu"
        assert msg["sensor"] == "magnetometer"
        assert msg["heading_deg"] == 90.0

    @pytest.mark.asyncio
    async def test_live_barometer_ws_broadcast(self, client, ws_client):
        ws_receive(ws_client)
        await client.post(
            "/data",
            json={"payload": [{"name": "barometer", "values": {"pressure_hpa": 1013.25}}]},
        )
        msg = ws_receive(ws_client)
        assert msg["type"] == "imu"
        assert msg["sensor"] == "barometer"
        assert msg["pressure_hpa"] == 1013.25

    @pytest.mark.asyncio
    async def test_multiple_clients(self, client):
        with ExitStack() as stack:
            ws1 = stack.enter_context(StarletteTestClient(app).websocket_connect("/ws"))
            ws2 = stack.enter_context(StarletteTestClient(app).websocket_connect("/ws"))
            ws3 = stack.enter_context(StarletteTestClient(app).websocket_connect("/ws"))

            for ws in (ws1, ws2, ws3):
                ws_receive(ws)

            await client.post(
                "/data",
                json={"payload": [{"name": "accelerometer", "values": {"x": 7.7, "y": 0.0, "z": 0.0}}]},
            )

            m1 = ws_receive(ws1)
            m2 = ws_receive(ws2)
            m3 = ws_receive(ws3)
            assert m1["x"] == m2["x"] == m3["x"] == 7.7


# ===================================================================
# 19.  Data buffer limits
# ===================================================================


class TestDataBuffers:
    @pytest.mark.asyncio
    async def test_buffer_capped_at_300(self, client):
        for i in range(400):
            await client.post(
                "/data",
                json={"payload": [{"name": "accelerometer", "values": {"x": float(i), "y": 0.0, "z": 0.0}}]},
            )

        with StarletteTestClient(app).websocket_connect("/ws") as ws:
            history = ws_receive(ws)
            assert history["type"] == "history"
            samples = _count_samples(history, "accelerometer")
            assert 0 < samples <= 300, f"Expected at most 300 buffered samples, got {samples}"


# ===================================================================
# 20-21.  Sensor simulator
# ===================================================================


class TestSimulator:
    @pytest.mark.asyncio
    async def test_simulator_enabled_by_default(self):
        """Simulator running — magnetometer and barometer in history."""
        transport = ASGITransport(app=src.server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for _ in range(5):
                await ac.post("/data", json={"payload": []})
                await asyncio.sleep(0.1)

        with StarletteTestClient(src.server.app).websocket_connect("/ws") as ws:
            history = ws_receive(ws)
            assert history["type"] == "history"
            data = history.get("data", {})
            assert "magnetometer" in data, "Expected magnetometer in simulated data"
            assert "barometer" in data, "Expected barometer in simulated data"

    def test_simulator_disabled_via_env(self, monkeypatch):
        """With DISABLE_SIMULATOR=1, no simulated data after history."""
        monkeypatch.setenv("DISABLE_SIMULATOR", "1")
        importlib.reload(src.server)
        sim_app = src.server.app

        with StarletteTestClient(sim_app).websocket_connect("/ws") as ws:
            history = ws_receive(ws)
            assert history["type"] == "history"
            with pytest.raises(TimeoutError):
                ws_receive(ws, timeout=3.0)


# ===================================================================
# 22-24.  Infrastructure: static files, CORS, concurrency
# ===================================================================


class TestServerInfrastructure:
    @pytest.mark.asyncio
    async def test_static_file_serving(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_cors_headers(self, client):
        resp = await client.post(
            "/data", json={"payload": []},
            headers={"Origin": "http://example.com"},
        )
        # With credentials=True, Starlette echoes the origin back per CORS spec
        assert resp.headers.get("access-control-allow-origin") is not None

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        async def _post():
            return await client.post(
                "/data",
                json={"payload": [{"name": "accelerometer", "values": {"x": 1.0, "y": 2.0, "z": 3.0}}]},
            )

        results = await asyncio.gather(*[_post() for _ in range(10)])
        for i, r in enumerate(results):
            assert r.status_code == 200, f"Request {i} failed with {r.status_code}"


# ===================================================================
# 25-27.  GET /api/state  --  Latest vessel state
# ===================================================================


class TestApiState:
    @pytest.mark.asyncio
    async def test_state_empty_by_default(self, monkeypatch):
        """GET /api/state with no data -- returns 200 with null fields."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/state")
            assert resp.status_code == 200
            data = resp.json()
            assert data["lat"] is None
            assert data["lon"] is None
            assert data["speed_kn"] is None
            assert data["heading_deg"] is None
            assert data["depth_m"] is None
            assert data["timestamp"] is None

    @pytest.mark.asyncio
    async def test_state_after_gps(self, client):
        """POST GPS data, then GET /api/state -- returns the posted lat/lon."""
        await client.post(
            "/gps",
            json={
                "lat": 59.4,
                "lon": 10.7,
                "speed_kn": 5.2,
                "heading_deg": 45.0,
            },
        )
        resp = await client.get("/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["lat"] == 59.4
        assert data["lon"] == 10.7
        assert data["speed_kn"] == 5.2
        assert data["heading_deg"] == 45.0

    @pytest.mark.asyncio
    async def test_state_after_pressure(self, client):
        """POST pressure, then GET /api/state -- depth_m is present."""
        await client.post("/pressure", json={"depth_m": 12.5, "pressure_bar": 2.3})
        resp = await client.get("/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["depth_m"] == 12.5


# ===================================================================
# 28-31.  GET /api/track  --  Track history as GeoJSON
# ===================================================================


class TestApiTrack:
    @pytest.mark.asyncio
    async def test_track_empty_by_default(self, monkeypatch):
        """GET /api/track with no data -- valid GeoJSON with empty coords."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/track")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "Feature"
            assert data["geometry"]["type"] == "LineString"
            assert data["geometry"]["coordinates"] == []
            assert data["properties"]["count"] == 0

    @pytest.mark.asyncio
    async def test_track_after_gps(self, monkeypatch):
        """POST 3 GPS points, GET /api/track -- 3 coordinates in LineString."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for lat, lon in [(59.4, 10.7), (59.5, 10.8), (59.6, 10.9)]:
                await ac.post("/gps", json={"lat": lat, "lon": lon})

            resp = await ac.get("/api/track")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["geometry"]["coordinates"]) == 3
            assert data["properties"]["count"] == 3

    @pytest.mark.asyncio
    async def test_track_geojson_format(self, client):
        """Verify returned track is valid GeoJSON Feature / LineString."""
        await client.post(
            "/gps",
            json={"lat": 59.4, "lon": 10.7, "speed_kn": 5.2, "heading_deg": 45.0},
        )
        resp = await client.get("/api/track")
        assert resp.status_code == 200
        data = resp.json()

        # Top-level Feature
        assert data["type"] == "Feature"

        # Geometry is a LineString
        geom = data["geometry"]
        assert geom["type"] == "LineString"
        coords = geom["coordinates"]
        assert isinstance(coords, list)
        assert len(coords) >= 1

        # Each coordinate is a [lon, lat] pair
        for pt in coords:
            assert isinstance(pt, list) and len(pt) == 2
            assert isinstance(pt[0], (int, float))
            assert isinstance(pt[1], (int, float))

        # Properties
        props = data["properties"]
        assert "count" in props
        assert isinstance(props["count"], int)
        assert props["count"] >= 1

    @pytest.mark.asyncio
    async def test_track_capped_at_2000(self, client):
        """Post 2100 GPS points -- track returns at most 2000."""
        for i in range(2100):
            await client.post(
                "/gps",
                json={
                    "lat": 59.0 + i * 0.0001,
                    "lon": 10.0 + i * 0.0001,
                },
            )

        resp = await client.get("/api/track")
        assert resp.status_code == 200
        data = resp.json()
        count = data["properties"]["count"]
        assert count <= 2000, f"Expected <= 2000, got {count}"
        assert len(data["geometry"]["coordinates"]) == count
