"""Tests for the maritime mission planner backend (FastAPI).

Run with:
    /home/tyhug/hackathon/venv/bin/python -m pytest tests/ -v
"""

import asyncio
import importlib
import math
import queue
import threading
from contextlib import ExitStack

import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient as StarletteTestClient

import src.server
from src.replay import SimrisReplay
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
            "/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.2, "heading_deg": 45.0},
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
            for lat, lon in [(56.16, 15.59), (56.18, 15.60), (56.13, 15.58)]:
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
            json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.2, "heading_deg": 45.0},
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


# ===================================================================
# Embedded planner route control
# ===================================================================


class TestEmbeddedPlanner:
    @pytest.mark.asyncio
    async def test_planner_route_lifecycle(self, monkeypatch):
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/planner/path", json={"waypoints": [[0, 0], [10, 0], [10, 10]]})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "planner_route_active"
            assert data["total_length_m"] == 20.0
            assert len(data["waypoints"]) == 3
            assert data["waypoints"][0]["lat"] == pytest.approx(src.server.BASE_LAT)
            assert data["waypoints"][0]["lon"] == pytest.approx(src.server.BASE_LON)
            assert data["waypoints"][1]["lat"] == pytest.approx(src.server.BASE_LAT)
            assert data["waypoints"][1]["lon"] - data["waypoints"][0]["lon"] == pytest.approx(
                10.0 / (111320.0 * math.cos(math.radians(src.server.BASE_LAT)))
            )
            assert data["waypoints"][2]["lat"] - data["waypoints"][1]["lat"] == pytest.approx(
                10.0 / 111320.0
            )
            assert data["waypoints"][2]["lon"] == pytest.approx(data["waypoints"][1]["lon"])

            resp = await ac.get("/api/planner/status")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True
            assert resp.json()["paused"] is False

            resp = await ac.post("/api/planner/pause")
            assert resp.status_code == 200
            assert resp.json()["paused"] is True

            resp = await ac.post("/api/planner/resume")
            assert resp.status_code == 200
            assert resp.json()["paused"] is False

            resp = await ac.post("/api/planner/clear")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is False

    @pytest.mark.asyncio
    async def test_planner_rejects_zero_length_route(self, monkeypatch):
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/planner/path", json={"waypoints": [[1, 1], [1, 1]]})
            assert resp.status_code == 400

    def test_planner_advances_exact_metric_distance(self, monkeypatch):
        _fresh_app(monkeypatch)
        src.server._set_planner_path([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])

        x_m, y_m, heading = src.server._advance_planner_route(2.5)
        assert src.server.state.planner_progress_m == pytest.approx(2.5)
        assert x_m == pytest.approx(2.5)
        assert y_m == pytest.approx(0.0)
        assert heading == pytest.approx(90.0)

        x_m, y_m, heading = src.server._advance_planner_route(10.5)
        assert src.server.state.planner_progress_m == pytest.approx(13.0)
        assert x_m == pytest.approx(10.0)
        assert y_m == pytest.approx(3.0)
        assert heading == pytest.approx(0.0)

        src.server.state.planner_paused = True
        x_m, y_m, heading = src.server._advance_planner_route(4.0)
        assert src.server.state.planner_progress_m == pytest.approx(13.0)
        assert x_m == pytest.approx(10.0)
        assert y_m == pytest.approx(3.0)
        assert heading == pytest.approx(0.0)

        src.server.state.planner_paused = False
        x_m, y_m, heading = src.server._advance_planner_route(100.0)
        assert src.server.state.planner_progress_m == pytest.approx(20.0)
        assert src.server.state.planner_done is True
        assert x_m == pytest.approx(10.0)
        assert y_m == pytest.approx(10.0)
        assert heading == pytest.approx(0.0)

        x_m, y_m, heading = src.server._advance_planner_route(5.0)
        assert src.server.state.planner_progress_m == pytest.approx(20.0)
        assert x_m == pytest.approx(10.0)
        assert y_m == pytest.approx(10.0)
        assert heading == pytest.approx(0.0)


# ===================================================================
# 32-36.  Demo 3 — Waypoints CRUD
# ===================================================================


class TestWaypoints:
    @pytest.mark.asyncio
    async def test_create_waypoint(self, monkeypatch):
        """POST /api/waypoints with valid lat/lon returns id."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/waypoints",
                json={"lat": 56.16, "lon": 15.59, "name": "WP1"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == 1
            assert data["lat"] == 56.16
            assert data["lon"] == 15.59
            assert data["name"] == "WP1"
            assert "depth_m" in data

    @pytest.mark.asyncio
    async def test_create_waypoint_invalid_coords(self, monkeypatch):
        """POST /api/waypoints with out-of-range lat returns 400/422."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/waypoints", json={"lat": 100.0, "lon": 15.6})
            assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_list_waypoints(self, monkeypatch):
        """POST 3 waypoints, GET /api/waypoints returns list of 3."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for lat, lon, name in [
                (56.16, 15.59, "WP1"),
                (56.18, 15.60, "WP2"),
                (56.13, 15.58, "WP3"),
            ]:
                await ac.post("/api/waypoints", json={"lat": lat, "lon": lon, "name": name})
            resp = await ac.get("/api/waypoints")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 3

    @pytest.mark.asyncio
    async def test_delete_waypoint(self, monkeypatch):
        """DELETE /api/waypoints/{id} removes the waypoint."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            create_resp = await ac.post("/api/waypoints", json={"lat": 56.16, "lon": 15.59})
            wp_id = create_resp.json()["id"]

            del_resp = await ac.delete(f"/api/waypoints/{wp_id}")
            assert del_resp.status_code == 200
            assert del_resp.json()["status"] == "deleted"

            list_resp = await ac.get("/api/waypoints")
            assert len(list_resp.json()) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_waypoint(self, monkeypatch):
        """DELETE /api/waypoints/99999 returns 404."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/api/waypoints/99999")
            assert resp.status_code == 404


# ===================================================================
# 37-39.  Demo 3 — IMU Drift Tracking
# ===================================================================


class TestImuDrift:
    @pytest.mark.asyncio
    async def test_drift_tracks_gyro_bias(self, monkeypatch):
        """Post gyro readings with consistent Z bias, verify drift tracking."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for _ in range(50):
                await ac.post(
                    "/data",
                    json={
                        "payload": [
                            {"name": "gyroscope", "values": {"x": 0.0, "y": 0.0, "z": 0.05}},
                        ]
                    },
                )
            resp = await ac.get("/api/risk")
            assert resp.status_code == 200
            imu = resp.json()["imu"]
            assert "gyro_bias_dps" in imu
            assert "drift_rate_dps" in imu
            assert imu["gyro_bias_dps"] > 0.01
            assert imu["drift_rate_dps"] > 0.01

    @pytest.mark.asyncio
    async def test_gps_timestamp_recorded(self, monkeypatch):
        """POST /gps, verify last_gps_time is reflected in /api/risk imu data."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59})
            resp = await ac.get("/api/risk")
            assert resp.status_code == 200
            imu = resp.json()["imu"]
            assert "last_gps_sec" in imu
            assert imu["last_gps_sec"] < 5.0

    @pytest.mark.asyncio
    async def test_drift_with_noise(self, monkeypatch):
        """Drift should be higher with noisy (biased) gyro data than with clean data."""
        # App A: clean (unbiased) gyro data
        app_a = _fresh_app(monkeypatch)
        transport_a = ASGITransport(app=app_a)
        async with AsyncClient(transport=transport_a, base_url="http://test") as ac_a:
            for _ in range(100):
                await ac_a.post(
                    "/data",
                    json={
                        "payload": [
                            {"name": "gyroscope", "values": {"x": 0.0, "y": 0.0, "z": 0.0}},
                        ]
                    },
                )
            resp_a = await ac_a.get("/api/risk")
            clean_drift = resp_a.json()["imu"]["drift_rate_dps"]

        # App B: noisy (biased) gyro data
        app_b = _fresh_app(monkeypatch)
        transport_b = ASGITransport(app=app_b)
        async with AsyncClient(transport=transport_b, base_url="http://test") as ac_b:
            for _ in range(100):
                await ac_b.post(
                    "/data",
                    json={
                        "payload": [
                            {"name": "gyroscope", "values": {"x": 0.02, "y": -0.01, "z": 0.05}},
                        ]
                    },
                )
            resp_b = await ac_b.get("/api/risk")
            noisy_drift = resp_b.json()["imu"]["drift_rate_dps"]

        assert noisy_drift > clean_drift


# ===================================================================
# 40-44.  Demo 3 — Risk Calculation
# ===================================================================


class TestRisk:
    @pytest.mark.asyncio
    async def test_risk_empty_waypoints(self, monkeypatch):
        """GET /api/risk with no waypoints returns 200 and empty segments."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/risk")
            assert resp.status_code == 200
            data = resp.json()
            assert data["segments"] == []
            assert data["mission_risk"] == 0.0

    @pytest.mark.asyncio
    async def test_risk_with_waypoints(self, monkeypatch):
        """POST waypoints, GET /api/risk returns segments for each pair."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/api/waypoints", json={"lat": 56.16, "lon": 15.59, "name": "WP1"})
            await ac.post("/api/waypoints", json={"lat": 56.18, "lon": 15.60, "name": "WP2"})
            await ac.post("/api/waypoints", json={"lat": 56.13, "lon": 15.58, "name": "WP3"})

            resp = await ac.get("/api/risk")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["segments"]) == 2  # 3 waypoints -> 2 segments
            assert "mission_risk" in data
            assert "imu" in data

    @pytest.mark.asyncio
    async def test_risk_increases_with_drift(self, monkeypatch):
        """Adding gyro bias increases mission risk."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/api/waypoints", json={"lat": 56.16, "lon": 15.59})
            await ac.post("/api/waypoints", json={"lat": 56.18, "lon": 15.60})
            # GPS fix so baseline risk isn't maxed by GPS age alone
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0})

            resp = await ac.get("/api/risk")
            risk_baseline = resp.json()["mission_risk"]

            # Inject gyro bias
            for _ in range(50):
                await ac.post(
                    "/data",
                    json={
                        "payload": [
                            {"name": "gyroscope", "values": {"x": 0.0, "y": 0.0, "z": 0.05}},
                        ]
                    },
                )

            resp = await ac.get("/api/risk")
            risk_drift = resp.json()["mission_risk"]

            assert risk_drift >= risk_baseline

    @pytest.mark.asyncio
    async def test_risk_increases_with_gps_age(self, monkeypatch):
        """Risk is higher with no GPS fix than with a fresh GPS fix."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/api/waypoints", json={"lat": 56.16, "lon": 15.59})
            await ac.post("/api/waypoints", json={"lat": 56.18, "lon": 15.60})

            # Risk before any GPS (gps_age defaults to 9999)
            resp = await ac.get("/api/risk")
            risk_no_gps = resp.json()["mission_risk"]

            # Fresh GPS fix
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59})

            resp = await ac.get("/api/risk")
            risk_fresh_gps = resp.json()["mission_risk"]

            assert risk_fresh_gps < risk_no_gps

    @pytest.mark.asyncio
    async def test_risk_segment_format(self, monkeypatch):
        """Each risk segment has the expected fields."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/api/waypoints", json={"lat": 56.16, "lon": 15.59, "name": "WP1"})
            await ac.post("/api/waypoints", json={"lat": 56.18, "lon": 15.60, "name": "WP2"})

            resp = await ac.get("/api/risk")
            data = resp.json()
            seg = data["segments"][0]

            assert "from_id" in seg
            assert "to_id" in seg
            assert "from_name" in seg
            assert "to_name" in seg
            assert "distance_m" in seg
            assert "risk_score" in seg
            assert "factors" in seg
            assert "heading_error_deg" in seg
            assert "avg_depth_m" in seg
            assert 0.0 <= seg["risk_score"] <= 1.0
            for key in ("drift", "gps", "distance", "depth"):
                assert key in seg["factors"]


# ===================================================================
# 45-48.  Demo 5 — Recording
# ===================================================================


class TestRecording:
    @pytest.mark.asyncio
    async def test_recording_start_stop(self, monkeypatch):
        """Start recording, post GPS data, stop -- verify frames and status flips."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start recording
            resp = await ac.get("/api/recording/start")
            assert resp.status_code == 200

            # Status shows recording=true
            resp = await ac.get("/api/recording/status")
            assert resp.status_code == 200
            assert resp.json()["recording"] is True

            # Post some GPS data
            for lat, lon in [(56.16, 15.59), (56.18, 15.60), (56.13, 15.58)]:
                await ac.post("/gps", json={"lat": lat, "lon": lon})

            # Stop recording -- returns frame count (frames) and duration
            resp = await ac.get("/api/recording/stop")
            assert resp.status_code == 200
            data = resp.json()
            assert "frames" in data
            assert data["frames"] >= 0
            assert "duration_sec" in data
            assert data["duration_sec"] >= 0

            # Status now shows recording=false
            resp = await ac.get("/api/recording/status")
            assert resp.status_code == 200
            assert resp.json()["recording"] is False

    @pytest.mark.asyncio
    async def test_recording_status_idle(self, monkeypatch):
        """Status before starting should show recording=false, frames=0, elapsed_sec=0."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/recording/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["recording"] is False
            assert data["frames"] == 0
            assert data["elapsed_sec"] == 0.0

    @pytest.mark.asyncio
    async def test_recording_playback(self, monkeypatch):
        """Start recording, stop, then playback with range params returns a list."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.get("/api/recording/start")

            for lat, lon in [(56.16, 15.59), (56.18, 15.60), (56.20, 15.62)]:
                await ac.post("/gps", json={"lat": lat, "lon": lon})

            await ac.get("/api/recording/stop")

            # Playback returns a raw list (not wrapped in a dict)
            resp = await ac.get("/api/recording/playback", params={"start": 0, "end": 100})
            assert resp.status_code == 200
            frames = resp.json()
            assert isinstance(frames, list)

    @pytest.mark.asyncio
    async def test_recording_playback_empty(self, monkeypatch):
        """Playback with no recording should return empty list."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/recording/playback", params={"start": 0, "end": 100})
            assert resp.status_code == 200
            frames = resp.json()
            assert isinstance(frames, list)
            assert len(frames) == 0


# ===================================================================
# 49-52.  Demo 5 — Export
# ===================================================================


class TestExport:
    @pytest.mark.asyncio
    async def test_export_geojson_format(self, monkeypatch):
        """GET /api/export/geojson returns valid GeoJSON FeatureCollection."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/export/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "FeatureCollection"
            assert "features" in data
            assert isinstance(data["features"], list)

    @pytest.mark.asyncio
    async def test_export_geojson_has_tracks(self, monkeypatch):
        """Send GPS data, verify export has features with correct structure."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for lat, lon in [(56.16, 15.59), (56.18, 15.60)]:
                await ac.post("/gps", json={"lat": lat, "lon": lon})

            resp = await ac.get("/api/export/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["features"]) >= 1
            for feat in data["features"]:
                assert feat["type"] == "Feature"
                assert feat["geometry"]["type"] == "LineString"
                assert "vessel" in feat["properties"]

    @pytest.mark.asyncio
    async def test_export_csv_format(self, monkeypatch):
        """GET /api/export/csv returns text/csv with correct header row."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/export/csv")
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("text/csv")
            header_line = resp.text.split("\n")[0]
            cols = [c.strip() for c in header_line.split(",")]
            for col in ("timestamp", "vessel", "lat", "lon", "heading_deg", "speed_kn", "depth_m"):
                assert col in cols, f"Missing CSV column: {col}"

    @pytest.mark.asyncio
    async def test_export_csv_has_data(self, monkeypatch):
        """Send GPS data, verify CSV has data rows."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for lat, lon in [(56.16, 15.59), (56.18, 15.60)]:
                await ac.post("/gps", json={"lat": lat, "lon": lon})

            resp = await ac.get("/api/export/csv")
            assert resp.status_code == 200
            lines = resp.text.strip().split("\n")
            # Header + at least one data row
            assert len(lines) >= 2


# ===================================================================
# 53-54.  Demo 5 — Multi-Vessel State
# ===================================================================


class TestMultiVessel:
    @pytest.mark.asyncio
    async def test_state_includes_vessel2(self, monkeypatch):
        """GET /api/state, verify response has vessel2 keys."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/state")
            assert resp.status_code == 200
            data = resp.json()
            assert "vessel2_lat" in data
            assert "vessel2_lon" in data
            assert "vessel2_heading_deg" in data
            assert "vessel2_speed_kn" in data

    @pytest.mark.asyncio
    async def test_state_vessel2_fields_populated(self, client):
        """With simulator running, vessel2 fields should have non-null values."""
        for _ in range(5):
            await client.post("/data", json={"payload": []})
            await asyncio.sleep(0.05)

        resp = await client.get("/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "vessel2_lat" in data
        assert "vessel2_lon" in data
        assert "vessel2_heading_deg" in data
        assert "vessel2_speed_kn" in data


# ===================================================================
# 55-58.  GPS Denial
# ===================================================================


class TestGpsDenial:
    @pytest.mark.asyncio
    async def test_gps_deny_on(self, client):
        """GET /api/gps-deny/on toggles denial on; status and state reflect it."""
        resp = await client.get("/api/gps-deny/on")
        assert resp.status_code == 200
        assert resp.json()["status"] == "gps_denied"

        resp = await client.get("/api/gps-deny/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["denied"] is True
        assert isinstance(data["seconds_without_gps"], float)

        resp = await client.get("/api/state")
        assert resp.status_code == 200
        assert resp.json()["gps_denied"] is True
        assert resp.json()["position_source"] == "dead_reckon"

    @pytest.mark.asyncio
    async def test_gps_deny_off(self, client):
        """Toggle on then off; status shows denied=false, position_source=gps."""
        await client.get("/api/gps-deny/on")
        resp = await client.get("/api/gps-deny/off")
        assert resp.status_code == 200
        assert resp.json()["status"] == "gps_restored"

        resp = await client.get("/api/gps-deny/status")
        assert resp.status_code == 200
        assert resp.json()["denied"] is False

        resp = await client.get("/api/state")
        assert resp.status_code == 200
        assert resp.json()["gps_denied"] is False
        assert resp.json()["position_source"] == "gps"

    @pytest.mark.asyncio
    async def test_gps_deny_status(self, monkeypatch):
        """GET /api/gps-deny/status returns denial state and seconds_without_gps field."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Status before denial
            resp = await ac.get("/api/gps-deny/status")
            assert resp.status_code == 200
            assert resp.json()["denied"] is False
            assert resp.json()["seconds_without_gps"] == 0.0

            # Enable denial
            await ac.get("/api/gps-deny/on")
            resp = await ac.get("/api/gps-deny/status")
            assert resp.status_code == 200
            assert resp.json()["denied"] is True
            assert resp.json()["seconds_without_gps"] >= 0.0

    @pytest.mark.asyncio
    async def test_gps_deny_stops_gps_broadcast(self, monkeypatch):
        """With GPS denied, POST /gps should not update track state position."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Post initial GPS fix
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59})
            resp = await ac.get("/api/track")
            initial_count = resp.json()["properties"]["count"]
            assert initial_count == 1

            # Enable denial
            await ac.get("/api/gps-deny/on")

            # Post GPS again — should not append to track_buffer
            await ac.post("/gps", json={"lat": 56.18, "lon": 15.60})

            # Track count should remain unchanged
            resp = await ac.get("/api/track")
            assert resp.json()["properties"]["count"] == initial_count

    @pytest.mark.asyncio
    async def test_documented_gps_denial_api(self, monkeypatch):
        """POST/GET /api/gps-denial mirror the documented contract."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0})

            resp = await ac.post("/api/gps-denial", json={"enabled": True})
            assert resp.status_code == 200
            data = resp.json()
            assert data["gps_denied"] is True
            assert data["last_gps"]["lat"] == pytest.approx(56.16)
            assert data["simulator_gps_stopped"] is True

            resp = await ac.get("/api/gps-denial")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True

            resp = await ac.post("/api/gps-denial", json={"enabled": False})
            assert resp.status_code == 200
            assert resp.json()["gps_denied"] is False


# ===================================================================
# 59-62.  Dead Reckoning
# ===================================================================


class TestDeadReckoning:
    @pytest.mark.asyncio
    async def test_dead_reckon_returns_position(self, client):
        """GET /api/dead-reckon returns DR position, uncertainty, and heading fields."""
        await client.post(
            "/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0, "heading_deg": 45.0},
        )
        await asyncio.sleep(0.2)
        await client.get("/api/gps-deny/on")

        resp = await client.get("/api/dead-reckon")
        assert resp.status_code == 200
        data = resp.json()

        assert data["dr_lat"] is not None
        assert data["dr_lon"] is not None
        assert "gps_lat" in data
        assert "gps_lon" in data
        assert data["heading_source"] == "magnetometer"
        assert isinstance(data["uncertainty_m"], float)
        assert data["position_source"] == "dead_reckon"
        assert isinstance(data["uncertainty_ellipse"], dict)

    @pytest.mark.asyncio
    async def test_dead_reckon_uncertainty_grows(self, monkeypatch):
        """Enable GPS denial, then verify uncertainty increases over time."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Inject gyro bias so drift_rate_dps > 0
            for _ in range(50):
                await ac.post(
                    "/data",
                    json={
                        "payload": [
                            {"name": "gyroscope", "values": {"x": 0.0, "y": 0.0, "z": 0.05}},
                        ]
                    },
                )

            # Post GPS with speed so uncertainty formula has inputs
            await ac.post(
                "/gps",
                json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0},
            )

            # Enable denial
            await ac.get("/api/gps-deny/on")

            # Uncertainty at t=0
            resp = await ac.get("/api/dead-reckon")
            u0 = resp.json()["uncertainty_m"]

            # Wait for uncertainty to accumulate
            await asyncio.sleep(2.0)

            resp = await ac.get("/api/dead-reckon")
            u1 = resp.json()["uncertainty_m"]

            assert u1 > u0, f"Expected uncertainty to grow over time: {u0} -> {u1}"

    @pytest.mark.asyncio
    async def test_dead_reckon_resets_on_gps(self, client):
        """Turn GPS denial off; verify position_source returns to 'gps'."""
        await client.get("/api/gps-deny/on")
        resp = await client.get("/api/dead-reckon")
        assert resp.json()["position_source"] == "dead_reckon"

        await client.get("/api/gps-deny/off")
        resp = await client.get("/api/dead-reckon")
        assert resp.json()["position_source"] == "gps"

    @pytest.mark.asyncio
    async def test_dead_reckon_ellipse_fields(self, client):
        """Verify uncertainty_ellipse has semi_major, semi_minor, angle_deg."""
        await client.post(
            "/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0},
        )
        await asyncio.sleep(0.2)
        await client.get("/api/gps-deny/on")

        resp = await client.get("/api/dead-reckon")
        assert resp.status_code == 200
        ellipse = resp.json()["uncertainty_ellipse"]

        assert "semi_major" in ellipse
        assert "semi_minor" in ellipse
        assert "angle_deg" in ellipse
        assert isinstance(ellipse["semi_major"], float)
        assert isinstance(ellipse["semi_minor"], float)
        assert isinstance(ellipse["angle_deg"], float)

    @pytest.mark.asyncio
    async def test_documented_dead_reckon_api(self, monkeypatch):
        """Documented POST/status dead-reckon endpoints return useful fields."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0})
            await ac.post("/api/gps-denial", json={"enabled": True})

            resp = await ac.post("/api/dead-reckon", json={})
            assert resp.status_code == 200
            data = resp.json()
            assert data["lat"] is not None
            assert data["fix_type"] == "dead_reckon"
            assert "semi_major_m" in data["ellipse"]

            resp = await ac.get("/api/dead-reckon/status")
            assert resp.status_code == 200
            status = resp.json()
            assert status["gps_denied"] is True
            assert status["estimated_position"]["lat"] is not None

    @pytest.mark.asyncio
    async def test_gps_denial_before_first_fix_has_fallback_position(self, monkeypatch):
        """GPS denial before the first GPS fix should not return a null DR position."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.get("/api/gps-deny/on")
            resp = await ac.get("/api/dead-reckon")
            assert resp.status_code == 200
            data = resp.json()
            assert data["lat"] is not None
            assert data["lon"] is not None
            assert data["estimated_position"]["lat"] is not None


# ===================================================================
# 63-65.  Magnetometer
# ===================================================================


class TestMagnetometer:
    @pytest.mark.asyncio
    async def test_mag_heading_stored(self, monkeypatch):
        """POST magnetometer data; verify mag_heading is computed and available."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Post magnetometer with known x,y
            await ac.post(
                "/data",
                json={
                    "payload": [
                        {"name": "magnetometer", "values": {"x": 0.0, "y": 1.0, "z": 45.0}},
                    ]
                },
            )
            # Post GPS and enable denial so dead-reckon endpoint is meaningful
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0})
            await ac.get("/api/gps-deny/on")

            resp = await ac.get("/api/dead-reckon")
            assert resp.status_code == 200
            data = resp.json()
            assert data["heading_source"] == "magnetometer"
            assert isinstance(data["uncertainty_ellipse"]["angle_deg"], float)

    @pytest.mark.asyncio
    async def test_dead_reckon_uses_mag_heading(self, client):
        """During GPS denial, verify heading_source is 'magnetometer' in /api/dead-reckon."""
        await client.get("/api/gps-deny/on")
        resp = await client.get("/api/dead-reckon")
        assert resp.status_code == 200
        data = resp.json()
        assert data["heading_source"] == "magnetometer"

    @pytest.mark.asyncio
    async def test_hard_iron_correction_applied(self, monkeypatch):
        """POST biased mag data; verify computed heading is within 0-360 range."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            test_cases = [(10.0, -5.0), (-10.0, 0.0), (0.0, -10.0), (-5.0, -5.0)]
            for x, y in test_cases:
                await ac.post(
                    "/data",
                    json={
                        "payload": [
                            {"name": "magnetometer", "values": {"x": float(x), "y": float(y), "z": 30.0}},
                        ]
                    },
                )
                mag_heading = src.server.state.mag_heading
                assert 0 <= mag_heading <= 360.0, (
                    f"mag_heading {mag_heading} out of range for x={x}, y={y}"
                )

    @pytest.mark.asyncio
    async def test_estimation_tools_are_wired(self, monkeypatch):
        """Server calls the real DVL/IMU and magnetometer package APIs."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post(
                "/data",
                json={
                    "payload": [
                        {"name": "accelerometer", "values": {"x": 0.0, "y": 0.0, "z": 9.81}},
                        {"name": "gyroscope", "values": {"x": 0.0, "y": 0.0, "z": 0.0}},
                    ]
                },
            )
            assert src.server.state.kf_output is not None

            await ac.post(
                "/data",
                json={"payload": [{"name": "magnetometer", "values": {"x": 50.0, "y": 0.0, "z": 0.0}}]},
            )
            assert src.server.state.mag_heading == pytest.approx(90.0)

            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 5.0})
            assert src.server.state.kf_output is not None
            assert len(src.server.state.kf_covariance) == 2

    @pytest.mark.asyncio
    async def test_mag_correction_toggle(self, monkeypatch):
        """Documented magnetometer correction toggle controls DR heading source."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/dead-reckon/mag-correction")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True

            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "heading_deg": 35.0})
            resp = await ac.post("/api/dead-reckon/mag-correction", json={"enabled": False})
            assert resp.status_code == 200
            assert resp.json()["enabled"] is False
            assert resp.json()["heading_source"] == "last_gps_heading"

            resp = await ac.post("/api/dead-reckon/mag-correction", json={"enabled": True})
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True


# ===================================================================
# 66-67.  Live DVL and embedded planner WebSocket
# ===================================================================


class TestLiveDvlAndPlannerWs:
    @pytest.mark.asyncio
    async def test_live_dvl_ingest_and_status(self, monkeypatch):
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 0.0})
            resp = await ac.post(
                "/api/dvl",
                json={
                    "velocity_dvl_m_s": [0.5, 0.0, 0.0],
                    "valid_beams": [True, True, True, True],
                    "altitude_m": 5.0,
                    "mode": "bottom",
                    "status": "valid",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "accepted"
            assert data["diagnostics"]["raw_measurements"] == 1

            resp = await ac.get("/api/dvl/status")
            assert resp.status_code == 200
            assert resp.json()["raw_measurements"] == 1

    def test_fastapi_serves_planner_websocket(self, monkeypatch):
        clean_app = _fresh_app(monkeypatch)
        with StarletteTestClient(clean_app).websocket_connect("/planner/ws") as ws:
            first = ws_receive(ws)
            assert first == {"type": "status", "state": "idle"}

            ws.send_json({"type": "set_path", "waypoints": [[0.0, 0.0], [2.0, 0.0]]})
            ws.send_json({"type": "play"})

            messages = [ws_receive(ws) for _ in range(4)]
            assert any(msg["type"] == "tick" for msg in messages)
            assert any(msg == {"type": "status", "state": "running"} for msg in messages)


class TestOperatorHardening:
    @pytest.mark.asyncio
    async def test_mode_health_events_and_reset(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EDTH_RUNTIME_DIR", str(tmp_path))
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/mode", json={"mode": "training"})
            assert resp.status_code == 200
            assert resp.json()["mode"] == "training"

            resp = await ac.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["mode"] == "training"

            resp = await ac.get("/api/events")
            assert resp.status_code == 200
            assert any(event["kind"] == "mode" for event in resp.json())

            resp = await ac.post("/api/estimator/reset", json={"clear_buffers": True})
            assert resp.status_code == 200
            assert resp.json()["status"] == "reset"

    @pytest.mark.asyncio
    async def test_config_persistence_and_mag_calibration(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EDTH_RUNTIME_DIR", str(tmp_path))
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/config",
                json={
                    "config": {
                        "mode": "live",
                        "magnetometer": {
                            "enabled": False,
                            "hard_iron_offset_uT": [1.0, 2.0, 3.0],
                            "soft_iron_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "expected_field_magnitude_uT": 48.0,
                        },
                        "safety": {"min_depth_m": 3.0},
                    }
                },
            )
            assert resp.status_code == 200
            data = resp.json()["config"]
            assert data["mode"] == "live"
            assert data["magnetometer"]["enabled"] is False
            assert data["safety"]["min_depth_m"] == 3.0

            resp = await ac.get("/api/calibration/magnetometer")
            assert resp.status_code == 200
            assert resp.json()["hard_iron_offset_uT"] == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_recording_is_persisted_and_listed(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EDTH_RUNTIME_DIR", str(tmp_path))
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.get("/api/recording/start")
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59, "speed_kn": 1.0})
            resp = await ac.get("/api/recording/stop")
            assert resp.status_code == 200
            recording_id = resp.json()["recording_id"]

            resp = await ac.get("/api/recording/list")
            assert resp.status_code == 200
            assert any(item["id"] == recording_id for item in resp.json()["recordings"])

            resp = await ac.get("/api/recording/playback", params={"recording_id": recording_id})
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_planner_validation_reports_safety_metadata(self, monkeypatch):
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/planner/validate", json={"waypoints": [[0, 0], [100, 0]]})
            assert resp.status_code == 200
            data = resp.json()
            assert "warnings" in data
            assert data["total_length_m"] == pytest.approx(100.0)


# ===================================================================
# 66-68.  Integration: GPS denial doesn't break other features
# ===================================================================


class TestIntegration:
    @pytest.mark.asyncio
    async def test_gps_deny_doesnt_break_waypoints(self, monkeypatch):
        """Waypoint CRUD still works while GPS denial is active."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Enable GPS denial
            await ac.get("/api/gps-deny/on")

            # Create a waypoint
            resp = await ac.post(
                "/api/waypoints",
                json={"lat": 56.16, "lon": 15.59, "name": "WP1"},
            )
            assert resp.status_code == 200
            assert resp.json()["id"] == 1

            # List waypoints
            resp = await ac.get("/api/waypoints")
            assert resp.status_code == 200
            assert len(resp.json()) == 1

            # Delete waypoint
            resp = await ac.delete("/api/waypoints/1")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_gps_deny_doesnt_break_risk(self, monkeypatch):
        """Risk assessment still returns valid segments during GPS denial."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Create waypoints and GPS fix
            await ac.post("/api/waypoints", json={"lat": 56.16, "lon": 15.59})
            await ac.post("/api/waypoints", json={"lat": 56.18, "lon": 15.60})
            await ac.post("/gps", json={"lat": 56.16, "lon": 15.59})

            # Enable GPS denial
            await ac.get("/api/gps-deny/on")

            # Risk should still work
            resp = await ac.get("/api/risk")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["segments"]) == 1
            assert "mission_risk" in data
            assert "imu" in data

    @pytest.mark.asyncio
    async def test_gps_deny_doesnt_break_recording(self, monkeypatch):
        """Recording start/stop/status still works during GPS denial."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Enable GPS denial
            await ac.get("/api/gps-deny/on")

            # Start recording
            resp = await ac.get("/api/recording/start")
            assert resp.status_code == 200
            assert resp.json()["status"] == "recording"

            # Status shows recording is active
            resp = await ac.get("/api/recording/status")
            assert resp.status_code == 200
            assert resp.json()["recording"] is True

            # Stop recording
            resp = await ac.get("/api/recording/stop")
            assert resp.status_code == 200
            data = resp.json()
            assert "frames" in data
            assert "duration_sec" in data

            # Status shows recording stopped
            resp = await ac.get("/api/recording/status")
            assert resp.status_code == 200
            assert resp.json()["recording"] is False


# ===================================================================
# 71-75.  Demo 6 — Real-data replay module unit tests
# ===================================================================


class TestReplay:
    @pytest.mark.asyncio
    async def test_replay_module_parses_data(self):
        """Instantiate SimrisReplay, verify gps_events and imu_events are non-empty lists."""
        replay = SimrisReplay("/home/tyhug/hackathon/simris_2min")
        assert isinstance(replay.gps_events, list)
        assert len(replay.gps_events) > 0
        assert isinstance(replay.imu_events, list)
        assert len(replay.imu_events) > 0

    @pytest.mark.asyncio
    async def test_replay_gps_format(self):
        """Verify parsed GPS events have lat, lon, timestamp fields in correct ranges."""
        replay = SimrisReplay("/home/tyhug/hackathon/simris_2min")
        event = replay.gps_events[0]
        assert "lat" in event
        assert "lon" in event
        assert "timestamp" in event
        assert 55.5 <= event["lat"] <= 55.6
        assert 14.3 <= event["lon"] <= 14.4
        assert isinstance(event["timestamp"], (int, float))

    @pytest.mark.asyncio
    async def test_replay_imu_format(self):
        """Verify parsed IMU events have acc_x, acc_y, acc_z, gyro_x, gyro_y,
        gyro_z, mag_x, mag_y, mag_z fields."""
        replay = SimrisReplay("/home/tyhug/hackathon/simris_2min")
        event = replay.imu_events[0]
        for field in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                      "mag_x", "mag_y", "mag_z"):
            assert field in event, f"Missing IMU field: {field}"
            assert isinstance(event[field], float)

    @pytest.mark.asyncio
    async def test_replay_start_stop(self, monkeypatch):
        """Start replay via POST /api/replay/start, verify status shows
        running=true, stop it, verify running=false."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start
            resp = await ac.post("/api/replay/start")
            assert resp.status_code == 200
            assert resp.json()["status"] == "running"

            # Status shows running
            resp = await ac.get("/api/replay/status")
            assert resp.status_code == 200
            assert resp.json()["running"] is True

            # Stop
            resp = await ac.post("/api/replay/stop")
            assert resp.status_code == 200
            assert resp.json()["status"] == "stopped"

            # Status shows stopped
            resp = await ac.get("/api/replay/status")
            assert resp.status_code == 200
            assert resp.json()["running"] is False

    @pytest.mark.asyncio
    async def test_replay_feeds_state(self, monkeypatch):
        """Start replay at high speed, wait briefly, GET /api/state —
        verify lat/lon are populated from real data (not default 55.5601)."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start replay at 100x speed
            resp = await ac.post("/api/replay/start", params={"speed": 100})
            assert resp.status_code == 200

            # Let replay feed some GPS events
            await asyncio.sleep(0.5)

            # State should have real Simris GPS data
            resp = await ac.get("/api/state")
            assert resp.status_code == 200
            data = resp.json()
            assert data["lat"] is not None
            assert data["lon"] is not None
            # Real data is around 55.56, not the default BASE_LAT 55.5601
            assert data["lat"] != 55.5601


# ===================================================================
# 76-77.  Demo 6 — Real-data drift and GPS denial
# ===================================================================


class TestRealDataDrift:
    @pytest.mark.asyncio
    async def test_real_imu_drift_nonzero(self, monkeypatch):
        """Start replay, let it run, GET /api/risk — verify imu drift_rate_dps
        is non-zero (real IMU has actual gyro bias)."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start replay at moderate speed to accumulate gyro samples
            resp = await ac.post("/api/replay/start", params={"speed": 50})
            assert resp.status_code == 200

            # Let replay feed IMU events to build up gyro bias
            await asyncio.sleep(1.0)

            # Risk should show non-zero drift from real IMU data
            resp = await ac.get("/api/risk")
            assert resp.status_code == 200
            imu = resp.json()["imu"]
            assert imu["drift_rate_dps"] > 0, (
                "Expected non-zero drift rate from real IMU gyro bias"
            )

    @pytest.mark.asyncio
    async def test_gps_denial_with_real_data(self, monkeypatch):
        """Start replay, enable GPS denial, GET /api/dead-reckon —
        verify it returns DR position with real data."""
        clean_app = _fresh_app(monkeypatch)
        transport = ASGITransport(app=clean_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start replay to feed real GPS data into state
            resp = await ac.post("/api/replay/start", params={"speed": 100})
            assert resp.status_code == 200
            await asyncio.sleep(0.3)

            # Enable GPS denial — captures last GPS fix as DR origin
            resp = await ac.get("/api/gps-deny/on")
            assert resp.status_code == 200

            # Dead-reckon should have position from replayed GPS
            resp = await ac.get("/api/dead-reckon")
            assert resp.status_code == 200
            data = resp.json()
            assert data["dr_lat"] is not None
            assert data["dr_lon"] is not None
            assert data["position_source"] == "dead_reckon"
