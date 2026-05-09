# Bug Report: USV Mission Planner Integration — Security & Edge Case Audit

## Critical Bugs

### C1. [HIGH] Set mutation race in WebSocket broadcast causes `RuntimeError`

**Files:** `src/server.py:229-238`

**Description:** `_broadcast()` iterates `state.websockets` directly. When it hits `await ws.send_json(message)` (line 234), the asyncio event loop yields control. During that yield, another task (e.g., `websocket_endpoint` connecting/disconnecting at lines 537-538 or 550) can call `state.websockets.add()` or `state.websockets.discard()`, mutating the set while iteration is in progress. On resume, Python raises `RuntimeError: Set changed size during iteration`.

This crashes the broadcast and kills the simulator loop if the exception isn't caught — all downstream broadcasts stop.

**Reproduction:** Start 10 WebSocket clients, connect/disconnect rapidly while the simulator broadcasts at 20 Hz. Within seconds the broadcast crashes.

**Fix:** Copy to a list before iterating:
```python
for ws in list(state.websockets):
```

---

### C2. [HIGH] NaN injection via sensor values propagates through all calculations

**Files:** `src/server.py:72-75` (SensorValue model), `src/server.py:256-266` (gyro/accel processing)

**Description:** `SensorValue.values` is `dict[str, float]`. Python `float` accepts `float('nan')`. A client POSTing `{"name": "gyroscope", "values": {"z": NaN}}` injects NaN into:
- `gyro_z_bias` (line 259) — mean of samples containing NaN → NaN
- `accel_variance` (line 266) — variance of samples containing NaN → NaN
- `drift_rate_dps` (line 185-186) — `abs(NaN) * factor` → NaN
- `heading_error` (line 439) — `NaN * travel_time` → NaN
- `risk_score` (line 447) — weighted sum of NaN → NaN
- All `round(NaN, ...)` calls → NaN
- `min(NaN, 1.0)` (line 443) → NaN
- `max(0.0, min(1.0, NaN))` (line 448) → NaN

Every client fetching `/api/risk` receives NaN values, and the mission risk dashboard shows no useful data.

**Fix:** Add NaN rejection in Pydantic model or validate values before use:
```python
import math
if sensor.name == "gyroscope" and "z" in sensor.values:
    z = sensor.values["z"]
    if math.isnan(z) or math.isinf(z):
        continue
```

---

### C3. [MEDIUM] Unbounded waypoint list enables memory/performance DoS

**Files:** `src/server.py:124` (waypoints list), `src/server.py:389-406` (create_waypoint)

**Description:** `state.waypoints` is a plain Python list with no maximum size. A client can POST thousands of waypoints, causing:
- Memory exhaustion on the server (each waypoint is a dict with 5+ keys)
- `/api/risk` endpoint degrades — O(n) haversine calculations per segment pair
- Frontend creates a Leaflet marker per waypoint — at 500+ markers, the browser tab becomes unresponsive

**Fix:** Cap waypoints at a reasonable limit (e.g., 50):
```python
if len(state.waypoints) >= 50:
    raise HTTPException(status_code=400, detail="Maximum 50 waypoints allowed")
```

---

### C4. [MEDIUM] CSV export has no output sanitization

**Files:** `src/server.py:526-530`

**Description:** CSV rows are built via f-strings with no escaping of values. If any `lat`, `lon`, `heading_deg`, or `speed_kn` value contained a comma, double-quote, or newline, the CSV output would be malformed. While floats are unlikely to contain these characters in normal operation, malicious input or edge cases (NaN, inf serialization) could produce unexpected output.

**Fix:** Use Python's `csv` module or at minimum wrap values in quotes.

---

### C5. [MEDIUM] Recording produces zero frames when simulator is disabled

**Files:** `src/server.py:703-719`

**Description:** Frame capture only runs inside the simulator loop (`_simulator_loop`). When `SIMULATOR_DISABLED=true` (real sensor mode), the loop doesn't execute, so `recorded_frames` remains empty regardless of incoming sensor data. `recording_start` and `recording_stop` succeed, but playback returns nothing.

This is a critical usability bug for real deployments.

**Fix:** Decouple recording from the simulator. Capture frames from the data stream (e.g., by hooking into `post_gps` and `post_data` endpoints), not from the simulated sensor loop.

---

## Edge Cases Not Handled

### E1. Zero speed inflates risk to max (line 438)

**Details:** `travel_time = d / max(speed, 0.1)` floors speed at 0.1 m/s. At true zero speed, a 1000m segment has travel_time = 10,000 seconds. With any non-zero gyro bias, `heading_error = drift_rate_dps * 10000` produces a huge heading error, making `risk_drift = min(huge / 30.0, 1.0)` = 1.0 (max risk). A stationary vessel should have low heading error, not maximum.

**Impact:** Mission risk shows 100% while vessel is stationary, causing operator confusion.

### E2. GPS denial during a turn (line 259)

**Details:** Gyro bias is computed as a simple arithmetic mean over the last 300 samples (15 seconds at 20 Hz). During a coordinated turn, yaw rate is non-zero for the turn duration. After leveling out, it takes up to 15 seconds for turning samples to flush from the deque. During this window, `gyro_z_bias` is inflated by turn data, causing false drift warnings.

**Impact:** False positive drift alerts for 15 seconds after every turn.

**Fix:** Use a running median or gate the samples by angular rate magnitude to exclude high-rate samples from bias estimation.

### E3. Negative depth from pressure sensor accepted (line 89-90)

**Details:** `PressureData.depth_m` has no `ge=0` validation. Clients can POST negative depth (above waterline). While this doesn't affect risk calculation (which uses bathymetry, not pressure depth), it could cause confusion in the UI and data logs.

### E4. `_last_ts` crashes on empty buffer

**Details:** `_last_ts()` (line 172-174) does `state.buffers[name][-1]` which raises `IndexError` if the buffer is empty. Currently it's only called after `_buffer_sensor` in production code, but this is a latent bug if anyone refactors and calls it on an empty buffer.

### E5. Waypoint created with zero-depth valid values via boundary coordinate

**Details:** `get_depth` returns values close to zero but above 0.5 for shallow water. If a waypoint has depth 0.6m (above the 2.0m USV threshold at line 392), it would be accepted by the `depth < 2.0` rejection check but the vessel would still run aground.

### E6. Simulator vessel2 heading uses `BASE_LAT` for cos conversion (line 664)

**Details:** `cos_lat = math.cos(math.radians(BASE_LAT))` is computed once and reused. V2's actual latitude varies up/down from `V2_CENTER_LAT`. The heading error from using `BASE_LAT` instead of the instantaneous latitude is ~0.01% at this scale, so this is minor but technically incorrect.

---

## Security Issues

### S1. No authentication on any API endpoint

**Details:** All endpoints (`/data`, `/gps`, `/pressure`, `/api/*`, `/ws`) are completely public with no authentication, authorization, or API keys. For a demo this is acceptable, but any production deployment of this code would expose mission control to anyone on the network.

### S2. No rate limiting

**Details:** A client can POST `/data` at any rate without throttling. Combined with the buffer size (300 entries per sensor), this allows a client to completely overwrite legitimate sensor data by flooding the buffers.

### S3. No Content Security Policy (CSP)

**Details:** The HTML served at `/` has no CSP meta tag or header. Combined with third-party CDN scripts (Leaflet from unpkg.com at line 7, 185), this creates a supply-chain XSS risk. A compromised CDN could inject malicious JavaScript into every mission planner session.

**Fix:** Add a CSP header:
```python
@app.middleware("http")
async def add_csp(request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' https://unpkg.com; style-src 'self' 'unsafe-inline' https://unpkg.com; img-src 'self' https://*.tile.openstreetmap.org https://server.arcgisonline.com https://tiles.openseamap.org https://ows.emodnet-bathymetry.eu data:;"
    return response
```

### S4. Missing SRI integrity on CDN resources

**Details:** `index.html` loads Leaflet CSS and JS from unpkg.com (lines 7, 185) without `integrity` attributes. Add SRI hashes to prevent supply-chain attacks.

---

## Race Conditions

### R1. WebSocket set modification race (see C1 above)

Already documented as critical bug C1.

### R2. Recording start races with simulator frame capture

**Details:** `recording_start` (line 480-482) clears `state.recorded_frames` and sets `state.recording = True`. The simulator loop checks `state.recording` (line 704) at its next tick. In between, a GPS or sensor POST could arrive but not be captured because the simulator hasn't ticked yet. This means the first 0-50ms of data after hitting record may not be in the recording. Minor timing issue.

---

## Missing Test Coverage

### T1. No test for NaN/inf sensor values
### T2. No test for very long waypoint names
### T3. No test for concurrent WebSocket connect/disconnect during broadcast
### T4. No test for recording with 0 waypoints
### T5. No test for risk calculation with 1 waypoint (should produce 0 segments)
### T6. No test for rapid recording start/stop cycling
### T7. No test for `get_api_depth` with out-of-range lat/lon
### T8. No test for waypoint creation with name = empty string

---

## Recommendations (Top 5 Fixes by Priority)

| Priority | Fix | Bug Ref | Effort |
|----------|-----|---------|--------|
| 1 | Copy `state.websockets` to list before iteration in `_broadcast` | C1 | 1 line |
| 2 | Reject NaN/inf in Pydantic `SensorValue.values` or before use in gyro/accel processing | C2 | 2 lines |
| 3 | Add waypoint count cap (max 50) in `create_waypoint` | C3 | 3 lines |
| 4 | Decouple recording from simulator loop — capture from data stream | C5 | 1-2 days |
| 5 | Add CSP header middleware and SRI integrity to CDN links | S3, S4 | 5 lines |
