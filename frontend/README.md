# Underwater Mission Planner

Simple UI for planning an underwater drone path and visualising the position
estimate (mean + covariance) as the drone moves along it.

The frontend is a React/TS/Vite app that talks to a Rust WebSocket server. The
server is a **stub** that simulates a drone moving along the path with a
covariance that grows over time and shrinks slightly at each waypoint
(landmark-fix style). The real estimator backend will replace this stub.

## Layout

```
frontend/
  server/   Rust WS server stub (axum + tokio)
  web/      Vite + React + TypeScript frontend (react-konva)
```

All commands below assume you start from this `frontend/` directory.

## Run

In one terminal:

```sh
cd server
cargo run
```

In another:

```sh
cd web
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5173`).

1. Click on the canvas to drop waypoints (need at least two).
2. Press **Play**. The drone interpolates along the path; the yellow ellipse
   shows the position uncertainty around the current mean.
3. Pause / Reset as needed. Toggle the confidence level (1σ / 2σ / 95%) to
   change the ellipse scaling.

## WebSocket contract

Endpoint: `ws://127.0.0.1:8080/ws`. JSON messages, one per frame.

### Client → Server

```json
{ "type": "set_path", "waypoints": [[x, y], ...] }
{ "type": "play" }
{ "type": "pause" }
{ "type": "reset" }
```

### Server → Client

```json
{ "type": "tick", "t": 0.05, "mean": [x, y], "cov": [[xx, xy], [xy, yy]] }
{ "type": "status", "state": "idle" | "running" | "paused" | "done" }
```

Ticks are emitted at ~20 Hz while `state == "running"`.

## Replacing the stub backend

Anything that speaks the contract above can replace `server/`. The frontend
makes no assumptions beyond the JSON shapes. A real estimator only needs to
consume `set_path` and stream `tick` messages of the same form.

## Coordinate system

- Units: metres.
- World bounds shown on canvas: `[-50, 50] × [-50, 50]`.
- World y axis points up; the canvas display flips it for screen coordinates.
