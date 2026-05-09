# Underwater Mission Planner

Simple UI for planning an underwater drone path and visualising the position
estimate (mean + covariance) as the drone moves along it.

The frontend is a React/TS/Vite app that talks to the command-center FastAPI
WebSocket at `/planner/ws` when embedded at `/planner/`. The older Rust server
under `frontend/server` is still useful as a standalone protocol stub.

## Layout

```
frontend/
  server/   Optional Rust WS server stub (axum + tokio)
  web/      Vite + React + TypeScript frontend (react-konva)
```

All commands below assume you start from this `frontend/` directory.

## Run

For the integrated command center, build the web bundle and run FastAPI from
the repository root:

```sh
cd web
npm install
npm run build
cd ../..
uvicorn src.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/planner/` or the command-center planner drawer.

For standalone Rust stub development, run this instead in one terminal:

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

Open the Vite URL (default `http://localhost:5173`). In dev mode the frontend
connects to `ws://127.0.0.1:8000/planner/ws`, so keep FastAPI running unless
you temporarily point `web/src/ws.ts` back to the Rust stub.

1. Click on the canvas to drop waypoints (need at least two).
2. Press **Play**. The drone interpolates along the path; the yellow ellipse
   shows the position uncertainty around the current mean.
3. Pause / Reset as needed. Toggle the confidence level (1σ / 2σ / 95%) to
   change the ellipse scaling.

## WebSocket contract

Integrated endpoint: `/planner/ws` on the command-center host. JSON messages,
one per frame. The standalone Rust stub keeps the same contract at
`ws://127.0.0.1:8080/ws`.

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
