# edth-luuv-hack

Monorepo for navigation fusion experiments on a corrected DVL + IMU stack.

## Components

- [`dvl_correction/`](dvl_correction/) - DVL/IMU error-state Kalman fusion
  layer with raw-DVL correction, frame calibration, optional DVL velocity
  dead-reckoning, runtime diagnostics, a CLI runner, and a streaming pipeline.
  See [`dvl_correction/README.md`](dvl_correction/README.md) for usage.
- [`imu-drift/`](imu-drift/) - Rust IMU drift and Allan variance library plus
  an offline Allan-variance CLI.
- [`frontend/`](frontend/) - React/Vite mission planner canvas embedded in the
  command center, with an optional Rust WebSocket simulator for standalone dev.
- [`src/`](src/) - FastAPI command-center app with maritime map UI, simulated
  and replayed sensor feeds, waypoints, recording/export, GPS-denial controls,
  and dead-reckoning visualization.

## Run The Command Center

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Run The React Planner

The command center serves the built planner at `http://127.0.0.1:8000/planner/`
and provides the planner WebSocket at `/planner/ws`.

For Vite development against the integrated FastAPI WebSocket:

```sh
cd frontend/web
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`, while the command center is
running on port 8000.

The optional Rust stub still speaks the same WebSocket contract:

```sh
cd frontend/server
cargo run
```

## Offline Allan Variance

```sh
cd imu-drift
cargo run --release -- allan-variance --samples imu.csv --dt 0.01 --column gyro_z
```

## Checks

Python command center:

```sh
pip install -r requirements.txt
python -m pytest tests -q
```

DVL correction package:

```sh
pip install -e ./dvl_correction
cd dvl_correction
python3 -m unittest discover -s tests
edth-luuv-fuse --help
```

Rust and frontend:

```sh
cd imu-drift && cargo test --locked && cargo test --locked --features sample
cd frontend/server && cargo test --locked
cd frontend/web && npm ci && npm run typecheck && npm run build
```
