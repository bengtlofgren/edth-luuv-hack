# edth-luuv-hack

Monorepo for navigation fusion experiments on a corrected DVL + IMU stack.

## Packages

- [`dvl-correction/`](dvl-correction/) — DVL/IMU error-state Kalman fusion
  layer with raw-DVL correction, frame calibration, and a streaming pipeline.
  See [`dvl-correction/README.md`](dvl-correction/README.md) for usage.

More packages (e.g. `imu-drift`) will land in follow-up branches.

## Working on a package

Each package is independently pip-installable from its own directory:

```sh
pip install -e ./dvl-correction
```

Run a package's tests from inside its directory:

```sh
cd dvl-correction
python3 -m unittest discover -s tests
```
