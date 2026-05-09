# edth-luuv-hack

Monorepo for navigation fusion experiments on a corrected DVL + IMU stack.

## Packages

- [`dvl_correction/`](dvl_correction/) - DVL/IMU error-state Kalman fusion
  layer with raw-DVL correction, frame calibration, optional DVL velocity
  dead-reckoning, runtime diagnostics, a CLI runner, and a streaming pipeline.
  See [`dvl_correction/README.md`](dvl_correction/README.md) for usage.

More packages, such as `imu-drift`, can land in follow-up branches.

## Working on a package

Each package is independently pip-installable from its own directory:

```sh
pip install -e ./dvl_correction
```

Run a package's tests from inside its directory:

```sh
cd dvl_correction
python3 -m unittest discover -s tests
```
