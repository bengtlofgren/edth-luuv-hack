"""Command-line runner for EDTH LUUV DVL/IMU fusion."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .adapters import iter_sensor_jsonl, load_imu_csv, load_raw_dvl_csv
from .config import build_navigation_pipeline_from_json
from .dead_reckoning import DvlDeadReckoningTrack
from .dvl_correction import RawDvlMeasurement
from .dvl_imu_kalman import ImuSample, NavigationOutput
from .synchronization import NavigationFusionPipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse IMU and DVL logs into navigation estimates.")
    parser.add_argument("--config", help="Optional JSON navigation config.")
    parser.add_argument("--jsonl", help="Mixed IMU/DVL JSON Lines log.")
    parser.add_argument("--imu-csv", help="IMU CSV log.")
    parser.add_argument("--dvl-csv", help="DVL CSV log.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for navigation estimates.")
    parser.add_argument("--diagnostics-json", help="Optional JSON path for pipeline diagnostics.")
    parser.add_argument(
        "--enable-dvl-track",
        action="store_true",
        help="Integrate corrected DVL velocity into a position track when DVL position is absent.",
    )
    args = parser.parse_args(argv)

    if args.jsonl and (args.imu_csv or args.dvl_csv):
        parser.error("use either --jsonl or --imu-csv/--dvl-csv, not both")
    if not args.jsonl and not (args.imu_csv and args.dvl_csv):
        parser.error("provide --jsonl or both --imu-csv and --dvl-csv")

    pipeline = build_navigation_pipeline_from_json(args.config) if args.config else NavigationFusionPipeline()
    if args.enable_dvl_track and pipeline.dvl_track is None:
        pipeline.dvl_track = DvlDeadReckoningTrack()

    outputs = _run_jsonl(pipeline, args.jsonl) if args.jsonl else _run_csv_pair(pipeline, args.imu_csv, args.dvl_csv)
    _write_outputs_csv(args.output_csv, outputs)

    if args.diagnostics_json:
        Path(args.diagnostics_json).write_text(
            json.dumps(asdict(pipeline.diagnostics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0


def _run_jsonl(pipeline: NavigationFusionPipeline, path: str) -> list[NavigationOutput]:
    records = list(iter_sensor_jsonl(path))
    return _run_events(pipeline, records)


def _run_csv_pair(pipeline: NavigationFusionPipeline, imu_csv: str, dvl_csv: str) -> list[NavigationOutput]:
    records: list[tuple[str, ImuSample | RawDvlMeasurement]] = []
    records.extend(("imu", sample) for sample in load_imu_csv(imu_csv))
    records.extend(("dvl", measurement) for measurement in load_raw_dvl_csv(dvl_csv))
    return _run_events(pipeline, records)


def _run_events(
    pipeline: NavigationFusionPipeline,
    records: list[tuple[str, ImuSample | RawDvlMeasurement]],
) -> list[NavigationOutput]:
    outputs: list[NavigationOutput] = []
    for record_type, payload in sorted(records, key=_event_sort_key):
        if record_type == "imu":
            if not isinstance(payload, ImuSample):
                raise TypeError("imu record must contain ImuSample")
            outputs.extend(pipeline.add_imu_sample(payload))
        elif record_type == "dvl":
            if not isinstance(payload, RawDvlMeasurement):
                raise TypeError("dvl record must contain RawDvlMeasurement")
            outputs.extend(pipeline.add_raw_dvl_measurement(payload))
        else:
            raise ValueError(f"unknown record type: {record_type}")
    outputs.extend(pipeline.flush())
    return outputs


def _event_sort_key(record: tuple[str, ImuSample | RawDvlMeasurement]) -> tuple[float, int]:
    record_type, payload = record
    priority = 0 if record_type == "imu" else 1
    return float(payload.timestamp_s), priority


def _write_outputs_csv(path: str, outputs: list[NavigationOutput]) -> None:
    fieldnames = [
        "timestamp_s",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "attitude_qw",
        "attitude_qx",
        "attitude_qy",
        "attitude_qz",
        "gyro_bias_x_rad_s",
        "gyro_bias_y_rad_s",
        "gyro_bias_z_rad_s",
        "accel_bias_x_m_s2",
        "accel_bias_y_m_s2",
        "accel_bias_z_m_s2",
        "dvl_update_applied",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for output in outputs:
            writer.writerow(
                {
                    "timestamp_s": output.timestamp_s,
                    "position_x_m": output.position_m[0],
                    "position_y_m": output.position_m[1],
                    "position_z_m": output.position_m[2],
                    "velocity_x_m_s": output.velocity_m_s[0],
                    "velocity_y_m_s": output.velocity_m_s[1],
                    "velocity_z_m_s": output.velocity_m_s[2],
                    "attitude_qw": output.attitude_quat_wxyz[0],
                    "attitude_qx": output.attitude_quat_wxyz[1],
                    "attitude_qy": output.attitude_quat_wxyz[2],
                    "attitude_qz": output.attitude_quat_wxyz[3],
                    "gyro_bias_x_rad_s": output.gyro_bias_rad_s[0],
                    "gyro_bias_y_rad_s": output.gyro_bias_rad_s[1],
                    "gyro_bias_z_rad_s": output.gyro_bias_rad_s[2],
                    "accel_bias_x_m_s2": output.accel_bias_m_s2[0],
                    "accel_bias_y_m_s2": output.accel_bias_m_s2[1],
                    "accel_bias_z_m_s2": output.accel_bias_m_s2[2],
                    "dvl_update_applied": output.dvl_update_applied,
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
