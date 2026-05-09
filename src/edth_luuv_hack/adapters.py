"""File adapters for IMU and DVL sample logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .dvl_correction import RawDvlMeasurement
from .dvl_imu_kalman import ImuSample


IMU_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "time", "t"),
    "gyro_x": ("gyro_x", "gx", "angular_velocity_x", "angular_velocity_rad_s_x"),
    "gyro_y": ("gyro_y", "gy", "angular_velocity_y", "angular_velocity_rad_s_y"),
    "gyro_z": ("gyro_z", "gz", "angular_velocity_z", "angular_velocity_rad_s_z"),
    "accel_x": ("accel_x", "ax", "linear_acceleration_x", "linear_acceleration_m_s2_x"),
    "accel_y": ("accel_y", "ay", "linear_acceleration_y", "linear_acceleration_m_s2_y"),
    "accel_z": ("accel_z", "az", "linear_acceleration_z", "linear_acceleration_m_s2_z"),
}

DVL_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "time", "t"),
    "vel_x": ("vel_x", "vx", "velocity_x", "velocity_dvl_x", "velocity_dvl_m_s_x"),
    "vel_y": ("vel_y", "vy", "velocity_y", "velocity_dvl_y", "velocity_dvl_m_s_y"),
    "vel_z": ("vel_z", "vz", "velocity_z", "velocity_dvl_z", "velocity_dvl_m_s_z"),
    "pos_x": ("pos_x", "position_x", "position_nav_x", "position_nav_m_x"),
    "pos_y": ("pos_y", "position_y", "position_nav_y", "position_nav_m_y"),
    "pos_z": ("pos_z", "position_z", "position_nav_z", "position_nav_m_z"),
    "altitude_m": ("altitude_m", "altitude", "range_m"),
    "mode": ("mode", "track_mode"),
    "status": ("status", "quality", "validity"),
}


def load_imu_csv(path: str | Path) -> list[ImuSample]:
    """Load IMU samples from a CSV file with common column aliases."""

    rows = _read_csv_rows(path)
    return [
        ImuSample(
            timestamp_s=_required_float(row, IMU_ALIASES["timestamp_s"]),
            angular_velocity_rad_s=[
                _required_float(row, IMU_ALIASES["gyro_x"]),
                _required_float(row, IMU_ALIASES["gyro_y"]),
                _required_float(row, IMU_ALIASES["gyro_z"]),
            ],
            linear_acceleration_m_s2=[
                _required_float(row, IMU_ALIASES["accel_x"]),
                _required_float(row, IMU_ALIASES["accel_y"]),
                _required_float(row, IMU_ALIASES["accel_z"]),
            ],
        )
        for row in rows
    ]


def load_raw_dvl_csv(path: str | Path) -> list[RawDvlMeasurement]:
    """Load raw DVL samples from a CSV file with common column aliases."""

    measurements = []
    for row in _read_csv_rows(path):
        velocity = _optional_vec3(row, DVL_ALIASES["vel_x"], DVL_ALIASES["vel_y"], DVL_ALIASES["vel_z"])
        position = _optional_vec3(row, DVL_ALIASES["pos_x"], DVL_ALIASES["pos_y"], DVL_ALIASES["pos_z"])
        measurements.append(
            RawDvlMeasurement(
                timestamp_s=_required_float(row, DVL_ALIASES["timestamp_s"]),
                velocity_dvl_m_s=velocity,
                velocity_covariance_dvl=_optional_diag_covariance(row, "vel_var"),
                position_nav_m=position,
                position_covariance_nav=_optional_diag_covariance(row, "pos_var"),
                valid_beams=_optional_beams(row),
                altitude_m=_optional_float(row, DVL_ALIASES["altitude_m"]),
                mode=_optional_text(row, DVL_ALIASES["mode"], default="bottom"),
                status=_optional_text(row, DVL_ALIASES["status"], default="valid"),
            )
        )
    return measurements


def iter_sensor_jsonl(path: str | Path) -> Iterable[tuple[str, ImuSample | RawDvlMeasurement]]:
    """Yield IMU and DVL records from a JSON Lines sensor log."""

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record_type = str(record.get("type", "")).lower()
            if record_type == "imu":
                yield "imu", ImuSample(
                    timestamp_s=float(record["timestamp_s"]),
                    angular_velocity_rad_s=record["angular_velocity_rad_s"],
                    linear_acceleration_m_s2=record["linear_acceleration_m_s2"],
                )
            elif record_type == "dvl":
                yield "dvl", RawDvlMeasurement(
                    timestamp_s=float(record["timestamp_s"]),
                    velocity_dvl_m_s=record.get("velocity_dvl_m_s"),
                    velocity_covariance_dvl=record.get("velocity_covariance_dvl"),
                    position_nav_m=record.get("position_nav_m"),
                    position_covariance_nav=record.get("position_covariance_nav"),
                    valid_beams=record.get("valid_beams"),
                    mode=record.get("mode", "bottom"),
                    status=record.get("status", "valid"),
                    altitude_m=record.get("altitude_m"),
                    velocity_scale_factor=float(record.get("velocity_scale_factor", 1.0)),
                )
            else:
                raise ValueError(f"unknown sensor record type on line {line_number}: {record_type!r}")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _required_float(row: dict[str, str], aliases: tuple[str, ...]) -> float:
    value = _optional_float(row, aliases)
    if value is None:
        raise ValueError(f"missing required column, expected one of {aliases}")
    return value


def _optional_float(row: dict[str, str], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = row.get(alias)
        if value not in (None, ""):
            return float(value)
    return None


def _optional_text(row: dict[str, str], aliases: tuple[str, ...], default: str) -> str:
    for alias in aliases:
        value = row.get(alias)
        if value not in (None, ""):
            return value
    return default


def _optional_vec3(
    row: dict[str, str],
    x_aliases: tuple[str, ...],
    y_aliases: tuple[str, ...],
    z_aliases: tuple[str, ...],
) -> np.ndarray | None:
    values = [_optional_float(row, x_aliases), _optional_float(row, y_aliases), _optional_float(row, z_aliases)]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("partial 3D vector in CSV row")
    return np.asarray(values, dtype=float)


def _optional_diag_covariance(row: dict[str, str], prefix: str) -> np.ndarray | None:
    aliases = ((f"{prefix}_x", f"{prefix}_xx"), (f"{prefix}_y", f"{prefix}_yy"), (f"{prefix}_z", f"{prefix}_zz"))
    values = [_optional_float(row, axis_aliases) for axis_aliases in aliases]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"partial diagonal covariance for {prefix}")
    return np.diag(np.asarray(values, dtype=float))


def _optional_beams(row: dict[str, str]) -> tuple[bool, ...] | None:
    beam_values = []
    for key in ("beam1_valid", "beam2_valid", "beam3_valid", "beam4_valid"):
        value = row.get(key)
        if value in (None, ""):
            continue
        beam_values.append(value.lower() in ("1", "true", "yes", "valid"))
    return tuple(beam_values) if beam_values else None
