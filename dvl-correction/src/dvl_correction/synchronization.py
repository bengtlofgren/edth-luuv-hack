"""Buffered synchronization for asynchronous IMU and DVL streams."""

from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass
from itertools import count

import numpy as np

from .dvl_correction import DvlCorrectionLayer, RawDvlMeasurement
from .dvl_imu_kalman import CorrectedDvlMeasurement, DvlImuKalmanLayer, ImuSample, NavigationOutput


@dataclass(frozen=True)
class SynchronizerConfig:
    """Settings for ordering asynchronous sensor events."""

    max_delay_s: float = 0.25


class NavigationFusionPipeline:
    """Run IMU propagation and DVL updates in timestamp order.

    The pipeline buffers events so late-arriving DVL messages can still be
    applied before later IMU samples. Call `flush()` after the log or stream
    ends to process all remaining events.
    """

    def __init__(
        self,
        kalman: DvlImuKalmanLayer | None = None,
        dvl_correction: DvlCorrectionLayer | None = None,
        config: SynchronizerConfig | None = None,
    ) -> None:
        self.kalman = kalman or DvlImuKalmanLayer()
        self.dvl_correction = dvl_correction or DvlCorrectionLayer()
        self.config = config or SynchronizerConfig()
        self._events: list[tuple[float, int, str, object]] = []
        self._event_counter = count()
        self._imu_samples: list[ImuSample] = []
        self._imu_timestamps: list[float] = []
        self._latest_seen_s: float | None = None

    def add_imu_sample(self, sample: ImuSample) -> list[NavigationOutput]:
        """Add an IMU sample and process events older than the delay window."""

        self._remember_imu(sample)
        self._push_event(sample.timestamp_s, "imu", sample)
        return self.process_ready(self._ready_watermark(sample.timestamp_s))

    def add_raw_dvl_measurement(self, measurement: RawDvlMeasurement) -> list[NavigationOutput]:
        """Add a raw DVL sample and process events older than the delay window."""

        self._push_event(measurement.timestamp_s, "raw_dvl", measurement)
        return self.process_ready(self._ready_watermark(measurement.timestamp_s))

    def add_corrected_dvl_measurement(self, measurement: CorrectedDvlMeasurement) -> list[NavigationOutput]:
        """Add a corrected DVL sample and process events older than the delay window."""

        self._push_event(measurement.timestamp_s, "corrected_dvl", measurement)
        return self.process_ready(self._ready_watermark(measurement.timestamp_s))

    def process_ready(self, watermark_s: float) -> list[NavigationOutput]:
        """Process buffered events with timestamps up to `watermark_s`."""

        outputs: list[NavigationOutput] = []
        while self._events and self._events[0][0] <= watermark_s:
            _, _, event_type, payload = heapq.heappop(self._events)
            output = self._process_event(event_type, payload)
            if output is not None:
                outputs.append(output)
        return outputs

    def flush(self) -> list[NavigationOutput]:
        """Process every buffered event."""

        return self.process_ready(float("inf"))

    def _push_event(self, timestamp_s: float, event_type: str, payload: object) -> None:
        timestamp_s = float(timestamp_s)
        if self.kalman.last_timestamp_s is not None and timestamp_s < self.kalman.last_timestamp_s:
            raise ValueError("cannot add an event older than the already processed filter state")
        heapq.heappush(self._events, (timestamp_s, next(self._event_counter), event_type, payload))
        self._latest_seen_s = timestamp_s if self._latest_seen_s is None else max(self._latest_seen_s, timestamp_s)

    def _ready_watermark(self, latest_timestamp_s: float) -> float:
        return float(latest_timestamp_s) - self.config.max_delay_s

    def _remember_imu(self, sample: ImuSample) -> None:
        timestamp_s = float(sample.timestamp_s)
        index = bisect.bisect_left(self._imu_timestamps, timestamp_s)
        if index < len(self._imu_timestamps) and np.isclose(self._imu_timestamps[index], timestamp_s):
            self._imu_samples[index] = sample
            return
        self._imu_timestamps.insert(index, timestamp_s)
        self._imu_samples.insert(index, sample)

    def _process_event(self, event_type: str, payload: object) -> NavigationOutput | None:
        if event_type == "imu":
            return self.kalman.process(payload)  # type: ignore[arg-type]
        if event_type == "corrected_dvl":
            measurement = payload  # type: ignore[assignment]
            self._propagate_filter_to(measurement.timestamp_s)
            update_applied = self.kalman.update_corrected_dvl(measurement)
            return self.kalman.output(measurement.timestamp_s, dvl_update_applied=update_applied)
        if event_type == "raw_dvl":
            raw = payload  # type: ignore[assignment]
            angular_velocity = self._interpolate_imu(raw.timestamp_s).angular_velocity_rad_s
            corrected = self.dvl_correction.correct(raw, angular_velocity_body_rad_s=angular_velocity)
            if corrected is None:
                return None
            self._propagate_filter_to(corrected.timestamp_s)
            update_applied = self.kalman.update_corrected_dvl(corrected)
            return self.kalman.output(corrected.timestamp_s, dvl_update_applied=update_applied)
        raise ValueError(f"unknown event type: {event_type}")

    def _propagate_filter_to(self, timestamp_s: float) -> None:
        timestamp_s = float(timestamp_s)
        last_timestamp_s = self.kalman.last_timestamp_s
        if last_timestamp_s is not None and timestamp_s < last_timestamp_s:
            raise ValueError("cannot apply a DVL update older than the current filter timestamp")
        if last_timestamp_s is None or timestamp_s > last_timestamp_s:
            self.kalman.propagate_imu(self._interpolate_imu(timestamp_s))

    def _interpolate_imu(self, timestamp_s: float) -> ImuSample:
        if not self._imu_samples:
            raise ValueError("DVL correction requires at least one IMU sample")

        timestamp_s = float(timestamp_s)
        index = bisect.bisect_left(self._imu_timestamps, timestamp_s)
        if index < len(self._imu_timestamps) and np.isclose(self._imu_timestamps[index], timestamp_s):
            sample = self._imu_samples[index]
            return ImuSample(
                timestamp_s=timestamp_s,
                angular_velocity_rad_s=sample.angular_velocity_rad_s,
                linear_acceleration_m_s2=sample.linear_acceleration_m_s2,
            )
        if index == 0:
            sample = self._imu_samples[0]
            return ImuSample(timestamp_s, sample.angular_velocity_rad_s, sample.linear_acceleration_m_s2)
        if index == len(self._imu_samples):
            sample = self._imu_samples[-1]
            return ImuSample(timestamp_s, sample.angular_velocity_rad_s, sample.linear_acceleration_m_s2)

        before = self._imu_samples[index - 1]
        after = self._imu_samples[index]
        span = after.timestamp_s - before.timestamp_s
        if span <= 0.0:
            raise ValueError("IMU timestamps must be strictly increasing for interpolation")
        fraction = (timestamp_s - before.timestamp_s) / span
        angular_velocity = _lerp(before.angular_velocity_rad_s, after.angular_velocity_rad_s, fraction)
        linear_acceleration = _lerp(before.linear_acceleration_m_s2, after.linear_acceleration_m_s2, fraction)
        return ImuSample(timestamp_s, angular_velocity, linear_acceleration)


def _lerp(start: object, end: object, fraction: float) -> np.ndarray:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    return start_array + fraction * (end_array - start_array)
