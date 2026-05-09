"""Real-data replay pipeline for Simrishamn USV field deployment.

Parses and replays GPS + IMU log files from the simris_2min dataset
collected near Simrishamn, Sweden (55.56N, 14.36E).

Independent module -- does not import from src.server (server imports this).
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Coroutine


def parse_gps_log(path: str) -> list[dict[str, Any]]:
    """Parse simris_2min/gps.log into list of GPS events."""
    events: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            events.append({
                "rel_time": float(parts[0]),
                "timestamp": int(parts[2]) / 1000.0,
                "lat": float(parts[3]),
                "lon": float(parts[4]),
            })
    return events


def parse_imu_log(path: str) -> list[dict[str, Any]]:
    """Parse simris_2min/imu.log into list of IMU sensor events."""
    events: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 13:
                continue
            events.append({
                "rel_time": float(parts[0]),
                "accelerometer": {
                    "x": float(parts[4]),
                    "y": float(parts[5]),
                    "z": float(parts[6]),
                },
                "gyroscope": {
                    "x": float(parts[7]),
                    "y": float(parts[8]),
                    "z": float(parts[9]),
                },
                "magnetometer": {
                    "x": float(parts[10]),
                    "y": float(parts[11]),
                    "z": float(parts[12]),
                },
                # Flat field names for convenience
                "acc_x": float(parts[4]),
                "acc_y": float(parts[5]),
                "acc_z": float(parts[6]),
                "gyro_x": float(parts[7]),
                "gyro_y": float(parts[8]),
                "gyro_z": float(parts[9]),
                "mag_x": float(parts[10]),
                "mag_y": float(parts[11]),
                "mag_z": float(parts[12]),
            })
    return events


class SimrisReplay:
    """Load and replay Simris field data with real-time event spacing.

    Accepts optional callbacks so the consumer can route events however it
    likes (direct function calls, HTTP posts, etc.) without the replay module
    needing to know about the server.
    """

    def __init__(self, data_dir: str = "simris_2min"):
        base = Path(data_dir)
        self.gps_events = parse_gps_log(str(base / "gps.log"))
        self.imu_events = parse_imu_log(str(base / "imu.log"))

        # Build merged, sorted timeline
        timeline: list[tuple[float, str, dict[str, Any]]] = []
        for e in self.gps_events:
            timeline.append((e["rel_time"], "gps", e))
        for e in self.imu_events:
            timeline.append((e["rel_time"], "imu", e))
        timeline.sort(key=lambda x: x[0])
        self.timeline = timeline
        self.total_events = len(timeline)

        # Replay state
        self.running = False
        self.progress = 0.0
        self.current_rel_time = 0.0
        self._task: asyncio.Task[None] | None = None

        # Starting position (for map centering)
        if self.gps_events:
            self.start_lat = self.gps_events[0]["lat"]
            self.start_lon = self.gps_events[0]["lon"]
        else:
            self.start_lat = 55.560
            self.start_lon = 14.363

    async def replay(
        self,
        speed: float = 1.0,
        on_gps: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
        on_imu: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Step through the timeline at *speed*, calling callbacks for each
        event and yielding individual sensor events.

        Each timeline step fires:
          * ``on_gps(event)``  -- receives the full GPS dict
          * ``on_imu(event)``  -- receives the full IMU row dict with three
                                  sensor sub-dicts (accelerometer, gyroscope,
                                  magnetometer)

        The method also **yields** individual sensor events that match the
        API contract::

            {"type": "gps", "lat": ..., "lon": ..., "timestamp": ...}
            {"type": "imu", "sensor": "accelerometer", "x": ..., "y": ..., "z": ...}
        """
        if not self.timeline:
            return

        self.running = True
        self.progress = 0.0

        start_wall = time.monotonic()
        first_rel = self.timeline[0][0]

        for i, (rel_t, evt_type, evt) in enumerate(self.timeline):
            if not self.running:
                break

            # Maintain real-time spacing (adjusted for speed)
            elapsed = time.monotonic() - start_wall
            target_offset = rel_t - first_rel
            delay = target_offset / speed - elapsed
            if delay > 0:
                await asyncio.sleep(delay)

            self.current_rel_time = rel_t
            self.progress = (i + 1) / self.total_events * 100.0

            if evt_type == "gps":
                if on_gps:
                    await on_gps(evt)
                yield {
                    "type": "gps",
                    "lat": evt["lat"],
                    "lon": evt["lon"],
                    "timestamp": evt["timestamp"],
                }

            elif evt_type == "imu":
                if on_imu:
                    await on_imu(evt)
                for sname in ("accelerometer", "gyroscope", "magnetometer"):
                    svals = evt[sname]
                    yield {
                        "type": "imu",
                        "sensor": sname,
                        **svals,
                        "timestamp": rel_t,
                    }

        self.running = False

    def stop(self) -> None:
        """Signal the replay loop to stop."""
        self.running = False

    def get_status(self) -> dict[str, Any]:
        """Return current replay status."""
        return {
            "running": self.running,
            "progress": round(self.progress, 1),
            "total_events": self.total_events,
            "current_rel_time": round(self.current_rel_time, 1),
            "start_lat": self.start_lat,
            "start_lon": self.start_lon,
        }
