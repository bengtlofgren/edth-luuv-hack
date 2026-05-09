#!/usr/bin/env python3
"""Replay Simris field data into the maritime mission planner.

Reads simris_2min/{gps,imu}.log and POSTs to the running FastAPI server.
Usage: python scripts/replay_simris.py [--speed 5.0] [--port 8000]
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.replay import SimrisReplay


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=10.0, help="Replay speed multiplier")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Replaying Simris data to {base_url} at {args.speed}x speed...")

    replay = SimrisReplay("simris_2min")
    print(f"Loaded {len(replay.gps_events)} GPS + {len(replay.imu_events)} IMU events")

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Verify server
        try:
            r = await client.get(f"{base_url}/api/state")
            r.raise_for_status()
        except Exception:
            print(f"ERROR: Cannot reach server at {base_url}")
            sys.exit(1)

        count = 0
        gps_count = 0
        imu_count = 0
        start = time.monotonic()

        async for evt in replay.replay(speed=args.speed, on_gps=None, on_imu=None):
            if evt["type"] == "gps":
                await client.post(f"{base_url}/gps", json=evt)
                gps_count += 1
            elif evt["type"] == "imu":
                # Build the multi-sensor payload format
                payload = {
                    "payload": [
                        {"name": evt["sensor"], "values": {k: v for k, v in evt.items() if k not in ("type", "sensor", "timestamp")}}
                    ]
                }
                await client.post(f"{base_url}/data", json=payload)
                imu_count += 1

            count += 1
            if count % 500 == 0:
                elapsed = time.monotonic() - start
                print(f"  {count}/{replay.total_events} events ({gps_count}gps + {imu_count}imu) in {elapsed:.1f}s")

        elapsed = time.monotonic() - start
        print(f"Done: {gps_count} GPS + {imu_count} IMU in {elapsed:.1f}s ({args.speed}x real-time)")


if __name__ == "__main__":
    asyncio.run(main())
