#!/usr/bin/env python3
"""Feed DVL samples into the command center.

Examples:
  python scripts/feed_dvl.py --csv dvl.csv --speed 1
  python scripts/feed_dvl.py --jsonl sensors.jsonl --speed 10
  python scripts/feed_dvl.py --udp 9999
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def dvl_to_payload(measurement: Any) -> dict[str, Any]:
    import numpy as np

    payload = asdict(measurement)
    for key, value in list(payload.items()):
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    return payload


async def post_dvl(client: Any, base_url: str, payload: dict[str, Any]) -> None:
    response = await client.post(f"{base_url}/api/dvl", json=payload)
    response.raise_for_status()


async def feed_file(args: argparse.Namespace) -> None:
    import httpx
    from dvl_correction import RawDvlMeasurement, iter_sensor_jsonl, load_raw_dvl_csv

    base_url = f"http://{args.host}:{args.port}"
    if args.csv:
        measurements = load_raw_dvl_csv(args.csv)
    else:
        measurements = [
            payload for kind, payload in iter_sensor_jsonl(args.jsonl)
            if kind == "dvl" and isinstance(payload, RawDvlMeasurement)
        ]
    print(f"Loaded {len(measurements)} DVL samples from file")
    if not measurements:
        return

    first_ts = measurements[0].timestamp_s
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as client:
        for idx, measurement in enumerate(measurements, start=1):
            if args.speed > 0:
                target = (measurement.timestamp_s - first_ts) / args.speed
                delay = target - (time.monotonic() - start)
                if delay > 0:
                    await asyncio.sleep(delay)
            await post_dvl(client, base_url, dvl_to_payload(measurement))
            if idx % args.report_every == 0:
                print(f"Sent {idx}/{len(measurements)} DVL samples")
    print(f"Done: sent {len(measurements)} DVL samples to {base_url}")


async def feed_udp(args: argparse.Namespace) -> None:
    import httpx

    base_url = f"http://{args.host}:{args.port}"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.udp))
    sock.setblocking(False)
    print(f"Listening for DVL UDP JSON on {args.bind}:{args.udp}; forwarding to {base_url}/api/dvl")
    loop = asyncio.get_running_loop()
    async with httpx.AsyncClient(timeout=5.0) as client:
        count = 0
        while True:
            data, _addr = await loop.sock_recvfrom(sock, 65535)
            record = json.loads(data.decode("utf-8"))
            if record.get("type") == "dvl":
                record.pop("type", None)
            await post_dvl(client, base_url, record)
            count += 1
            if count % args.report_every == 0:
                print(f"Forwarded {count} DVL samples")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV with common DVL column names")
    source.add_argument("--jsonl", type=Path, help="JSONL stream with records of type=dvl")
    source.add_argument("--udp", type=int, help="UDP port receiving one JSON DVL object per datagram")
    parser.add_argument("--bind", default="127.0.0.1", help="UDP bind address")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed; 0 sends as fast as possible")
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()

    if args.udp:
        await feed_udp(args)
    else:
        await feed_file(args)


if __name__ == "__main__":
    asyncio.run(main())
