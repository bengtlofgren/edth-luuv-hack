"""Data loaders: ANSFL Snapir DVL (V_test.npy) and SINTEF SOLAQUA rosbags.

Both loaders prefer a local cache and fall back to downloading from the
public source. rosbags is only imported inside load_solaqua_bag, so this
module can be imported (and load_snapir used) without it installed.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np

# --- Snapir (ANSFL BeamsNet) ------------------------------------------------

SNAPIR_URL = "https://raw.githubusercontent.com/ansfl/BeamsNet/main/dataset/Test/V_test.npy"
SNAPIR_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "V_test.npy",
    Path.home() / "dvl-gp" / "data" / "V_test.npy",
]


def load_snapir(path=None, timeout=15):
    """Load the ANSFL Snapir AUV test-mission DVL velocity.

    1 Hz, 3-axis body-frame velocity [m/s], from the BeamsNet repo's
    dataset/Test/V_test.npy (CC-BY-4.0). Uses a local cache
    (~/dvl-gp/data/V_test.npy or gp-research/data/V_test.npy) if present,
    else downloads from SNAPIR_URL.

    Returns dict(t=(N,) mission time [s], v=(3,N) velocity [m/s]).
    """
    if path is not None:
        p = Path(path)
    else:
        p = next((c for c in SNAPIR_CANDIDATES if c.exists()), None)
        if p is None:
            p = SNAPIR_CANDIDATES[0]
            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SNAPIR_URL, p)
    v = np.load(p)                       # (3, N)
    t = np.arange(v.shape[1], dtype=float)
    return dict(t=t, v=v)


# --- SOLAQUA (SINTEF Ocean) -------------------------------------------------

SOLAQUA_DIR_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "solaqua",
    Path.home() / "dvl-gp" / "data" / "solaqua",
]
SOLAQUA_URL_TMPL = "https://data.sintef.no/api/public/data/{data_id}/file"

# IMU topic/units per vehicle style (see poc_06_imu_bridge.py docstring):
#   Nucleus /nucleus1000dvl/imu    -- vector3 fields, already SI (m/s^2, rad/s)
#   Pixhawk /sensor/imu            -- flat fields, MAVLink RAW_IMU units
#                                      (acc in mg, gyro in mrad/s)
_PIXHAWK = dict(imu_topic="/sensor/imu", imu_style="flat",
                 acc_scale=9.80665e-3, gyr_scale=1e-3)
_NUCLEUS = dict(imu_topic="/nucleus1000dvl/imu", imu_style="vector3",
                 acc_scale=1.0, gyr_scale=1.0)

# Registry: filename -> dict(data_id, imu_topic, imu_style, acc_scale, gyr_scale).
# data_id is the SINTEF data.sintef.no public data UUID used to download the
# bag if it is not already cached locally (see _locate_or_download).
#
# data_ids recovered from the data.sintef.no public listing API:
# GET /api/public/features/fe-a8f86232-5107-495e-a3dd-a86460eebef6/data?size=1000
# (paginated via searchAfter).
SOLAQUA_BAGS = {
    "2024-08-22_14-29-05_data.bag": dict(
        data_id="5026fa35-2df8-44eb-aa06-2ed1df6c53ae", **_NUCLEUS),
    "2024-08-20_15-18-27_data.bag": dict(
        data_id="ab14ebc0-a2d2-4c7f-89e2-b7c1fcd11b9b", **_PIXHAWK),
    "2024-08-20_15-09-34_data.bag": dict(
        data_id="66e1a803-ee9f-4298-9d1a-35ec523ffcdf", **_PIXHAWK),
    "2024-08-20_15-20-29_data.bag": dict(
        data_id="53893962-70d7-447a-be53-c77fa73b1833", **_PIXHAWK),
    "2024-08-20_14-57-38_data.bag": dict(
        data_id="3bda0651-12fd-4b4d-87a2-27f2fe50cac8", **_PIXHAWK),
    "2024-08-20_17-14-36_data.bag": dict(
        data_id="b4679c55-32e0-41f7-809f-a6e9e1b2ad87", **_PIXHAWK),
    "2024-08-22_14-47-39_data.bag": dict(
        data_id="af730fb6-4f81-48a5-a23c-4a1c12d3c0de", **_NUCLEUS),
    "2024-08-22_14-06-43_data.bag": dict(
        data_id="fdc2ea98-02ed-4b38-bcc4-80fd36f310f5", **_NUCLEUS),
}


def _locate_or_download(filename, data_id, timeout=15):
    cands = [d / filename for d in SOLAQUA_DIR_CANDIDATES]
    p = next((c for c in cands if c.exists()), None)
    if p is not None:
        return p
    if not data_id:
        raise FileNotFoundError(
            f"{filename} not found locally in {[str(c) for c in cands]} and no "
            "data_id is registered to download it (see SOLAQUA_BAGS note).")
    p = cands[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    url = SOLAQUA_URL_TMPL.format(data_id=data_id)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(p, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return p


def load_solaqua_bag(filename, data_id=None, timeout=15):
    """Load one SOLAQUA rosbag: A50 DVL velocity+fom, IMU (SI units), and
    Nortek bottomtrack velocity if that topic carries any valid fixes.

    filename must be a key of SOLAQUA_BAGS, which supplies the IMU topic,
    field layout, and unit-conversion scale for that bag, and (unless
    overridden by the data_id argument) the SINTEF data_id used to download
    it when not already cached under ~/dvl-gp/data/solaqua/ or
    gp-research/data/solaqua/.

    Times are seconds, zeroed at the first A50 fix (matches poc_06).

    Returns dict:
      a50:    dict(t=(n,), v=(n,3) m/s, fom=(n,))          -- WaterLinked A50
      imu:    dict(t=(m,), f=(m,3) m/s^2, w=(m,3) rad/s)   -- specific force / rate
      nortek: dict(t=(k,), v=(k,3) m/s) or None            -- independent 2nd DVL
      imu_style, filename: passthrough metadata.
    """
    from rosbags.highlevel import AnyReader  # heavy/optional; import lazily

    if filename not in SOLAQUA_BAGS:
        raise KeyError(f"{filename!r} is not in SOLAQUA_BAGS registry")
    cfg = SOLAQUA_BAGS[filename]
    data_id = data_id if data_id is not None else cfg["data_id"]
    path = _locate_or_download(filename, data_id, timeout=timeout)
    imu_topic = cfg["imu_topic"]

    a50, imu, nortek = [], [], []
    with AnyReader([path]) as r:
        topics = ("/sensor/dvl_velocity", imu_topic, "/nucleus1000dvl/bottomtrack")
        conns = [c for c in r.connections if c.topic in topics]
        for conn, ts, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            t = ts / 1e9
            if conn.topic == "/sensor/dvl_velocity":
                if m.velocity_valid:
                    a50.append((t, m.velocity.x, m.velocity.y, m.velocity.z, m.fom))
            elif conn.topic == imu_topic:
                if cfg["imu_style"] == "vector3":
                    imu.append((t, m.accelerometer.x, m.accelerometer.y, m.accelerometer.z,
                                m.gyroscope.x, m.gyroscope.y, m.gyroscope.z))
                else:
                    imu.append((t, m.acc_x, m.acc_y, m.acc_z, m.gyro_x, m.gyro_y, m.gyro_z))
            else:
                v = m.dvl_velocity_xyz
                if m.data_valid and (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5 < 10.0:
                    nortek.append((t, v.x, v.y, v.z))    # -32.77 sentinel = invalid

    if not a50:
        raise ValueError(f"no valid A50 fixes in {filename}")
    if not imu:
        raise ValueError(f"no IMU samples on {imu_topic} in {filename}")

    a50 = np.asarray(a50, dtype=float)
    imu = np.asarray(imu, dtype=float)
    nortek = np.asarray(nortek, dtype=float) if nortek else np.empty((0, 4))

    imu[:, 1:4] *= cfg["acc_scale"]
    imu[:, 4:7] *= cfg["gyr_scale"]

    t0 = a50[0, 0]
    a50[:, 0] -= t0
    imu[:, 0] -= t0
    if nortek.size:
        nortek[:, 0] -= t0

    return dict(
        a50=dict(t=a50[:, 0], v=a50[:, 1:4], fom=a50[:, 4]),
        imu=dict(t=imu[:, 0], f=imu[:, 1:4], w=imu[:, 4:7]),
        nortek=dict(t=nortek[:, 0], v=nortek[:, 1:4]) if nortek.size else None,
        imu_style=cfg["imu_style"],
        filename=filename,
    )
