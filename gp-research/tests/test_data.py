"""Tests for eval.data loaders."""
import urllib.request

import numpy as np
import pytest

from eval.data import (
    SOLAQUA_BAGS,
    SOLAQUA_DIR_CANDIDATES,
    SOLAQUA_URL_TMPL,
    load_snapir,
    load_solaqua_bag,
)


def _quick_network_check(url, timeout=5):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _skip_unless_available(filename, timeout=5):
    cands = [d / filename for d in SOLAQUA_DIR_CANDIDATES]
    if any(c.exists() for c in cands):
        return
    data_id = SOLAQUA_BAGS[filename]["data_id"]
    if not data_id or not _quick_network_check(
            SOLAQUA_URL_TMPL.format(data_id=data_id), timeout=timeout):
        pytest.skip(f"{filename} not cached locally and not reachable quickly")


@pytest.mark.network
def test_load_snapir_shape_and_finite():
    d = load_snapir()
    assert d["v"].ndim == 2 and d["v"].shape[0] == 3
    n = d["v"].shape[1]
    assert d["t"].shape == (n,)
    assert np.all(np.isfinite(d["v"]))
    assert np.all(np.isfinite(d["t"]))


@pytest.mark.network
def test_load_solaqua_gentle_bag():
    filename = "2024-08-22_14-29-05_data.bag"
    _skip_unless_available(filename)

    d = load_solaqua_bag(filename)
    assert d["imu_style"] == "vector3"

    fom = d["a50"]["fom"]
    assert 0.01 < np.median(fom) < 0.05, f"median fom {np.median(fom)} (expect ~0.025)"

    f = d["imu"]["f"]
    mean_norm = float(np.linalg.norm(f.mean(axis=0)))
    assert 9.0 < mean_norm < 11.0, f"|mean specific force| {mean_norm} (expect ~9.81 m/s^2)"

    assert d["a50"]["t"].shape[0] == d["a50"]["v"].shape[0] == d["a50"]["fom"].shape[0]
    assert d["imu"]["t"].shape[0] == d["imu"]["f"].shape[0] == d["imu"]["w"].shape[0]
