import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import calib_compare as cc  # noqa: E402


def _write(path, t, xy, th):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "x", "y", "rot"])
        for row in zip(t, xy[:, 0], xy[:, 1], th):
            w.writerow(row)


def test_metrics_on_analytic_profiles(tmp_path):
    t = np.arange(0, 2.5, 0.02)
    v = np.minimum(8.0 * t, 4.5)
    x = np.cumsum(v) * 0.02
    m = cc.metrics_straight(t, np.stack([x, np.zeros_like(x)], 1))
    assert abs(m["t_to_3mps"] - 3 / 8) < 0.05 and abs(m["top_speed"] - 4.5) < 0.05
    w = np.minimum(20 * t, 9.0)
    m = cc.metrics_spin(t, np.cumsum(w) * 0.02)
    assert abs(m["top_yaw_rate"] - 9.0) < 0.1 and 0.3 < m["t_to_180deg"] < 0.7


def test_sim_against_itself_passes(tmp_path):
    sa, sb = cc.simulate()
    assert cc.compare(sa, sa) and cc.compare(sb, sb)
    assert not cc.compare({"top_speed": sa["top_speed"] * 1.2}, sa)
