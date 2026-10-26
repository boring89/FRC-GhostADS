"""Compare real-robot drivetrain tests with the simulator (deliverable: errors within 10 %).

Test protocol on the real robot (open carpet, full battery, log pose at >= 50 Hz, export CSV from AdvantageScope):
  A) straight: from rest, full forward stick for 2.5 s     -> t(0->3 m/s), top speed
  B) spin:     from rest, full rotation stick for 2.0 s    -> t(0->180 deg), top yaw rate

    uv run python tools/calib_compare.py --straight a.csv --spin b.csv \\
        --time Timestamp --x "/Pose/x" --y "/Pose/y" --theta "/Pose/rot"
Column names are whatever your log uses (theta in radians unless --degrees).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def load_csv(path, cols, degrees=False):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if all(r.get(c, "") not in ("", None) for c in cols)]
    a = np.array([[float(r[c]) for c in cols] for r in rows])
    if degrees:
        a[:, 3] = np.radians(a[:, 3])
    return a[:, 0], a[:, 1:3], np.unwrap(a[:, 3])


def metrics_straight(t, xy, v_thr=3.0):
    v = np.linalg.norm(np.gradient(xy, t, axis=0), axis=1)
    k0 = np.argmax(v > 0.05)
    t0 = t[k0] - v[k0] / max(np.gradient(v, t)[k0], 1e-6) if k0 > 0 else t[0]
    k = np.argmax(v >= v_thr) if (v >= v_thr).any() else len(v) - 1
    return {"t_to_3mps": float(t[k] - t0), "top_speed": float(np.percentile(v, 98))}


def metrics_spin(t, th):
    w = np.abs(np.gradient(th, t))
    k0 = np.argmax(w > 0.1)
    turned = np.abs(th - th[k0])
    k = np.argmax(turned >= np.pi) if (turned >= np.pi).any() else len(t) - 1
    return {"t_to_180deg": float(t[k] - t[k0]), "top_yaw_rate": float(np.percentile(w, 98))}


def simulate(profile="ours.json", seconds=(2.5, 2.0)):
    """Run the same two tests in the simulator (one robot, open field). Returns metric dicts."""
    import warnings
    warnings.filterwarnings("ignore")
    from ghostsim import SimCfg, VecSim
    out = []
    for test, T in zip(("straight", "spin"), seconds):
        sim = VecSim(SimCfg(n_envs=1, sensors=False, substeps=1, slots=("ours",) * 6, seed=0))
        b = sim.body
        far = np.array([[2.0, 7.4], [2.0, 0.7], [15.5, 7.4], [15.5, 0.7], [12.0, 7.4]])
        b.pose[0, 1:, :2] = far
        b.pose[0, 0] = [2.5, 6.0, 0.0]
        b.vel[:] = 0
        stick = np.zeros((1, 6, 3))
        stick[0, 0] = [1, 0, 0] if test == "straight" else [0, 0, 1]
        ts, xy, th = [0.0], [b.pose[0, 0, :2].copy()], [0.0]
        for k in range(int(T / 0.02)):
            sim.step(stick, np.zeros((1, 6), int))
            ts.append((k + 1) * 0.02); xy.append(b.pose[0, 0, :2].copy()); th.append(b.pose[0, 0, 2])
        t, xy, th = np.array(ts), np.array(xy), np.unwrap(np.array(th))
        out.append(metrics_straight(t, xy) if test == "straight" else metrics_spin(t, th))
    return out


def compare(real: dict, sim: dict, tol=0.10):
    ok = True
    for k, rv in real.items():
        sv = sim[k]
        err = abs(sv - rv) / max(abs(rv), 1e-9)
        ok &= err <= tol
        print(f"  {k:14s} real {rv:7.3f}  sim {sv:7.3f}  error {100 * err:5.1f} %  {'OK' if err <= tol else 'OUT'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--straight", required=True)
    ap.add_argument("--spin", required=True)
    ap.add_argument("--time", default="Timestamp")
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--theta", required=True)
    ap.add_argument("--degrees", action="store_true")
    ap.add_argument("--profile", default="ours.json")
    a = ap.parse_args()
    cols = [a.time, a.x, a.y, a.theta]
    t, xy, _ = load_csv(a.straight, cols, a.degrees)
    real_a = metrics_straight(t, xy)
    t, _, th = load_csv(a.spin, cols, a.degrees)
    real_b = metrics_spin(t, th)
    sim_a, sim_b = simulate(a.profile)
    print("straight:"); ok = compare(real_a, sim_a)
    print("spin:"); ok &= compare(real_b, sim_b)
    print("PASS (all within 10 %)" if ok else "FAIL: tune robots/ours.json drive params and re-run")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
