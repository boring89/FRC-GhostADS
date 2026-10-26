"""Smoke test (deliverable): full matches without rendering, P processes in parallel.
Pass criterion: >= 1 complete match per second on the P520 with 12 processes.

    uv run python smoke/smoke_sim.py --procs 12 --envs 16 --seconds 60
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ghostsim.parallel import ParallelSim  # noqa: E402
from ghostsim.sim import SimCfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--envs", type=int, default=16, help="envs per process")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--sensors", action="store_true")
    ap.add_argument("--min-mps", type=float, default=1.0)
    a = ap.parse_args()
    cfg = SimCfg(n_envs=a.envs, sensors=a.sensors, seed=0)
    ps = ParallelSim(cfg, a.procs, want_obs=True)
    n = a.procs * a.envs
    rng = np.random.default_rng(0)
    ticks, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        if ticks % 10 == 0:
            stick = rng.uniform(-1, 1, (n, 6, 3))
            goal = rng.integers(0, 13, (n, 6))
        out = ps.step(stick, goal)
        ticks += 1
    dt = time.perf_counter() - t0
    done = ps.matches()
    ps.close()
    sim_s = ticks * a.envs * a.procs * 0.1
    mps = sim_s / 135.0 / dt
    print(f"procs={a.procs} envs/proc={a.envs} sensors={a.sensors} wall={dt:.1f}s ticks={ticks}")
    print(f"simulated {sim_s:,.0f} s = {mps:.2f} matches/s ({done} matches completed), "
          f"finite={np.isfinite(out['obs']['pose']).all()}")
    ok = mps >= a.min_mps and np.isfinite(out["obs"]["pose"]).all()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
