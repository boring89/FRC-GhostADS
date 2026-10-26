"""Deliverable: render FPS and VRAM for 24 cameras at 160x120 on the GTX 1070.

    uv run python smoke/smoke_render.py [--frames 200] [--method forward_plus|mobile|gl_compatibility]
Reports both layouts: 1 env x 6 robots x 4 cams, and 6 envs x 1 robot x 4 cams; render-only and with the
simulator stepping in the loop.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ghostrender import dr  # noqa: E402
from ghostrender.client import RenderClient  # noqa: E402
from ghostsim import SimCfg, VecSim  # noqa: E402


def vram_mib(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        p, m = [x.strip() for x in line.split(",")]
        if int(p) == pid:
            return int(m)
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return f"{out.strip()} (whole GPU; per-process not reported)"


def bench(method, n_envs, cams, frames, W, H):
    sim = VecSim(SimCfg(n_envs=n_envs, sensors=False, seed=0))
    rng = np.random.default_rng(0)
    with RenderClient(method=method) as rc:
        rc.configure(sim.game, sim.ours["cameras"], n_envs=n_envs, width=W, height=H, max_cams=len(cams))
        for e in range(n_envs):
            rc.set_appearance(e, dr.sample_appearance(rng, sim.prm.half[e], sim.prm.height[e], sim.ours["bumper_z_m"]),
                              dr.sample_dr(rng))
        post = [(1.0, 0.02, 2.0, 0.0, float(i)) for i in range(len(cams))]
        for _ in range(5):                                     # warm-up (shader compilation)
            rc.render(sim.render_state(), cams, post)
        t0 = time.perf_counter()
        ms = []
        for _ in range(frames):
            _, h = rc.render(sim.render_state(), cams, post)
            ms.append(h["render_ms"])
        fps_render = frames / (time.perf_counter() - t0)
        stick = rng.uniform(-1, 1, (n_envs, 6, 3))
        goal = rng.integers(0, 13, (n_envs, 6))
        t0 = time.perf_counter()
        for _ in range(frames // 2):
            sim.step(stick, goal)
            img, _ = rc.render(sim.render_state(), cams, post)
        fps_loop = (frames // 2) / (time.perf_counter() - t0)
        vram = vram_mib(rc.proc.pid)
    return fps_render, fps_loop, np.median(ms), vram, img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--method", default="forward_plus")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    a = ap.parse_args()
    layouts = {"1 env x 6 robots x 4 cams": (1, [(0, r, c) for r in range(6) for c in range(4)]),
               "6 envs x 1 robot x 4 cams": (6, [(e, 0, c) for e in range(6) for c in range(4)])}
    for name, (n_envs, cams) in layouts.items():
        fr, fl, med, vram, img = bench(a.method, n_envs, cams, a.frames, a.width, a.height)
        print(f"[{a.method}] {name}: {fr:.1f} requests/s = {fr * len(cams):.0f} images/s render-only "
              f"(engine {med:.1f} ms/request); with sim step in loop {fl:.1f} requests/s; VRAM {vram} MiB; "
              f"mean pixel {img.mean():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
