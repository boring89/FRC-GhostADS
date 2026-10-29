"""Replay videos of full matches (phase-2 deliverable: 10 matches per version for human inspection).

    uv run python -m ghosttrain.replay runs/t1/latest_net.pt --matches 10 --out runs/t1/replays [--opponent baseline|self]

Runs the policy in the simulator (in-process), records the render state of every tick, renders a fixed
spectator camera with the Godot renderer, writes one mp4 per match (score + clock overlay) and prints the
audit (stuck robots, rule/score inconsistencies, ...) for exactly these matches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ghostsim import SimCfg, load
from ghostsim.robots import ALLIANCE, R

from .audit import Auditor
from .net import TeacherNet
from .obs import Encoder
from .vecenv import _Worker

KEYS = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")


def spectator_rig(game, width=640, height=360):
    fx = (width / 2) / np.tan(np.radians(37))
    return {"name": "spectator", "world": True, "robot_to_camera": [game.mid[0], -8.5, 11.0, 0.0, 41.0, 90.0],
            "capture": [width, height], "fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


@torch.no_grad()
def record(net_path, matches=10, opponent="baseline", seed=0, device="cuda", deterministic=False):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    game = load("2025")
    enc = Encoder(game)
    net = None
    if net_path != "baseline":
        net = TeacherNet(enc.n_goals, enc.G).to(dev)
        sd = torch.load(net_path, map_location=dev, weights_only=False)
        net.load_state_dict(sd["net"] if "net" in sd else sd); net.eval()
    w = _Worker(SimCfg(n_envs=matches, sensors=False, seed=seed))
    w.trunc_at[:] = 10 ** 9
    ctrl = np.zeros((matches, 2), bool)
    ctrl[:, 1] = opponent == "baseline"
    if net is None:
        ctrl[:] = True
    w.set_ctrl(ctrl)
    o = w.reset()
    aud = Auditor(w.sim)
    frames, scores, times = [], [], []
    ticks = int(round((w.sim.end_t - w.sim.t[0]) / 0.1))
    N = matches
    for _ in range(ticks):
        if net is None:
            stick, goal = np.zeros((N, R, 3)), np.zeros((N, R), int)
        else:
            t = {k: torch.from_numpy(np.ascontiguousarray(o[k])).to(dev).reshape(N * R, *o[k].shape[2:]) for k in KEYS}
            t["glob"] = torch.from_numpy(o["glob"]).to(dev)[:, ALLIANCE].reshape(N * R, -1)
            logits, mu, _ = net(t)
            g = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
            s = mu if deterministic else mu + net.stick_log_std().exp() * torch.randn_like(mu)
            stick, goal = np.clip(s.cpu().numpy().reshape(N, R, 3), -1, 1), g.cpu().numpy().reshape(N, R)
        frames.append({k: v.copy() for k, v in w.sim.render_state().items()})
        scores.append(w.sim.score.copy()); times.append(w.sim.t.copy())
        res = w.step(stick, goal)
        aud.observe(res["act_stick"], res["act_goal"], SimpleNamespace(done=res["done"]))
        o = res["obs"]
        if res["done"].all():
            break
    final = w.sim.score.copy() if not res["done"].any() else np.stack(scores)[-1]
    return game, w, frames, np.stack(scores), np.stack(times), aud


def render_videos(game, w, frames, scores, times, out_dir, fps=10, width=640, height=360):
    import cv2

    from ghostrender import dr
    from ghostrender.client import RenderClient
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = scores.shape[1]
    rng = np.random.default_rng(0)
    cam = spectator_rig(game, width, height)
    paths = []
    with RenderClient() as rc:
        rc.configure(game, [cam], n_envs=n, width=width, height=height, max_cams=n)
        for e in range(n):
            rc.set_appearance(e, dr.sample_appearance(rng, w.sim.prm.half[e], w.sim.prm.height[e], w.sim.ours["bumper_z_m"]),
                              {**dr.sample_dr(rng), "crowd_density": 0.6})
        writers = []
        for e in range(n):
            p = out / f"match_{e:02d}.mp4"
            wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not wr.isOpened():
                raise RuntimeError("cv2.VideoWriter could not open mp4v output")
            writers.append(wr); paths.append(p)
        for f, st in enumerate(frames):
            img, _ = rc.render(st, [(e, 0, 0) for e in range(n)])
            for e in range(n):
                fr = np.ascontiguousarray(img[e][..., ::-1])
                txt = f"BLUE {scores[f, e, 0]:.0f}   RED {scores[f, e, 1]:.0f}   t={times[f, e]:.0f}s"
                cv2.putText(fr, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(fr, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writers[e].write(fr)
        for wr in writers:
            wr.release()
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--opponent", default="baseline", choices=["baseline", "self"])
    ap.add_argument("--out", default="runs/replays")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    game, w, frames, scores, times, aud = record(a.net, a.matches, a.opponent, a.seed)
    paths = render_videos(game, w, frames, scores, times, a.out)
    fin = scores[-1]
    print(f"{len(paths)} videos in {a.out}; final scores (blue, red): " + ", ".join(f"{int(b)}-{int(r)}" for b, r in fin))
    print(aud.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
