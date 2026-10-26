"""Behaviour-cloning warm start from the rule-based baseline.

    uv run python -m ghosttrain.bc --rounds 40 --out runs/bc

Why: MAPPO from scratch needs ~10^8-10^9 agent steps before it beats the baseline (measured: 7M steps give
0 % win rate). Cloning the baseline first gives a ~baseline-level policy; MAPPO + league then improves it.
Data: both alliances are baseline-driven with execution noise (labels stay clean, DART-style) so the
dataset also covers states slightly off the baseline's own trajectories.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ghostsim import SimCfg, load
from ghostsim.robots import ALLIANCE, R

from .net import TeacherNet
from .obs import Encoder
from .vecenv import TeacherVecEnv

KEYS = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")


def to_t(o, dev, N):
    t = {k: torch.from_numpy(np.ascontiguousarray(o[k])).to(dev).reshape(N * R, *o[k].shape[2:]) for k in KEYS}
    t["glob"] = torch.from_numpy(o["glob"]).to(dev)[:, ALLIANCE].reshape(N * R, -1)
    return t


def run(rounds=40, ticks=40, n_procs=12, envs_per_proc=16, epochs=3, minibatch=8192, lr=1e-3, noise=0.15,
        out="runs/bc", device="cuda", seed=0, log=print):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    enc = Encoder(load("2025"))
    net = TeacherNet(enc.n_goals, enc.G).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rounds, eta_min=lr * 0.05)
    env = TeacherVecEnv(SimCfg(n_envs=envs_per_proc, sensors=False, seed=seed), n_procs)
    N = env.n_envs
    env.set_ctrl(np.ones((N, 2), bool), noise)
    obs = env.reset()
    zero = (np.zeros((N, R, 3)), np.zeros((N, R), int))
    t0 = time.time()
    hist = []
    for rd in range(rounds):
        data = {k: [] for k in KEYS + ("glob",)}
        gl, st = [], []
        for _ in range(ticks):
            t = to_t(obs, dev, N)
            res = env.step(*zero)
            for k in data:
                data[k].append(t[k])
            gl.append(torch.from_numpy(res["act_goal"]).to(dev).reshape(-1))
            st.append(torch.from_numpy(res["act_stick"]).to(dev).reshape(-1, 3))
            obs = res["obs"]
        data = {k: torch.cat(v) for k, v in data.items()}
        gl, st = torch.cat(gl), torch.cat(st)
        n = gl.numel()
        agg = np.zeros(3); cnt = 0
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, minibatch):
                mb = perm[s:s + minibatch]
                logits, mu, _ = net({k: v[mb] for k, v in data.items()})
                ce = F.cross_entropy(logits, gl[mb])
                mse = F.smooth_l1_loss(mu, st[mb], beta=0.02)                # L1-like: exact stops matter
                loss = ce + 20.0 * mse
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                agg += [ce.item(), mse.item(), (logits.argmax(-1) == gl[mb]).float().mean().item()]; cnt += 1
        agg /= cnt
        sched.step()
        hist.append(agg)
        if rd % 5 == 0 or rd == rounds - 1:
            log(f"bc round {rd + 1:3d}/{rounds} samples {n:,} goal-CE {agg[0]:.3f} acc {agg[2]:.3f} stick-L1loss {agg[1]:.4f} [{time.time() - t0:.0f}s]")
    env.close()
    with torch.no_grad():
        net.log_std.fill_(-1.6)
    import pathlib
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, f"{out}/bc_net.pt")
    return net, hist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--out", default="runs/bc")
    a = ap.parse_args(argv)
    run(a.rounds, a.ticks, noise=a.noise, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
