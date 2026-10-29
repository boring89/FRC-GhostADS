"""Behaviour-cloning warm start from the rule-based baseline.

    uv run python -m ghosttrain.bc --rounds 40 --out runs/bc

Why: MAPPO from scratch needs ~10^8-10^9 agent steps before it beats the baseline (measured: 7M steps give
0 % win rate). Cloning the baseline first gives a ~baseline-level policy; MAPPO + league then improves it.
Data: first both alliances are baseline-driven with execution noise (labels stay clean, DART-style); from
`dagger_start` on, a growing share of the envs is driven by the net being trained while the baseline still labels
every robot (DAgger), and the last `keep` rounds are trained on together.
Why DAgger: the observation holds the filtered stick command (the chassis filter state), and on the baseline's
own trajectories "this stick = last stick" predicts the label almost perfectly. Cloned from those alone (bc8),
the net copied its own command: starting from rest it kept the stick near 0 (mean |stick| 0.07 vs the baseline's
0.23, robots stalled 0.3-3 m short of their targets) and scored 24 vs 112. On its own states the baseline's labels
say "accelerate", which breaks the copying.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ghostsim import SimCfg, load
from ghostsim.robots import ALLIANCE, R

from .net import LOG_STD_INIT, TeacherNet
from .obs import Encoder
from .vecenv import TeacherVecEnv

KEYS = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")


def to_t(o, dev, N):
    t = {k: torch.from_numpy(np.ascontiguousarray(o[k])).to(dev).reshape(N * R, *o[k].shape[2:]) for k in KEYS}
    t["glob"] = torch.from_numpy(o["glob"]).to(dev)[:, ALLIANCE].reshape(N * R, -1)
    return t


def run(rounds=150, ticks=40, n_procs=12, envs_per_proc=16, epochs=3, minibatch=8192, micro=2048, lr=1e-3,
        noise=0.15, dagger_start=30, dagger_ramp=40, dagger_max=0.8, keep=3, act_noise=0.05, out="runs/bc",
        device="cuda", seed=0, log=lambda m: print(m, flush=True)):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    enc = Encoder(load("2025"))
    net = TeacherNet(enc.n_goals, enc.G).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rounds, eta_min=lr * 0.05)
    env = TeacherVecEnv(SimCfg(n_envs=envs_per_proc, sensors=False, seed=seed), n_procs)
    N = env.n_envs
    rng = np.random.default_rng(seed)
    env.set_ctrl(np.ones((N, 2), bool), noise, label_all=True)
    obs = env.reset()
    t0 = time.time()
    hist = []
    buffer = collections.deque(maxlen=keep)
    for rd in range(rounds):
        beta = 0.0 if rd < dagger_start else min(dagger_max, dagger_max * (rd - dagger_start + 1) / dagger_ramp)
        pol = np.zeros(N, bool)
        pol[rng.permutation(N)[:int(round(beta * N))]] = True               # envs driven by the net this round
        ctrl = np.ones((N, 2), bool)
        ctrl[pol] = False
        env.set_ctrl(ctrl)
        data = {k: [] for k in KEYS + ("glob",)}
        gl, st = [], []
        mag = np.zeros(2)
        for _ in range(ticks):
            t = to_t(obs, dev, N)
            if pol.any():
                with torch.no_grad():
                    logits, mu, _ = net(t)
                goal = logits.argmax(-1).cpu().numpy().reshape(N, R)
                stick = np.clip((mu + act_noise * torch.randn_like(mu)).cpu().numpy().reshape(N, R, 3), -1, 1)
            else:
                goal, stick = np.zeros((N, R), int), np.zeros((N, R, 3))
            res = env.step(stick, goal)
            for k in data:
                data[k].append(t[k])
            gl.append(torch.from_numpy(res["act_goal"]).to(dev).reshape(-1))
            st.append(torch.from_numpy(res["act_stick"]).to(dev).reshape(-1, 3))
            if pol.any():
                mag += [np.abs(stick[pol]).mean(), np.abs(res["act_stick"][pol]).mean()]
            obs = res["obs"]
        buffer.append(({k: torch.cat(v) for k, v in data.items()}, torch.cat(gl), torch.cat(st)))
        data = {k: torch.cat([b[0][k] for b in buffer]) for k in buffer[0][0]}
        gl, st = torch.cat([b[1] for b in buffer]), torch.cat([b[2] for b in buffer])
        n = gl.numel()
        agg = np.zeros(3); cnt = 0
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, minibatch):
                mb = perm[s:s + minibatch]
                opt.zero_grad()
                for c in mb.split(micro):                                   # gradient accumulation (memory only)
                    logits, mu, _ = net({k: v[c] for k, v in data.items()})
                    ce = F.cross_entropy(logits, gl[c])
                    mse = F.smooth_l1_loss(mu, st[c], beta=0.02)            # L1-like: exact stops matter
                    w = c.numel() / mb.numel()
                    ((ce + 20.0 * mse) * w).backward()
                    agg += np.array([ce.item(), mse.item(), (logits.argmax(-1) == gl[c]).float().mean().item()]) * w
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                cnt += 1
        agg /= cnt
        sched.step()
        hist.append(agg)
        if rd % 5 == 0 or rd == rounds - 1:
            m = f" net-driven {pol.mean():.2f}: |stick| net {mag[0] / ticks:.3f} baseline label {mag[1] / ticks:.3f}" if pol.any() else ""
            log(f"bc round {rd + 1:3d}/{rounds} samples {n:,} goal-CE {agg[0]:.3f} acc {agg[2]:.3f} "
                f"stick-L1loss {agg[1]:.4f}{m} [{time.time() - t0:.0f}s]")
    env.close()
    with torch.no_grad():
        net.log_std.fill_(LOG_STD_INIT)
    import pathlib
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, f"{out}/bc_net.pt")
    return net, hist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--dagger_start", type=int, default=30)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--out", default="runs/bc")
    a = ap.parse_args(argv)
    run(a.rounds, a.ticks, noise=a.noise, dagger_start=a.dagger_start, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
