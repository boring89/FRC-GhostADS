"""Matches between two policies, or a policy and the rule-based baseline, for evaluation.

    uv run python -m ghosttrain.arena runs/teacher8/iter_001540.pt runs/teacher8/iter_000500.pt [--episodes 192]
    uv run python -m ghosttrain.arena runs/teacher8/iter_001540.pt baseline [--stoch] [--asym]

Every env plays whole matches from the start (the training envs cut their first match short to desynchronise);
A is blue in even envs and red in odd envs. Slots are symmetric by default: our robot in both alliances (slots 0 and
3), so the result measures the policies and not which side drew our robot (with the training slots the team with our
robot won ~0.8 of the teacher8 head-to-heads). Policies act deterministically by default (argmax goal, mean stick),
which is less noisy than sampling. team "ours" (default): each policy drives only its alliance's copy of our robot and
the baseline drives the partners (as nets trained with team "ours" play); team "all": each policy drives its whole
alliance (teacher5-8 nets). Reports A's win rate (Wilson 95 % CI) and the mean score difference A - B with a 95 % CI,
overall and per side.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from dataclasses import replace

import numpy as np
import torch

from ghostsim import SimCfg
from ghostsim.robots import ALLIANCE, R

from .vecenv import TeacherVecEnv, _Worker

SYMMETRIC = ("ours", "others", "others", "ours", "others", "others")
KEYS = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")


def _whole_matches(cfg):
    w = _Worker(cfg)
    w.trunc_at[:] = 10 ** 9                                  # no desync cut: every match is a full one
    return w


def _arena_main(conn, cfg):
    w = _whole_matches(cfg)
    while True:
        cmd, *a = conn.recv()
        if cmd == "reset":
            conn.send(w.reset())
        elif cmd == "step":
            conn.send(w.step(*a))
        elif cmd == "ctrl":
            w.set_ctrl(*a)
            conn.send(True)
        elif cmd == "close":
            return


class ArenaEnv(TeacherVecEnv):
    """TeacherVecEnv whose workers play whole matches."""

    def __init__(self, cfg, n_procs: int, in_process: bool = False):
        self.n_procs, self.per, self.in_process = n_procs, cfg.n_envs, in_process
        self.n_envs = n_procs * cfg.n_envs
        if in_process:
            self.local = [_whole_matches(replace(cfg, seed=cfg.seed + 1000 * i)) for i in range(n_procs)]
            return
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n_procs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_arena_main, args=(b, replace(cfg, seed=cfg.seed + 1000 * i)), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)


def load_policy(path, dev):
    """'baseline' or a checkpoint / bc_net / latest_net file -> TeacherNet (eval mode) or None."""
    if path == "baseline":
        return None
    from ghostsim import load
    from .net import TeacherNet
    from .obs import Encoder
    enc = Encoder(load("2025"))
    net = TeacherNet(enc.n_goals, enc.G).to(dev)
    sd = torch.load(path, map_location=dev, weights_only=False)
    net.load_state_dict(sd["net"] if "net" in sd else sd)
    return net.eval()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def summarise(res):
    """res (n,3): A score, B score, A is red. Win rate counts ties as half."""
    res = np.asarray(res, float)
    out = {"episodes": len(res)}
    for name, sel in (("", np.ones(len(res), bool)), ("_blue", res[:, 2] == 0), ("_red", res[:, 2] == 1)):
        x = res[sel]
        if len(x) == 0:
            continue
        w = (x[:, 0] > x[:, 1]) + 0.5 * (x[:, 0] == x[:, 1])
        d = x[:, 0] - x[:, 1]
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        out[f"win{name}"] = float(w.mean())
        out[f"diff{name}"] = float(d.mean())
        if not name:
            out["win_ci"] = wilson(w.sum(), len(w))
            out["diff_ci"] = (float(d.mean() - half), float(d.mean() + half))
            out["score_a"], out["score_b"] = float(x[:, 0].mean()), float(x[:, 1].mean())
    return out


def masks(N, team="ours", b_is_net=True):
    """(N,R) robots driven by A and by B: A on blue in even envs, on red in odd envs. team "ours": our robot only
    (slot 0 / 3), else the whole alliance. B = baseline: nothing (the workers' baseline drives the rest)."""
    a_red = np.arange(N) % 2 == 1
    a_team = (ALLIANCE[None] == 1) == a_red[:, None]
    if team == "ours":
        ours = np.zeros((N, R), bool)
        ours[:, [0, 3]] = True
        a_np, b_np = a_team & ours, ~a_team & ours
    else:
        a_np, b_np = a_team, ~a_team
    return a_np, (b_np if b_is_net else np.zeros_like(b_np))


@torch.no_grad()
def play(a, b, episodes=192, deterministic=True, symmetric=True, n_procs=12, envs_per_proc=8, seed=123,
         device="cuda", in_process=False, team="ours"):
    """A vs B (paths, 'baseline', or loaded nets for B). Returns summarise() of whole matches."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    A = load_policy(a, dev) if isinstance(a, str) else a
    B = load_policy(b, dev) if isinstance(b, str) else b
    assert A is not None, "A must be a policy"
    cfg = SimCfg(n_envs=envs_per_proc, sensors=False, seed=seed, **({"slots": SYMMETRIC} if symmetric else {}))
    env = ArenaEnv(cfg, n_procs, in_process=in_process)
    N = env.n_envs
    a_red = np.arange(N) % 2 == 1
    a_np, b_np = masks(N, team, B is not None)
    env.set_ctrl(~(a_np | b_np))                                                      # the baseline drives the rest
    a_mask, b_mask = torch.from_numpy(a_np.reshape(-1)).to(dev), torch.from_numpy(b_np.reshape(-1)).to(dev)
    obs = env.reset()
    res = []
    try:
        while len(res) < episodes:
            seen = np.zeros(N, bool)
            while not seen.all():                             # one round: every env finishes one match
                t = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(dev).reshape(N * R, *obs[k].shape[2:])
                     for k in KEYS}
                t["glob"] = torch.from_numpy(obs["glob"]).to(dev)[:, ALLIANCE].reshape(N * R, -1)
                goal = torch.zeros(N * R, dtype=torch.long, device=dev)
                stick = torch.zeros(N * R, 3, device=dev)
                for net, m in ((A, a_mask), (B, b_mask)):
                    if net is None or not bool(m.any()):
                        continue
                    lg, mu, _ = net({k: v[m] for k, v in t.items()})
                    goal[m] = lg.argmax(-1) if deterministic else torch.distributions.Categorical(logits=lg).sample()
                    stick[m] = mu if deterministic else mu + net.stick_log_std().exp() * torch.randn_like(mu)
                r = env.step(np.clip(stick.cpu().numpy().reshape(N, R, 3), -1, 1), goal.cpu().numpy().reshape(N, R))
                obs = r["obs"]
                for e in np.flatnonzero(r["done"] & ~seen):
                    blue, red = r["final"][e]
                    if not np.isnan(blue):
                        res.append(((red, blue) if a_red[e] else (blue, red)) + (float(a_red[e]),))
                    seen[e] = True
    finally:
        env.close()
    return summarise(res[:episodes])


def describe(r, a="A", b="B"):
    return (f"{a} vs {b}: win {r['win']:.3f} (CI {r['win_ci'][0]:.2f}-{r['win_ci'][1]:.2f}), score {r['score_a']:.1f} vs "
            f"{r['score_b']:.1f}, diff {r['diff']:+.1f} (CI {r['diff_ci'][0]:+.1f} to {r['diff_ci'][1]:+.1f}); "
            f"as blue {r.get('win_blue', float('nan')):.2f} / as red {r.get('win_red', float('nan')):.2f}; "
            f"n={r['episodes']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", help="checkpoint or 'baseline'")
    ap.add_argument("--episodes", type=int, default=192)
    ap.add_argument("--stoch", action="store_true", help="sample goals and sticks as in training")
    ap.add_argument("--asym", action="store_true", help="training slots (our robot only in slot 0, blue)")
    ap.add_argument("--team", default="ours", choices=["ours", "all"], help="what each policy drives (see the top)")
    ap.add_argument("--n_procs", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    x = ap.parse_args(argv)
    r = play(x.a, x.b, x.episodes, deterministic=not x.stoch, symmetric=not x.asym, n_procs=x.n_procs, team=x.team)
    print(json.dumps(r) if x.json else describe(r, x.a, x.b), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
