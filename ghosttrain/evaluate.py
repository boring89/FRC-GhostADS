"""Win rate of a policy against the rule-based baseline (both colours) + audit of the rollouts.

    uv run python -m ghosttrain.evaluate runs/teacher/latest_net.pt --episodes 200
    uv run python -m ghosttrain.evaluate baseline --episodes 200      # baseline vs baseline sanity check
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from ghostsim import SimCfg
from ghostsim.robots import ALLIANCE, R

from .net import TeacherNet
from .obs import Encoder
from .vecenv import TeacherVecEnv


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


@torch.no_grad()
def evaluate(net_path, episodes=100, n_procs=12, envs_per_proc=8, deterministic=True, seed=123, device="cuda"):
    """Learner on one alliance, baseline on the other; half of the envs each colour."""
    from ghostsim import load
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    enc = Encoder(load("2025"))
    net = None
    if net_path != "baseline":
        net = TeacherNet(enc.n_goals, enc.G).to(dev)
        sd = torch.load(net_path, map_location=dev, weights_only=False)
        net.load_state_dict(sd["net"] if "net" in sd else sd); net.eval()
    env = TeacherVecEnv(SimCfg(n_envs=envs_per_proc, sensors=False, seed=seed), n_procs)
    N = env.n_envs
    learner_red = np.arange(N) % 2 == 1                    # learner = red in odd envs, blue in even envs
    ctrl = np.zeros((N, 2), bool)
    ctrl[np.arange(N), np.where(learner_red, 0, 1)] = True  # the other alliance is the baseline
    if net is None:
        ctrl[:] = True                                       # sanity mode: baseline vs baseline
    env.set_ctrl(ctrl)
    obs = env.reset()
    results = []
    while len(results) < episodes:
        if net is None:
            stick, goal = np.zeros((N, R, 3)), np.zeros((N, R), int)
        else:
            t = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(dev).reshape(N * R, *obs[k].shape[2:])
                 for k in ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")}
            t["glob"] = torch.from_numpy(obs["glob"]).to(dev)[:, ALLIANCE].reshape(N * R, -1)
            logits, mu, _ = net(t)
            goal = (logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample())
            stick = (mu if deterministic else mu + net.log_std.exp() * torch.randn_like(mu))
            stick = np.clip(stick.cpu().numpy().reshape(N, R, 3), -1, 1)
            goal = goal.cpu().numpy().reshape(N, R)
        res = env.step(stick, goal)
        obs = res["obs"]
        for e in np.flatnonzero(res["done"]):
            b, r = res["final"][e]
            if np.isnan(b):                                   # desync truncation, not a finished match
                continue
            mine, theirs = (r, b) if learner_red[e] else (b, r)
            results.append((mine, theirs, learner_red[e]))
    env.close()
    a = np.array(results[:episodes])
    wins = (a[:, 0] > a[:, 1]).sum() + 0.5 * (a[:, 0] == a[:, 1]).sum()
    lo, hi = wilson(wins, len(a))
    return {"episodes": len(a), "win_rate": wins / len(a), "ci95": (lo, hi), "score_learner": a[:, 0].mean(),
            "score_baseline": a[:, 1].mean(), "win_as_blue": ((a[~a[:, 2].astype(bool), 0] > a[~a[:, 2].astype(bool), 1]).mean()),
            "win_as_red": ((a[a[:, 2].astype(bool), 0] > a[a[:, 2].astype(bool), 1]).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args(argv)
    r = evaluate(a.net, a.episodes)
    print(f"{a.net}: win rate {r['win_rate']:.3f} (95% CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}) over {r['episodes']} episodes; "
          f"score {r['score_learner']:.1f} vs {r['score_baseline']:.1f}; as blue {r['win_as_blue']:.2f}, as red {r['win_as_red']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
