"""Teammate-intent predictor: guesses a robot's current goal from what a teammate can observe about it.

Input per (viewer, teammate) pair: the teammate's ego-frame features as the viewer sees them (pose, velocity,
holding, capabilities, distance; the 'others' row of obs.py, first FO_RAW values) + time left + endgame flag.
Output: a distribution over the 13 goals. The teacher observation gets this distribution for its 2 allies.

Trained on rule-based baseline matches (privileged goal labels) with a plain MLP. Weights are stored in
ghosttrain/intent_weights.npz and evaluated in numpy, so the vectorised environment workers need no torch.

    uv run python -m ghosttrain.intent collect --envs 32 --steps 1500 --seed 100 --out runs/intent/val.npz
    uv run python -m ghosttrain.intent collect --envs 96 --steps 1500 --seed 1 --out runs/intent/train.npz
    uv run python -m ghosttrain.intent train runs/intent/train.npz --val runs/intent/val.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAW_FEATS = 25                                    # FO_RAW (23) + time left + endgame
HIDDEN = 128
WEIGHTS = Path(__file__).with_name("intent_weights.npz")
SCORING = ("score_L1", "score_L2", "score_L3", "score_L4", "net", "processor", "climb")


def load_weights(path=WEIGHTS):
    p = Path(path)
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_np(w, x):
    """x (..., RAW_FEATS) -> probabilities (..., K). Numpy only."""
    h = (x.reshape(-1, RAW_FEATS) - w["mu"]) / w["sd"]
    h = np.maximum(h @ w["W1"] + w["b1"], 0.0)
    h = np.maximum(h @ w["W2"] + w["b2"], 0.0)
    z = h @ w["W3"] + w["b3"]
    z -= z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return p.reshape(x.shape[:-1] + (w["W3"].shape[1],)).astype(np.float32)


def collect(game_name="2025", n_envs=32, steps=1500, seed=0, noise=0.1):
    """Baseline vs baseline. Returns X (M,RAW_FEATS) float32, Y (M,) goal ids, S (M,) 1 if goal is a scoring one."""
    from ghostsim import SimCfg, load
    from ghostsim.robots import R
    from ghostsim.sim import VecSim

    from .baseline import Baseline
    from .obs import OTHERS, Encoder
    game = load(game_name)
    sim = VecSim(SimCfg(n_envs=n_envs, sensors=False, seed=seed))
    enc = Encoder(game, intent_weights=None)            # features do not depend on the intent input
    bl = Baseline(game)
    rng = np.random.default_rng(seed)
    scoring_ids = np.array([game.goals.index(g) for g in SCORING])
    Xs, Ys = [], []
    for k in range(steps):
        stick, goal = bl.act(sim, bl.plan(sim))
        if noise > 0:                                    # small stick noise: more varied teammate states
            stick = np.clip(stick + rng.normal(0, noise, stick.shape), -1, 1)
        enc.encode(sim, goal)
        tf = enc.last_teammate                               # (N,R,2,RAW_FEATS)
        lab = goal[:, OTHERS[:, :2]]                         # (N,R,2) teammate goals
        Xs.append(tf.reshape(-1, RAW_FEATS)); Ys.append(lab.reshape(-1))
        info = sim.step(stick, goal)
        if info.done.any():
            sim.reset(info.done)
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.int64)
    S = np.isin(Y, scoring_ids).astype(np.int8)
    return X, Y, S, game.goals


def _net(in_dim, k):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, k))


def train(X, Y, K, val=None, epochs=6, lr=1e-3, seed=0, batch=2048, log=print, scoring_ids=()):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd)
    Yt = torch.from_numpy(Y)
    net = _net(RAW_FEATS, K)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xn)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(net(Xn[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        msg = f"epoch {ep + 1}/{epochs} train loss {tot / n:.4f}"
        if val is not None:
            msg += "  " + " ".join(f"{k} {v:.3f}" for k, v in evaluate_net(net, mu, sd, *val, scoring_ids).items())
        log(msg)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    w = {"W1": lin[0].weight.detach().numpy().T, "b1": lin[0].bias.detach().numpy(),
         "W2": lin[1].weight.detach().numpy().T, "b2": lin[1].bias.detach().numpy(),
         "W3": lin[2].weight.detach().numpy().T, "b3": lin[2].bias.detach().numpy(),
         "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    return {k: v.astype(np.float32) for k, v in w.items()}


def evaluate_net(net, mu, sd, X, Y, S, scoring_ids):
    import torch
    with torch.no_grad():
        p = torch.softmax(net(torch.from_numpy((X - mu) / sd)), -1).numpy()
    return score_probs(p, Y, S, scoring_ids)


def score_probs(p, Y, S, scoring_ids):
    """acc vs majority-class baseline; recall/precision for 'the teammate is scoring-type' (S = true flag)."""
    pred = p.argmax(-1)
    maj = np.bincount(Y, minlength=p.shape[1]).argmax()
    out = {"acc": float((pred == Y).mean()), "majority": float((Y == maj).mean())}
    s_pred = np.isin(pred, np.asarray(scoring_ids, int))
    if S.sum() > 0:
        out["scoring_recall"] = float((s_pred & (S == 1)).sum() / S.sum())
    if s_pred.sum() > 0:
        out["scoring_precision"] = float((S[s_pred] == 1).mean())
    return out


def scoring_ids_for(goals):
    return [goals.index(g) for g in SCORING]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--envs", type=int, default=32)
    c.add_argument("--steps", type=int, default=1500)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("data")
    t.add_argument("--val")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--out", default=str(WEIGHTS))
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        X, Y, S, goals = collect(n_envs=a.envs, steps=a.steps, seed=a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, X=X, Y=Y, S=S, goals=np.array(goals))
        print(f"{a.out}: {len(X)} teammate samples, scoring-goal share {S.mean():.2f}")
        return 0
    d = np.load(a.data)
    goals = list(d["goals"])
    K = len(goals)
    sids = scoring_ids_for(goals)
    val = None
    if a.val:
        v = np.load(a.val)
        val = (v["X"], v["Y"], v["S"])
    w = train(d["X"], d["Y"], K, val=val, epochs=a.epochs, log=lambda m: print(m, flush=True), scoring_ids=sids)
    if val is not None:
        p = predict_np(w, val[0])
        print("final validation (numpy):", {k: round(v, 3) for k, v in score_probs(p, val[1], val[2], sids).items()})
    np.savez(a.out, **w)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
