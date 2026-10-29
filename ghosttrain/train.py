"""Teacher training entry point.

    uv run python -m ghosttrain.train --iters 1000 --out runs/teacher
    uv run python -m ghosttrain.train --resume runs/teacher/iter_000500.pt --iters 500
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields

import torch

from .mappo import TrainCfg, Trainer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--resume", default=None)
    as_bool = lambda v: str(v).lower() in ("1", "true", "yes", "on")
    for f in fields(TrainCfg):
        t = {int: int, float: float, str: str, bool: as_bool}.get(type(f.default), str)
        ap.add_argument(f"--{f.name}", type=t, default=None)
    a = ap.parse_args(argv)
    cfg = TrainCfg(**{f.name: getattr(a, f.name) for f in fields(TrainCfg) if getattr(a, f.name) is not None})
    tr = Trainer(cfg)
    if a.resume:
        ck = torch.load(a.resume, map_location=tr.dev, weights_only=False)
        tr.net.load_state_dict(ck["net"]); tr.opt.load_state_dict(ck["opt"]); tr.it, tr.steps = ck["it"], ck["steps"]
        tr.ret_rms.mean, tr.ret_rms.var, tr.ret_rms.count = ck["ret_rms"]
        for k, v in ck.get("pool", {}).items():
            tr.league.snaps[k] = v; tr.league.win[k] = ck["win"].get(k, 0.5); tr.league.games[k] = 0
        tr.league.win[0] = ck["win"].get(0, 0.0)
        tr.league.anchors = set(ck.get("anchors", [])) & set(tr.league.snaps)
        tr.league.next_id = max([0, *tr.league.snaps]) + 1
    t0 = time.time()
    try:
        for i in range(a.iters):
            row = tr.train(1)
            if i % 5 == 0 or i == a.iters - 1:
                print(f"it {row['it']:5d} steps {row['steps'] / 1e6:6.2f}M sps {row['sps']:7.0f} "
                      f"ret {row['ret_mean']:+.4f} ent {row['ent_goal']:.2f} kl {row['kl']:+.4f} ev {row['expl_var']:+.2f} "
                      f"clip s/g {row['clip_stick']:.2f}/{row['clip_goal']:.2f} act {row['actor_frac']:.2f} std {row['std_stick']:.3f} "
                      f"gn a/c {row['gn_actor']:.1f}/{row['gn_critic']:.1f} "
                      f"learner {row.get('score_learner', float('nan')):5.1f} opp {row.get('score_opp', float('nan')):5.1f} "
                      f"win_bl {row.get('win_baseline', float('nan')):.2f} ({row.get('n_baseline', 0)}) "
                      f"win_self {row.get('win_self', float('nan')):.2f} pool {row['pool']} [{time.time() - t0:.0f}s]", flush=True)
    finally:
        tr.save()
        tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
