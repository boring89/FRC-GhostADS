"""Smoke test for the teacher training pipeline (a few minutes on the P520).

Checks: rollout + PPO update run and stay finite; the cloned warm start keeps the actor intact during the critic
warm-up (KL ~ 0); the critic learns (explained variance); checkpoint save -> load round-trips exactly;
evaluation against the baseline runs. It does NOT claim the policy has learned the game - that takes
~10^8-10^9 agent steps (see docs/training.md).

    uv run python smoke/smoke_train.py [--init runs/bc2/bc_net.pt]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="")
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    from ghosttrain.evaluate import evaluate
    from ghosttrain.mappo import TrainCfg, Trainer
    cfg = TrainCfg(n_procs=6, envs_per_proc=8, horizon=32, minibatch=2048, out="runs/smoke", init=a.init,
                   critic_warmup=4 if a.init else 0, lr=1e-4 if a.init else 3e-4, snapshot_every=4, pool_size=2)
    tr = Trainer(cfg)
    w0 = {k: v.clone() for k, v in tr.net.state_dict().items()}
    t0 = time.time()
    rows = [tr.train(1) for _ in range(a.iters)]
    dt = time.time() - t0
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {name} {detail}")

    check("all losses finite", all(np.isfinite([r["pg"], r["vf"], r["ent_goal"], r["kl"]]).all() for r in rows))
    check("throughput", rows[-1]["sps"] > 1500, f"{rows[-1]['sps']:.0f} agent-steps/s")
    check("critic learns (explained variance > 0)", max(r["expl_var"] for r in rows) > 0.0, f"max {max(r['expl_var'] for r in rows):.2f}")
    if a.init:
        warm = [r for r in rows[:4]]
        check("actor untouched during critic warm-up (|KL| < 1e-3)", all(abs(r["kl"]) < 1e-3 for r in warm), f"{[round(r['kl'], 5) for r in warm]}")
    moved = sum(float((tr.net.state_dict()[k] - w0[k]).abs().sum()) for k in w0)
    check("parameters changed", moved > 0)
    path = tr.save()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    check("checkpoint round-trip", all(torch.equal(ck["net"][k].cpu(), v.cpu()) for k, v in tr.net.state_dict().items()))
    tr.close()
    print(f"trained {rows[-1]['steps']:,} agent-steps in {dt:.0f}s")
    r = evaluate(str(Path("runs/smoke/latest_net.pt")), episodes=8, n_procs=2, envs_per_proc=8)
    check("evaluate() runs", r["episodes"] == 8, f"win {r['win_rate']:.2f}, score {r['score_learner']:.0f} vs {r['score_baseline']:.0f}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
