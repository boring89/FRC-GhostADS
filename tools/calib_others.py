"""Calibrate robots/others.json against the 2025 Championship divisions (tools/champs_data.py).

    uv run python tools/calib_others.py solo [q ...]     # one robot alone, full match, baseline driver: pieces vs our robot
    uv run python tools/calib_others.py match            # baseline vs baseline matches: alliance metrics vs real

The baseline is a mediocre driver (congestion, slow docking), so absolute numbers are below real ones; the
calibration target is RELATIVE: an 'others' robot of quality q against our robot (FRC 254 numbers), both driven by
the same baseline, compared with real teams against the 97th-percentile team (pieces = coral + net + processor).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# measured with tools/champs_data.py on NEWTON + HOPPER qualifications (250 matches, 150 teams)
REAL_REL = {0.1: 0.30, 0.25: 0.45, 0.5: 0.61, 0.75: 0.75, 0.9: 0.85}
REAL_ALLIANCE = {"score": 179.5, "coral_n": 29.2, "L4": 10.5, "L3": 9.2, "L2": 6.7, "L1": 2.8, "auto_coral": 4.3,
                 "net": 4.5, "proc": 1.6, "barge": 20.0}
PARK = np.array([[5.5, 0.5], [5.5, 7.55], [12.0, 0.5], [12.0, 7.55], [7.0, 0.5]])      # away from stations / cages


def _solo(q, n=16, seed=3):
    import warnings
    warnings.filterwarnings("ignore")
    from ghostsim import SimCfg, VecSim
    from ghosttrain.baseline import Baseline
    sim = VecSim(SimCfg(n_envs=n, sensors=False, start_mode="teleop", seed=seed,
                        slots=("ours",) + ("others",) * 5 if q is None else ("others",) * 6))
    if q is not None:                              # fixed quality, every capability (pieces depend on cycle time only)
        prof = copy.deepcopy(sim.profiles[0])
        rnd = prof["randomize"]
        rnd["quality"] = [q, q]
        for k in rnd["by_quality"]["capability_p"]:
            rnd["by_quality"]["capability_p"][k] = [0.0, 0.0] if k == "climb_shallow" else [1.0, 1.0]
        sim.profiles = [prof] + sim.profiles[1:]
        sim.reset()
    sim.body.pose[:, 1:, :2] = PARK
    sim.gs.hold_coral[:, 1:] = False
    bl = Baseline(sim.game)
    while not (sim.t >= sim.end_t - 1e-9).all():
        st, gl = bl.act(sim, bl.plan(sim))
        st[:, 1:], gl[:, 1:] = 0.0, 0
        sim.step(st, gl)
    return float((sim.gs.scored_coral[:, 0].sum(-1) + sim.gs.scored_algae[:, 0].sum(-1)).mean())


def solo(qs):
    with mp.get_context("spawn").Pool(min(12, len(qs) + 1)) as pool:
        res = pool.map(_solo, [None] + qs)
    print(f"our robot alone: {res[0]:.1f} pieces per match")
    for q, v in zip(qs, res[1:]):
        tgt = f"   real {REAL_REL[q]:.2f}" if q in REAL_REL else ""
        print(f"  q={q:.2f}: {v:5.1f} pieces, ratio {v / res[0]:.2f}{tgt}")


def _match(seed, n=8):
    import warnings
    warnings.filterwarnings("ignore")
    from ghostsim import SimCfg, VecSim
    from ghosttrain.baseline import Baseline
    sim = VecSim(SimCfg(n_envs=n, sensors=False, start_mode="auto", seed=seed))
    bl = Baseline(sim.game)
    auto = None
    while not (sim.t >= sim.end_t - 1e-9).all():
        sim.step(*bl.act(sim, bl.plan(sim)))
        if auto is None and (sim.t >= sim.game.auto_s).all():
            auto = sim.gs.scored_coral.sum(-1).copy()
    gs = sim.gs
    deep = np.stack([((gs.cage_occ == r) & gs.cage_deep).any(1) for r in range(6)], 1)
    return dict(score=gs.score - gs.cat[..., 4], sc=gs.scored_coral.copy(), sa=gs.scored_algae.copy(), net=gs.net.copy(),
                proc=gs.processor.copy(), barge=gs.cat[..., 3].copy(), auto=auto, climbed=gs.climbed.copy(), deep=deep)


def match(n_procs=12):
    with mp.get_context("spawn").Pool(n_procs) as pool:
        outs = pool.map(_match, [100 + i for i in range(n_procs)])
    cat = lambda k: np.concatenate([o[k] for o in outs])
    sc, sa = cat("sc"), cat("sa")
    per_all = lambda x: np.stack([x[:, :3].sum(1), x[:, 3:].sum(1)], 1).reshape(-1, *x.shape[2:])
    a = {"score": cat("score").reshape(-1), "coral_n": per_all(sc).sum(-1), "L4": per_all(sc)[:, 3], "L3": per_all(sc)[:, 2],
         "L2": per_all(sc)[:, 1], "L1": per_all(sc)[:, 0], "auto_coral": per_all(cat("auto")), "net": cat("net").reshape(-1),
         "proc": cat("proc").reshape(-1), "barge": cat("barge").reshape(-1)}
    print(f"{len(sc)} matches (baseline driver): sim mean | real")
    for k, v in a.items():
        print(f"  {k:10s} {np.mean(v):6.1f} | {REAL_ALLIANCE[k]:.1f}")
    cl, dp = cat("climbed")[:, 1:], cat("deep")[:, 1:]
    print(f"  others' end state: deep {dp.mean():.2f} shallow {(cl & ~dp).mean():.2f} | real deep 0.48 shallow 0.00")
    pieces = sc.sum(-1) + sa.sum(-1)
    rel = pieces[:, 1:].reshape(-1) / pieces[:, 0].mean()
    qs = list(REAL_REL)
    print("  others' pieces / ours:", " ".join(f"p{int(q * 100)} {v:.2f}" for q, v in zip(qs, np.percentile(rel, [q * 100 for q in qs]))),
          "| real:", " ".join(f"{v:.2f}" for v in REAL_REL.values()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "solo":
        solo([float(x) for x in sys.argv[2:]] or list(REAL_REL))
    else:
        match()
