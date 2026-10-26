"""Run VecSim in worker processes (one numpy thread each). Finished envs auto-reset in the worker."""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import replace

import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")


def _worker(conn, cfg, want_obs):
    import warnings
    warnings.filterwarnings("ignore")
    from .sim import VecSim
    sim = VecSim(cfg)
    matches = 0
    while True:
        cmd, *args = conn.recv()
        if cmd == "step":
            info = sim.step(*args)
            out = {"points": info.points, "fouls": info.fouls, "done": info.done.copy(), "score": sim.score.copy()}
            if want_obs:
                out["obs"] = {k: np.asarray(v).copy() for k, v in sim.privileged().items()}
            if info.done.any():
                matches += int(info.done.sum())
                sim.reset(info.done)
            conn.send(out)
        elif cmd == "matches":
            conn.send(matches)
        elif cmd == "close":
            conn.close()
            return


class ParallelSim:
    def __init__(self, cfg, n_procs: int, want_obs: bool = True):
        ctx = mp.get_context("spawn")
        self.n_procs, self.per = n_procs, cfg.n_envs
        self.conns, self.procs = [], []
        for i in range(n_procs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, replace(cfg, seed=cfg.seed + 1000 * i), want_obs), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def step(self, stick, goal):
        """stick (P*N,R,3), goal (P*N,R) -> dict of concatenated arrays."""
        for i, c in enumerate(self.conns):
            s = slice(i * self.per, (i + 1) * self.per)
            c.send(("step", stick[s], goal[s]))
        outs = [c.recv() for c in self.conns]
        res = {k: np.concatenate([o[k] for o in outs]) for k in ("points", "fouls", "done", "score")}
        if "obs" in outs[0]:
            res["obs"] = {k: np.concatenate([o["obs"][k] for o in outs]) for k in outs[0]["obs"]}
        return res

    def matches(self):
        for c in self.conns:
            c.send(("matches",))
        return sum(c.recv() for c in self.conns)

    def close(self):
        for c in self.conns:
            c.send(("close",))
        for p in self.procs:
            p.join(timeout=5)
