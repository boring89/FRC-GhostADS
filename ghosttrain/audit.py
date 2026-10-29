"""Automatic exploit / bug detection on rollouts (phase-2 'vulnerability detection').

Flags (per episode, then aggregated):
  stuck_wall      a robot commanded to move (|stick| > 0.5) but speed < 0.2 m/s next to a wall/obstacle for > 20 % of the match
  out_of_bounds   a robot outside the field boundary by > 10 cm, or NaN/inf state
  too_fast        speed > 1.1 x its free speed (physics bug / push exploit)
  score_mismatch  alliance score != value recomputed from the field state (+ foul credits)   [teleop episodes]
  conservation    coral != 126 or algae != 18
  score_outlier   alliance score > outlier_pts (default 300) - implausible for this game
  foul_heavy      alliance conceded > 40 foul points in one match
  cycle_loop      the same robot acquired > 40 game pieces (acquire/release loops)
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ghostsim import geometry as geo
from ghostsim.robots import ALLIANCE, R


def state_points(game, gs):
    """Teleop point value of everything scored on the field, per alliance (N,2)."""
    P = game.points
    n = gs.branch.shape[0]
    pts = np.zeros((n, 2))
    for lv, key in enumerate(("L2", "L3", "L4")):
        pts += gs.branch[..., lv].sum((2, 3)) * P["coral"][key][1]
    pts += gs.l1.sum(2) * P["coral"]["L1"][1]
    pts += gs.processor * P["processor"][1] + gs.net * P["net"][1]
    return pts


class Auditor:
    def __init__(self, sim, outlier_pts=300.0):
        self.sim, self.g = sim, sim.game
        self.n = sim.cfg.n_envs
        self.outlier = outlier_pts
        self.stuck = np.zeros((self.n, R))
        self.ticks = np.zeros(self.n)
        self.acq = np.zeros((self.n, R))
        self.prev_hold = sim.game.holding(sim.gs).copy()
        self.flags: dict[str, set] = defaultdict(set)         # flag -> {(env, detail)}
        self.last_check = 0

    def observe(self, stick, goal, info):
        sim, g, b, p, gs = self.sim, self.g, self.sim.body, self.sim.prm, self.sim.gs
        live = sim.t < sim.end_t - 1e-6
        speed = np.linalg.norm(b.vel[..., :2], axis=-1)
        th = b.pose[..., 2]
        hd, _ = geo.halfplane_box(b.pose[..., :2], p.half, np.cos(th), np.sin(th), g.hp_n, g.hp_off, points=False)
        od, _, _ = geo.box_poly_sat(b.pose[..., :2], p.half, np.cos(th), np.sin(th), g.polys, points=False)
        touching = (hd.max(-1) > -0.03) | (od.max(-1) > -0.03)
        cmd = np.linalg.norm(stick[..., :2], axis=-1) > 0.5
        self.stuck += (cmd & (speed < 0.2) & touching & live[:, None])
        self.ticks += live
        for e in range(self.n):
            if not np.isfinite(b.pose[e]).all() or not np.isfinite(b.vel[e]).all():
                self.flags["out_of_bounds"].add((e, "non-finite state"))
        if (hd.max(-1) > 0.10).any():
            for e in np.flatnonzero((hd.max(-1) > 0.10).any(-1)):
                self.flags["out_of_bounds"].add((int(e), "outside boundary"))
        fast = speed > 1.1 * p.free_speed
        for e, r in zip(*np.nonzero(fast)):
            self.flags["too_fast"].add((int(e), f"robot {r}: {speed[e, r]:.1f} m/s"))
        hold = g.holding(gs)
        self.acq += (hold & ~self.prev_hold).sum(-1)
        self.prev_hold = hold.copy()
        # state-derived score check (before the match ends, in teleop-start episodes)
        if self.sim.mode == "teleop" and (sim.t - self.last_check > 5).any():
            self.last_check = sim.t.copy()
            ok = live & (sim.t < sim.end_t - 1.0)
            exp = state_points(g, gs) + gs.fouls[:, ::-1]
            for e in np.flatnonzero(ok & (np.abs(gs.score - exp) > 1e-6).any(-1)):
                self.flags["score_mismatch"].add((int(e), f"score {gs.score[e]} expected {exp[e]}"))
        coral = gs.stock.sum(1) + gs.coral.active.sum(1) + gs.hold_coral.sum(1) + gs.branch.sum((1, 2, 3, 4)) + gs.l1.sum((1, 2)) + gs.coral_lost
        alg = gs.reef_algae.sum((1, 2)) + gs.algae.active.sum(1) + gs.hold_algae.sum(1) + gs.hp_stock.sum(1) + gs.net.sum(1)
        for e in np.flatnonzero((coral != g.coral_count) | (alg != 18)):
            self.flags["conservation"].add((int(e), f"coral {coral[e]} algae {alg[e]}"))
        if info.done.any():
            self._episode_end(info.done)

    def _episode_end(self, done):
        gs = self.sim.gs
        for e in np.flatnonzero(done):
            frac = self.stuck[e] / max(self.ticks[e], 1)
            for r in np.flatnonzero(frac > 0.2):
                self.flags["stuck_wall"].add((int(e), f"robot {r}: {100 * frac[r]:.0f} % of the match"))
            for r in np.flatnonzero(self.acq[e] > 40):
                self.flags["cycle_loop"].add((int(e), f"robot {r}: {self.acq[e, r]:.0f} acquisitions"))
            if (gs.score[e] > self.outlier).any():
                self.flags["score_outlier"].add((int(e), f"scores {gs.score[e]}"))
            if (gs.fouls[e] > 40).any():
                self.flags["foul_heavy"].add((int(e), f"fouls {gs.fouls[e]}"))
            self.stuck[e], self.ticks[e], self.acq[e] = 0, 0, 0

    def report(self) -> dict:
        return {k: {"count": len(v), "examples": sorted(v)[:3]} for k, v in self.flags.items()}

    def summary(self) -> str:
        r = self.report()
        if not r:
            return "audit: no flags"
        return "audit: " + "; ".join(f"{k} x{v['count']} (e.g. {v['examples'][0][1]})" for k, v in r.items())
