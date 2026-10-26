"""Rule-based baseline policy (privileged state, vectorised over envs x robots).

Pure state machine, no learning. It is the denominator of the win-rate metric, a league opponent and an
end-to-end test of the simulator rules. Output is the same as every policy: stick (N,R,3), goal (N,R).

Priority per robot:  endgame -> park in own barge zone;  holding algae (net capable) -> net;
holding coral -> highest level that still has a free branch on the nearest own reef face;
otherwise the cheapest of: coral station (stock left) / floor coral / floor algae.
Steering: P-controller to an approach pose with a speed profile, repulsion from other robots and obstacles.
"""
from __future__ import annotations

import numpy as np

from ghostsim import geometry as geo
from ghostsim.robots import ALLIANCE, R

LV = ("score_L1", "score_L2", "score_L3", "score_L4")


class Baseline:
    def __init__(self, game, park_s: float = 14.0, kp: float = 2.2, accel: float = 3.0):
        self.g, self.park_s, self.kp, self.accel = game, park_s, kp, accel
        self.gi = {n: i for i, n in enumerate(game.goals)}
        obst = [game.polys.center[i] for i in range(len(game.polys))]
        rad = [1.55, 1.55, 0.75] + [0.45] * 6
        self.obst_c, self.obst_r = np.array(obst), np.array(rad)

    # ------------------------------------------------------------------ helpers
    def _score_target(self, sim, pos, hx):
        """Returns goal id (N,R), approach xy (N,R,2), yaw (N,R), valid (N,R)."""
        g, gs, p = self.g, sim.gs, sim.prm
        n = pos.shape[0]
        env = np.arange(n)[:, None]
        al = ALLIANCE[None].repeat(n, 0)
        fc = g.reef_fc[ALLIANCE]                                    # (R,6,2)
        nn = g.reef_n[ALLIANCE]
        app = fc[None] + nn[None] * (hx[..., None, None] + 0.10)     # (N,R,6,2)
        dist = np.linalg.norm(app - pos[:, :, None], axis=-1)        # (N,R,6)
        br = gs.branch[env, al]                                      # (N,R,6,2,3)
        alg = gs.reef_algae[env, al]                                 # (N,R,6)
        hi = g.algae_high[None, None]
        free = np.zeros((n, R, 4, 6), bool)
        free[:, :, 0] = gs.l1[env, al] < g.l1_cap
        for li in range(3):
            blocked = alg & np.where(hi, g.block["high"][li], g.block["low"][li])
            free[:, :, li + 1] = (~br[..., li]).any(-1) & ~blocked
        free &= p.levels[..., None]
        has = free.any(-1)                                           # (N,R,4)
        lvl = np.where(has.any(-1), 3 - np.argmax(has[..., ::-1], -1), 0)
        ok = has.any(-1)
        pick = np.where(np.take_along_axis(free, lvl[..., None, None].repeat(6, -1), 2)[:, :, 0], dist, np.inf)
        f = np.argmin(pick, -1)
        xy = np.take_along_axis(app, f[..., None, None].repeat(2, -1), 2)[:, :, 0]
        nf = nn[None].repeat(n, 0)[np.arange(n)[:, None], np.arange(R)[None], f]
        yaw = np.arctan2(-nf[..., 1], -nf[..., 0])
        goal = np.array([self.gi[LV[i]] for i in range(4)])[lvl]
        return goal, xy, yaw, ok & np.isfinite(pick.min(-1))

    def _acquire_target(self, sim, pos, hx):
        g, gs, p = self.g, sim.gs, sim.prm
        n = pos.shape[0]
        env = np.arange(n)[:, None]
        al = ALLIANCE[None].repeat(n, 0)
        best = np.full((n, R), np.inf)
        goal = np.zeros((n, R), int)
        xy = pos.copy()
        yaw = np.zeros((n, R))
        # coral station
        sc, sn = g.station_c[ALLIANCE], g.station_n[ALLIANCE]                      # (R,2,2)
        app = sc[None] + sn[None] * (hx[..., None, None] + 0.10)
        d = np.linalg.norm(app - pos[:, :, None], axis=-1)
        k = np.argmin(d, -1)
        dmin = np.take_along_axis(d, k[..., None], -1)[..., 0]
        okc = (gs.stock[env, al] > 0) & p.station & ~gs.hold_coral
        cost = np.where(okc, dmin, np.inf)
        xy_s = np.take_along_axis(app, k[..., None, None].repeat(2, -1), 2)[:, :, 0]
        nk = sn[None].repeat(n, 0)[np.arange(n)[:, None], np.arange(R)[None], k]
        upd = cost < best
        best = np.where(upd, cost, best); goal = np.where(upd, self.gi["coral_station"], goal)
        xy = np.where(upd[..., None], xy_s, xy); yaw = np.where(upd, np.arctan2(-nk[..., 1], -nk[..., 0]), yaw)
        # floor pieces: stand `hx + 0.07` short of the piece, facing it
        under = np.zeros(gs.coral.active.shape, bool)
        ue, uk = np.nonzero(gs.stacked >= 0)
        under[ue, gs.stacked[ue, uk]] = True
        for name, pcs, act, cap, bias, gname in (
                ("coral", gs.coral.pos, gs.coral.active & ~under, p.floor_coral & ~gs.hold_coral, 1.15, "coral_floor"),
                ("algae", gs.algae.pos, gs.algae.active, p.floor_algae & p.net & ~gs.hold_algae, 1.6, "algae_floor")):
            d = np.linalg.norm(pcs[:, None] - pos[:, :, None], axis=-1)            # (N,R,K)
            d = np.where(act[:, None], d, np.inf)
            k = np.argmin(d, -1)
            dmin = np.take_along_axis(d, k[..., None], -1)[..., 0]
            tgt = pcs[np.arange(n)[:, None], k]                                     # (N,R,2)
            vec = tgt - pos
            dirv = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-9)
            stand = tgt - dirv * (hx[..., None] + 0.07)
            cost = np.where(cap & np.isfinite(dmin), dmin * bias, np.inf)
            upd = cost < best
            best = np.where(upd, cost, best); goal = np.where(upd, self.gi[gname], goal)
            xy = np.where(upd[..., None], stand, xy); yaw = np.where(upd, np.arctan2(vec[..., 1], vec[..., 0]), yaw)
        return goal, xy, yaw, np.isfinite(best)

    # ------------------------------------------------------------------ main
    def act(self, sim, plan=None):
        pos, yaw = sim.body.pose[..., :2], sim.body.pose[..., 2]
        goal, tgt, tyaw = plan if plan is not None else self.plan(sim)
        return self._drive(sim, pos, yaw, tgt, tyaw), goal

    def plan(self, sim):
        """Goal id, approach xy and yaw for every robot."""
        g, gs, p, b = self.g, sim.gs, sim.prm, sim.body
        n = sim.cfg.n_envs
        pos, yaw, hx = b.pose[..., :2], b.pose[..., 2], p.half[..., 0]
        gi = self.gi
        goal = np.zeros((n, R), int)
        tgt = pos.copy()
        tyaw = yaw.copy()

        a_g, a_xy, a_yaw, a_ok = self._acquire_target(sim, pos, hx)
        s_g, s_xy, s_yaw, s_ok = self._score_target(sim, pos, hx)
        # default: acquire
        goal, tgt, tyaw = np.where(a_ok, a_g, 0), np.where(a_ok[..., None], a_xy, tgt), np.where(a_ok, a_yaw, tyaw)
        # holding coral and a free branch exists -> score
        m = gs.hold_coral & s_ok
        goal, tgt, tyaw = np.where(m, s_g, goal), np.where(m[..., None], s_xy, tgt), np.where(m, s_yaw, tyaw)
        # holding algae -> net (blue: own side of the barge, y in own net range)
        net_y = g.net_y[ALLIANCE].mean(-1)
        side = np.where(ALLIANCE == 0, -1.0, 1.0)
        net_xy = np.stack([g.mid[0] + side * np.mean(g.model["net_zone_dx_m"]) * np.ones(R), net_y], -1)[None].repeat(n, 0)
        m = gs.hold_algae & p.net & ~(gs.hold_coral & s_ok)
        goal = np.where(m, gi["net"], goal)
        tgt = np.where(m[..., None], net_xy, tgt)
        tyaw = np.where(m, np.where(ALLIANCE == 0, 0.0, np.pi), tyaw)
        # endgame: park in the own barge zone
        zone_y = g.mid[1] + np.where(ALLIANCE == 0, 1.0, -1.0) * (0.8 + 1.0 * (np.arange(R) % 3))
        park_xy = np.stack([g.mid[0] + side * (g.bz_half + hx.mean() - 0.12), zone_y], -1)[None].repeat(n, 0)
        park = (sim.t >= sim.end_t - self.park_s)[:, None] & np.ones((n, R), bool)
        goal = np.where(park, 0, goal)
        tgt = np.where(park[..., None], park_xy, tgt)
        tyaw = np.where(park, 0.0, tyaw)
        goal = np.where(gs.climbed, 0, goal)
        return goal, tgt, tyaw

    def _drive(self, sim, pos, yaw, tgt, tyaw):
        p, g = sim.prm, self.g
        n = pos.shape[0]
        vmax = p.max_v
        err = tgt - pos
        dist = np.linalg.norm(err, axis=-1, keepdims=True)
        v = err / np.maximum(dist, 1e-9) * np.minimum(vmax[..., None], np.minimum(self.kp * dist, np.sqrt(2 * self.accel * dist)))
        att = np.clip((dist - 0.35) / 1.1, 0.0, 1.0)                              # ignore obstacles at the goal
        # robots
        d = pos[:, :, None] - pos[:, None]                                         # (N,R,R,2) from j to i
        dd = np.linalg.norm(d, axis=-1)
        rr = (p.half[..., 0][:, :, None] + p.half[..., 0][:, None]) + 0.35
        w = np.clip((rr - dd) / 0.35, 0, 1) * (1 - np.eye(R))
        near = np.clip((dist - 0.25) / 0.9, 0.12, 1.0)                           # keep a little when docking
        v += ((d / np.maximum(dd[..., None], 1e-9)) * w[..., None]).sum(2) * vmax[..., None] * 1.2 * near
        # static obstacles
        d = pos[:, :, None] - self.obst_c[None, None]
        dd = np.linalg.norm(d, axis=-1)
        rr = self.obst_r[None, None] + p.half[..., 0][..., None] * 0.6
        w = np.clip((rr - dd) / 0.5, 0, 1)
        v += ((d / np.maximum(dd[..., None], 1e-9)) * w[..., None]).sum(2) * vmax[..., None] * att * 0.8
        sp = np.linalg.norm(v, axis=-1, keepdims=True)
        v = v / np.maximum(sp / vmax[..., None], 1.0)
        # heading: face the target yaw when close, otherwise the direction of travel
        trav = np.arctan2(v[..., 1], v[..., 0])
        far = (dist[..., 0] > 1.2)
        want = np.where(far & (sp[..., 0] > 0.3), trav, tyaw)
        # near the target the yaw must be exact; slow translation while the heading error is large
        herr = geo.wrap(want - yaw)
        v *= np.clip(1.4 - np.abs(herr) / 1.2, 0.25, 1.0)[..., None] * np.where(dist < 0.7, 1.0, 1.0)
        om = np.clip(3.0 * herr, -0.8 * p.max_w, 0.8 * p.max_w)
        flip = np.where(ALLIANCE == 0, 1.0, -1.0)[None, :, None]
        stick = np.concatenate([flip * v / p.max_v[..., None], (om / p.max_w)[..., None]], -1)
        return np.clip(stick, -1.0, 1.0)
