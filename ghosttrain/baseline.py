"""Rule-based baseline policy (privileged state, vectorised over envs x robots).

Pure state machine, no learning. It is the denominator of the win-rate metric, a league opponent and an
end-to-end test of the simulator rules. Output is the same as every policy: stick (N,R,3), goal (N,R).

Priority per robot:  endgame -> climb / park;  holding algae -> processor (if capable) else net;
holding coral -> highest level that still has a free branch on the nearest own reef face;
otherwise the cheapest of: coral station (stock left) / floor coral / floor algae / reef algae.
Steering: P-controller to an approach pose with a speed profile, repulsion from other robots and obstacles.
"""
from __future__ import annotations

import numpy as np

from ghostsim import geometry as geo
from ghostsim.robots import ALLIANCE, R

LV = ("score_L1", "score_L2", "score_L3", "score_L4")


class Baseline:
    def __init__(self, game, park_s: float = 14.0, kp: float = 2.2, accel: float = 3.0,
                 algae_floor_bias: float = 1.0, algae_reef_bias: float = 1.0):
        self.g, self.park_s, self.kp, self.accel = game, park_s, kp, accel
        self.algae_floor_bias, self.algae_reef_bias = algae_floor_bias, algae_reef_bias
        self.gi = {n: i for i, n in enumerate(game.goals)}
        obst = [game.polys.center[i] for i in range(len(game.polys))]
        rad = [1.55, 1.55, 0.75] + [0.45] * 6
        self.obst_c, self.obst_r = np.array(obst), np.array(rad)
        # keep-out zones: OPPONENT cages are off-limits in TELEOP (G418, MAJOR FOUL per contact)
        self.opp_cage_xy = game.cage_xy[np.arange(6).reshape(2, 3)[1 - ALLIANCE]]          # (R,3,2) per robot slot
        self.cage_half_diag = game.cfg["field"]["cages"]["size"] * 0.0254 / 2 * np.sqrt(2)
        self.keepout_margin = 0.10

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
        # reef algae (robots with algae_reef): stand at the nearest own face that still has algae, as for scoring
        ra = gs.reef_algae[env, al]                                                  # (N,R,6)
        fR, nR = g.reef_fc[ALLIANCE], g.reef_n[ALLIANCE]                              # (R,6,2)
        appR = fR[None] + nR[None] * (hx[..., None, None] + 0.10)                     # (N,R,6,2)
        dR = np.linalg.norm(appR - pos[:, :, None], axis=-1)
        dR = np.where(ra & (p.algae_reef & ~gs.hold_algae)[..., None], dR, np.inf)
        k = np.argmin(dR, -1)
        dmin = np.take_along_axis(dR, k[..., None], -1)[..., 0]
        xyR = np.take_along_axis(appR, k[..., None, None].repeat(2, -1), 2)[:, :, 0]
        nk = nR[None].repeat(n, 0)[np.arange(n)[:, None], np.arange(R)[None], k]
        cost = np.where(np.isfinite(dmin), dmin * self.algae_reef_bias, np.inf)
        upd = cost < best
        best = np.where(upd, cost, best); goal = np.where(upd, self.gi["algae_reef"], goal)
        xy = np.where(upd[..., None], xyR, xy); yaw = np.where(upd, np.arctan2(-nk[..., 1], -nk[..., 0]), yaw)
        # floor pieces: stand `hx + 0.07` short of the piece, facing it
        under = np.zeros(gs.coral.active.shape, bool)
        ue, uk = np.nonzero(gs.stacked >= 0)
        under[ue, gs.stacked[ue, uk]] = True
        for name, pcs, act, cap, bias, gname in (
                ("coral", gs.coral.pos, gs.coral.active & ~under, p.floor_coral & ~gs.hold_coral, 1.15, "coral_floor"),
                ("algae", gs.algae.pos, gs.algae.active, p.floor_algae & (p.net | p.processor) & ~gs.hold_algae,
                 self.algae_floor_bias, "algae_floor")):
            d = np.linalg.norm(pcs[:, None] - pos[:, :, None], axis=-1)            # (N,R,K)
            d = np.where(act[:, None], d, np.inf)
            k = np.argmin(d, -1)
            dmin = np.take_along_axis(d, k[..., None], -1)[..., 0]
            tgt = pcs[np.arange(n)[:, None], k]                                     # (N,R,2)
            vec = tgt - pos
            dirv = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-9)
            stand = tgt - dirv * (hx[..., None] + 0.07)
            bad = self._in_keepout(stand, hx) | self._in_keepout(tgt, hx)          # piece itself next to a cage
            cost = np.where(cap & np.isfinite(dmin) & ~bad, dmin * bias, np.inf)
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
        # auto-align: approach with the align goal (the controller in ghostsim.align drives the stick), then fire the
        # best level free on the aligned branch once it is aligned
        from ghostsim.align import aligned_goal
        ag = aligned_goal(sim)
        use = m & gs.aligned & (ag >= 0)
        goal = np.where(use, ag, goal)
        # trough (L1) has no branch lateral tolerance: it scores directly, no alignment needed
        need_align = m & ~use & (s_g != gi["score_L1"]) & bool(g.model.get("auto_align", False))
        goal = np.where(need_align, g.gi["align_score"], goal)
        # holding algae: processor (6 pts, 0.5 s) if the robot can, else net (4 pts, 1.8 s).
        # A robot that holds a coral with a free branch scores the coral first (checked above).
        hold_c = gs.hold_coral & s_ok
        pc, pn = g.proc_c[ALLIANCE], g.proc_n[ALLIANCE]                              # (R,2)
        proc_xy = pc[None] + pn[None] * (hx[..., None] + 0.10)                        # (N,R,2)
        proc_yaw = np.broadcast_to(np.arctan2(-pn[:, 1], -pn[:, 0]), (n, R))
        m_proc = gs.hold_algae & p.processor & ~hold_c
        goal = np.where(m_proc, gi["processor"], goal)
        tgt = np.where(m_proc[..., None], proc_xy, tgt)
        tyaw = np.where(m_proc, proc_yaw, tyaw)
        net_y = g.net_y[ALLIANCE].mean(-1)                                           # (R,)
        side = np.where(ALLIANCE == 0, -1.0, 1.0)
        net_xy = np.stack([g.mid[0] + side * np.mean(g.model["net_zone_dx_m"]) * np.ones(R), net_y], -1)[None].repeat(n, 0)
        m_net = gs.hold_algae & p.net & ~hold_c & ~m_proc
        goal = np.where(m_net, gi["net"], goal)
        tgt = np.where(m_net[..., None], net_xy, tgt)
        tyaw = np.where(m_net, np.where(ALLIANCE == 0, 0.0, np.pi), tyaw)
        # endgame: park in the own barge zone
        zone_y = g.mid[1] + np.where(ALLIANCE == 0, 1.0, -1.0) * (0.8 + 1.0 * (np.arange(R) % 3))
        park_xy = np.stack([g.mid[0] + side * (g.bz_half + hx.mean() - 0.12), zone_y], -1)[None].repeat(n, 0)
        endg = (sim.t >= sim.end_t - self.park_s)[:, None] & np.ones((n, R), bool)
        # endgame: hang on a free own cage if this robot's climb capability fits the cage (6 / 12 pts, vs 2 for park)
        c_g, c_xy, c_yaw, c_ok = self._climb_target(sim, pos, hx)
        climb = endg & c_ok
        goal = np.where(climb, c_g, goal)
        tgt = np.where(climb[..., None], c_xy, tgt)
        tyaw = np.where(climb, c_yaw, tyaw)
        park = endg & ~climb
        goal = np.where(park, 0, goal)
        tgt = np.where(park[..., None], park_xy, tgt)
        tyaw = np.where(park, 0.0, tyaw)
        goal = np.where(gs.climbed, 0, goal)
        return goal, tgt, tyaw

    def _climb_target(self, sim, pos, hx):
        """Free own cage that the robot's climb capability can use (rules: deep needs climb 2, shallow needs climb 1).
        Approach from the side the robot is on, at hx + 0.1 m from the cage centre (cage_approach_m is 0.25)."""
        g, gs, p = self.g, sim.gs, sim.prm
        n = pos.shape[0]
        env = np.arange(n)[:, None, None]
        own = np.arange(6).reshape(2, 3)[ALLIANCE]                                  # (R,3) own cage ids
        cc = g.cage_xy[own][None].repeat(n, 0)                                      # (N,R,3,2)
        free = gs.cage_occ[env, own[None]] < 0                                      # (N,R,3)
        deep = gs.cage_deep[env, own[None]]
        cap = p.climb[..., None]
        ok = free & np.where(deep, cap == 2, cap == 1)
        side = np.where(pos[:, :, None, 0] < cc[..., 0], -1.0, 1.0)
        app = cc + np.stack([side * (hx[..., None] + 0.1), np.zeros_like(side)], -1)
        dist = np.where(ok, np.linalg.norm(app - pos[:, :, None], axis=-1), np.inf)
        k = np.argmin(dist, -1)
        xy = np.take_along_axis(app, k[..., None, None].repeat(2, -1), 2)[:, :, 0]
        cxy = np.take_along_axis(cc, k[..., None, None].repeat(2, -1), 2)[:, :, 0]
        yaw = np.arctan2(cxy[..., 1] - xy[..., 1], cxy[..., 0] - xy[..., 0])
        goal = np.full((n, R), self.gi["climb"])
        return goal, xy, yaw, ok.any(-1)

    def _keepout_radius(self, hx):
        """Centre-to-cage distance below which a robot (circumscribed radius hx*sqrt2) would touch an opponent cage."""
        return self.cage_half_diag + hx * np.sqrt(2) + self.keepout_margin

    def _in_keepout(self, xy, hx):
        """xy (N,R,2) points, hx (N,R): True if a point is inside any opponent-cage keep-out disc. Returns (N,R)."""
        d = xy[:, :, None] - self.opp_cage_xy[None]                                      # (N,R,3,2)
        dd = np.linalg.norm(d, axis=-1)
        return (dd < self._keepout_radius(hx)[..., None]).any(-1)

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
        # hard keep-out around opponent cages: push out, and remove any velocity that drives into the disc
        hx = p.half[..., 0]
        rk = self._keepout_radius(hx)                                                   # (N,R)
        dc = pos[:, :, None] - self.opp_cage_xy[None]                                   # (N,R,3,2)
        dcn = np.linalg.norm(dc, axis=-1)
        u = dc / np.maximum(dcn[..., None], 1e-9)
        pen = np.clip((rk[..., None] + 0.35 - dcn) / 0.35, 0.0, 1.0)                    # 0 far, 1 at the boundary
        v += (u * pen[..., None]).sum(2) * vmax[..., None] * 2.0
        inside = (dcn < rk[..., None] + 0.05)
        for k in range(3):
            into = (v * u[:, :, k]).sum(-1, keepdims=True) < 0
            cut = inside[:, :, k, None] & into
            v = np.where(cut, v - (v * u[:, :, k]).sum(-1, keepdims=True) * u[:, :, k], v)
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
