"""Teacher observation (privileged, ego-centric, alliance-canonical).

The field is 180-degree rotationally symmetric, so every viewer sees the world as if it were blue
(red: p -> [L, W] - p, angles + pi). Features are entity sets so the network does not depend on slot order:

  self      (N,R,S)        own state, capabilities, current goal one-hot, timers, scores, filtered stick command
  others    (N,R,5,Fo)     2 allies then 3 opponents, in the ego frame; Fo = 23 features + 13 intent
                           probabilities (allies only, from ghosttrain/intent.py, zeros for opponents)
  pieces    (N,R,30,Fp)    12 floor coral + 18 floor algae slots (+ mask)
  faces     (N,R,12,Ff)    own 6 reef faces then opponent 6 (occupancy, algae, relative pose)
  landmarks (N,R,15,Fl)    4 stations, 2 processors, 2 net zones, 6 cages, barge support
  glob      (N,2,G)        centralised critic input, absolute canonical frame, one per alliance
  gmask     (N,R,13)       which goals this robot can do (capabilities)

The reef state in `faces` can come from a ReefReport (the training and evaluation workers use one): a person next to
the field relays it, so the actor sees it about once a second, 0.5-1 s late, with a few cells wrong. The critic
(`glob`) keeps the true state.
"""
from __future__ import annotations

import numpy as np

from ghostsim import geometry as geo
from ghostsim.robots import ALLIANCE, R

S, FP, FF, FL = 51, 6, 15, 12                                # S: self features incl. 14-goal one-hot + 4 align + 3 cmd
FO_RAW = 23                                       # per-other features before the intent block
N_GOALS = 14                                      # game.json goals (checked in Encoder): 13 + align_score
FO = FO_RAW + N_GOALS                             # 36
N_OTH, N_PC, N_FACE, N_LM = 5, 30, 12, 15
OTHERS = np.array([[q for q in range(R) if q != r and ALLIANCE[q] == ALLIANCE[r]] +
                   [q for q in range(R) if ALLIANCE[q] != ALLIANCE[r]] for r in range(R)])      # (R,5)


def _ego(v, c, s):
    """world->ego rotation of vectors v (...,2) with cos/sin of the viewer yaw."""
    return np.stack([v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c], -1)


class ReefReport:
    """Reef state as relayed by an observer: updated every `period_s` (random phase per env) with the state of
    `delay_s` ago (drawn per env and episode), each branch cell, trough count and algae flag wrong with probability
    `err_p` per update (a wrong trough count is off by one). Call step(sim) once per policy tick (before encoding)
    and reset(mask) when envs reset."""

    def __init__(self, n, l1_cap, rng, tick_s=0.1, period_s=1.0, delay_s=(0.5, 1.0), err_p=0.02):
        self.n, self.l1_cap, self.rng = n, l1_cap, rng
        self.period = max(1, int(round(period_s / tick_s)))
        self.dmin, self.dmax = (int(round(d / tick_s)) for d in delay_s)
        self.err_p = err_p
        self.hist = None                                   # ring of true states, length dmax + 1
        self.t = np.zeros(n, int)
        self.delay = np.zeros(n, int)
        self.phase = np.zeros(n, int)
        self.view = None
        self._fresh = np.ones(n, bool)

    def reset(self, mask=None):
        m = np.ones(self.n, bool) if mask is None else np.asarray(mask, bool)
        self._fresh |= m

    def _state(self, gs):
        return {"branch": gs.branch.copy(), "l1": gs.l1.copy(), "reef_algae": gs.reef_algae.copy()}

    def step(self, sim):
        st = self._state(sim.gs)
        if self.hist is None:
            self.hist = [st] * (self.dmax + 1)
            self.view = {k: v.copy() for k, v in st.items()}
        self.hist = self.hist[1:] + [st]
        f = self._fresh
        if f.any():                                        # a new match: the observer starts from the true state
            k = int(f.sum())
            self.t[f] = 0
            self.delay[f] = self.rng.integers(self.dmin, self.dmax + 1, k)
            self.phase[f] = self.rng.integers(0, self.period, k)
            for key in st:
                self.view[key][f] = st[key][f]
                for h in self.hist:
                    h[key][f] = st[key][f]
            self._fresh = np.zeros(self.n, bool)
        upd = (self.t + self.phase) % self.period == 0
        self.t += 1
        if not upd.any():
            return
        for e in np.flatnonzero(upd):
            old = self.hist[-1 - self.delay[e]]
            br, l1, alg = old["branch"][e].copy(), old["l1"][e].copy(), old["reef_algae"][e].copy()
            br ^= self.rng.random(br.shape) < self.err_p
            alg ^= self.rng.random(alg.shape) < self.err_p
            wrong = self.rng.random(l1.shape) < self.err_p
            l1 = np.where(wrong, np.clip(l1 + self.rng.choice([-1, 1], l1.shape), 0, self.l1_cap), l1)
            self.view["branch"][e], self.view["l1"][e], self.view["reef_algae"][e] = br, l1, alg


class Encoder:
    def __init__(self, game, intent_weights="default"):
        """intent_weights: npz path, None (zeros; used while collecting intent data), or 'default'."""
        from .intent import WEIGHTS, load_weights
        self.g = game
        self.intent = load_weights(WEIGHTS if intent_weights == "default" else intent_weights) \
            if intent_weights is not None else None
        self.last_teammate = None                         # (N,R,2,25) teammate features of the last encode()
        L, W = game.length, game.width
        self.LW = np.array([L, W])
        self.G = self._glob_dim()
        mx, my = game.mid
        dx = np.mean(game.model["net_zone_dx_m"])
        # world-frame static landmarks per alliance: (2, 15, 2) position, normal, scalar-type
        self.lm_pos = np.zeros((2, N_LM, 2)); self.lm_nrm = np.zeros((2, N_LM, 2)); self.lm_type = np.zeros((N_LM, 5))
        for a in range(2):
            o = 1 - a
            for sl, side in ((slice(0, 2), a), (slice(2, 4), o)):             # sorted by canonical y: same order for both viewers
                cy = game.station_c[side][:, 1] if a == 0 else W - game.station_c[side][:, 1]
                k = np.argsort(cy)
                self.lm_pos[a, sl], self.lm_nrm[a, sl] = game.station_c[side][k], game.station_n[side][k]
            self.lm_pos[a, 4], self.lm_nrm[a, 4] = game.proc_c[a], game.proc_n[a]
            self.lm_pos[a, 5], self.lm_nrm[a, 5] = game.proc_c[o], game.proc_n[o]
            self.lm_pos[a, 6] = [mx + (-1 if a == 0 else 1) * dx, game.net_y[a].mean()]
            self.lm_pos[a, 7] = [mx + (-1 if o == 0 else 1) * dx, game.net_y[o].mean()]
            self.lm_pos[a, 8:11] = game.cage_xy[3 * a:3 * a + 3]
            self.lm_pos[a, 11:14] = game.cage_xy[3 * o:3 * o + 3]
            self.lm_pos[a, 14] = game.mid
        self.lm_type[0:4, 0] = 1; self.lm_type[4:6, 1] = 1; self.lm_type[6:8, 2] = 1
        self.lm_type[8:14, 3] = 1; self.lm_type[14, 4] = 1
        self.lm_owner = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0], float)
        self.goal_names = game.goals
        gi = {n: i for i, n in enumerate(game.goals)}
        self.gi = gi
        self.n_goals = len(game.goals)
        assert self.n_goals == N_GOALS, self.n_goals

    def _glob_dim(self):
        return R * 21 + 2 * 6 * 8 + 10 + 30 * 3 + 6 + 3 + 2

    # ------------------------------------------------------------------ world -> viewer helpers
    def _flip_pts(self, p, flip):
        return np.where(flip, self.LW - p, p)

    def goal_mask(self, prm):
        n = prm.mass.shape[0]
        m = np.zeros((n, R, self.n_goals), bool)
        gi = self.gi
        m[..., gi["idle"]] = True
        m[..., gi["defend"]] = True
        m[..., gi["coral_station"]] = prm.station
        m[..., gi["coral_floor"]] = prm.floor_coral
        m[..., gi["algae_floor"]] = prm.floor_algae
        m[..., gi["algae_reef"]] = prm.algae_reef
        for i in range(4):
            m[..., gi[f"score_L{i + 1}"]] = prm.levels[..., i]
        m[..., gi["processor"]] = prm.processor
        m[..., gi["net"]] = prm.net
        m[..., gi["climb"]] = prm.climb > 0
        m[..., gi["align_score"]] = (prm.levels[..., 1:].any(-1) & bool(self.g.model.get("auto_align", False)))
        return m

    def caps(self, prm):
        return np.concatenate([prm.levels.astype(np.float32), prm.station[..., None], prm.floor_coral[..., None],
                               prm.floor_algae[..., None], prm.algae_reef[..., None], prm.processor[..., None],
                               prm.net[..., None], (prm.climb / 2.0)[..., None]], -1).astype(np.float32)

    # ------------------------------------------------------------------ main
    def encode(self, sim, goal_applied, reef=None):
        """goal_applied (N,R) int: goal currently executed by each robot. reef: ReefReport (actor view of the reef)
        or None (true state)."""
        g, gs, b, prm = self.g, sim.gs, sim.body, sim.prm
        rv = reef.view if reef is not None and reef.view is not None else \
            {"branch": gs.branch, "l1": gs.l1, "reef_algae": gs.reef_algae}
        n = b.pose.shape[0]
        env = np.arange(n)
        al = ALLIANCE
        flip = (al == 1)[None, :, None]                                    # (1,R,1)
        L, W = self.LW
        pos_w, yaw_w, vel_w, om = b.pose[..., :2], b.pose[..., 2], b.vel[..., :2], b.vel[..., 2]
        pos = self._flip_pts(pos_w, flip)                                  # canonical pos of every robot (own view)
        sgn = np.where(flip, -1.0, 1.0)
        yaw = yaw_w + np.where(al == 1, np.pi, 0.0)[None]
        c, s = np.cos(yaw), np.sin(yaw)
        vel = vel_w * sgn
        vel_e = _ego(vel, c, s)
        caps = self.caps(prm)
        t_left = (sim.end_t - sim.t) / g.teleop_s
        is_auto = (sim.t < g.auto_s).astype(np.float32)
        endg = (sim.t >= sim.end_t - g.endgame_s).astype(np.float32)
        own_sc, opp_sc = gs.score[:, al], gs.score[:, 1 - al]
        need = np.take_along_axis(prm.times, goal_applied[..., None], -1)[..., 0]
        tprog = np.where(need > 0, gs.timer / np.maximum(need, 1e-6), 0.0)
        goal_oh = np.eye(self.n_goals, dtype=np.float32)[goal_applied]
        own_stock, hp_own = gs.stock[:, al], gs.hp_stock[:, al]
        selfv = np.concatenate([
            (pos / self.LW), np.stack([c, s], -1), vel_e / 5.0, (om / 10.0)[..., None],
            gs.hold_coral[..., None], gs.hold_algae[..., None], caps, (prm.max_v / 5.0)[..., None], goal_oh,
            tprog[..., None], np.broadcast_to(t_left[:, None, None], (n, R, 1)), np.broadcast_to(is_auto[:, None, None], (n, R, 1)),
            np.broadcast_to(endg[:, None, None], (n, R, 1)), own_sc[..., None] / 100.0, opp_sc[..., None] / 100.0,
            own_stock[..., None] / 60.0, hp_own[..., None] / 18.0,
            *self._align_feats(sim, pos_w, yaw_w),
            sim.cmd], -1).astype(np.float32)                 # command filter state (alliance-relative stick units)
        assert selfv.shape[-1] == S, selfv.shape

        # others
        oi = OTHERS
        o_pos = pos[:, oi]                                                 # (N,R,5,2) in the viewer's canonical frame
        o_pos = np.where(flip[..., None], self.LW - pos_w[:, oi], pos_w[:, oi])
        rel = _ego(o_pos - pos[:, :, None], c[..., None], s[..., None])
        o_yaw = yaw_w[:, oi] + np.where(al == 1, np.pi, 0.0)[None, :, None]
        dyaw = o_yaw - yaw[:, :, None]
        o_vel = _ego(vel_w[:, oi] * sgn[..., None], c[..., None], s[..., None])
        ally = np.broadcast_to((al[oi] == al[:, None]).astype(np.float32), (n, R, 5))
        hs = prm.half[:, oi, 0]
        oth = np.concatenate([rel / 10.0, np.stack([np.cos(dyaw), np.sin(dyaw)], -1), o_vel / 5.0, (om[:, oi] / 10.0)[..., None],
                              ally[..., None], gs.hold_coral[:, oi][..., None], gs.hold_algae[:, oi][..., None], hs[..., None] / 0.5,
                              caps[:, oi], (np.linalg.norm(rel, axis=-1) / 20.0)[..., None]], -1).astype(np.float32)
        assert oth.shape[-1] == FO_RAW, oth.shape
        # teammate intent: predicted goal distribution of the 2 allies (features are in this viewer's frame)
        tm = np.stack([np.broadcast_to(t_left[:, None], (n, R)), np.broadcast_to(endg[:, None], (n, R))], -1)
        tf = np.concatenate([oth[:, :, :2], np.broadcast_to(tm[:, :, None], (n, R, 2, 2))], -1).astype(np.float32)
        self.last_teammate = tf
        intent = np.zeros((n, R, 5, N_GOALS), np.float32)
        if self.intent is not None:
            from .intent import predict_np
            intent[:, :, :2] = predict_np(self.intent, tf)
        oth = np.concatenate([oth, intent], -1)
        assert oth.shape[-1] == FO, oth.shape

        # pieces
        under = np.zeros(gs.coral.active.shape, bool)
        ue, uk = np.nonzero(gs.stacked >= 0)
        under[ue, gs.stacked[ue, uk]] = True
        pcs = []
        for typ, p_, act, extra in ((0.0, gs.coral.pos, gs.coral.active, under.astype(np.float32)),
                                    (1.0, gs.algae.pos, gs.algae.active, (gs.stacked >= 0).astype(np.float32))):
            pw = np.where(flip[..., None], self.LW - p_[:, None], p_[:, None])           # (N,R,K,2)
            rl = _ego(pw - pos[:, :, None], c[..., None], s[..., None])
            k = rl.shape[2]
            f = np.concatenate([rl / 10.0, (np.linalg.norm(rl, axis=-1) / 20.0)[..., None],
                                np.full((n, R, k, 1), typ, np.float32), np.broadcast_to(extra[:, None, :, None], (n, R, k, 1)),
                                np.ones((n, R, k, 1), np.float32)], -1)
            pcs.append((f, np.broadcast_to(act[:, None], (n, R, k))))
        pieces = np.concatenate([p[0] for p in pcs], 2).astype(np.float32)
        pmask = np.concatenate([p[1] for p in pcs], 2)

        # reef faces: own 6 then opponent 6
        fc_all = np.concatenate([g.reef_fc[al], g.reef_fc[1 - al]], 1)                    # (R,12,2) world
        fn_all = np.concatenate([g.reef_n[al], g.reef_n[1 - al]], 1)
        fc_v = np.where(flip[..., None], self.LW - fc_all[None], fc_all[None])             # (N?,R,12,2) -> (1,R,12,2)
        fn_v = np.where(flip[..., None], -fn_all[None], fn_all[None])
        rel = _ego(fc_v - pos[:, :, None], c[..., None], s[..., None])
        nrm = _ego(np.broadcast_to(fn_v, rel.shape), c[..., None], s[..., None])
        br = np.concatenate([rv["branch"][:, al], rv["branch"][:, 1 - al]], 2).reshape(n, R, 12, 6)
        l1 = np.concatenate([rv["l1"][:, al], rv["l1"][:, 1 - al]], 2) / max(g.l1_cap, 1)
        alg = np.concatenate([rv["reef_algae"][:, al], rv["reef_algae"][:, 1 - al]], 2)
        hi = np.concatenate([g.algae_high, g.algae_high])
        owner = np.concatenate([np.ones(6), np.zeros(6)])
        faces = np.concatenate([rel / 10.0, nrm, (np.linalg.norm(rel, axis=-1) / 20.0)[..., None], np.broadcast_to(owner[:, None], (n, R, 12, 1)),
                                br, l1[..., None], alg[..., None], np.broadcast_to(hi[:, None], (n, R, 12, 1))], -1).astype(np.float32)
        assert faces.shape[-1] == FF, faces.shape

        # landmarks
        lp = np.where(flip[..., None], self.LW - self.lm_pos[al][None], self.lm_pos[al][None])      # (1,R,15,2)
        ln = np.where(flip[..., None], -self.lm_nrm[al][None], self.lm_nrm[al][None])
        rel = _ego(lp - pos[:, :, None], c[..., None], s[..., None])
        nrm = _ego(np.broadcast_to(ln, rel.shape), c[..., None], s[..., None])
        scal = np.zeros((n, R, N_LM))
        scal[:, :, 0:2] = gs.stock[:, al][:, :, None] / 60.0          # own stations: remaining stock
        scal[:, :, 2:4] = gs.stock[:, 1 - al][:, :, None] / 60.0
        scal[:, :, 4] = gs.processor[:, al] / 6.0
        scal[:, :, 5] = gs.processor[:, 1 - al] / 6.0
        scal[:, :, 6] = gs.net[:, al] / 6.0
        scal[:, :, 7] = gs.net[:, 1 - al] / 6.0
        cage_occ = (gs.cage_occ >= 0).astype(np.float32)
        own_cg = np.stack([cage_occ[:, 3 * a:3 * a + 3] for a in al], 1)
        opp_cg = np.stack([cage_occ[:, 3 * (1 - a):3 * (1 - a) + 3] for a in al], 1)
        scal[:, :, 8:11], scal[:, :, 11:14] = own_cg, opp_cg
        lm = np.concatenate([rel / 10.0, nrm, (np.linalg.norm(rel, axis=-1) / 20.0)[..., None],
                             np.broadcast_to(self.lm_owner[:, None], (n, R, N_LM, 1)), scal[..., None],
                             np.broadcast_to(self.lm_type, (n, R, N_LM, 5))], -1).astype(np.float32)
        assert lm.shape[-1] == FL, lm.shape
        return {"self": selfv, "others": oth, "pieces": pieces, "pmask": pmask, "faces": faces, "lm": lm,
                "glob": self._glob(sim, caps), "gmask": self.goal_mask(prm)}

    def _align_feats(self, sim, pos_w, yaw_w):
        """Auto-align state the policy needs to know it is aligned: [aligned, dwell progress, has target,
        target offset in the robot frame / 5 m]. Zeros when the robot has no target."""
        from ghostsim.robots import R as _R
        gs, g = sim.gs, self.g
        need = np.maximum(sim.prm.times[..., g.gi["align_score"]], 1e-6)
        has = (gs.align_face >= 0).astype(np.float32)
        tgt = gs.align_tgt[..., :2]
        rel = _ego(tgt - pos_w, np.cos(yaw_w), np.sin(yaw_w)) * has[..., None]
        flip_sign = 1.0 - 2.0 * (ALLIANCE == 1).astype(np.float32)                 # canonical (blue) frame
        return [gs.aligned.astype(np.float32)[..., None], np.minimum(gs.align_t / need, 1.0)[..., None].astype(np.float32),
                has[..., None], (rel * flip_sign[None, :, None] / 5.0).astype(np.float32)]

    def _glob(self, sim, caps):
        g, gs, b, prm = self.g, sim.gs, sim.body, sim.prm
        n = b.pose.shape[0]
        out = np.zeros((n, 2, self.G), np.float32)
        for a in range(2):
            order = np.array([q for q in range(R) if ALLIANCE[q] == a] + [q for q in range(R) if ALLIANCE[q] != a])
            fl = a == 1
            p = np.where(fl, self.LW - b.pose[:, order, :2], b.pose[:, order, :2])
            sg = -1.0 if fl else 1.0
            yw = b.pose[:, order, 2] + (np.pi if fl else 0.0)
            rob = np.concatenate([p / self.LW, np.stack([np.cos(yw), np.sin(yw)], -1), b.vel[:, order, :2] * sg / 5.0,
                                  (b.vel[:, order, 2] / 10.0)[..., None], gs.hold_coral[:, order][..., None],
                                  gs.hold_algae[:, order][..., None], caps[:, order],
                                  np.broadcast_to((np.arange(R) < 3).astype(np.float32)[:, None], (n, R, 1))], -1)   # (N,6,21)
            o = 1 - a
            reef = np.concatenate([
                np.concatenate([gs.branch[:, x].reshape(n, 6, 6), gs.l1[:, x][..., None] / max(g.l1_cap, 1),
                                gs.reef_algae[:, x][..., None]], -1).reshape(n, -1) for x in (a, o)], -1)
            pcs = []
            for p_, act in ((gs.coral.pos, gs.coral.active), (gs.algae.pos, gs.algae.active)):
                if fl:                      # staged pieces: slots 0-2 blue marks, 3-5 red marks -> put own marks first
                    k = p_.shape[1]
                    perm = np.r_[3, 4, 5, 0, 1, 2, np.arange(6, k)]
                    p_, act = p_[:, perm], act[:, perm]
                q = np.where(fl, self.LW - p_, p_) * act[..., None]            # unused slots stay at 0 for both viewers
                pcs.append(np.concatenate([q / self.LW, act[..., None]], -1).reshape(n, -1))
            pcs = np.concatenate(pcs, -1)
            sc = [gs.stock[:, a] / 60.0, gs.stock[:, o] / 60.0, gs.hp_stock[:, a] / 18.0, gs.hp_stock[:, o] / 18.0,
                  gs.net[:, a] / 6.0, gs.net[:, o] / 6.0, gs.processor[:, a] / 6.0, gs.processor[:, o] / 6.0,
                  np.zeros(n), np.zeros(n)]
            cg = np.concatenate([(gs.cage_occ[:, 3 * a:3 * a + 3] >= 0), (gs.cage_occ[:, 3 * o:3 * o + 3] >= 0)], -1)
            tm = np.stack([(sim.end_t - sim.t) / g.teleop_s, (sim.t < g.auto_s).astype(float),
                           (sim.t >= sim.end_t - g.endgame_s).astype(float)], -1)
            scr = np.stack([gs.score[:, a] / 100.0, gs.score[:, o] / 100.0], -1)
            vec = np.concatenate([rob.reshape(n, -1), reef, np.stack(sc, -1), pcs, cg, tm, scr], -1)
            assert vec.shape[-1] == self.G, (vec.shape, self.G)
            out[:, a] = vec
        return out
