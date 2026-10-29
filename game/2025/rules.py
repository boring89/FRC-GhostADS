"""REEFSCAPE (2025) game logic, vectorised over N envs.

Everything numeric comes from game.json; modelling assumptions live in game.json['model'].
Robots 0-2 blue, 3-5 red. The field is rotationally symmetric (red = blue rotated 180 deg).
"""
from __future__ import annotations

import numpy as np

from ghostsim import geometry as geo
from ghostsim.game import GameBase
from ghostsim.geometry import IN
from ghostsim.physics import EI, EJ, PAIRS, Pieces, piece_step
from ghostsim.robots import ALLIANCE, R

LEVELS = ("L1", "L2", "L3", "L4")


class State:
    def __init__(self, n, kc, ka):
        z, b, i = (lambda *s: np.zeros(s)), (lambda *s: np.zeros(s, bool)), (lambda *s: np.zeros(s, np.int64))
        self.hold_coral, self.hold_algae, self.climbed = b(n, R), b(n, R), b(n, R)
        self.timer, self.timer_goal = z(n, R), i(n, R)
        self.stock = i(n, 2)
        self.coral = Pieces(z(n, kc, 2), z(n, kc, 2), z(n, kc), b(n, kc))
        self.algae = Pieces(z(n, ka, 2), z(n, ka, 2), z(n, ka), b(n, ka))
        self.stacked = -np.ones((n, ka), np.int64)        # algae resting on coral slot k
        self.branch, self.branch_auto = b(n, 2, 6, 2, 3), b(n, 2, 6, 2, 3)   # [alliance, face, side, L2..L4]
        self.l1, self.l1_auto = i(n, 2, 6), i(n, 2, 6)
        self.reef_algae = b(n, 2, 6)
        self.hp_stock, self.hp_timer = i(n, 2), z(n, 2)
        self.processor, self.net = i(n, 2), i(n, 2)
        self.cage_occ, self.cage_deep = -np.ones((n, 6), np.int64), b(n, 6)
        self.score, self.fouls = z(n, 2), z(n, 2)
        # points by category per alliance: 0 LEAVE, 1 CORAL, 2 ALGAE (net + processor), 3 BARGE (park/cage), 4 PENALTY
        # (points received from opponent fouls). Sum over categories == score (tested).
        self.cat = z(n, 2, 5)
        self.prev_pair, self.prev_cage = b(n, len(PAIRS)), b(n, R, 6)
        self.g421_t = z(n, 2)
        # not game points: damage tally (own robots hit hard), teammate-contact timer (reward shaping only)
        self.damage, self.dmg_t, self.press_t = z(n, R), z(n, R), z(n, R)
        # pushing a teammate that is scoring (reward only): events per robot, and last substep's teammate contacts
        self.push_events, self.prev_tm = z(n, R), b(n, len(PAIRS))
        # auto-align (ghostsim/align.py): target face/side, dwell timer, aligned flag
        self.align_face, self.align_side = i(n, R) - 1, i(n, R) - 1
        self.align_t, self.aligned = z(n, R), b(n, R)
        self.align_tgt = z(n, R, 3)
        self.align_stuck, self.align_last = z(n, R), z(n, R, 2)
        self.align_bad = b(n, R, 12)              # candidates given up after being stuck (re-chosen next)
        self.pair_last = np.full((n, len(PAIRS)), -1e9)   # time of last G427/G428 foul per robot pair
        # failed actions (robot.success_p): misses per robot; CORAL that found no free floor slot (left the game)
        self.miss, self.coral_lost = z(n, R), i(n)


class Game(GameBase):
    def __init__(self, cfg, tags, game_dir):
        self.cfg, self.tags, self.dir = cfg, tags, game_dir
        f, m = cfg["field"], cfg["model"]
        self.name, self.goals, self.model = cfg["name"], cfg["goals"], m
        self.gi = {g: k for k, g in enumerate(self.goals)}
        self.scoring_ids = np.array([self.gi[g] for g in ("score_L1", "score_L2", "score_L3", "score_L4", "net",
                                                          "processor", "climb")])
        self.auto_s, self.teleop_s = cfg["match"]["auto_s"], cfg["match"]["teleop_s"]
        self.endgame_s, self.start_mode = cfg["match"]["endgame_s"], cfg["match"]["start_mode"]
        self.points = cfg["points"]
        L, W = f["length"] * IN, f["width"] * IN
        self.length, self.width = L, W
        mx, my = L / 2, W / 2
        self.mid = np.array([mx, my])
        txy = lambda i: tags[i]["pos"][:2]
        tn = lambda i: np.array([np.cos(tags[i]["yaw"]), np.sin(tags[i]["yaw"])])

        # field boundary: 4 walls + coral-station chamfers through the station tags
        gi = f.get("guardrail_inset_m", 0.0)
        hn, ho = [[1, 0], [-1, 0], [0, 1], [0, -1]], [0.0, -L, gi, -(W - gi)]
        for a in ("blue", "red"):
            for i in f["coral_station_tags"][a]:
                hn.append(tn(i)); ho.append(float(tn(i) @ txy(i)))
        self.hp_n, self.hp_off = np.array(hn, float), np.array(ho)

        # reef: centre/apothem from the tags; face k normal = base + 60k (checked)
        self.reef_c, self.reef_n = np.zeros((2, 2)), np.zeros((2, 6, 2))
        aps = []
        for a, name in enumerate(("blue", "red")):
            ids = f["reef_tags"][name]
            c = np.mean([txy(i) for i in ids], 0)
            self.reef_c[a] = c
            aps += [np.linalg.norm(txy(i) - c) for i in ids]
            base = np.pi if a == 0 else 0.0
            for k, i in enumerate(ids):
                assert abs(geo.wrap(tags[i]["yaw"] - (base + k * np.pi / 3))) < 1e-6, "reef_tags order"
                self.reef_n[a, k] = tn(i)
        self.reef_ap = float(np.mean(aps))
        self.reef_t = np.stack([-self.reef_n[..., 1], self.reef_n[..., 0]], -1)
        self.reef_fc = self.reef_c[:, None] + self.reef_ap * self.reef_n
        rf = f["reef"]
        reef_poly = [geo.regular_polygon(self.reef_c[a], self.reef_ap, 6, np.degrees(np.arctan2(*self.reef_n[a, 0][::-1])))
                     for a in range(2)]
        zone_poly = [geo.regular_polygon(self.reef_c[a], self.reef_ap + rf["zone_offset"] * IN, 6,
                                         np.degrees(np.arctan2(*self.reef_n[a, 0][::-1]))) for a in range(2)]
        bg, cg, bz = f["barge"], f["cages"], f["barge_zone"]
        sup = geo.rect(mx, my, bg["center_support"][0] * IN, bg["center_support"][1] * IN)
        sgn = np.array([1, -1])                                   # blue cages at y > mid
        self.cage_xy = np.array([[mx, my + s * o * IN] for s in sgn for o in cg["offsets"]])   # (6,2) blue 0-2, red 3-5
        cages = [geo.rect(x, y, cg["size"] * IN, cg["size"] * IN) for x, y in self.cage_xy]
        self.polys = geo.Polygons(reef_poly + [sup] + cages, [0.6, 0.6, bg["height"] * IN] + [0.0] * 6)
        self.poly_ids = ["reef_blue", "reef_red", "barge_support"] + [f"cage_{k}" for k in range(6)]
        self.cage_poly = np.arange(3, 9)
        bzr = [geo.rect(mx, my + s * bz["length"] * IN / 2, bz["depth"] * IN, bz["length"] * IN) for s in sgn]
        self.zones = geo.Polygons(zone_poly + bzr)             # reef zone b/r, barge zone b/r
        self.bz_half = bz["depth"] * IN / 2

        st = f["coral_station_tags"]
        self.station_c = np.array([[txy(i) for i in st[a]] for a in ("blue", "red")])        # (2,2,2)
        self.station_n = np.array([[tn(i) for i in st[a]] for a in ("blue", "red")])
        self.station_half = f["coral_station_opening"]["width"] * IN / 2
        pt = f["processor_tags"]
        self.proc_c = np.array([txy(pt["blue"]), txy(pt["red"])])
        self.proc_n = np.array([tn(pt["blue"]), tn(pt["red"])])
        self.proc_half = f["processor_opening"]["width"] * IN / 2
        cm = f["coral_marks"]
        blue_marks = np.array([[cm["x_from_wall"] * IN, y * IN] for y in cm["y"]])
        self.marks = np.concatenate([blue_marks, [L, W] - blue_marks])                      # (6,2)
        self.start_x = np.array([f["starting_line_x_blue"] * IN, L - f["starting_line_x_blue"] * IN])
        net_len = 144 * IN
        self.net_y = np.array([[my, my + net_len], [my - net_len, my]])

        self.algae_high = np.isin(np.arange(6), m["reef_algae_high_faces"])
        blk = m["algae_blocks"]
        self.block = {k: np.array([f"L{l}" in blk[k] for l in (2, 3, 4)]) for k in ("low", "high")}
        self.l1_cap = m["l1_capacity_per_face"]
        ts = f["reef"]["trough_slots"]
        assert self.l1_cap <= ts["per_layer"] * len(ts["z"]), "l1_capacity_per_face exceeds the trough slots"
        self.stop_ids = np.array([self.gi[g] for g in m.get("stop_goals", [])], np.int64)
        pc = cfg["pieces"]
        self.kc, self.ka = pc["coral"]["floor_slots"], pc["algae"]["floor_slots"]
        self.coral_count = pc["coral"]["count"]
        self.rad = m["piece_radius_m"]
        self._reef_slots(rf)

    # ------------------------------------------------------------------ helpers
    def _reef_slots(self, rf):
        """Static 3D poses [x,y,z,yaw,pitch] of reef coral slots (render only, approximate)."""
        br, l1 = [], []
        for a in range(2):
            for k in range(6):
                n, t, fc = self.reef_n[a, k], self.reef_t[a, k], self.reef_fc[a, k]
                yaw = np.arctan2(n[1], n[0])
                for s in (-1, 1):
                    for lv in ("L2", "L3", "L4"):
                        xy = fc - n * rf["branch_inset"][lv] * IN + t * s * rf["pipe_spacing"] * IN / 2
                        if lv == "L4":
                            br.append([*xy, rf["branch_top"][lv] * IN - 0.17, yaw, np.pi / 2])
                        else:
                            xy = xy + n * 0.06
                            br.append([*xy, rf["branch_top"][lv] * IN - 0.06, yaw, np.radians(rf["branch_angle_deg"][lv])])
                ts = rf["trough_slots"]                                  # layer by layer, end to end along the face
                for j in range(self.l1_cap):
                    layer, i = divmod(j, ts["per_layer"])
                    xy = fc + n * ts["outward"][layer] * IN + t * (i - (ts["per_layer"] - 1) / 2) * ts["lateral_spacing"] * IN
                    l1.append([*xy, ts["z"][layer] * IN, yaw + np.pi / 2, 0.0])
        self.branch_pose = np.array(br).reshape(2, 6, 2, 3, 5)
        self.l1_pose = np.array(l1).reshape(2, 6, self.l1_cap, 5)
        ar = rf["algae_rest"]                                            # cradled by the L2 / L3 branch pair
        self.reef_algae_pose = np.array([[[*(self.reef_fc[a, k] + self.reef_n[a, k] * ar["outward"] * IN),
                                           ar["z"]["high" if self.algae_high[k] else "low"] * IN] for k in range(6)]
                                         for a in range(2)])

    def new_state(self, n):
        return State(n, self.kc, self.ka)

    def fixed(self, st):
        return st.climbed

    def holding(self, st):
        return np.stack([st.hold_coral, st.hold_algae], -1)

    def _phase_pts(self, val, auto):
        v = np.asarray(val)
        return np.where(auto, v[0], v[1])

    # ------------------------------------------------------------------ reset
    def _valid(self, xy, rad, placed):
        if np.any(xy @ self.hp_n.T - self.hp_off < rad + 0.05):
            return False
        d, _ = geo.circle_poly(xy, rad + 0.05, self.polys)
        if np.any(d > 0):
            return False
        return all(np.linalg.norm(xy - q) > rad + rq + 0.05 for q, rq in placed)

    def reset(self, st, idx, body, prm, rng, mode=None):
        mode = mode or self.start_mode
        mx = self.mid[0]
        for e in idx:
            placed = []
            for r in range(R):
                a = ALLIANCE[r]
                rad = float(np.hypot(*prm.half[e, r]))
                for _ in range(200):
                    if mode == "auto":
                        x = self.start_x[a] + rng.uniform(-0.15, 0.15)
                        xy, yaw = np.array([x, rng.uniform(0.6, self.width - 0.6)]), (np.pi if a == 0 else 0.0) + rng.uniform(-0.3, 0.3)
                    else:
                        lo, hi = (0.0, mx - 0.8) if a == 0 else (mx + 0.8, self.length)
                        xy, yaw = np.array([rng.uniform(lo, hi), rng.uniform(0, self.width)]), rng.uniform(-np.pi, np.pi)
                    if self._valid(xy, rad, placed):
                        break
                placed.append((xy, rad))
                body.pose[e, r] = [*xy, geo.wrap(yaw)]
        body.vel[idx] = 0
        body.az[idx] = 0
        body.wheel[idx] = 0
        body.vbatt[idx] = prm.v_oc[idx]
        k = len(idx)
        if mode == "auto":
            hold = np.ones((k, R), bool)
        else:
            hold = rng.random((k, R)) < 0.5
        st.hold_coral[idx], st.hold_algae[idx], st.climbed[idx] = hold, False, False
        st.timer[idx], st.timer_goal[idx] = 0, 0
        per_alliance = (self.coral_count - 6) // 2                       # 60 = 57 at stations + 3 preload
        st.stock[idx] = per_alliance - np.stack([hold[:, :3].sum(1), hold[:, 3:].sum(1)], 1)
        for pcs in (st.coral, st.algae):
            pcs.vel[idx] = 0
            pcs.active[idx] = False
        st.coral.pos[idx, :6], st.coral.active[idx, :6] = self.marks, True
        st.coral.yaw[idx, :6] = np.pi / 2                                # covering the mark, axis along y
        st.algae.pos[idx, :6], st.algae.active[idx, :6] = self.marks, True
        st.stacked[idx] = -1
        st.stacked[idx, :6] = np.arange(6)
        st.branch[idx] = st.branch_auto[idx] = False
        st.l1[idx] = st.l1_auto[idx] = 0
        st.reef_algae[idx] = True
        st.hp_stock[idx], st.hp_timer[idx] = 0, 0
        st.processor[idx], st.net[idx] = 0, 0
        st.cage_occ[idx] = -1
        st.cage_deep[idx] = prm.climb[idx] != 1                          # cage k <-> driver station k
        st.score[idx], st.fouls[idx] = 0, 0
        st.cat[idx] = 0
        st.prev_pair[idx], st.prev_cage[idx], st.g421_t[idx] = False, False, 0
        st.damage[idx], st.dmg_t[idx], st.press_t[idx] = 0, 0, 0
        st.push_events[idx], st.prev_tm[idx] = 0, False
        st.align_face[idx], st.align_side[idx] = -1, -1
        st.align_t[idx], st.aligned[idx], st.align_tgt[idx] = 0, False, 0
        st.align_stuck[idx], st.align_last[idx], st.align_bad[idx] = 0, 0, False
        st.pair_last[idx] = -1e9
        st.miss[idx], st.coral_lost[idx] = 0, 0

    # ------------------------------------------------------------------ geometry tests
    def _at_faces(self, pos, fwd, hx, fc, fn, half_w, facing):
        """pos/fwd (N,R,2), fc/fn (R,F,2) -> face index (N,R) (-1 none), and for that face: lateral offset, gap
        (bumper to face) and heading error [rad], each (N,R)."""
        m = self.model
        rel = pos[:, :, None] - fc[None]
        gap = (rel * fn).sum(-1) - hx[..., None]
        t = np.stack([-fn[..., 1], fn[..., 0]], -1)
        lat = (rel * t).sum(-1)
        head = -(fwd[:, :, None] * fn).sum(-1) * facing[..., None]
        ok = (gap > -0.05) & (gap <= m["approach_m"]) & (np.abs(lat) <= half_w) & (head >= np.cos(np.radians(m["heading_tol_deg"])))
        score = np.where(ok, np.abs(lat), np.inf)
        f = np.argmin(score, -1)
        pick = lambda x: np.take_along_axis(x, f[..., None], -1)[..., 0]
        return np.where(ok.any(-1), f, -1), pick(lat), pick(gap), np.arccos(np.clip(pick(head), -1.0, 1.0))

    def _in_intake(self, pp, body, prm, radius):
        """Pieces (N,K,2) of the given radius inside each robot's front intake (N,R,K). The reach is measured to the
        near edge of the piece: a piece resting against the bumper is inside."""
        d = pp[:, None] - body.pose[:, :, None, :2]
        th = body.pose[..., 2][..., None]
        lx = d[..., 0] * np.cos(th) + d[..., 1] * np.sin(th)
        ly = -d[..., 0] * np.sin(th) + d[..., 1] * np.cos(th)
        hx = prm.half[..., 0][..., None]
        return (lx >= hx - 0.15) & (lx <= hx + prm.intake_depth[..., None] + radius) & \
            (np.abs(ly) <= prm.intake_half_width[..., None])

    def _spawn_algae(self, st, e, xy, vel):
        """Put one algae on the floor in env e (first free slot)."""
        free = np.flatnonzero(~st.algae.active[e])
        if len(free):
            k = free[0]
            st.algae.pos[e, k], st.algae.vel[e, k], st.algae.active[e, k], st.stacked[e, k] = xy, vel, True, -1

    def _spawn_coral(self, st, e, xy, vel, yaw):
        """Put one coral on the floor in env e (first free slot that no algae rests on). No free slot: the coral
        leaves the game (counted in st.coral_lost)."""
        on = np.zeros(self.kc, bool)
        on[st.stacked[e][st.stacked[e] >= 0]] = True
        free = np.flatnonzero(~st.coral.active[e] & ~on)
        if len(free):
            k = free[0]
            st.coral.pos[e, k], st.coral.vel[e, k], st.coral.yaw[e, k], st.coral.active[e, k] = xy, vel, yaw, True
        else:
            st.coral_lost[e] += 1

    def _beside(self, body, prm, e, r, radius, rng):
        """Floor point just left or right of robot r (a missed piece falls next to it) and a small velocity."""
        th = body.pose[e, r, 2]
        side = np.array([-np.sin(th), np.cos(th)]) * rng.choice([-1.0, 1.0])
        xy = body.pose[e, r, :2] + side * (prm.half[e, r, 1] + radius + self.model["miss_spawn_gap_m"])
        return xy, side * rng.uniform(0.1, 0.4) + body.vel[e, r, :2]

    # ------------------------------------------------------------------ step
    def substep(self, st, body, prm, goal, t, live, dt, rng):
        n, m, gi = len(t), self.model, self.gi
        auto = t < self.auto_s
        env = np.arange(n)
        pts, foul, bonus = np.zeros((n, R)), np.zeros((n, R)), np.zeros((n, 2))
        pos, th = body.pose[..., :2], body.pose[..., 2]
        cs, sn = np.cos(th), np.sin(th)
        fwd = np.stack([cs, sn], -1)
        hx = prm.half[..., 0]

        # ---- floor pieces (stacked algae ride on their coral until a robot touches them)
        piece_step(st.coral, self.rad["coral"], m["piece_decel_mps2"]["coral"], m["piece_restitution"]["coral"],
                   body, prm.half, self, dt)
        stk = st.stacked >= 0
        if stk.any():
            par = np.maximum(st.stacked, 0)
            st.algae.pos = np.where(stk[..., None], np.take_along_axis(st.coral.pos, par[..., None].repeat(2, -1), 1), st.algae.pos)
            d, _ = geo.circle_box(st.algae.pos, self.rad["coral"] + 0.05, pos, prm.half, cs, sn)
            touch = (d > 0) & stk[:, None]
            knock = touch.any(1)
            if knock.any():
                rv = np.take_along_axis(body.vel[..., :2], np.argmax(touch, 1)[..., None].repeat(2, -1), 1)
                st.algae.vel = np.where(knock[..., None], rv * 1.2 + rng.normal(0, 0.3, rv.shape), st.algae.vel)
                st.stacked[knock] = -1
        act = st.algae.active.copy()
        st.algae.active &= st.stacked < 0
        piece_step(st.algae, self.rad["algae"], m["piece_decel_mps2"]["algae"], m["piece_restitution"]["algae"],
                   body, prm.half, self, dt)
        st.algae.active = act

        # ---- floor intakes: a piece inside the intake of a robot that is intaking it moves with that robot (the
        # rollers hold it) instead of bouncing off the bumper, so the dwell timer can complete on the move
        in_c = self._in_intake(st.coral.pos, body, prm, self.rad["coral"])
        under = np.zeros(st.coral.active.shape, bool)
        ue, uk = np.nonzero(st.stacked >= 0)
        under[ue, st.stacked[ue, uk]] = True
        cfloor = in_c & (st.coral.active & ~under)[:, None]
        afloor = self._in_intake(st.algae.pos, body, prm, self.rad["algae"]) & st.algae.active[:, None]
        for pcs, inside, grab in ((st.coral, cfloor, (goal == gi["coral_floor"]) & ~st.hold_coral & prm.floor_coral),
                                  (st.algae, afloor & (st.stacked < 0)[:, None], (goal == gi["algae_floor"]) & ~st.hold_algae & prm.floor_algae)):
            e_, r_, k_ = np.nonzero(inside & grab[..., None])
            if len(e_):
                rel = pcs.pos[e_, k_] - body.pose[e_, r_, :2]
                w = body.vel[e_, r_, 2]
                pcs.vel[e_, k_] = body.vel[e_, r_, :2] + np.stack([-w * rel[:, 1], w * rel[:, 0]], -1)

        # ---- mechanism predicates
        goal = np.where(st.climbed, gi["idle"], goal)
        al = ALLIANCE[None, :].repeat(n, 0)
        face, lat, gap_r, ang_r = self._at_faces(pos, fwd, hx, self.reef_fc[ALLIANCE], self.reef_n[ALLIANCE],
                                                 m["lateral_tol_m"], np.ones((n, R)))
        f0 = np.maximum(face, 0)
        at_reef = face >= 0
        sta, _, gap_s, ang_s = self._at_faces(pos, fwd, hx, self.station_c[ALLIANCE], self.station_n[ALLIANCE],
                                              self.station_half, prm.station_side)
        prc, lat_p, gap_p, ang_p = self._at_faces(pos, fwd, hx, self.proc_c[ALLIANCE][:, None], self.proc_n[ALLIANCE][:, None],
                                                  self.proc_half + 0.1, np.ones((n, R)))
        dxm = np.abs(pos[..., 0] - self.mid[0])
        ny = self.net_y[ALLIANCE]
        net_ok = (dxm >= m["net_zone_dx_m"][0]) & (dxm <= m["net_zone_dx_m"][1]) & \
            (pos[..., 1] >= ny[:, 0]) & (pos[..., 1] <= ny[:, 1])
        own_cages = np.arange(6).reshape(2, 3)[ALLIANCE]                                    # (R,3)
        cdist = np.linalg.norm(pos[:, :, None] - self.cage_xy[own_cages][None], axis=-1)     # (N,R,3)
        cage_ok = (cdist <= hx[..., None] + m["cage_approach_m"]) & (st.cage_occ[env[:, None, None], own_cages[None]] < 0)
        deep = st.cage_deep[env[:, None, None], own_cages[None]]
        cage_ok &= np.where(deep, prm.climb[..., None] == 2, prm.climb[..., None] == 1)
        cage_sel = own_cages[None].repeat(n, 0)[env[:, None], np.arange(R)[None], np.argmax(cage_ok, -1)]
        ralg = st.reef_algae[env[:, None], al, f0]
        hi = self.algae_high[f0]
        blocked = ralg[..., None] & np.where(hi[..., None], self.block["high"], self.block["low"])     # (N,R,3)
        free = ~st.branch[env[:, None], al, f0] & ~blocked[..., None, :]                                 # (N,R,2,3)
        l1_ok = st.l1[env[:, None], al, f0] < self.l1_cap
        # branch lateral tolerance: side index 0 is -t, side index 1 is +t (see branch layout in _build_scene)
        half_pipe = self.cfg["field"]["reef"]["pipe_spacing"] * IN / 2
        lat_b = lat[..., None] - np.array([-half_pipe, half_pipe])                       # (N,R,2)
        tol_b = m["branch_lateral_tol_m"] if m.get("auto_align", False) else m["lateral_tol_m"]
        okb = np.abs(lat_b) <= tol_b                       # auto-align off: the original 0.4 m tolerance

        c = np.zeros((n, R), bool)
        sel = lambda g: goal == gi[g]
        c |= sel("coral_station") & ~st.hold_coral & prm.station & (sta >= 0) & (st.stock[env[:, None], al] > 0)
        c |= sel("coral_floor") & ~st.hold_coral & prm.floor_coral & cfloor.any(-1)
        c |= sel("algae_floor") & ~st.hold_algae & prm.floor_algae & afloor.any(-1)
        c |= sel("algae_reef") & prm.algae_reef & at_reef & ralg
        c |= sel("score_L1") & st.hold_coral & prm.levels[..., 0] & at_reef & l1_ok
        for li in range(3):
            c |= sel(f"score_L{li + 2}") & st.hold_coral & prm.levels[..., li + 1] & at_reef & (free[..., li] & okb).any(-1)
        c |= sel("processor") & st.hold_algae & prm.processor & (prc >= 0)
        c |= sel("net") & st.hold_algae & prm.net & net_ok
        c |= sel("climb") & ~auto[:, None] & (prm.climb > 0) & cage_ok.any(-1)
        c &= live[:, None] & ~st.climbed
        # placing / receiving a piece needs the robot (nearly) stopped
        slow = (np.linalg.norm(body.vel[..., :2], axis=-1) <= m.get("stop_speed_mps", np.inf)) & \
            (np.abs(body.vel[..., 2]) <= m.get("stop_omega_radps", np.inf))
        c &= slow | ~np.isin(goal, self.stop_ids)
        st.timer = np.where(goal != st.timer_goal, 0.0, st.timer)
        st.timer_goal = goal
        st.timer = np.where(c, st.timer + dt, 0.0)
        need = np.take_along_axis(prm.times, goal[..., None], -1)[..., 0]
        va = None if m.get("auto_align", False) else m.get("virtual_align")
        if va:                                       # time of the robot's own drive-to-pose, charged as dwell
            def align_t(lat_err, gap, ang):
                d = np.hypot(lat_err, np.maximum(gap - va["gap_m"], 0.0))
                return va["settle_s"] + d / va["speed_mps"] + ang / va["omega_radps"]
            for li in range(3):                      # nearest free branch of that level
                lat_lv = np.where(free[..., li] & okb, np.abs(lat_b), np.inf).min(-1)
                lat_lv = np.where(np.isfinite(lat_lv), lat_lv, 0.0)
                need = np.where(sel(f"score_L{li + 2}"), need + align_t(lat_lv, gap_r, ang_r), need)
            need = np.where(sel("score_L1"), need + align_t(0.0, gap_r, ang_r), need)
            need = np.where(sel("algae_reef"), need + align_t(np.abs(lat), gap_r, ang_r), need)
            need = np.where(sel("coral_station"), need + align_t(0.0, gap_s, ang_s), need)
            need = np.where(sel("processor"), need + align_t(np.abs(lat_p), gap_p, ang_p), need)
        fire = c & (st.timer >= need)
        st.timer[fire] = 0.0

        # ---- effects, robot by robot (shared resources)
        P = self.points
        rc, ra = self.rad["coral"], self.rad["algae"]
        for r in np.flatnonzero(fire.any(0)):
            a = ALLIANCE[r]
            for e in np.flatnonzero(fire[:, r]):
                g, ph = self.goals[goal[e, r]], 0 if auto[e] else 1
                f = f0[e, r]
                if rng.random() >= prm.succ[e, r, goal[e, r]]:              # missed: the piece falls next to the robot
                    st.miss[e, r] += 1
                    if g == "coral_station" and st.stock[e, a] > 0:
                        st.stock[e, a] -= 1
                        self._spawn_coral(st, e, *self._beside(body, prm, e, r, rc, rng), rng.uniform(-np.pi, np.pi))
                    elif g in ("score_L1", "score_L2", "score_L3", "score_L4"):
                        st.hold_coral[e, r] = False
                        self._spawn_coral(st, e, *self._beside(body, prm, e, r, rc, rng), rng.uniform(-np.pi, np.pi))
                    elif g == "algae_reef" and st.reef_algae[e, a, f]:
                        st.reef_algae[e, a, f] = False
                        self._spawn_algae(st, e, *self._beside(body, prm, e, r, ra, rng))
                    elif g in ("processor", "net"):
                        st.hold_algae[e, r] = False
                        self._spawn_algae(st, e, *self._beside(body, prm, e, r, ra, rng))
                    continue                                                 # climb: not hanging, may retry
                if g == "coral_station" and st.stock[e, a] > 0:
                    st.stock[e, a] -= 1; st.hold_coral[e, r] = True
                elif g == "coral_floor":
                    cand = np.flatnonzero(in_c[e, r] & st.coral.active[e] & ~under[e])
                    if len(cand):
                        st.coral.active[e, cand[0]] = False; st.hold_coral[e, r] = True
                elif g == "algae_floor":
                    cand = np.flatnonzero(afloor[e, r] & st.algae.active[e])
                    if len(cand):
                        st.algae.active[e, cand[0]] = False; st.stacked[e, cand[0]] = -1; st.hold_algae[e, r] = True
                elif g == "algae_reef" and st.reef_algae[e, a, f]:
                    st.reef_algae[e, a, f] = False
                    if st.hold_algae[e, r]:
                        self._spawn_algae(st, e, self.reef_fc[a, f] + self.reef_n[a, f] * 0.35, self.reef_n[a, f] * 1.0)
                    else:
                        st.hold_algae[e, r] = True
                elif g == "score_L1" and st.l1[e, a, f] < self.l1_cap:
                    st.l1[e, a, f] += 1; st.l1_auto[e, a, f] += ph == 0
                    st.hold_coral[e, r] = False; pts[e, r] += P["coral"]["L1"][ph]; st.cat[e, a, 1] += P["coral"]["L1"][ph]
                elif g in ("score_L2", "score_L3", "score_L4"):
                    li = int(g[-1]) - 2
                    hi_ = st.reef_algae[e, a, f] and self.block["high" if self.algae_high[f] else "low"][li]
                    order = (1, 0) if lat[e, r] > 0 else (0, 1)
                    for s in order:
                        if not st.branch[e, a, f, s, li] and not hi_ and okb[e, r, s]:
                            st.branch[e, a, f, s, li] = True; st.branch_auto[e, a, f, s, li] = ph == 0
                            st.hold_coral[e, r] = False; pts[e, r] += P["coral"][g[-2:]][ph]
                            st.cat[e, a, 1] += P["coral"][g[-2:]][ph]
                            break
                elif g == "processor":
                    st.processor[e, a] += 1; st.hp_stock[e, 1 - a] += 1
                    st.hold_algae[e, r] = False; pts[e, r] += P["processor"][ph]; st.cat[e, a, 2] += P["processor"][ph]
                elif g == "net":
                    st.net[e, a] += 1; st.hold_algae[e, r] = False; pts[e, r] += P["net"][ph]; st.cat[e, a, 2] += P["net"][ph]
                elif g == "climb":
                    cg = cage_sel[e, r]
                    if st.cage_occ[e, cg] < 0:
                        st.cage_occ[e, cg] = r; st.climbed[e, r] = True

        # ---- human players: throw algae received through the opponent's processor into their net
        thr = (st.hp_stock > 0) & ~auto[:, None] & live[:, None]
        st.hp_timer = np.where(thr, st.hp_timer + dt, 0.0)
        for e, a in zip(*np.nonzero(st.hp_timer >= m["hp_throw_interval_s"])):
            st.hp_timer[e, a] = 0.0
            st.hp_stock[e, a] -= 1
            if rng.random() < m["hp_net_success"]:
                st.net[e, a] += 1; bonus[e, a] += P["net"][1]; st.cat[e, a, 2] += P["net"][1]
            else:
                side = rng.choice([-1, 1])
                xy = np.array([self.mid[0] + side * rng.uniform(0.6, 1.5), rng.uniform(*self.net_y[a])])
                self._spawn_algae(st, e, xy, rng.normal(0, 0.5, 2))

        # ---- fouls (contact based, assessed on contact onset)
        foul += self._fouls(st, body, prm, t, live, auto, dt)
        self._damage(st, body, live, dt)
        self._teammate_press(st, body, live, dt)
        self._push_teammate(st, body, goal, live, dt)

        for a in range(2):
            st.score[:, a] += pts[:, ALLIANCE == a].sum(1) + bonus[:, a] + foul[:, ALLIANCE != a].sum(1)
            st.cat[:, a, 4] += foul[:, ALLIANCE != a].sum(1)
        return pts, foul, bonus

    def _damage(self, st, body, live, dt):
        """Robot hits something (robot, cage, reef, wall) at >= damage_speed_mps. Each hit costs damage_points,
        at most once per damage_cooldown_s per robot. Kept in st.damage: NOT part of st.score."""
        m = self.model
        hit = (body.impact >= m["damage_speed_mps"]) & live[:, None] & (st.dmg_t <= 0)
        st.dmg_t = np.where(hit, m["damage_cooldown_s"], np.maximum(st.dmg_t - dt, 0.0))
        st.damage += hit * m["damage_points"]

    def _teammate_press(self, st, body, live, dt):
        """Seconds each robot has been continuously touching a teammate (any teammate contact resets at 0)."""
        same = ~(ALLIANCE[PAIRS[:, 0]] != ALLIANCE[PAIRS[:, 1]])
        touch = body.pair_contact & same
        on = (touch.astype(float) @ EI.T + touch.astype(float) @ EJ.T) > 0
        st.press_t = np.where(on & live[:, None], st.press_t + dt, 0.0)

    def _push_teammate(self, st, body, goal, live, dt):
        """Onset of a teammate contact in which the robot itself moves into the teammate (>0.3 m/s toward it) while
        that teammate's current goal is a scoring goal. Counted as one push event for the pusher."""
        a_, b_ = PAIRS[:, 0], PAIRS[:, 1]
        same = ALLIANCE[a_] == ALLIANCE[b_]
        on = body.pair_contact & same & live[:, None]
        onset = on & ~st.prev_tm
        st.prev_tm = on.copy()
        d = body.pose[:, b_, :2] - body.pose[:, a_, :2]
        u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        va = (body.vel[:, a_, :2] * u).sum(-1)                  # speed of a toward b
        vb = -(body.vel[:, b_, :2] * u).sum(-1)                 # speed of b toward a
        sc = np.isin(goal, self.scoring_ids)
        push_a = onset & (va > 0.3) & sc[:, b_]
        push_b = onset & (vb > 0.3) & sc[:, a_]
        st.push_events += (push_a.astype(float) @ EI.T + push_b.astype(float) @ EJ.T)

    def _fouls(self, st, body, prm, t, live, auto, dt):
        """G427 zone protection, G428 cage protection, G418 opponent cage, G421 one defender.
        Returns points given to the opponent, attributed to the offending robot (N,R)."""
        n, P = len(t), self.points
        th = body.pose[..., 2]
        cs, sn = np.cos(th), np.sin(th)
        zd, _, _ = geo.box_poly_sat(body.pose[..., :2], prm.half, cs, sn, self.zones, points=False)      # (N,R,4)
        rr = np.arange(R)
        in_own = (zd[:, rr, ALLIANCE] > 0) | (zd[:, rr, 2 + ALLIANCE] > 0)
        a_, b_ = PAIRS[:, 0], PAIRS[:, 1]
        opp = ALLIANCE[a_] != ALLIANCE[b_]
        new = body.pair_contact & ~st.prev_pair & opp & live[:, None]
        st.prev_pair = body.pair_contact.copy()
        cage_c = body.poly_contact[..., self.cage_poly]                                      # (N,R,6)
        own_cage = np.arange(6)[None, :] // 3 == ALLIANCE[:, None]                           # (R,6)
        touching_own = st.climbed | (cage_c & own_cage).any(-1)
        endgame = (t >= self.auto_s + self.teleop_s - self.endgame_s)[:, None]
        by_a = new & (in_own[:, b_] | (endgame & touching_own[:, b_]))                      # a hit protected b
        by_b = new & (in_own[:, a_] | (endgame & touching_own[:, a_]))
        # at most one foul per robot pair per `foul_cooldown_s` (referees do not call every re-contact)
        cd = (t[:, None] - st.pair_last) >= self.model.get("foul_cooldown_s", 3.0)
        by_a, by_b = by_a & cd, by_b & cd
        st.pair_last = np.where(by_a | by_b, t[:, None], st.pair_last)
        foul = (by_a @ EI.T + by_b @ EJ.T) * P["major_foul"]
        g418 = cage_c & ~own_cage & ~st.prev_cage & (~auto & live)[:, None, None]
        foul += g418.sum(-1) * P["major_foul"]
        st.prev_cage = cage_c.copy()
        ext_x = prm.half[..., 0] * np.abs(cs) + prm.half[..., 1] * np.abs(sn)
        x = body.pose[..., 0]
        beyond = np.where(ALLIANCE == 0, x - ext_x > self.mid[0] + self.bz_half, x + ext_x < self.mid[0] - self.bz_half)
        for a in range(2):
            mine = beyond & (ALLIANCE == a)
            many = (mine.sum(1) > 1) & live
            t0 = st.g421_t[:, a]
            t1 = np.where(many, t0 + dt, 0.0)
            pts = np.where(many & (t0 == 0), P["minor_foul"], 0.0) \
                + np.where(many & (t0 > 0) & (np.floor(t1 / 3.0) > np.floor(t0 / 3.0)), P["major_foul"], 0.0)
            st.g421_t[:, a] = t1
            foul += mine * (pts / np.maximum(mine.sum(1), 1))[:, None]
        for a in range(2):
            st.fouls[:, a] += foul[:, ALLIANCE == a].sum(1)
        return foul

    # ------------------------------------------------------------------ match events
    def end_auto(self, st, body, prm, mask):
        pts = np.zeros((len(mask), R))
        th = body.pose[..., 2]
        ext_x = prm.half[..., 0] * np.abs(np.cos(th)) + prm.half[..., 1] * np.abs(np.sin(th))
        left = np.abs(body.pose[..., 0] - self.start_x[ALLIANCE]) > ext_x
        pts += (left & mask[:, None]) * self.points["leave"][0]
        self._add(st, pts, 0)
        return pts

    def end_match(self, st, body, prm, mask):
        th = body.pose[..., 2]
        zd, _, _ = geo.box_poly_sat(body.pose[..., :2], prm.half, np.cos(th), np.sin(th), self.zones, points=False)
        in_bz = np.take_along_axis(zd, 2 + ALLIANCE[None, :, None].repeat(len(mask), 0), -1)[..., 0] > 0
        pts = np.zeros((len(mask), R))
        for r in range(R):
            occ = st.cage_occ == r
            deep = (occ & st.cage_deep).any(1)
            pts[:, r] = np.where(st.climbed[:, r], np.where(deep, self.points["cage"]["deep"], self.points["cage"]["shallow"]),
                                 in_bz[:, r] * self.points["park"])
        pts *= mask[:, None]
        self._add(st, pts, 3)
        return pts

    def _add(self, st, pts, cat):
        for a in range(2):
            st.score[:, a] += pts[:, ALLIANCE == a].sum(1)
            st.cat[:, a, cat] += pts[:, ALLIANCE == a].sum(1)

    # ------------------------------------------------------------------ observation / render
    def privileged(self, st):
        return {
            "hold": self.holding(st), "climbed": st.climbed,
            "coral_pos": st.coral.pos, "coral_on_floor": st.coral.active,
            "algae_pos": st.algae.pos, "algae_on_floor": st.algae.active,
            "branch": st.branch, "l1": st.l1, "reef_algae": st.reef_algae,
            "stock": st.stock, "hp_stock": st.hp_stock, "cage_occ": st.cage_occ, "cage_deep": st.cage_deep,
            "score": st.score,
        }

    def render_objects(self, st, body, prm, held_offsets):
        """Dynamic objects. coral (N,Mc,6)[x,y,z,yaw,pitch,vis], algae (N,Ma,4)[x,y,z,vis], cages (N,6,4)."""
        n = len(st.score)
        th = body.pose[..., 2]
        cs, sn = np.cos(th), np.sin(th)

        def held(off):
            o = np.asarray(off)
            return np.stack([body.pose[..., 0] + o[0] * cs - o[1] * sn, body.pose[..., 1] + o[0] * sn + o[1] * cs,
                             np.full(th.shape, o[2])], -1)
        rc, ra = self.rad["coral"], self.rad["algae"]
        floor_c = np.concatenate([st.coral.pos, np.full((n, self.kc, 1), 0.057), st.coral.yaw[..., None],
                                  np.zeros((n, self.kc, 1)), st.coral.active[..., None]], -1)
        hc = np.concatenate([held(held_offsets["coral"]), th[..., None], np.zeros((n, R, 1)), st.hold_coral[..., None]], -1)
        bp = np.broadcast_to(self.branch_pose, (n,) + self.branch_pose.shape).reshape(n, -1, 5)
        brc = np.concatenate([bp, st.branch.reshape(n, -1, 1)], -1)
        lp = np.broadcast_to(self.l1_pose, (n,) + self.l1_pose.shape)
        lvis = np.arange(self.l1_cap)[None, None, None] < st.l1[..., None]
        l1c = np.concatenate([lp.reshape(n, -1, 5), lvis.reshape(n, -1, 1)], -1)
        coral = np.concatenate([floor_c, hc, brc, l1c], 1).astype(np.float32)
        z_alg = np.where(st.stacked >= 0, 0.114 + ra, ra)
        fa = np.concatenate([st.algae.pos, z_alg[..., None], st.algae.active[..., None]], -1)
        ha = np.concatenate([held(held_offsets["algae"]), st.hold_algae[..., None]], -1)
        rp = np.broadcast_to(self.reef_algae_pose, (n, 2, 6, 3)).reshape(n, -1, 3)
        reef_a = np.concatenate([rp, st.reef_algae.reshape(n, -1, 1)], -1)
        algae = np.concatenate([fa, ha, reef_a], 1).astype(np.float32)
        cz = np.where(st.cage_deep, self.cfg["field"]["cages"]["bottom"]["deep"], self.cfg["field"]["cages"]["bottom"]["shallow"]) * IN \
            + self.cfg["field"]["cages"]["height"] * IN / 2
        cages = np.concatenate([np.broadcast_to(self.cage_xy, (n, 6, 2)), cz[..., None], st.cage_deep[..., None]], -1).astype(np.float32)
        robots_z = st.climbed * 0.25
        return {"coral": coral, "algae": algae, "cages": cages, "robot_z": robots_z.astype(np.float32)}

    def scene(self, cad=False):
        """Static primitives for the renderer (field frame, metres). cad=True: the structure comes from the
        optimised CAD model (visual/field.glb); only the AprilTags (placed from the tag file) remain."""
        f = self.cfg["field"]
        L, W = self.length, self.width
        mx, my = self.mid
        P = []
        box = lambda pos, size, mat, yaw=0.0: P.append({"kind": "box", "pos": list(map(float, pos)),
                                                        "rpy": [0.0, 0.0, float(yaw)], "size": list(map(float, size)), "mat": mat})
        box([mx, my, -0.005], [L + 3, W + 3, 0.01], "carpet")
        # walls: long guardrails and alliance walls (between chamfers)
        box([mx, -0.03, 0.255], [L, 0.06, 0.51], "guardrail")
        box([mx, W + 0.03, 0.255], [L, 0.06, 0.51], "guardrail")
        cy = [self.station_c[0, :, 1].min(), self.station_c[0, :, 1].max()]
        for x in (-0.03, L + 0.03):
            box([x, my, 0.99], [0.06, W - 2 * 1.05, 1.98], "alliance_wall")
        for a in range(2):
            for k in range(2):
                c, nn = self.station_c[a, k], self.station_n[a, k]
                yaw = np.arctan2(nn[1], nn[0])
                box([*(c - nn * 0.15), 0.85], [0.3, 2.0, 1.7], f"station", yaw)
            c, nn = self.proc_c[a], self.proc_n[a]
            box([*(c - nn * 0.2), 0.6], [0.4, 1.2, 1.2], "processor", np.arctan2(nn[1], nn[0]))
            yaw0 = np.arctan2(self.reef_n[a, 0, 1], self.reef_n[a, 0, 0])
            P.append({"kind": "prism", "verts": self.polys.V[a].tolist(), "z": [0.0, f["reef"]["trough_height"] * IN],
                      "mat": "reef_base"})
            for k in range(6):
                for s in (-1, 1):
                    xy = self.reef_fc[a, k] - self.reef_n[a, k] * 0.04 + self.reef_t[a, k] * s * f["reef"]["pipe_spacing"] * IN / 2
                    P.append({"kind": "cylinder", "pos": [*map(float, xy), 1.28], "rpy": [0, 0, 0],
                              "radius": 0.021, "height": 1.11, "mat": "reef_pipe"})
            # tape: reef zone outline, barge zone outline, starting line
            for poly, mat in ((self.zones.V[a], "tape_blue" if a == 0 else "tape_red"),
                              (self.zones.V[2 + a], "tape_blue" if a == 0 else "tape_red")):
                for i in range(len(poly)):
                    p0, p1 = poly[i], poly[(i + 1) % len(poly)]
                    if np.allclose(p0, p1):
                        continue
                    d = p1 - p0
                    box([*((p0 + p1) / 2), 0.001], [np.linalg.norm(d), 2 * IN, 0.002], mat, np.arctan2(d[1], d[0]))
            box([self.start_x[a], my, 0.001], [2 * IN, W, 0.002], "tape_black")
        for xy in self.marks:
            box([*xy, 0.001], [4 * IN, 1 * IN, 0.002], "tape_black")
            box([*xy, 0.001], [1 * IN, 4 * IN, 0.002], "tape_black")
        bg = f["barge"]
        box([mx, my, bg["truss_height"] * IN + 0.2], [bg["depth"] * IN, W + 0.8, 0.4], "barge")
        box([mx, my, bg["truss_height"] * IN / 2], [bg["center_support"][0] * IN, bg["center_support"][1] * IN,
                                                    bg["truss_height"] * IN], "truss")
        for y in (-0.4, W + 0.4):
            box([mx, y, bg["height"] * IN / 2], [0.35, 0.35, bg["height"] * IN], "truss")
        for i, tg in sorted(self.tags.items()):
            P.append({"kind": "tag", "id": int(i), "pos": list(map(float, tg["pos"])), "quat_wxyz": list(map(float, tg["quat"])),
                      "size": f["tags"]["black_square"] * IN, "panel": f["tags"]["panel"] * IN})
        if cad:
            P = [p for p in P if p["kind"] == "tag"]
        return {"field": [L, W], "primitives": P}
