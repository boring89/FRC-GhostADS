"""Close-range auto-align to a reef scoring branch (learned trigger: the policy chooses goal `align_score`).

- Target: among the 6 own reef faces x 2 sides (left/right branch), the nearest branch that still has a free
  level (L2-L4). The robot stops `align_gap_m` in front of the face, centred on that branch, facing the reef.
- Control: P controller on the pose (KF estimate when sensors are on, true pose otherwise), speed limited.
- Dwell: the robot counts as ALIGNED after it has stayed within `align_tol_m` / `align_tol_deg` for the
  `align_score` time in ours.json (0.7 s). Scoring goals then hold the aligned pose until the piece is placed.
- The rules check the branch lateral tolerance (model.branch_lateral_tol_m) when a score goal fires.

Applied inside VecSim.step, once per policy tick, so every caller (policy, baseline, evaluation) behaves the same.
"""
from __future__ import annotations

import numpy as np

from ghostsim import geometry as geo
from ghostsim.robots import ALLIANCE, R

SCORE_GOALS = ("score_L1", "score_L2", "score_L3", "score_L4")


def _pose(sim):
    return sim.sens.est if sim.sens is not None else sim.body.pose


def candidates(sim, pos, hx):
    """All 12 (face, side) approach points for every robot, plus validity (a free L2-L4 level on that branch)."""
    g, gs = sim.game, sim.gs
    n = pos.shape[0]
    half_pipe = g.cfg["field"]["reef"]["pipe_spacing"] * 0.0254 / 2
    gap = g.model["align_gap_m"]
    fc, nn, tt = g.reef_fc[ALLIANCE], g.reef_n[ALLIANCE], g.reef_t[ALLIANCE]          # (R,6,2)
    pts, yaw, valid = [], [], []
    br_free = ~gs.branch[:, ALLIANCE]                                                # (N,R,6,2,3)
    for si, sgn in enumerate((-1.0, 1.0)):
        centre = fc[None] + tt[None] * (sgn * half_pipe)                              # (1,R,6,2) branch centre
        app = centre + nn[None] * (hx[..., None, None] + gap)                         # approach point (N,R,6,2)
        pts.append(app)
        yaw.append(np.broadcast_to(np.arctan2(-nn[None, :, :, 1], -nn[None, :, :, 0]), (n,) + nn.shape[:2]))
        valid.append(br_free[:, :, :, si, :].any(-1) & sim.prm.levels[:, :, 1:].any(-1)[..., None])   # L2..L4
    pts = np.stack(pts, 2).reshape(n, R, 12, 2)                                       # index = side * 6 + face
    yaw = np.stack(yaw, 2).reshape(n, R, 12)
    valid = np.stack(valid, 2).reshape(n, R, 12)
    # a branch with another robot (teammate or opponent) within 0.6 m of its approach point is not a target now
    others = sim_pos_all(sim)                                                         # (N,R,2)
    dd = np.linalg.norm(pts[:, :, :, None, :] - others[:, None, None, :, :], axis=-1)  # (N,R,12,R)
    dd = np.where(np.eye(R, dtype=bool)[None, :, None, :], np.inf, dd)                  # not itself
    valid &= dd.min(-1) > 1.0
    return pts, yaw, valid


def sim_pos_all(sim):
    return _pose(sim)[..., :2]


def choose(sim, pos, hx, bad=None):
    """Nearest valid candidate per robot -> (face, side, target xy, target yaw, ok). `bad` (N,R,12): given up."""
    pts, yaw, valid = candidates(sim, pos, hx)
    if bad is not None:
        valid = valid & ~bad
    d = np.linalg.norm(pts - pos[:, :, None], axis=-1)
    d = np.where(valid, d, np.inf)
    k = np.argmin(d, -1)
    ok = np.isfinite(d.min(-1))
    xy = np.take_along_axis(pts, k[..., None, None].repeat(2, -1), 2)[:, :, 0]
    th = np.take_along_axis(yaw, k[..., None], -1)[..., 0]
    return k % 6, k // 6, xy, th, ok


def drive(sim, pos, yaw, tgt, tyaw):
    """Stick (N,R,3) toward tgt/tyaw, in the same convention as the baseline (alliance-relative field frame).
    Other robots repel (as in the baseline) so a teammate in the way is walked around, not pushed."""
    p = sim.prm
    err = tgt - pos
    dist = np.linalg.norm(err, axis=-1, keepdims=True)
    vmax = np.minimum(p.max_v, 2.0)[..., None]
    # speed profile: 2.5 /s * distance, but never below a stick of ~0.12 (the chassis deadband is 0.05 per axis, so
    # smaller commands do not move the robot at all); the floor is off within 3 cm so the robot settles
    floor = np.where(dist > 0.03, 0.55, 0.0)
    speed = np.maximum(np.clip(2.5 * dist, 0.0, None), floor)
    speed = np.minimum(vmax, speed)
    v = err / np.maximum(dist, 1e-9) * speed
    d = pos[:, :, None] - pos[:, None]                                    # (N,R,R,2) from j to i
    dd = np.linalg.norm(d, axis=-1)
    # repel only when nearly touching (0.25 m margin): a wide field pushes the robot off its target otherwise
    rr = (p.half[..., 0][:, :, None] + p.half[..., 0][:, None]) + 0.25
    w = np.clip((rr - dd) / 0.25, 0, 1) * (1 - np.eye(R))
    v += ((d / np.maximum(dd[..., None], 1e-9)) * w[..., None]).sum(2) * vmax * 1.2
    herr = geo.wrap(tyaw - yaw)
    # proportional, but never below a turn rate that clears the deadband (0.05 of full turn) when the heading error is
    # larger than 0.02 rad; otherwise the robot stalls ~4 deg off target and never meets the 3 deg tolerance
    om = np.clip(6.0 * herr, -0.8 * p.max_w, 0.8 * p.max_w)
    flip = np.where(ALLIANCE == 0, 1.0, -1.0)[None, :, None]
    stick = np.concatenate([flip * v / p.max_v[..., None], (om / p.max_w)[..., None]], -1)
    return np.clip(stick, -1.0, 1.0)


def aligned_error(sim, pos, yaw, tgt, tyaw):
    g = sim.game
    d = np.linalg.norm(tgt - pos, axis=-1)
    dy = np.abs(geo.wrap(tyaw - yaw))
    return (d <= g.model["align_tol_m"]) & (dy <= np.radians(g.model["align_tol_deg"]))


def apply(sim, stick, goal):
    """Override the stick of robots whose goal is align_score (or a score goal while holding an aligned pose).

    Returns the stick actually executed. Updates gs.align_* (target, dwell timer, aligned flag)."""
    g, gs, p = sim.game, sim.gs, sim.prm
    ai = g.gi["align_score"]
    dt = sim.cfg.dt * sim.cfg.substeps
    pos_all = _pose(sim)[..., :2]
    yaw_all = _pose(sim)[..., 2]
    hx = sim.prm.half[..., 0]
    score_ids = np.array([g.gi[s] for s in SCORE_GOALS])
    is_align = goal == ai
    is_score = np.isin(goal, score_ids)
    out = stick.copy()
    # robots with no align goal and no score goal lose their target
    drop = ~(is_align | is_score)
    gs.align_face[drop], gs.align_side[drop] = -1, -1
    gs.aligned[drop], gs.align_t[drop] = False, 0.0
    # pick a target for robots that are aligning and have none yet
    need = is_align & (gs.align_face < 0)
    if need.any():
        f, sd, xy, th, ok = choose(sim, pos_all, hx, bad=gs.align_bad)
        gs.align_face = np.where(need & ok, f, gs.align_face)
        gs.align_side = np.where(need & ok, sd, gs.align_side)
        gs.align_tgt = np.where((need & ok)[..., None], np.concatenate([xy, th[..., None]], -1), gs.align_tgt)
        gs.align_stuck = np.where(need, 0.0, gs.align_stuck)
    if (gs.align_bad.all(-1) & need).any():                  # every candidate given up: start over
        gs.align_bad[need & gs.align_bad.all(-1)] = False
    active = (is_align | is_score) & (gs.align_face >= 0)
    if active.any():
        tgt = gs.align_tgt[..., :2]
        tyaw = gs.align_tgt[..., 2]
        # two-phase approach: first a waypoint 0.5 m out from the reef (lateral alignment happens there, far from
        # the hexagon), then straight in along the face normal. Approaching the branch diagonally clips the
        # adjacent reef face with the robot's corner (half-width 0.41 m > branch offset 0.165 m).
        normal = np.stack([-np.cos(tyaw), -np.sin(tyaw)], -1)
        wp = tgt + 0.5 * normal
        tangent = np.stack([-normal[..., 1], normal[..., 0]], -1)
        lat = np.abs(((tgt - pos_all) * tangent).sum(-1))
        goal_pt = np.where((lat > 0.03)[..., None], wp, tgt)        # straight in only once laterally aligned
        ctrl = drive(sim, pos_all, yaw_all, goal_pt, tyaw)
        out = np.where(active[..., None], ctrl, out)
        inside = aligned_error(sim, pos_all, yaw_all, tgt, tyaw)
        dwell = is_align & active
        # stuck (no progress for 2 s, not aligned): give this candidate up and choose another next tick
        moved = np.linalg.norm(pos_all - gs.align_last, axis=-1)
        gs.align_last = pos_all.copy()
        gs.align_stuck = np.where(active & ~inside & (moved < 0.01), gs.align_stuck + dt, np.where(active, 0.0, gs.align_stuck))
        give_up = active & ~inside & (gs.align_stuck > 2.0)
        if give_up.any():
            cand = gs.align_side * 6 + gs.align_face
            ee, rr = np.nonzero(give_up)
            gs.align_bad[ee, rr, cand[ee, rr]] = True
            gs.align_face[give_up], gs.align_side[give_up] = -1, -1
            gs.align_stuck[give_up] = 0.0
        gs.align_t = np.where(dwell & inside, gs.align_t + dt, np.where(dwell, 0.0, gs.align_t))
        need_t = p.times[..., ai]
        gs.aligned = np.where(dwell, gs.aligned | (gs.align_t >= need_t - 1e-9), gs.aligned)
    return out, goal


def aligned_goal(sim, r_index_env=None):
    """Best scoring goal that can fire on the aligned branch (highest free level), or None. Used by the baseline."""
    g, gs = sim.game, sim.gs
    out = np.full(gs.align_face.shape, -1)
    for e, r in zip(*np.nonzero(gs.aligned)):
        f, s = gs.align_face[e, r], gs.align_side[e, r]
        if f < 0:
            continue
        al = ALLIANCE[r]
        best = None
        # reef algae on this face blocks the levels it covers (game.json model.algae_blocks), same as the rules
        blocked_face = gs.reef_algae[e, al, f]
        high = g.algae_high[f]
        for li, name in ((2, "score_L4"), (1, "score_L3"), (0, "score_L2")):
            block = g.block["high" if high else "low"][li]
            if blocked_face and block:
                continue
            if not gs.branch[e, al, f, s, li] and sim.prm.levels[e, r, li + 1]:
                best = g.gi[name]
                break
        out[e, r] = -1 if best is None else best
    return out
