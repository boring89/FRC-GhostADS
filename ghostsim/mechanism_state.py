"""Joint state of the mechanism for each robot (renderer only; not physics, not observations).

The broadcast simulation process updates it once per policy tick from what the rules are doing, and the renderer
interpolates the joints between ticks:
  - driving: carry pose (elevator down, dispenser level, algae arm stowed or cradling a ball). Nothing moves on the
    way, whatever goal the policy currently names.
  - dwell (the rules' timer for the current goal is running, i.e. the robot is in place and stopped): the joints
    follow the dwell progress p = timer / need, so a score pose is reached at ~70 % of the dwell and the piece leaves
    at 100 %. After a release the pose is held for LINGER_S, then everything returns to carry.
  - coral station: a coral slides down the funnel during the dwell.
  - floor intake (algae; coral for partners / opponents that have one): the arm is down while the robot intakes;
    climb: the climber reaches out, then pulls.
Joint rates are limited (elevator 3 m/s), so a move takes several frames even if a dwell is short.
"""
from __future__ import annotations

import numpy as np

from .mechanism import ARM, SLOT_W, SLOTS, STOW, coral_target, pose_moving, pose_static, solid_rows

JOINTS = ("h", "wrist", "arm", "climb")
RATE = np.array([3.0, 300.0, 360.0, 120.0])        # m/s, deg/s, deg/s, deg/s
LINGER_S = 0.35
RISE_AT = 0.7                                      # share of the dwell at which a score pose is reached
REEF_ALGAE_Z = {"low": 36.3 * 0.0254, "high": 52.1 * 0.0254}     # game.json field.reef.algae_rest
# score poses: (coral centre height, wrist); heights a little above each branch top / the trough
CORAL_AT = {"score_L1": (0.50, -20.0), "score_L2": (0.84, -35.0), "score_L3": (1.24, -35.0), "score_L4": (1.86, -35.0)}
SCORE = {g: (coral_target(z, w), w) for g, (z, w) in CORAL_AT.items()}


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def carry(hold_algae):
    q = np.array([STOW[j] for j in JOINTS], float)
    q[2] = ARM["hold"] if hold_algae else ARM["stow"]
    return q


def target(goal, p, hold_coral, hold_algae, climbed, algae_high=False):
    """Joint target and funnel coral position (None or 0..1) for one robot. p: dwell progress (0 = not dwelling)."""
    q = carry(hold_algae)
    funnel = None
    u = ease(p / RISE_AT) if p > 0 else 0.0
    if goal in SCORE and p > 0:
        h, w = SCORE[goal]
        q[0], q[1] = h * u, w * u
    elif goal == "algae_reef" and p > 0:
        q[0] = max(0.0, REEF_ALGAE_Z["high" if algae_high else "low"] - 0.40) * u   # dispenser nose at the algae
        q[2] = ARM["catch"]
    elif goal == "net" and p > 0:
        q[2] = ARM["hold"] + (ARM["net"] - ARM["hold"]) * ease(p)
    elif goal == "processor" and p > 0:
        q[2] = ARM["processor"]
    elif goal == "coral_station" and p > 0 and not hold_coral:
        funnel = ease(p)
    if (goal == "algae_floor" and not hold_algae) or (goal == "coral_floor" and not hold_coral):
        q[2] = ARM["floor"]                        # the front arm is the only floor intake drawn (others' coral too)
    if climbed:
        q[3] = 60.0
    elif goal == "climb":
        q[3] = 15.0
    return q, funnel


class JointTracker:
    def __init__(self, goals, n_robots=6):
        self.goals = list(goals)
        self.n = n_robots
        self.q = np.tile(carry(False), (n_robots, 1))
        self.linger = np.zeros(n_robots)
        self.linger_goal = [""] * n_robots
        self.prev_hold = np.zeros((n_robots, 2), bool)
        self.hold = [None] * n_robots
        self.funnel = [None] * n_robots

    def update(self, goal_ids, progress, hold_coral, hold_algae, climbed, algae_high=None, dt=0.1):
        """goal_ids, progress (0..1, 0 = not dwelling), hold_*, climbed, algae_high: per robot."""
        algae_high = np.zeros(self.n, bool) if algae_high is None else algae_high
        for r in range(self.n):
            g = self.goals[int(goal_ids[r])]
            hc, ha = bool(hold_coral[r]), bool(hold_algae[r])
            released = (self.prev_hold[r, 0] and not hc) or (self.prev_hold[r, 1] and not ha)
            if released and (g in SCORE or g in ("net", "processor")):
                self.linger[r], self.linger_goal[r] = LINGER_S, g
            if self.linger[r] > 0:
                self.linger[r] -= dt
                g, p = self.linger_goal[r], 1.0
            else:
                p = float(progress[r])
            tgt, self.funnel[r] = target(g, p, hc, ha, bool(climbed[r]), bool(algae_high[r]))
            step = RATE * dt
            self.q[r] = self.q[r] + np.clip(tgt - self.q[r], -step, step)
            self.prev_hold[r] = (hc, ha)
            self.hold[r] = "both" if hc and ha else ("coral" if hc else ("algae" if ha else None))

    def state(self):
        """Per-tick snapshot for the renderer: joints (R,4) and the piece state per robot."""
        return {"q": self.q.copy(), "hold": list(self.hold), "funnel": list(self.funnel)}


_STATIC: dict = {}


def mech_slots(q, hold, funnel, frames, n=6):
    """(1, R, SLOTS, SLOT_W) renderer array from joint values (e.g. interpolated between two ticks). The static part of
    each robot (drivebase, bumpers, outer stage, funnel) is computed once per frame size and alliance."""
    out = np.zeros((1, n, SLOTS, SLOT_W), np.float32)
    for r in range(n):
        key = (0 if r < 3 else 1, round(float(frames[r]), 5))
        if key not in _STATIC:
            _STATIC[key] = solid_rows(pose_static(key[0], frames[r]))
        st = _STATIC[key]
        h, w, a, c = q[r]
        mv = solid_rows(pose_moving(h, w, a, c, hold[r], frames[r], funnel[r]))
        out[0, r, :len(st)] = st
        out[0, r, len(st):len(st) + len(mv)] = mv
    return out
