"""Joint state of the 254-style mechanism for each robot, driven by the policy's goal (renderer only).

It is NOT part of the physics or the observations: the broadcast simulation process runs it once per tick and sends the
resulting geometry to the renderer. Joints move toward a per-goal target at a limited rate, so the animation shows
elevator / wrist / intake motion that follows what the robot is doing.
"""
from __future__ import annotations

import numpy as np

from .mechanism import SLOTS, SLOT_W, WRIST_MIN_DEG, pose, slots_for

STOW = (0.0, 80.0, 75.0)                    # (h m, theta deg, psi deg)
TARGETS = {
    "coral_floor": (0.0, 80.0, -10.0),      # intake down, wrist stowed
    "algae_floor": (0.0, WRIST_MIN_DEG, 75.0),
    "algae_reef": (0.5, 0.0, 75.0),
    "score_L1": (0.0, -10.0, 75.0),         # trough
    "score_L2": (0.30, 0.0, 75.0),          # branch top 31.9 in
    "score_L3": (0.70, 0.0, 75.0),          # 47.6 in
    "score_L4": (1.32, 0.0, 75.0),          # 72 in
    "processor": (0.0, WRIST_MIN_DEG, 75.0),
    "net": (1.32, 20.0, 75.0),
}
RATE = np.array([4.4, 1050.0, 600.0])       # elevator m/s, wrist deg/s, intake deg/s (254 binder timings)
H_MAX_STATE = 1.32


class JointTracker:
    def __init__(self, goals, n_robots=6):
        self.goals = list(goals)
        self.q = np.tile(np.array(STOW, float), (n_robots, 1))            # (R, 3): h, theta, psi
        self.n = n_robots

    def update(self, goal_ids, hold_coral, hold_algae, dt=0.1):
        for r in range(self.n):
            name = self.goals[int(goal_ids[r])]
            tgt = np.array(TARGETS.get(name, STOW), float)
            step = RATE * dt
            self.q[r] = self.q[r] + np.clip(tgt - self.q[r], -step, step)
        self.hold = [("coral" if hold_coral[r] else ("algae" if hold_algae[r] else None)) for r in range(self.n)]

    def slots(self):
        """(1, R, SLOTS, SLOT_W) for the renderer (one environment)."""
        out = np.zeros((1, self.n, SLOTS, SLOT_W), np.float32)
        for r in range(self.n):
            h, th, ps = self.q[r]
            out[0, r] = slots_for(pose(h=h, theta=th, psi=ps, hold=self.hold[r]))
        return out
