"""254 mechanism in the live view: joint tracker (goal -> joint targets, rate limits) and renderer slot layout."""
import numpy as np

from ghostsim import load
from ghostsim.mechanism import H_MAX, SLOT_W, SLOTS, WRIST_MIN_DEG, pose, slots_for
from ghostsim.mechanism_state import TARGETS, JointTracker

GAME = load("2025")


def test_joints_reach_score_targets_and_respect_rate_limits():
    tr = JointTracker(GAME.goals, 6)
    gl = np.array([GAME.gi["score_L4"]] * 6)
    z = [False] * 6
    tr.update(gl, z, z, 0.1)
    h1 = tr.q[0, 0]
    assert 0 < h1 <= 4.4 * 0.1 + 1e-9                      # elevator moves at most 4.4 m/s
    for _ in range(40):
        tr.update(gl, z, z, 0.1)
    assert np.allclose(tr.q[0], TARGETS["score_L4"], atol=1e-6)
    assert 1.32 <= H_MAX + 1e-9 and TARGETS["score_L4"][0] <= H_MAX + 1e-9


def test_processor_and_floor_targets():
    tr = JointTracker(GAME.goals, 6)
    gl = np.array([GAME.gi["processor"]] * 3 + [GAME.gi["coral_floor"]] * 3)
    z = [False] * 6
    for _ in range(40):
        tr.update(gl, z, z, 0.1)
    assert abs(tr.q[0, 1] - WRIST_MIN_DEG) < 1e-6          # processor: wrist at the lowest allowed angle
    assert abs(tr.q[3, 2] - (-10.0)) < 1e-6                # coral floor: intake down


def test_slots_layout_and_visibility():
    solids = pose(h=0.5, theta=10, psi=75, hold="coral")
    s = slots_for(solids)
    assert s.shape == (SLOTS, SLOT_W)
    assert s[:, 0].sum() == len(solids)                   # visible slots == solids
    assert np.allclose(s[0, 8:17].reshape(3, 3) @ s[0, 8:17].reshape(3, 3).T, np.eye(3), atol=1e-5)   # rotation
    algae = slots_for(pose(h=0.5, theta=10, psi=75, hold="algae"))
    assert algae[:, 1].sum() == 1                          # the held algae is the only sphere
