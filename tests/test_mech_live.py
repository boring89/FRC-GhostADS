"""Mechanism in the live view: the joint tracker moves only while the rules dwell, follows the dwell progress,
holds the score pose briefly after a release, and returns to the carry pose."""
import numpy as np

from ghostsim import SimCfg, VecSim, load
from ghostsim.mechanism import ARM, SLOT_W, SLOTS
from ghostsim.mechanism_state import LINGER_S, RATE, RISE_AT, SCORE, JointTracker, mech_slots

GAME = load("2025")
G = GAME.gi
Z = [False] * 6


def run(tr, goal, prog, hc=Z, ha=Z, cl=Z, n=1, dt=0.1):
    for _ in range(n):
        tr.update(np.array(goal), np.array(prog, float), hc, ha, cl, dt=dt)


def test_nothing_moves_while_driving_whatever_the_goal():
    tr = JointTracker(GAME.goals, 6)
    rng = np.random.default_rng(0)
    for _ in range(50):                                         # the policy switching goals on the way
        goals = rng.choice([G["score_L4"], G["score_L2"], G["net"], G["algae_reef"], G["coral_station"]], 6)
        run(tr, goals, [0.0] * 6, hc=[True] * 6)
    assert np.allclose(tr.q[:, 0], 0.0) and np.allclose(tr.q[:, 1], 0.0) and np.allclose(tr.q[:, 2], ARM["stow"])


def test_elevator_follows_the_dwell_and_is_rate_limited():
    tr = JointTracker(GAME.goals, 6)
    h_top = SCORE["score_L4"][0]
    hs, n = [], 12
    for k in range(n):                                          # a 1.2 s dwell at the reef
        run(tr, [G["score_L4"]] * 6, [(k + 1) / n] * 6, hc=[True] * 6)
        hs.append(tr.q[0, 0])
    assert np.all(np.diff(hs) >= -1e-9) and np.max(np.diff(hs)) <= RATE[0] * 0.1 + 1e-9
    assert abs(hs[-1] - h_top) < 1e-6
    assert hs[int(n * RISE_AT) - 1] < h_top and hs[int(n * RISE_AT) + 2] > 0.9 * h_top


def test_pose_is_held_after_release_then_returns():
    tr = JointTracker(GAME.goals, 6)
    for k in range(10):
        run(tr, [G["score_L4"]] * 6, [(k + 1) / 10] * 6, hc=[True] * 6)
    top = tr.q[0, 0]
    run(tr, [G["score_L4"]] * 6, [0.0] * 6, hc=Z)               # released: rules timer reset, coral gone
    assert abs(tr.q[0, 0] - top) < 1e-6
    run(tr, [G["score_L4"]] * 6, [0.0] * 6, n=int(LINGER_S / 0.1) + 12)
    assert tr.q[0, 0] < 1e-6


def test_station_funnel_floor_arm_and_climber():
    tr = JointTracker(GAME.goals, 6)
    run(tr, [G["coral_station"]] * 3 + [G["algae_floor"]] * 3, [0.5] * 3 + [0.0] * 3, n=10)
    assert tr.funnel[0] is not None and 0.0 < tr.funnel[0] < 1.0 and tr.q[0, 0] == 0.0
    assert abs(tr.q[3, 2] - ARM["floor"]) < 1e-6
    run(tr, [G["climb"]] * 6, [0.0] * 6, cl=[True] * 6, n=20)
    assert tr.q[0, 3] < 90.0


def test_mech_inputs_and_slots_from_a_live_sim():
    from ghosttrain.broadcast_sim import mech_inputs
    sim = VecSim(SimCfg(n_envs=1, sensors=False, seed=0))
    tr = JointTracker(GAME.goals, 6)
    for _ in range(5):
        sim.step(np.zeros((1, 6, 3)), np.zeros((1, 6), int))
        tr.update(*mech_inputs(sim), dt=0.1)
    st = tr.state()
    frames = (2 * (sim.prm.half[0, :, 0] - 3.25 * 0.0254)).tolist()
    s = mech_slots(st["q"], st["hold"], st["funnel"], frames)
    assert s.shape == (1, 6, SLOTS, SLOT_W) and s[0, :, :, 0].sum(1).min() > 30
