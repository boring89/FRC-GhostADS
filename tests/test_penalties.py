"""Damage tally (not game score), teammate-press timer, baseline climbing."""
import numpy as np

from conftest import idle, make_sim, place
from ghosttrain.baseline import Baseline


def _clear_others(sim, keep):
    """Park every robot except `keep` far from the action (their random teleop spawns would touch the test robots)."""
    spots = [(2.0, 7.8), (2.0, 0.6), (15.8, 7.8), (15.8, 0.6), (9.0, 0.5)]
    k = 0
    for r in range(6):
        if r in keep:
            continue
        place(sim, r, spots[k], 0.0)
        k += 1


def test_damage_counts_hard_impacts_not_score():
    sim = make_sim(seed=0)
    g = sim.game
    _clear_others(sim, keep={0})
    place(sim, 0, [g.mid[0] - 3.0, 1.2], 0.0)                 # drive into cage_5 (~4.6 m/s)
    st, gl = idle(sim)
    st[:, 0, 0] = 1.0
    blue0 = sim.score[0, 0]
    for _ in range(12):
        sim.step(st, gl)
    assert sim.gs.damage[0, 0] >= 1.0
    assert sim.score[0, 0] == blue0                           # damage is not part of the game score
    # (red gains the G418 opponent-cage foul points here; that is game score, checked elsewhere)


def test_damage_cooldown_and_threshold():
    sim = make_sim(seed=0)
    g = sim.game
    _clear_others(sim, keep={0})
    place(sim, 0, [g.mid[0] - 3.0, 1.2], 0.0)
    st, gl = idle(sim)
    st[:, 0, 0] = 1.0
    for _ in range(12):
        sim.step(st, gl)
    d = sim.gs.damage[0, 0]
    # keep pushing into the same cage for 0.5 s: still inside the 1 s cooldown, no extra hit
    for _ in range(5):
        sim.step(st, gl)
    assert sim.gs.damage[0, 0] == d
    # a slow bump (below 2.5 m/s) costs nothing
    sim2 = make_sim(seed=0)
    _clear_others(sim2, keep={0})
    place(sim2, 0, [g.mid[0] - 3.0, 1.2], 0.0)
    st2, gl2 = idle(sim2)
    st2[:, 0, 0] = 0.4
    for _ in range(40):
        sim2.step(st2, gl2)
    assert sim2.gs.damage[0, 0] == 0.0


def test_teammate_press_timer():
    sim = make_sim(seed=0)
    g = sim.game
    _clear_others(sim, keep={0, 1})
    place(sim, 0, [g.mid[0] - 3.0, 6.5], 0.0)
    place(sim, 1, [g.mid[0] - 3.0 + 0.78, 6.5], 0.0)          # teammate, bumpers overlapping
    st, gl = idle(sim)
    st[:, 0, 0] = 0.0
    st[:, 1, 0] = 0.0
    for _ in range(20):                                       # 2 s
        sim.step(st, gl)
    assert sim.gs.press_t[0, 0] > g.model["teammate_press_grace_s"]
    place(sim, 1, [g.mid[0] + 2.5, 1.0], 0.0)                 # separate
    sim.step(st, gl)
    assert sim.gs.press_t[0, 0] == 0.0


def test_baseline_hangs_deep_capable_robot():
    """Full auto-start matches: deep-capable robots hang on their own cages. Rate is ~68 % (noisy); require > 0."""
    from ghostsim import SimCfg, VecSim
    sim = VecSim(SimCfg(n_envs=16, sensors=False, start_mode="auto", seed=7))
    bl = Baseline(sim.game)
    sim.prm.climb[:, 3] = 2
    sim.prm.climb[:, 0] = 0
    for _ in range(int(round(sim.end_t / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
    assert sim.gs.climbed[:, 3].sum() > 0
    assert sim.gs.climbed[:, 0].sum() == 0                    # climb 0: never


def _push_run(teammate_goal_name):
    sim = make_sim(seed=0)
    g = sim.game
    _clear_others(sim, keep={0, 1})
    place(sim, 0, [g.mid[0] - 3.0, 6.5], 0.0)
    place(sim, 1, [g.mid[0] - 3.0 + 1.0, 6.5], 0.0)            # teammate 1 m ahead
    st, gl = idle(sim)
    gl[:, 1] = g.goals.index(teammate_goal_name)
    st[:, 0, 0] = 1.0                                           # robot 0 drives into its teammate
    for _ in range(30):
        sim.step(st, gl)
    return sim.gs.push_events[0, 0]


def test_push_scoring_teammate_counts_once():
    assert _push_run("score_L4") >= 1
    assert _push_run("idle") == 0
