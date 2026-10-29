"""G425 pins, G403 / G405 (AUTO), per-rule foul accounting (game/2025/rules.py _fouls, _pins)."""
import numpy as np

from conftest import idle, make_sim, place, run
from test_penalties import _clear_others


def pts(sim, rule, alliance):
    return float(sim.gs.foul_rule[0, alliance, sim.game.foul_rules.index(rule)])


def _wall_pin(sim, x=6.0):
    """Blue robot 0 against the lower guardrail (outside every protected zone), red robot 3 right above it.
    Returns the stick array with red pushing blue into the wall (red stick 'left' = world -y)."""
    _clear_others(sim, keep={0, 3})
    h0, h3 = sim.prm.half[0, 0, 1], sim.prm.half[0, 3, 1]
    y0 = sim.game.cfg["field"]["guardrail_inset_m"] + h0 + 0.005
    place(sim, 0, [x, y0], 0.0)
    place(sim, 3, [x, y0 + h0 + h3 + 0.02], np.pi)
    st, gl = idle(sim)
    st[:, 3, 1] = 1.0
    return st, gl


def test_pin_against_the_wall_minor_then_majors():
    sim = make_sim()
    st, gl = _wall_pin(sim)
    run(sim, st, gl, 100)                                       # 10 s
    assert pts(sim, "G425", 1) == 2 + 6 + 6                     # MINOR past 3 s, MAJOR at 6 s and 9 s
    assert pts(sim, "G427", 1) == 0 and pts(sim, "G425", 0) == 0


def test_pushing_in_the_open_is_not_a_pin():
    sim = make_sim()
    _clear_others(sim, keep={0, 3})
    place(sim, 0, [6.0, 3.0], 0.0)
    place(sim, 3, [6.0, 3.0 + sim.prm.half[0, 0, 1] + sim.prm.half[0, 3, 1] + 0.02], np.pi)
    st, gl = idle(sim)
    st[:, 3, 1] = 1.0
    run(sim, st, gl, 60)
    assert pts(sim, "G425", 1) == 0


def test_pin_count_survives_a_short_break_and_ends_after_separation():
    sim = make_sim()
    st, gl = _wall_pin(sim)
    run(sim, st, gl, 25)                                        # 2.5 s pinned
    p_red = sim.body.pose[0, 3].copy()
    place(sim, 3, p_red[:2] + [0, 0.6], np.pi)                  # backs off 0.6 m (inside 6 ft) for 1 s
    run(sim, idle(sim)[0], gl, 10)
    run(sim, st, gl, 12)                                        # pins again: the count continues past 3 s
    assert pts(sim, "G425", 1) == 2
    sim2 = make_sim()
    st, gl = _wall_pin(sim2)
    run(sim2, st, gl, 25)
    p_red = sim2.body.pose[0, 3].copy()
    place(sim2, 3, p_red[:2] + [0, 2.5], np.pi)                 # 2.5 m away for 3.5 s: the pin ends
    run(sim2, idle(sim2)[0], gl, 35)
    place(sim2, 3, p_red[:2], np.pi)
    run(sim2, st, gl, 25)                                       # a new pin, 2.5 s: no foul yet
    assert pts(sim2, "G425", 1) == 0


def test_auto_contact_across_the_barge_zone_g403():
    for start, expect in (("auto", 6.0), ("teleop", 0.0)):
        sim = make_sim(start_mode=start)
        _clear_others(sim, keep={0, 3})
        g = sim.game
        x = g.mid[0] + 2.4                                      # blue robot on the red side, past the barge zone
        place(sim, 0, [x, 1.2], 0.0)
        place(sim, 3, [x + sim.prm.half[0, 0, 0] + sim.prm.half[0, 3, 0] + 0.15, 1.2], np.pi)
        st, gl = idle(sim)
        st[:, 0, 0] = 0.5
        run(sim, st, gl, 8)
        assert pts(sim, "G403", 0) == expect, start


def test_auto_opponent_cage_g405():
    sim = make_sim(start_mode="auto")
    _clear_others(sim, keep={0})
    g = sim.game
    place(sim, 0, g.cage_xy[3] - [sim.prm.half[0, 0, 0] + 0.3, 0], 0.0)
    st, gl = idle(sim)
    st[:, 0, 0] = 0.4
    run(sim, st, gl, 10)
    assert sim.t[0] < g.auto_s and pts(sim, "G405/G418", 0) == 6


def test_rule_totals_add_up_to_fouls():
    sim = make_sim(n=6, seed=4)
    rng = np.random.default_rng(4)
    for k in range(400):
        if k % 10 == 0:
            st, gl = rng.uniform(-1, 1, (6, 6, 3)), rng.integers(0, len(sim.game.goals), (6, 6))
        sim.step(st, gl)
    assert sim.gs.fouls.sum() > 0
    assert np.allclose(sim.gs.foul_rule.sum(-1), sim.gs.fouls)
