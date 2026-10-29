"""Scoring realism: stop to place, drive-to-pose time, misses, floor intakes, reef render poses."""
import numpy as np

from conftest import at_branch, idle, make_sim, place, run
from test_penalties import _clear_others


def gid(sim, name):
    return sim.game.goals.index(name)


def _ticks_to_score(sim, goal, max_ticks=40):
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, goal)
    for k in range(max_ticks):
        sim.step(st, gl)
        if not sim.gs.hold_coral[0, 0]:
            return k + 1
    return None


def test_no_scoring_while_driving_past_the_reef():
    """Driving along a reef face at 1 m/s with a coral and the L4 goal does not score; stopped in front of it does."""
    sim = make_sim(full=False)
    g = sim.game
    _clear_others(sim, keep={0})
    n, t = g.reef_n[0, 0], g.reef_t[0, 0]
    hx = sim.prm.half[0, 0, 0]
    place(sim, 0, g.reef_fc[0, 0] + n * (hx + 0.08) - t * 1.2, np.arctan2(-n[1], -n[0]))
    sim.gs.hold_coral[:, 0] = True
    st, gl = idle(sim)
    st[0, 0, :2] = t * 1.0 / sim.prm.max_v[0, 0]                              # blue: stick is field-relative
    sim.body.vel[0, 0, :2] = t * 1.0
    gl[:, 0] = gid(sim, "score_L4")
    run(sim, st, gl, 30)
    assert sim.gs.hold_coral[0, 0] and sim.score[0, 0] == 0
    at_branch(sim, 0, 0, 0, side=1)
    assert _ticks_to_score(sim, "score_L4") is not None and sim.score[0, 0] == 5


def test_drive_to_pose_time_grows_with_the_lateral_offset():
    """auto_align off: the dwell includes settle + distance / speed (game.json model.virtual_align)."""
    va = make_sim().game.model["virtual_align"]
    ticks = []
    for off in (0.0, 0.2):
        sim = make_sim(full=False)
        _clear_others(sim, keep={0})
        at_branch(sim, 0, 0, 0, side=1)
        sim.body.pose[:, 0, :2] += sim.game.reef_t[0, 0] * off                # 0.2 m beside the +t branch (outer side)
        sim.gs.reef_algae[:] = False
        sim.gs.hold_coral[:, 0] = True
        ticks.append(_ticks_to_score(sim, "score_L4"))
    extra = 0.2 / va["speed_mps"] / 0.1
    assert ticks[0] is not None and ticks[1] is not None
    assert extra - 1 <= ticks[1] - ticks[0] <= extra + 1, ticks


def test_floor_algae_is_picked_up_on_the_move():
    """Regression: the bumper used to push the ball (r 0.206 m) out of the 0.15 m intake, so it was never acquired."""
    sim = make_sim(full=False)
    g, gs = sim.game, sim.gs
    _clear_others(sim, keep={0})
    gs.algae.active[0], gs.stacked[0] = False, -1
    gs.algae.active[0, 6], gs.algae.pos[0, 6], gs.algae.vel[0, 6] = True, [5.0, 6.0], 0.0
    place(sim, 0, [5.0 - sim.prm.half[0, 0, 0] - 0.8, 6.0], 0.0)
    st, gl = idle(sim)
    st[0, 0, 0] = 1.0 / sim.prm.max_v[0, 0]
    gl[:, 0] = gid(sim, "algae_floor")
    run(sim, st, gl, 20)
    assert gs.hold_algae[0, 0] and not gs.algae.active[0, 6]


def test_missed_score_drops_the_coral_next_to_the_robot():
    sim = make_sim()
    gs = sim.gs
    at_branch(sim, 0, 0, 3, side=1)
    sim.prm.succ[:, 0, gid(sim, "score_L4")] = 0.0
    gs.hold_coral[:, 0] = True
    floor0 = gs.coral.active[0].sum()
    assert _ticks_to_score(sim, "score_L4") is not None                         # the dwell completes, then misses
    assert sim.score[0, 0] == 0 and not gs.branch[0].any() and gs.miss[0, 0] == 1
    assert gs.coral.active[0].sum() == floor0 + 1
    k = np.flatnonzero(gs.coral.active[0])[-1]
    d = np.linalg.norm(gs.coral.pos[0, k] - sim.body.pose[0, 0, :2])
    assert d < sim.prm.half[0, 0, 1] + 0.5


def test_missed_station_pickup_leaves_a_floor_coral():
    sim = make_sim()
    g, gs = sim.game, sim.gs
    c, n = g.station_c[0, 0], g.station_n[0, 0]
    place(sim, 0, c + n * (sim.prm.half[0, 0, 0] + 0.1), np.arctan2(-n[1], -n[0]))
    sim.prm.succ[:, 0, gid(sim, "coral_station")] = 0.0
    stock, floor0 = gs.stock[0, 0], gs.coral.active[0].sum()
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "coral_station")
    run(sim, st, gl, 10)
    assert not gs.hold_coral[0, 0] and gs.stock[0, 0] == stock - 1 and gs.coral.active[0].sum() == floor0 + 1


def test_pieces_conserved_with_misses():
    sim = make_sim(n=6, seed=5, perfect=False)
    sim.prm.succ[:] = 0.5
    rng = np.random.default_rng(5)
    for k in range(400):
        if k % 10 == 0:
            st, gl = rng.uniform(-1, 1, (6, 6, 3)), rng.integers(0, len(sim.game.goals), (6, 6))
        sim.step(st, gl)
    gs = sim.gs
    coral = gs.stock.sum(1) + gs.coral.active.sum(1) + gs.hold_coral.sum(1) + gs.branch.sum((1, 2, 3, 4)) \
        + gs.l1.sum((1, 2)) + gs.coral_lost
    algae = gs.reef_algae.sum((1, 2)) + gs.algae.active.sum(1) + gs.hold_algae.sum(1) + gs.hp_stock.sum(1) + gs.net.sum(1)
    assert (coral == 126).all() and (algae == 18).all()


def test_reef_render_poses_follow_the_cad():
    """Staged algae sit on the branch pair behind the tag plane (CAD-measured); L1 coral lie end to end in 2 layers."""
    g = make_sim().game
    IN = 0.0254
    for a in range(2):
        for f in range(6):
            n, t, fc = g.reef_n[a, f], g.reef_t[a, f], g.reef_fc[a, f]
            p = g.reef_algae_pose[a, f]
            assert abs((p[:2] - fc) @ n + 5.7 * IN) < 1e-6 and abs((p[:2] - fc) @ t) < 1e-6
            assert abs(p[2] - (52.1 if g.algae_high[f] else 36.3) * IN) < 1e-6
            l1 = g.l1_pose[a, f]                                               # (cap, 5)
            lat = (l1[:, :2] - fc) @ t
            for layer in range(2):
                s = np.sort(lat[layer * 3:(layer + 1) * 3])
                assert (np.diff(s) >= g.cfg["pieces"]["coral"]["length"] * IN).all()      # no overlap
                assert s[-1] - s[0] + g.cfg["pieces"]["coral"]["length"] * IN <= 2 * g.reef_ap * np.tan(np.pi / 6)
            assert (l1[3:, 2] > l1[:3, 2] + 0.09).all()                          # 2nd layer rests on the 1st
            assert ((l1[:, :2] - fc) @ n < 0).all()                              # inside the trough, not in front
