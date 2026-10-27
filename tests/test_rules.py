"""REEFSCAPE rules (game/2025/rules.py) scenario tests."""
import numpy as np

from conftest import at_face, idle, make_sim, place, run
from ghostsim import SimCfg, VecSim


def gid(sim, name):
    return sim.game.goals.index(name)


def test_staging_matches_manual():
    sim = VecSim(SimCfg(n_envs=4, sensors=False, start_mode="auto"))
    gs, g = sim.gs, sim.game
    assert (gs.reef_algae.sum((1, 2)) == 12).all()                        # 6 per reef
    assert (gs.coral.active.sum(1) == 6).all() and (gs.algae.active.sum(1) == 6).all()
    assert ((gs.stacked >= 0).sum(1) == 6).all()                          # algae on top of coral marks
    assert (gs.stock == 57).all() and gs.hold_coral.all()                 # 3 preloads per alliance
    total = gs.stock.sum(1) + gs.coral.active.sum(1) + gs.hold_coral.sum(1)
    assert (total == 126).all()
    assert np.allclose(g.marks[0], [48 * 0.0254, 86.5 * 0.0254])


def test_score_l4_and_points(sim):
    at_face(sim, 0, 0, 3)
    sim.gs.hold_coral[:, 0] = True
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "score_L4")
    run(sim, st, gl, 15)
    assert not sim.gs.hold_coral[0, 0]
    assert sim.gs.branch[0, 0, 3, :, 2].sum() == 1
    assert sim.score[0, 0] == 5                                             # teleop L4


def test_algae_blocks_level(sim):
    g = sim.game
    low_face = int(np.flatnonzero(~g.algae_high)[0])
    at_face(sim, 0, 0, low_face)
    sim.gs.hold_coral[:, 0] = True
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "score_L2")
    run(sim, st, gl, 15)
    assert sim.gs.hold_coral[0, 0]                                         # blocked by low algae
    gl[:, 0] = gid(sim, "algae_reef")
    run(sim, st, gl, 12)
    assert sim.gs.hold_algae[0, 0] and not sim.gs.reef_algae[0, 0, low_face]
    gl[:, 0] = gid(sim, "score_L2")
    run(sim, st, gl, 12)
    assert not sim.gs.hold_coral[0, 0]


def test_l1_capacity_and_both_branches(sim):
    at_face(sim, 0, 0, 3)
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "score_L1")
    for _ in range(5):
        sim.gs.hold_coral[:, 0] = True
        run(sim, st, gl, 8)
    assert sim.gs.l1[0, 0, 3] == sim.game.l1_cap
    gl[:, 0] = gid(sim, "score_L4")
    for _ in range(3):
        sim.gs.hold_coral[:, 0] = True
        run(sim, st, gl, 15)
    assert sim.gs.branch[0, 0, 3, :, 2].all() and sim.gs.hold_coral[0, 0]   # both L4 branches full


def test_must_face_the_reef(sim):
    at_face(sim, 0, 0, 3)
    sim.body.pose[:, 0, 2] += np.pi / 2
    sim.gs.hold_coral[:, 0] = True
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "score_L4")
    run(sim, st, gl, 20)
    assert sim.gs.hold_coral[0, 0]


def test_station_intake(sim):
    g = sim.game
    c, n = g.station_c[0, 0], g.station_n[0, 0]
    place(sim, 0, c + n * (sim.prm.half[0, 0, 0] + 0.1), np.arctan2(-n[1], -n[0]))
    stock = sim.gs.stock[0, 0]
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "coral_station")
    run(sim, st, gl, 10)
    assert sim.gs.hold_coral[0, 0] and sim.gs.stock[0, 0] == stock - 1


def test_processor_then_opponent_human_player(sim):
    g = sim.game
    c, n = g.proc_c[0], g.proc_n[0]
    place(sim, 0, c + n * (sim.prm.half[0, 0, 0] + 0.1), np.arctan2(-n[1], -n[0]))
    sim.gs.hold_algae[:, 0] = True
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "processor")
    run(sim, st, gl, 8)
    assert sim.score[0, 0] == 6 and sim.gs.hp_stock[0, 1] == 1
    run(sim, st, np.zeros_like(gl), 40)
    assert sim.gs.hp_stock[0, 1] == 0
    assert sim.score[0, 1] in (0, 4)                                        # red HP either scored or missed


def test_climb_deep_and_park():
    sim = make_sim()
    g = sim.game
    place(sim, 0, g.cage_xy[0] - [sim.prm.half[0, 0, 0] + 0.15, 0], 0.0)
    place(sim, 1, [g.mid[0] - 0.2, g.mid[1] + 3.2], 0.0)                     # inside blue barge zone
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "climb")
    run(sim, st, gl, 40)
    assert sim.gs.climbed[0, 0]
    st[:, 0, 0] = 1.0
    x0 = sim.body.pose[0, 0, 0]
    sim.t[:] = sim.end_t - 0.5
    info = run(sim, st, gl, 5)
    assert info.done.all() and abs(sim.body.pose[0, 0, 0] - x0) < 1e-9     # hanging robots do not move
    assert sim.score[0, 0] == 12 + 2


def test_zone_protection_foul(sim):
    g = sim.game
    at_face(sim, 0, 0, 0, gap=0.05)                                        # blue robot in its reef zone
    hx = sim.prm.half[0, 3, 0]
    n = g.reef_n[0, 0]
    p = g.reef_fc[0, 0] + n * (sim.prm.half[0, 0, 0] * 2 + 0.05 + hx + 0.3)
    place(sim, 3, p, np.arctan2(-n[1], -n[0]))                              # red robot behind it
    st, gl = idle(sim)
    st[:, 3, 0] = -1.0                                                     # red forward is -x -> toward the blue reef
    run(sim, st, gl, 10)
    assert sim.score[0, 0] >= 6 and sim.gs.fouls[0, 1] >= 6


def _bounce_fouls(sim, steps=120):
    """Red robot rams the protected blue robot, backs off, rams again, ... Returns the foul points assessed."""
    g = sim.game
    at_face(sim, 0, 0, 0, gap=0.05)
    hx = sim.prm.half[0, 3, 0]
    n = g.reef_n[0, 0]
    p = g.reef_fc[0, 0] + n * (sim.prm.half[0, 0, 0] * 2 + 0.05 + hx + 0.3)
    place(sim, 3, p, np.arctan2(-n[1], -n[0]))
    st, gl = idle(sim)
    for k in range(steps):
        st[:, 3, 0] = -1.0 if (k // 10) % 2 == 0 else 1.0                    # 1 s in, 1 s out
        run(sim, st, gl, 1)
    return sim.gs.fouls[0, 1]


def test_foul_cooldown_per_pair(sim):
    capped = _bounce_fouls(sim)
    assert 6 <= capped <= 6 * (12 // 3 + 1)                                # <= 1 foul per 3 s
    sim.game.model["foul_cooldown_s"] = 0.0
    sim.reset()
    free = _bounce_fouls(sim)
    assert free > capped


def test_one_defender_rule(sim):
    g = sim.game
    place(sim, 0, [g.mid[0] + 2.0, 1.0], 0.0)
    place(sim, 1, [g.mid[0] + 2.0, 7.0], 0.0)
    st, gl = idle(sim)
    run(sim, st, gl, 32)                                                   # 3.2 s: minor at onset + 1 major
    assert sim.gs.fouls[0, 0] == 2 + 6 and sim.score[0, 1] == 8


def test_opponent_cage_foul(sim):
    g = sim.game
    place(sim, 0, g.cage_xy[3] - [sim.prm.half[0, 0, 0] + 0.3, 0], 0.0)   # blue robot next to a red cage
    st, gl = idle(sim)
    st[:, 0, 0] = 0.4
    run(sim, st, gl, 10)
    assert sim.gs.fouls[0, 0] >= 6


def test_knock_algae_off_coral_mark_then_intake(sim):
    g = sim.game
    m = g.marks[0]
    place(sim, 0, m - [sim.prm.half[0, 0, 0] + 0.5, 0], 0.0)
    st, gl = idle(sim)
    gl[:, 0] = gid(sim, "coral_floor")
    st[:, 0, 0] = 0.25
    run(sim, st, gl, 30)
    assert sim.gs.hold_coral[0, 0]
    assert (sim.gs.stacked[0] >= 0).sum() == 5


def test_pieces_conserved_random_rollout():
    sim = make_sim(n=6, seed=3)
    rng = np.random.default_rng(3)
    for k in range(400):
        if k % 10 == 0:
            st, gl = rng.uniform(-1, 1, (6, 6, 3)), rng.integers(0, len(sim.game.goals), (6, 6))
        sim.step(st, gl)
    gs = sim.gs
    coral = gs.stock.sum(1) + gs.coral.active.sum(1) + gs.hold_coral.sum(1) + gs.branch.sum((1, 2, 3, 4)) + gs.l1.sum((1, 2))
    algae = gs.reef_algae.sum((1, 2)) + gs.algae.active.sum(1) + gs.hold_algae.sum(1) + gs.hp_stock.sum(1) + gs.net.sum(1)
    assert (coral == 126).all() and (algae == 18).all()


def test_category_points_add_up_to_score():
    """Per-category points (LEAVE/CORAL/ALGAE/BARGE/PENALTY) must add up to the match score, both alliances."""
    from ghostsim import SimCfg, VecSim
    from ghosttrain.baseline import Baseline
    sim = VecSim(SimCfg(n_envs=8, sensors=False, start_mode="auto", seed=9))
    bl = Baseline(sim.game)
    for _ in range(int(round(sim.end_t / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
    cat = sim.gs.cat.sum(-1)
    assert np.allclose(cat, sim.score, atol=1e-6), (cat[0], sim.score[0])
    assert sim.gs.cat[..., 1].sum() > 0 and sim.gs.cat[..., 2].sum() > 0 and sim.gs.cat[..., 0].sum() > 0
