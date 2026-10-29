import numpy as np

from conftest import idle, make_sim, place, run
from ghostsim import geometry as geo


def test_accel_top_speed_and_battery_sag(sim):
    st, gl = idle(sim)
    st[:, 0, 0] = 1.0
    vs, vb = [], []
    for _ in range(15):
        sim.step(st, gl)
        vs.append(sim.body.vel[0, 0, 0]); vb.append(sim.body.vbatt[0, 0])
    vs = np.array(vs)
    g = 9.81 * sim.prm.mu_s[0, 0]
    t3 = (np.argmax(vs > 3.0) + 1) * 0.1
    assert 3.0 / g < t3 < 1.0                                   # traction-limited, not instantaneous
    assert np.isclose(vs[-1], sim.prm.max_v[0, 0], atol=1e-3)
    assert min(vb) < sim.prm.v_oc[0, 0] - 0.5                     # sag while accelerating


def test_wheel_slip_shows_in_encoders(sim):
    sim.prm.mu_s[:] = 0.3
    sim.prm.mu_k[:] = 0.25
    sim.prm.cmd_tau[:], sim.prm.cmd_accel[:], sim.prm.cmd_alpha[:] = 1e-6, 1e6, 1e6   # physics only: filter pass-through
    st, gl = idle(sim)
    st[:, 0, 0] = 1.0
    sim.step(st, gl)
    assert sim.body.slip[0, 0].all()
    assert (sim.body.wheel[0, 0] - sim.body.vel[0, 0, 0] > 0.5).all()


def _face_off(sim, mu_blue=None, gap=0.3):
    """Open lane y=1.2 between the blue reef and the scoring-table wall."""
    place(sim, 0, [2.0, 5.5], 0.0)
    place(sim, 1, [3.4, 1.2], 0.0)
    place(sim, 4, [3.4 + 2 * sim.prm.half[0, 1, 0] + gap, 1.2], np.pi)
    if mu_blue:
        sim.prm.mu_s[:, 1] *= mu_blue; sim.prm.mu_k[:, 1] *= mu_blue
    st, gl = idle(sim)
    st[:, 1, 0] = 1.0
    st[:, 4, 0] = 1.0
    return st, gl


def test_equal_robots_stalemate():
    sim = make_sim()
    st, gl = _face_off(sim)
    run(sim, st, gl, 30)
    x1 = sim.body.pose[0, 1, 0]
    run(sim, st, gl, 20)
    assert abs(sim.body.pose[0, 1, 0] - x1) < 0.05


def test_more_traction_wins_and_braking_resists():
    sim = make_sim()
    st, gl = _face_off(sim, mu_blue=1.4)
    run(sim, st, gl, 5)
    x4 = sim.body.pose[0, 4, 0]
    run(sim, st, gl, 10)
    assert sim.body.pose[0, 4, 0] - x4 > 0.5
    sim = make_sim()
    st, gl = _face_off(sim, gap=0.002)                              # already in contact, both at rest
    st[:, 4] = 0                                                    # red only brakes
    run(sim, st, gl, 20)
    x4 = sim.body.pose[0, 4, 0]
    run(sim, st, gl, 20)
    assert sim.body.pose[0, 4, 0] - x4 < 0.1                         # same traction: barely moves


def test_off_centre_hit_spins_target():
    sim = make_sim()
    place(sim, 0, [2.0, 5.5], 0.0)
    place(sim, 1, [3.0, 1.2], 0.0)
    place(sim, 4, [4.5, 1.55], 0.0)
    sim.prm.mu_s[:, 4] = 0.2; sim.prm.mu_k[:, 4] = 0.2
    st, gl = idle(sim)
    st[:, 1, 0] = 1.0
    run(sim, st, gl, 10)
    assert abs(geo.wrap(sim.body.pose[0, 4, 2])) > 0.05


def test_walls_reef_and_no_overlap_random():
    sim = make_sim(n=8, seed=4)
    rng = np.random.default_rng(4)
    worst = 0.0
    for k in range(300):
        if k % 10 == 0:
            st = rng.uniform(-1, 1, (8, 6, 3))
        sim.step(st, np.zeros((8, 6), int))
        b, p = sim.body, sim.prm
        th = b.pose[..., 2]
        hd, _ = geo.halfplane_box(b.pose[..., :2], p.half, np.cos(th), np.sin(th), sim.game.hp_n, sim.game.hp_off, False)
        od, _, _ = geo.box_poly_sat(b.pose[..., :2], p.half, np.cos(th), np.sin(th), sim.game.polys, False)
        worst = max(worst, hd.max(), od.max())
    assert np.isfinite(sim.body.pose).all() and worst < 0.05
    assert (np.linalg.norm(sim.body.vel[..., :2], axis=-1) <= sim.prm.free_speed * 1.05).all()
