import numpy as np

from conftest import idle, make_sim
from ghostsim import SimCfg, VecSim


def test_full_match_teleop_and_auto():
    for mode, ticks in (("teleop", 1350), ("auto", 1500)):
        sim = VecSim(SimCfg(n_envs=2, sensors=False, start_mode=mode, seed=1))
        st, gl = idle(sim)
        info = None
        for k in range(ticks):
            info = sim.step(st, gl)
            assert not info.done.any() or k == ticks - 1
        assert info.done.all()


def test_leave_points_in_auto():
    sim = VecSim(SimCfg(n_envs=1, sensors=False, start_mode="auto", seed=2))
    sim.gs.hold_coral[:] = False
    st, gl = idle(sim)
    st[:, :, 0] = -1.0                                   # drive back toward own wall
    for _ in range(150):
        sim.step(st, gl)
    assert sim.score[0, 0] == 9 and sim.score[0, 1] == 9


def test_determinism_and_partial_reset():
    a, b = make_sim(n=3, seed=5), make_sim(n=3, seed=5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        st, gl = rng.uniform(-1, 1, (3, 6, 3)), rng.integers(0, 13, (3, 6))
        a.step(st, gl), b.step(st, gl)
    assert np.array_equal(a.body.pose, b.body.pose)
    before = a.body.pose.copy()
    a.reset(np.array([True, False, False]))
    assert np.array_equal(a.body.pose[1:], before[1:]) and not np.array_equal(a.body.pose[0], before[0])


def test_observation_and_render_shapes():
    sim = make_sim(n=2, sensors=True)
    pv = sim.privileged()
    assert pv["pose"].shape == (2, 6, 3) and pv["branch"].shape == (2, 2, 6, 2, 3)
    rs = sim.render_state()
    assert rs["robots"].shape == (2, 6, 4) and rs["coral"].shape == (2, 12 + 6 + 72 + 12 * sim.game.l1_cap, 6)   # floor, held, branches, trough
    assert rs["algae"].shape == (2, 36, 4) and rs["cages"].shape == (2, 6, 4)
    assert all(v.dtype == np.float32 for v in rs.values())
    assert rs["coral"][..., 5].sum() == 2 * 6                    # coral on marks visible


def test_env_wrapper_autoreset():
    from ghostsim.env import GhostEnv
    env = GhostEnv(SimCfg(n_envs=2, sensors=True, seed=1), agents=(0, 3))
    o = env.reset()
    assert o["sensor"]["pose"].shape == (2, 2, 3) and "images" not in o["sensor"]
    env.sim.t[0] = env.sim.end_t - 0.05
    st, gl = idle(env.sim)
    o, info, done = env.step({"stick": st, "goal": gl})
    assert done[0] and not done[1] and np.isfinite(info["final_score"][0]).all()
    assert env.sim.t[0] < env.sim.end_t - 10                          # env 0 restarted
