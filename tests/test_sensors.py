import numpy as np

from conftest import idle, make_sim, place, run


def test_odometry_tracks_without_vision():
    sim = make_sim(sensors=True)
    sim.sens.v["max_dist_m"] = 0.0                    # no tags
    sim.full_mask[:] = True
    sim.prm.force_limit[:] = 0.8 * sim.prm.mu_s * sim.prm.mass * 9.81 / 4     # current limit below traction: no slip
    sim.sens.kf["init_std"] = [1e-4, 1e-4, 1e-4]
    place(sim, 0, [2.0, 5.6], 0.0)
    st, gl = idle(sim)
    st[:, 0] = [0.2, 0.0, 0.1]
    run(sim, st, gl, 15)
    err = np.linalg.norm(sim.sens.est[0, 0, :2] - sim.body.pose[0, 0, :2])
    assert not sim.body.slip.any() and err < 0.05 and sim.sens.var[0, 0, 0] > sim.sens.kf["init_std"][0] ** 2


def test_slip_degrades_odometry():
    sim = make_sim(sensors=True)
    sim.sens.v["max_dist_m"] = 0.0
    sim.full_mask[:] = True
    sim.prm.mu_s[:] = 0.3; sim.prm.mu_k[:] = 0.25
    st, gl = idle(sim)
    st[:, 0, 0] = 1.0
    run(sim, st, gl, 10)
    assert np.linalg.norm(sim.sens.est[0, 0, :2] - sim.body.pose[0, 0, :2]) > 0.2


def test_tags_visible_and_occluded():
    sim = make_sim(sensors=True)
    g = sim.game
    n = g.reef_n[0, 0]
    place(sim, 0, g.reef_fc[0, 0] + n * 2.0, np.arctan2(-n[1], -n[0]))        # 2 m in front of tag 18
    cnt, _ = sim.sens._visible(sim.body, np.array([0]), np.array([0]))
    assert cnt[0, 0] >= 1                                                       # front camera sees the reef
    place(sim, 1, g.reef_fc[0, 0] + n * 1.0, np.arctan2(-n[1], -n[0]))         # robot in between
    cnt2, _ = sim.sens._visible(sim.body, np.array([0]), np.array([0]))
    assert cnt2[0, 0] < cnt[0, 0]


def test_vision_shrinks_uncertainty_and_rejects_outliers():
    sim = make_sim(sensors=True)
    g = sim.game
    n = g.reef_n[0, 0]
    place(sim, 0, g.reef_fc[0, 0] + n * 2.0, np.arctan2(-n[1], -n[0]))
    sim.sens.est[0, 0, :2] += 0.3
    sim.sens.var[0, 0] = 0.5 ** 2
    st, gl = idle(sim)
    run(sim, st, gl, 10)
    assert np.sqrt(sim.sens.var[0, 0, 0]) < 0.1
    assert np.linalg.norm(sim.sens.est[0, 0, :2] - sim.body.pose[0, 0, :2]) < 0.1
    sim.sens.var[0, 0] = 0.01 ** 2
    sim.sens.v["ambiguity_p"] = 1.0                 # every single-tag view is an outlier
    run(sim, st, gl, 10)
    assert np.linalg.norm(sim.sens.est[0, 0, :2] - sim.body.pose[0, 0, :2]) < 0.1


def test_obs_shapes():
    sim = make_sim(n=2, sensors=True)
    o = sim.sensor_obs((0,))
    for k, v in o.items():
        assert v.dtype == np.float32 and v.shape[:2] == (2, 1) and np.isfinite(v).all(), k
