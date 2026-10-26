import numpy as np

from conftest import at_face, idle, make_sim, run
from ghostsim import SimCfg, VecSim
from ghostsim.robots import load_profile, sample_params

IN = 0.0254
GOALS = ["idle", "coral_station", "coral_floor", "algae_floor", "algae_reef", "score_L1", "score_L2", "score_L3",
         "score_L4", "processor", "net", "climb", "defend"]


def test_ours_matches_design_spec():
    rng = np.random.default_rng(0)
    p = sample_params([load_profile("ours.json")] * 6, 400, GOALS, rng)
    frame = (p.half[..., 0] * 2 - 2 * 3.25 * IN) / IN               # frame side in inches
    assert 25.0 <= frame.min() and frame.max() <= 27.0 and frame.std() > 0.3
    assert (p.half[..., 0] == p.half[..., 1]).all()                  # square
    assert p.levels.all() and p.net.all() and p.floor_coral.all() and p.floor_algae.all() and p.station.all()
    assert not p.processor.any() and not p.algae_reef.any() and (p.climb == 0).all()
    assert (p.max_v <= 0.95 * p.free_speed + 1e-9).all()
    assert 50 <= p.mass.min() and p.mass.max() <= 66
    assert (np.abs(p.offsets) < p.half[..., None, :] - 3.25 * IN).all()   # modules inside the frame


def test_nominal_values_without_randomisation():
    p = sample_params([load_profile("ours.json")] * 6, 3, GOALS, np.random.default_rng(0), randomize=False)
    assert np.allclose((p.half[..., 0] * 2 - 2 * 3.25 * IN) / IN, 26.0) and np.allclose(p.mass, 58.0)


def test_others_vary_capabilities():
    p = sample_params([load_profile("others.json")] * 6, 300, GOALS, np.random.default_rng(1))
    assert 0.3 < p.levels[..., 3].mean() < 0.9 and 0 < p.net.mean() < 1 and p.climb.max() == 2


def test_randomised_episodes_differ_per_reset():
    sim = VecSim(SimCfg(n_envs=4, sensors=False, seed=2))
    before = sim.prm.half.copy()
    sim.reset(np.array([True, False, False, False]))
    assert not np.allclose(sim.prm.half[0, 0], before[0, 0]) and np.allclose(sim.prm.half[1:], before[1:])


def test_elevator_robot_cannot_climb_or_use_processor():
    sim = make_sim(full=False)
    g = sim.game
    at_face(sim, 0, 0, 3)
    sim.gs.hold_coral[:, 0] = True
    st, gl = idle(sim)
    gl[:, 0] = g.goals.index("score_L4")
    run(sim, st, gl, 20)                                              # elevator reaches L4 (1.7 s)
    assert not sim.gs.hold_coral[0, 0]
    c, n = g.proc_c[0], g.proc_n[0]
    sim.body.pose[:, 0] = [*(c + n * (sim.prm.half[0, 0, 0] + 0.1)), np.arctan2(-n[1], -n[0])]
    sim.gs.hold_algae[:, 0] = True
    gl[:, 0] = g.goals.index("processor")
    run(sim, st, gl, 20)
    assert sim.gs.hold_algae[0, 0]                                    # no processor mechanism
