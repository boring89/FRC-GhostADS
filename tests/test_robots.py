import numpy as np

from conftest import at_branch, idle, make_sim, run
from ghostsim import SimCfg, VecSim
from ghostsim.robots import load_profile, sample_params

IN = 0.0254
GOALS = ["idle", "coral_station", "coral_floor", "algae_floor", "algae_reef", "score_L1", "score_L2", "score_L3",
         "score_L4", "processor", "net", "climb", "defend"]


def test_ours_matches_design_spec():
    rng = np.random.default_rng(0)
    p = sample_params([load_profile("ours.json")] * 6, 400, GOALS, rng)
    frame = (p.half[..., 0] * 2 - 2 * 3.25 * IN) / IN               # frame side in inches
    assert 29.0 <= frame.min() and frame.max() <= 30.0 and frame.std() > 0.1     # 254 binder: 29.5 in frame
    assert (p.half[..., 0] == p.half[..., 1]).all()                  # square
    assert p.levels.all() and p.net.all() and p.floor_algae.all() and p.station.all()
    assert not p.floor_coral.any() and (p.station_side == -1).all()        # 6328 layout: back funnel, no coral ground intake
    assert p.processor.all() and p.algae_reef.all() and (p.climb == 2).all()     # 254: deep cage climb (binder p.17)
    assert (p.processor == p.processor[0, 0]).all() and (p.algae_reef == p.algae_reef[0, 0]).all()   # fixed, not randomised
    assert (p.max_v <= 0.95 * p.free_speed + 1e-9).all()
    assert 50 <= p.mass.min() and p.mass.max() <= 66
    assert (np.abs(p.offsets) < p.half[..., None, :] - 3.25 * IN).all()   # modules inside the frame


def test_nominal_values_without_randomisation():
    p = sample_params([load_profile("ours.json")] * 6, 3, GOALS, np.random.default_rng(0), randomize=False)
    assert np.allclose((p.half[..., 0] * 2 - 2 * 3.25 * IN) / IN, 29.5) and np.allclose(p.mass, 58.0)


def test_others_vary_capabilities():
    p = sample_params([load_profile("others.json")] * 6, 300, GOALS, np.random.default_rng(1))
    assert 0.3 < p.levels[..., 3].mean() < 0.9 and 0 < p.net.mean() < 1 and p.climb.max() == 2


def test_randomised_episodes_differ_per_reset():
    sim = VecSim(SimCfg(n_envs=4, sensors=False, seed=2))
    before = sim.prm.half.copy()
    sim.reset(np.array([True, False, False, False]))
    assert not np.allclose(sim.prm.half[0, 0], before[0, 0]) and np.allclose(sim.prm.half[1:], before[1:])


def test_elevator_robot_cannot_climb_but_uses_processor():
    sim = make_sim(full=False)
    g = sim.game
    at_branch(sim, 0, 0, 3, side=1)
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
    assert not sim.gs.hold_algae[0, 0]                                # processor scored (algae released)
    assert sim.gs.processor[0, 0] == 1
    assert sim.prm.climb[0, 0] == 2 and not sim.gs.climbed[0, 0]      # deep climb capable (254), not climbing here

def test_others_quality_makes_strong_robots_strong_in_every_respect():
    """robots/others.json: one quality q per robot drives mechanism time, alignment, speed, capabilities and success
    together (2025 Champs pool); q never exceeds our robot (FRC 254 numbers)."""
    p = sample_params([load_profile("others.json")] * 6, 400, GOALS, np.random.default_rng(3))
    ours = sample_params([load_profile("ours.json")] * 6, 1, GOALS, np.random.default_rng(3), randomize=False)
    q = p.quality.reshape(-1)
    assert 0.0 <= q.min() and q.max() <= 0.95
    t_l4 = p.times[..., GOALS.index("score_L4")].reshape(-1)
    for v, sign in ((t_l4, -1), (p.align_scale.reshape(-1), -1), (p.max_v.reshape(-1), 1),
                    (p.succ[..., GOALS.index("score_L4")].reshape(-1), 1)):
        assert sign * np.corrcoef(q, v)[0, 1] > 0.8
    assert t_l4.min() >= ours.times[0, 0, GOALS.index("score_L4")] - 1e-9                 # nobody faster than 254
    assert (p.align_scale >= 1.0).all()
    hi, lo = q > 0.7, q < 0.3
    assert p.levels[..., 3].reshape(-1)[hi].mean() > p.levels[..., 3].reshape(-1)[lo].mean()
    assert (p.climb == 1).mean() < 0.05 and 0.55 < (p.climb == 2).mean() < 0.85             # deep cages at Champs
