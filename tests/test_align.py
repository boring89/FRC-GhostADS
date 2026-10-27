"""Auto-align to a reef branch (ghostsim/align.py): converges within tolerance, dwell time, baseline still scores."""
import numpy as np

from ghostsim import SimCfg, VecSim
from ghostsim.robots import R


def _sim(seed=5):
    sim = VecSim(SimCfg(n_envs=1, sensors=False, seed=seed, randomize=False, start_mode="teleop", slots=("ours",) * 6))
    sim.prm.levels[:] = True
    sim.gs.hold_coral[:] = True
    return sim


def test_aligns_within_tolerance_and_dwells():
    sim = _sim(seed=5)
    g = sim.game
    gl = np.zeros((1, R), int)
    gl[:] = g.gi["align_score"]
    st = np.zeros((1, R, 3))
    sim.body.pose[0, 0] = [g.reef_fc[0, 0, 0] + 2.2 * g.reef_n[0, 0, 0], g.reef_fc[0, 0, 1] + 2.2 * g.reef_n[0, 0, 1], 0.0]
    first = None
    for k in range(300):                                  # 30 s
        sim.step(st, gl)
        if sim.gs.aligned[0, 0] and first is None:
            first = k
            break
    assert first is not None, "robot never aligned"
    tgt = sim.gs.align_tgt[0, 0]
    assert np.linalg.norm(tgt[:2] - sim.body.pose[0, 0, :2]) <= g.model["align_tol_m"] + 1e-6
    assert abs(np.arctan2(np.sin(tgt[2] - sim.body.pose[0, 0, 2]), np.cos(tgt[2] - sim.body.pose[0, 0, 2]))) \
        <= np.radians(g.model["align_tol_deg"]) + 1e-6
    # dwell: the aligned flag is only set after 0.7 s continuously inside tolerance
    need = sim.prm.times[0, 0, g.gi["align_score"]]
    assert sim.gs.align_t[0, 0] >= need - 1e-6


def test_score_fires_only_on_the_aligned_branch():
    sim = _sim(seed=5)
    g = sim.game
    gl = np.zeros((1, R), int)
    gl[:] = g.gi["align_score"]
    st = np.zeros((1, R, 3))
    sim.body.pose[0, 0] = [g.reef_fc[0, 0, 0] + 2.2 * g.reef_n[0, 0, 0], g.reef_fc[0, 0, 1] + 2.2 * g.reef_n[0, 0, 1], 0.0]
    for _ in range(300):
        sim.step(st, gl)
        if sim.gs.aligned[0, 0]:
            break
    assert sim.gs.aligned[0, 0]
    gl[:] = g.gi["score_L4"]
    for _ in range(25):
        sim.step(st, gl)
    assert sim.gs.branch[0, 0].any(), "aligned robot did not place coral on the aligned branch"


def test_baseline_with_align_still_scores_branches():
    from ghosttrain.baseline import Baseline
    sim = VecSim(SimCfg(n_envs=8, sensors=False, start_mode="auto", seed=21))
    bl = Baseline(sim.game)
    for _ in range(int(round(sim.end_t / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
    assert sim.gs.branch.sum() > 8 * 4, "baseline placed too few coral on the reef"
    assert np.allclose(sim.gs.cat.sum(-1), sim.score)
