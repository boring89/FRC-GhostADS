"""Odometry + vision estimator: honest under wheel slip (robot pushing into walls, cages, robots)."""
import numpy as np

from ghostsim import SimCfg, VecSim
from ghosttrain.baseline import Baseline


def _run(seconds=15, vision=True, seed=3):
    sim = VecSim(SimCfg(n_envs=4, sensors=True, seed=seed, start_mode="auto"))
    if not vision:
        sim.sens.v["sigma_xy_m"] = 1e9
    bl = Baseline(sim.game)
    for _ in range(int(round(seconds / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
    err = np.hypot(sim.sens.est[..., 0] - sim.body.pose[..., 0], sim.sens.est[..., 1] - sim.body.pose[..., 1])
    return sim, err


def test_estimate_tracks_truth_with_vision():
    _, err = _run()
    assert np.median(err) < 0.3 and err.max() < 1.5, (np.median(err), err.max())


def test_variance_is_honest_without_vision():
    # without vision the filter must not be overconfident: its 1-sigma should cover most of the real error
    sim, err = _run(vision=False)
    sigma = np.sqrt(sim.sens.var[..., 0] + sim.sens.var[..., 1])
    assert np.mean(err <= 2 * sigma) > 0.9
