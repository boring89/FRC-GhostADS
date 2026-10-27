"""Baseline uses the full algae mechanism: reef removal, floor pick-up, processor (preferred) and net."""
import numpy as np

from ghostsim import SimCfg, VecSim
from ghosttrain.baseline import Baseline


def test_baseline_removes_reef_algae_and_scores_processor():
    sim = VecSim(SimCfg(n_envs=8, sensors=False, start_mode="auto", seed=21, slots=("ours",) * 6))
    assert sim.prm.algae_reef.all() and sim.prm.processor.all()          # our robot spec (fixed)
    bl = Baseline(sim.game)
    reef0 = sim.gs.reef_algae.sum((1, 2)).copy()
    for _ in range(int(round(sim.end_t / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
    removed = (reef0 - sim.gs.reef_algae.sum((1, 2))).sum()
    assert removed > 0, "baseline never removed algae from the reef"
    assert sim.gs.processor.sum() > 0, "baseline never scored in a processor"


def test_baseline_keeps_out_of_opponent_cages():
    """Opponent cages are off-limits in TELEOP (G418): the baseline must not touch them while chasing algae."""
    from ghostsim.robots import ALLIANCE, R
    sim = VecSim(SimCfg(n_envs=8, sensors=False, start_mode="auto", seed=21, slots=("ours",) * 6))
    bl = Baseline(sim.game)
    own = np.arange(6)[None, :] // 3 == ALLIANCE[:, None]
    prev = np.zeros((8, R, 6), bool)
    onsets = 0
    for _ in range(int(round(sim.end_t / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
        cc = sim.body.poly_contact[..., sim.game.cage_poly]
        onsets += int((cc & ~own[None] & ~prev).sum())
        prev = cc.copy()
    assert onsets <= 8 * 2, f"{onsets} opponent-cage contacts in 8 matches"   # ~0.1 per match measured
