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
