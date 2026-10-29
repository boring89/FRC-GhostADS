"""Baseline uses the full algae mechanism: reef removal, floor pick-up, net (preferred, as at the 2025 Champs) and
processor."""
import numpy as np

from ghostsim import SimCfg, VecSim
from ghosttrain.baseline import Baseline


def _algae_match(**kw):
    sim = VecSim(SimCfg(n_envs=8, sensors=False, start_mode="auto", seed=21, slots=("ours",) * 6))
    assert sim.prm.algae_reef.all() and sim.prm.processor.all() and sim.prm.net.all()   # our robot spec (fixed)
    bl = Baseline(sim.game, **kw)
    reef0 = sim.gs.reef_algae.sum((1, 2)).copy()
    for _ in range(int(round(sim.end_t / 0.1))):
        sim.step(*bl.act(sim, bl.plan(sim)))
    return sim, (reef0 - sim.gs.reef_algae.sum((1, 2))).sum()


def test_baseline_removes_reef_algae_and_scores_net_first():
    sim, removed = _algae_match()
    assert removed > 0, "baseline never removed algae from the reef"
    assert sim.gs.scored_algae[..., 1].sum() > 0 and sim.gs.scored_algae[..., 0].sum() == 0   # net-capable: net only
    sim, _ = _algae_match(prefer_net=False)
    assert sim.gs.processor.sum() > 0, "processor-first baseline never scored in a processor"


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


def test_baseline_reaches_a_reef_face_from_the_side():
    """Regression: driving straight at a reef approach point from beside the reef wedged the robot on a reef vertex
    for the rest of the match; the baseline now goes through a point in front of the face."""
    import sys
    sys.path.insert(0, "tests")
    from conftest import make_sim, place
    from test_penalties import _clear_others
    sim = make_sim(full=False)
    g = sim.game
    _clear_others(sim, keep={0})
    n1 = g.reef_n[0, 1]
    place(sim, 0, g.reef_c[0] + n1 * 1.55, 0.0)                      # beside face 1, target on face 0
    sim.gs.hold_coral[:, 0] = False
    sim.gs.reef_algae[0, 0, :] = False
    sim.gs.reef_algae[0, 0, 0] = True                                # only face 0 still has algae
    sim.prm.algae_reef[:, 0] = True
    sim.prm.station[:, 0] = sim.prm.floor_coral[:, 0] = sim.prm.floor_algae[:, 0] = False
    bl = Baseline(sim.game)
    for _ in range(60):
        st, gl = bl.act(sim, bl.plan(sim))
        st[:, 1:], gl[:, 1:] = 0.0, 0
        sim.step(st, gl)
        if sim.gs.hold_algae[0, 0]:
            break
    assert sim.gs.hold_algae[0, 0] and not sim.gs.reef_algae[0, 0, 0]
