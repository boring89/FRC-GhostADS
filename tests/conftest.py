import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", message=".*unverified placeholder.*")

from ghostsim import SimCfg, VecSim  # noqa: E402
from ghostsim.robots import R  # noqa: E402

_L, _W = 690.875 * 0.0254, 317.0 * 0.0254
_BLUE = np.array([[2.5, 1.0], [2.5, 7.05], [6.5, 7.0]])
PARK = np.concatenate([_BLUE, [_L, _W] - _BLUE])


def make_sim(n=1, sensors=False, seed=0, full=True, perfect=True, **kw):
    """full=True: every robot can do everything (rules tests). full=False: the actual design spec.
    perfect=True: every action succeeds (success_p = 1), so scenario tests are deterministic."""
    kw = {"randomize": False, "start_mode": "teleop", **kw}          # scenario tests: teleop unless asked otherwise
    sim = VecSim(SimCfg(n_envs=n, seed=seed, sensors=sensors, slots=("ours",) * 6, **kw))
    if full:
        p = sim.prm
        p.processor[:], p.algae_reef[:], p.climb[:] = True, True, 2
        sim.gs.cage_deep[:] = True
        p.times[:] = np.minimum(p.times, 1.0)
    if perfect:
        sim.prm.succ[:] = 1.0
    park(sim)
    return sim


def park(sim):
    b = sim.body
    b.pose[:, :, :2] = PARK
    b.pose[:, :3, 2], b.pose[:, 3:, 2] = 0.0, np.pi
    b.vel[:] = 0
    gs = sim.gs
    gs.stock[:, 0] += gs.hold_coral[:, :3].sum(1)
    gs.stock[:, 1] += gs.hold_coral[:, 3:].sum(1)
    gs.hold_coral[:], gs.hold_algae[:] = False, False
    if sim.sens:
        sim.sens.reset(np.arange(sim.cfg.n_envs), b, sim.prm)


def place(sim, r, xy, yaw):
    sim.body.pose[:, r] = [xy[0], xy[1], yaw]
    sim.body.vel[:, r] = 0
    if sim.sens:
        sim.sens.reset(np.arange(sim.cfg.n_envs), sim.body, sim.prm)


def at_face(sim, r, a, f, gap=0.1):
    g = sim.game
    hx = sim.prm.half[0, r, 0]
    n = g.reef_n[a, f]
    place(sim, r, g.reef_fc[a, f] + n * (hx + gap), np.arctan2(-n[1], -n[0]))


def at_branch(sim, r, a, f, side=1, gap=0.1):
    """Robot in front of face f, centred on one of its two branches (side -1 or +1). L2-L4 scoring needs this."""
    g = sim.game
    hx = sim.prm.half[0, r, 0]
    n, t = g.reef_n[a, f], g.reef_t[a, f]
    half_pipe = g.cfg["field"]["reef"]["pipe_spacing"] * 0.0254 / 2
    xy = g.reef_fc[a, f] + n * (hx + gap) + t * side * half_pipe
    place(sim, r, xy, np.arctan2(-n[1], -n[0]))


def idle(sim):
    return np.zeros((sim.cfg.n_envs, R, 3)), np.zeros((sim.cfg.n_envs, R), int)


def run(sim, stick, goal, ticks):
    info = None
    for _ in range(ticks):
        info = sim.step(stick, goal)
    return info


@pytest.fixture
def sim():
    return make_sim()
