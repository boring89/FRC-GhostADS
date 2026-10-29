"""Renderer (Godot, skipped when it is not installed): robots are drawn in renders without a mechanism array (robot
cameras in training, replays: the randomised box robot) and only by the mechanism when one is sent (live view).
Regression: from eb3a764 on, the box robots were never built, so training images showed no robots at all."""
import os

import numpy as np
import pytest

from ghostrender.client import DEFAULT_BIN

BIN = os.environ.get("GODOT_BIN", str(DEFAULT_BIN))
pytestmark = pytest.mark.skipif(not os.path.exists(BIN), reason="Godot binary not installed")


def test_robots_visible_without_mechanism_and_drawn_only_by_it_with_one():
    from ghostrender import dr
    from ghostrender.client import RenderClient
    from ghostsim import SimCfg, VecSim
    from ghostsim.mechanism import SLOT_W, SLOTS
    from ghosttrain.broadcast import look_rig

    sim = VecSim(SimCfg(n_envs=1, sensors=False, seed=0, start_mode="teleop", randomize=False))
    P = np.array([6.4, 6.6])
    sim.body.pose[0, :, :2] = [P, [1.2, 7.4], [1.2, 6.2], [16.3, 0.7], [16.3, 1.9], [16.3, 3.1]]
    rng = np.random.default_rng(0)
    rig = look_rig([P[0] + 2.5, P[1] - 2.0, 1.6], [P[0], P[1], 0.4], 40.0, (320, 240))
    with RenderClient(godot_bin=BIN) as rc:
        rc.configure(sim.game, [rig], n_envs=1, width=320, height=240, max_cams=1, visual="proxy")
        rc.set_appearance(0, dr.sample_appearance(rng, sim.prm.half[0], sim.prm.height[0], sim.ours["bumper_z_m"]),
                          dr.sample_dr(rng))
        st = sim.render_state()
        shot = lambda s: rc.render(s, [(0, 0, 0)])[0][0].astype(int)
        with_robot = shot(st)
        away = dict(st, robots=st["robots"].copy())
        away["robots"][0, 0, :2] = [-30.0, -30.0]                     # robot 0 out of the picture
        without = shot(away)
        assert np.abs(with_robot - without).mean() > 2.0                # the box robot is drawn
        gone = dict(st, robots=st["robots"].copy())
        gone["robots"][0, :, :2] = [-30.0, -30.0]                       # every robot out of the picture
        empty = dict(st, mech=np.zeros((1, 6, SLOTS, SLOT_W), np.float32))
        assert np.abs(shot(empty) - shot(gone)).mean() < 0.3            # with a mechanism array only it draws robots
