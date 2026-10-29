"""Motion layer = Systemcore Java implementation. Golden vectors in tests/golden/motion_v1.json are
the contract for the Java port (regenerate only on purpose: python tests/make_golden.py)."""
import json
from pathlib import Path

import numpy as np

from ghostsim import motion as m

GOLDEN = Path(__file__).parent / "golden" / "motion_v1.json"
OFF = np.array([[0.29, 0.29], [0.29, -0.29], [-0.29, 0.29], [-0.29, -0.29]])


def test_deadband_matches_wpilib():
    assert np.allclose(m.apply_deadband(np.array([0.04, 0.5, -0.5, 2.0]), 0.05), [0, 0.45 / 0.95, -0.45 / 0.95, 1])


def test_red_alliance_rotates_inputs():
    s = np.array([[1.0, 0.5, 0.3]] * 2)
    v = m.stick_to_field_speeds(s, np.array([0, 1]), np.array([4.0, 4.0]), np.array([9.0, 9.0]), 0.0)
    assert np.allclose(v[0, :2], -v[1, :2]) and np.isclose(v[0, 2], v[1, 2])


def test_discretize_identity_without_rotation_and_arc_with():
    v = np.array([3.0, -1.0, 0.0])
    assert np.allclose(m.discretize(v, 0.02), v)
    # constant twist over dt must end at the desired pose (vx dt, vy dt, w dt)
    v = np.array([2.0, 1.0, 3.0])
    tw = m.discretize(v, 0.02)
    dx, dy, dth = tw * 0.02
    s, c = np.sin(dth) / dth, (1 - np.cos(dth)) / dth
    end = [dx * s - dy * c, dx * c + dy * s, dth]
    assert np.allclose(end, v * 0.02, atol=1e-12)


def test_module_states_pure_rotation_and_zero_keeps_angle():
    spd, ang = m.to_module_states(np.array([0.0, 0.0, 1.0]), OFF, np.zeros(4))
    assert np.allclose(spd, np.hypot(0.29, 0.29))
    assert np.allclose(np.cos(ang) * OFF[:, 0] + np.sin(ang) * OFF[:, 1], 0)   # tangential
    prev = np.array([0.1, 0.2, 0.3, 0.4])
    spd, ang = m.to_module_states(np.zeros(3), OFF, prev)
    assert np.allclose(spd, 0) and np.allclose(ang, prev)


def test_desaturate_and_optimize():
    s = m.desaturate(np.array([2.0, 6.0, -3.0, 1.0]), np.array(4.0))
    assert np.isclose(np.abs(s).max(), 4.0) and np.isclose(s[0] / s[1], 2 / 6)
    spd, ang = m.optimize(np.array([1.0]), np.array([np.pi]), np.array([0.0]))
    assert np.allclose(spd, -1) and np.allclose(np.cos(ang), 1)


def test_golden_vectors():
    gold = json.loads(GOLDEN.read_text())
    for case in gold["cases"]:
        p = {k: np.asarray(v) for k, v in gold["params"].items()}
        spd, ang = m.motion_layer(np.array(case["stick"]), np.array(case["alliance"]), np.array(case["yaw"]),
                                  np.array(case["current_angle"]), np.array(case["prev_target_angle"]), p, gold["dt"])
        assert np.allclose(spd, case["speed"], atol=1e-9) and np.allclose(ang, case["angle"], atol=1e-9)


FILTER = Path(__file__).parent / "golden" / "cmd_filter_v1.json"


def test_command_filter_golden():
    gold = json.loads(FILTER.read_text())
    p, cmd = gold["params"], np.zeros(3)
    for st, want in zip(gold["stick"], gold["cmd"]):
        cmd = m.filter_stick(cmd, np.array(st), p["max_v"], p["max_w"], p["tau"], p["accel"], p["alpha"], gold["dt"])
        assert np.allclose(cmd, want, atol=1e-12)


def test_command_filter_limits_acceleration_and_smooths_jitter():
    max_v, max_w, tau, acc, alpha, dt = 3.8, 9.0, 0.15, 8.0, 30.0, 0.02
    f = lambda c, s: m.filter_stick(c, s, max_v, max_w, tau, acc, alpha, dt)
    # step to full forward: never faster than accel, reaches > 95 % within 1 s
    cmd, prev, hist = np.zeros(3), np.zeros(3), []
    for _ in range(50):
        cmd = f(cmd, np.array([1.0, 0.0, 1.0]))
        assert np.linalg.norm(cmd[:2] - prev[:2]) * max_v / dt <= acc + 1e-9
        assert abs(cmd[2] - prev[2]) * max_w / dt <= alpha + 1e-9
        prev = cmd.copy()
    assert cmd[0] > 0.95 and cmd[2] > 0.95
    # a policy flipping between +-0.3 every 0.1 s (5 periods): the command swings far less than the input
    cmd, xs = np.zeros(3), []
    for k in range(200):
        cmd = f(cmd, np.array([0.3 if (k // 5) % 2 == 0 else -0.3, 0.0, 0.0]))
        xs.append(cmd[0])
    assert np.ptp(xs[100:]) < 0.5 * 0.6
