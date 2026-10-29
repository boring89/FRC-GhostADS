"""Live view: 1920x1080 layout, field camera framing, follow cameras, the overlay state (/api/live)."""
import numpy as np

from ghostsim import load
from ghostsim.sensors import rot_rpy
from ghosttrain.broadcast import (CANVAS, TILE, TOP, FollowCam, follow_cams, live_appearance, live_state, project,
                                  robot_rig, tag_corners, top_rig)

GAME = load("2025")


def test_layout_is_1080p_and_rendered_one_to_one():
    assert CANVAS == (1920, 1080) and TOP == (1920, 720) and TILE == (960, 360)
    assert TOP[1] + TILE[1] == CANVAS[1] and 2 * TILE[0] == CANVAS[0]


def _uv(rig, pts):
    t = rig["robot_to_camera"]
    R = rot_rpy(*np.radians(t[3:]))
    return project(pts, np.asarray(t[:3]), R, (rig["fx"], rig["fy"], rig["cx"], rig["cy"]), None)


def test_field_camera_frames_the_field_below_the_scoreboard():
    L, W = GAME.length, GAME.width
    rig = top_rig(GAME, *TOP)
    u, v, z = _uv(rig, np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0], [L / 2, W, 2.6]], float))
    assert (z > 0).all() and u.min() > 0 and u.max() < TOP[0] and v[:2].max() < TOP[1]
    assert (u[1] - u[0]) / TOP[0] > 0.85                               # near edge fills the width
    assert min(v[2:]) > 120                                            # far side clears the scoreboard overlay


def test_follow_camera_keeps_the_robot_centred_and_zooms_with_distance():
    cam = FollowCam([5.0, -2.6, 4.2])
    for xy in ([5.0, 2.0], [12.0, 7.5]):
        cam.robot = None
        rig = cam.rig(2, np.array(xy), 0.033)
        u, v, z = _uv(rig, np.array([[*xy, 0.55]]))
        assert z[0] > 0 and abs(u[0] - rig["cx"]) < 1 and abs(v[0] - rig["cy"]) < 1
    near = FollowCam([5.0, -2.6, 4.2]).rig(2, np.array([5.0, 2.0]), 0.0)
    far = FollowCam([5.0, -2.6, 4.2]).rig(2, np.array([12.0, 7.5]), 0.0)
    assert far["fx"] > near["fx"]                                      # longer lens on a far robot
    a, b = follow_cams(GAME)
    assert a.eye[1] < 0 and b.eye[1] < 0 and a.eye[0] < GAME.mid[0] < b.eye[0]   # outside the field, one per half


def test_live_state_for_the_overlay():
    status = [{"goal": "score_L4", "hold_coral": True, "hold_algae": False, "speed_mps": 1.23, "climbed": False}] * 6
    snap = {"t": 140.0, "score": np.array([120.0, 98.0]), "cats": np.zeros((2, 5)), "status": status}
    s = live_state(snap, 150.0, 1.2e6, [3, 1], {"blue": 0, "red": 4}, "baseline", None, 7)
    assert s["phase"] == "ENDGAME" and abs(s["t_left"] - 10.0) < 1e-9 and s["score"] == [120.0, 98.0]
    assert s["tiles"][0]["ours"] and s["tiles"][1]["robot"] == 5 and s["tiles"][0]["goal"] == "SCORE L4"
    assert s["red_name"] == "BASELINE" and s["card"] is None and s["match"] == 7
    assert live_state(dict(snap, t=3.0), 150.0, 0, [0, 0], {"blue": 1, "red": 3}, "self")["phase"] == "AUTO"
    c = live_state(None, 150.0, 0, [1, 0], {"blue": 0, "red": 3}, "self", (np.array([10.0, 20.0]), np.zeros((2, 5)), 0.0))
    assert c["card"]["winner"] == 1 and "score" not in c


def test_live_look_has_no_crowd():
    a = live_appearance({"crowd_density": 1.2, "carpet_rough": 0.1, "sun_energy": 0.3})
    assert a["crowd_density"] == 0.0 and a["sun_energy"] > 1.0


def test_robot_rig_intrinsics_scale_with_render_size():
    full, small = robot_rig((640, 480)), robot_rig((320, 240))
    assert full["fx"] == 500.0 and full["capture"] == [640, 480] and abs(small["fx"] - 250.0) < 1e-9


def test_tag_corners_are_square_and_on_the_field():
    corners, ids = tag_corners(GAME)
    assert corners.shape == (len(ids), 4, 3) and len(ids) > 0
    side = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=-1)
    assert np.allclose(side, side[0], rtol=1e-3)
