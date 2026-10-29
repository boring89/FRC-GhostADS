"""Geometry checks of the mechanism model (ghostsim/mechanism.py, FRC 6328's 2025 robot from its CAD)."""
import numpy as np
import pytest

from ghostsim.mechanism import (ALGAE_R, AXIS, BLUE, BODIES, BUMPER_T, CAD_FRAME, CLIMB, CORAL_LEN, CORAL_R, EFF_PIVOT,
                                FRAME, H_MAX, PERIMETER_HALF, RED, SLOTS, STAGES, STOW, STOW_TOP_Z, WRIST, Body, Box,
                                Cyl, Sphere, algae_held, algae_target, body_transforms, coral_held, coral_pitch,
                                funnel_pose, pose, rot_y, scale_of, slots_for, solid_rows)
from ghostsim.mechanism_state import CORAL_AT, REEF_ALGAE_Z, SCORE

IN = 0.0254
PIECES = ("coral", "algae")
# the poses the live view moves through (ghostsim/mechanism_state.py target(): elevator and wrist together, eased)
GOAL_POSES = [*SCORE.values(), *((algae_target(z), WRIST["reach"]) for z in REEF_ALGAE_Z.values()),
              (H_MAX, WRIST["net"]), (0.0, WRIST["processor"]), (0.0, WRIST["floor"])]


def robot(solids):
    return [s for s in solids if s.name not in PIECES]


@pytest.mark.parametrize("u", np.linspace(0, 1, 5))
@pytest.mark.parametrize("hold", [None, "coral", "algae", "both"])
def test_operating_range_keeps_everything_above_the_floor(u, hold):
    for h, w in GOAL_POSES:
        for climb in CLIMB.values():
            assert min(s.min_z() for s in pose(h=h * u, wrist=w * u, climb=climb, hold=hold)) > -1e-6, (h, w, climb)
    for h in np.linspace(0, H_MAX, 5):                    # the wrist ahead of a lagging elevator (rate limits)
        assert min(s.min_z() for s in pose(h=h, wrist=-50.0 * u, hold=hold if h > 0.05 else None)) > -1e-6


def test_starting_configuration_height_r104():
    # R104: starting configuration no taller than 3 ft 6 in (1.067 m); the CAD stands 0.94 m
    assert max(s.max_z() for s in pose(**STOW)) <= 1.067 and STOW_TOP_Z <= 1.067


def test_extension_within_r105():
    # R105: no more than 1 ft 6 in beyond the frame perimeter (bumpers excluded); game pieces are not robot parts
    lim = PERIMETER_HALF + 18 * IN + 1e-9
    for h, w in [*SCORE.values(), (0.0, WRIST["floor"]), (H_MAX, WRIST["net"]), (algae_target(1.4), WRIST["reach"])]:
        for climb in CLIMB.values():
            parts = robot(pose(h=h, wrist=w, climb=climb))
            assert max(s.max_x() for s in parts) <= lim and min(s.min_x() for s in parts) >= -lim


def test_elevator_is_full_width_and_leans_forward():
    """The fixed stage spans the frame (tubes at the frame edge); the top comes forward as the elevator rises, so at
    L4 the coral reaches over the front bumper toward the reef while the bottom stays put."""
    outer = [s for s in pose() if s.name == "outer"]
    assert max(abs(c[1]) for s in outer for c in s.corners()) > PERIMETER_HALF - 0.06
    assert AXIS[0] > 0.1
    h, w = SCORE["score_L4"]
    S = scale_of(FRAME)
    tip = S @ coral_held(h, w) + rot_y(coral_pitch(w)) @ [CORAL_LEN / 2, 0, 0]
    assert (S @ coral_held(h, w))[0] > (S @ coral_held(0.0, 0.0))[0] + 0.2
    assert tip[0] > PERIMETER_HALF + BUMPER_T


def test_elevator_stages_stay_nested():
    """Cascade along the tilted axis: every stage overlaps the one it rides in, the carriage's lower bearings stay
    inside the innermost stage."""
    for h in np.linspace(0, H_MAX, 7):
        spans = [(z0 / AXIS[2] + k * h, z0 / AXIS[2] + k * h + length) for _, _, length, z0, k in STAGES]
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert b0 < a1 - 0.15                                               # >= 15 cm overlap
        assert 0.166 / AXIS[2] + h < spans[-1][1] - 0.1


@pytest.mark.parametrize("goal", list(CORAL_AT))
def test_score_poses_put_the_coral_at_each_level(goal):
    h, w = SCORE[goal]
    assert 0.0 <= h <= H_MAX
    c = coral_held(h, w)
    assert abs(c[2] - CORAL_AT[goal][0]) < 1e-9 and abs(coral_pitch(w) - CORAL_AT[goal][1]) < 1e-9


def test_algae_poses():
    for z in REEF_ALGAE_Z.values():
        h = algae_target(z)
        assert 0.0 <= h <= H_MAX and abs(algae_held(h, WRIST["reach"])[2] - z) < 1e-9
    b = algae_held(0.0, WRIST["floor"])
    assert abs(b[2] - ALGAE_R) < 0.01 and b[0] > PERIMETER_HALF                   # at the floor, in front
    carry = algae_held(0.0, 0.0)
    assert carry[2] - ALGAE_R > 0.14 and carry[0] > PERIMETER_HALF                # held clear of the bumper
    assert algae_held(H_MAX, WRIST["net"])[2] > 2.1                              # over the net


def test_hopper_feeds_the_effector_from_the_back():
    (top, a0), (bot, a1) = funnel_pose(0.0), funnel_pose(1.0)
    assert top[0] < -PERIMETER_HALF + 0.05 and top[2] < 37.5 * IN                # under the station chute exit
    assert np.allclose(bot, coral_held(0.0, 0.0)) and abs(a1 - coral_pitch(0.0)) < 1e-9
    zs = [funnel_pose(s)[0][2] for s in np.linspace(0, 1, 11)]
    assert all(b <= a + 1e-9 for a, b in zip(zs, zs[1:]))                       # it only goes down


def test_bumpers_follow_alliance_and_frame_size():
    for alliance, col in ((0, BLUE), (1, RED)):
        b = [s for s in pose(alliance=alliance) if s.name == "bumper"]
        assert len(b) == 4 and all(s.color == col for s in b)
    small = 25 * IN
    b = [s for s in pose(frame=small) if s.name == "bumper"]
    assert abs(max(s.max_x() for s in b) - (small / 2 + BUMPER_T)) < 1e-9
    assert max(s.max_x() for s in robot(pose(frame=small)) if s.name != "bumper") < small / 2 + BUMPER_T


def test_held_pieces():
    assert sum(isinstance(s, Sphere) for s in pose(hold="algae")) == 1
    coral = [s for s in pose(hold="both", funnel=0.5) if s.name == "coral"]
    assert len(coral) == 2 and all(isinstance(s, Cyl) and abs(s.radius - CORAL_R) < 1e-9 for s in coral)
    c = coral[0]                                                                  # axis along the effector channel
    axis = c.R @ [0, 0, 1.0]
    assert abs(np.degrees(np.arctan2(axis[2], axis[0])) - coral_pitch(0.0)) < 1e-6


def test_bodies_match_the_box_model_kinematics():
    """robot.glb bodies (CAD pose = joint zero) and the pieces agree: a point of the effector moves exactly like the
    held coral, the pivot stays on the carriage, the stages follow the cascade."""
    I = body_transforms()
    assert all(np.allclose(A, np.eye(3)) and np.allclose(t, 0) for A, t in I.values()) and set(I) == set(BODIES)
    for h, w, c in [(0.7, -35.0, 40.0), (H_MAX, WRIST["net"], CLIMB["hang"])]:
        T = body_transforms(h, w, c)
        A, t = T["effector"]
        assert np.allclose(A @ coral_held(0, 0) + t, coral_held(h, w))
        assert np.allclose(A @ EFF_PIVOT + t, EFF_PIVOT + AXIS * h)
        assert np.allclose(T["stage3"][1], AXIS * 2 * h / 3) and np.allclose(T["carriage"][1], AXIS * h)
    solids = pose(h=0.5, wrist=-20, climb=10, hold="both", cad=True)
    bodies = [s for s in solids if isinstance(s, Body)]
    assert sorted(b.name for b in bodies) == sorted(BODIES)
    rows = solid_rows(solids)
    for s, row in zip(solids, rows):
        if isinstance(s, Body):
            assert row[1] == 3 + BODIES.index(s.name)


def test_slots_layout():
    solids = pose(h=0.5, wrist=-20, hold="both", funnel=0.3, frame=CAD_FRAME)
    assert len(solids) <= SLOTS
    s = slots_for(solids)
    assert s[:, 0].sum() == len(solids)
    kinds = s[: len(solids), 1]
    assert (kinds == 1).sum() == 1 and (kinds == 2).sum() == sum(isinstance(x, Cyl) for x in solids)
    for row in s[: len(solids)]:
        R = row[8:17].reshape(3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    # a wheel cylinder: axis along robot y = Godot -z, so its Godot local Y (the CylinderMesh axis) is along z
    i = next(k for k, x in enumerate(solids) if x.name == "wheel")
    assert abs(abs(s[i, 8:17].reshape(3, 3)[2, 1]) - 1.0) < 1e-6
    assert FRAME == 29.5 * IN and isinstance(solids[0], Box)
    assert len(pose(h=H_MAX, wrist=-50, climb=70, hold="both", funnel=0.5)) <= SLOTS      # the most crowded case
