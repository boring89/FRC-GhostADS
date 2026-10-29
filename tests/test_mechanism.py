"""Geometry checks of the mechanism model (ghostsim/mechanism.py, 6328-style layout)."""
import numpy as np
import pytest

from ghostsim.mechanism import (ALGAE_R, ARM, BLUE, BUMPER_T, CORAL_R, FRAME, H_MAX, PERIMETER_HALF, RED, SLOTS, STAGES,
                                STOW, Box, Cyl, Sphere, arm_tip, ball_in_arm, coral_held, funnel_point, pose, slots_for)
from ghostsim.mechanism_state import CORAL_AT, SCORE

IN = 0.0254
PIECES = ("coral", "algae")


def robot(solids):
    return [s for s in solids if s.name not in PIECES]


@pytest.mark.parametrize("h", np.linspace(0, H_MAX, 4))
@pytest.mark.parametrize("wrist", [-40.0, -20.0, 0.0])
@pytest.mark.parametrize("hold", [None, "coral", "algae", "both"])
def test_operating_range_keeps_everything_above_the_floor(h, wrist, hold):
    for arm in (ARM["floor"], ARM["processor"], ARM["hold"], ARM["catch"], ARM["stow"], ARM["net"]):
        for climb in (15.0, 60.0, 95.0):
            assert min(s.min_z() for s in pose(h=h, wrist=wrist, arm=arm, climb=climb, hold=hold)) > -1e-6


def test_starting_configuration_height_r104():
    # R104: starting configuration no taller than 3 ft 6 in (1.067 m)
    assert max(s.max_z() for s in pose(**STOW)) <= 1.067


def test_algae_arm_extension_within_r105():
    # R105: no more than 1 ft 6 in beyond the frame perimeter (bumpers excluded); game pieces are not robot parts
    for arm in ARM.values():
        assert max(s.max_x() for s in robot(pose(arm=arm))) <= PERIMETER_HALF + 18 * IN + 1e-9


def test_algae_arm_reaches_the_floor_and_cradles_a_ball():
    roller = [s for s in pose(arm=ARM["floor"]) if s.name == "arm_roller"][0]
    assert 0.0 <= roller.min_z() < 0.03
    assert arm_tip(ARM["floor"])[0] > PERIMETER_HALF + BUMPER_T                  # in front of the bumper
    b = ball_in_arm(ARM["hold"])
    assert b[2] > ALGAE_R and b[0] > PERIMETER_HALF                             # held in front, off the floor


def test_elevator_stages_stay_nested():
    """Cascade: every stage overlaps the one it rides in, and the carriage stays inside the innermost stage."""
    for h in np.linspace(0, H_MAX, 7):
        spans = [(z0 + k * h, z1 + k * h) for _, _, _, _, z0, z1, k in STAGES]
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert b0 < a1 - 0.25                                               # >= 25 cm overlap
        car = [s for s in pose(h=h) if s.name == "carriage"]
        assert max(s.max_z() for s in car) <= spans[-1][1] + 1e-9


@pytest.mark.parametrize("goal", list(CORAL_AT))
def test_score_poses_put_the_coral_at_each_level(goal):
    h, w = SCORE[goal]
    assert 0.0 <= h <= H_MAX
    c = coral_held(h, w)
    assert abs(c[2] - CORAL_AT[goal][0]) < 1e-9
    assert c[0] > 0.2                                                          # in front of the elevator


def test_funnel_feeds_the_dispenser_from_the_back():
    top, bot = funnel_point(0.0), funnel_point(1.0)
    assert top[0] < -PERIMETER_HALF + 0.05 and top[2] < 37.5 * IN + 0.06        # just under the station chute exit
    assert bot[0] > top[0] and bot[2] < top[2]
    assert abs(bot[2] - coral_held(0.0, 0.0)[2]) < 0.06                         # level with the dispenser at h = 0


def test_bumpers_follow_alliance_and_frame_size():
    for alliance, col in ((0, BLUE), (1, RED)):
        b = [s for s in pose(alliance=alliance) if s.name == "bumper"]
        assert len(b) == 4 and all(s.color == col for s in b)
    small = 25 * IN
    b = [s for s in pose(frame=small) if s.name == "bumper"]
    assert abs(max(s.max_x() for s in b) - (small / 2 + BUMPER_T)) < 1e-9


def test_held_pieces():
    assert sum(isinstance(s, Sphere) for s in pose(hold="algae")) == 1
    coral = [s for s in pose(hold="both", funnel=0.5) if s.name == "coral"]
    assert len(coral) == 2 and all(isinstance(s, Cyl) and abs(s.radius - CORAL_R) < 1e-9 for s in coral)


def test_slots_layout():
    solids = pose(h=0.5, wrist=-20, hold="both", funnel=0.3)
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
