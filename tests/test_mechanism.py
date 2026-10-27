"""Geometry checks of the cube-built mechanism model (ghostsim/mechanism.py)."""
import numpy as np
import pytest

from ghostsim.mechanism import (ALGAE_R, EE_LEN, FRAME, H_MAX, INTAKE_LEN, PERIMETER_HALF, WRIST_MIN_DEG, Box, Sphere,
                                ee_wrist, intake_tip, pose)

BRANCH_TOP = {"trough": 18.0 * 0.0254, "L2": 31.875 * 0.0254, "L3": 47.625 * 0.0254, "L4": 72.0 * 0.0254}


def tip(h, theta):
    P, _ = ee_wrist(h)
    t = np.radians(theta)
    return P + EE_LEN * np.array([np.cos(t), 0.0, np.sin(t)])


def roller(psi):
    return [s for s in pose(psi=psi) if isinstance(s, Box) and s.name == "intake_roller"][0]


def solids_min_z(solids):
    return min(s.min_z() for s in solids)


@pytest.mark.parametrize("h", np.linspace(0, H_MAX, 5))
@pytest.mark.parametrize("theta", np.linspace(WRIST_MIN_DEG, 90, 6))
@pytest.mark.parametrize("hold", [None, "coral", "algae"])
def test_operating_range_keeps_everything_above_the_floor(h, theta, hold):
    for psi in (-10, 0, 40, 75):
        assert solids_min_z(pose(h=h, theta=theta, psi=psi, hold=hold)) > -1e-6


def test_wrist_below_limit_drives_a_held_algae_into_the_floor():
    # documents why WRIST_MIN_DEG exists: below it the held algae would pass through the floor at h=0
    assert solids_min_z(pose(h=0, theta=WRIST_MIN_DEG - 20, psi=-10, hold="algae")) < 0


def test_intake_roller_touches_floor_when_deployed():
    r = roller(-10)
    assert 0.0 <= r.min_z() < 0.01                       # on the floor, not through it


def test_intake_extension_within_r105():
    # R105: no more than 1 ft 6 in beyond the ROBOT PERIMETER (frame edge, bumpers excluded)
    assert roller(-10).min_x() >= -(PERIMETER_HALF + 0.457)


def test_starting_configuration_height_r104():
    # R104: starting configuration no taller than 3 ft 6 in (1.067 m)
    top = max(max(s.corners()[:, 2].max() if isinstance(s, Box) else s.center[2] + s.radius for s in pose(h=0, theta=80, psi=75)), 0)
    assert top <= 1.067


def test_frame_inside_25_27_in():
    frame = [s for s in pose() if isinstance(s, Box) and s.name == "chassis"][0]
    assert 25 * 0.0254 <= 2 * frame.half[0] <= 27 * 0.0254 + 1e-9


@pytest.mark.parametrize("level", ["trough", "L2", "L3", "L4"])
def test_end_effector_reaches_every_scoring_height(level):
    # some elevator height places the wrist tip exactly at the branch top with theta = 0 (horizontal)
    target = BRANCH_TOP[level]
    h = target - 0.41                                    # wrist tip height at theta=0 is 0.41 + h
    assert 0.0 <= h <= H_MAX
    assert abs(tip(h, 0.0)[2] - target) < 1e-9


def test_processor_pose_holds_algae_above_floor():
    # processor opening sits at 7 in (0.18 m): at the processor pose the held algae centre is about 0.21 m up
    s = [x for x in pose(h=0.0, theta=-20.0, hold="algae") if isinstance(x, Sphere)][0]
    assert s.center[2] >= ALGAE_R and s.center[2] < 0.25


def test_held_algae_is_a_sphere_on_the_end_effector():
    spheres = [s for s in pose(h=0.5, theta=10, hold="algae") if isinstance(s, Sphere)]
    assert len(spheres) == 1 and abs(spheres[0].radius - ALGAE_R) < 1e-9


def test_intake_arm_length_and_stow():
    P = np.array([-0.30, 0.0, 0.13])
    assert abs(np.linalg.norm(intake_tip(np.radians(75)) - P) - INTAKE_LEN) < 1e-9
    assert intake_tip(np.radians(75))[2] > intake_tip(np.radians(-10))[2]
