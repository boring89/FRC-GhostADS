"""Broadcast helpers: canvas sizes, robot camera scaling, tag geometry, result card, category table."""
import numpy as np

from ghosttrain.broadcast import CANVAS, TILE, TILE_RENDER, TOP, result_card, robot_rig, tag_corners, tile
from ghostsim import load


def test_layout_sizes():
    assert CANVAS == (1280, 720)
    assert TOP[0] == CANVAS[0] and TOP[1] + TILE[1] == CANVAS[1]
    assert TILE[0] * 2 == CANVAS[0]
    assert abs(TILE_RENDER[0] / TILE_RENDER[1] - 4 / 3) < 0.01            # robot POV keeps 4:3


def test_robot_rig_intrinsics_scale_with_render_size():
    full = robot_rig((640, 480))
    small = robot_rig(TILE_RENDER)
    assert full["fx"] == 500.0 and full["capture"] == [640, 480]
    assert abs(small["fx"] - 500.0 * TILE_RENDER[0] / 640) < 1e-9
    assert abs(small["cy"] - 240.0 * TILE_RENDER[1] / 480) < 1e-9


def test_tag_corners_are_square_and_on_the_field():
    game = load("2025")
    corners, ids = tag_corners(game)
    assert corners.shape == (len(ids), 4, 3) and len(ids) > 0
    side = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=-1)
    assert np.allclose(side, side[0], rtol=1e-3)
    assert np.allclose(np.linalg.norm(corners[:, 1] - corners[:, 2], axis=-1), side[0], rtol=1e-3)


def test_result_card_and_tile_shapes():
    card = result_card(np.array([293.0, 276.0]), np.array([[9, 208, 40, 36, 0], [9, 209, 32, 26, 0]], float), [2, 1], "x")
    assert card.shape == (720, 1280, 3) and card.dtype == np.uint8
    t = tile(np.zeros((TILE_RENDER[1], TILE_RENDER[0], 3), np.uint8), "BLUE  robot 1", (200, 90, 25), ["a", "b"])
    assert t.shape == (TILE[1], TILE[0], 3)
