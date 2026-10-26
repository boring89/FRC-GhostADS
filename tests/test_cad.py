import sys
from pathlib import Path

import numpy as np
import pytest

from ghostsim import geometry as geo
from ghostsim import load

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted(ROOT.glob("REEFSCAPE FIELD*.glb"))


def test_polygon_center_ignores_padding():
    p = geo.Polygons([geo.regular_polygon([3, 4], 1.0, 6, 0.0), geo.rect(10, 20, 2, 2)])
    assert np.allclose(p.center, [[3, 4], [10, 20]])
    assert (p.radius >= np.linalg.norm(geo.rect(10, 20, 2, 2) - [10, 20], axis=1).max() - 1e-9)[1]


def test_scene_with_cad_keeps_only_tags():
    g = load("2025")
    full, cad = g.scene(), g.scene(cad=True)
    assert len(cad["primitives"]) == 22 and all(p["kind"] == "tag" for p in cad["primitives"])
    assert len(full["primitives"]) > 22


def test_guardrail_inset_in_boundary():
    g = load("2025")
    assert np.isclose(g.hp_off[2], 0.027) and np.isclose(g.hp_off[3], -(g.width - 0.027))


@pytest.mark.skipif(not SRC, reason="source CAD GLB not present in the repo root")
def test_cad_matches_collision_geometry(capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    import check_field
    old = sys.argv
    sys.argv = ["check_field.py", str(SRC[0])]
    try:
        rc = check_field.main()
    finally:
        sys.argv = old
    assert rc == 0, capsys.readouterr().out
