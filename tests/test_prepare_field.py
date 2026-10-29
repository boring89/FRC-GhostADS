"""tools/prepare_field.py on a tiny synthetic CAD scene: faces stay outward (the field was exported inside out:
its winding was flipped after a pure rotation and the materials were single-sided), thin parts keep their shape."""
import json
import struct

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("fast_simplification")


def _glb_tris(path):
    """(triangles (k,3,3), stored vertex normals (k,3,3), materials) straight from the .glb."""
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(b[20:20 + jl])
    bin0 = 20 + jl + 8

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
        n = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        x = np.frombuffer(b, dt, count=a["count"] * n, offset=bin0 + bv.get("byteOffset", 0) + a.get("byteOffset", 0))
        return x.reshape(-1, n) if n > 1 else x

    tris, nrms = [], []
    for mesh in js["meshes"]:
        for p in mesh["primitives"]:
            v, nv = acc(p["attributes"]["POSITION"]), acc(p["attributes"]["NORMAL"])
            f = acc(p["indices"]).reshape(-1, 3)
            tris.append(v[f]); nrms.append(nv[f])
    return np.concatenate(tris), np.concatenate(nrms), js["materials"]


def test_field_export_faces_outward_and_double_sided(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import prepare_field as pf
    monkeypatch.setattr(pf, "DROP_NAME", __import__("re").compile("^$"))
    box = trimesh.creation.box((0.5, 0.3, 0.2))
    pipe = trimesh.creation.cylinder(radius=0.02, height=1.0, sections=32)           # thin, long
    sc = trimesh.Scene()
    sc.add_geometry(box, node_name="box", geom_name="box", transform=trimesh.transformations.rotation_matrix(0.7, [0, 0, 1]))
    for k in range(20):                                                               # many instances: tight budget
        T = trimesh.transformations.translation_matrix([1.0 + 0.1 * k, 0.0, 0.5])
        sc.add_geometry(pipe, node_name=f"pipe{k}", geom_name="pipe", transform=T)
    out = tmp_path / "f.glb"
    pf.build(Path("synthetic"), out, 400, 2.0, 2.0, sc, pf.MIN_TRIS["lo"])
    tris, nrms, mats = _glb_tris(out)
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert ((fn * nrms.sum(1)).sum(1) > 0).mean() > 0.99                               # winding matches normals
    # every triangle faces away from the centre of its own part (field + shift (1, 1, 0) -> glTF Y-up (x, z, -y))
    to_gl = lambda p: np.array([p[0] + 1.0, p[2], -(p[1] + 1.0)])
    centres = np.array([to_gl([0.0, 0.0, 0.0])] + [to_gl([1.0 + 0.1 * k, 0.0, 0.5]) for k in range(20)])
    mid = tris.mean(1)
    own = centres[np.argmin(np.linalg.norm(mid[:, None] - centres[None], axis=-1), 1)]
    assert (((mid - own) * fn).sum(1) > 0).mean() > 0.99                              # outward, not inside out
    assert all(m.get("doubleSided") for m in mats)
    # the pipes kept at least the thin-part floor (they collapsed to slivers before)
    assert len(tris) >= 20 * pf.MIN_TRIS["lo"][0]
