"""tools/prepare_robot.py on a tiny synthetic Onshape-style glTF: parts land in the right bodies (sub-assemblies, and
the elevator stage of each part from the stage tubes), fasteners go, the output has one node per body and the CAD axes
become the robot frame."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("fast_simplification")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

# (sub-assembly, part, centre in CAD axes (x right, y front, z up), size, expected body)
PARTS = [
    ("R-0100 Drivetrain <1>", "Front Tube", (0.0, 0.33, 0.05), (0.6, 0.05, 0.05), "base"),
    ("R-0600 Hopper <1>", "Side Fridge", (0.33, -0.1, 0.45), (0.004, 0.5, 0.3), "base"),
    ("R-0200 Elevator <1>", "R-0201-Outer Elevator LEFT Tube", (0.29, 0.14, 0.43), (0.025, 0.05, 0.72), "base"),
    ("R-0200 Elevator <1>", "R-0204-Middle Elevator LEFT Tube", (0.26, 0.14, 0.44), (0.025, 0.05, 0.75), "stage2"),
    ("R-0200 Elevator <1>", "R-0207-Inner Elevator LEFT Tube", (0.23, 0.15, 0.51), (0.025, 0.05, 0.85), "stage3"),
    ("R-0200 Elevator <1>", "Bearing Block", (0.276, 0.1, 0.07), (0.03, 0.05, 0.03), "stage2"),      # rides on the outer
    ("R-0200 Elevator <1>", "Bearing Block", (0.276, 0.17, 0.78), (0.03, 0.05, 0.03), "base"),       # holds stage 2
    ("R-0200 Elevator <1>", "Bearing Block", (0.244, 0.1, 0.09), (0.03, 0.05, 0.03), "stage3"),
    ("R-0800 Carriage <1>", "Side Plate", (0.21, 0.12, 0.31), (0.005, 0.14, 0.32), "carriage"),
    ("R-0900 End Effector <1>", "Roller", (-0.04, 0.30, 0.30), (0.18, 0.05, 0.05), "effector"),
    ("R-0700 Climber Arm <1>", "Grapple Arm", (0.0, -0.37, 0.56), (0.025, 0.15, 0.014), "climber"),
    ("R-0700 Climber Arm <1>", "Socket head cap screw #10-32 x 0.5", (0.0, -0.41, 0.55), (0.03, 0.03, 0.03), None),
]


def _gltf(path):
    """Onshape layout: root assembly -> sub-assembly -> 'occurrence of <part>' (transform) -> '<part>' (mesh)."""
    blob, acc, views, meshes, nodes = b"", [], [], [], [{"name": "R-0000", "children": []}]
    subs = {}
    for sub, part, c, size, _ in PARTS:
        m = trimesh.creation.box(size)
        v, f = m.vertices.astype(np.float32), m.faces.astype(np.uint32)
        for arr, kind in ((v, "VEC3"), (f.ravel(), "SCALAR")):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes})
            a = {"bufferView": len(views) - 1, "componentType": 5126 if kind == "VEC3" else 5125,
                 "count": len(arr), "type": kind}
            if kind == "VEC3":
                a.update(min=v.min(0).tolist(), max=v.max(0).tolist())
            acc.append(a)
            blob += arr.tobytes()
        meshes.append({"primitives": [{"attributes": {"POSITION": len(acc) - 2}, "indices": len(acc) - 1, "material": 0}]})
        if sub not in subs:
            subs[sub] = len(nodes)
            nodes.append({"name": sub, "children": []})
            nodes[0]["children"].append(subs[sub])
        nodes.append({"name": f"occurrence of {part}", "translation": list(c), "children": [len(nodes) + 1]})
        nodes[subs[sub]]["children"].append(len(nodes) - 1)
        nodes.append({"name": part, "mesh": len(meshes) - 1})
    g = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes,
         "accessors": acc, "bufferViews": views,
         "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.24, 0.73, 1.0]}}],
         "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}]}
    path.write_text(json.dumps(g))
    return path


def test_parts_go_to_their_bodies(tmp_path):
    import prepare_robot as pr
    meshes, inst = pr.load(_gltf(tmp_path / "r.gltf"))
    bodies, tx = pr.assign_bodies(meshes, inst)
    got = {(ch[1], ch[-1], round(float(W[2, 3]), 3)): b for (m, W, ch), b in zip(inst, bodies)}
    for sub, part, c, _, want in PARTS:
        if want is not None:
            assert got[(sub, part, c[2])] == want, (part, c)
    assert set(tx) == {"outer", "middle", "inner"}


def test_export_has_one_node_per_body_in_the_robot_frame(tmp_path):
    import prepare_robot as pr
    from ghostsim.mechanism import BODIES
    out = tmp_path / "robot.glb"
    tris, _ = pr.build(_gltf(tmp_path / "r.gltf"), out, 10_000)
    assert sum(tris.values()) == 12 * (len(PARTS) - 1)                       # boxes kept whole, the screw dropped
    sc = trimesh.load(str(out), force="scene")
    assert set(BODIES) <= set(sc.graph.nodes)
    eff = [g for n, g in ((n, sc.graph[n][1]) for n in sc.graph.nodes_geometry) if n.startswith("effector")]
    v = np.concatenate([sc.geometry[g].vertices for g in eff])
    # CAD (x, y, z) = (-0.04, 0.30, 0.30) -> robot (0.30, 0.04, 0.30) -> glTF Y-up (x, z, -y) = (0.30, 0.30, -0.04)
    assert np.allclose(v.mean(0), [0.30, 0.30, -0.04], atol=1e-5)
