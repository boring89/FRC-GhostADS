"""Official FE-2025 CAD (Onshape GLB) -> optimised visual model for the renderer.

    uv run python tools/prepare_field.py [source.glb] [--budget 300000] [--budget-hi 1500000]

What it does
  - removes what the renderer draws itself: AprilTags (FE-00083, placed from the tag file), the six CAGE
    assemblies (animated), any leftover CORAL/ALGAE, plus the scoring table and small hardware (< 4 cm)
  - decimates every unique part with one ratio, raised to a floor for thin long parts (pipes keep their shape), so
    that the drawn total meets the budget; crease-angle normals; merges the instances into one mesh per material
  - re-bases the model onto the field frame used everywhere else and writes Y-up glTF (Godot convention); keeps the
    source's triangle winding (the axis change is a rotation) and its double-sided materials
Output: game/2025/visual/field.glb (robot cameras, --budget) and field_hi.glb (live view, --budget-hi). The source and the output are NOT committed (size);
the renderer falls back to the procedural proxy scene when field.glb is absent.

Source frame (verified against the tag file, <= 5 mm): metres, Z up, origin at the field centre,
+X toward red. Field frame: origin at the blue alliance-wall / scoring-table corner.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DROP_NAME = re.compile(r"APRILTAG|FE-00083|CAGE \(GE-25231\)|CORAL \(GE-25500\)|ALGAE|Scoring Table|"
                       r"Carabiner|eyenut|eyebolt|^hex_|^washer_", re.I)
MIN_EXTENT_M = 0.04       # parts smaller than this are invisible at 160x120


def find_source(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    hits = sorted(ROOT.glob("REEFSCAPE FIELD*.glb"))
    if not hits:
        sys.exit("no source GLB given and none found in the repo root")
    return hits[0]


def _rgba(mat) -> np.ndarray:
    """Onshape materials are named 'r_g_b_a_x' with float components; fall back to baseColorFactor."""
    nm = getattr(mat, "name", "") or ""
    parts = nm.split("_")
    try:
        v = [float(x) for x in parts[:3]]
        if all(0.0 <= x <= 1.0 for x in v):
            return np.array(v + [1.0])
    except ValueError:
        pass
    bc = getattr(mat, "baseColorFactor", None)
    if bc is None:
        return np.array([0.78, 0.78, 0.78, 1.0])
    bc = np.asarray(bc, float)
    return bc / 255.0 if bc.max() > 1.0 else bc


def clean(name: str) -> str:
    return name.replace("occurrence of ", "")


def kept_instances(scene):
    """Yield (node, geometry_name, transform) for every instance that survives the filters."""
    import trimesh  # noqa: F401
    parents = {v: u for u, v, _ in scene.graph.to_edgelist()}
    drop_cache: dict[str, bool] = {}
    carpet_cache: dict[str, bool] = {}

    def memo(node: str, cache: dict, test) -> bool:
        stack, n = [], node
        while n is not None and n not in cache:
            stack.append(n)
            n = parents.get(n)
        val = cache[n] if n is not None else False
        for c in reversed(stack):                       # root -> leaf: inherit from the parent
            val = val or test(clean(c))
            cache[c] = val
        return val

    dropped = lambda node: memo(node, drop_cache, lambda nm: bool(DROP_NAME.search(nm)))
    under_carpet = lambda node: memo(node, carpet_cache, lambda nm: "CARPET" in nm)

    for node in scene.graph.nodes_geometry:
        if dropped(node):
            continue
        T, g = scene.graph[node]
        yield node, g, T, under_carpet(node)


ELONGATED = (0.15, 3.0)   # parts longer than 0.15 m and 3x their next extent (pipes, rods, rails) ...
MIN_TRIS = {"hi": (64, 12), "lo": (24, 4)}   # ... keep at least this many triangles (thin parts collapsed to slivers),
                                              # other parts at least this many; per level of detail
CREASE_DEG = 35.0         # smooth normals across edges flatter than this (cylinders look round, boxes stay sharp)


def _plan(scene, inst, budget, floors):
    """Per unique geometry: target triangle count. One ratio r for all parts, raised to each part's floor, chosen so
    the drawn total (instances included) meets the budget."""
    use = collections.Counter(g for _, g, _, _ in inst)
    n, floor = {}, {}
    for g in use:
        m = scene.geometry[g]
        e = np.sort(np.ptp(m.bounds, axis=0))[::-1]
        if e[0] < MIN_EXTENT_M:
            continue
        n[g] = len(m.faces)
        lo = floors[0] if (e[0] > ELONGATED[0] and e[0] > ELONGATED[1] * max(e[1], 1e-6)) else floors[1]
        floor[g] = min(n[g], lo)
    drawn = lambda r: sum(use[g] * max(floor[g], min(n[g], r * n[g])) for g in n)
    lo_r, hi_r = 0.0, 1.0
    for _ in range(40):
        mid = (lo_r + hi_r) / 2
        lo_r, hi_r = (mid, hi_r) if drawn(mid) < budget else (lo_r, mid)
    r = lo_r
    return {g: int(max(floor[g], min(n[g], r * n[g]))) for g in n}, r, drawn(r), sum(use[g] * n[g] for g in n)


def _part(m, target):
    """Decimate one unique part and give it crease-angle normals. Returns (v, f, normals) in the part frame."""
    import fast_simplification as fs
    import trimesh
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)
    if target < len(f):
        v, f = fs.simplify(v, f, target_count=target, agg=5)
    tm = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f, np.int64), process=True)
    if len(tm.faces) == 0:
        return None
    tm = trimesh.graph.smooth_shade(tm, angle=np.radians(CREASE_DEG), facet_minarea=None)
    return np.asarray(tm.vertices), np.asarray(tm.faces), np.asarray(tm.vertex_normals)


def build(src: Path, dst: Path, budget: int, field_len: float, field_wid: float, scene=None, floors=MIN_TRIS["hi"]):
    import trimesh

    t0 = time.time()
    scene = scene or trimesh.load(str(src), force="scene")
    inst = list(kept_instances(scene))
    targets, r, drawn, total_in = _plan(scene, inst, budget, floors)
    print(f"{len(scene.graph.nodes_geometry)} instances; kept {len(inst)} ({total_in:,} tris drawn) -> ratio {r:.3f}, "
          f"{drawn:,.0f} tris for budget {budget:,}  [{time.time() - t0:.0f}s]")
    cache: dict[str, tuple | None] = {}
    groups: dict[str, dict] = {}
    shift = np.array([field_len / 2, field_wid / 2, 0.0])
    t1 = time.time()
    for node, g, T, is_carpet in inst:
        if g not in targets:
            continue
        if g not in cache:
            m = scene.geometry[g]
            part = _part(m, targets[g])
            cache[g] = None if part is None else (*part, _rgba(getattr(m.visual, "material", None)))
        if cache[g] is None:
            continue
        v, f, nv, rgba = cache[g]
        A = T[:3, :3]
        wv = v @ A.T + T[:3, 3] + shift
        nw = nv @ np.linalg.inv(A)                                          # normals: inverse-transpose
        nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12)
        ff = f[:, [0, 2, 1]] if np.linalg.det(A) < 0 else f                 # mirrored instance: keep the outside out
        key = "carpet" if is_carpet else "m_" + "".join(f"{int(round(c * 255)):02x}" for c in rgba[:3])
        d = groups.setdefault(key, {"v": [], "f": [], "n": [], "k": 0, "rgba": rgba})
        d["v"].append(wv)
        d["n"].append(nw)
        d["f"].append(ff + d["k"])
        d["k"] += len(wv)
    print(f"decimated {len(cache)} unique parts, merged into {len(groups)} material groups [{time.time() - t1:.0f}s]")
    out = trimesh.Scene()
    tris = 0
    for key, d in sorted(groups.items()):
        v, f, nn = np.concatenate(d["v"]), np.concatenate(d["f"]), np.concatenate(d["n"])
        # field (x, y, z) -> glTF Y-up (x, z, -y): a rotation (det +1), so the triangle winding stays as it is
        gv = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
        gn = np.stack([nn[:, 0], nn[:, 2], -nn[:, 1]], 1)
        mesh = trimesh.Trimesh(gv, f, vertex_normals=gn, process=False)
        rough, metal = (0.95, 0.0) if key == "carpet" else (0.6, 0.35)
        mat = trimesh.visual.material.PBRMaterial(name=key, baseColorFactor=(d["rgba"] * 255).astype(np.uint8).tolist(),
                                                  roughnessFactor=rough, metallicFactor=metal, doubleSided=True)
        mesh.visual = trimesh.visual.TextureVisuals(material=mat)
        out.add_geometry(mesh, node_name=key, geom_name=key)
        tris += len(f)
        print(f"  {key:12s} {len(f):9,d} tris   rgba {np.round(d['rgba'], 2)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(dst), include_normals=True)
    print(f"wrote {dst} : {tris:,} tris in {len(groups)} meshes, {dst.stat().st_size / 1e6:.1f} MB  [{time.time() - t0:.0f}s]")
    return tris


def main() -> int:
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--budget", type=int, default=300_000, help="field.glb: robot cameras (many small renders)")
    ap.add_argument("--budget-hi", type=int, default=1_500_000, help="field_hi.glb: live view; 0 = skip")
    ap.add_argument("--out", default=str(ROOT / "game" / "2025" / "visual" / "field.glb"))
    a = ap.parse_args()
    from ghostsim.geometry import IN
    src = find_source(a.source)
    scene = trimesh.load(str(src), force="scene")
    out = Path(a.out)
    build(src, out, a.budget, 690.875 * IN, 317.0 * IN, scene, MIN_TRIS["lo"])
    if a.budget_hi:
        build(src, out.with_name(out.stem + "_hi.glb"), a.budget_hi, 690.875 * IN, 317.0 * IN, scene, MIN_TRIS["hi"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
