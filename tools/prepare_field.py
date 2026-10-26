"""Official FE-2025 CAD (Onshape GLB) -> optimised visual model for the renderer.

    uv run python tools/prepare_field.py [source.glb] [--budget 300000]

What it does
  - removes what the renderer draws itself: AprilTags (FE-00083, placed from the tag file), the six CAGE
    assemblies (animated), any leftover CORAL/ALGAE, plus the scoring table and small hardware (< 4 cm)
  - merges the ~62 000 part instances into one mesh per material, decimating to the triangle budget
  - re-bases the model onto the field frame used everywhere else and writes Y-up glTF (Godot convention)
Output: game/2025/visual/field.glb (+ report). The source and the output are NOT committed (size);
the renderer falls back to the procedural proxy scene when field.glb is absent.

Source frame (verified against the tag file, <= 5 mm): metres, Z up, origin at the field centre,
+X toward red. Field frame: origin at the blue alliance-wall / scoring-table corner.
"""
from __future__ import annotations

import argparse
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


def build(src: Path, dst: Path, budget: int, field_len: float, field_wid: float):
    import fast_simplification as fs
    import trimesh

    t0 = time.time()
    scene = trimesh.load(str(src), force="scene")
    inst = list(kept_instances(scene))
    total_in = sum(len(scene.geometry[g].faces) for _, g, _, _ in inst)
    ratio = min(1.0, budget / max(total_in, 1))
    print(f"loaded {len(scene.graph.nodes_geometry)} instances; kept {len(inst)} "
          f"({total_in:,} tris) -> ratio {ratio:.3f} for budget {budget:,}  [{time.time() - t0:.0f}s]")
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    groups: dict[str, dict] = {}
    shift = np.array([field_len / 2, field_wid / 2, 0.0])
    t1 = time.time()
    for node, g, T, is_carpet in inst:
        if g not in cache:
            m = scene.geometry[g]
            if np.ptp(m.bounds, axis=0).max() < MIN_EXTENT_M:
                cache[g] = None
            else:
                v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)
                tgt = max(min(len(f), 6), int(len(f) * ratio))
                if tgt < len(f):
                    v, f = fs.simplify(v, f, target_count=tgt, agg=7)
                cache[g] = (np.asarray(v, np.float64), np.asarray(f, np.int64), _rgba(getattr(m.visual, "material", None)))
        if cache[g] is None:
            continue
        v, f, rgba = cache[g]
        key = "carpet" if is_carpet else "m_" + "".join(f"{int(round(c * 255)):02x}" for c in rgba[:3])
        wv = v @ T[:3, :3].T + T[:3, 3] + shift
        d = groups.setdefault(key, {"v": [], "f": [], "n": 0, "rgba": rgba})
        d["v"].append(wv)
        d["f"].append(f + d["n"])
        d["n"] += len(wv)
    print(f"merged {len(groups)} material groups [{time.time() - t1:.0f}s]")
    out = trimesh.Scene()
    tris = 0
    for key, d in sorted(groups.items()):
        v, f = np.concatenate(d["v"]), np.concatenate(d["f"])
        gv = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)                       # field (x,y,z) -> Y-up (x,z,-y)
        f = f[:, [0, 2, 1]]                                                   # keep CCW after the axis swap
        mesh = trimesh.Trimesh(gv, f, process=False)
        mesh.unmerge_vertices()
        _ = mesh.vertex_normals
        rough, metal = (0.95, 0.0) if key == "carpet" else (0.6, 0.35)
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=key, baseColorFactor=(d["rgba"] * 255).astype(np.uint8).tolist(), roughnessFactor=rough, metallicFactor=metal))
        out.add_geometry(mesh, node_name=key, geom_name=key)
        tris += len(f)
        print(f"  {key:12s} {len(f):8,d} tris   rgba {np.round(d['rgba'], 2)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(dst), include_normals=True)
    print(f"wrote {dst} : {tris:,} tris in {len(groups)} meshes, {dst.stat().st_size / 1e6:.1f} MB  [{time.time() - t0:.0f}s]")
    return tris


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--out", default=str(ROOT / "game" / "2025" / "visual" / "field.glb"))
    a = ap.parse_args()
    from ghostsim.geometry import IN
    build(find_source(a.source), Path(a.out), a.budget, 690.875 * IN, 317.0 * IN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
