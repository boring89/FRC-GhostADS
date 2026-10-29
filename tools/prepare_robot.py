"""Robot CAD (Onshape glTF export, e.g. FRC 6328's 2025 robot) -> visual model of our robot for the live view.

    uv run python tools/prepare_robot.py ~/下載/6328-25B-0000.gltf [--budget 120000]

What it does
  - reads the export (base64 or external buffer; Draco-compressed meshes need DracoPy, a dev dependency)
  - splits it into the rigid bodies the mechanism model moves (ghostsim/mechanism.py BODIES): base (drivetrain,
    hopper, fixed elevator stage), stage2, stage3 (inner elevator stages), carriage, effector (pivots on the
    carriage), climber. Top-level sub-assemblies give the body; inside the elevator, the stage of each part comes
    from the stage tubes (named) or from where it sits between them (bearing blocks: the lower ones move with the
    inner stage, the upper ones stay on the outer)
  - drops fasteners and parts smaller than 2 cm; makes levels of detail of every unique part (quadric decimation or
    vertex clustering, whichever keeps its surface) and picks, within the triangle budget, the levels that lose the
    least surface anywhere (pocketed plates and drilled tubes keep their faces, motors and bearings give theirs up);
    crease-angle normals as tools/prepare_field.py; merges each body into one mesh per colour
  - prints the joint geometry it finds (elevator tilt, effector and climber pivots, coral channel) next to the
    constants in ghostsim/mechanism.py so a different CAD can be checked
Every body keeps its CAD pose: joint zero of the mechanism model is the CAD configuration, and the renderer moves the
bodies with the transforms from ghostsim/mechanism.py body_transforms().

Frames: the Onshape export is Z up, +Y = robot front, +X = robot right, origin on the floor at the frame centre.
Robot frame (ghostsim): x forward, y left, z up = (cad_y, -cad_x, cad_z); written as glTF Y-up (x, z, -y). Both
changes are rotations, so the triangle winding stays.
Output: game/2025/visual/robot.glb (not committed, like the field). Without it the renderer draws the box model.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DROP_NAME = re.compile(r"screw|locknut|nut\b|washer|rivet|zip ?tie", re.I)
MIN_EXTENT_M = 0.02
FLOORS = (48, 12)         # coarsest level for thin long parts / other parts
CAD_TO_ROBOT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # (x, y, z) = (cad_y, -cad_x, cad_z)
STAGE_TUBE = re.compile(r"(Outer|Middle|Inner) Elevator (LEFT|RIGHT|BOTTOM) Tube|CROSS BAR \(Attached to (Outer|Middle)",
                        re.I)
STAGE_OF = {"outer": "base", "middle": "stage2", "inner": "stage3"}


# ---------------------------------------------------------------- glTF reading
def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _local(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in n:
        M[:3, :3] = np.diag(n["scale"])
    if "rotation" in n:
        M[:3, :3] = _quat(n["rotation"]) @ M[:3, :3]
    if "translation" in n:
        M[:3, 3] = n["translation"]
    return M


def _buffers(g, src: Path):
    out = []
    for b in g["buffers"]:
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            out.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            out.append((src.parent / uri).read_bytes())
    return out


def _accessor(g, bufs, i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
    k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    stride = bv.get("byteStride", 0)
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    if stride and stride != k * np.dtype(dt).itemsize:
        raw = np.frombuffer(bufs[bv["buffer"]], np.uint8, count=stride * a["count"], offset=off).reshape(-1, stride)
        return raw[:, :k * np.dtype(dt).itemsize].copy().view(dt).reshape(-1, k)
    x = np.frombuffer(bufs[bv["buffer"]], dt, count=a["count"] * k, offset=off)
    return x.reshape(-1, k) if k > 1 else x


def _primitive(g, bufs, p):
    """(vertices (n,3) float32, faces (m,3) int32) of one primitive."""
    ext = p.get("extensions", {}).get("KHR_draco_mesh_compression")
    if ext:
        import DracoPy
        bv = g["bufferViews"][ext["bufferView"]]
        o = bv.get("byteOffset", 0)
        d = DracoPy.decode(bytes(bufs[bv["buffer"]][o:o + bv["byteLength"]]))
        return np.asarray(d.points, np.float32).reshape(-1, 3), np.asarray(d.faces, np.int32).reshape(-1, 3)
    v = _accessor(g, bufs, p["attributes"]["POSITION"]).astype(np.float32)
    f = (_accessor(g, bufs, p["indices"]) if "indices" in p else np.arange(len(v))).astype(np.int32).reshape(-1, 3)
    return v, f


def _rgba(mat) -> np.ndarray:
    c = np.array(mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0.7, 0.7, 0.7, 1.0]), float)
    return c


def load(src: Path):
    """Unique meshes {mesh: [(v, f, rgba)]} and instances [(mesh, world 4x4, name chain from the top assembly)]."""
    raw = src.read_bytes()
    if raw[:4] == b"glTF":                              # binary container: JSON chunk + BIN chunk
        jl = int.from_bytes(raw[12:16], "little")
        g = json.loads(raw[20:20 + jl])
        bufs = [raw[20 + jl + 8:]]
    else:
        g = json.loads(raw)
        bufs = _buffers(g, src)
    mats = [_rgba(m) for m in g.get("materials", [])] or [np.array([0.7, 0.7, 0.7, 1.0])]
    nodes = g["nodes"]
    inst = []

    def walk(i, M, chain):
        n = nodes[i]
        W = M @ _local(n)
        ch = chain + [re.sub(r"^occurrence of ", "", n.get("name", ""))]
        if "mesh" in n:
            inst.append((n["mesh"], W, ch))
        for c in n.get("children", []):
            walk(c, W, ch)

    for r in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4), [])
    used = {m for m, _, _ in inst}
    meshes = {}
    for mi in used:
        prims = []
        for p in g["meshes"][mi]["primitives"]:
            v, f = _primitive(g, bufs, p)
            prims.append((v, f, mats[p.get("material", 0)] if p.get("material") is not None else mats[0]))
        meshes[mi] = prims
    return meshes, inst


# ---------------------------------------------------------------- bodies
def _mesh_box(prims, W):
    v = np.concatenate([p[0] for p in prims]) @ W[:3, :3].T + W[:3, 3]
    return v.min(0), v.max(0)


def assign_bodies(meshes, inst):
    """Body name per instance (parallel list), using the CAD frame."""
    top = [ch[1] if len(ch) > 1 else ch[0] for _, _, ch in inst]       # ch[0] is the root assembly
    boxes = [_mesh_box(meshes[m], W) for m, W, _ in inst]
    # stage tube lateral positions (|cad x|) from the named tubes
    tube_x = collections.defaultdict(list)
    for (m, W, ch), (lo, hi) in zip(inst, boxes):
        mt = STAGE_TUBE.search(ch[-1])
        if mt and (mt.group(1) or "") and "BOTTOM" not in ch[-1].upper():
            tube_x[mt.group(1).lower()].append(abs((lo[0] + hi[0]) / 2))
    tx = {k: float(np.median(v)) for k, v in tube_x.items()}
    order = [k for k in ("outer", "middle", "inner") if k in tx]
    out = []
    for (m, W, ch), t, (lo, hi) in zip(inst, top, boxes):
        T = t.lower()
        if "drivetrain" in T or "hopper" in T:
            out.append("base")
        elif "climber" in T:
            out.append("climber")
        elif "effector" in T:
            out.append("effector")
        elif "carriage" in T:
            out.append("carriage")
        elif "elevator" in T:
            out.append(_elevator_stage(ch, lo, hi, tx, order))
        else:
            out.append("base")
    return out, tx


def _elevator_stage(ch, lo, hi, tx, order):
    mt = STAGE_TUBE.search(ch[-1])
    if mt:
        return STAGE_OF[(mt.group(1) or mt.group(3)).lower()]
    if len(ch) > 4 or not order:                       # nested sub-assembly (cable chain): on the fixed stage
        return "base"
    a, zc = abs((lo[0] + hi[0]) / 2), (lo[2] + hi[2]) / 2
    if hi[0] - lo[0] > 0.3 or zc < 0.035:              # full-width parts and the plates under the stages: base
        return "base"
    xs = [tx[k] for k in order]
    for k, x in zip(order, xs):
        if abs(a - x) < 0.008:
            return STAGE_OF[k]
    if a > xs[0]:
        return "base"
    if a < xs[-1]:
        return STAGE_OF[order[-1]]
    for i in range(len(xs) - 1):
        if xs[i + 1] <= a <= xs[i]:                     # between two tubes: lower blocks ride with the inner stage
            return STAGE_OF[order[i + 1]] if zc < 0.45 else STAGE_OF[order[i]]
    return "base"


# ---------------------------------------------------------------- joint geometry (printed for checking)
def measure(meshes, inst, bodies):
    """Joint geometry in the robot frame, found from named parts of the 6328 CAD (None when a part is missing)."""
    def pts(pred):
        v = [np.concatenate([p[0] for p in meshes[m]]) @ W[:3, :3].T + W[:3, 3]
             for (m, W, ch) in inst if pred(ch[-1])]
        return np.concatenate(v) @ CAD_TO_ROBOT.T if v else None

    def centre(pred):
        p = pts(pred)
        return None if p is None else (p.min(0) + p.max(0)) / 2

    out = {}
    p = pts(lambda n: re.search(r"Outer Elevator LEFT Tube", n))
    if p is not None:
        p = p[p[:, 1] > 0] if (p[:, 1] > 0).any() else p    # one tube (robot left)
        c = p - p.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        axis *= np.sign(axis[2])
        out["tilt_deg"] = float(np.degrees(np.arctan2(axis[0], axis[2])))
    out["effector_pivot"] = centre(lambda n: n.startswith("Pulley-HTD5-T72"))
    out["climber_pivot"] = centre(lambda n: "24T 35 Chain" in n)
    top, bot = pts(lambda n: n == "Part 30"), pts(lambda n: n == "Part 31")
    if top is not None and bot is not None:
        c = np.concatenate([top, bot]) - np.concatenate([top, bot]).mean(0)
        d = np.linalg.svd(bot - bot.mean(0), full_matrices=False)[2]
        lng = d[0] * np.sign(d[0][0])
        out["coral_centre"] = (top.mean(0) + bot.mean(0)) / 2
        out["coral_pitch_deg"] = float(np.degrees(np.arctan2(lng[2], lng[0])))
    out["stow_top_z"] = float(max(_mesh_box(meshes[m], W)[1][2] for m, W, _ in inst))
    return {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(v, 4) if v is not None else None)
            for k, v in out.items()}


# ---------------------------------------------------------------- decimation
def _cluster(v, f, cell):
    """Vertex clustering: vertices in one grid cell merge into their mean; collapsed and duplicate faces go."""
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nv = np.zeros((inv.max() + 1, 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv)[:, None]
    ff = inv[f]
    ff = ff[(ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])]
    _, keep = np.unique(np.sort(ff, axis=1), axis=0, return_index=True)
    return nv, ff[np.sort(keep)]


LADDER_MAX = 16_000       # finest level tried for a part (the CAD has parts of 80k triangles)


def _area(v, f):
    return float(np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum() / 2)


def _ladder(v, f, floor):
    """Levels of detail of one part from fine to coarse, halving the triangle count down to `floor`:
    [(triangles, surface area lost, v, f)]. Each level takes quadric decimation (fast_simplification) or vertex
    clustering, whichever keeps the surface area closer to the original: what decimation loses on this CAD are the
    pockets and hole patterns of thin plates and tubes (they open into gaps), and toothed or many-featured parts
    (pulleys, bearings, motors) that quadric decimation cannot reduce at all."""
    import fast_simplification as fs
    import trimesh
    tm = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f, np.int64), process=True)      # weld per-face vertices
    v, f = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64)
    a0 = _area(v, f)
    out = [(len(f), 0.0, v, f)] if len(f) <= LADDER_MAX else []
    t = min(len(f), LADDER_MAX) if len(f) > LADDER_MAX else len(f) // 2
    ext = float(np.ptp(v, axis=0).max())
    while t >= max(floor, 4):
        cands = []
        vq, fq = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_count=t, agg=7)
        if 0 < len(fq) <= 1.3 * t:
            cands.append((np.asarray(vq, np.float64), np.asarray(fq, np.int64)))
        lo, hi = ext / 1000, ext / 2
        for _ in range(12):                                   # finest grid that meets the target
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if len(_cluster(v, f, mid)[1]) > t else (lo, mid)
        vc, fc = _cluster(v, f, hi)
        if len(fc):
            cands.append((vc, fc))
        if cands:
            vb, fb = min(cands, key=lambda c: abs(_area(*c) - a0))
            out.append((len(fb), abs(_area(vb, fb) - a0), vb, fb))
        if t == floor:
            break
        t = max(t // 2, floor)
    return out


def _choose(units, budget):
    """Per unit (use count, ladder): the coarsest level whose lost area is <= tau, with tau the smallest that fits the
    budget (the drawn total over all instances). Returns ({unit: level}, tau, drawn)."""
    def pick(tau):
        sel, tot = {}, 0
        for u, (use, lad) in units.items():
            ok = [i for i, l in enumerate(lad) if l[1] <= tau]
            i = min(ok, key=lambda i: lad[i][0]) if ok else 0
            sel[u] = i
            tot += use * lad[i][0]
        return sel, tot
    lo, hi = 0.0, max(l[1] for _, lad in units.values() for l in lad) + 1e-9
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pick(mid)[1] <= budget else (mid, hi)
    sel, tot = pick(hi)
    return sel, hi, tot


def _shade(v, f):
    import trimesh
    from prepare_field import CREASE_DEG
    tm = trimesh.Trimesh(v, f, process=True)
    if len(tm.faces) == 0:
        return None
    tm = trimesh.graph.smooth_shade(tm, angle=np.radians(CREASE_DEG), facet_minarea=None)
    return np.asarray(tm.vertices), np.asarray(tm.faces), np.asarray(tm.vertex_normals)


# ---------------------------------------------------------------- build
def build(src: Path, dst: Path, budget: int):
    import trimesh
    from prepare_field import ELONGATED

    t0 = time.time()
    meshes, inst = load(src)
    bodies, tx = assign_bodies(meshes, inst)
    print(f"{len(inst)} instances, {len(meshes)} unique meshes  [{time.time() - t0:.0f}s]; stage tubes |x| {tx}")
    keep = []
    for (m, W, ch), b in zip(inst, bodies):
        lo, hi = _mesh_box(meshes[m], W)
        if DROP_NAME.search(ch[-1]) or (hi - lo).max() < MIN_EXTENT_M:
            continue
        keep.append((m, W, b))
    print("kept per body:", dict(collections.Counter(b for _, _, b in keep)))
    # levels of detail per unique (mesh, colour), then the levels that lose the least surface for the budget
    use = collections.Counter(m for m, _, _ in keep)
    units = {}
    t1 = time.time()
    for m in use:
        allv = np.concatenate([p[0] for p in meshes[m]])
        e = np.sort(np.ptp(allv, axis=0))[::-1]
        thin_long = e[0] > ELONGATED[0] and e[0] > ELONGATED[1] * max(e[1], 1e-6)
        by_col = collections.defaultdict(list)
        for v, f, rgba in meshes[m]:
            by_col[tuple(np.round(rgba, 4))].append((v, f))
        for col, pl in by_col.items():
            v = np.concatenate([p[0] for p in pl])
            k = np.cumsum([0] + [len(p[0]) for p in pl[:-1]])
            f = np.concatenate([p[1] + kk for p, kk in zip(pl, k)])
            lad = _ladder(v, f, FLOORS[0] if thin_long else FLOORS[1])
            if lad:
                units[(m, col)] = (use[m], lad)
    sel, tau, drawn = _choose(units, budget)
    total_in = sum(use[m] * sum(len(p[1]) for p in meshes[m]) for m in use)
    print(f"{total_in:,} tris drawn -> {drawn:,} for budget {budget:,}; surface lost per part <= {tau * 1e4:.1f} cm^2 "
          f"[{time.time() - t1:.0f}s]")
    cache = {}
    for (m, col), (_, lad) in units.items():
        part = _shade(lad[sel[(m, col)]][2], lad[sel[(m, col)]][3])
        if part is not None:
            cache.setdefault(m, []).append((*part, np.array(col)))
    groups: dict[tuple, dict] = {}
    for m, W, b in keep:
        if m not in cache:
            continue
        A = CAD_TO_ROBOT @ W[:3, :3]
        t = CAD_TO_ROBOT @ W[:3, 3]
        for v, f, nv, rgba in cache[m]:
            wv = v @ A.T + t
            nw = nv @ np.linalg.inv(A)
            nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12)
            ff = f[:, [0, 2, 1]] if np.linalg.det(A) < 0 else f
            key = (b, "".join(f"{int(round(c * 255)):02x}" for c in rgba[:3]))
            d = groups.setdefault(key, {"v": [], "f": [], "n": [], "k": 0, "rgba": rgba})
            d["v"].append(wv)
            d["n"].append(nw)
            d["f"].append(ff + d["k"])
            d["k"] += len(wv)
    out = trimesh.Scene()
    tris = collections.Counter()
    from ghostsim.mechanism import BODIES
    for b in BODIES:
        out.graph.update(frame_to=b, frame_from=out.graph.base_frame)
    for (b, col), d in sorted(groups.items()):
        v, f, nn = np.concatenate(d["v"]), np.concatenate(d["f"]), np.concatenate(d["n"])
        gv = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)               # robot -> glTF Y-up (a rotation)
        gn = np.stack([nn[:, 0], nn[:, 2], -nn[:, 1]], 1)
        mesh = trimesh.Trimesh(gv, f, vertex_normals=gn, process=False)
        mat = trimesh.visual.material.PBRMaterial(name=f"m_{col}", baseColorFactor=(d["rgba"] * 255).astype(np.uint8).tolist(),
                                                  roughnessFactor=0.55, metallicFactor=0.3, doubleSided=True)
        mesh.visual = trimesh.visual.TextureVisuals(material=mat)
        out.add_geometry(mesh, node_name=f"{b}__{col}", geom_name=f"{b}__{col}", parent_node_name=b)
        tris[b] += len(f)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(dst), include_normals=True)
    print("tris per body:", dict(tris), "total", sum(tris.values()))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)  [{time.time() - t0:.0f}s]")
    geo = measure(meshes, inst, bodies)
    from ghostsim import mechanism as mech
    print("joint geometry from the CAD (robot frame), next to ghostsim/mechanism.py:")
    print(f"  elevator tilt     {geo.get('tilt_deg')} deg   (TILT_DEG {mech.TILT_DEG})")
    print(f"  effector pivot    {geo.get('effector_pivot')}   (EFF_PIVOT {mech.EFF_PIVOT.tolist()})")
    print(f"  climber pivot     {geo.get('climber_pivot')}   (CLIMB_PIVOT {mech.CLIMB_PIVOT.tolist()})")
    print(f"  coral centre      {geo.get('coral_centre')}   (EFF_PIVOT + CORAL_IN_EFF "
          f"{np.round(mech.EFF_PIVOT + mech.CORAL_IN_EFF, 4).tolist()})")
    print(f"  coral pitch       {geo.get('coral_pitch_deg')} deg   (CORAL_PITCH0 {mech.CORAL_PITCH0})")
    print(f"  stowed height     {geo.get('stow_top_z')} m   (STOW_TOP_Z {mech.STOW_TOP_Z})")
    return dict(tris), geo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Onshape glTF export of the robot (.gltf or .glb)")
    ap.add_argument("--budget", type=int, default=150_000, help="triangles for the whole robot")
    ap.add_argument("--out", default=str(ROOT / "game" / "2025" / "visual" / "robot.glb"))
    a = ap.parse_args()
    build(Path(a.source).expanduser(), Path(a.out), a.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
