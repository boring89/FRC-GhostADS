"""Cross-check the official CAD (source GLB) against the simulator's collision geometry and tag file.

    uv run python tools/check_field.py [source.glb]

Checks (tolerance in brackets): all 22 tag panels vs the tag file [1 cm]; the 6 cages vs Game.cage_xy [1 cm]
and size; the barge centre support vs Game.polys [1 cm]; the reef base hexagon vs Game.polys [3 cm];
and (reported, not asserted) where the guardrail / alliance-wall inner faces sit relative to the boundary.
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_field import clean, find_source  # noqa: E402

IN = 0.0254


class Cad:
    def __init__(self, path, shift):
        import trimesh
        self.sc = trimesh.load(str(path), force="scene")
        self.parents = {v: u for u, v, _ in self.sc.graph.to_edgelist()}
        self.shift = np.asarray(shift, float)
        self._box: dict[str, tuple] = {}

    def chain(self, node):
        out = [node]
        while out[-1] in self.parents:
            out.append(self.parents[out[-1]])
        return [clean(c) for c in out]

    def bbox(self, node):
        if node not in self._box:
            T, g = self.sc.graph[node]
            b = self.sc.geometry[g].bounds
            c = np.array([[x, y, z, 1] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]]) @ T.T
            self._box[node] = (c[:, :3].min(0) + self.shift, c[:, :3].max(0) + self.shift)
        return self._box[node]

    def union(self, nodes):
        bs = [self.bbox(n) for n in nodes]
        return np.min([b[0] for b in bs], 0), np.max([b[1] for b in bs], 0)

    def select(self, pred):
        return [n for n in self.sc.graph.nodes_geometry if pred(self.chain(n))]


def reef_face_points(cad, g, a, f, per_m2=40000):
    """Surface samples of the reef parts in front of face f of alliance a, in the face frame
    (outward along the face normal from the tag plane, lateral, z). (P,3)"""
    import trimesh
    fc, n, t = g.reef_fc[a, f], g.reef_n[a, f], g.reef_t[a, f]
    pts = []
    for node in cad.sc.graph.nodes_geometry:
        lo, hi = cad.bbox(node)
        rel = (lo + hi)[:2] / 2 - fc
        if abs(rel @ t) < 0.45 and -0.6 < rel @ n < 0.35 and hi[2] > 0.4 and lo[2] < 2.0:
            T, gname = cad.sc.graph[node]
            m = cad.sc.geometry[gname]
            if not hasattr(m, "faces") or len(m.faces) == 0:
                continue
            m = m.copy()
            m.apply_transform(T)
            s, _ = trimesh.sample.sample_surface(m, max(200, int(m.area * per_m2)))
            pts.append(s + cad.shift)
    p = np.concatenate(pts)
    rel = p[:, :2] - fc
    return np.stack([rel @ n, rel @ t, p[:, 2]], -1)


def _nearest(pts, q):
    from scipy.spatial import cKDTree
    return float(cKDTree(pts).query(np.atleast_2d(q))[0][0])


def algae_rest(pts, r, z_lo, z_hi):
    """Lowest sphere centre (outward, z) at lateral 0 between z_lo and z_hi that touches but does not penetrate the
    branches, with the ball behind the branch tips (cradled by the pair, not falling off the front)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    best = None
    zs = np.arange(z_lo, z_hi, 0.002)
    for o in np.arange(-0.30, -0.02, 0.005):
        d, _ = tree.query(np.stack([np.full_like(zs, o), np.zeros_like(zs), zs], -1))
        free = d >= r
        if free[0] or not free.any():                 # nothing below holds the ball here
            continue
        z = zs[np.argmax(free)]
        if best is None or z < best[1]:
            best = (o, z)
    return best


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from ghostsim import load
    g = load("2025")
    cad = Cad(find_source(sys.argv[1] if len(sys.argv) > 1 else None), [g.length / 2, g.width / 2, 0.0])
    ok = True

    def report(name, err, tol, unit="mm", scale=1000.0):
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name:44s} {err * scale:8.1f} {unit} (tol {tol * scale:.0f})")

    print("AprilTag panels vs tag file")
    by_id = collections.defaultdict(list)
    for n in cad.select(lambda ch: any(re.match(r"FE-00083_ID\d+", c) for c in ch)):
        i = int(next(re.match(r"FE-00083_ID(\d+)", c).group(1) for c in cad.chain(n) if re.match(r"FE-00083_ID(\d+)", c)))
        by_id[i].append(n)
    worst = 0.0
    for i, nodes in sorted(by_id.items()):
        lo, hi = cad.union(nodes)
        worst = max(worst, np.linalg.norm((lo + hi) / 2 - g.tags[i]["pos"]))
    report(f"{len(by_id)} tags, worst centre error", worst, 0.01)

    print("Cages")
    cages = {}
    for n in cad.select(lambda ch: any(c.startswith("CAGE (GE-25231)") for c in ch)):
        key = next(c for c in cad.chain(n) if c.startswith("CAGE (GE-25231)"))
        cages.setdefault(key, []).append(n)
    cxy = []
    for key, nodes in sorted(cages.items()):
        lo, hi = cad.union(nodes)
        cxy.append(((lo + hi)[:2] / 2, hi[:2] - lo[:2], lo[2]))
    centres = np.array([c[0] for c in cxy])
    d = max(np.min(np.linalg.norm(centres - q, axis=1)) for q in g.cage_xy)
    report("cage centres vs Game.cage_xy (matched)", d, 0.01)
    report("cage footprint size", max(abs(c[1] - g.cfg["field"]["cages"]["size"] * IN).max() for c in cxy), 0.005)
    print("   cage bottoms [m]:", sorted(round(c[2], 3) for c in cxy),
          "(deep 0.079 / shallow 0.765 expected from game.json)")

    print("Barge centre support")
    sup = cad.select(lambda ch: any(c.startswith("BARGE") for c in ch) and ch[1].startswith("FE-01003"))
    sup = [n for n in sup if abs((cad.bbox(n)[0][1] + cad.bbox(n)[1][1]) / 2 - g.width / 2) < 0.3]
    lo, hi = cad.union(sup)
    poly = g.polys.V[2]
    report("support x extent", abs((hi[0] - lo[0]) - np.ptp(poly[:, 0])), 0.01)
    report("support y extent", abs((hi[1] - lo[1]) - np.ptp(poly[:, 1])), 0.01)
    report("support centre", np.linalg.norm((lo + hi)[:2] / 2 - g.polys.center[2]), 0.01)

    print("Reef (parts in the bumper height band z 0.065..0.14 m)")
    for a, name in enumerate(("<2>", "<1>")):                          # blue reef = REEF <2> in the source
        nodes = cad.select(lambda ch, name=name: any(c.startswith("REEF (GE-25300) " + name) for c in ch))
        band = [n for n in nodes if cad.bbox(n)[0][2] < 0.14 and cad.bbox(n)[1][2] > 0.065]
        lo, hi = cad.union(band)
        c = g.reef_c[a]
        ap, cr = g.reef_ap, g.reef_ap / np.cos(np.pi / 6)
        label = "blue" if a == 0 else "red"
        report(f"reef {label}: x half-extent vs apothem", abs(max(hi[0] - c[0], c[0] - lo[0]) - ap), 0.02)
        report(f"reef {label}: y half-extent vs circumradius", abs(max(hi[1] - c[1], c[1] - lo[1]) - cr), 0.02)

    print("Staged reef ALGAE (blue face 0 region, surface-sampled CAD)")
    pts = reef_face_points(cad, g, 0, 0)
    for lvl, z_lo, z_hi in (("low", 0.72, 1.25), ("high", 1.12, 1.70)):
        o_rest, z_rest = algae_rest(pts, g.rad["algae"], z_lo, z_hi)
        ar = g.cfg["field"]["reef"]["algae_rest"]
        o_cfg, z_cfg = ar["outward"] * IN, ar["z"][lvl] * IN
        report(f"algae {lvl}: game.json rest pose vs CAD cradle", np.hypot(o_cfg - o_rest, z_cfg - z_rest), 0.01)
        report(f"algae {lvl}: penetration into branches at that pose", max(0.0, g.rad["algae"] - _nearest(pts, [o_cfg, 0.0, z_cfg])), 0.005)

    print("Boundary (reported only; bumper band)")
    mid = g.mid
    wall = cad.select(lambda ch: any(c.startswith("FIELD WALL") for c in ch))
    wall = [n for n in wall if cad.bbox(n)[0][2] < 0.14 and cad.bbox(n)[1][2] > 0.065]
    top = min(cad.bbox(n)[0][1] for n in wall if cad.bbox(n)[0][1] > mid[1])
    bot = max(cad.bbox(n)[1][1] for n in wall if cad.bbox(n)[1][1] < mid[1])
    print(f"   guardrail inner faces at y = {bot:.3f} and {top:.3f} (boundary 0 and {g.width:.3f}): "
          f"{bot * 100:+.1f} cm / {(top - g.width) * 100:+.1f} cm")
    ds = cad.select(lambda ch: any(c.startswith("DRIVER STATIONS") for c in ch))
    ds = [n for n in ds if cad.bbox(n)[0][2] < 0.9 and cad.bbox(n)[1][2] > 0.1]
    left = max(cad.bbox(n)[1][0] for n in ds if cad.bbox(n)[1][0] < mid[0])
    right = min(cad.bbox(n)[0][0] for n in ds if cad.bbox(n)[0][0] > mid[0])
    print(f"   alliance-wall inner faces at x = {left:.3f} and {right:.3f} (boundary 0 and {g.length:.3f}): "
          f"{left * 100:+.1f} cm / {(right - g.length) * 100:+.1f} cm")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
