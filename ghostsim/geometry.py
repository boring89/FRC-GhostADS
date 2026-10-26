"""Broadcastable 2D geometry (field frame, metres). Hot paths are written component-wise."""
from __future__ import annotations

import numpy as np

IN = 0.0254


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rotate(v, c, s):
    """Rotate vectors v (...,2) by angle with cos c, sin s."""
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)


def box_corners(c, h, ca, sa):
    """(...,2) centre, (...,2) half extents, (...) cos/sin -> (...,4,2)."""
    sg = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)
    loc = sg * h[..., None, :]
    return c[..., None, :] + rotate(loc, ca[..., None], sa[..., None])


def support(pts, n, deepest_min=True, tol=0.01):
    """Average of the points (...,M,2) that are extreme along n (...,2) (within tol)."""
    d = (pts * n[..., None, :]).sum(-1)
    ext = d.min(-1, keepdims=True) if deepest_min else d.max(-1, keepdims=True)
    w = (np.abs(d - ext) <= tol).astype(float)
    return (pts * w[..., None]).sum(-2) / w.sum(-1, keepdims=True)


def obb_sat(c1, h1, ca, sa, c2, h2, cb, sb):
    """SAT between two oriented boxes. Returns (depth, normal box1->box2, axis_idx 0..3).
    depth > 0 means overlap. axis 0,1 belong to box 1; 2,3 to box 2."""
    ax, ay, bx, by = h1[..., 0], h1[..., 1], h2[..., 0], h2[..., 1]
    cd = np.abs(ca * cb + sa * sb)
    sd = np.abs(ca * sb - sa * cb)
    dx, dy = c2[..., 0] - c1[..., 0], c2[..., 1] - c1[..., 1]
    cands = (
        (ax + bx * cd + by * sd, dx * ca + dy * sa, ca, sa),
        (ay + bx * sd + by * cd, -dx * sa + dy * ca, -sa, ca),
        (bx + ax * cd + ay * sd, dx * cb + dy * sb, cb, sb),
        (by + ax * sd + ay * cd, -dx * sb + dy * cb, -sb, cb),
    )
    depth = nx = ny = idx = None
    for k, (r, p, ux, uy) in enumerate(cands):
        o = r - np.abs(p)
        sg = np.where(p < 0, -1.0, 1.0)
        if depth is None:
            depth, nx, ny, idx = o, ux * sg, uy * sg, np.zeros(np.shape(o), np.int8)
        else:
            b = o < depth
            depth, nx, ny = np.where(b, o, depth), np.where(b, ux * sg, nx), np.where(b, uy * sg, ny)
            idx = np.where(b, k, idx)
    return depth, np.stack(np.broadcast_arrays(nx, ny), -1), idx


class Polygons:
    """Static convex polygons padded to K vertices. verts (P,K,2) CCW, normals outward."""

    def __init__(self, polys: list[np.ndarray], heights: list[float] | None = None):
        k = max(len(p) for p in polys)
        V = np.zeros((len(polys), k, 2))
        for i, p in enumerate(polys):
            p = np.asarray(p, float)
            if cross2(p[1] - p[0], p[2] - p[1]) < 0:   # make CCW
                p = p[::-1]
            V[i] = np.concatenate([p, np.repeat(p[-1:], k - len(p), 0)])
        e = np.roll(V, -1, 1) - V
        n = np.stack([e[..., 1], -e[..., 0]], -1)
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.where(ln > 1e-9, n / np.maximum(ln, 1e-12), np.nan)
        for i in range(len(polys)):          # padded (zero length) edges reuse a real normal
            bad = np.isnan(n[i, :, 0])
            n[i, bad] = n[i, ~bad][0]
        self.V, self.N = V, n
        self.center = V.mean(1)
        self.radius = np.linalg.norm(V - self.center[:, None], axis=-1).max(1)
        proj = np.einsum("pkd,pjd->pkj", n, V)              # (P, axis k, vertex j)
        self.own_min, self.own_max = proj.min(-1), proj.max(-1)
        self.height = np.asarray(heights if heights is not None else [10.0] * len(polys), float)

    def __len__(self):
        return len(self.V)

    def subset(self, keep):
        out = object.__new__(Polygons)
        for k in ("V", "N", "center", "radius", "own_min", "own_max", "height"):
            setattr(out, k, getattr(self, k)[np.asarray(keep)])
        return out


def box_poly(c, h, ca, sa, poly: Polygons, pid, points=True):
    """Paired SAT: box i vs polygon pid[i]. All args share a leading shape S.
    Returns depth S, normal box->poly S+(2,), contact point S+(2,) (or None)."""
    V, Nr = poly.V[pid], poly.N[pid]                                                    # S+(K,2)
    omin, omax = poly.own_min[pid], poly.own_max[pid]
    cx, cy = c[..., 0:1], c[..., 1:2]
    a_, s_ = ca[..., None], sa[..., None]
    proj_c = cx * Nr[..., 0] + cy * Nr[..., 1]
    rb = h[..., 0:1] * np.abs(a_ * Nr[..., 0] + s_ * Nr[..., 1]) + h[..., 1:2] * np.abs(-s_ * Nr[..., 0] + a_ * Nr[..., 1])
    pc = (omin + omax) / 2
    ov = rb + (omax - omin) / 2 - np.abs(proj_c - pc)
    sg = np.where(pc - proj_c < 0, -1.0, 1.0)
    k = np.argmin(ov, -1)[..., None]
    depth = np.take_along_axis(ov, k, -1)[..., 0]
    g = np.take_along_axis(sg, k, -1)[..., 0]
    nx = np.take_along_axis(Nr[..., 0], k, -1)[..., 0] * g
    ny = np.take_along_axis(Nr[..., 1], k, -1)[..., 0] * g
    from_poly = np.ones(depth.shape, bool)
    for ux, uy, hh in ((ca, sa, h[..., 0]), (-sa, ca, h[..., 1])):
        pv = V[..., 0] * ux[..., None] + V[..., 1] * uy[..., None]
        pmin, pmax = pv.min(-1), pv.max(-1)
        cb = c[..., 0] * ux + c[..., 1] * uy
        o = hh + (pmax - pmin) / 2 - np.abs(cb - (pmin + pmax) / 2)
        sgn = np.where((pmin + pmax) / 2 - cb < 0, -1.0, 1.0)
        b = o < depth
        depth, nx, ny = np.where(b, o, depth), np.where(b, ux * sgn, nx), np.where(b, uy * sgn, ny)
        from_poly &= ~b
    nrm = np.stack([nx, ny], -1)
    if not points:
        return depth, nrm, None
    p_box = support(box_corners(c, h, ca, sa), nrm, deepest_min=False)
    p_poly = support(V, nrm, deepest_min=True)
    return depth, nrm, np.where(from_poly[..., None], p_box, p_poly)


def near_polys(c, rad, poly: Polygons, margin=0.0):
    """Broadphase: (...,2) centres with bounding radius (...) -> candidate mask (...,P)."""
    d = np.linalg.norm(c[..., None, :] - poly.center, axis=-1)
    return d < rad[..., None] + poly.radius + margin


def box_poly_sat(c, h, ca, sa, poly: Polygons, points=True, margin=0.05):
    """All box/polygon pairs with broadphase. Dense outputs (...,P); far pairs get depth -inf."""
    rad = np.hypot(h[..., 0], h[..., 1]) * np.ones(c.shape[:-1])
    cand = near_polys(c, rad, poly, margin)
    depth = np.full(cand.shape, -np.inf)
    nrm = np.zeros(cand.shape + (2,))
    pt = np.zeros(cand.shape + (2,)) if points else None
    idx = np.nonzero(cand)
    if len(idx[0]):
        lead, m = idx[:-1], len(idx[-1])
        g2 = lambda x: np.broadcast_to(np.broadcast_to(x, c.shape)[lead], (m, 2))
        g1 = lambda x: np.broadcast_to(np.broadcast_to(x, c.shape[:-1])[lead], (m,))
        dd, nn, pp = box_poly(g2(c), g2(h), g1(ca), g1(sa), poly, idx[-1], points)
        depth[idx], nrm[idx] = dd, nn
        if points:
            pt[idx] = pp
    return depth, nrm, pt


def halfplane_box(c, h, ca, sa, n, off, points=True):
    """Valid region n.p >= off. Boxes (...,2) vs H half-planes -> depth (...,H), contact point (...,H,2) or None."""
    ext = h[..., None, 0] * np.abs(ca[..., None] * n[:, 0] + sa[..., None] * n[:, 1]) \
        + h[..., None, 1] * np.abs(-sa[..., None] * n[:, 0] + ca[..., None] * n[:, 1])
    depth = off - (c[..., None, 0] * n[:, 0] + c[..., None, 1] * n[:, 1] - ext)
    if not points:
        return depth, None
    corners = box_corners(c, h, ca, sa)[..., None, :, :]
    return depth, support(np.broadcast_to(corners, depth.shape + (4, 2)), np.broadcast_to(n, depth.shape + (2,)))


def circle_box(p, r, c, h, ca, sa):
    """Circles p (...,K,2) vs boxes (...,R,2) -> depth (...,R,K), normal box->circle (...,R,K,2)."""
    d = p[..., None, :, :] - c[..., :, None, :]
    lx = d[..., 0] * ca[..., None] + d[..., 1] * sa[..., None]
    ly = -d[..., 0] * sa[..., None] + d[..., 1] * ca[..., None]
    hx, hy = h[..., None, 0], h[..., None, 1]
    qx, qy = np.clip(lx, -hx, hx), np.clip(ly, -hy, hy)
    ex, ey = lx - qx, ly - qy
    dist = np.hypot(ex, ey)
    inside = dist < 1e-9
    # inside: push out through nearest face
    fx, fy = hx - np.abs(lx), hy - np.abs(ly)
    use_x = fx < fy
    nlx = np.where(inside, np.where(use_x, np.sign(lx) + (lx == 0), 0.0), ex / np.maximum(dist, 1e-9))
    nly = np.where(inside, np.where(use_x, 0.0, np.sign(ly) + (ly == 0)), ey / np.maximum(dist, 1e-9))
    depth = np.where(inside, r + np.minimum(fx, fy), r - dist)
    nx = nlx * ca[..., None] - nly * sa[..., None]
    ny = nlx * sa[..., None] + nly * ca[..., None]
    return depth, np.stack([nx, ny], -1)


def circle_poly(p, r, poly: Polygons):
    """Circles (...,2) vs polygons (approximate: deepest face). Dense (...,P); far pairs depth -inf."""
    cand = near_polys(p, np.full(p.shape[:-1], float(r)), poly, 0.01)
    depth = np.full(cand.shape, -np.inf)
    nrm = np.zeros(cand.shape + (2,))
    idx = np.nonzero(cand)
    if len(idx[0]):
        pid = idx[-1]
        q = np.broadcast_to(p[idx[:-1]], (len(pid), 2))
        sd = (q[:, None, :] * poly.N[pid]).sum(-1) - poly.own_max[pid]
        k = np.argmax(sd, -1)
        depth[idx] = r - sd[np.arange(len(k)), k]
        nrm[idx] = poly.N[pid, k]
    return depth, nrm


def segment_box_hit(a, b, c, h, ca, sa):
    """Does segment a->b (...,2) intersect box (...,2)? Uses SAT with a degenerate box."""
    d = b - a
    ln = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
    hs = np.stack([ln / 2, np.full(ln.shape, 1e-4)], -1)
    depth, _, _ = obb_sat((a + b) / 2, hs, d[..., 0] / ln, d[..., 1] / ln, c, h, ca, sa)
    return depth > 0


def segment_poly_hit(a, b, poly: Polygons):
    """(...,2) segments vs polygons -> (...,P) bool."""
    d = b - a
    ln = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
    hs = np.stack([ln / 2, np.full(ln.shape, 1e-4)], -1)
    depth, _, _ = box_poly_sat((a + b) / 2, hs, d[..., 0] / ln, d[..., 1] / ln, poly, points=False)
    return depth > 0


def regular_polygon(center, apothem, n, first_normal_deg):
    """Vertices of a regular n-gon whose face k has outward normal first_normal + k*360/n."""
    R = apothem / np.cos(np.pi / n)
    ang = np.deg2rad(first_normal_deg) + (np.arange(n) + 0.5) * 2 * np.pi / n
    return np.asarray(center, float) + R * np.stack([np.cos(ang), np.sin(ang)], -1)


def rect(cx, cy, lx, ly, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[lx, ly], [-lx, ly], [-lx, -ly], [lx, -ly]]) / 2
    return np.array([cx, cy]) + rotate(loc, c, s)
