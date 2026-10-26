"""Vectorised rigid-body physics for N envs x R swerve robots + floor pieces.

Per 20 ms substep:
  1. steer modules toward target azimuth (rate limited)
  2. projected Gauss-Seidel/Jacobi sequential impulses, solved together:
       - per-module tyre impulse toward the commanded contact velocity,
         longitudinal part bounded by motor force (battery voltage, back-EMF, current limit),
         total bounded by mu_s * N (else slip, mu_k)
       - robot-robot, robot-obstacle, robot-wall normal impulses at the contact point (with torque)
  3. integrate, then positional correction of residual penetration
  4. wheel encoder speed (slip-aware), battery sag
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import geometry as geo
from .robots import R

G = 9.81
PAIRS = np.array(list(combinations(range(R), 2)))
EI = np.eye(R)[:, PAIRS[:, 0]]
EJ = np.eye(R)[:, PAIRS[:, 1]]


@dataclass
class Body:
    pose: np.ndarray        # (N,R,3)
    vel: np.ndarray         # (N,R,3) field frame
    az: np.ndarray          # (N,R,4) module azimuth, robot frame
    wheel: np.ndarray       # (N,R,4) wheel surface speed seen by encoders
    vbatt: np.ndarray       # (N,R)
    fixed: np.ndarray       # (N,R) bool, immovable (e.g. hanging from a cage)
    slip: np.ndarray        # (N,R,4) bool
    pair_contact: np.ndarray    # (N,15) bool
    poly_contact: np.ndarray    # (N,R,P) bool
    wall_contact: np.ndarray    # (N,R) bool, touching a field half-plane (walls, chamfers)
    impact: np.ndarray          # (N,R) m/s, largest approach speed of a contact that STARTED this substep (0 if none)


@dataclass
class PhysCfg:
    iters: int = 6
    relax: float = 0.7
    slop: float = 0.002
    margin: float = 0.01
    pos_passes: int = 2


def _cross_w(w, r):
    """omega (scalar) x r (...,2)."""
    return np.stack([-w * r[..., 1], w * r[..., 0]], -1)


class Physics:
    def __init__(self, polys: geo.Polygons, hp_n: np.ndarray, hp_off: np.ndarray, cfg: PhysCfg = PhysCfg()):
        self.polys, self.hp_n, self.hp_off, self.cfg = polys, hp_n, hp_off, cfg

    def new_body(self, n):
        z = np.zeros
        return Body(pose=z((n, R, 3)), vel=z((n, R, 3)), az=z((n, R, 4)), wheel=z((n, R, 4)),
                    vbatt=np.full((n, R), 12.6), fixed=z((n, R), bool), slip=z((n, R, 4), bool),
                    pair_contact=z((n, len(PAIRS)), bool), poly_contact=z((n, R, len(self.polys)), bool),
                    wall_contact=z((n, R), bool), impact=z((n, R)))

    # ------------------------------------------------------------------ contacts (sparse lists)
    def _contacts(self, pose, half, cs, sn, inv_m, inv_I, prev):
        """Returns (contacts or None, pact, oact, wact, onset). contacts: (env, robot_a, robot_b or -1, normal a->b,
        r_a, r_b, k_eff, has_b, imb, iIb); onset: per contact, True if that contact was not touching last substep.
        prev = (pair_contact, poly_contact, wall_contact) from the previous substep."""
        c = self.cfg
        n = pose.shape[0]
        a, b = PAIRS[:, 0], PAIRS[:, 1]
        pd, pn, pidx = geo.obb_sat(pose[:, a, :2], half[:, a], cs[:, a], sn[:, a],
                                   pose[:, b, :2], half[:, b], cs[:, b], sn[:, b])
        pact = pd > -c.margin
        E, A, B, NRM, PT, ONS = [], [], [], [], [], []
        e, q = np.nonzero(pact)
        if len(e):
            ia, ib, nn = a[q], b[q], pn[e, q]
            cA = geo.box_corners(pose[e, ia, :2], half[e, ia], cs[e, ia], sn[e, ia])
            cB = geo.box_corners(pose[e, ib, :2], half[e, ib], cs[e, ib], sn[e, ib])
            pt = np.where((pidx[e, q] < 2)[:, None], geo.support(cB, nn, True), geo.support(cA, nn, False))
            E.append(e); A.append(ia); B.append(ib); NRM.append(nn); PT.append(pt); ONS.append(~prev[0][e, q])
        od, on, opt = geo.box_poly_sat(pose[..., :2], half, cs, sn, self.polys)            # robot -> polygon
        oact = od > -c.margin
        e, r_, pi = np.nonzero(oact)
        if len(e):
            E.append(e); A.append(r_); B.append(np.full(len(e), -1)); NRM.append(on[oact]); PT.append(opt[oact])
            ONS.append(~prev[1][e, r_, pi])
        hd, _ = geo.halfplane_box(pose[..., :2], half, cs, sn, self.hp_n, self.hp_off, points=False)
        wact = hd > -c.margin
        e, r_, h = np.nonzero(wact)
        if len(e):
            corners = geo.box_corners(pose[e, r_, :2], half[e, r_], cs[e, r_], sn[e, r_])
            E.append(e); A.append(r_); B.append(np.full(len(e), -1)); NRM.append(-self.hp_n[h])
            PT.append(geo.support(corners, self.hp_n[h], True)); ONS.append(~prev[2][e, r_])
        if not E:
            return None, pact, oact, wact, np.zeros(0, bool)
        E, A, B, NRM, PT, ONS = map(np.concatenate, (E, A, B, NRM, PT, ONS))
        ra = PT - pose[E, A, :2]
        Bc = np.maximum(B, 0)
        rb = np.where((B >= 0)[:, None], PT - pose[E, Bc, :2], 0.0)
        imb = np.where(B >= 0, inv_m[E, Bc], 0.0)
        iIb = np.where(B >= 0, inv_I[E, Bc], 0.0)
        kk = inv_m[E, A] + imb + inv_I[E, A] * geo.cross2(ra, NRM) ** 2 + iIb * geo.cross2(rb, NRM) ** 2
        keff = np.where(kk > 0, 1.0 / np.maximum(kk, 1e-12), 0.0)
        return (E, A, Bc, B >= 0, NRM, ra, rb, keff, imb, iIb), pact, oact, wact, ONS

    # ------------------------------------------------------------------ step
    def substep(self, s: Body, p, tgt_spd, tgt_az, dt):
        c = self.cfg
        free = ~s.fixed
        s.az = geo.wrap(s.az + np.clip(geo.wrap(tgt_az - s.az), -(p.steer_rate * dt)[..., None],
                                       (p.steer_rate * dt)[..., None]))
        th = s.pose[..., 2]
        cs, sn = np.cos(th), np.sin(th)
        r = geo.rotate(p.offsets, cs[..., None], sn[..., None])                      # (N,R,4,2)
        d = np.stack([np.cos(th[..., None] + s.az), np.sin(th[..., None] + s.az)], -1)
        t = np.stack([-d[..., 1], d[..., 0]], -1)
        inv_m = np.where(free, 1.0 / p.mass, 0.0)
        r2 = (p.offsets ** 2).sum(-1).mean(-1)
        inv_I = np.where(free, 1.0 / np.maximum(p.inertia, p.mass * r2), 0.0)       # relaxed for Jacobi stability
        Nn = (p.mass * G / 4)[..., None]
        vel = s.vel
        vc0 = vel[..., None, :2] + _cross_w(vel[..., None, 2], r)
        w0 = (vc0 * d).sum(-1)
        u = tgt_spd[..., None] * d
        vr = (s.vbatt / 12.0)[..., None]
        wfree = p.free_speed[..., None]
        acc_b = np.minimum(p.force_limit[..., None], p.stall_force[..., None] * np.maximum(0, vr - np.abs(w0) / wfree)) * dt
        brk_b = np.minimum(p.force_limit[..., None], p.stall_force[..., None] * (vr + np.abs(w0) / wfree)) * dt
        lim_s, lim_k = p.mu_s[..., None] * Nn * dt, p.mu_k[..., None] * Nn * dt
        J = np.zeros(r.shape)
        slip = np.zeros(r.shape[:-1], bool)
        con, pact, oact, wact, ons = self._contacts(s.pose, p.half, cs, sn, inv_m, inv_I,
                                                    (s.pair_contact, s.poly_contact, s.wall_contact))
        s.impact = np.zeros(s.pose.shape[:2])
        if con is not None:
            E, A, B, hasb, NRM, ra, rb, keff, imb, iIb = con
            lam = np.zeros(len(E))
            # approach speed at the start of the substep, for contacts that just began (damage model)
            va0 = vel[E, A, :2] + _cross_w(vel[E, A, 2], ra)
            vb0 = np.where(hasb[:, None], vel[E, B, :2] + _cross_w(vel[E, B, 2], rb), 0.0)
            app = np.maximum(0.0, -((vb0 - va0) * NRM).sum(-1)) * ons
            np.maximum.at(s.impact, (E, A), app)
            np.maximum.at(s.impact, (E[hasb], B[hasb]), app[hasb])
        quarter = (p.mass / 4)[..., None, None]
        for _ in range(c.iters):
            vc = vel[..., None, :2] + _cross_w(vel[..., None, 2], r)
            Jn = J + quarter * (u - vc)
            jl, jt = (Jn * d).sum(-1), (Jn * t).sum(-1)
            bnd = np.where(jl * w0 >= 0, acc_b, brk_b)
            jl = np.clip(jl, -bnd, bnd)
            mag = np.hypot(jl, jt)
            slip = mag > lim_s
            sc = np.where(slip, lim_k / np.maximum(mag, 1e-12), 1.0)
            Jn = ((jl * sc)[..., None] * d + (jt * sc)[..., None] * t) * free[..., None, None]
            dJ, J = Jn - J, Jn
            vel[..., :2] += dJ.sum(-2) * inv_m[..., None]
            vel[..., 2] += geo.cross2(r, dJ).sum(-1) * inv_I
            if con is not None:
                va = vel[E, A, :2] + _cross_w(vel[E, A, 2], ra)
                vb = np.where(hasb[:, None], vel[E, B, :2] + _cross_w(vel[E, B, 2], rb), 0.0)
                vn = ((vb - va) * NRM).sum(-1)                                      # < 0: approaching
                new = np.maximum(0.0, lam - c.relax * vn * keff)
                dl = (new - lam)[:, None] * NRM
                lam = new
                np.add.at(vel, (E, A, slice(0, 2)), -dl * inv_m[E, A][:, None])
                np.add.at(vel, (E, A, 2), -geo.cross2(ra, dl) * inv_I[E, A])
                np.add.at(vel, (E, B, slice(0, 2)), dl * imb[:, None])
                np.add.at(vel, (E, B, 2), geo.cross2(rb, dl) * iIb)

        vel[s.fixed] = 0.0
        s.pose += vel * dt
        s.pose[..., 2] = geo.wrap(s.pose[..., 2])
        self._position_correction(s, p, inv_m)

        # encoders and battery
        th = s.pose[..., 2]
        r = geo.rotate(p.offsets, np.cos(th)[..., None], np.sin(th)[..., None])
        vc = vel[..., None, :2] + _cross_w(vel[..., None, 2], r)
        spin = np.clip(tgt_spd, -(p.free_speed * s.vbatt / 12)[..., None], (p.free_speed * s.vbatt / 12)[..., None])
        s.wheel = np.where(slip, spin, (vc * d).sum(-1))
        s.slip = slip
        F = np.abs((J * d).sum(-1)) / dt
        duty = np.minimum(1.0, 12.0 * (F / p.stall_force[..., None] + np.abs(w0) / wfree) / s.vbatt[..., None])
        I = (F / p.stall_force[..., None] * p.stall_current[..., None] * duty).sum(-1) + p.other_load
        target = np.maximum(p.v_oc - p.r_int * I, 6.0)
        s.vbatt += (target - s.vbatt) * min(1.0, dt / 0.05)
        s.pair_contact, s.poly_contact, s.wall_contact = pact, oact, wact.any(-1)

    def _position_correction(self, s, p, inv_m):
        c = self.cfg
        a, b = PAIRS[:, 0], PAIRS[:, 1]
        mov = (inv_m > 0)[..., None]
        for _ in range(c.pos_passes):
            th = s.pose[..., 2]
            cs, sn = np.cos(th), np.sin(th)
            depth, n, _ = geo.obb_sat(s.pose[:, a, :2], p.half[:, a], cs[:, a], sn[:, a],
                                      s.pose[:, b, :2], p.half[:, b], cs[:, b], sn[:, b])
            corr = np.maximum(depth - c.slop, 0.0)
            if corr.any():
                wsum = inv_m[:, a] + inv_m[:, b]
                k = np.where(wsum > 0, corr / np.maximum(wsum, 1e-12), 0.0)[..., None] * n
                s.pose[..., :2] += EI @ (-k * inv_m[:, a, None]) + EJ @ (k * inv_m[:, b, None])
            od, on, _ = geo.box_poly_sat(s.pose[..., :2], p.half, cs, sn, self.polys, points=False)
            s.pose[..., :2] -= (np.maximum(od - c.slop, 0)[..., None] * on).sum(-2) * mov
            hd, _ = geo.halfplane_box(s.pose[..., :2], p.half, cs, sn, self.hp_n, self.hp_off, points=False)
            s.pose[..., :2] += (np.maximum(hd - c.slop, 0)[..., None] * self.hp_n).sum(-2) * mov


# ---------------------------------------------------------------------- pieces
@dataclass
class Pieces:
    pos: np.ndarray      # (N,K,2)
    vel: np.ndarray      # (N,K,2)
    yaw: np.ndarray      # (N,K)
    active: np.ndarray   # (N,K) bool, on the floor and simulated


def piece_step(pc: Pieces, radius, decel, rest, body: Body, half, phys, dt):
    """Light pieces on the floor (sparse over active ones): pushed by robots, walls and obstacles;
    no reaction on robots."""
    e, k = np.nonzero(pc.active)
    if len(e) == 0:
        return
    pos, v = pc.pos[e, k], pc.vel[e, k]
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v * np.maximum(0.0, sp - decel * dt) / np.maximum(sp, 1e-9)
    pos = pos + v * dt
    bp, bv = body.pose[e], body.vel[e]                                                      # (M,R,3)
    depth, n = geo.circle_box(pos[:, None], radius, bp[..., :2], half[e], np.cos(bp[..., 2]), np.sin(bp[..., 2]))
    depth, n = depth[..., 0], n[..., 0, :]                                                  # (M,R)
    hit = depth > 0
    if hit.any():
        rel = pos[:, None] - bp[..., :2]
        vr = bv[..., :2] + _cross_w(bv[..., 2], rel)
        vn = ((v[:, None] - vr) * n).sum(-1)
        v = v + (np.where(hit & (vn < 0), -(1 + rest) * vn, 0.0)[..., None] * n).sum(1)
        pos = pos + (np.where(hit, depth, 0.0)[..., None] * n).sum(1)
    dist = pos @ phys.hp_n.T - phys.hp_off                                                  # (M,H)
    pen = np.maximum(radius - dist, 0.0)
    if pen.any():
        pos = pos + pen @ phys.hp_n
        vn = v @ phys.hp_n.T
        v = v - np.where((pen > 0) & (vn < 0), (1 + rest) * vn, 0.0) @ phys.hp_n
    od, on = geo.circle_poly(pos, radius, phys.polys)                                       # (M,P)
    pen = np.maximum(od, 0.0)
    if pen.any():
        pos = pos + (pen[..., None] * on).sum(1)
        vn = (v[:, None] * on).sum(-1)
        v = v - (np.where((pen > 0) & (vn < 0), (1 + rest) * vn, 0.0)[..., None] * on).sum(1)
    pc.pos[e, k], pc.vel[e, k] = pos, v
