"""Cube-built kinematic model of our robot (visualisation and geometry checks; not used by the physics).

Frame: x forward (scoring/elevator side), y left, z up, origin at the floor centre. Lengths in metres.
Layout after FRC 254 (2025 Technical Binder): swerve drivebase, multi-stage elevator at the FRONT carrying one
wrist-driven end effector that holds coral AND algae, and an arm-style intake at the REAR that drops to the floor.

Joints (all values are placeholders except where noted in the comments):
  h       elevator travel 0..H_MAX (254 travels ~52 in = 1.32 m; ours is a placeholder sized to reach L4)
  theta   end-effector wrist angle, 0 = horizontal forward, + up, - down
  psi     intake arm angle, + up (stowed), - down (deployed onto the floor)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IN = 0.0254
FRAME = 26.0 * IN                 # frame outer size without bumpers (robots/ours.json frame_in)
BUMPER_T = 3.25 * IN              # bumper thickness (ours.json bumper_t_in)
H_MAX = 1.42                      # elevator travel (carriage), metres; wrist tip at 1.83 m = L4 branch top (72 in)
EE_LEN = 0.30                     # end-effector arm length from wrist to tip
INTAKE_LEN = 0.40                 # intake arm length (pivot to roller centre)
ROLLER_R = 0.05                   # intake roller bar half height
ALGAE_R = 0.206                   # from game.json piece radius (algae)
PERIMETER_HALF = FRAME / 2
WRIST_MIN_DEG = -20.0            # lowest wrist angle that keeps a held algae (r=0.206) above the floor at h=0

PIVOT_EE_X = 0.30                 # wrist pivot x at h=0 (in front of the carriage)
PIVOT_EE_Z0 = 0.41                # wrist pivot z at h=0
INTAKE_PIVOT = np.array([-0.30, 0.0, 0.13])     # keeps the roller inside R105 (1 ft 6 in) behind the frame

COL = {
    "chassis": (0.27, 0.32, 0.45), "module": (0.12, 0.12, 0.13), "bumper": (0.12, 0.33, 0.85),
    "upright": (0.18, 0.42, 0.95), "carriage": (0.30, 0.30, 0.33), "ee": (0.95, 0.52, 0.12),
    "ee_head": (0.85, 0.85, 0.85), "intake": (0.20, 0.20, 0.22), "roller": (0.80, 0.80, 0.78),
    "coral": (0.97, 0.97, 0.97), "algae": (0.25, 0.80, 0.60), "floor": (0.62, 0.62, 0.60),
}


def rot_y(th):
    """Pitch about y: R @ e_x = (cos, 0, sin)."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


@dataclass
class Box:
    name: str
    color: tuple
    center: np.ndarray
    half: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    def corners(self):
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return self.center + (s * self.half) @ self.R.T          # (8,3); index = 4*(sx)+2*(sy)+(sz)

    def min_z(self):
        return float(self.corners()[:, 2].min())

    def max_x(self):
        return float(self.corners()[:, 0].max())

    def min_x(self):
        return float(self.corners()[:, 0].min())


@dataclass
class Sphere:
    name: str
    color: tuple
    center: np.ndarray
    radius: float

    def min_z(self):
        return float(self.center[2] - self.radius)


def _box(name, color, center, size, R=None):
    return Box(name, color, np.asarray(center, float), np.asarray(size, float) / 2,
               np.eye(3) if R is None else R)


def ee_wrist(h):
    """Wrist pivot position for elevator travel h."""
    z_c = 0.25 + h
    return np.array([PIVOT_EE_X, 0.0, z_c + 0.16]), z_c


def intake_tip(psi):
    """Roller bar centre for intake arm angle psi (+ up, - down)."""
    d = np.array([-np.cos(psi), 0.0, np.sin(psi)])
    return INTAKE_PIVOT + INTAKE_LEN * d


def pose(h=0.0, theta=0.0, psi=75.0, hold=None):
    """All solids for one configuration. psi, theta in degrees. hold: None | 'coral' | 'algae'."""
    h = float(np.clip(h, 0.0, H_MAX))
    th = np.radians(theta)
    ps = np.radians(psi)
    P, z_c = ee_wrist(h)
    z_s1 = 1.03 + 0.5 * h                                    # stage-1 top (stowed height kept under 1.07 m)
    out = []
    # drivebase and bumpers
    out.append(_box("chassis", COL["chassis"], [0, 0, 0.115], [FRAME, FRAME, 0.03]))
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(_box("module", COL["module"], [sx * 0.27, sy * 0.27, 0.07], [0.09, 0.09, 0.10]))
    half_out = PERIMETER_HALF + BUMPER_T
    bz, bh = 0.08, 0.10
    out.append(_box("bumper", COL["bumper"], [half_out - BUMPER_T / 2, 0, bz], [BUMPER_T, 2 * half_out, bh]))
    out.append(_box("bumper", COL["bumper"], [-half_out + BUMPER_T / 2, 0, bz], [BUMPER_T, 2 * half_out, bh]))
    out.append(_box("bumper", COL["bumper"], [0, half_out - BUMPER_T / 2, bz], [2 * half_out, BUMPER_T, bh]))
    out.append(_box("bumper", COL["bumper"], [0, -half_out + BUMPER_T / 2, bz], [2 * half_out, BUMPER_T, bh]))
    # elevator: two stage-1 uprights (y = +-0.20) and the carriage
    for sy in (-1, 1):
        out.append(_box("upright", COL["upright"], [0.22, sy * 0.20, (0.13 + z_s1) / 2], [0.05, 0.05, z_s1 - 0.13]))
    out.append(_box("carriage", COL["carriage"], [0.22, 0.0, z_c + 0.10], [0.12, 0.44, 0.20]))
    # end effector: arm from the wrist to the tip, head with the roller, held piece
    d = np.array([np.cos(th), 0.0, np.sin(th)])
    R_ee = rot_y(th)
    tip = P + EE_LEN * d
    out.append(_box("ee_arm", COL["ee"], P + 0.5 * EE_LEN * d, [EE_LEN, 0.10, 0.08], R_ee))
    out.append(_box("ee_head", COL["ee_head"], tip, [0.08, 0.34, 0.10], R_ee))
    if hold == "coral":
        out.append(_box("coral", COL["coral"], tip + 0.10 * d + np.array([0, 0, 0.0]), [0.30, 0.11, 0.11], R_ee))
    elif hold == "algae":
        out.append(Sphere("algae", COL["algae"], tip + (0.10 + ALGAE_R * 0.9) * d, ALGAE_R))
    # rear intake: arm from the pivot to the roller bar
    Rp = rot_y(np.pi - ps)
    tipi = intake_tip(ps)
    mid = 0.5 * (INTAKE_PIVOT + tipi)
    out.append(_box("intake_arm", COL["intake"], mid, [INTAKE_LEN, 0.05, 0.05], Rp))
    out.append(_box("intake_roller", COL["roller"], tipi, [0.12, 0.56, 2 * ROLLER_R], Rp))
    out.append(_box("intake_roller", COL["intake"], tipi + np.array([0, 0, 0.0]), [0.10, 0.50, 0.04], Rp))
    return out


def sphere_faces(s: Sphere, nlat=8, nlon=12):
    """Quads of a lat-long sphere as (corners(4,3), normal(3,)) list."""
    faces = []
    for i in range(nlat):
        t0, t1 = np.pi * i / nlat, np.pi * (i + 1) / nlat
        for j in range(nlon):
            p0, p1 = 2 * np.pi * j / nlon, 2 * np.pi * (j + 1) / nlon

            def pt(t, p):
                return s.center + s.radius * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            quad = np.array([pt(t0, p0), pt(t1, p0), pt(t1, p1), pt(t0, p1)])
            faces.append((quad, s.color, quad.mean(0) - s.center))
    return faces


def box_faces(b: Box):
    c = b.corners()
    # faces of the box as corner index loops; corner index = 4*sx + 2*sy + sz with bits (0 -> -1, 1 -> +1)
    idx = lambda sx, sy, sz: 4 * sx + 2 * sy + sz
    loops = {
        "-x": [idx(0, 0, 0), idx(0, 0, 1), idx(0, 1, 1), idx(0, 1, 0)],
        "+x": [idx(1, 0, 0), idx(1, 1, 0), idx(1, 1, 1), idx(1, 0, 1)],
        "-y": [idx(0, 0, 0), idx(1, 0, 0), idx(1, 0, 1), idx(0, 0, 1)],
        "+y": [idx(0, 1, 0), idx(0, 1, 1), idx(1, 1, 1), idx(1, 1, 0)],
        "-z": [idx(0, 0, 0), idx(0, 1, 0), idx(1, 1, 0), idx(1, 0, 0)],
        "+z": [idx(0, 0, 1), idx(1, 0, 1), idx(1, 1, 1), idx(0, 1, 1)],
    }
    normals = {"-x": (-1, 0, 0), "+x": (1, 0, 0), "-y": (0, -1, 0), "+y": (0, 1, 0), "-z": (0, 0, -1), "+z": (0, 0, 1)}
    out = []
    for k, loop in loops.items():
        n = b.R @ np.array(normals[k], float)
        out.append((c[loop], b.color, n))
    return out


def all_faces(solids):
    faces = []
    for s in solids:
        if isinstance(s, Sphere):
            faces += sphere_faces(s)
        else:
            faces += box_faces(s)
    return faces
