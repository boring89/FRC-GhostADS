"""Kinematic model of our robot: FRC 6328's 2025 robot (Onshape CAD 6328-25B-0000, numbers measured by
tools/prepare_robot.py). Visualisation and geometry checks only; not used by the physics.

  - swerve drivebase, alliance-coloured bumpers sized per robot
  - full-width 3-stage elevator leaning forward TILT_DEG: the bottom stays where it is, the top comes out over the
    front bumper, so the effector reaches toward the reef as it rises. Fixed outer stage, stage 2 (h/3), stage 3
    (2h/3) and the carriage (h) all move along AXIS
  - effector on the carriage, pitched by the wrist about EFF_PIVOT: coral passes through its channel (in from the
    hopper at the back, out over the front bumper); algae is held in front of it, against its front wheel and top
    roller (no separate algae arm: floor algae is picked up with the effector nose down at h = 0)
  - hopper at the BACK: the robot backs into the coral station, coral drops in and the belt feeds it into the effector
    (h = 0, wrist 0)
  - climber arm at the back (deep cage)

Two ways to draw it: renderer bodies (BODIES, posed by body_transforms; meshes in game/2025/visual/robot.glb made by
tools/prepare_robot.py) or boxes, cylinders and spheres of the same layout (tests, tools/mech_view.py, and the renderer
when robot.glb is absent). Both are scaled to each robot's frame (the CAD frame is 28 in).

Frame: x forward (scoring side), y left, z up, origin on the floor at the robot centre. Metres, angles in degrees.
Joint zero is the CAD configuration (elevator down, effector as drawn: coral channel in line with the hopper belt,
climber stowed).
Joints:  h      elevator travel along AXIS, 0..H_MAX
         wrist  effector pitch about EFF_PIVOT, + nose up
         climb  climber arm angle about CLIMB_PIVOT, + swings back and out
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IN = 0.0254
FRAME = 29.5 * IN                 # frame outer size without bumpers (robots/ours.json frame_in)
CAD_FRAME = 28.0 * IN             # frame of the CAD (drivetrain tubes at +-14 in)
BUMPER_T = 3.25 * IN              # bumper thickness (ours.json bumper_t_in)
BUMPER_Z = (0.065, 0.14)          # ours.json bumper_z_m
PERIMETER_HALF = FRAME / 2
ALGAE_R = 16.25 * IN / 2
CORAL_R, CORAL_LEN = 4.5 * IN / 2, 11.875 * IN

# ---- measured on the CAD (robot frame, 28 in frame; tools/prepare_robot.py prints them for a new export)
TILT_DEG = 8.0                                   # elevator axis leans forward (CAD 7.97)
AXIS = np.array([np.sin(np.radians(TILT_DEG)), 0.0, np.cos(np.radians(TILT_DEG))])
STOW_TOP_Z = 0.94                                # tallest point as drawn (R104 starting height 42 in)
H_MAX = 1.60                                     # elevator travel (L4 needs ~1.55)
EFF_PIVOT = np.array([0.141, 0.0, 0.3195])       # effector pivot at h = 0 (axis along y)
EFF_Y = 0.044                                    # effector centre line (it sits left of the robot centre)
CORAL_IN_EFF = np.array([0.017, EFF_Y, 0.005])   # held coral centre - pivot, wrist 0 (CAD 0.1576, 0.3239)
CORAL_PITCH0 = 15.0                              # coral channel pitch at wrist 0 (nose up)
ALGAE_IN_EFF = np.array([0.344, EFF_Y, 0.165])   # held algae centre - pivot, wrist 0: touching the front wheel and
                                                 # the top roller
CLIMB_PIVOT = np.array([-0.305, 0.0, 0.378])
# elevator stage tubes: (name, lateral |y|, bottom-centre (x, z) at h = 0, length along AXIS, share of h)
X0_AXIS = 0.0803                                 # x of the elevator centre line at z = 0
STAGES = (("outer", 0.292, 0.723, 0.065, 0.0), ("stage2", 0.260, 0.757, 0.056, 1 / 3),
          ("stage3", 0.229, 0.859, 0.081, 2 / 3))
BODIES = ("base", "stage2", "stage3", "carriage", "effector", "climber")

STOW = {"h": 0.0, "wrist": 0.0, "climb": 0.0}
WRIST = {"carry": 0.0, "floor": -42.8, "reach": -25.6, "processor": -20.0, "net": 55.0}   # floor: ball on the carpet
CLIMB = {"stow": 0.0, "reach": 70.0, "hang": 25.0}

BLUE, RED = (0.10, 0.25, 0.80), (0.80, 0.10, 0.10)
COL = {
    "tube": (0.0, 0.24, 0.73), "plate": (0.25, 0.25, 0.25), "pan": (0.25, 0.25, 0.25), "module": (0.67, 0.70, 0.77),
    "wheel": (0.23, 0.38, 0.71), "battery": (0.08, 0.08, 0.09), "poly": (0.25, 0.25, 0.25), "belt": (0.53, 0.53, 0.53),
    "roller": (0.09, 0.32, 0.69), "wheel3": (0.49, 0.49, 0.49), "coral": (0.97, 0.97, 0.97), "algae": (0.25, 0.80, 0.60),
    "floor": (0.62, 0.62, 0.60),
}


def rot_y(deg):
    """Pitch about y (degrees): R @ e_x = (cos, 0, sin), + up."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


AXIS_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # cylinder local z -> world y
R_TILT = rot_y(-TILT_DEG)                                                 # local z -> AXIS


@dataclass
class Box:
    name: str
    color: tuple
    center: np.ndarray
    half: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))      # columns: local axes (may be scaled, see scaled())

    def corners(self):
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return self.center + (s * self.half) @ self.R.T          # (8,3); index = 4*(sx)+2*(sy)+(sz)

    def min_z(self):
        return float(self.corners()[:, 2].min())

    def max_z(self):
        return float(self.corners()[:, 2].max())

    def max_x(self):
        return float(self.corners()[:, 0].max())

    def min_x(self):
        return float(self.corners()[:, 0].min())


@dataclass
class Cyl:
    """Cylinder: axis = local z of R, `half_len` along it."""
    name: str
    color: tuple
    center: np.ndarray
    radius: float
    half_len: float
    R: np.ndarray = field(default_factory=lambda: AXIS_Y.copy())

    def rim(self, k=12):
        a = np.linspace(0, 2 * np.pi, k, endpoint=False)
        ring = np.stack([np.cos(a), np.sin(a), np.zeros(k)], -1) * self.radius
        ends = [ring + [0, 0, s * self.half_len] for s in (-1, 1)]
        return [self.center + e @ self.R.T for e in ends]          # two (k,3) rims

    def min_z(self):
        return float(min(r[:, 2].min() for r in self.rim()))

    def max_z(self):
        return float(max(r[:, 2].max() for r in self.rim()))

    def max_x(self):
        return float(max(r[:, 0].max() for r in self.rim()))

    def min_x(self):
        return float(min(r[:, 0].min() for r in self.rim()))


@dataclass
class Sphere:
    name: str
    color: tuple
    center: np.ndarray
    radius: float

    def min_z(self):
        return float(self.center[2] - self.radius)

    def max_z(self):
        return float(self.center[2] + self.radius)

    def max_x(self):
        return float(self.center[0] + self.radius)

    def min_x(self):
        return float(self.center[0] - self.radius)


@dataclass
class Body:
    """A renderer body (one mesh of robot.glb): world point = A @ cad point + t."""
    name: str
    A: np.ndarray
    t: np.ndarray


def _box(name, color, center, size, R=None):
    return Box(name, color, np.asarray(center, float), np.asarray(size, float) / 2, np.eye(3) if R is None else R)


def _cyl_y(name, color, center, radius, length, R=None):
    """Cylinder along the (possibly pitched) y axis."""
    Rw = AXIS_Y if R is None else R @ AXIS_Y
    return Cyl(name, color, np.asarray(center, float), radius, length / 2, Rw)


def _seg(name, color, a, b, w, t):
    """Box from point a to point b (in the x-z plane at a's y), w across (y), t thick."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ang = np.degrees(np.arctan2(d[2], d[0]))
    return _box(name, color, (a + b) / 2, [float(np.linalg.norm(d)), w, t], rot_y(ang))


def scale_of(frame):
    return np.diag([frame / CAD_FRAME, frame / CAD_FRAME, 1.0])


def scaled(solids, S):
    """Apply the frame scale S (diagonal) to CAD-frame solids: exact for boxes and cylinders (their axes are scaled),
    pieces keep their shape."""
    out = []
    for s in solids:
        if isinstance(s, Box):
            out.append(Box(s.name, s.color, S @ s.center, s.half, S @ s.R))
        elif isinstance(s, Cyl):
            keep = s.name in ("coral",)
            out.append(Cyl(s.name, s.color, S @ s.center, s.radius, s.half_len, s.R if keep else S @ s.R))
        elif isinstance(s, Sphere):
            out.append(Sphere(s.name, s.color, S @ s.center, s.radius))
        else:
            out.append(Body(s.name, S @ s.A, S @ s.t))
    return out


# ---------------------------------------------------------------- kinematics (CAD frame; scale with scale_of(frame))
def eff_pivot(h):
    return EFF_PIVOT + AXIS * float(np.clip(h, 0.0, H_MAX))


def coral_held(h, wrist):
    """Centre of a coral in the effector."""
    return eff_pivot(h) + rot_y(wrist) @ CORAL_IN_EFF


def coral_pitch(wrist):
    return CORAL_PITCH0 + wrist


def algae_held(h, wrist):
    return eff_pivot(h) + rot_y(wrist) @ ALGAE_IN_EFF


def coral_target(z_coral, pitch):
    """(h, wrist) that put the held coral centre at height z_coral with the coral pitched `pitch` degrees."""
    w = pitch - CORAL_PITCH0
    return (z_coral - (EFF_PIVOT[2] + (rot_y(w) @ CORAL_IN_EFF)[2])) / AXIS[2], w


def algae_target(z_algae, wrist=WRIST["reach"]):
    """Elevator travel that puts the held algae centre at height z_algae with the given wrist angle."""
    return (z_algae - (EFF_PIVOT[2] + (rot_y(wrist) @ ALGAE_IN_EFF)[2])) / AXIS[2]


# coral path from the hopper mouth to the effector: (x, y, z) of the coral centre and its pitch, s = 0 .. 1
FUNNEL_PATH = (((-0.33, 0.0, 0.68), -25.0), ((-0.274, 0.0, 0.415), -24.7), ((-0.094, 0.0, 0.332), -24.7),
               (tuple(EFF_PIVOT + CORAL_IN_EFF), CORAL_PITCH0))


def funnel_pose(s):
    """Coral centre and pitch on the way down the hopper: s = 0 just dropped in from the station chute (at the back),
    s = 1 in the effector (h = 0, wrist 0)."""
    p = np.array([q for q, _ in FUNNEL_PATH], float)
    a = np.array([w for _, w in FUNNEL_PATH], float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = float(np.clip(s, 0.0, 1.0)) * seg.sum()
    k = int(min(np.searchsorted(np.cumsum(seg), u, side="right"), len(seg) - 1))
    f = (u - np.concatenate([[0.0], np.cumsum(seg)])[k]) / seg[k]
    return p[k] + (p[k + 1] - p[k]) * f, float(a[k] + (a[k + 1] - a[k]) * f)


def body_transforms(h=0.0, wrist=0.0, climb=0.0):
    """Moving bodies relative to the CAD pose: {name: (A, t)} with world = A @ p + t (base: identity)."""
    h = float(np.clip(h, 0.0, H_MAX))
    I = np.eye(3)
    out = {"base": (I, np.zeros(3))}
    for name, _, _, _, k in STAGES[1:]:
        out[name] = (I, AXIS * k * h)
    out["carriage"] = (I, AXIS * h)
    Rw = rot_y(wrist)
    out["effector"] = (Rw, EFF_PIVOT - Rw @ EFF_PIVOT + AXIS * h)
    Rc = rot_y(climb)
    out["climber"] = (Rc, CLIMB_PIVOT - Rc @ CLIMB_PIVOT)
    return out


# ---------------------------------------------------------------- solids
def pose(h=0.0, wrist=0.0, climb=0.0, hold=None, alliance=0, frame=FRAME, funnel=None, cad=False):
    """Everything for one configuration. hold: None | 'coral' | 'algae' | 'both'. funnel: None or the position 0..1 of
    a coral coming down the hopper. frame: frame outer size (m) of this robot. cad: renderer bodies instead of boxes."""
    return pose_static(alliance, frame, cad) + pose_moving(h, wrist, climb, hold, frame, funnel, cad)


def pose_static(alliance=0, frame=FRAME, cad=False):
    """What never moves relative to the robot (drivebase, bumpers, fixed elevator stage, hopper): cacheable."""
    half = frame / 2
    out = []
    S = scale_of(frame)
    if cad:
        out.append(Body("base", S.copy(), np.zeros(3)))
    else:
        # ---- drivebase (lowered swerve: frame tubes low, modules on top)
        out.append(_box("pan", COL["pan"], [0, 0, 0.025], [frame - 0.05, frame - 0.05, 0.006]))
        for s in (-1, 1):
            out.append(_box("rail", COL["tube"], [0, s * (half - 0.025), 0.048], [frame, 0.05, 0.05]))
            out.append(_box("rail", COL["tube"], [s * (half - 0.025), 0, 0.048], [0.05, frame - 0.10, 0.05]))
        m = half - 0.085
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(_box("module", COL["module"], [sx * m, sy * m, 0.12], [0.12, 0.12, 0.10]))
                out.append(_cyl_y("wheel", COL["wheel"], [sx * m, sy * m, 0.05], 0.05, 0.04))
        out.append(_box("battery", COL["battery"], [-0.10, -0.20, 0.10], [0.18, 0.08, 0.17]))
        out += scaled(_stage(STAGES[0], 0.0) + _hopper(), S)
    # ---- bumpers (alliance colour)
    bc = BLUE if alliance == 0 else RED
    ho = half + BUMPER_T
    bz, bh = (BUMPER_Z[0] + BUMPER_Z[1]) / 2, BUMPER_Z[1] - BUMPER_Z[0]
    for s in (-1, 1):
        out.append(_box("bumper", bc, [s * (ho - BUMPER_T / 2), 0, bz], [BUMPER_T, 2 * ho, bh]))
        out.append(_box("bumper", bc, [0, s * (ho - BUMPER_T / 2), bz], [2 * ho - 2 * BUMPER_T, BUMPER_T, bh]))
    return out


def _axis_point(z):
    """Point on the elevator centre line at height z (h = 0)."""
    return np.array([X0_AXIS + z * AXIS[0] / AXIS[2], 0.0, z])


def _stage(st, h):
    """Two tilted tubes (2 x 1 in), a top cross member (plate behind the tubes, or a tube on stage 3) and a bottom
    cross member, moved k * h along AXIS."""
    name, y, length, z0, k = st
    b = _axis_point(z0) + AXIS * k * h
    out = [_box(name, COL["tube"], b + AXIS * length / 2 + [0, s * y, 0], [2 * IN, IN, length], R_TILT)
           for s in (-1, 1)]
    if name == "stage3":
        out.append(_box(name, COL["tube"], b + AXIS * (length - 0.015), [2 * IN, 2 * y + IN, 0.03], R_TILT))
    else:
        out.append(_box(name, COL["plate"], b + AXIS * (length - 0.005) - [0.057, 0, 0], [0.164, 2 * y + IN, 0.03],
                        R_TILT))
    out.append(_box(name, COL["tube"], b + AXIS * 0.015, [2 * IN, 2 * y - IN, 0.03], R_TILT))
    return out


def _hopper():
    """Hopper at the back: polycarbonate side walls, a V of ramp plates over a centre belt that runs down to the
    effector, the front backboard and two diagonal frame tubes."""
    out = []
    for s in (-1, 1):
        out.append(_box("hopper_wall", COL["poly"], [-0.083, s * 0.331, 0.447], [0.545, 0.004, 0.30]))
        out.append(_seg("hopper_tube", COL["tube"], [-0.343, s * 0.318, 0.13], [0.178, s * 0.318, 0.587], IN, IN))
        out.append(_box("hopper_ramp", COL["poly"], [-0.16, s * 0.19, 0.495], [0.34, 0.29, 0.005], rot_x(s * 40.8)))
    out.append(_seg("hopper_belt", COL["belt"], [-0.33, 0.0, 0.38], [-0.06, 0.0, 0.255], 0.12, 0.02))
    out.append(_box("hopper_board", COL["poly"], [0.017, 0.0, 0.446], [0.006, 0.607, 0.241]))
    return out


def pose_moving(h=0.0, wrist=0.0, climb=0.0, hold=None, frame=FRAME, funnel=None, cad=False):
    """What follows the joints, and the game pieces."""
    h = float(np.clip(h, 0.0, H_MAX))
    S = scale_of(frame)
    if cad:
        bt = body_transforms(h, wrist, climb)
        out = [Body(b, *bt[b]) for b in BODIES[1:]]
    else:
        out = []
        for st in STAGES[1:]:
            out += _stage(st, h)
        out += _carriage(h) + _effector(h, wrist) + _climber(climb)
    # ---- game pieces
    Rw = rot_y(coral_pitch(wrist))
    if hold in ("coral", "both"):
        out.append(_cyl_y_axis_x("coral", coral_held(h, wrist), Rw))
    if funnel is not None:
        p, a = funnel_pose(funnel)
        out.append(_cyl_y_axis_x("coral", p, rot_y(a)))
    if hold in ("algae", "both"):
        out.append(Sphere("algae", COL["algae"], algae_held(h, wrist), ALGAE_R))
    return scaled(out, S)


def _cyl_y_axis_x(name, centre, R):
    """A coral: cylinder whose axis is the local x of R (along the effector channel)."""
    Rx = R @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])        # local z -> R e_x
    return Cyl(name, COL["coral"], np.asarray(centre, float), CORAL_R, CORAL_LEN / 2, Rx)


def _carriage(h):
    c = np.array([0.123, 0.0, 0.311]) + AXIS * h
    out = [_box("carriage", COL["plate"], c + [0, s * 0.209, 0], [0.142, 0.005, 0.32], R_TILT) for s in (-1, 1)]
    out.append(_cyl_y("carriage_tube", COL["tube"], np.array([0.156, 0.0, 0.443]) + AXIS * h, 0.0145, 0.413))
    out.append(_cyl_y("carriage_axle", COL["plate"], np.array([0.074, 0.0, 0.46]) + AXIS * h, 0.018, 0.30))
    return out


def _effector(h, wrist):
    """Side plates (around the coral channel, an arm up to the top roller, a nose down to the front wheel), the
    channel floor and roof, the front wheel and the top roller; given relative to the pivot at wrist 0."""
    P, R = eff_pivot(h), rot_y(wrist)
    parts = []
    for y in (-0.045, 0.133):
        parts.append(_box("eff_side", COL["plate"], [0.02, y, -0.01], [0.24, 0.006, 0.17], rot_y(CORAL_PITCH0)))
        parts.append(_seg("eff_side", COL["plate"], [0.06, y, 0.08], [0.146, y, 0.2865], 0.006, 0.04))
        parts.append(_seg("eff_side", COL["plate"], [0.10, y, -0.06], [0.217, y, -0.0435], 0.006, 0.05))
    parts.append(_box("eff_plate", COL["plate"], [0.043, EFF_Y, -0.0605], [0.18, 0.17, 0.012], rot_y(CORAL_PITCH0)))
    parts.append(_box("eff_plate", COL["plate"], [0.007, EFF_Y, 0.0745], [0.18, 0.17, 0.012], rot_y(CORAL_PITCH0)))
    out = [Box(b.name, b.color, P + R @ b.center, b.half, R @ b.R) for b in parts]
    out.append(_cyl_y("eff_roller", COL["roller"], P + R @ [0.146, EFF_Y, 0.2865], 0.027, 0.19, R))
    out.append(_cyl_y("eff_wheel", COL["wheel3"], P + R @ [0.217, EFF_Y, -0.0435], 0.038, 0.19, R))
    return out


def _climber(climb):
    R = rot_y(climb)

    def at(v):
        return CLIMB_PIVOT + R @ np.asarray(v, float)
    out = [Box("climber", COL["plate"], at([-0.023, s * 0.015, 0.079]), np.array([0.109, 0.005, 0.222]) / 2, R)
           for s in (-1, 1)]
    out.append(Box("climber", COL["tube"], at([-0.065, 0.0, 0.183]), np.array([0.152, 0.025, 0.014]) / 2, R))
    out.append(Box("climber_grapple", COL["plate"], at([-0.07, 0.0, 0.192]), np.array([0.132, 0.17, 0.03]) / 2, R))
    out.append(_cyl_y("climber_bar", COL["tube"], at([-0.026, 0.0, 0.063]), 0.013, 0.225, R))
    return out


# ---------------------------------------------------------------- flat-shaded faces (tools/mech_view.py)
def sphere_faces(s: Sphere, nlat=8, nlon=12):
    """Quads of a lat-long sphere as (corners(4,3), colour, normal(3,)) list."""
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


def cyl_faces(c: Cyl, k=12):
    a, b = c.rim(k)
    axis = c.R @ np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    faces = [(a[::-1], c.color, -axis), (b, c.color, axis)]
    for i in range(k):
        j = (i + 1) % k
        quad = np.array([a[i], a[j], b[j], b[i]])
        faces.append((quad, c.color, quad.mean(0) - c.center - axis * ((quad.mean(0) - c.center) @ axis)))
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
    Rn = np.linalg.inv(b.R).T                                   # normals under a scaled basis
    nrm = lambda k: Rn @ np.array(normals[k], float) / np.linalg.norm(Rn @ np.array(normals[k], float))
    return [(c[loop], b.color, nrm(k)) for k, loop in loops.items()]


def all_faces(solids):
    faces = []
    for s in solids:
        if isinstance(s, Body):
            continue
        faces += sphere_faces(s) if isinstance(s, Sphere) else (cyl_faces(s) if isinstance(s, Cyl) else box_faces(s))
    return faces


# ---------------------------------------------------------------- renderer slots (fixed layout per robot)
SLOTS = 64                  # solids per robot in the renderer (the box model gives ~60 with pieces)
SLOT_W = 20                 # [visible, kind, centre(3), half(3), basis rows (9), colour(3)]
KIND_BODY = 3               # kind 0 box, 1 sphere, 2 cylinder, 3 + i body BODIES[i] (basis = full linear map)
_MX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # field-local -> Godot (matches main.gd gv)


def slots_for(solids):
    """Robot-local solids -> (SLOTS, SLOT_W) float32 in Godot axes (unused slots have visible = 0)."""
    out = np.zeros((SLOTS, SLOT_W), np.float32)
    assert len(solids) <= SLOTS, len(solids)
    out[:len(solids)] = solid_rows(solids)
    return out


def solid_rows(solids):
    """Robot-local solids -> (k, SLOT_W) float32 renderer rows in Godot axes. Boxes and cylinders: half sizes along
    the local axes in Godot order (x, z, y) -> a Godot CylinderMesh (axis Y) takes the solid's local z as its axis.
    Bodies: the linear map in Godot axes (the robot.glb meshes are stored in Godot axes)."""
    out = np.zeros((len(solids), SLOT_W), np.float32)
    for i, s in enumerate(solids):
        out[i, 0] = 1.0
        if isinstance(s, Body):
            out[i, 1] = KIND_BODY + BODIES.index(s.name)
            out[i, 2:5] = _MX @ s.t
            out[i, 8:17] = (_MX @ s.A @ _MX.T).ravel()
            continue
        out[i, 2:5] = _MX @ s.center
        if isinstance(s, Sphere):
            out[i, 1] = 1.0
            out[i, 5:8] = s.radius
            out[i, 8:17] = np.eye(3).ravel()
        else:
            hl = s.half if isinstance(s, Box) else np.array([s.radius, s.radius, s.half_len])
            out[i, 1] = 0.0 if isinstance(s, Box) else 2.0
            out[i, 5:8] = [hl[0], hl[2], hl[1]]
            out[i, 8:17] = (_MX @ s.R @ _MX.T).ravel()
        out[i, 17:20] = s.color
    return out
