"""Kinematic model of our robot built from boxes, cylinders and spheres (visualisation and geometry checks; not used by
the physics). Layout after FRC 6328's 2025 robot (team reference pictures), sized by robots/ours.json:

  - swerve drivebase, alliance-coloured bumpers
  - cascade elevator in the middle: fixed outer stage (blue), stage 2 (orange, h/3), stage 3 (grey, 2h/3), carriage (h)
  - coral dispenser on the carriage, pitched by a wrist; it scores over the FRONT bumper
  - funnel at the BACK: the robot backs into the coral station, coral slides down the chute into the dispenser (h = 0)
  - algae arm at the FRONT: two bars with rollers that pick algae off the floor and cradle it in front of the robot
  - climber arm on the left side (deep cage)

Frame: x forward (scoring side), y left, z up, origin on the floor at the robot centre. Metres, angles in degrees.
Joints:  h      elevator travel 0..H_MAX
         wrist  dispenser pitch, 0 = level, - = nose down
         arm    algae arm angle from +x, + up (60 = stowed, leaning forward clear of the dispenser's path)
         climb  climber arm angle from +x, + up (95 = stowed)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IN = 0.0254
FRAME = 29.5 * IN                 # frame outer size without bumpers (robots/ours.json frame_in)
BUMPER_T = 3.25 * IN              # bumper thickness (ours.json bumper_t_in)
BUMPER_Z = (0.065, 0.14)          # ours.json bumper_z_m
PERIMETER_HALF = FRAME / 2
H_MAX = 1.50                      # elevator travel (3 stages, starting height under R104's 42 in)
ALGAE_R = 16.25 * IN / 2
CORAL_R, CORAL_LEN = 4.5 * IN / 2, 11.875 * IN

CARRIAGE_Z0, CARRIAGE_LEN = 0.30, 0.20     # carriage bottom at h = 0, its height
PIVOT_X, PIVOT_DZ = 0.13, 0.10    # dispenser wrist pivot: x, height above the carriage bottom
DISP_LEN = 0.26                   # dispenser plate length (pivot -> nose)
CORAL_IN_DISP = np.array([0.14, 0.0, 0.065])     # held coral centre in the dispenser frame (on the plate)
ARM_LEN = 0.42                    # algae arm pivot -> tip roller
ARM_PIVOT_DZ = 0.32               # algae arm pivot height
BALL_IN_ARM = np.array([0.27, 0.0, 0.17])        # cradled algae centre in the arm frame
CLIMB_LEN = 0.55
FUNNEL_TOP_Z, FUNNEL_BOT = 0.92, np.array([-0.02, 0.46])   # chute top just under the station chute (37.5 in), bottom end (x, z)
FUNNEL_WALL = 0.16
# elevator stages: (name, colour key, x, upright y, bottom z at h = 0, top z at h = 0, share of h)
STAGES = (("outer", "outer", 0.02, 0.20, 0.125, 1.06, 0.0), ("stage2", "stage2", 0.055, 0.17, 0.14, 1.03, 1 / 3),
          ("stage3", "stage3", 0.085, 0.14, 0.155, 1.00, 2 / 3))

STOW = {"h": 0.0, "wrist": 0.0, "arm": 60.0, "climb": 95.0}
ARM = {"stow": 60.0, "hold": 10.0, "floor": -38.0, "processor": -15.0, "net": 80.0, "catch": 35.0}

BLUE, RED = (0.10, 0.25, 0.80), (0.80, 0.10, 0.10)
COL = {
    "rail": (0.16, 0.30, 0.72), "pan": (0.30, 0.30, 0.32), "module": (0.14, 0.14, 0.15), "wheel": (0.05, 0.05, 0.05),
    "battery": (0.08, 0.08, 0.09), "outer": (0.18, 0.36, 0.86), "stage2": (0.95, 0.50, 0.12), "stage3": (0.46, 0.46, 0.49),
    "carriage": (0.22, 0.22, 0.24), "inner": (0.46, 0.46, 0.49),
    "disp": (0.24, 0.24, 0.26), "roller": (0.78, 0.78, 0.76), "funnel": (0.55, 0.56, 0.60), "arm": (0.52, 0.52, 0.54),
    "climb": (0.95, 0.50, 0.12), "coral": (0.97, 0.97, 0.97), "algae": (0.25, 0.80, 0.60), "floor": (0.62, 0.62, 0.60),
}


def rot_y(deg):
    """Pitch about y (degrees): R @ e_x = (cos, 0, sin), + up."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


AXIS_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # cylinder local z -> world y


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


def _box(name, color, center, size, R=None):
    return Box(name, color, np.asarray(center, float), np.asarray(size, float) / 2, np.eye(3) if R is None else R)


def _cyl_y(name, color, center, radius, length, R=None):
    """Cylinder along the (possibly pitched) y axis."""
    Rw = AXIS_Y if R is None else R @ AXIS_Y
    return Cyl(name, color, np.asarray(center, float), radius, length / 2, Rw)


# ---------------------------------------------------------------- kinematics
def carriage_z(h):
    return CARRIAGE_Z0 + float(np.clip(h, 0.0, H_MAX))


def wrist_pivot(h):
    return np.array([PIVOT_X, 0.0, carriage_z(h) + PIVOT_DZ])


def coral_held(h, wrist):
    """Centre of a coral in the dispenser."""
    return wrist_pivot(h) + rot_y(wrist) @ CORAL_IN_DISP


def coral_target(z_coral, wrist):
    """Elevator travel that puts the held coral centre at height z_coral with the given wrist angle."""
    return z_coral - (CARRIAGE_Z0 + PIVOT_DZ + (rot_y(wrist) @ CORAL_IN_DISP)[2])


def arm_pivot(half=PERIMETER_HALF):
    return np.array([half - 0.075, 0.0, ARM_PIVOT_DZ])


def arm_tip(arm, half=PERIMETER_HALF):
    return arm_pivot(half) + rot_y(arm) @ np.array([ARM_LEN, 0.0, 0.0])


def ball_in_arm(arm, half=PERIMETER_HALF):
    return arm_pivot(half) + rot_y(arm) @ BALL_IN_ARM


def funnel_top_x(half=PERIMETER_HALF):
    return -half - 0.045


def funnel_point(s, half=PERIMETER_HALF):
    """Coral centre on the chute: s = 0 top (entering from the station) .. 1 bottom (into the dispenser)."""
    top = np.array([funnel_top_x(half), FUNNEL_TOP_Z])
    bot = FUNNEL_BOT
    p = top + (bot - top) * float(np.clip(s, 0.0, 1.0))
    d = (bot - top) / np.linalg.norm(bot - top)
    n = np.array([-d[1], d[0]])                              # chute normal (up-forward)
    q = p + n * (CORAL_R + 0.01)
    return np.array([q[0], 0.0, q[1]])


def chute_angle(half=PERIMETER_HALF):
    top = np.array([funnel_top_x(half), FUNNEL_TOP_Z])
    d = FUNNEL_BOT - top
    return float(np.degrees(np.arctan2(d[1], d[0])))


# ---------------------------------------------------------------- solids
def pose(h=0.0, wrist=0.0, arm=60.0, climb=95.0, hold=None, alliance=0, frame=FRAME, funnel=None):
    """All solids for one configuration. hold: None | 'coral' | 'algae' | 'both'. funnel: None or chute position
    0..1 of a coral sliding down the funnel. frame: frame outer size (m) of this robot."""
    return pose_static(alliance, frame) + pose_moving(h, wrist, arm, climb, hold, frame, funnel)


def pose_static(alliance=0, frame=FRAME):
    """Solids that never move relative to the robot (drivebase, bumpers, outer stage, funnel): cacheable."""
    half = frame / 2
    out = []
    # ---- drivebase
    out.append(_box("pan", COL["pan"], [0, 0, 0.05], [frame - 0.05, frame - 0.05, 0.006]))
    for s in (-1, 1):
        out.append(_box("rail", COL["rail"], [0, s * (half - 0.025), 0.10], [frame, 0.05, 0.05]))
        out.append(_box("rail", COL["rail"], [s * (half - 0.025), 0, 0.10], [0.05, frame - 0.10, 0.05]))
    m = half - 0.085
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(_box("module", COL["module"], [sx * m, sy * m, 0.135], [0.12, 0.12, 0.07]))
            out.append(_cyl_y("wheel", COL["wheel"], [sx * m, sy * m, 0.05], 0.05, 0.04))
    out.append(_box("battery", COL["battery"], [-0.12, -0.22, 0.13], [0.18, 0.08, 0.17]))
    # ---- bumpers (alliance colour)
    bc = BLUE if alliance == 0 else RED
    ho = half + BUMPER_T
    bz, bh = (BUMPER_Z[0] + BUMPER_Z[1]) / 2, BUMPER_Z[1] - BUMPER_Z[0]
    for s in (-1, 1):
        out.append(_box("bumper", bc, [s * (ho - BUMPER_T / 2), 0, bz], [BUMPER_T, 2 * ho, bh]))
        out.append(_box("bumper", bc, [0, s * (ho - BUMPER_T / 2), bz], [2 * ho - 2 * BUMPER_T, BUMPER_T, bh]))
    # ---- elevator outer stage (fixed)
    out += _stage(STAGES[0], 0.0)
    # ---- funnel at the back (chute + side walls + supports)
    Rf = rot_y(chute_angle(half))
    top = np.array([funnel_top_x(half), 0.0, FUNNEL_TOP_Z])
    bot = np.array([FUNNEL_BOT[0], 0.0, FUNNEL_BOT[1]])
    out.append(_box("funnel", COL["funnel"], (top + bot) / 2, [float(np.linalg.norm(bot - top)), 0.38, 0.01], Rf))
    wz0, wz1 = bot[2] - 0.02, FUNNEL_TOP_Z + FUNNEL_WALL / 2               # vertical side walls of the hopper
    for s in (-1, 1):
        out.append(_box("funnel_wall", COL["funnel"], [(top[0] + bot[0]) / 2, s * 0.195, (wz0 + wz1) / 2],
                        [bot[0] - top[0], 0.01, wz1 - wz0]))
    for s in (-1, 1):
        out.append(_box("funnel_post", COL["inner"], [top[0] + 0.03, s * 0.17, (0.125 + wz0) / 2], [0.025, 0.025, wz0 - 0.125]))
    return out


def _stage(st, h):
    name, ck, x, y, z0, z1, k = st
    e, w = k * h, (0.05 if name == "outer" else 0.035)
    out = [_box(name, COL[ck], [x, s * y, (z0 + z1) / 2 + e], [w, w, z1 - z0]) for s in (-1, 1)]
    out.append(_box(name, COL[ck], [x, 0, z1 - w / 2 + e], [w, 2 * y + w, w]))
    return out


def pose_moving(h=0.0, wrist=0.0, arm=60.0, climb=95.0, hold=None, frame=FRAME, funnel=None):
    """Solids that follow the joints, and the game pieces."""
    h = float(np.clip(h, 0.0, H_MAX))
    half = frame / 2
    out = []
    # ---- elevator: inner stages (cascade) and the carriage
    for st in STAGES[1:]:
        out += _stage(st, h)
    zc = carriage_z(h)
    for s in (-1, 1):
        out.append(_box("carriage", COL["carriage"], [0.11, s * 0.12, zc + CARRIAGE_LEN / 2], [0.02, 0.06, CARRIAGE_LEN]))
    # ---- dispenser on the carriage (wrist)
    P = wrist_pivot(h)
    Rw = rot_y(wrist)
    out.append(_box("disp", COL["disp"], P + Rw @ [DISP_LEN / 2, 0, 0.0], [DISP_LEN, 0.36, 0.02], Rw))
    for s in (-1, 1):
        out.append(_box("disp_side", COL["disp"], P + Rw @ [DISP_LEN / 2, s * 0.17, 0.06], [DISP_LEN, 0.01, 0.12], Rw))
    out.append(_cyl_y("disp_roller", COL["roller"], P + Rw @ [DISP_LEN - 0.02, 0, 0.13], 0.025, 0.33, Rw))
    # ---- algae arm at the front: two bars, two rollers
    Ra = rot_y(arm)
    ap = arm_pivot(half)
    for s in (-1, 1):
        out.append(_box("arm", COL["arm"], ap + Ra @ [ARM_LEN / 2, 0, 0] + [0, s * 0.29, 0], [ARM_LEN, 0.03, 0.04], Ra))
    out.append(_cyl_y("arm_roller", COL["roller"], ap + Ra @ [ARM_LEN, 0, 0], 0.035, 0.61, Ra))
    out.append(_cyl_y("arm_roller", COL["arm"], ap + Ra @ [0.28, 0, 0], 0.025, 0.61, Ra))
    # ---- climber (left side)
    Rc = rot_y(climb)
    cp = np.array([-half + 0.07, half - 0.04, 0.22])
    out.append(_box("climber", COL["climb"], cp + Rc @ [CLIMB_LEN / 2, 0, 0], [CLIMB_LEN, 0.03, 0.06], Rc))
    out.append(_cyl_y("climber_roller", COL["roller"], cp + Rc @ [CLIMB_LEN, 0, 0] + [0, -0.10, 0], 0.03, 0.20, Rc))
    # ---- game pieces
    if hold in ("coral", "both"):
        out.append(_cyl_y("coral", COL["coral"], coral_held(h, wrist), CORAL_R, CORAL_LEN, Rw))
    if funnel is not None:
        out.append(_cyl_y("coral", COL["coral"], funnel_point(funnel, half), CORAL_R, CORAL_LEN, rot_y(chute_angle(half))))
    if hold in ("algae", "both"):
        out.append(Sphere("algae", COL["algae"], ball_in_arm(arm, half), ALGAE_R))
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
    return [(c[loop], b.color, b.R @ np.array(normals[k], float)) for k, loop in loops.items()]


def all_faces(solids):
    faces = []
    for s in solids:
        faces += sphere_faces(s) if isinstance(s, Sphere) else (cyl_faces(s) if isinstance(s, Cyl) else box_faces(s))
    return faces


# ---------------------------------------------------------------- renderer slots (fixed layout per robot)
SLOTS = 48                  # solids per robot in the renderer (pose() gives ~45)
SLOT_W = 20                 # [visible, kind(0 box, 1 sphere, 2 cylinder), centre(3), half(3), basis rows (9), colour(3)]
_MX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # field-local -> Godot (matches main.gd gv)


def slots_for(solids):
    """Robot-local solids -> (SLOTS, SLOT_W) float32 in Godot axes (unused slots have visible = 0)."""
    out = np.zeros((SLOTS, SLOT_W), np.float32)
    assert len(solids) <= SLOTS, len(solids)
    out[:len(solids)] = solid_rows(solids)
    return out


def solid_rows(solids):
    """Robot-local solids -> (k, SLOT_W) float32 renderer rows in Godot axes. Boxes and cylinders: half sizes along
    the local axes in Godot order (x, z, y) -> a Godot CylinderMesh (axis Y) takes the solid's local z as its axis."""
    out = np.zeros((len(solids), SLOT_W), np.float32)
    for i, s in enumerate(solids):
        out[i, 0] = 1.0
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
