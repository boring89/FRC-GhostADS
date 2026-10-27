"""Keyframed animation of the cube model: one coral from the floor to L4, one algae from the floor to the processor.

    uv run python tools/mech_anim.py                  # -> runs/mech/anim.mp4 (30 fps)
    uv run python tools/mech_anim.py --gif-frames 8   # also dump every 8th frame as PNG into runs/mech/frames/

Timeline is in seconds. Each segment interpolates (h, theta, psi) with smoothstep and defines where the loose
game pieces are. Pieces attach to the intake roller, to the end-effector tip, or to the floor/branch.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ghostsim.mechanism import (ALGAE_R, COL, EE_LEN, H_MAX, INTAKE_LEN, INTAKE_PIVOT, WRIST_MIN_DEG, Box, Sphere,
                                ee_wrist, intake_tip, pose, rot_y)
from tools.mech_view import label, render

FPS = 30


@dataclass
class State:
    h: float
    theta: float
    psi: float


STOW = State(0.0, 80.0, 75.0)


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _lerp_state(a: State, b: State, u):
    s = smooth(u)
    return State(a.h + (b.h - a.h) * s, a.theta + (b.theta - a.theta) * s, a.psi + (b.psi - a.psi) * s)


def _ee_geom(st: State):
    P, _ = ee_wrist(st.h)
    t = np.radians(st.theta)
    d = np.array([np.cos(t), 0.0, np.sin(t)])
    return P + EE_LEN * d, d, rot_y(t)


def _intake_geom(st: State):
    p = np.radians(st.psi)
    return intake_tip(p), rot_y(np.pi - p)


# loose piece placement: returns (kind, centre, rotation) or None if the piece is not present
def coral_at(t, seg_u, st, seg):
    kind = seg.get("coral")
    if kind is None:
        return None
    if kind == "floor":
        return (np.array([-0.72, 0.0, 0.055]), np.eye(3))
    if kind == "intake":
        tip, R = _intake_geom(st)
        return (tip + np.array([0.0, 0.0, 0.10]), R)
    if kind == "handoff":                          # from the intake top to the end-effector head
        a, Ra = _intake_geom(seg["a"])
        a = a + np.array([0.0, 0.0, 0.10])
        b, d, Rb = _ee_geom(st)
        b = b + 0.10 * d
        s = smooth(seg_u)
        return ((1 - s) * a + s * b, Rb if s > 0.5 else Ra)
    if kind == "ee":
        tip, d, R = _ee_geom(st)
        return (tip + 0.10 * d, R)
    if kind == "drop":                              # released onto the L4 branch
        tip, d, R = _ee_geom(seg["a"])
        c0 = tip + 0.10 * d
        s = smooth(seg_u)
        return (c0 + np.array([0.20 * s, 0.0, -0.10 * s]), R)
    return None


def algae_at(t, seg_u, st, seg):
    kind = seg.get("algae")
    if kind is None:
        return None
    if kind == "floor":
        return np.array([0.85, 0.0, ALGAE_R])
    if kind == "floor_to_ee":                       # taken from the floor into the wrist
        tip, d, _ = _ee_geom(st)
        b = tip + (0.10 + ALGAE_R * 0.9) * d
        s = smooth(seg_u)
        return (1 - s) * np.array([0.85, 0.0, ALGAE_R]) + s * b
    if kind == "ee":
        tip, d, _ = _ee_geom(st)
        return tip + (0.10 + ALGAE_R * 0.9) * d
    if kind == "launch":                            # pushed out of the processor opening and gone
        tip, d, _ = _ee_geom(seg["a"])
        c0 = tip + (0.10 + ALGAE_R * 0.9) * d
        s = smooth(seg_u)
        return c0 + np.array([0.35 * s, 0.0, 0.0])
    return None


SEGMENTS = [
    dict(name="stowed", dur=1.0, a=STOW, b=STOW),
    dict(name="intake deploys to the floor", dur=1.0, a=STOW, b=State(0.0, 80.0, -10.0), coral="floor"),
    dict(name="intake picks coral from the floor", dur=0.8, a=State(0, 80, -10), b=State(0, 80, -10), coral="intake"),
    dict(name="intake stows with coral", dur=1.0, a=State(0, 80, -10), b=State(0, 80, 75), coral="intake"),
    dict(name="coral handed to end effector", dur=1.0, a=State(0, 80, 75), b=State(0, 0, 75), coral="handoff"),
    dict(name="elevator to L4 (1.83 m)", dur=2.0, a=State(0, 0, 75), b=State(H_MAX, 0, 75), coral="ee"),
    dict(name="score coral on L4", dur=0.8, a=State(H_MAX, 0, 75), b=State(H_MAX, 0, 75), coral="drop"),
    dict(name="elevator down, wrist to floor", dur=1.6, a=State(H_MAX, 0, 75), b=State(0, WRIST_MIN_DEG, 75)),
    dict(name="algae picked from the floor", dur=1.0, a=State(0, WRIST_MIN_DEG, 75), b=State(0, WRIST_MIN_DEG, 75), algae="floor_to_ee"),
    dict(name="algae into processor", dur=0.8, a=State(0, WRIST_MIN_DEG, 75), b=State(0, WRIST_MIN_DEG, 75), algae="ee"),
    dict(name="processor release", dur=0.6, a=State(0, WRIST_MIN_DEG, 75), b=State(0, WRIST_MIN_DEG, 75), algae="launch"),
    dict(name="return to stow", dur=1.6, a=State(0, WRIST_MIN_DEG, 75), b=STOW),
]


def _segments_with_starts():
    t0, out = 0.0, []
    for seg in SEGMENTS:
        out.append((t0, seg))
        t0 += seg["dur"]
    return out, t0


def state_at(t):
    """(State, seg, u, coral_centre_rot|None, algae_centre|None) at time t."""
    segs, total = _segments_with_starts()
    t = min(max(t, 0.0), total - 1e-9)
    for t0, seg in segs:
        if t <= t0 + seg["dur"] or seg is segs[-1][1]:
            u = (t - t0) / seg["dur"]
            st = _lerp_state(seg["a"], seg["b"], u)
            return st, seg, u, coral_at(t, u, st, seg), algae_at(t, u, st, seg)
    raise AssertionError


def solids_at(t):
    st, seg, u, cr, al = state_at(t)
    solids = pose(h=st.h, theta=st.theta, psi=st.psi, hold=None)
    if cr is not None:
        c, R = cr
        solids.append(Box("coral", COL["coral"], c, np.array([0.15, 0.055, 0.055]), R))
    if al is not None:
        solids.append(Sphere("algae", COL["algae"], al, ALGAE_R))
    return solids, st, seg


def timeline():
    _, total = _segments_with_starts()
    return total


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mech/anim.mp4")
    ap.add_argument("--frames-png", type=int, default=0, help="if >0, also save every Nth frame as PNG")
    a = ap.parse_args(argv)
    total = timeline()
    n = int(round(total * FPS))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (900, 700))
    if not wr.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4v")
    png_dir = out.parent / "frames"
    if a.frames_png:
        png_dir.mkdir(exist_ok=True)
    for i in range(n):
        t = i / FPS
        solids, st, seg = solids_at(t)
        img = label(render(solids), f"{t:5.2f}s  {seg['name']}")
        wr.write(img)
        if a.frames_png and i % a.frames_png == 0:
            cv2.imwrite(str(png_dir / f"f{i:04d}.png"), img)
    wr.release()
    print("wrote", out, f"({n} frames, {total:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
