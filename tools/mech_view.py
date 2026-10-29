"""Render the box model of the robot to PNG (numpy + OpenCV painter's algorithm, flat shading). The live view draws
the CAD meshes (game/2025/visual/robot.glb) with the same joints.

    uv run python tools/mech_view.py                 # standard poses -> runs/mech/*.png and runs/mech/sheet.png
    uv run python tools/mech_view.py --pose h=1.4,wrist=-35,hold=coral --out /tmp/x.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ghostsim.mechanism import CLIMB, COL, H_MAX, WRIST, algae_target, all_faces, pose
from ghostsim.mechanism_state import REEF_ALGAE_Z, SCORE

LIGHT = np.array([-0.45, 0.6, 1.0])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

POSES = {
    "stowed": dict(),
    "station_hopper": dict(funnel=0.45),
    "L2_coral": dict(h=SCORE["score_L2"][0], wrist=SCORE["score_L2"][1], hold="coral"),
    "L4_coral": dict(h=SCORE["score_L4"][0], wrist=SCORE["score_L4"][1], hold="coral"),
    "algae_floor": dict(wrist=WRIST["floor"], hold="algae"),
    "algae_reef_high": dict(h=algae_target(REEF_ALGAE_Z["high"]), wrist=WRIST["reach"], hold="algae"),
    "net_fling": dict(h=H_MAX, wrist=WRIST["net"], hold="algae"),
    "red_climb": dict(climb=CLIMB["reach"], alliance=1),
    "full_up": dict(h=H_MAX, hold="both"),
}


def _camera(eye, target, up=(0, 0, 1.0), f=1.0):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return eye, fwd, right, upv


def render(solids, size=(900, 700), eye=(2.2, -2.4, 2.0), target=(0.0, 0.0, 0.8), focal=1.5):
    import cv2
    W, H = size
    eye, fwd, right, upv = _camera(eye, target)
    faces = all_faces(solids)
    # floor
    fl = np.array([[-0.8, -0.8, 0], [0.8, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0]], float)
    faces.append((fl, COL["floor"], np.array([0, 0, 1.0])))

    def proj(p):
        d = p - eye
        xc, yc, zc = d @ right, d @ upv, d @ fwd
        return np.stack([W / 2 + focal * W / 2 * xc / zc, H / 2 - focal * W / 2 * yc / zc], -1), zc

    img = np.zeros((H, W, 3), np.uint8)
    yy = np.linspace(0, 1, H)[:, None]
    img[:] = (np.stack([0.93 - 0.08 * yy, 0.94 - 0.08 * yy, 0.96 - 0.06 * yy], -1) * 255).astype(np.uint8)[:, :1].repeat(W, 1)
    polys = []
    for quad, color, normal in faces:
        centre = quad.mean(0)
        if np.dot(normal, eye - centre) <= 0:                  # back face
            continue
        pts, zc = proj(quad)
        if np.any(zc <= 0.05):
            continue
        shade = 0.42 + 0.58 * max(0.0, float(np.dot(normal, LIGHT)))
        polys.append((float(zc.mean()), pts, np.clip(np.asarray(color) * shade, 0, 1)))
    polys.sort(key=lambda t: -t[0])                              # far first
    for _, pts, col in polys:
        bgr = tuple(int(255 * c) for c in col[::-1])
        cv2.fillPoly(img, [pts.astype(np.int32)], bgr, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts.astype(np.int32)], True, (40, 40, 40), 1, lineType=cv2.LINE_AA)
    return img


def label(img, text):
    import cv2
    cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def parse_pose(s):
    kw = {}
    for part in s.split(","):
        k, v = part.split("=")
        kw[k] = (None if v == "none" else v) if k == "hold" else (int(v) if k == "alliance" else float(v))
    return kw


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mech")
    ap.add_argument("--pose", help="one custom pose, e.g. h=0.8,wrist=-35,climb=0,hold=coral")
    a = ap.parse_args(argv)
    if a.pose:
        kw = parse_pose(a.pose)
        cv2.imwrite(a.out, label(render(pose(**kw)), a.pose))
        print("wrote", a.out)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for name, kw in POSES.items():
        img = label(render(pose(**kw)), name)
        cv2.imwrite(str(out / f"{name}.png"), img)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    rows = [r if r.shape[1] == rows[0].shape[1] else np.hstack([r, np.zeros((r.shape[0], rows[0].shape[1] - r.shape[1], 3), np.uint8)]) for r in rows]
    cv2.imwrite(str(out / "sheet.png"), np.vstack(rows))
    print("wrote", out / "sheet.png", "and", len(tiles), "single poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
