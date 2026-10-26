"""Deliverable check: the same state gives the same object positions in Python and in the engine.

1. echo: robot transforms applied by Godot, converted back to field frame, match Python (< 1 mm, < 1e-4 rad)
2. projection: unlit markers at known 3D points; centroid in the rendered image vs Python pinhole
   projection with the robot camera rig (incl. an off-centre principal point) (< 1 px)
3. AprilTag: detect a rendered tag with OpenCV; ID and corner order must match the field layout (< 2 px)

    uv run python smoke/smoke_consistency.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ghostrender.client import RenderClient  # noqa: E402
from ghostsim import SimCfg, VecSim  # noqa: E402
from ghostsim.sensors import Cameras, quat_to_mat  # noqa: E402


def project(cam: dict, pose, pts, W, H):
    c = Cameras([cam])
    pos, rot = c.world_poses(np.asarray(pose, float))
    pc = (pts - pos[0]) @ rot[0]                                  # camera frame (x fwd, y left, z up)
    z, x, y = pc[:, 0], -pc[:, 1], -pc[:, 2]
    sx, sy = W / cam["capture"][0], H / cam["capture"][1]
    return np.stack([cam["fx"] * sx * x / z + cam["cx"] * sx, cam["fy"] * sy * y / z + cam["cy"] * sy], -1), z


def main() -> int:
    W, H = 320, 240
    sim = VecSim(SimCfg(n_envs=2, sensors=False, seed=7))
    cams = list(sim.ours["cameras"])
    off = dict(cams[0], name="offcentre", cx=390.0, cy=205.0, fx=470.0, fy=470.0)
    cams.append(off)
    rng = np.random.default_rng(1)
    ok = True
    with RenderClient() as rc:
        rc.configure(sim.game, cams, n_envs=2, width=W, height=H, max_cams=8)
        # put robots at open spots, random yaw, one climbed (z offset)
        sim.body.pose[0, 0] = [6.5, 6.5, 0.7]
        sim.body.pose[1, 2] = [11.0, 1.5, -2.0]
        sim.gs.climbed[1, 2] = True
        state = sim.render_state()
        per = 6                                    # hues 0,60,..300 deg: invariant to sRGB conversion
        markers = np.zeros((2, per, 4), np.float32)
        checks = []
        for e, r, rig in ((0, 0, 4), (1, 2, 0)):
            cam = cams[rig]
            pose = state["robots"][e, r]
            c = Cameras([cam])
            pos, rot = c.world_poses(np.array([pose[0], pose[1], pose[2]]))
            pos = pos[0] + [0, 0, pose[3]]
            for i in range(per):
                u, v, depth = rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H, rng.uniform(1.5, 3.5)
                sx, sy = W / cam["capture"][0], H / cam["capture"][1]
                x, y = (u - cam["cx"] * sx) / (cam["fx"] * sx) * depth, (v - cam["cy"] * sy) / (cam["fy"] * sy) * depth
                p = pos + rot[0] @ np.array([depth, -x, -y])
                markers[e, i] = [*p, 0.03]
            checks.append((e, r, rig, pose))
        img, h = rc.render(state, [(e, r, rig) for e, r, rig, _ in checks], check=True, markers=markers)
        # 1. echo
        echo = np.array(h["echo"])
        err_xy = err_yaw = 0.0
        for e, r, x, y, z, yaw in echo:
            p = state["robots"][int(e), int(r)]
            err_xy = max(err_xy, np.abs([x - p[0], y - p[1], z - p[3]]).max())
            err_yaw = max(err_yaw, abs((yaw - p[2] + np.pi) % (2 * np.pi) - np.pi))
        print(f"echo: max position error {err_xy * 1000:.3f} mm, yaw {err_yaw:.2e} rad")
        ok &= err_xy < 1e-3 and err_yaw < 1e-4
        # 2. markers
        worst = 0.0
        for k, (e, r, rig, pose) in enumerate(checks):
            hsv = cv2.cvtColor(img[k][..., ::-1], cv2.COLOR_BGR2HSV).astype(float)
            pose3 = np.array([pose[0], pose[1], pose[2]])
            pts = markers[e, :, :3].astype(float) - [0, 0, pose[3]]          # climbed robots are lifted
            uv, _ = project(cams[rig], pose3, pts, W, H)
            for i in range(per):
                hue = 180.0 * i / per
                dh = np.abs((hsv[..., 0] - hue + 90) % 180 - 90)
                mask = (dh < 8) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
                if mask.sum() < 3:
                    print(f"  marker {i} of cam {k} not found"); ok = False; continue
                vv, uu = np.nonzero(mask)
                c = np.array([uu.mean() + 0.5, vv.mean() + 0.5])
                worst = max(worst, np.linalg.norm(c - uv[i]))
        print(f"markers: max centroid error {worst:.2f} px (incl. off-centre principal point)")
        ok &= worst < 1.0
        # 3. AprilTag detection at 640x480 is done with a separate configure
    with RenderClient() as rc:
        W2, H2 = 640, 480
        rc.configure(sim.game, cams, n_envs=1, width=W2, height=H2, max_cams=1)
        g = sim.game
        tid = 18
        tag = g.tags[tid]
        n = np.array([np.cos(tag["yaw"]), np.sin(tag["yaw"])])
        p = tag["pos"][:2] + n * 1.2
        sim.body.pose[0, 0] = [p[0], p[1], np.arctan2(-n[1], -n[0]) + 0.15]
        state = sim.render_state()
        rc.set_appearance(0, [{"bumper_rgb": [0, 0, 1], "body_rgb": [.5, .5, .5], "team": "1", "font": 0,
                               "size": [0.9, 0.9, 1.0], "bumper_z": [0.065, 0.14], "seed": 1}] * 6,
                          {"sun_energy": 1.0, "sun_elev_deg": 60, "sun_az_deg": 30, "sun_rgb": [1, 1, 1],
                           "ambient_energy": 0.8, "ambient_rgb": [1, 1, 1], "bg_rgb": [.2, .2, .2],
                           "carpet_rgb": [.25, .25, .25], "carpet_rough": 0.9, "crowd_seed": 1, "crowd_density": 0.2})
        img, _ = rc.render(state, [(0, 0, 0)])
        gray = cv2.cvtColor(img[0], cv2.COLOR_RGB2GRAY)
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
        corners, ids, _ = det.detectMarkers(gray)
        s = g.cfg["field"]["tags"]["black_square"] * 0.0254 / 2
        Rt = quat_to_mat(tag["quat"])
        local = np.array([[0, -s, s], [0, s, s], [0, s, -s], [0, -s, -s]])     # TL, TR, BR, BL (viewer's view)
        exp, _ = project(cams[0], sim.body.pose[0, 0], tag["pos"] + local @ Rt.T, W2, H2)
        if ids is None or tid not in ids.ravel():
            print(f"apriltag: tag {tid} NOT detected (ids={None if ids is None else ids.ravel()})"); ok = False
        else:
            got = corners[list(ids.ravel()).index(tid)][0]
            err = np.linalg.norm(got - exp, axis=-1).max()
            print(f"apriltag: tag {tid} detected, max corner error {err:.2f} px (corner order checked)")
            ok &= err < 2.0
        cv2.imwrite("/tmp/ghost_consistency_tag.png", img[0][..., ::-1])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
