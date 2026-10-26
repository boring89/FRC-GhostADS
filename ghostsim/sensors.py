"""Robot-side sensors and pose fusion (must match the Systemcore implementation for the KF).

- IMU: yaw = true + bias(random walk) + noise; rate = omega + noise
- module encoders: azimuth + noise; wheel speed from physics (wrong while slipping) + noise
- odometry: SwerveDriveOdometry (least-squares module kinematics, heading change from gyro, Pose2d.exp)
- AprilTags ("full" robots): per camera frustum / distance / incidence / pixel size / occlusion by robots
  and tall obstacles; pose measurement sigma = s0 * d^2 / sqrt(n); single-tag ambiguity outliers; latency.
  Other robots get "synthetic" measurements with the same noise model (cheap).
- diagonal Kalman filter on (x, y, theta) with chi^2 gating and latency compensation.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from .robots import R


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def rot_rpy(roll, pitch, yaw):
    """WPILib Rotation3d(roll, pitch, yaw) = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_exp(dx, dy, dth):
    """Pose2d.exp of a twist -> local translation (…,2)."""
    small = np.abs(dth) < 1e-9
    s = np.where(small, 1 - dth ** 2 / 6, np.sin(dth) / np.where(small, 1, dth))
    c = np.where(small, dth / 2, (1 - np.cos(dth)) / np.where(small, 1, dth))
    return np.stack([dx * s - dy * c, dx * c + dy * s], -1)


class Cameras:
    def __init__(self, cams: list[dict]):
        self.names = [c["name"] for c in cams]
        self.t = np.array([c["robot_to_camera"][:3] for c in cams])                     # (C,3)
        self.R = np.stack([rot_rpy(*np.radians(c["robot_to_camera"][3:])) for c in cams])  # (C,3,3) robot<-cam
        self.K = np.array([[c["fx"], c["fy"], c["cx"], c["cy"]] for c in cams])
        self.size = np.array([c["capture"] for c in cams], float)

    def world_poses(self, pose):
        """pose (...,3) -> camera position (...,C,3), rotation field<-cam (...,C,3,3)."""
        Rz = rotz(pose[..., 2])[..., None, :, :]                                        # (...,1,3,3)
        pos = np.concatenate([pose[..., :2], np.zeros(pose.shape[:-1] + (1,))], -1)[..., None, :] \
            + (Rz @ self.t[..., None])[..., 0]
        return pos, Rz @ self.R


class Sensors:
    def __init__(self, game, sp: dict, cams: list[dict], n: int, dt: float, rng):
        self.sp, self.v, self.kf, self.dt, self.rng = sp, sp["vision"], sp["kf"], dt, rng
        ids = sorted(game.tags)
        self.tag_id = np.array(ids)
        self.tag_pos = np.array([game.tags[i]["pos"] for i in ids])
        self.tag_nrm = np.array([quat_to_mat(game.tags[i]["quat"])[:, 0] for i in ids])
        self.tag_size = game.cfg["field"]["tags"]["black_square"] * geo.IN
        self.occ = game.polys.subset(np.flatnonzero(game.polys.height > 0))
        self.cams = Cameras(cams)
        self.every = max(1, int(round(1.0 / (self.v["rate_hz"] * dt))))
        self.lag = max(1, int(round(self.v["latency_s"] / dt)))
        z = np.zeros
        self.bias, self.yaw, self.rate = z((n, R)), z((n, R)), z((n, R))
        self.az, self.wheel, self.odom_vel = z((n, R, 4)), z((n, R, 4)), z((n, R, 3))
        self.est, self.var = z((n, R, 3)), z((n, R, 3))
        self.hist_true, self.hist_est = z((n, R, self.lag + 1, 3)), z((n, R, self.lag + 1, 3))
        self.prev_yaw, self.pinv = z((n, R)), z((n, R, 3, 8))
        self.tags_seen, self.k = z((n, R)), 0

    def reset(self, idx, body, prm):
        s0 = np.array(self.kf["init_std"])
        k = len(idx)
        self.bias[idx] = 0.0
        self.yaw[idx] = body.pose[idx, :, 2]
        self.prev_yaw[idx] = self.yaw[idx]
        self.est[idx] = body.pose[idx] + self.rng.normal(0, 1, (k, R, 3)) * s0
        self.var[idx] = s0 ** 2
        self.hist_true[idx] = body.pose[idx, :, None]
        self.hist_est[idx] = self.est[idx, :, None]
        off = prm.offsets[idx]                                                            # (k,R,4,2)
        A = np.zeros((k, R, 8, 3))
        A[..., 0::2, 0], A[..., 1::2, 1] = 1, 1
        A[..., 0::2, 2], A[..., 1::2, 2] = -off[..., 1], off[..., 0]
        self.pinv[idx] = np.linalg.pinv(A)
        self.tags_seen[idx] = 0

    # ------------------------------------------------------------------ per substep
    def update(self, body, full_mask, live):
        sp, dt, rng = self.sp, self.dt, self.rng
        n = len(live)
        self.bias += rng.normal(0, sp["gyro_bias_rw_rad_per_sqrt_s"] * np.sqrt(dt), self.bias.shape) * live[:, None]
        self.yaw = geo.wrap(body.pose[..., 2] + self.bias + rng.normal(0, sp["gyro_noise_rad"], self.bias.shape))
        self.rate = body.vel[..., 2] + rng.normal(0, sp["gyro_rate_noise_radps"], self.rate.shape)
        self.az = geo.wrap(body.az + rng.normal(0, sp["az_noise_rad"], self.az.shape))
        self.wheel = body.wheel + rng.normal(0, sp["wheel_speed_noise_mps"], self.wheel.shape)
        mv = np.stack([self.wheel * np.cos(self.az), self.wheel * np.sin(self.az)], -1).reshape(n, R, 8)
        self.odom_vel = (self.pinv @ mv[..., None])[..., 0]                               # robot-frame chassis speeds
        # odometry predict
        dth = geo.wrap(self.yaw - self.prev_yaw)
        self.prev_yaw = self.yaw
        loc = pose_exp(self.odom_vel[..., 0] * dt, self.odom_vel[..., 1] * dt, dth)
        c, s = np.cos(self.est[..., 2]), np.sin(self.est[..., 2])
        self.est[..., 0] += loc[..., 0] * c - loc[..., 1] * s
        self.est[..., 1] += loc[..., 0] * s + loc[..., 1] * c
        self.est[..., 2] = geo.wrap(self.est[..., 2] + dth)
        kf = self.kf
        dist = np.linalg.norm(loc, axis=-1)
        self.var[..., :2] += (kf["odom_std_per_m"] ** 2 * dist + kf["odom_std_per_s"] ** 2 * dt)[..., None]
        self.var[..., 2] += kf["yaw_std_per_rad"] ** 2 * np.abs(dth) + kf["yaw_std_per_s"] ** 2 * dt
        self.hist_true = np.roll(self.hist_true, 1, 2); self.hist_true[:, :, 0] = body.pose
        self.hist_est = np.roll(self.hist_est, 1, 2); self.hist_est[:, :, 0] = self.est
        self.k += 1
        if self.k % self.every == 0:
            self._vision(body, full_mask, live)

    # ------------------------------------------------------------------ vision
    def _vision(self, body, full_mask, live):
        v, rng = self.v, self.rng
        n = len(live)
        then = self.hist_true[:, :, self.lag]
        est_then = self.hist_est[:, :, self.lag]
        nvis = np.zeros((n, R, len(self.cams.names)))
        dmean = np.full(nvis.shape, 3.0)
        e_idx, r_idx = np.nonzero(full_mask & live[:, None])
        if len(e_idx):
            cnt, dm = self._visible(body, e_idx, r_idx)
            nvis[e_idx, r_idx], dmean[e_idx, r_idx] = cnt, dm
        syn = ~full_mask & live[:, None]
        nvis[syn] = (rng.random((syn.sum(), nvis.shape[-1])) < 0.35) * 2.0
        self.tags_seen = nvis.sum(-1)
        for ci in range(nvis.shape[-1]):
            k = nvis[..., ci]
            has = k > 0
            if not has.any():
                continue
            dd = np.maximum(dmean[..., ci], 1.0) ** 2 / np.sqrt(np.maximum(k, 1))
            sig = np.stack([v["sigma_xy_m"] * dd, v["sigma_xy_m"] * dd, v["sigma_theta_rad"] * dd], -1)
            z = then + rng.normal(0, 1, then.shape) * sig
            amb = has & (k == 1) & (rng.random(k.shape) < v["ambiguity_p"])
            z = np.where(amb[..., None], z + rng.normal(0, 1, z.shape) * [1.0, 1.0, 0.5], z)
            inn = z - est_then
            inn[..., 2] = geo.wrap(inn[..., 2])
            S = self.var + sig ** 2
            ok = has & ((inn ** 2 / S).sum(-1) < self.kf["gate_chi2"])
            K = self.var / S * ok[..., None]
            self.est += K * inn
            self.est[..., 2] = geo.wrap(self.est[..., 2])
            self.var *= 1 - K

    def _visible(self, body, e, r):
        """Count visible tags per camera for selected (env, robot) pairs -> (M,C), mean distance (M,C)."""
        v, cams = self.v, self.cams
        pose = body.pose[e, r]                                                            # (M,3)
        cpos, crot = cams.world_poses(pose)                                               # (M,C,3), (M,C,3,3)
        d = self.tag_pos[None, None] - cpos[:, :, None]                                   # (M,C,T,3)
        pc = np.einsum("mcji,mctj->mcti", crot, d)                                        # camera frame (x fwd,y left,z up)
        zc, xc, yc = pc[..., 0], -pc[..., 1], -pc[..., 2]
        fx, fy, cx, cy = (cams.K[:, i][None, :, None] for i in range(4))
        W, H = cams.size[:, 0][None, :, None], cams.size[:, 1][None, :, None]
        zz = np.maximum(zc, 1e-6)
        u, w_ = fx * xc / zz + cx, fy * yc / zz + cy
        dist = np.linalg.norm(d, axis=-1)
        inc = (self.tag_nrm[None, None] * -d).sum(-1) / np.maximum(dist, 1e-9)
        m = v["edge_margin_px"]
        vis = (zc > 0.1) & (u > m) & (u < W - m) & (w_ > m) & (w_ < H - m) & (dist < v["max_dist_m"]) \
            & (inc > np.cos(np.radians(v["max_incidence_deg"]))) & (fx * self.tag_size / zz > v["min_px"])
        # occlusion (2D segment tests with height check), only for candidates
        mi, ci, ti = np.nonzero(vis)
        if len(mi):
            a2 = cpos[mi, ci, :2]
            b2 = self.tag_pos[ti, :2]
            dd = dist[mi, ci, ti][:, None]
            b2 = b2 + (a2 - b2) * (0.03 / np.maximum(dd, 0.03))
            low = np.minimum(cpos[mi, ci, 2], self.tag_pos[ti, 2])
            blocked = (geo.segment_poly_hit(a2, b2, self.occ) & (self.occ.height > low[:, None])).any(-1)
            ee, rr = e[mi], r[mi]
            others = np.array([[q for q in range(R) if q != x] for x in range(R)])[rr]       # (K,R-1)
            th = body.pose[ee[:, None], others, 2]
            hit = geo.segment_box_hit(a2[:, None], b2[:, None], body.pose[ee[:, None], others, :2],
                                      self.prm_half[ee[:, None], others], np.cos(th), np.sin(th))
            blocked |= (hit & (self.prm_height[ee[:, None], others] > low[:, None])).any(-1)
            vis[mi[blocked], ci[blocked], ti[blocked]] = False
        cnt = vis.sum(-1)
        dm = (dist * vis).sum(-1) / np.maximum(cnt, 1)
        return cnt, dm

    def bind(self, prm):
        self.prm_half, self.prm_height = prm.half, prm.height

    def obs(self, holding, match):
        f32 = lambda x: np.asarray(x, np.float32)
        return {
            "gyro": f32(np.stack([self.yaw, self.rate], -1)),
            "module_angle": f32(np.concatenate([np.sin(self.az), np.cos(self.az)], -1)),
            "wheel_speed": f32(self.wheel),
            "odom_vel": f32(self.odom_vel),
            "holding": f32(holding),
            "pose": f32(self.est),
            "pose_std": f32(np.sqrt(self.var)),
            "tags_seen": f32(self.tags_seen[..., None]),
            "match": f32(match),
        }
