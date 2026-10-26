"""Gym-like multi-agent wrapper.

obs["privileged"]: full true state (teacher)
obs["sensor"]:     per agent robot, exactly what the real robot has (student): IMU, encoders, odometry,
                   game-piece sensor, fused pose + std, match info, and (if rendering) 4 camera images
Finished envs are reset automatically; their final scores are in info["final_score"].
"""
from __future__ import annotations

import numpy as np

from .robots import ALLIANCE
from .sim import SimCfg, VecSim


class GhostEnv:
    def __init__(self, cfg: SimCfg = SimCfg(), agents=(0,), render=False, width=160, height=120,
                 method="forward_plus", seed=0):
        self.sim = VecSim(cfg)
        self.agents = tuple(agents)
        self.rng = np.random.default_rng(seed + 12345)
        self.rc = None
        self.W, self.H = width, height
        n = cfg.n_envs
        self.cams = self.sim.ours["cameras"]
        if render:
            from ghostrender.client import RenderClient
            self.rc = RenderClient(method=method)
            self.rc.configure(self.sim.game, self.cams, n_envs=n, width=width, height=height,
                              max_cams=n * len(self.agents) * len(self.cams))
            self.cam_list = [(e, r, c) for e in range(n) for r in self.agents for c in range(len(self.cams))]
            self.post = None
        self._episode(np.ones(n, bool))

    def _episode(self, mask):
        if self.rc is None:
            return
        from ghostrender import dr
        s = self.sim
        k = len(self.agents) * len(self.cams)
        if self.post is None:
            self.post = {key: np.zeros((s.cfg.n_envs, k)) for key in ("gain", "noise", "t_exp")}
        for e in np.flatnonzero(mask):
            self.rc.set_appearance(int(e), dr.sample_appearance(self.rng, s.prm.half[e], s.prm.height[e],
                                                                s.ours["bumper_z_m"]), dr.sample_dr(self.rng))
            p = dr.sample_camera_post(self.rng, k)
            for key in p:
                self.post[key][e] = p[key]

    def _images(self):
        from ghostrender import dr
        s = self.sim
        omega = s.body.vel[..., 2]
        post = []
        for e, r, c in self.cam_list:
            j = self.agents.index(r) * len(self.cams) + c
            fx = self.cams[c]["fx"] * self.W / self.cams[c]["capture"][0]
            post.append((self.post["gain"][e, j], self.post["noise"][e, j],
                         float(dr.blur_px(omega[e, r], fx, self.post["t_exp"][e, j])), 0.0, float(self.rng.random() * 100)))
        img, _ = self.rc.render(s.render_state(), self.cam_list, post)
        return img.reshape(s.cfg.n_envs, len(self.agents), len(self.cams), self.H, self.W, 3)

    def obs(self):
        o = {"privileged": self.sim.privileged()}
        if self.sim.sens is not None:
            o["sensor"] = self.sim.sensor_obs(self.agents)
            if self.rc is not None:
                o["sensor"]["images"] = self._images()
        return o

    def reset(self):
        self.sim.reset()
        self._episode(np.ones(self.sim.cfg.n_envs, bool))
        return self.obs()

    def step(self, action: dict):
        """action: {"stick": (N,R,3), "goal": (N,R) int} for all robots (non-agents come from other policies)."""
        info = self.sim.step(action["stick"], action["goal"])
        extra = {"points": info.points, "fouls": info.fouls, "bonus": info.bonus,
                 "score": self.sim.score.copy(), "alliance": ALLIANCE}
        if info.done.any():
            extra["final_score"] = np.where(info.done[:, None], self.sim.score, np.nan)
            self.sim.reset(info.done)
            self._episode(info.done)
        return self.obs(), extra, info.done

    def close(self):
        if self.rc is not None:
            self.rc.close()
