"""Vectorised match simulator (single source of truth). One step() = one policy tick
(default 0.1 s = 5 substeps of 20 ms, the Systemcore loop period)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from .game import load
from .motion import motion_layer
from .physics import Physics, PhysCfg
from .robots import ALLIANCE, R, load_profile, sample_params
from .sensors import Sensors


@dataclass
class SimCfg:
    n_envs: int = 16
    seed: int = 0
    game: str = "2025"
    tags: str | None = None                       # "welded" / "andymark"
    dt: float = 0.02
    substeps: int = 5
    slots: tuple = ("ours", "others", "others", "others", "others", "others")
    vision: tuple = (0,)                          # robots with full AprilTag visibility simulation
    sensors: bool = True                          # False: teacher training (privileged state only)
    start_mode: str | None = None                 # "teleop" / "auto" (default from game.json)
    phys: PhysCfg = field(default_factory=PhysCfg)


@dataclass
class StepInfo:
    points: np.ndarray        # (N,R) points earned by each robot this tick
    fouls: np.ndarray         # (N,R) points conceded by each robot this tick
    bonus: np.ndarray         # (N,2) alliance points not attributable to a robot (human player)
    done: np.ndarray          # (N,)


class VecSim:
    def __init__(self, cfg: SimCfg = SimCfg()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.game = g = load(cfg.game, cfg.tags)
        n = cfg.n_envs
        ours, others = load_profile("ours.json"), load_profile("others.json")
        self.ours = ours
        self.profiles = [ours if s == "ours" else others for s in cfg.slots]
        self.prm = sample_params(self.profiles, n, g.goals, self.rng)
        self.phys = Physics(g.polys, g.hp_n, g.hp_off, cfg.phys)
        self.body = self.phys.new_body(n)
        self.gs = g.new_state(n)
        self.sens = Sensors(g, ours["sensors"], ours["cameras"], n, cfg.dt, self.rng) if cfg.sensors else None
        if self.sens:
            self.sens.bind(self.prm)
        self.full_mask = np.zeros((n, R), bool)
        self.full_mask[:, list(cfg.vision)] = True
        self.mode = cfg.start_mode or g.start_mode
        self.end_t = g.auto_s + g.teleop_s
        self.t = np.zeros(n)
        self.auto_done = np.zeros(n, bool)
        self.tgt_az = np.zeros((n, R, 4))
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, mask=None):
        n = self.cfg.n_envs
        idx = np.arange(n) if mask is None else np.flatnonzero(mask)
        if len(idx) == 0:
            return
        new = sample_params(self.profiles, len(idx), self.game.goals, self.rng)
        for f in fields(new):
            getattr(self.prm, f.name)[idx] = getattr(new, f.name)
        self.game.reset(self.gs, idx, self.body, self.prm, self.rng, self.mode)
        self.t[idx] = 0.0 if self.mode == "auto" else self.game.auto_s
        self.auto_done[idx] = self.mode != "auto"
        self.tgt_az[idx] = 0.0
        self.body.fixed[idx] = False
        if self.sens:
            self.sens.reset(idx, self.body, self.prm)

    # ------------------------------------------------------------------ step
    def step(self, stick, goal) -> StepInfo:
        c, g, b = self.cfg, self.game, self.body
        n = c.n_envs
        stick = np.nan_to_num(np.asarray(stick, float))
        goal = np.clip(np.asarray(goal, np.int64), 0, len(g.goals) - 1)
        pts, fouls, bonus = np.zeros((n, R)), np.zeros((n, R)), np.zeros((n, 2))
        for _ in range(c.substeps):
            live = self.t < self.end_t - 1e-9
            yaw = self.sens.est[..., 2] if self.sens else b.pose[..., 2]
            az = self.sens.az if self.sens else b.az
            spd, ang = motion_layer(stick, ALLIANCE[None], yaw, az, self.tgt_az, self.prm.control(), c.dt)
            spd[~live] = 0.0                                   # robots are disabled after the match
            self.tgt_az = ang
            b.fixed = g.fixed(self.gs).copy()
            self.phys.substep(b, self.prm, spd, ang, c.dt)
            p, f, bo = g.substep(self.gs, b, self.prm, goal, self.t, live, c.dt, self.rng)
            pts, fouls, bonus = pts + p, fouls + f, bonus + bo
            if self.sens:
                self.sens.update(b, self.full_mask, live)
            self.t[live] += c.dt
            ea = live & ~self.auto_done & (self.t >= g.auto_s - 1e-9)
            if ea.any():
                pts += g.end_auto(self.gs, b, self.prm, ea)
                self.auto_done |= ea
            em = live & (self.t >= self.end_t - 1e-9)
            if em.any():
                pts += g.end_match(self.gs, b, self.prm, em)
        return StepInfo(points=pts, fouls=fouls, bonus=bonus, done=self.t >= self.end_t - 1e-9)

    # ------------------------------------------------------------------ observations
    def match_features(self):
        n = self.cfg.n_envs
        rem = (self.end_t - self.t) / self.game.teleop_s
        return np.stack(np.broadcast_arrays(rem[:, None], (self.t < self.game.auto_s)[:, None].astype(float),
                                            ALLIANCE[None].astype(float)), -1).reshape(n, R, 3)

    def privileged(self) -> dict:
        b = self.body
        out = {"pose": b.pose, "vel": b.vel, "alliance": np.broadcast_to(ALLIANCE, b.pose.shape[:2]),
               "half": self.prm.half, "t": self.t, "match": self.match_features()}
        out.update(self.game.privileged(self.gs))
        return out

    def sensor_obs(self, robots=(0,)) -> dict:
        o = self.sens.obs(self.game.holding(self.gs), self.match_features())
        return {k: v[:, list(robots)] for k, v in o.items()}

    @property
    def score(self):
        return self.gs.score

    def render_state(self) -> dict:
        objs = self.game.render_objects(self.gs, self.body, self.prm, self.ours["mechanism"]["held_offset_m"])
        robots = np.concatenate([self.body.pose, objs.pop("robot_z")[..., None]], -1).astype(np.float32)
        return {"robots": robots, **objs}
