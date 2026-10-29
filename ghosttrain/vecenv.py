"""Parallel teacher environments. Each worker owns a VecSim (no sensors, no rendering), the observation
encoder, a baseline policy (for baseline-controlled alliances) and the reward computation."""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import replace

import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

DECIDE_EVERY = 10          # goals are re-decided once per second (10 policy ticks)
GAMMA = 0.998
R_POINT = 0.1              # reward per match point
W_ACQUIRE = (0.10, 0.05)   # shaping: coral / algae acquired
W_POTENTIAL = 1.0          # shaping: potential-based distance to the baseline target (x -dist/20)
W_WIN = 1.0
DAMAGE_W = 1.0            # reward per damage point (damage is not game score; it is a training cost)


class _Worker:
    def __init__(self, cfg):
        import warnings
        warnings.filterwarnings("ignore")
        from ghostsim.sim import VecSim
        from .baseline import Baseline
        from .obs import Encoder
        self.sim = VecSim(cfg)
        self.n = cfg.n_envs
        self.enc, self.bl = Encoder(self.sim.game), Baseline(self.sim.game)
        self.ctrl = np.zeros((self.n, 2), bool)                  # True: alliance is driven by the baseline
        self._fresh()
        self.prev_score = self.sim.score.copy()
        self.prev_dmg = self.sim.gs.damage.copy()
        self.prev_push = self.sim.gs.push_events.copy()
        self.plan_cache = None
        self.noise = 0.0                                          # exec noise on baseline-driven robots (labels stay clean)
        self.label_all = False                                    # DAgger: baseline labels for every robot, whoever drives
        self.rng = np.random.default_rng(cfg.seed + 7)
        self.trunc_at = self.rng.integers(20, 1300, self.n)       # desynchronise the envs: first episode is cut short

    def _fresh(self, mask=None):
        m = np.ones(self.n, bool) if mask is None else mask
        if not hasattr(self, "goal"):
            self.goal = np.zeros((self.n, 6), int)
            self.tick = np.zeros(self.n, int)
            self.prev_hold = np.zeros((self.n, 6, 2), bool)
            self.phi = np.zeros((self.n, 6))
        self.goal[m], self.tick[m] = 0, 0

    def _phi(self, plan):
        _, tgt, _ = plan
        d = np.linalg.norm(tgt - self.sim.body.pose[..., :2], axis=-1)
        return -np.minimum(d, 20.0) / 20.0

    def _obs(self):
        from ghostsim.robots import ALLIANCE
        o = self.enc.encode(self.sim, self.goal)
        o["decide"] = (self.tick % DECIDE_EVERY == 0)
        o["bl"] = self.ctrl[:, ALLIANCE]
        return o

    def reset(self):
        self.sim.reset()
        self._fresh()
        self.prev_score = self.sim.score.copy()
        self.prev_dmg = self.sim.gs.damage.copy()
        self.prev_push = self.sim.gs.push_events.copy()
        self.prev_hold = self.sim.game.holding(self.sim.gs).copy()
        self.plan_cache = self.bl.plan(self.sim)
        self.phi = self._phi(self.plan_cache)
        return self._obs()

    def step(self, stick, goal_new):
        from ghostsim.robots import ALLIANCE
        sim, gs = self.sim, self.sim.gs
        plan = self.plan_cache if self.plan_cache is not None else self.bl.plan(sim)
        dec = (self.tick % DECIDE_EVERY == 0)[:, None]
        self.goal = np.where(dec, goal_new, self.goal)                          # held goal = what the observation shows
        exec_goal = self.goal.copy()
        bl_m = self.ctrl[:, ALLIANCE]                                           # (n,R)
        bst = None
        if bl_m.any() or self.label_all:
            bst, _ = self.bl.act(sim, plan)
            stick = np.where(bl_m[..., None], bst, stick)
            exec_goal = np.where(bl_m, plan[0], exec_goal)                      # the baseline re-plans every tick
            self.goal = np.where(bl_m & dec, plan[0], self.goal)                # ...but "decides" once per second
        if self.label_all:                                                      # what the baseline would do here
            act_stick, act_goal = bst.copy(), np.asarray(plan[0]).copy()
        else:
            act_stick, act_goal = stick.copy(), exec_goal.copy()                # clean labels for behaviour cloning
        if self.noise > 0 and bl_m.any():
            stick = np.where(bl_m[..., None], np.clip(stick + self.rng.normal(0, self.noise, stick.shape), -1, 1), stick)
        info = sim.step(stick, exec_goal)
        self.tick += 1
        diff = sim.score - self.prev_score
        self.prev_score = sim.score.copy()
        dmg_inc = sim.gs.damage - self.prev_dmg                                # (n,R) damage points taken this tick
        self.prev_dmg = sim.gs.damage.copy()
        g = sim.game.model
        press_pen = (sim.gs.press_t > g["teammate_press_grace_s"]) * g.get("teammate_press_pen_per_s", 0.0) * 0.1
        push_inc = sim.gs.push_events - self.prev_push
        self.prev_push = sim.gs.push_events.copy()
        pen = -R_POINT * (DAMAGE_W * dmg_inc + press_pen + g.get("push_scoring_teammate_pen", 0.0) * push_inc)
        team = (diff[:, ALLIANCE] - diff[:, 1 - ALLIANCE]) * R_POINT                # (n,R)
        hold = sim.game.holding(gs)
        acq = (hold & ~self.prev_hold).astype(float)
        self.prev_hold = hold.copy()
        shape_acq = acq[..., 0] * W_ACQUIRE[0] + acq[..., 1] * W_ACQUIRE[1]
        plan2 = self.bl.plan(sim)
        phi2 = self._phi(plan2)
        self.plan_cache = plan2
        shape_pot = W_POTENTIAL * (GAMMA * phi2 - self.phi)
        self.phi = phi2
        final = np.full((self.n, 2), np.nan)
        final_cat = np.full((self.n, 2, 5), np.nan)     # category points before the auto-reset below
        final_rule = np.full((self.n,) + sim.gs.foul_rule.shape[1:], np.nan)   # foul points conceded per rule
        win = np.zeros((self.n, R_))
        done = info.done
        trunc = (self.tick >= self.trunc_at) & ~done
        self.trunc_at[trunc] = 10 ** 9
        if done.any():
            final[done] = sim.score[done]
            final_cat[done] = sim.gs.cat[done]
            final_rule[done] = sim.gs.foul_rule[done]
            w = np.sign(sim.score[done][:, ALLIANCE] - sim.score[done][:, 1 - ALLIANCE])
            win[done] = w * W_WIN
            sim.reset(done)
            self._fresh(done)
            self.prev_score[done] = sim.score[done]
            self.prev_dmg[done] = 0.0
            self.prev_push[done] = 0.0
            self.prev_hold[done] = sim.game.holding(sim.gs)[done]
            plan3 = self.bl.plan(sim)
            self.phi[done] = self._phi(plan3)[done]
            self.plan_cache = plan3
        if trunc.any():
            sim.reset(trunc)
            self._fresh(trunc)
            self.prev_score[trunc] = sim.score[trunc]
            self.prev_dmg[trunc] = 0.0
            self.prev_push[trunc] = 0.0
            self.prev_hold[trunc] = sim.game.holding(sim.gs)[trunc]
            self.plan_cache = self.bl.plan(sim)
            self.phi[trunc] = self._phi(self.plan_cache)[trunc]
            done = done | trunc
        return {"obs": self._obs(), "act_stick": act_stick.astype(np.float32), "act_goal": act_goal,
                "r_team": (team + win + pen).astype(np.float32),
                "r_shape": (shape_acq + shape_pot).astype(np.float32), "done": done.copy(),
                "final": final, "final_cat": final_cat, "final_rule": final_rule,
                "pts": info.points.astype(np.float32), "fouls": info.fouls.astype(np.float32)}

    def set_ctrl(self, c, noise=None, label_all=None):
        self.ctrl = np.asarray(c, bool)
        if noise is not None:
            self.noise = float(noise)
        if label_all is not None:
            self.label_all = bool(label_all)


R_ = 6


def _main(conn, cfg):
    w = _Worker(cfg)
    while True:
        cmd, *a = conn.recv()
        if cmd == "reset":
            conn.send(w.reset())
        elif cmd == "step":
            conn.send(w.step(*a))
        elif cmd == "ctrl":
            w.set_ctrl(*a)
            conn.send(True)
        elif cmd == "eval_state":
            conn.send({"score": w.sim.score.copy(), "t": w.sim.t.copy()})
        elif cmd == "close":
            return


def _cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class TeacherVecEnv:
    def __init__(self, cfg, n_procs: int, in_process: bool = False):
        self.n_procs, self.per, self.in_process = n_procs, cfg.n_envs, in_process
        self.n_envs = n_procs * cfg.n_envs
        if in_process:
            self.local = [_Worker(replace(cfg, seed=cfg.seed + 1000 * i)) for i in range(n_procs)]
            return
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n_procs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_main, args=(b, replace(cfg, seed=cfg.seed + 1000 * i)), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)

    def _sl(self, x, i):
        return x[i * self.per:(i + 1) * self.per]

    def reset(self):
        if self.in_process:
            return _cat([w.reset() for w in self.local])
        for c in self.conns:
            c.send(("reset",))
        return _cat([c.recv() for c in self.conns])

    def step(self, stick, goal):
        if self.in_process:
            outs = [w.step(self._sl(stick, i), self._sl(goal, i)) for i, w in enumerate(self.local)]
        else:
            for i, c in enumerate(self.conns):
                c.send(("step", self._sl(stick, i), self._sl(goal, i)))
            outs = [c.recv() for c in self.conns]
        res = _cat([{k: v for k, v in o.items() if k != "obs"} for o in outs])
        res["obs"] = _cat([o["obs"] for o in outs])
        return res

    def set_ctrl(self, ctrl, noise=None, label_all=None):
        """ctrl (n_envs, 2) bool: alliance driven by the baseline policy. noise: exec noise std on its stick.
        label_all: act_stick / act_goal are the baseline's actions for every robot (DAgger labels)."""
        if self.in_process:
            for i, w in enumerate(self.local):
                w.set_ctrl(self._sl(ctrl, i), noise, label_all)
            return
        for i, c in enumerate(self.conns):
            c.send(("ctrl", self._sl(ctrl, i), noise, label_all))
        [c.recv() for c in self.conns]

    def close(self):
        if not self.in_process:
            for c in self.conns:
                c.send(("close",))
            for p in self.procs:
                p.join(timeout=5)
