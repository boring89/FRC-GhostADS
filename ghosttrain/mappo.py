"""MAPPO self-play + league trainer (own implementation).

- one shared actor for all robots of the learner, centralised critic input (see net.py)
- goal head decides once per second (DECIDE_EVERY ticks), stick head every tick
- opponents per env: SELF (both alliances learn), BASELINE (rule-based, driven inside the worker) or a frozen
  POOL snapshot; the opponent of an episode is re-drawn at every reset with PFSP weights
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ghostsim.robots import ALLIANCE, R

from .net import TeacherNet, goal_logp_entropy, stick_entropy, stick_logp
from .vecenv import DECIDE_EVERY, GAMMA

KEYS = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")


@dataclass
class TrainCfg:
    n_procs: int = 12
    envs_per_proc: int = 32
    horizon: int = 64
    epochs: int = 3
    minibatch: int = 8192
    lr: float = 3e-4
    gamma: float = GAMMA
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_goal: float = 0.01
    ent_stick: float = 0.001
    max_grad: float = 0.5
    shape_w0: float = 1.0
    shape_anneal_iters: int = 1500
    p_self: float = 0.35
    pool_size: int = 6
    snapshot_every: int = 100
    d: int = 64
    device: str = "cuda"
    seed: int = 0
    out: str = "runs/teacher"
    log_every: int = 1
    save_every: int = 50
    init: str = ""                  # warm start (behaviour-cloned net); also seeds the league pool
    critic_warmup: int = 15          # iterations that train only the critic after a warm start


def compute_gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done (T,B); last_val (B,). done[t] = the episode ended after step t (next value is not used)."""
    T = rew.shape[0]
    adv = torch.zeros_like(val)
    g = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = (~done[t]).float()
        g = rew[t] + gamma * nv * nonterm - val[t] + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + val


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        d, tot = bm - self.mean, self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return max(self.var, 1e-8) ** 0.5


class League:
    """Opponent selection. id -1 = SELF, 0 = BASELINE, k>=1 = pool snapshot."""

    def __init__(self, cfg: TrainCfg, rng):
        self.cfg, self.rng = cfg, rng
        self.snaps: dict[int, dict] = {}            # id -> state_dict
        self.win: dict[int, float] = {0: 0.0}       # EMA of the learner win rate (ties = 0.5)
        self.games: dict[int, int] = {0: 0}
        self.next_id = 1

    def add(self, state, win=0.5):
        if len(self.snaps) >= self.cfg.pool_size:                # drop the easiest (highest learner win rate)
            drop = max(self.snaps, key=lambda k: self.win.get(k, 0.5))
            self.snaps.pop(drop); self.win.pop(drop, None); self.games.pop(drop, None)
        i = self.next_id; self.next_id += 1
        self.snaps[i] = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.win[i], self.games[i] = win, 0
        return i

    def draw(self, n):
        ids = [0] + sorted(self.snaps)
        w = np.array([(1.0 - self.win.get(i, 0.5)) ** 2 + 0.05 for i in ids])
        p = np.concatenate([[self.cfg.p_self], (1 - self.cfg.p_self) * w / w.sum()])
        choice = self.rng.choice(len(p), size=n, p=p)
        return np.array([-1] + ids)[choice]

    def record(self, opp, learner_win):
        if opp < 0:
            return
        a = 0.05 if self.games.get(opp, 0) > 20 else 0.2
        self.win[opp] = (1 - a) * self.win.get(opp, 0.5) + a * learner_win
        self.games[opp] = self.games.get(opp, 0) + 1


def stale_opponents(opp, league):
    """Envs whose opponent snapshot was evicted from the pool (they must draw a new one)."""
    valid = [-1, 0, *league.snaps]
    return ~np.isin(opp, valid)


class Trainer:
    def __init__(self, cfg: TrainCfg, env=None):
        from ghostsim.sim import SimCfg
        from .vecenv import TeacherVecEnv
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.env = env or TeacherVecEnv(SimCfg(n_envs=cfg.envs_per_proc, sensors=False, seed=cfg.seed), cfg.n_procs)
        self.N = self.env.n_envs
        from .obs import Encoder
        from ghostsim import load
        self.enc = Encoder(load("2025"))
        self.net = TeacherNet(self.enc.n_goals, self.enc.G, cfg.d).to(self.dev)
        if cfg.init:
            sd = torch.load(cfg.init, map_location=self.dev, weights_only=False)
            self.net.load_state_dict(sd["net"] if "net" in sd else sd)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.league = League(cfg, self.rng)
        if cfg.init:
            self.league.add(self.net.state_dict(), win=0.5)
        self.pool_nets: dict[int, TeacherNet] = {}
        self.opp = np.full(self.N, -1)
        self.ret_rms = RunningMeanStd()
        self.it, self.steps = 0, 0
        self.out = Path(cfg.out); self.out.mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.out / "log.jsonl", "a")
        self.obs = None
        self.stats = {"ep": [], "returns": []}
        self.ep_pts = np.zeros((self.N, 2)); self.ep_foul = np.zeros((self.N, 2))      # running per-episode audit
        self.anom_f = open(self.out / "anomalies.jsonl", "a")
        self._assign(np.ones(self.N, bool))

    # ------------------------------------------------------------------ opponents
    def _assign(self, mask):
        self.opp[mask] = self.league.draw(int(mask.sum()))
        ctrl = np.zeros((self.N, 2), bool)
        ctrl[:, 1] = self.opp == 0
        self.env.set_ctrl(ctrl)

    def _pool_net(self, i):
        if i not in self.pool_nets:
            n = TeacherNet(self.enc.n_goals, self.enc.G, self.cfg.d).to(self.dev)
            n.load_state_dict(self.league.snaps[i]); n.eval()
            self.pool_nets[i] = n
        return self.pool_nets[i]

    # ------------------------------------------------------------------ obs helpers
    def _tensors(self, o):
        N = self.N
        t = {}
        for k in KEYS:
            v = torch.from_numpy(np.ascontiguousarray(o[k])).to(self.dev)
            t[k] = v.reshape(N * R, *v.shape[2:])
        g = torch.from_numpy(o["glob"]).to(self.dev)[:, ALLIANCE]                 # (N,R,G)
        t["glob"] = g.reshape(N * R, -1)
        t["decide"] = torch.from_numpy(o["decide"]).to(self.dev)[:, None].expand(N, R).reshape(-1)
        return t

    def _learner_mask(self):
        m = np.zeros((self.N, R), bool)
        m[:, :3] = True
        m[:, 3:] = (self.opp < 0)[:, None]
        return m.reshape(-1)

    @torch.no_grad()
    def _act(self, t, deterministic=False):
        logits, mu, value = self.net(t)
        std = self.net.log_std.exp().expand_as(mu)
        goal = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        stick = mu if deterministic else mu + std * torch.randn_like(mu)
        # frozen pool opponents (red alliance of their envs)
        opp = torch.from_numpy(np.repeat(self.opp, R)).to(self.dev)
        red = torch.from_numpy(np.tile(ALLIANCE == 1, self.N)).to(self.dev)
        for i in np.unique(self.opp[self.opp > 0]):
            sel = (opp == i) & red
            sub = {k: v[sel] for k, v in t.items()}
            lg, m2, _ = self._pool_net(int(i))(sub)
            goal[sel] = torch.distributions.Categorical(logits=lg).sample()
            stick[sel] = m2 + self._pool_net(int(i)).log_std.exp() * torch.randn_like(m2)
        return goal, stick, mu, logits, value

    # ------------------------------------------------------------------ rollout
    def rollout(self):
        cfg, T, N = self.cfg, self.cfg.horizon, self.N
        if self.obs is None:
            self.obs = self.env.reset()
        B = N * R
        buf = {k: [] for k in KEYS + ("glob", "decide")}
        acts = {k: [] for k in ("goal", "stick", "lp", "val", "rew", "done", "lm")}
        w_shape = cfg.shape_w0 * max(0.0, 1.0 - self.it / max(cfg.shape_anneal_iters, 1))
        for _ in range(T):
            t = self._tensors(self.obs)
            goal, stick, mu, logits, value = self._act(t)
            with torch.no_grad():
                lg, ent = goal_logp_entropy(logits, goal)
                lp = stick_logp(mu, self.net.log_std.expand_as(mu), stick) + torch.where(t["decide"], lg, torch.zeros_like(lg))
            for k in buf:
                buf[k].append(t[k])
            res = self.env.step(np.clip(stick.cpu().numpy().reshape(N, R, 3), -1, 1), goal.cpu().numpy().reshape(N, R))
            rew = torch.from_numpy(res["r_team"] + w_shape * res["r_shape"]).to(self.dev).reshape(-1)
            done = torch.from_numpy(np.repeat(res["done"], R)).to(self.dev)
            for k, v in (("goal", goal), ("stick", stick), ("lp", lp), ("val", value), ("rew", rew), ("done", done),
                         ("lm", torch.from_numpy(self._learner_mask()).to(self.dev))):
                acts[k].append(v)
            for a in range(2):
                self.ep_pts[:, a] += res["pts"][:, ALLIANCE == a].sum(1)
                self.ep_foul[:, a] += res["fouls"][:, ALLIANCE == a].sum(1)
            self.obs = res["obs"]
            self.steps += B
            if res["done"].any():
                self._episode_end(res)
                self._assign(res["done"])
        t = self._tensors(self.obs)
        with torch.no_grad():
            last_val = self.net(t)[2]
        return {k: torch.stack(v) for k, v in buf.items()}, {k: torch.stack(v) for k, v in acts.items()}, last_val, w_shape

    def _episode_end(self, res):
        for e in np.flatnonzero(res["done"]):
            b, r = res["final"][e]
            if np.isnan(b):                                   # desync truncation, not a finished match
                continue
            opp = int(self.opp[e])
            win = 1.0 if b > r else (0.5 if b == r else 0.0)
            self.league.record(opp, win)
            fb, fr = self.ep_foul[e]
            self.stats["ep"].append((opp, float(b), float(r), win, float(fb), float(fr)))
            if max(b, r) > 250 or max(fb, fr) > 40:                       # exploit candidates -> anomalies.jsonl
                self.anom_f.write(json.dumps({"it": self.it, "env": int(e), "opp": opp, "score": [float(b), float(r)],
                                              "robot_points": self.ep_pts[e].tolist(), "fouls_committed": [float(fb), float(fr)]}) + "\n")
                self.anom_f.flush()
        d = res["done"]
        self.ep_pts[d] = 0.0; self.ep_foul[d] = 0.0

    # ------------------------------------------------------------------ PPO
    def gae(self, acts, last_val):
        mean, std = self.ret_rms.mean, self.ret_rms.std
        return compute_gae(acts["rew"], acts["val"] * std + mean, acts["done"], last_val * std + mean,
                           self.cfg.gamma, self.cfg.lam)

    def update(self, buf, acts, last_val):
        cfg = self.cfg
        adv, ret = self.gae(acts, last_val)
        lm = acts["lm"]
        self.ret_rms.update(ret[lm].cpu().numpy())
        std, mean = self.ret_rms.std, self.ret_rms.mean
        idx = lm.reshape(-1).nonzero()[:, 0]
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        data = {k: flat(v)[idx] for k, v in buf.items()}
        goal, stick, lp_old = flat(acts["goal"])[idx], flat(acts["stick"])[idx], flat(acts["lp"])[idx]
        advf, retf = flat(adv)[idx], flat(ret)[idx]
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        retn = (retf - mean) / std
        n = idx.numel()
        stats = {"pg": 0.0, "vf": 0.0, "ent_goal": 0.0, "kl": 0.0, "clipfrac": 0.0}
        cnt = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.dev)
            for s in range(0, n, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                o = {k: v[mb] for k, v in data.items() if k != "decide"}
                logits, mu, value = self.net(o)
                lg, ent = goal_logp_entropy(logits, goal[mb])
                dec = data["decide"][mb]
                lp = stick_logp(mu, self.net.log_std.expand_as(mu), stick[mb]) + torch.where(dec, lg, torch.zeros_like(lg))
                ratio = (lp - lp_old[mb]).exp()
                a = advf[mb]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = ((value - retn[mb]) ** 2).mean()
                ent_g = (ent * dec).sum() / dec.sum().clamp(min=1)
                if self.it < cfg.critic_warmup and cfg.init:
                    loss = cfg.vf_coef * vf                                   # critic only; the cloned actor stays put
                else:
                    loss = pg + cfg.vf_coef * vf - cfg.ent_goal * ent_g - cfg.ent_stick * stick_entropy(self.net.log_std).sum()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad)
                self.opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent_goal"] += ent_g.item()
                    stats["kl"] += ((ratio - 1) - (lp - lp_old[mb])).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                cnt += 1
        stats = {k: v / max(cnt, 1) for k, v in stats.items()}
        with torch.no_grad():
            var_y = retn.var()
            stats["expl_var"] = float(1 - ((retn - self._value_of(data, idx)) ** 2).mean() / (var_y + 1e-8))
        return stats

    @torch.no_grad()
    def _value_of(self, data, idx):
        n = idx.numel()
        out = []
        for s in range(0, min(n, 16384), 4096):
            out.append(self.net({k: v[s:s + 4096] for k, v in data.items() if k != "decide"})[2])
        v = torch.cat(out)
        return torch.cat([v, torch.zeros(n - v.numel(), device=self.dev)]) if v.numel() < n else v

    # ------------------------------------------------------------------ loop
    def train(self, iters):
        for _ in range(iters):
            t0 = time.time()
            buf, acts, last_val, w_shape = self.rollout()
            t1 = time.time()
            st = self.update(buf, acts, last_val)
            self.it += 1
            if self.it % self.cfg.snapshot_every == 0:
                self.league.add(self.net.state_dict(), win=0.5)
                self.pool_nets.clear()
                stale = stale_opponents(self.opp, self.league)
                if stale.any():
                    self._assign(stale)
            ep = self.stats["ep"]
            self.stats["ep"] = []
            summ = self._summarise(ep)
            row = {"it": self.it, "steps": self.steps, "sps": self.cfg.horizon * self.N * R / (time.time() - t0),
                   "t_roll": t1 - t0, "t_upd": time.time() - t1, "w_shape": w_shape,
                   "ret_mean": float(acts["rew"].mean()), **st, **summ,
                   "pool": len(self.league.snaps), "winrate_baseline": self.league.win[0]}
            if self.it % self.cfg.log_every == 0:
                self.log_f.write(json.dumps(row) + "\n"); self.log_f.flush()
            if self.it % self.cfg.save_every == 0:
                self.save()
        return row

    def _summarise(self, ep):
        out = {"episodes": len(ep)}
        if ep:
            a = np.array(ep)
            out["score_blue"], out["score_red"] = float(a[:, 1].mean()), float(a[:, 2].mean())
            out["score_median"] = float(np.median(np.concatenate([a[:, 1], a[:, 2]])))
            out["score_max"] = float(max(a[:, 1].max(), a[:, 2].max()))
            out["foul_mean"] = float(a[:, 4:6].mean())
            out["foul_max"] = float(a[:, 4:6].max())
            for name, sel in (("self", a[:, 0] < 0), ("baseline", a[:, 0] == 0), ("pool", a[:, 0] > 0)):
                if sel.any():
                    out[f"win_{name}"], out[f"n_{name}"] = float(a[sel, 3].mean()), int(sel.sum())
                    out[f"score_{name}"] = float(a[sel, 1].mean())
        return out

    def save(self, name=None):
        path = self.out / (name or f"iter_{self.it:06d}.pt")
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), "it": self.it, "steps": self.steps,
                    "cfg": asdict(self.cfg), "ret_rms": (self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count),
                    "pool": self.league.snaps, "win": self.league.win}, path)
        torch.save(self.net.state_dict(), self.out / "latest_net.pt")
        return path

    def close(self):
        self.env.close()
