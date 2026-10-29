"""MAPPO self-play + league trainer (own implementation).

- one shared actor for all robots of the learner, centralised critic input (see net.py)
- goal head decides once per second (DECIDE_EVERY ticks), stick head every tick
- opponents per env: SELF (both alliances learn), BASELINE (rule-based, driven inside the worker) or a frozen
  POOL snapshot; the opponent of an episode is re-drawn at every reset with PFSP weights
- the learner's alliance is drawn per episode (side_swap): our robot (slot 0) is on the learner's side or on the
  opponent's half of the time, so the win rates are not biased by it (teacher5-7: learner always blue = always with
  our robot, self-play blue won ~70 %)
- PPO with one clipped surrogate per head (stick every tick, goal on decide ticks only, each averaged over its own
  samples); actor and critic gradients clipped separately; soft floor on the stick std (net.LOG_STD_MIN)
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
    n_procs: int = 6                # measured: 12 -> 6 procs costs ~20 % sps (12.4k -> 9.8k), frees CPU and GPU for the live view
    envs_per_proc: int = 32
    horizon: int = 64
    epochs: int = 3
    minibatch: int = 8192
    micro: int = 2048               # gradient accumulation chunk (memory only: same update as one 8192 minibatch)
    lr: float = 2e-5                # actor (encoder, trunk, stick head). The stick std is small (0.05): one Adam
                                    # step at 1e-4 moved the stick distribution by KL 0.1-0.5 (also with a 10x
                                    # smaller rate on the stick head alone: the shared trunk moves it); at 2e-5 the
                                    # KL is ~0.007 per iteration with every minibatch used
    lr_goal: float = 1e-4           # goal head (last layer: moves only the goal logits)
    lr_critic: float = 1e-4         # critic (reads a detached embedding: cannot move the actor)
    gamma: float = GAMMA
    lam: float = 0.95
    clip: float = 0.2               # stick ratio
    clip_goal: float = 0.2          # goal ratio (decide ticks)
    vf_coef: float = 0.5
    ent_goal: float = 0.005         # per decision (the goal loss and entropy are both averaged over decide ticks)
    target_kl: float = 0.02         # actor updates of an iteration stop once a minibatch's KL exceeds 1.5x this
                                    # (the critic keeps training); the stick std is small, so its ratio moves fast
    ent_stick: float = 0.001
    max_grad: float = 5.0           # actor gradient norm (clips outliers only; measured norms are logged)
    max_grad_critic: float = 2.0    # critic gradient norm
    shape_w0: float = 1.0
    shape_anneal_iters: int = 1500
    p_self: float = 0.35
    side_swap: bool = True          # learner alliance drawn per episode
    pool_size: int = 8
    snapshot_every: int = 25
    d: int = 64
    device: str = "cuda"
    seed: int = 0
    out: str = "runs/teacher"
    log_every: int = 1
    save_every: int = 10
    init: str = ""                  # warm start (behaviour-cloned net); also seeds the league pool
    init_log_std: float = -3.0      # stick log std set at a warm start (net.LOG_STD_INIT; nan keeps the file's)
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
        self.anchors: set[int] = set()              # never evicted (the warm start: keeps an old style in the pool)
        self.next_id = 1

    def add(self, state, win=0.5, anchor=False):
        free = [k for k in self.snaps if k not in self.anchors]
        if len(self.snaps) >= self.cfg.pool_size and free:       # drop the easiest (highest learner win rate)
            drop = max(free, key=lambda k: self.win.get(k, 0.5))
            self.snaps.pop(drop); self.win.pop(drop, None); self.games.pop(drop, None)
        i = self.next_id; self.next_id += 1
        self.snaps[i] = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.win[i], self.games[i] = win, 0
        if anchor:
            self.anchors.add(i)
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
        self.foul_rules = self.enc.g.foul_rules              # anomaly log: conceded points per rule
        self.net = TeacherNet(self.enc.n_goals, self.enc.G, cfg.d).to(self.dev)
        if cfg.init:
            sd = torch.load(cfg.init, map_location=self.dev, weights_only=False)
            self.net.load_state_dict(sd["net"] if "net" in sd else sd)
            if np.isfinite(cfg.init_log_std):
                with torch.no_grad():
                    self.net.log_std.fill_(cfg.init_log_std)
        self.critic_params = list(self.net.critic.parameters())
        self.actor_params = [p for n, p in self.net.named_parameters() if not n.startswith("critic.")]
        goal_p = list(self.net.goal.parameters())
        shared = [p for p in self.actor_params if all(p is not q for q in goal_p)]
        self.opt = torch.optim.Adam([{"params": shared, "lr": cfg.lr}, {"params": goal_p, "lr": cfg.lr_goal},
                                     {"params": self.critic_params, "lr": cfg.lr_critic}], eps=1e-5)
        self.league = League(cfg, self.rng)
        if cfg.init:
            self.league.add(self.net.state_dict(), win=0.5, anchor=True)
        self.pool_nets: dict[int, TeacherNet] = {}
        self.opp = np.full(self.N, -1)
        self.side = np.zeros(self.N, int)                     # learner alliance per env
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
        if self.cfg.side_swap:
            self.side[mask] = self.rng.integers(0, 2, int(mask.sum()))
        self.env.set_ctrl(self._ctrl())

    def _ctrl(self):
        """(N,2) alliances driven by the baseline inside the workers: the opponent side of baseline episodes."""
        ctrl = np.zeros((self.N, 2), bool)
        ctrl[np.arange(self.N), 1 - self.side] = self.opp == 0
        return ctrl

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
        """(N*R,) robots driven by the learning net: its alliance, and both alliances in self-play."""
        return ((ALLIANCE[None, :] == self.side[:, None]) | (self.opp < 0)[:, None]).reshape(-1)

    @torch.no_grad()
    def _act(self, t, deterministic=False):
        logits, mu, value = self.net(t)
        std = self.net.stick_log_std().exp().expand_as(mu)
        goal = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        stick = mu if deterministic else mu + std * torch.randn_like(mu)
        # frozen pool opponents (the other alliance of their envs)
        opp = torch.from_numpy(np.repeat(self.opp, R)).to(self.dev)
        other = torch.from_numpy((ALLIANCE[None, :] != self.side[:, None]).reshape(-1)).to(self.dev)
        for i in np.unique(self.opp[self.opp > 0]):
            sel = (opp == i) & other
            sub = {k: v[sel] for k, v in t.items()}
            pn = self._pool_net(int(i))
            lg, m2, _ = pn(sub)
            goal[sel] = torch.distributions.Categorical(logits=lg).sample()
            stick[sel] = m2 + pn.stick_log_std().exp() * torch.randn_like(m2)
        return goal, stick, mu, logits, value

    # ------------------------------------------------------------------ rollout
    def rollout(self):
        cfg, T, N = self.cfg, self.cfg.horizon, self.N
        if self.obs is None:
            self.obs = self.env.reset()
        B = N * R
        buf = {k: [] for k in KEYS + ("glob", "decide")}
        acts = {k: [] for k in ("goal", "stick", "lp_s", "lp_g", "val", "rew", "done", "lm")}
        w_shape = cfg.shape_w0 * max(0.0, 1.0 - self.it / max(cfg.shape_anneal_iters, 1))
        for _ in range(T):
            t = self._tensors(self.obs)
            goal, stick, mu, logits, value = self._act(t)
            with torch.no_grad():
                lg, ent = goal_logp_entropy(logits, goal)
                lp_s = stick_logp(mu, self.net.stick_log_std().expand_as(mu), stick)
                lp_g = torch.where(t["decide"], lg, torch.zeros_like(lg))
            for k in buf:
                buf[k].append(t[k])
            res = self.env.step(np.clip(stick.cpu().numpy().reshape(N, R, 3), -1, 1), goal.cpu().numpy().reshape(N, R))
            rew = torch.from_numpy(res["r_team"] + w_shape * res["r_shape"]).to(self.dev).reshape(-1)
            done = torch.from_numpy(np.repeat(res["done"], R)).to(self.dev)
            for k, v in (("goal", goal), ("stick", stick), ("lp_s", lp_s), ("lp_g", lp_g), ("val", value), ("rew", rew), ("done", done),
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
            opp, sd = int(self.opp[e]), int(self.side[e])
            me, them = res["final"][e][sd], res["final"][e][1 - sd]               # learner / opponent alliance
            win = 1.0 if me > them else (0.5 if me == them else 0.0)
            self.league.record(opp, win)
            fb, fr = self.ep_foul[e]
            self.stats["ep"].append((opp, float(me), float(them), win, float(self.ep_foul[e, sd]),
                                     float(self.ep_foul[e, 1 - sd]), sd))
            if max(b, r) > 350 or max(fb, fr) > 60:                       # exploit candidates -> anomalies.jsonl
                rules = getattr(self, "foul_rules", None)
                fr_rule = res["final_rule"][e] if "final_rule" in res else None
                by_rule = {} if fr_rule is None or rules is None else \
                    {k: [float(fr_rule[0, i]), float(fr_rule[1, i])] for i, k in enumerate(rules) if fr_rule[:, i].any()}
                self.anom_f.write(json.dumps({"it": self.it, "env": int(e), "opp": opp, "score": [float(b), float(r)],
                                              "robot_points": self.ep_pts[e].tolist(), "fouls_committed": [float(fb), float(fr)],
                                              "fouls_by_rule": by_rule}) + "\n")
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
        goal, stick = flat(acts["goal"])[idx], flat(acts["stick"])[idx]
        lp_s_old, lp_g_old = flat(acts["lp_s"])[idx], flat(acts["lp_g"])[idx]
        advf, retf = flat(adv)[idx], flat(ret)[idx]
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        retn = (retf - mean) / std
        n = idx.numel()
        keys = ("pg_stick", "pg_goal", "vf", "ent_goal", "kl_stick", "kl_goal", "clip_stick", "clip_goal",
                "gn_actor", "gn_critic")
        stats = dict.fromkeys(keys, 0.0)
        cnt = act_cnt = 0
        warm = self.it < cfg.critic_warmup and bool(cfg.init)
        actor_on = not warm
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.dev)
            for s in range(0, n, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                n_mb, nd = float(mb.numel()), data["decide"][mb].float().sum().clamp(min=1.0)
                self.opt.zero_grad()
                acc = dict.fromkeys(("pg_stick", "pg_goal", "vf", "ent_goal", "kl_stick", "kl_goal", "clip_stick",
                                     "clip_goal"), 0.0)
                for c in mb.split(cfg.micro):                                       # sums / minibatch counts
                    o = {k: v[c] for k, v in data.items() if k != "decide"}
                    logits, mu, value = self.net(o)
                    lg, ent = goal_logp_entropy(logits, goal[c])
                    dec = data["decide"][c].float()
                    log_std = self.net.stick_log_std()
                    a = advf[c]
                    d_s = stick_logp(mu, log_std.expand_as(mu), stick[c]) - lp_s_old[c]
                    d_g = (lg - lp_g_old[c]) * dec                                  # 0 (ratio 1) off the decide ticks
                    r_s, r_g = d_s.exp(), d_g.exp()
                    pg_s = -torch.min(r_s * a, r_s.clamp(1 - cfg.clip, 1 + cfg.clip) * a).sum() / n_mb
                    pg_g = -(torch.min(r_g * a, r_g.clamp(1 - cfg.clip_goal, 1 + cfg.clip_goal) * a) * dec).sum() / nd
                    vf = ((value - retn[c]) ** 2).sum() / n_mb
                    ent_g = (ent * dec).sum() / nd
                    if not actor_on:
                        loss = cfg.vf_coef * vf                                     # critic only (warm-up, or KL reached)
                    else:
                        loss = pg_s + pg_g + cfg.vf_coef * vf - cfg.ent_goal * ent_g \
                            - cfg.ent_stick * stick_entropy(log_std) * (c.numel() / n_mb)
                    loss.backward()
                    with torch.no_grad():
                        for k, v in (("pg_stick", pg_s), ("pg_goal", pg_g), ("vf", vf), ("ent_goal", ent_g),
                                     ("kl_stick", ((r_s - 1) - d_s).sum() / n_mb),
                                     ("kl_goal", (((r_g - 1) - d_g) * dec).sum() / nd),
                                     ("clip_stick", ((r_s - 1).abs() > cfg.clip).float().sum() / n_mb),
                                     ("clip_goal", (((r_g - 1).abs() > cfg.clip_goal).float() * dec).sum() / nd)):
                            acc[k] += float(v)
                if actor_on and acc["kl_stick"] + acc["kl_goal"] > 1.5 * cfg.target_kl:
                    actor_on = False                                                # this minibatch went too far:
                    for p in self.actor_params:                                    # drop its actor gradient
                        p.grad = None
                gn_a = torch.nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad) if actor_on else torch.zeros(())
                gn_c = torch.nn.utils.clip_grad_norm_(self.critic_params, cfg.max_grad_critic)
                self.opt.step()
                for k, v in acc.items():
                    stats[k] += v
                stats["gn_actor"] += float(gn_a); stats["gn_critic"] += float(gn_c)
                cnt += 1
                act_cnt += int(actor_on)
        stats = {k: v / max(cnt, 1) for k, v in stats.items()}
        stats["gn_actor"] *= cnt / max(act_cnt, 1)
        stats["kl"], stats["clipfrac"] = stats["kl_stick"] + stats["kl_goal"], stats["clip_stick"]
        stats["actor_frac"] = act_cnt / max(cnt, 1)                         # share of minibatches that moved the actor
        with torch.no_grad():
            sd = self.net.stick_log_std().exp()
            stats["std_stick"], stats["std_stick_min"] = float(sd.mean()), float(sd.min())
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
        """Episode stats from the learner's side: score_learner / score_opp, wins per opponent kind, and vs the
        baseline with and without our robot (slot 0, blue) on the learner's side."""
        out = {"episodes": len(ep)}
        if ep:
            a = np.array(ep)
            out["score_learner"], out["score_opp"] = float(a[:, 1].mean()), float(a[:, 2].mean())
            out["score_median"] = float(np.median(np.concatenate([a[:, 1], a[:, 2]])))
            out["score_max"] = float(max(a[:, 1].max(), a[:, 2].max()))
            out["foul_mean"] = float(a[:, 4:6].mean())
            out["foul_max"] = float(a[:, 4:6].max())
            for name, sel in (("self", a[:, 0] < 0), ("baseline", a[:, 0] == 0), ("pool", a[:, 0] > 0)):
                if sel.any():
                    out[f"win_{name}"], out[f"n_{name}"] = float(a[sel, 3].mean()), int(sel.sum())
                    out[f"score_{name}"] = float(a[sel, 1].mean())
            for name, sd in (("with_ours", 0), ("without_ours", 1)):
                sel = (a[:, 0] == 0) & (a[:, 6] == sd)
                if sel.any():
                    out[f"win_baseline_{name}"] = float(a[sel, 3].mean())
        return out

    def save(self, name=None):
        path = self.out / (name or f"iter_{self.it:06d}.pt")
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), "it": self.it, "steps": self.steps,
                    "cfg": asdict(self.cfg), "ret_rms": (self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count),
                    "pool": self.league.snaps, "win": self.league.win, "anchors": sorted(self.league.anchors)}, path)
        torch.save(self.net.state_dict(), self.out / "latest_net.pt")
        return path

    def close(self):
        self.env.close()
