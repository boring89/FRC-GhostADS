"""Teacher actor-critic. Actor: entity-set attention over the ego observation. Critic: centralised
(global state of the alliance + own embedding). Goals are masked by robot capabilities."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs import FF, FL, FO, FP, S

LOG_STD_MIN = -3.5        # stick exploration floor (std 0.03): without one log_std collapsed -1.6 -> -4.2 in teacher5-7.
                          # Not higher: scoring needs the robot stopped (<= 0.3 m/s), and stick noise costs points
                          # (bc9 vs baseline: std 0.05 131-117, 0.10 107-120, 0.20 31-116)
LOG_STD_INIT = -3.0       # std 0.05 for a warm start (behaviour cloning has no std of its own)


def mlp(i, h, o, ln=True):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o)) if not ln else \
        nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o), nn.LayerNorm(o))


class Block(nn.Module):
    """Single-query cross attention + feed-forward (pre-norm, residual)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.d = heads, d
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, tok, mask):
        """x (B,d), tok (B,T,d), mask (B,T) True = valid."""
        B, T, d = tok.shape
        q = self.q(self.nq(x)).view(B, self.h, 1, d // self.h)
        k, v = self.kv(self.nk(tok)).view(B, T, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h)
        att = att.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        x = x + self.o((att @ v).reshape(B, d))
        return x + self.ff(x)


class TeacherNet(nn.Module):
    def __init__(self, n_goals=14, glob_dim=333, d=64, heads=4, layers=2):      # 14 goals (align_score added)
        super().__init__()
        self.n_goals = n_goals
        self.e_self = mlp(S, 2 * d, d)
        self.e_oth, self.e_pc, self.e_face, self.e_lm = mlp(FO, d, d), mlp(FP, d, d), mlp(FF, d, d), mlp(FL, d, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.trunk = nn.Sequential(nn.Linear(d + S, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU())
        self.goal = nn.Linear(2 * d, n_goals)
        self.mu = nn.Linear(2 * d, 3)
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        self.critic = nn.Sequential(nn.Linear(glob_dim + d, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.goal.weight, 0.01); nn.init.zeros_(self.goal.bias)
        nn.init.orthogonal_(self.mu.weight, 0.01); nn.init.zeros_(self.mu.bias)

    def encode(self, o):
        x = self.e_self(o["self"])
        tok = torch.cat([self.e_oth(o["others"]) + self.type_emb[0], self.e_pc(o["pieces"]) + self.type_emb[1],
                         self.e_face(o["faces"]) + self.type_emb[2], self.e_lm(o["lm"]) + self.type_emb[3]], 1)
        B = tok.shape[0]
        mask = torch.cat([torch.ones(B, o["others"].shape[1], dtype=torch.bool, device=tok.device), o["pmask"],
                          torch.ones(B, o["faces"].shape[1] + o["lm"].shape[1], dtype=torch.bool, device=tok.device)], 1)
        for blk in self.blocks:
            x = blk(x, tok, mask)
        return x

    def forward(self, o):
        """o: dict of tensors with leading batch B (robot-agents). Returns goal logits (masked), mean, value."""
        x = self.encode(o)
        h = self.trunk(torch.cat([x, o["self"]], -1))
        logits = self.goal(h).masked_fill(~o["gmask"], -1e9)
        value = self.critic(torch.cat([o["glob"], x.detach()], -1))[:, 0]      # critic never perturbs the actor's encoder
        return logits, self.mu(h), value

    def stick_log_std(self):
        """log std of the stick with a soft floor at LOG_STD_MIN (identity well above it, never below it)."""
        return LOG_STD_MIN + F.softplus(5.0 * (self.log_std - LOG_STD_MIN)) / 5.0

    def dist(self, o):
        logits, mu, value = self.forward(o)
        return logits, mu, self.stick_log_std().expand_as(mu), value


def stick_logp(mu, log_std, a):
    var = (2 * log_std).exp()
    return (-0.5 * ((a - mu) ** 2 / var + 2 * log_std + math.log(2 * math.pi))).sum(-1)


def stick_entropy(log_std):
    return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)


def goal_logp_entropy(logits, goal):
    lp = F.log_softmax(logits, -1)
    p = lp.exp()
    ent = -(p * lp.clamp(min=-30)).sum(-1)
    return lp.gather(-1, goal[:, None])[:, 0], ent
