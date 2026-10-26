"""Teammate-intent predictor: shapes, probabilities, and the obs block it feeds."""
import numpy as np

from ghostsim import SimCfg, load
from ghostsim.sim import VecSim
from ghosttrain.intent import RAW_FEATS, load_weights, predict_np
from ghosttrain.obs import FO, FO_RAW, Encoder


def test_obs_has_intent_block_for_allies_only():
    sim = VecSim(SimCfg(n_envs=3, sensors=False, seed=4))
    enc = Encoder(sim.game)
    o = enc.encode(sim, np.zeros((3, 6), int))
    assert o["others"].shape[-1] == FO == FO_RAW + 13
    assert enc.last_teammate.shape == (3, 6, 2, RAW_FEATS)
    intent = o["others"][..., FO_RAW:]
    assert np.allclose(intent[:, :, 2:].sum(), 0.0)           # opponents: no intent block
    assert np.allclose(intent[:, :, :2].sum(-1), 1.0, atol=1e-4)   # allies: a distribution


def test_zero_block_without_weights():
    sim = VecSim(SimCfg(n_envs=2, sensors=False, seed=4))
    enc = Encoder(sim.game, intent_weights=None)
    o = enc.encode(sim, np.zeros((2, 6), int))
    assert np.allclose(o["others"][..., FO_RAW:], 0.0)


def test_shipped_weights_are_a_probability_model():
    w = load_weights()
    assert w is not None, "ghosttrain/intent_weights.npz missing: run ghosttrain.intent collect/train"
    x = np.random.default_rng(0).normal(size=(50, RAW_FEATS)).astype(np.float32)
    p = predict_np(w, x)
    assert p.shape == (50, 13)
    assert np.allclose(p.sum(-1), 1.0, atol=1e-5) and (p >= 0).all()
    assert len(load("2025").goals) == 13
