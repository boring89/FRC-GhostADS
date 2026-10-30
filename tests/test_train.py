import numpy as np
import pytest
import torch

from conftest import make_sim, place
from ghostsim import SimCfg, VecSim
from ghostsim.robots import ALLIANCE, R
from ghosttrain.audit import Auditor, state_points
from ghosttrain.baseline import Baseline
from ghosttrain.mappo import League, TrainCfg, Trainer, compute_gae, stale_opponents
from ghosttrain.net import LOG_STD_MIN, TeacherNet, goal_logp_entropy, stick_logp
from ghosttrain.obs import Encoder
from ghosttrain.vecenv import DECIDE_EVERY, TeacherVecEnv, _Worker

IN = 0.0254


def quiet_sim(n=2, seed=0, **kw):
    return VecSim(SimCfg(n_envs=n, sensors=False, seed=seed, slots=("ours",) * 6, randomize=False,
                         **{"start_mode": "teleop", **kw}))


# ------------------------------------------------------------------ baseline
def test_baseline_scores_both_alliances_and_audit_is_clean():
    sim = quiet_sim(n=4, seed=1)
    bl, aud = Baseline(sim.game), Auditor(sim)
    for _ in range(1350):
        st, gl = bl.act(sim)
        aud.observe(st, gl, sim.step(st, gl))
    assert (sim.score > 20).all(), sim.score
    bad = {k: v for k, v in aud.report().items() if k in ("out_of_bounds", "conservation", "score_mismatch", "too_fast")}
    assert not bad, bad


def test_baseline_beats_idle_and_random():
    sim = quiet_sim(n=4, seed=2)
    bl = Baseline(sim.game)
    rng = np.random.default_rng(0)
    for _ in range(1350):
        st, gl = bl.act(sim)
        st[:, 3:] = rng.uniform(-1, 1, (4, 3, 3))                 # red plays randomly
        gl[:, 3:] = rng.integers(0, 14, (4, 3))
        sim.step(st, gl)
    assert (sim.score[:, 0] > sim.score[:, 1]).all()


# ------------------------------------------------------------------ observation
def test_obs_shapes_and_goal_mask():
    sim = VecSim(SimCfg(n_envs=3, sensors=False, seed=1))
    enc = Encoder(sim.game)
    o = enc.encode(sim, np.zeros((3, 6), int))
    assert o["self"].shape[:2] == (3, 6) and o["glob"].shape == (3, 2, enc.G)
    assert all(np.isfinite(v.astype(float)).all() for v in o.values())
    gm = o["gmask"]
    assert gm[..., enc.gi["idle"]].all()
    assert (gm[..., enc.gi["processor"]] == sim.prm.processor).all()
    assert (gm[..., enc.gi["score_L4"]] == sim.prm.levels[..., 3]).all()


def test_canonical_symmetry_red_sees_what_blue_sees():
    sim = make_sim(n=1, full=False)
    g, enc = sim.game, Encoder(sim.game)
    L, W = g.length, g.width
    P = [(4.0, 2.5, 0.6), (6.5, 6.5, -1.0), (2.0, 5.0, 2.0)]
    for r, (x, y, yaw) in enumerate(P):                          # blue robots
        place(sim, r, [x, y], yaw)
        place(sim, r + 3, [L - x, W - y], yaw + np.pi)          # red robots: 180-degree rotated copies
    sim.body.vel[:, :3, :2] = [[1.0, 0.5], [-0.3, 0.2], [0.0, 1.0]]
    sim.body.vel[:, 3:, :2] = -sim.body.vel[:, :3, :2]
    sim.body.vel[:, 3:, 2] = sim.body.vel[:, :3, 2] = [[0.3, -0.2, 0.1]]
    o = enc.encode(sim, np.zeros((1, 6), int))
    for r in range(3):
        a, b = r, r + 3
        assert np.allclose(o["self"][0, a, :12], o["self"][0, b, :12], atol=1e-5), "own state must be identical"
        assert np.allclose(o["faces"][0, a], o["faces"][0, b], atol=1e-5), "faces"
        assert np.allclose(o["lm"][0, a], o["lm"][0, b], atol=1e-4), "landmarks"   # tag file is symmetric to ~0.1 mm
        assert np.allclose(np.sort(o["others"][0, a], axis=0), np.sort(o["others"][0, b], axis=0), atol=1e-5)
        pa = o["pieces"][0, a][o["pmask"][0, a]]
        pb = o["pieces"][0, b][o["pmask"][0, b]]
        assert np.allclose(np.sort(pa, axis=0), np.sort(pb, axis=0), atol=1e-5)
    assert np.allclose(o["glob"][0, 0], o["glob"][0, 1], atol=1e-5)


# ------------------------------------------------------------------ network
def test_net_masks_goals_and_is_permutation_invariant():
    sim = VecSim(SimCfg(n_envs=2, sensors=False, seed=3))
    enc = Encoder(sim.game)
    o = enc.encode(sim, np.zeros((2, 6), int))
    t = {k: torch.from_numpy(np.ascontiguousarray(o[k])).reshape(12, *o[k].shape[2:]) for k in
         ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")}
    t["glob"] = torch.from_numpy(o["glob"])[:, ALLIANCE].reshape(12, -1)
    net = TeacherNet(enc.n_goals, enc.G).eval()
    logits, mu, v = net(t)
    assert logits.shape == (12, 14) and mu.shape == (12, 3) and v.shape == (12,)
    assert (logits[~t["gmask"]] < -1e8).all() and (logits[t["gmask"]] > -1e8).all()
    perm = torch.randperm(30)
    t2 = dict(t, pieces=t["pieces"][:, perm], pmask=t["pmask"][:, perm])
    l2, m2, v2 = net(t2)
    assert torch.allclose(m2, mu, atol=1e-5) and torch.allclose(v2, v, atol=1e-5)


def test_net_gradients_flow_to_all_parts():
    sim = VecSim(SimCfg(n_envs=2, sensors=False, seed=4))
    enc = Encoder(sim.game)
    o = enc.encode(sim, np.zeros((2, 6), int))
    t = {k: torch.from_numpy(np.ascontiguousarray(o[k])).reshape(12, *o[k].shape[2:]) for k in
         ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")}
    t["glob"] = torch.from_numpy(o["glob"])[:, ALLIANCE].reshape(12, -1)
    net = TeacherNet(enc.n_goals, enc.G)
    logits, mu, v = net(t)
    goal = torch.distributions.Categorical(logits=logits).sample()
    lg, ent = goal_logp_entropy(logits, goal)
    (lg.sum() + stick_logp(mu, net.log_std.expand_as(mu), mu.detach() + 0.1).sum() + v.sum()).backward()
    dead = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not dead, dead
    # the critic must not change the actor: value loss alone leaves actor-side parameters untouched
    net.zero_grad()
    net(t)[2].sum().backward()
    actor_side = [n for n, p in net.named_parameters() if not n.startswith("critic") and p.grad is not None and p.grad.abs().sum() > 0]
    assert not actor_side, actor_side


# ------------------------------------------------------------------ PPO pieces
def test_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    rew = torch.tensor([[1.0], [0.0], [2.0]]); val = torch.tensor([[0.5], [0.4], [0.3]])
    done = torch.tensor([[False], [False], [False]]); last = torch.tensor([0.2])
    adv, ret = compute_gae(rew, val, done, last, gamma, lam)
    d2 = 2.0 + gamma * 0.2 - 0.3
    d1 = 0.0 + gamma * 0.3 - 0.4
    d0 = 1.0 + gamma * 0.4 - 0.5
    a2 = d2; a1 = d1 + gamma * lam * a2; a0 = d0 + gamma * lam * a1
    assert torch.allclose(adv[:, 0], torch.tensor([a0, a1, a2]), atol=1e-6)
    done[1] = True                                               # episode ends after step 1: no bootstrap, no leakage
    adv, _ = compute_gae(rew, val, done, last, gamma, lam)
    assert torch.isclose(adv[1, 0], torch.tensor(0.0 - 0.4)) and torch.isclose(adv[0, 0], torch.tensor(1.0 + gamma * 0.4 - 0.5 + gamma * lam * (-0.4)))


def test_league_pfsp_prefers_hard_opponents_and_evicts_easy():
    cfg = TrainCfg(pool_size=2, p_self=0.0)
    lg = League(cfg, np.random.default_rng(0))
    a = lg.add({"w": torch.zeros(1)}, win=0.95)
    b = lg.add({"w": torch.zeros(1)}, win=0.1)
    lg.win[0] = 0.9
    d = lg.draw(4000)
    assert (d == b).sum() > 4 * (d == a).sum() and (d == b).sum() > (d == 0).sum()
    c = lg.add({"w": torch.zeros(1)}, win=0.5)                  # pool full: the easiest (a) leaves
    assert a not in lg.snaps and b in lg.snaps and c in lg.snaps


def test_league_anchor_is_never_evicted():
    lg = League(TrainCfg(pool_size=2), np.random.default_rng(0))
    a = lg.add({"w": torch.zeros(1)}, win=0.99, anchor=True)      # the warm start: easiest, but kept
    for _ in range(4):
        lg.add({"w": torch.zeros(1)}, win=0.5)
    assert a in lg.snaps and len(lg.snaps) == 2


def test_stick_std_has_a_soft_floor():
    net = TeacherNet(14, 333)
    with torch.no_grad():
        net.log_std.fill_(-1.6)
    assert torch.allclose(net.stick_log_std(), torch.full((3,), -1.6), atol=0.01)     # identity well above the floor
    with torch.no_grad():
        net.log_std.fill_(-8.0)
    assert (net.stick_log_std() >= LOG_STD_MIN - 1e-6).all()
    net.stick_log_std().sum().backward()
    assert net.log_std.grad.abs().sum() > 0                                          # still differentiable down there


def test_trainer_sides_learner_mask_and_per_head_update(tmp_path):
    """team "all": the learner may play either alliance: the mask, the baseline control and the pool side follow
    it; one PPO iteration reports finite per-head statistics."""
    cfg = TrainCfg(n_procs=1, envs_per_proc=4, horizon=8, epochs=1, minibatch=128, device="cpu", out=str(tmp_path),
                   team="all")
    tr = Trainer(cfg, TeacherVecEnv(SimCfg(n_envs=4, sensors=False, seed=0), 1, in_process=True))
    try:
        blue = ALLIANCE == 0
        for e, (opp, side) in enumerate(((0, 0), (-1, 1), (0, 1), (0, 0))):
            tr.opp[e], tr.side[e] = opp, side
            tr.drv[e] = tr._drivers(side, opp)
        m = tr._learner_mask().reshape(4, R)
        assert (m[0] == blue).all() and m[1].all() and (m[2] == ~blue).all() and (m[3] == blue).all()
        c = tr._ctrl()
        assert (c[0] == ~blue).all() and not c[1].any() and (c[2] == blue).all() and (c[3] == ~blue).all()
        tr._assign(np.ones(4, bool))
        assert set(np.unique(tr.side)) <= {0, 1}
        row = tr.train(1)
        for k in ("pg_stick", "pg_goal", "clip_stick", "clip_goal", "kl_stick", "kl_goal", "gn_actor", "gn_critic"):
            assert np.isfinite(row[k]), k
        assert row["std_stick_min"] >= np.exp(LOG_STD_MIN) - 1e-6
    finally:
        tr.close()


# ------------------------------------------------------------------ environment
def test_worker_rewards_decisions_and_autoreset():
    w = _Worker(SimCfg(n_envs=2, sensors=False, seed=0, randomize=False, slots=("ours",) * 6))
    o = w.reset()
    assert o["decide"].all() and o["bl"].shape == (2, 6) and not o["bl"].any()
    st, gl = np.zeros((2, 6, 3)), np.zeros((2, 6), int)
    seen = []
    for k in range(DECIDE_EVERY + 1):
        r = w.step(st, gl)
        seen.append(bool(r["obs"]["decide"][0]))
    assert seen[:DECIDE_EVERY - 1] == [False] * (DECIDE_EVERY - 1) and seen[DECIDE_EVERY - 1] and not seen[-1]
    # a blue score gives + to blue robots and - to red robots (zero-sum team reward)
    g = w.sim.game
    n = g.reef_n[0, 3]
    hx = w.sim.prm.half[0, 0, 0]
    half_pipe = g.cfg["field"]["reef"]["pipe_spacing"] * 0.0254 / 2
    w.sim.body.pose[0, 0] = [*(g.reef_fc[0, 3] + n * (hx + 0.1) + g.reef_t[0, 3] * half_pipe), np.arctan2(-n[1], -n[0])]
    w.sim.body.vel[:] = 0
    w.sim.gs.hold_coral[0, 0] = True
    for _ in range(3 * DECIDE_EVERY):                              # next decision tick + 1.7 s elevator
        r = w.step(st, np.where(np.arange(6) == 0, g.goals.index("score_L4"), 0)[None].repeat(2, 0))
        if r["r_team"][0, 0] > 0:
            break
    assert r["r_team"][0, 0] > 0 and np.isclose(r["r_team"][0, 0], -r["r_team"][0, 3]) and r["r_team"][0, 1] == r["r_team"][0, 0]
    # match end: winner bonus on done, env restarts
    w.sim.t[0] = w.sim.end_t - 0.1
    r = w.step(st, gl)
    assert r["done"][0] and not np.isnan(r["final"][0]).any() and w.sim.t[0] < w.sim.end_t - 10


def test_baseline_control_inside_worker_overrides_policy_actions():
    w = _Worker(SimCfg(n_envs=2, sensors=False, seed=0, randomize=False, slots=("ours",) * 6))
    w.reset()
    w.set_ctrl(np.array([[False, True], [False, True]]))
    st, gl = np.zeros((2, 6, 3)), np.zeros((2, 6), int)
    p0 = w.sim.body.pose[..., :2].copy()
    for _ in range(30):
        r = w.step(st, gl)
    # displacement over the window (the final velocity alone is ~0 once a robot has arrived at its target)
    moved = np.linalg.norm(w.sim.body.pose[..., :2] - p0, axis=-1)
    assert moved[:, :3].max() < 0.05 < moved[:, 3:].max()


def test_parallel_env_matches_in_process_shapes():
    cfg = SimCfg(n_envs=2, sensors=False, seed=0)
    env = TeacherVecEnv(cfg, 2, in_process=True)
    o = env.reset()
    r = env.step(np.zeros((4, 6, 3)), np.zeros((4, 6), int))
    assert o["self"].shape[0] == 4 and r["r_team"].shape == (4, 6) and r["done"].shape == (4,)


# ------------------------------------------------------------------ audit
def test_audit_detects_injected_problems():
    sim = quiet_sim(n=2, seed=5)
    aud = Auditor(sim)
    st, gl = np.zeros((2, 6, 3)), np.zeros((2, 6), int)
    g = sim.game
    place(sim, 0, [0.5, 2.0], np.pi)                              # driving into the alliance wall forever
    place(sim, 1, [6.0, 1.0], 0.0)
    st[:, 0] = [-1.0, 0.0, 0.0]
    for _ in range(300):
        aud.observe(st, gl, sim.step(st, gl))
    aud._episode_end(np.array([True, True]))
    assert "stuck_wall" in aud.report()
    sim.gs.stock[1, 0] -= 3                                       # lose three coral
    sim.gs.score[0, 0] += 5                                       # score that no scoring event explains
    aud.last_check = np.zeros(2) - 10
    aud.observe(st, gl, sim.step(st, gl))
    assert "conservation" in aud.report() and "score_mismatch" in aud.report()
    assert np.allclose(state_points(g, sim.gs), 0)


def test_evicted_pool_snapshot_is_not_used_by_running_envs():
    lg = League(TrainCfg(pool_size=2), np.random.default_rng(0))
    a = lg.add({"w": torch.zeros(1)}, win=0.9)
    b = lg.add({"w": torch.zeros(1)}, win=0.2)
    opp = np.array([-1, 0, a, b, a])
    assert not stale_opponents(opp, lg).any()
    lg.add({"w": torch.zeros(1)}, win=0.5)                      # evicts the easiest snapshot (a)
    stale = stale_opponents(opp, lg)
    assert stale.tolist() == [False, False, True, False, True]


def test_micro_batches_give_the_same_update(tmp_path):
    """Gradient accumulation over micro-batches is a memory setting only: same parameters after an update."""
    import copy
    cfg = TrainCfg(n_procs=1, envs_per_proc=4, horizon=8, epochs=1, minibatch=128, micro=128, device="cpu",
                   out=str(tmp_path))
    tr = Trainer(cfg, TeacherVecEnv(SimCfg(n_envs=4, sensors=False, seed=0), 1, in_process=True))
    try:
        buf, acts, last, _ = tr.rollout()
        net0, opt0, rms0 = copy.deepcopy(tr.net.state_dict()), copy.deepcopy(tr.opt.state_dict()), copy.deepcopy(tr.ret_rms)
        out = []
        for micro in (128, 32):
            tr.net.load_state_dict(net0); tr.opt.load_state_dict(opt0); tr.ret_rms = copy.deepcopy(rms0)
            tr.cfg.micro = micro
            torch.manual_seed(0)
            st = tr.update(buf, acts, last)
            out.append(({k: v.clone() for k, v in tr.net.state_dict().items()}, st))
        for k in out[0][0]:
            assert torch.allclose(out[0][0][k], out[1][0][k], atol=1e-5), k
        assert abs(out[0][1]["clip_stick"] - out[1][1]["clip_stick"]) < 1e-6
    finally:
        tr.close()


def test_dagger_labels_come_from_the_baseline_while_the_policy_drives():
    """label_all: the worker returns the baseline's stick / goal for every robot as the label, but executes the
    given actions for the robots the baseline does not drive (bc.py DAgger rounds)."""
    w = _Worker(SimCfg(n_envs=2, sensors=False, seed=0, randomize=False, slots=("ours",) * 6))
    w.reset()
    w.set_ctrl(np.zeros((2, 2), bool), label_all=True)
    plan = w.bl.plan(w.sim)
    bst, _ = w.bl.act(w.sim, plan)
    r = w.step(np.zeros((2, 6, 3)), np.zeros((2, 6), int))
    assert np.allclose(r["act_stick"], bst, atol=1e-6) and np.abs(bst).max() > 0.1
    assert (r["act_goal"] == plan[0]).all()
    assert np.abs(w.sim.body.vel[..., :2]).max() < 0.05                    # the zero stick was executed, not the label


def test_kl_target_stops_actor_updates_but_not_the_critic(tmp_path):
    cfg = TrainCfg(n_procs=1, envs_per_proc=4, horizon=8, epochs=3, minibatch=64, device="cpu", out=str(tmp_path),
                   target_kl=1e-7, lr=1e-3)
    tr = Trainer(cfg, TeacherVecEnv(SimCfg(n_envs=4, sensors=False, seed=0), 1, in_process=True))
    try:
        critic0 = [p.detach().clone() for p in tr.critic_params]
        row = tr.train(1)
        assert row["actor_frac"] < 0.5                                        # stopped early
        assert any(not torch.equal(a, b) for a, b in zip(critic0, tr.critic_params))     # critic still trained
        tr.cfg.target_kl = 1e9
        assert tr.train(1)["actor_frac"] == 1.0
    finally:
        tr.close()


def test_explained_variance_uses_every_sample(tmp_path):
    """_value_of returns the critic value of every sample (it used to stop at 16384 and pad with zeros)."""
    cfg = TrainCfg(n_procs=1, envs_per_proc=4, horizon=8, device="cpu", out=str(tmp_path))
    tr = Trainer(cfg, TeacherVecEnv(SimCfg(n_envs=4, sensors=False, seed=0), 1, in_process=True))
    try:
        buf, acts, _, _ = tr.rollout()
        idx = acts["lm"].reshape(-1).nonzero()[:, 0]
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        data = {k: flat(v)[idx] for k, v in buf.items()}
        v = tr._value_of(data, idx)
        with torch.no_grad():
            ref = tr.net({k: d for k, d in data.items() if k != "decide"})[2]
        assert v.shape == ref.shape and torch.allclose(v, ref, atol=1e-5)
    finally:
        tr.close()


def test_independent_team_learner_drives_only_our_robot(tmp_path):
    """team "ours": the learner drives our robot of its alliance (slot 0 blue, 3 red) and, in self-play, the other
    alliance's one; partners are the styled baseline or pool snapshots; a baseline / pool opponent plays its whole
    alliance."""
    from ghosttrain.arena import SYMMETRIC
    cfg = TrainCfg(n_procs=1, envs_per_proc=4, horizon=8, epochs=1, minibatch=128, device="cpu", out=str(tmp_path),
                   p_partner_baseline=1.0)
    tr = Trainer(cfg, TeacherVecEnv(SimCfg(n_envs=4, sensors=False, seed=0, slots=SYMMETRIC), 1, in_process=True))
    try:
        assert tr._drivers(0, 0).tolist() == [-1, 0, 0, 0, 0, 0]
        assert tr._drivers(1, -1).tolist() == [-1, 0, 0, -1, 0, 0]
        tr.league.add(tr.net.state_dict(), win=0.5)
        k = max(tr.league.snaps)
        assert tr._drivers(1, k).tolist() == [k, k, k, -1, 0, 0]
        tr.cfg.p_partner_baseline = 0.0
        d = tr._drivers(0, 0)
        assert d[0] == -1 and (d[1:3] == k).all() and (d[3:] == 0).all()        # snapshot partners
        tr._assign(np.ones(4, bool))
        assert ((tr.drv == -1).sum(1) >= 1).all() and tr.env.local[0].styles   # baseline robots get styles
        assert (tr.env.local[0].ctrl_r == (tr.drv == 0)).all()
        row = tr.train(1)
        assert np.isfinite(row["pg_stick"]) and np.isfinite(row["kl"])
    finally:
        tr.close()


def test_baseline_styles_vary_per_robot_and_reset_to_defaults():
    from ghosttrain.baseline import Baseline
    sim = quiet_sim(n=3, seed=3)
    bl = Baseline(sim.game)
    rng = np.random.default_rng(0)
    bl.restyle(3, np.array([True, False, True]), rng)
    assert bl.kp.shape == (3, R) and (bl.kp[1] == bl.defaults["kp"]).all() and bl.kp[0].std() > 0
    lo, hi = Baseline.STYLE_RANGES["algae_reef_bias"]
    assert ((bl.algae_reef_bias[[0, 2]] >= lo) & (bl.algae_reef_bias[[0, 2]] <= hi)).all()
    st, gl = bl.act(sim)                                                      # array parameters run
    assert np.isfinite(st).all()
    bl.restyle(3, np.ones(3, bool), rng, on=False)
    assert (bl.kp == bl.defaults["kp"]).all() and bl.prefer_net.all()


def test_reef_report_is_late_sparse_and_sometimes_wrong():
    from ghosttrain.obs import ReefReport
    sim = quiet_sim(n=2, seed=4)
    rep = ReefReport(2, sim.game.l1_cap, np.random.default_rng(0), err_p=0.0)
    rep.step(sim)
    assert not rep.view["branch"].any()
    sim.gs.branch[:, 0, 2, 0, 2] = True                                        # an L4 coral appears now
    seen = []
    for t in range(25):
        rep.step(sim)
        seen.append(bool(rep.view["branch"][0, 0, 2, 0, 2]))
    first = seen.index(True)
    assert 4 <= first <= 20 and all(seen[first:])                              # 0.5-1 s late, at a 1 s update
    noisy = ReefReport(2, sim.game.l1_cap, np.random.default_rng(1), err_p=0.5)
    for _ in range(12):
        noisy.step(sim)
    assert (noisy.view["branch"] != sim.gs.branch).any()                       # errors show up
    rep.reset(np.array([True, False]))
    sim.gs.branch[:] = False
    rep.step(sim)
    assert not rep.view["branch"][0].any() and rep.view["branch"][1, 0, 2, 0, 2]   # a reset env restarts from the truth


def test_worker_actor_sees_the_reported_reef_and_the_critic_the_truth():
    w = _Worker(SimCfg(n_envs=1, sensors=False, seed=0, randomize=False, slots=("ours",) * 6))
    w.reset()
    w.sim.gs.branch[0, 0, 1, 1, 1] = True
    o = w._obs()
    enc_true = w.enc.encode(w.sim, w.goal)
    assert not np.array_equal(o["faces"], enc_true["faces"])                  # actor: not reported yet
    assert np.array_equal(o["glob"], enc_true["glob"])                         # critic: true state


def test_intent_collection_with_a_net_driving_some_robots(tmp_path):
    from ghosttrain.intent import RAW_FEATS, collect
    net = TeacherNet(14, 333)
    torch.save({"net": net.state_dict()}, tmp_path / "n.pt")
    X, Y, S, goals = collect(n_envs=2, steps=5, seed=0, net_path=str(tmp_path / "n.pt"), p_net=0.5)
    assert X.shape == (2 * 6 * 2 * 5, RAW_FEATS) and len(Y) == len(X) and Y.max() < len(goals)
