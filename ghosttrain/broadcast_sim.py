"""Match simulation for the broadcast, in its own process.

Why a process: the policy tick (~26 ms of Python) and the renderer thread both need the GIL; in one process the
render thread waited on the GIL and capped the broadcast near 29 fps. Here this process only simulates and sends
snapshots over a pipe; the parent process renders.

Protocol (child -> parent, over a Pipe):
  {"kind": "match", "appearance": (robots, dr)}        new match started
  {"kind": "tick", "snap": {...}}                      after every policy tick (10 per match second)
  {"kind": "over", "score": (2,), "cats": (2, 5)}      match finished; the child then waits RESULT_S seconds
Parent -> child: `watching` (multiprocessing.Event) pauses the match clock; `sel` (Value) selects the robot POV.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

TICK_S = 0.1
RESULT_S = 10.0


def robot_status(sim, r, goal):
    gs, b = sim.gs, sim.body
    d = {"robot": r + 1, "alliance": "blue" if r < 3 else "red",
         "goal": sim.game.goals[goal[0, r]], "hold_coral": bool(gs.hold_coral[0, r]),
         "hold_algae": bool(gs.hold_algae[0, r]), "speed_mps": float(np.hypot(*b.vel[0, r, :2])),
         "battery_v": float(b.vbatt[0, r]), "pose": [float(x) for x in b.pose[0, r]],
         "climbed": bool(gs.climbed[0, r]), "t_s": float(sim.t[0])}
    if sim.sens is not None:
        est = sim.sens.est[0, r]
        err = est[:2] - b.pose[0, r, :2]
        d["est_error_m"] = float(np.hypot(*err))
        dth = est[2] - b.pose[0, r, 2]
        d["est_error_deg"] = float(np.degrees(abs(np.arctan2(np.sin(dth), np.cos(dth)))))
    return d


def mech_inputs(sim):
    """JointTracker.update arguments for env 0: the goal the rules executed, dwell progress, held pieces, hanging,
    and whether the nearest own reef face holds its algae high."""
    gs, g = sim.gs, sim.game
    from ghostsim.robots import ALLIANCE
    prog = np.where(gs.timer[0] > 0, gs.timer[0] / np.maximum(gs.need[0], 1e-6), 0.0)
    d = np.linalg.norm(g.reef_fc[ALLIANCE] - sim.body.pose[0, :, None, :2], axis=-1)        # (R,6)
    high = g.algae_high[np.argmin(d, -1)]
    return gs.timer_goal[0], np.clip(prog, 0.0, 1.0), gs.hold_coral[0], gs.hold_algae[0], gs.climbed[0], high


def pov_data(sim, r):
    """Camera pose and the tags the detector sees for robot r (same visibility rule as ghostsim.sensors)."""
    if sim.sens is None:
        return None
    vis, _ = sim.sens.visible_mask(sim.body, np.array([0]), np.array([r]))
    cams = sim.sens.cams
    cpos, crot = cams.world_poses(sim.body.pose[0, r][None])
    return {"seen": [int(i) for i in np.flatnonzero(vis[0, 0])], "cam_pos": cpos[0, 0].tolist(),
            "cam_rot": crot[0, 0].tolist(), "K": [float(x) for x in cams.K[0]]}


def run_matches(run, opponent, seed, conn, watching, sel, stop):
    """Entry point of the simulation process (must be importable for multiprocessing 'spawn')."""
    import torch
    torch.set_num_threads(2)
    from ghostrender import dr
    from ghostsim import SimCfg, load
    from ghostsim.robots import ALLIANCE, R
    from ghostsim.mechanism_state import JointTracker
    from .net import TeacherNet
    from .obs import Encoder
    from .vecenv import _Worker

    run = Path(run)
    game = load("2025")
    enc = Encoder(game)
    dev = torch.device("cpu")          # a 6-agent forward is ~3 ms on CPU; GPU kernels queue behind the renderers
    net = TeacherNet(enc.n_goals, enc.G).to(dev)
    ckpt, mtime = run / "latest_net.pt", 0.0
    keys = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")
    rng = np.random.default_rng(seed)
    while not stop.is_set():
        if not watching.is_set():
            time.sleep(0.2)
            continue
        if not ckpt.exists():
            time.sleep(2)
            continue
        mt = ckpt.stat().st_mtime
        if mt != mtime:
            sd = torch.load(ckpt, map_location=dev, weights_only=False)
            net.load_state_dict(sd["net"] if "net" in sd else sd)
            net.eval()
            mtime = mt
        w = _Worker(SimCfg(n_envs=1, sensors=True, seed=int(rng.integers(1 << 30))))
        w.trunc_at[:] = 10 ** 9
        w.set_ctrl(np.array([[False, opponent == "baseline"]]))
        o = w.reset()
        sim = w.sim
        appearance = (dr.sample_appearance(rng, sim.prm.half[0], sim.prm.height[0], sim.ours["bumper_z_m"]),
                      {**dr.sample_dr(rng), "crowd_density": 0.6})
        conn.send({"kind": "match", "appearance": appearance})
        goal = np.zeros((1, R), int)
        res = {"done": np.array([False])}
        joints = JointTracker(game.goals, R)                  # mechanism pose for the renderer (not used by training)
        frames = (2 * (sim.prm.half[0, :, 0] - sim.ours["bumper_t_in"] * 0.0254)).tolist()
        # match clock target: advances by the wall time that passed while watching (never by sim.t itself;
        # that re-counted each tick's overshoot and ran the match ~3x faster than real time)
        t_wall = time.time()
        sim_target = float(sim.t[0])
        while not stop.is_set() and sim.t[0] < sim.end_t - 1e-6:
            if not watching.is_set():
                t_wall = time.time()
                sim_target = float(sim.t[0])
                time.sleep(0.2)
                continue
            now = time.time()
            sim_target = min(sim_target + min(now - t_wall, 1.0), sim.end_t)
            target = sim_target
            t_wall = now
            while sim.t[0] + 1e-9 < target and not stop.is_set():
                t = {key: torch.from_numpy(np.ascontiguousarray(o[key])).reshape(R, *o[key].shape[2:]) for key in keys}
                t["glob"] = torch.from_numpy(o["glob"])[:, ALLIANCE].reshape(R, -1)
                with torch.no_grad():
                    logits, mu, _ = net(t)
                    goal = torch.distributions.Categorical(logits=logits).sample().numpy().reshape(1, R)
                    stick = np.clip((mu + net.log_std.exp() * torch.randn_like(mu)).numpy().reshape(1, R, 3), -1, 1)
                res = w.step(stick, goal)
                o = res["obs"]
                joints.update(*mech_inputs(sim), dt=TICK_S)
                r_sel = int(sel.value)
                state = sim.render_state()
                for key, k0 in (("coral", game.kc), ("algae", game.ka)):      # held pieces: drawn by the mechanism
                    state[key][:, k0:k0 + R, -1] = 0.0
                snap = {"t": float(sim.t[0]), "state": state, "score": sim.score[0].copy(),
                        "cats": sim.gs.cat[0].copy(), "goal": goal[0].copy(),
                        "status": [robot_status(sim, r, goal) for r in range(R)],
                        "pov": pov_data(sim, r_sel), "sel": r_sel, "wall": time.time(),
                        "mech": {**joints.state(), "frames": frames}}
                conn.send({"kind": "tick", "snap": snap})
                if res["done"].any():
                    break
            if res["done"].any():
                break
            time.sleep(0.005)
        if stop.is_set():
            break
        # the worker auto-resets on the final tick: take the score and categories saved before the reset
        fin = res["final"][0]
        fin_cat = res["final_cat"][0]
        if np.isnan(fin).any():
            fin, fin_cat = sim.score[0].copy(), sim.gs.cat[0].copy()
        conn.send({"kind": "over", "score": np.asarray(fin, float), "cats": np.asarray(fin_cat, float)})
        t_end = time.time() + RESULT_S                 # the parent shows the result card for this long
        while time.time() < t_end and not stop.is_set():
            time.sleep(0.2)
