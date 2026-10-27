"""Live broadcast of training matches: 1280x720 at 30 fps, rendered only while someone is watching.

Layout (official-style, red/blue, no sponsor or FIRST artwork):
  top      1280x400  fixed field camera, scoreboard bar on top (blue left, red right), training step badge
  bottom   two 640x320 tiles: a random BLUE robot and a random RED robot POV (re-picked every 10 s)
  after a match: 10 s result card (LEAVE / CORAL / ALGAE / BARGE / PENALTY / WINS, score-based wins)

A separate robot page (/robot) shows one selected robot POV at 640x480 with the tags the detector sees
(same visibility rule as ghostsim.sensors) and the robot status.

Pacing: the match clock is wall-clock time while viewers are connected; the policy acts every 0.1 s tick and the
render interpolates robot poses between ticks, so 30 fps does not need 30 policy steps per second.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

TOP = (1280, 400)
TILE = (640, 320)
TILE_RENDER = (427, 320)          # robot POV rendered at the size it is shown (4:3)
CANVAS = (1280, 720)
FPS = 30
RESULT_S = 10.0
TILE_CHANGE_S = 10.0
BLUE = (200, 90, 25)            # BGR
RED = (40, 30, 200)
DARK = (22, 20, 18)
WHITE = (245, 245, 245)


def top_rig(game, width, height, vfov_deg=46.0, hfov_deg=74.0):
    fx = (width / 2) / np.tan(np.radians(hfov_deg / 2))
    fy = (height / 2) / np.tan(np.radians(vfov_deg / 2))
    return {"name": "spectator", "world": True, "robot_to_camera": [game.mid[0], -8.5, 11.0, 0.0, 41.0, 90.0],
            "capture": [width, height], "fx": fx, "fy": fy, "cx": width / 2, "cy": height / 2}


def robot_rig(size=(640, 480)):
    """Front camera of ours.json (640x480, fx 500), with intrinsics scaled to render at `size`."""
    from pathlib import Path as P
    import ghostsim
    prof = json.loads((P(ghostsim.__file__).parent.parent / "robots" / "ours.json").read_text())
    cam = dict(prof["cameras"][0])
    sx, sy = size[0] / cam["capture"][0], size[1] / cam["capture"][1]
    cam.update(capture=list(size), fx=cam["fx"] * sx, fy=cam["fy"] * sy, cx=cam["cx"] * sx, cy=cam["cy"] * sy)
    return cam


def tag_corners(game):
    """World corners (T,4,3) of every AprilTag, for overlays."""
    from ghostsim.geometry import IN
    from ghostsim.sensors import quat_to_mat
    ids = sorted(game.tags.keys())
    s = game.cfg["field"]["tags"]["black_square"] * IN
    out = []
    for i in ids:
        M = quat_to_mat(game.tags[i]["quat"])
        c = np.asarray(game.tags[i]["pos"], float)
        t1, t2 = M[:, 1], M[:, 2]
        out.append(np.stack([c + 0.5 * s * (sx * t1 + sy * t2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
    return np.array(out), ids


def project(points, cpos, crot, K, size):
    """points (...,3) world, camera position cpos (3,), rotation crot (3,3) field<-cam. Returns u,v,z (camera depth)."""
    d = points - cpos
    pc = d @ crot                                                   # camera frame: x fwd, y left, z up
    zc = pc[..., 0]
    xc, yc = -pc[..., 1], -pc[..., 2]
    fx, fy, cx, cy = K
    zz = np.maximum(zc, 1e-6)
    return fx * xc / zz + cx, fy * yc / zz + cy, zc


def draw_bar(img, text, x, y, w, h, color, scale=0.7, thick=2):
    import cv2
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, (x + (w - tw) // 2, y + (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def text(img, s, x, y, scale=0.55, color=WHITE, thick=1):
    import cv2
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


_BAR_CACHE = {}


def _bar_template(w, blue_name, red_name):
    key = (w, blue_name, red_name)
    if key not in _BAR_CACHE:
        import cv2
        base = np.zeros((62, w, 3), np.uint8)
        cv2.rectangle(base, (0, 0), (w, 62), DARK, -1)
        draw_bar(base, blue_name, 0, 0, 330, 30, BLUE, 0.55, 1)
        cv2.rectangle(base, (0, 30), (330, 62), (255, 255, 255), -1)
        draw_bar(base, red_name, w - 330, 0, 330, 30, RED, 0.55, 1)
        cv2.rectangle(base, (w - 330, 30), (w, 62), (255, 255, 255), -1)
        cv2.rectangle(base, (w // 2 - 70, 0), (w // 2 + 70, 62), (255, 255, 255), -1)
        _BAR_CACHE[key] = base
    return _BAR_CACHE[key]


def scoreboard(img, score, t_left, step_text, blue_name="BLUE  learner", red_name="RED  baseline"):
    """Top bar of the top view. score: (blue, red). Static parts are cached; numbers are drawn each frame."""
    import cv2
    w = img.shape[1]
    img[:62] = _bar_template(w, blue_name, red_name)
    cv2.putText(img, f"{score[0]:.0f}", (140, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, DARK, 2, cv2.LINE_8)
    cv2.putText(img, f"{score[1]:.0f}", (w - 190, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, DARK, 2, cv2.LINE_8)
    mm, ss = divmod(int(max(t_left, 0)), 60)
    cv2.putText(img, f"{mm}:{ss:02d}", (w // 2 - 46, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, DARK, 2, cv2.LINE_8)
    text(img, step_text, w // 2 - 120, 62 + 22, 0.6, WHITE, 1)


def _scoreboard_old(img, score, t_left, step_text, blue_name="BLUE  learner", red_name="RED  baseline"):
    import cv2
    w = img.shape[1]
    bar_h = 62
    cv2.rectangle(img, (0, 0), (w, bar_h), DARK, -1)
    # blue (left): name strip + score
    draw_bar(img, blue_name, 0, 0, 330, 30, BLUE, 0.55, 1)
    cv2.rectangle(img, (0, 30), (330, bar_h), (255, 255, 255), -1)
    cv2.putText(img, f"{score[0]:.0f}", (140, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, DARK, 2, cv2.LINE_AA)
    # red (right): name strip + score
    draw_bar(img, red_name, w - 330, 0, 330, 30, RED, 0.55, 1)
    cv2.rectangle(img, (w - 330, 30), (w, bar_h), (255, 255, 255), -1)
    cv2.putText(img, f"{score[1]:.0f}", (w - 190, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, DARK, 2, cv2.LINE_AA)
    # centre: match clock (counts down from the match length)
    mm, ss = divmod(int(max(t_left, 0)), 60)
    cv2.rectangle(img, (w // 2 - 70, 0), (w // 2 + 70, bar_h), (255, 255, 255), -1)
    cv2.putText(img, f"{mm}:{ss:02d}", (w // 2 - 46, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, DARK, 2, cv2.LINE_AA)
    text(img, step_text, w // 2 - 120, bar_h + 22, 0.6, WHITE, 1)


def result_card(scores, cats, wins, step_text, blue="BLUE", red="RED"):
    """1280x720 result card: per-category points table like the official post-match screen, no sponsor artwork."""
    import cv2
    img = np.full((720, 1280, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 0), (640, 720), BLUE, -1)
    cv2.rectangle(img, (640, 0), (1280, 720), RED, -1)
    head = "WINNER: BLUE" if scores[0] > scores[1] else ("WINNER: RED" if scores[1] > scores[0] else "TIE")
    cv2.putText(img, head, (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, f"{scores[0]:.0f}", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 5.0, WHITE, 8, cv2.LINE_AA)
    cv2.putText(img, f"{scores[1]:.0f}", (700, 330), cv2.FONT_HERSHEY_SIMPLEX, 5.0, WHITE, 8, cv2.LINE_AA)
    cv2.putText(img, blue, (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, red, (700, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    # category table in a white panel centred between the two colours
    rows = [("LEAVE", 0), ("CORAL", 1), ("ALGAE", 2), ("BARGE", 3), ("PENALTY", 4)]
    cv2.rectangle(img, (400, 370), (880, 650), (255, 255, 255), -1)
    for i, (label, k) in enumerate(rows):
        y = 420 + i * 46
        cv2.putText(img, f"{cats[0][k]:.0f}", (425, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, BLUE, 2, cv2.LINE_AA)
        cv2.putText(img, label, (560, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
        txt = f"{cats[1][k]:.0f}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        cv2.putText(img, txt, (855 - tw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, RED, 2, cv2.LINE_AA)
    cv2.putText(img, f"WINS  {wins[0]}", (60, 690), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, f"WINS  {wins[1]}", (700, 690), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, step_text, (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    return img


def tile(img_cam, label, color, status_lines):
    """Fit a 640x480 robot POV into a 640x320 tile (letterboxed) with a coloured label strip."""
    import cv2
    out = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    h = TILE[1] - 22
    w = int(img_cam.shape[1] * h / img_cam.shape[0])
    body = img_cam if img_cam.shape[:2] == (h, w) else cv2.resize(img_cam, (w, h))
    x0 = (TILE[0] - w) // 2
    out[22:, x0:x0 + w] = body
    cv2.rectangle(out, (0, 0), (TILE[0], 22), color, -1)
    cv2.putText(out, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    for i, s in enumerate(status_lines[:2]):
        text(out, s, 8, TILE[1] - 12 - 16 * i, 0.45, WHITE, 1)
    return out


class Broadcast:
    """Background match player. Renders only while `viewers` > 0."""

    def __init__(self, run: Path, opponent="baseline", seed=0, render_top=TOP, render_pov=(640, 480)):
        self.run = Path(run)
        self.opponent, self.seed = opponent, seed
        self.top_size, self.pov_size = render_top, render_pov
        self.viewers = {"bc": 0, "robot": 0}
        self.robot_sel = 0
        self.cond = threading.Condition()
        self.frames = {"bc": None, "robot": None}
        self.seq = {"bc": 0, "robot": 0}
        self.status = {}                              # robot id -> dict (for /api/robot)
        self.info = "starting"
        self.wins = [0, 0]
        self.stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.rng = np.random.default_rng(seed)
        self._goal_now = np.zeros((1, 6), int)
        self.ports = (48631, 48632, 48633)
        self.timing = {}                              # seconds spent per stage (diagnostics)
        self.infer_device = "cpu"
        self.match_s = 150.0                          # match length shown on the clock (game.json match.teleop_s + auto_s)

    # ------------------------------------------------------------------ public
    def start(self):
        self.thread.start()

    def _publish(self, key, img, quality=80):
        import cv2
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        with self.cond:
            self.frames[key], self.seq[key] = jpg, self.seq[key] + 1
            self.cond.notify_all()

    def training_step(self):
        """Latest training step from log.jsonl, re-read at most every 2 s (the file is small but read often)."""
        now = time.time()
        if now - getattr(self, "_step_t", 0.0) < 2.0:
            return getattr(self, "_step_v", 0)
        self._step_t, self._step_v = now, self._read_step()
        return self._step_v

    def _read_step(self):
        p = self.run / "log.jsonl"
        if not p.exists():
            return 0
        lines = p.read_text().splitlines()
        for line in reversed(lines[-20:]):
            try:
                return int(json.loads(line)["steps"])
            except (ValueError, KeyError):
                continue
        return 0

    # ------------------------------------------------------------------ internals
    def _loop(self):
        try:
            self._run()
        except Exception as e:                       # keep the dashboard alive, show the reason
            import traceback
            self.info = f"broadcast stopped: {e!r}"
            traceback.print_exc()

    def _run(self):
        clients = []
        try:
            self._run_inner(clients)
        finally:
            for rc in clients:                           # no orphan Godot processes after a stop or crash
                try:
                    rc.close()
                except Exception:
                    pass

    def _run_inner(self, clients):
        """Two threads per match: this one simulates (policy + physics, one tick per 0.1 s of match time) and the
        render thread draws at 30 fps by interpolating robot poses between the last two ticks. The render thread
        also shows the 10 s result card."""
        import torch
        torch.set_num_threads(2)                      # renderers and the sim share the cores
        from ghostrender import dr
        from ghostrender.client import RenderClient
        from ghostsim import SimCfg, load
        from ghostsim.robots import ALLIANCE, R
        from .net import TeacherNet
        from .obs import Encoder
        from .vecenv import _Worker

        game = load("2025")
        enc = Encoder(game)
        # policy inference on CPU: a 6-agent forward is ~3 ms, but small CUDA kernels queue behind the Godot
        # renderers that share the GPU (measured: tick 20 ms -> 70 ms with GPU inference)
        dev = torch.device(self.infer_device)
        net = TeacherNet(enc.n_goals, enc.G).to(dev)
        ckpt, mtime = self.run / "latest_net.pt", 0.0
        keys = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")
        top_cam = top_rig(game, *self.top_size)
        rc_top = RenderClient(port=self.ports[0])     # separate Godot instances: same pid would share a port
        clients.append(rc_top)
        rc_top.configure(game, [top_cam], n_envs=1, width=self.top_size[0], height=self.top_size[1], max_cams=1, msaa=2)
        rc_pov = RenderClient(port=self.ports[1])
        clients.append(rc_pov)
        pov_cam = robot_rig(TILE_RENDER)
        full_cam = robot_rig(self.pov_size)
        rc_pov.configure(game, [pov_cam], n_envs=1, width=TILE_RENDER[0], height=TILE_RENDER[1], max_cams=2, msaa=0)
        rc_robot = None                               # 640x480 client, started only when the robot page is open
        tags, tag_ids = tag_corners(game)
        rng = self.rng                                # all match and camera randomness comes from here
        while not self.stop:
            if sum(self.viewers.values()) == 0:
                time.sleep(0.5)
                continue
            while not ckpt.exists():
                self.info = f"waiting for {ckpt}"
                time.sleep(2)
            mt = ckpt.stat().st_mtime
            if mt != mtime:
                sd = torch.load(ckpt, map_location=dev, weights_only=False)
                net.load_state_dict(sd["net"] if "net" in sd else sd)
                net.eval()
                mtime = mt
            w = _Worker(SimCfg(n_envs=1, sensors=True, seed=int(rng.integers(1 << 30))))
            w.trunc_at[:] = 10 ** 9
            w.set_ctrl(np.array([[False, self.opponent == "baseline"]]))
            o = w.reset()
            sim = w.sim
            appearance = (dr.sample_appearance(rng, sim.prm.half[0], sim.prm.height[0], sim.ours["bumper_z_m"]),
                          {**dr.sample_dr(rng), "crowd_density": 0.6})
            rc_top.set_appearance(0, *appearance)
            rc_pov.set_appearance(0, *appearance)
            # shared snapshot: written by the sim thread at every tick, read by the render thread
            snap = {"prev": sim.render_state(), "now": sim.render_state(), "t_tick": time.time(), "score": sim.score[0].copy(),
                    "t": float(sim.t[0]), "goal": np.zeros((1, R), int), "done": False, "cats": sim.gs.cat[0].copy(),
                    "status_src": None}
            lock = threading.Lock()
            pick = {"blue": int(rng.integers(0, 3)), "red": int(rng.integers(3, 6)), "next": 0.0}
            match_over = threading.Event()
            card_done = threading.Event()
            state_holder = {"sim": sim, "w": w}

            def render_loop():
                nonlocal rc_robot
                import cv2
                frame_idx = 0
                canvas = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)     # reused every frame
                while not self.stop:
                    t_frame = time.time()
                    if sum(self.viewers.values()) == 0 and not snap["done"]:
                        time.sleep(0.3)                  # nobody watching: no rendering
                        continue
                    with lock:
                        prev, now, t_tick, score, t_sim, goal = (snap["prev"], snap["now"], snap["t_tick"],
                                                                 snap["score"], snap["t"], snap["goal"])
                        over = snap["done"]
                        cats = snap["cats"]
                    if over:
                        card = result_card(score, cats, self.wins, f"training step {self.training_step() / 1e6:.1f}M")
                        t_end = time.time() + RESULT_S
                        while time.time() < t_end and not self.stop:
                            self._publish("bc", card)
                            time.sleep(1.0 / FPS)
                        card_done.set()
                        return
                    alpha = min(1.0, max(0.0, (time.time() - t_tick) / 0.1))
                    state = dict(now)
                    state["robots"] = prev["robots"] + alpha * (now["robots"] - prev["robots"])   # pose interpolation
                    img_top, _ = rc_top.render(state, [(0, 0, 0)])
                    if time.time() >= pick["next"]:
                        pick.update(blue=int(rng.integers(0, 3)), red=int(rng.integers(3, 6)), next=time.time() + TILE_CHANGE_S)
                    img_pov, _ = rc_pov.render(state, [(0, pick["blue"], 0), (0, pick["red"], 0)])
                    top = cv2.cvtColor(img_top[0], cv2.COLOR_RGB2BGR)
                    scoreboard(top, score, self.match_s - t_sim, f"training step {self.training_step() / 1e6:.1f}M")
                    canvas[:TOP[1]] = top
                    canvas[TOP[1]:, :640] = tile(cv2.cvtColor(img_pov[0], cv2.COLOR_RGB2BGR), f"BLUE  robot {pick['blue'] + 1}",
                                                 BLUE, self._status_lines_from(goal, pick["blue"], snap))
                    canvas[TOP[1]:, 640:] = tile(cv2.cvtColor(img_pov[1], cv2.COLOR_RGB2BGR), f"RED  robot {pick['red'] + 1}",
                                                 RED, self._status_lines_from(goal, pick["red"], snap))
                    self._publish("bc", canvas)
                    frame_idx += 1
                    if self.viewers["robot"] > 0 and frame_idx % 2 == 0:       # robot page at 15 fps: the broadcast keeps 30
                        if rc_robot is None:
                            rc_robot = RenderClient(port=self.ports[2])
                            clients.append(rc_robot)
                            rc_robot.configure(game, [full_cam], n_envs=1, width=self.pov_size[0], height=self.pov_size[1],
                                               max_cams=1, msaa=0)
                            rc_robot.set_appearance(0, *appearance)
                        img_rob, _ = rc_robot.render(state, [(0, self.robot_sel, 0)])
                        cam = np.ascontiguousarray(img_rob[0][..., ::-1])
                        self._goal_now = goal
                        self._overlay_robot(cam, state_holder["sim"], self.robot_sel, tags, tag_ids)
                        self._publish("robot", cam)
                    time.sleep(max(0.0, 1.0 / FPS - (time.time() - t_frame)))

            renderer = threading.Thread(target=render_loop, daemon=True)
            renderer.start()
            # ---- sim thread: one policy tick per 0.1 s, paced by the wall clock while someone is watching
            res = {"done": np.array([False])}
            goal = np.zeros((1, R), int)
            t_wall = time.time()
            while not self.stop and sim.t[0] < sim.end_t - 1e-6:
                if sum(self.viewers.values()) == 0:
                    t_wall = time.time()
                    time.sleep(0.2)
                    continue
                now_w = time.time()
                target = min(sim.t[0] + min(now_w - t_wall, 1.0), sim.end_t)
                t_wall = now_w
                while sim.t[0] + 1e-9 < target and not self.stop:
                    _t1 = time.perf_counter()
                    t = {key: torch.from_numpy(np.ascontiguousarray(o[key])).to(dev).reshape(R, *o[key].shape[2:]) for key in keys}
                    t["glob"] = torch.from_numpy(o["glob"]).to(dev)[:, ALLIANCE].reshape(R, -1)
                    with torch.no_grad():
                        logits, mu, _ = net(t)
                        goal = torch.distributions.Categorical(logits=logits).sample().cpu().numpy().reshape(1, R)
                        stick = np.clip((mu + net.log_std.exp() * torch.randn_like(mu)).cpu().numpy().reshape(1, R, 3), -1, 1)
                    res = w.step(stick, goal)
                    o = res["obs"]
                    with lock:
                        snap["prev"], snap["now"] = snap["now"], sim.render_state()
                        snap["t_tick"], snap["t"], snap["goal"] = time.time(), float(sim.t[0]), goal.copy()
                        snap["score"], snap["cats"] = sim.score[0].copy(), sim.gs.cat[0].copy()
                    self.timing["tick"] = self.timing.get("tick", 0.0) + time.perf_counter() - _t1
                    for rr in range(R):                  # status once per tick (not per poll)
                        self.status[rr] = self._robot_status(sim, rr, goal)
                    if res["done"].any():
                        break
                if res["done"].any():
                    break
                time.sleep(0.005)
            if self.stop:
                break
            # ---- match over: winner count, then wait for the render thread to show the 10 s card
            scores = sim.score[0].copy()
            if scores[0] != scores[1]:
                self.wins[0 if scores[0] > scores[1] else 1] += 1
            with lock:
                snap["done"] = True
            card_done.wait(timeout=RESULT_S + 5)
            renderer.join(timeout=2)
            self.info = f"last match {scores[0]:.0f}-{scores[1]:.0f}"

    def _status_lines_from(self, goal, r, snap):
        g = self.g_names[goal[0, r]] if hasattr(self, "g_names") else str(goal[0, r])
        st = self.status.get(r, {})
        hold = "coral" if st.get("hold_coral") else ("algae" if st.get("hold_algae") else "-")
        return [f"goal {st.get('goal', g)}   hold {hold}",
                f"speed {st.get('speed_mps', 0.0):.1f} m/s   batt {st.get('battery_v', 0.0):.1f} V"]

    # ------------------------------------------------------------------ status / overlays
    def _status_lines(self, sim, r, goal):
        from ghostsim.robots import ALLIANCE  # noqa: F401
        gs = sim.gs
        g = sim.game.goals[goal[0, r]]
        hold = "coral" if gs.hold_coral[0, r] else ("algae" if gs.hold_algae[0, r] else "-")
        speed = float(np.hypot(*sim.body.vel[0, r, :2]))
        return [f"goal {g}   hold {hold}", f"speed {speed:.1f} m/s   batt {sim.body.vbatt[0, r]:.1f} V"]

    def _robot_status(self, sim, r, goal):
        gs, b = sim.gs, sim.body
        d = {"robot": r + 1, "alliance": "blue" if r < 3 else "red",
             "goal": sim.game.goals[goal[0, r]], "hold_coral": bool(gs.hold_coral[0, r]),
             "hold_algae": bool(gs.hold_algae[0, r]), "speed_mps": float(np.hypot(*b.vel[0, r, :2])),
             "battery_v": float(b.vbatt[0, r]), "pose": [float(x) for x in b.pose[0, r]],
             "climbed": bool(gs.climbed[0, r]), "t_s": float(sim.t[0])}
        if sim.sens is not None:
            est = sim.sens.est[0, r]
            d["est_error_m"] = float(np.hypot(*(est[:2] - b.pose[0, r, :2])))
            d["est_error_deg"] = float(np.degrees(abs(np.arctan2(np.sin(est[2] - b.pose[0, r, 2]),
                                                                  np.cos(est[2] - b.pose[0, r, 2])))))
        return d

    def _overlay_robot(self, cam, sim, r, tags, tag_ids):
        """Draw the tags the detector sees (same visibility as ghostsim.sensors) and a status panel."""
        import cv2
        if sim.sens is None:
            return
        vis, _ = sim.sens.visible_mask(sim.body, np.array([0]), np.array([r]))
        cams = sim.sens.cams
        cpos, crot = cams.world_poses(sim.body.pose[0, r][None])
        K = cams.K[0]
        seen = np.flatnonzero(vis[0, 0])
        for ti in range(tags.shape[0]):
            u, v, z = project(tags[ti], cpos[0, 0], crot[0, 0], K, None)
            if np.any(z <= 0.05):
                continue
            pts = np.stack([u, v], -1).astype(np.int32)
            color = (80, 230, 80) if ti in seen else (120, 120, 120)
            if ti in seen:
                cv2.polylines(cam, [pts], True, color, 2, cv2.LINE_AA)
                cv2.putText(cam, str(tag_ids[ti]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        st = self._robot_status(sim, r, self._goal_now)
        lines = [f"robot {r + 1} ({'blue' if r < 3 else 'red'})  t={sim.t[0]:.1f}s",
                 f"goal {st['goal']}",
                 f"hold coral {st['hold_coral']}  algae {st['hold_algae']}",
                 f"speed {st['speed_mps']:.2f} m/s  batt {st['battery_v']:.1f} V",
                 f"tags seen {len(seen)}  est err {st.get('est_error_m', 0):.2f} m",
                 f"score {sim.score[0, 0 if r < 3 else 1]:.0f}"]
        for i, s in enumerate(lines):
            text(cam, s, 10, 24 + 22 * i, 0.55, WHITE, 1)
