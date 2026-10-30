"""Live broadcast of training matches: 1920x1080 at 30 fps (15 while the run is training), rendered only while
someone is watching (pages stop their stream when hidden; the curves page plays it on demand).

Layout of the video (no graphics in it: the scoreboard, tile labels and the result card are an HTML overlay on the
/live page, fed by Broadcast.live / live_state(), so they stay sharp at any size and cost no rendering):
  top      1920x720  field camera behind the scoring table, rendered 1:1, MSAA 4x + FXAA
  bottom   two 960x360 follow cameras (tripods outside the field, pan and zoom) on a BLUE and a RED robot,
           re-picked every 10 s, 1:1
The live view has no crowd, a dark background and fixed lighting (training cameras keep their randomisation).

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

import cv2
import numpy as np

from .broadcast_sim import RESULT_S, TICK_S

CANVAS = (1920, 1080)
TOP = (1920, 720)
TILE = (960, 360)
LITE = {"canvas": (1280, 720), "top": (1280, 480), "tile": (640, 240), "fps": 20, "fps_training": 10, "msaa": 0,
        "visual": "auto", "jpeg": 80}          # --lite: low GPU cost (field.glb, no MSAA, 720p, fewer frames)
FPS = 30
FPS_TRAINING = 15         # while the run is training (log.jsonl written in the last TRAINING_IDLE_S): the live view
TRAINING_IDLE_S = 120.0   # shares the GPU with the learner (at 30 fps the training ran at ~45 % of its speed)
RESULT_S = 10.0
TILE_CHANGE_S = 10.0
GOAL_LABEL = {"idle": "IDLE", "coral_station": "CORAL STATION", "coral_floor": "CORAL FLOOR", "algae_floor": "ALGAE FLOOR",
              "algae_reef": "ALGAE REEF", "score_L1": "SCORE L1", "score_L2": "SCORE L2", "score_L3": "SCORE L3",
              "score_L4": "SCORE L4", "processor": "PROCESSOR", "net": "NET", "climb": "CLIMB", "defend": "DEFEND",
              "align_score": "ALIGN"}


def top_rig(game, width, height, hfov_deg=54.0):
    """Field camera high behind the scoring table: the near field edge spans ~90 % of the width, the far wall clears
    the scoreboard overlay (framing searched numerically for 1920x720)."""
    fx = (width / 2) / np.tan(np.radians(hfov_deg / 2))
    return {"name": "spectator", "world": True, "robot_to_camera": [game.mid[0], -14.25, 13.0, 0.0, 32.0, 90.0],
            "capture": [width, height], "fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def look_rig(eye, target, hfov_deg, size=TILE):
    """World camera at eye looking at target (field frame, metres)."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw, pitch = np.degrees(np.arctan2(d[1], d[0])), np.degrees(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    fx = (size[0] / 2) / np.tan(np.radians(hfov_deg / 2))
    return {"name": "follow", "world": True, "robot_to_camera": [*map(float, eye), 0.0, float(pitch), float(yaw)],
            "capture": list(size), "fx": fx, "fy": fx, "cx": size[0] / 2, "cy": size[1] / 2}


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("ghostsim", "ghosttrain", "game", "robots")


def code_signature(root=ROOT):
    """Newest modification time of the simulation / policy code and data. The live simulation runs in a child
    process: when this changes, the child is restarted after the current match (new code, no dashboard restart)."""
    m = 0.0
    for d in CODE_DIRS:
        for p in (root / d).rglob("*"):
            if p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
                m = max(m, p.stat().st_mtime)
    return m


def free_ports(base: int, n: int = 3):
    """n local TCP ports from `base` up that nobody listens on (renderer ports of this dashboard: a second dashboard
    must not talk to the first one's Godot instances)."""
    import socket
    out, p = [], base
    while len(out) < n:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                out.append(p)
            except OSError:
                pass
        p += 1
    return tuple(out)


def active_run(root: Path, follow: bool) -> Path:
    """The run to show: `root` itself, or with follow the run directory under `root` whose log.jsonl was written
    last (a new training run takes over the live view by itself)."""
    root = Path(root)
    if not follow:
        return root
    best, t = root, -1.0
    for d in root.iterdir() if root.is_dir() else []:
        p = d / "log.jsonl"
        if d.is_dir() and p.exists() and p.stat().st_mtime > t:
            best, t = d, p.stat().st_mtime
    return best


class FollowCam:
    """Broadcast camera on a tripod outside the field (scoring-table side, raised) that pans and zooms to keep one
    robot framed: the aim point follows the robot with a lag, the field of view shows ~`span` metres around it."""

    def __init__(self, eye, span=3.6, tau=0.35, hfov=(12.0, 55.0), size=TILE):
        self.eye, self.span, self.tau, self.hfov = np.asarray(eye, float), span, tau, hfov
        self.aim, self.robot, self.size = None, None, size

    def rig(self, robot, xy, dt):
        tgt = np.array([xy[0], xy[1], 0.55])
        if self.aim is None or robot != self.robot:                 # new robot: cut, no swing across the field
            self.aim, self.robot = tgt, robot
        else:
            self.aim = self.aim + (tgt - self.aim) * (1.0 - np.exp(-dt / self.tau))
        dist = np.linalg.norm(self.aim - self.eye)
        h = float(np.clip(np.degrees(2 * np.arctan(self.span / 2 / dist)), *self.hfov))
        return look_rig(self.eye, self.aim, h, self.size)


def follow_cams(game):
    """Blue tile camera at the blue end of the scoring-table side, red at the red end (4.2 m up, 2.6 m off the wall)."""
    L = game.length
    return FollowCam([L * 0.30, -2.6, 4.2]), FollowCam([L * 0.70, -2.6, 4.2])


def live_appearance(dr_values):
    """Broadcast look: no crowd, dark arena, fixed light (keeps the rest of the domain-randomisation values)."""
    return {**dr_values, "crowd_density": 0.0, "bg_rgb": [0.035, 0.04, 0.055], "sun_energy": 1.15, "sun_elev_deg": 62.0,
            "sun_az_deg": 35.0, "sun_rgb": [1.0, 0.98, 0.95], "ambient_energy": 0.85, "ambient_rgb": [0.95, 0.96, 1.0],
            "carpet_rgb": [0.27, 0.27, 0.28], "carpet_rough": 0.9}


def live_state(snap, match_s, step, wins, pick, opponent, card=None, match_no=0, auto_s=15.0, endgame_s=20.0):
    """JSON for the HTML overlay (/api/live): clock, phase, scores, category points, the robots in the tiles, and the
    result card while it is shown."""
    out = {"match": match_no, "step": int(step), "wins": list(wins), "red_name": "SAME NET" if opponent == "self" else "BASELINE",
           "blue_name": "LEARNER", "card": None}
    if card is not None:
        scores, cats = card[0], card[1]
        out["card"] = {"scores": [float(x) for x in scores], "cats": np.asarray(cats, float).round(1).tolist(),
                       "winner": 0 if scores[0] > scores[1] else (1 if scores[1] > scores[0] else -1)}
    if snap is None:
        return out
    t = float(snap["t"])
    t_left = max(match_s - t, 0.0)
    out.update(t_left=t_left, phase="AUTO" if t < auto_s else ("ENDGAME" if t_left <= endgame_s else "TELEOP"),
               score=[float(x) for x in snap["score"]], cats=np.asarray(snap["cats"], float).round(1).tolist())
    tiles = []
    for r in (pick["blue"], pick["red"]):
        st = snap["status"][r]
        hold = "CORAL + ALGAE" if st["hold_coral"] and st["hold_algae"] else (
            "CORAL" if st["hold_coral"] else ("ALGAE" if st["hold_algae"] else ""))
        tiles.append({"robot": r + 1, "alliance": "blue" if r < 3 else "red", "ours": r == 0,
                      "goal": "HANGING" if st["climbed"] else GOAL_LABEL.get(st["goal"], st["goal"].upper()),
                      "hold": hold, "speed": round(float(st["speed_mps"]), 1)})
    out["tiles"] = tiles
    return out


WHITE = (245, 245, 245)


def text(img, s, x, y, scale=0.55, color=WHITE, thick=1):
    """Plain status text on the robot page."""
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


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


class Broadcast:
    """Parent process: renders the broadcast and the robot page from snapshots sent by the simulation process.

    Rendering runs at 30 fps only while a page is open; the simulation process pauses while nobody watches.
    """

    def __init__(self, run: Path, opponent="self", seed=0, render_top=TOP, render_pov=(640, 480), team="ours",
                 follow=False, lite=False, http_port=8777):
        self.root, self.follow = Path(run), follow
        self.opponent, self.seed, self.team = opponent, seed, team
        self.lite = lite
        self.canvas_size = LITE["canvas"] if lite else CANVAS
        self.top_size = LITE["top"] if lite else render_top
        self.tile_size = LITE["tile"] if lite else TILE
        self.pov_size = render_pov
        self.viewers = {"bc": 0, "robot": 0}
        self.robot_sel = 0
        self.cond = threading.Condition()
        self.frames = {"bc": None, "robot": None}
        self.seq = {"bc": 0, "robot": 0}
        self._pending = {}
        self.status = {}                              # robot id -> dict (for /api/robot)
        self.info = "starting"
        self.wins = [0, 0]
        self.matches = 0
        self.live = {"card": None, "match": 0, "wins": [0, 0], "step": 0}     # /api/live (HTML overlay)
        self.stop = False
        self.timing = {}                              # seconds per stage (diagnostics)
        self.ports = free_ports(48631 + 3 * (http_port - 8777))       # per dashboard (port 8777: 48631-48633)
        self.infer_device = "cpu"
        self.match_s = 150.0                          # auto 15 s + teleop 135 s (game.json)
        self.rng = np.random.default_rng(seed)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    @property
    def run(self) -> Path:
        """The run shown (re-resolved every 10 s with follow)."""
        now = time.time()
        if now - getattr(self, "_run_t", -1e9) > 10.0:
            self._run_t, self._run_v = now, active_run(self.root, self.follow)
        return self._run_v

    # ------------------------------------------------------------------ public
    def start(self):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                 # no fork: the parent holds Godot sockets and threads
        self._ctx = ctx
        self._watching, self._stop_evt = ctx.Event(), ctx.Event()
        self._sel = ctx.Value("i", 0)
        self._spawn_child()
        self.thread.start()

    def _spawn_child(self, delay=0.0):
        """(Re)start the simulation process; a fresh interpreter imports the code as it is on disk now. delay: wait
        before its first match (the result card of the last one is still on screen)."""
        from .broadcast_sim import run_matches
        self._sig = code_signature()
        self._sig_t = time.time()
        self._conn, child_conn = self._ctx.Pipe(duplex=False)
        self._child = self._ctx.Process(target=run_matches, daemon=True,
                                        args=(str(self.root), self.opponent, self.seed + self.matches, child_conn,
                                              self._watching, self._sel, self._stop_evt, self.team, self.follow, delay))
        self._child.start()

    def _maybe_reload(self, delay=0.0):
        """Between matches (or while nobody watches): restart the simulation process if its code or data changed on
        disk. The HTTP server, the renderers and the stream keep running."""
        self._sig_t = time.time()
        if code_signature() == self._sig:
            return False
        self._child.terminate()
        self._child.join(timeout=5)
        self._spawn_child(delay)
        self.info = "code changed on disk: simulation reloaded"
        return True

    def shutdown(self):
        """Stop the simulation process and the renderer; the renderer closes its Godot clients on the way out."""
        self.stop = True
        if hasattr(self, "_stop_evt"):
            self._stop_evt.set()
        if hasattr(self, "_child"):
            self._child.join(timeout=5)
            if self._child.is_alive():
                self._child.terminate()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=10)

    def _publish(self, key, img, quality=80):
        """Hand a frame to the JPEG thread (keeps only the newest frame per key; never blocks the renderer)."""
        with self.cond:
            self._pending[key] = (img.copy(), quality)
            self.cond.notify_all()
        if not hasattr(self, "_enc_thread") or not self._enc_thread.is_alive():
            self._enc_thread = threading.Thread(target=self._encode_loop, daemon=True)
            self._enc_thread.start()

    def _encode_loop(self):
        import cv2
        while not self.stop:
            with self.cond:
                self.cond.wait_for(lambda: self._pending or self.stop, timeout=1.0)
                items, self._pending = self._pending, {}
            for key, (img, q) in items.items():
                jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
                with self.cond:
                    self.frames[key], self.seq[key] = jpg, self.seq[key] + 1
                    self.cond.notify_all()

    def fps(self):
        """Frame rate: FPS, or FPS_TRAINING while log.jsonl is being written (checked at most every 5 s)."""
        now = time.time()
        if now - getattr(self, "_fps_t", 0.0) >= 5.0:
            p = self.run / "log.jsonl"
            busy = p.exists() and now - p.stat().st_mtime < TRAINING_IDLE_S
            hi, lo = (LITE["fps"], LITE["fps_training"]) if self.lite else (FPS, FPS_TRAINING)
            self._fps_t, self._fps_v = now, lo if busy else hi
        return self._fps_v

    def training_step(self):
        """Latest training step from log.jsonl, re-read at most every 2 s."""
        now = time.time()
        if now - getattr(self, "_step_t", 0.0) < 2.0:
            return getattr(self, "_step_v", 0)
        self._step_t, self._step_v = now, self._read_step()
        return self._step_v

    def _read_step(self):
        p = self.run / "log.jsonl"
        if not p.exists():
            return 0
        for line in reversed(p.read_text().splitlines()[-20:]):
            try:
                return int(json.loads(line)["steps"])
            except (ValueError, KeyError):
                continue
        return 0

    # ------------------------------------------------------------------ parent render loop
    def _loop(self):
        import sys
        sys.setswitchinterval(0.0005)
        # A renderer can die (GPU device lost, observed with the training run on the same GPU). Restart the renderers
        # and keep going instead of leaving the live view dead until someone notices.
        restarts = 0
        try:
            while not self.stop:
                try:
                    self._run()
                    break
                except (ConnectionError, RuntimeError, TimeoutError, OSError) as e:
                    restarts += 1
                    self.info = f"renderer restarted ({restarts}): {e!r}"
                    print(self.info, flush=True)
                    time.sleep(2.0)
                except Exception as e:                # unexpected: show the reason and stop
                    import traceback
                    self.info = f"broadcast stopped: {e!r}"
                    traceback.print_exc()
                    break
        finally:
            self.shutdown()

    def _run(self):
        from ghostrender.client import RenderClient
        from ghostsim.mechanism_state import mech_slots
        from ghostsim import load

        game = load("2025")
        self._game = game
        tags, tag_ids = tag_corners(game)
        rc_top = RenderClient(port=self.ports[0])     # separate Godot instances: same pid would share a port
        msaa, visual = (LITE["msaa"], LITE["visual"]) if self.lite else (1, "hi")
        TOP_, TILE_ = self.top_size, self.tile_size
        rc_top.configure(game, [top_rig(game, *TOP_)], n_envs=1, width=TOP_[0], height=TOP_[1], max_cams=1,
                         msaa=msaa, fxaa=True, visual=visual, robot="auto")
        rc_pov = RenderClient(port=self.ports[1])                 # the two follow cameras
        follow = follow_cams(game)
        for f in follow:
            f.size = TILE_
        rc_pov.configure(game, [look_rig(f.eye, [game.mid[0], game.mid[1], 0.5], 40.0, TILE_) for f in follow], n_envs=1,
                         width=TILE_[0], height=TILE_[1], max_cams=2, msaa=msaa, fxaa=True, visual=visual, robot="auto")
        cad_robot = rc_top.robot == "cad" and rc_pov.robot == "cad"     # robot.glb bodies, else the box model
        t_prev = time.time()
        rc_robot = None                               # 640x480 client, started only when the robot page is open
        full_cam = robot_rig(self.pov_size)
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), np.uint8)
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(2)
        clients = [rc_top, rc_pov]
        try:
            appearance = None                         # set by each new match (robot page client uses it too)
            latest = prev = None                      # last two tick snapshots (for pose interpolation)
            t_recv = 0.0
            card = None                               # (scores, cats, until) while the result card is shown
            pick = {"blue": 0, "red": 3, "next": 0.0}
            frame_idx = 0
            while not self.stop:
                t_frame = time.time()
                self._watching.set() if sum(self.viewers.values()) else self._watching.clear()
                self._sel.value = int(self.robot_sel)
                # ---- drain the simulation process
                while self._conn.poll():
                    msg = self._conn.recv()
                    if msg["kind"] == "match":
                        appearance = msg["appearance"]
                        self.matches += 1
                        rc_top.set_appearance(0, *appearance)
                        rc_pov.set_appearance(0, *appearance)
                        if rc_robot is not None:
                            rc_robot.set_appearance(0, *appearance)
                        latest = prev = None
                        card = None
                    elif msg["kind"] == "tick":
                        prev, latest, t_recv = latest, msg["snap"], time.time()
                        self.status = {r: d for r, d in enumerate(latest["status"])}
                    elif msg["kind"] == "info":
                        self.info = msg["msg"]
                    elif msg["kind"] == "over":
                        scores, cats = msg["score"], msg["cats"]
                        if scores[0] != scores[1]:
                            self.wins[0 if scores[0] > scores[1] else 1] += 1
                        card = (scores, cats, time.time() + RESULT_S)
                        self.info = f"last match {scores[0]:.0f}-{scores[1]:.0f}"
                        if self._maybe_reload(delay=RESULT_S):         # new code: next match runs on it
                            break
                if sum(self.viewers.values()) == 0:
                    if time.time() - self._sig_t > 30.0:
                        self._maybe_reload()
                    time.sleep(0.2)
                    continue
                if card is not None:
                    if time.time() < card[2]:                     # result card: an overlay on the page, no 3D needed
                        self.live = live_state(None, self.match_s, self.training_step(), self.wins, pick, self.opponent,
                                               card, self.matches)
                        time.sleep(0.1)
                        continue
                    card = None
                if latest is None:
                    time.sleep(0.02)
                    continue
                # ---- frame: interpolate robot poses between the last two ticks
                alpha = min(1.0, max(0.0, (time.time() - t_recv) / TICK_S))
                state = dict(latest["state"])
                if prev is not None:
                    state["robots"] = prev["state"]["robots"] + alpha * (latest["state"]["robots"] - prev["state"]["robots"])
                mq = latest["mech"]["q"] if prev is None else prev["mech"]["q"] + alpha * (latest["mech"]["q"] - prev["mech"]["q"])
                state["mech"] = mech_slots(mq, latest["mech"]["hold"], latest["mech"]["funnel"], latest["mech"]["frames"],
                                           cad=cad_robot)
                if time.time() >= pick["next"]:
                    pick.update(blue=int(self.rng.integers(0, 3)), red=int(self.rng.integers(3, 6)),
                                next=time.time() + TILE_CHANGE_S)
                now = time.time()
                dt, t_prev = min(now - t_prev, 0.2), now
                rob = state["robots"][0]
                rigs = [follow[0].rig(pick["blue"], rob[pick["blue"], :2], dt), follow[1].rig(pick["red"], rob[pick["red"], :2], dt)]
                # the two Godot instances render in parallel (the wait is mostly readback and transfer)
                f_top = pool.submit(rc_top.render, state, [(0, 0, 0)])
                f_pov = pool.submit(rc_pov.render, state, [(0, pick["blue"], 0), (0, pick["red"], 1)], rigs=rigs)
                img_top, _ = f_top.result()
                img_pov, _ = f_pov.result()
                canvas[:TOP_[1]] = cv2.cvtColor(img_top[0], cv2.COLOR_RGB2BGR)
                canvas[TOP_[1]:, :TILE_[0]] = cv2.cvtColor(img_pov[0], cv2.COLOR_RGB2BGR)
                canvas[TOP_[1]:, TILE_[0]:] = cv2.cvtColor(img_pov[1], cv2.COLOR_RGB2BGR)
                self._publish("bc", canvas, quality=LITE["jpeg"] if self.lite else 88)
                self.live = live_state(latest, self.match_s, self.training_step(), self.wins, pick, self.opponent,
                                       None, self.matches)
                frame_idx += 1
                if self.viewers["robot"] > 0 and frame_idx % 2 == 0:       # robot page at 15 fps
                    if rc_robot is None:
                        rc_robot = RenderClient(port=self.ports[2])
                        clients.append(rc_robot)
                        rc_robot.configure(game, [full_cam], n_envs=1, width=self.pov_size[0], height=self.pov_size[1],
                                           max_cams=1, msaa=0, visual=visual, robot="cad" if cad_robot else "box")
                        if appearance is not None:
                            rc_robot.set_appearance(0, *appearance)
                    img_rob, _ = rc_robot.render(state, [(0, latest["sel"], 0)])
                    cam = cv2.cvtColor(img_rob[0], cv2.COLOR_RGB2BGR)
                    self._overlay_robot(cam, latest, tags, tag_ids)
                    self._publish("robot", cam)
                time.sleep(max(0.0, 1.0 / self.fps() - (time.time() - t_frame)))
        finally:
            for rc in clients + ([rc_robot] if rc_robot is not None else []):
                try:
                    rc.close()                        # no orphan Godot processes after a stop or crash
                except Exception:
                    pass

    def _overlay_robot(self, cam, snap, tags, tag_ids):
        """Tags the detector sees (green) and the status panel. Projection uses the camera pose from the child."""
        import cv2
        pov = snap["pov"]
        r = snap["sel"]
        st = snap["status"][r]
        seen = set(pov["seen"]) if pov else set()
        if pov:
            cpos = np.asarray(pov["cam_pos"])
            crot = np.asarray(pov["cam_rot"])
            K = pov["K"]
            for ti in range(tags.shape[0]):
                u, v, z = project(tags[ti], cpos, crot, K, None)
                if np.any(z <= 0.05):
                    continue
                pts = np.stack([u, v], -1).astype(np.int32)
                if ti in seen:
                    cv2.polylines(cam, [pts], True, (80, 230, 80), 2, cv2.LINE_AA)
                    cv2.putText(cam, str(tag_ids[ti]), tuple(int(x) for x in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (80, 230, 80), 1, cv2.LINE_8)
        lines = [f"robot {r + 1} ({'blue' if r < 3 else 'red'})  t={st['t_s']:.1f}s",
                 f"goal {st['goal']}",
                 f"hold coral {st['hold_coral']}  algae {st['hold_algae']}",
                 f"speed {st['speed_mps']:.2f} m/s  batt {st['battery_v']:.1f} V",
                 f"tags seen {len(seen)}  est err {st.get('est_error_m', 0.0):.2f} m"]
        for i, line in enumerate(lines):
            text(cam, line, 10, 24 + 22 * i, 0.55, WHITE, 1)
