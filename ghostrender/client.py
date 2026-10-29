"""Python side of the renderer: starts Godot, configures it from the game/robot definitions,
and renders batches of cameras."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from ghostsim.sensors import rot_rpy

from . import protocol

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parent / "godot"
DEFAULT_BIN = ROOT / "tools" / "godot" / "Godot_v4.7.2-stable_linux.x86_64"


def rig_dict(cam: dict) -> dict:
    return {"t": cam["robot_to_camera"][:3], "R": rot_rpy(*np.radians(cam["robot_to_camera"][3:])).tolist(),
            "fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"], "capture": cam["capture"],
            "world": bool(cam.get("world", False))}


class RenderClient:
    def __init__(self, godot_bin=None, port=None, method="forward_plus", timeout=30.0, log=None):
        self.port = port or int(os.environ.get("GHOST_RENDER_PORT", 47800 + os.getpid() % 1000))
        bin_ = str(godot_bin or os.environ.get("GODOT_BIN", DEFAULT_BIN))
        # renderer output goes to a file (not a pipe nobody reads: a full pipe would block or lose the crash message)
        self.log_path = Path(os.environ.get("GHOST_RENDER_LOG", f"/tmp/ghost_render_{self.port}.log"))
        self._logf = open(self.log_path, "w")
        self.proc = subprocess.Popen([bin_, "--path", str(PROJECT), "--rendering-method", method, "--",
                                      f"--port={self.port}"], stdout=self._logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        with open(self.log_path) as rd:
            while True:                                   # poll the log file until the listen line appears
                line = rd.readline()
                if not line:
                    if self.proc.poll() is not None:
                        raise RuntimeError(f"Godot exited during start, see {self.log_path}")
                    if time.time() - t0 > timeout:
                        self.proc.kill()
                        raise TimeoutError("Godot did not start")
                    time.sleep(0.05)
                    continue
                if log:
                    log.write(line)
                if line.startswith("GHOST_RENDER_LISTEN"):
                    break
                if "SCRIPT ERROR" in line or "Parse Error" in line:
                    self.proc.kill()
                    raise RuntimeError("Godot script error: " + line.strip())
        self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.frame = 0
        self.W = self.H = None

    def _call(self, header, blob=b""):
        protocol.send(self.sock, header, blob)
        return protocol.recv(self.sock)

    def configure(self, game, cameras: list[dict], n_envs: int, width=160, height=120, max_cams=24, visual="auto", msaa=0,
                  fxaa=False):
        """visual: "auto" (CAD model if game/<year>/visual/field.glb exists), "cad", "hi" (the detailed field_hi.glb when
        present, for the live view; else as "auto"), "proxy" (procedural), or a path to a .glb."""
        IN = 0.0254
        pc, f = game.cfg["pieces"], game.cfg["field"]
        bits = json.loads((game.dir / "tag36h11_bits.json").read_text())["bits"]
        glb = game.dir / "visual" / "field.glb"
        if str(visual).endswith(".glb"):
            glb, visual = Path(visual), "cad"
        elif visual == "hi":
            hi = game.dir / "visual" / "field_hi.glb"
            glb, visual = (hi, "cad") if hi.exists() else (glb, "auto")
        use_cad = visual == "cad" or (visual == "auto" and glb.exists())
        if use_cad and not glb.exists():
            raise FileNotFoundError(f"{glb} missing: run tools/prepare_field.py")
        self.visual = "cad" if use_cad else "proxy"
        self.W, self.H, self.n_envs, self.cameras = width, height, n_envs, cameras
        self.rigs = [rig_dict(c) for c in cameras]
        h, _ = self._call({
            "type": "configure", "version": protocol.VERSION, "n_envs": n_envs, "width": width, "height": height,
            "max_cams": max_cams, "scene": game.scene(cad=use_cad), "rigs": self.rigs, "tag_bits": bits,
            "visual_glb": str(glb) if use_cad else "", "msaa": int(msaa), "fxaa": bool(fxaa),
            "pieces": {"mc": game.kc + 6 + game.branch_pose[..., 0].size + game.l1_pose[..., 0].size,   # floor, held, branches, trough
                       "ma": game.ka + 6 + game.reef_algae_pose[..., 0].size, "coral_od": pc["coral"]["od"] * IN, "coral_len": pc["coral"]["length"] * IN,
                       "algae_d": pc["algae"]["diameter"] * IN, "cage_size": f["cages"]["size"] * IN,
                       "cage_h": f["cages"]["height"] * IN}})
        assert h["type"] == "ok"
        self.field = [game.length, game.width]

    def set_appearance(self, env: int, robots: list[dict], dr: dict):
        h, _ = self._call({"type": "appearance", "env": env, "robots": robots, "dr": dr, "field": self.field})
        assert h["type"] == "ok"

    def render(self, state: dict, cams: list, post=None, check=False, markers=None, rigs=None):
        """state: VecSim.render_state() (all envs). cams: [(env, robot, rig_index)].
        post: [(gain, noise, blur_dx_px, blur_dy_px, seed)] per camera. rigs: optional camera dicts (as for
        configure) replacing the configured rig of each camera for this render (None keeps it).
        Returns (K,H,W,3) uint8, header."""
        arrays = {k: state[k] for k in ("robots", "coral", "algae", "cages")}
        if "mech" in state:                              # 254 mechanism geometry (per robot solids), optional
            arrays["mech"] = state["mech"]
        if markers is not None:
            arrays["markers"] = markers
        meta, blob = protocol.pack_arrays(arrays)
        post = post if post is not None else [(1.0, 0.0, 0.0, 0.0, 0.0)] * len(cams)
        self.frame += 1
        msg = {"type": "render", "frame": self.frame, "cams": [list(map(int, c)) for c in cams],
               "post": [list(map(float, p)) for p in post], "check": bool(check), "arrays": meta}
        if rigs is not None:
            msg["rig_override"] = [rig_dict(r) if r is not None else None for r in rigs]
        h, data = self._call(msg, blob)
        cols, rows = h["atlas"]
        img = np.frombuffer(data, np.uint8).reshape(rows, self.H, cols, self.W, 3).transpose(0, 2, 1, 3, 4)
        img = img.reshape(rows * cols, self.H, self.W, 3)[: h["n"]]
        return img, h

    def close(self):
        try:
            self._call({"type": "quit"})
        except Exception:
            pass
        self.sock.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
