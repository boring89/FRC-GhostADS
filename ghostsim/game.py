"""Game modules. A game lives in game/<year>/: game.json (single source of truth, shared with the
renderer) + rules.py (vectorised game logic implementing GameBase)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from .geometry import IN

ROOT = Path(__file__).resolve().parents[1]


class GameBase:
    """Interface the simulator relies on. Coordinates: WPILib field frame, metres."""
    name: str
    goals: list[str]
    length: float
    width: float
    polys = None            # geometry.Polygons: static obstacles (physics + occlusion)
    poly_ids: list[str]
    hp_n: np.ndarray        # (H,2) field boundary half-planes, valid where n.p >= off
    hp_off: np.ndarray
    tags: dict              # id -> dict(pos=(3,), quat=(4,) wxyz, yaw, pitch)
    auto_s: float
    teleop_s: float

    def new_state(self, n): ...
    def reset(self, st, idx, body, prm, rng, mode): ...
    def substep(self, st, body, prm, goal, t, live, dt, rng): ...   # -> points (N,R), fouls (N,R), bonus (N,2)
    def end_auto(self, st, body, prm, idx): ...                     # -> points (N,R)
    def end_match(self, st, body, prm, idx): ...                    # -> points (N,R)
    def fixed(self, st): ...                                        # (N,R) robots that cannot move
    def holding(self, st): ...                                      # (N,R,k) game-piece sensor
    def privileged(self, st): ...
    def render_objects(self, st, body, prm): ...
    def scene(self): ...                                            # static primitives for the renderer


def load_tags(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {}
    for t in d["tags"]:
        p, q = t["pose"]["translation"], t["pose"]["rotation"]["quaternion"]
        w, x, y, z = q["W"], q["X"], q["Y"], q["Z"]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        out[t["ID"]] = dict(pos=np.array([p["x"], p["y"], p["z"]]), quat=np.array([w, x, y, z]),
                            yaw=float(yaw), pitch=float(pitch))
    return out


def load(game_dir: str | Path, tag_set: str | None = None) -> GameBase:
    game_dir = Path(game_dir)
    if not game_dir.is_absolute() and not game_dir.exists():
        game_dir = ROOT / "game" / str(game_dir)
    cfg = json.loads((game_dir / "game.json").read_text())
    if cfg.get("schema_version") != 2:
        raise ValueError("game.json schema_version must be 2")
    spec = importlib.util.spec_from_file_location(f"game_rules_{game_dir.name}", game_dir / cfg["rules_module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tags_cfg = cfg["field"]["tags"]
    tags = load_tags(game_dir / tags_cfg[tag_set or tags_cfg["default"]])
    return mod.Game(cfg, tags, game_dir)


__all__ = ["GameBase", "load", "load_tags", "IN"]
