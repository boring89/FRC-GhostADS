"""Robot profiles (robots/*.json) -> per-environment parameter arrays of shape (N, R)."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

R = 6
ALLIANCE = np.array([0, 0, 0, 1, 1, 1])
LEVELS = ("L1", "L2", "L3", "L4")
ROOT = Path(__file__).resolve().parents[1]


def load_profile(path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / "robots" / path
    d = json.loads(path.read_text())
    if "base" in d:
        base = load_profile(path.parent / d["base"])
        base.update({"randomize": d})
        d = base
    if not d.get("verified", False) and "randomize" not in d:
        warnings.warn(f"{path.name}: robot profile contains unverified placeholder values", stacklevel=2)
    return d


@dataclass
class RobotParams:
    mass: np.ndarray; inertia: np.ndarray; half: np.ndarray; height: np.ndarray
    offsets: np.ndarray
    free_speed: np.ndarray; stall_force: np.ndarray; stall_current: np.ndarray; force_limit: np.ndarray
    steer_rate: np.ndarray; mu_s: np.ndarray; mu_k: np.ndarray
    v_oc: np.ndarray; r_int: np.ndarray; other_load: np.ndarray
    max_v: np.ndarray; max_w: np.ndarray; deadband: np.ndarray
    levels: np.ndarray; floor_coral: np.ndarray; floor_algae: np.ndarray; algae_reef: np.ndarray
    processor: np.ndarray; net: np.ndarray; climb: np.ndarray; station_side: np.ndarray
    times: np.ndarray; intake_depth: np.ndarray; intake_half_width: np.ndarray

    def control(self):
        return {"max_v": self.max_v, "max_w": self.max_w, "deadband": self.deadband, "offsets": self.offsets}


def _get(d, key):
    for k in key.split("."):
        d = d[k]
    return d


def sample_params(profiles: list[dict], n: int, goals: list[str], rng) -> RobotParams:
    """profiles: one dict per robot slot (len R)."""
    out = {}
    for r, p in enumerate(profiles):
        rnd = p.get("randomize", {})
        uni = rnd.get("uniform", {})
        cap = rnd.get("capability_p")

        def num(key, e=None, p=p, uni=uni):
            if key in uni:
                return rng.uniform(*uni[key])
            return float(_get(p, key))

        for e in range(n):
            side = rng.uniform(*uni["bumper_side_m"]) if "bumper_side_m" in uni else None
            half = np.array([side, side]) / 2 if side else np.array(p["bumper_outer_m"]) / 2
            scale = (half * 2 - 0.16) / (np.array(p["bumper_outer_m"]) - 0.16)   # module spacing follows frame
            offs = np.array(p["module_offsets_m"]) * scale
            mass = num("mass_kg")
            dr, ctl, mech = p["drive"], p["control"], p["mechanism"]
            vals = {
                "mass": mass, "inertia": mass * ((2 * half) ** 2).sum() / 12.0, "half": half,
                "height": num("height_m"), "offsets": offs,
                "free_speed": num("drive.free_speed_mps"), "stall_force": dr["stall_force_N"],
                "stall_current": dr["stall_current_A"],
                "force_limit": dr["stall_force_N"] * num("drive.current_limit_A") / dr["stall_current_A"],
                "steer_rate": dr["steer_rate_radps"], "mu_s": num("drive.mu_static"),
                "mu_k": dr["mu_kinetic"] * num("drive.mu_static") / dr["mu_static"],
                "v_oc": p["battery"]["v_oc"], "r_int": p["battery"]["r_int_ohm"],
                "other_load": p["battery"]["other_load_A"],
                "max_v": num("control.max_speed_mps"), "max_w": num("control.max_omega_radps"),
                "deadband": ctl["deadband"],
                "station_side": 1.0 if mech["station_side"] == "front" else -1.0,
                "intake_depth": mech["intake_depth_m"], "intake_half_width": mech["intake_half_width_m"],
            }
            ts = rng.uniform(*rnd["times_scale"]) if "times_scale" in rnd else 1.0
            vals["times"] = np.array([mech["times_s"].get(g, 0.0) * ts for g in goals])
            if cap:
                lv = np.array([rng.random() < cap[l] for l in LEVELS])
                c = {k: rng.random() < cap[k] for k in ("floor_coral", "floor_algae", "algae_reef", "processor", "net")}
                u = rng.random()
                climb = 2 if u < cap["climb_deep"] else (1 if u < cap["climb_deep"] + cap["climb_shallow"] else 0)
            else:
                lv = np.array([i + 1 in mech["levels"] for i in range(4)])
                c = {k: bool(mech[k]) for k in ("floor_coral", "floor_algae", "algae_reef", "processor", "net")}
                climb = {"none": 0, "shallow": 1, "deep": 2}[mech["climb"]]
            vals.update(levels=lv, climb=climb, **c)
            for k, v in vals.items():
                out.setdefault(k, [[None] * R for _ in range(n)])[e][r] = v
    return RobotParams(**{k: np.array(v) for k, v in out.items()})
