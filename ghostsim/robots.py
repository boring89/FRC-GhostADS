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
IN = 0.0254


def load_profile(path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / "robots" / path
    d = json.loads(path.read_text())
    if "base" in d:
        base = load_profile(path.parent / d["base"])
        base.update({"randomize": d})
        d = base
    if not d.get("verified", False) and "base" not in d:
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
    levels: np.ndarray; station: np.ndarray; floor_coral: np.ndarray; floor_algae: np.ndarray; algae_reef: np.ndarray
    processor: np.ndarray; net: np.ndarray; climb: np.ndarray; station_side: np.ndarray
    times: np.ndarray; succ: np.ndarray; align_scale: np.ndarray; intake_depth: np.ndarray; intake_half_width: np.ndarray
    quality: np.ndarray

    def control(self):
        return {"max_v": self.max_v, "max_w": self.max_w, "deadband": self.deadband, "offsets": self.offsets}


def _get(d, key):
    for k in key.split("."):
        d = d[k]
    return d


def sample_params(profiles: list[dict], n: int, goals: list[str], rng, randomize: bool = True) -> RobotParams:
    """profiles: one dict per robot slot (len R). randomize=False gives the nominal values of the base profile
    (used for calibration and deterministic tests).

    A profile's "randomize" block may draw one quality q ~ U(quality) per robot and reset; every entry of
    "by_quality" is then [value at q=0, value at q=1(, gamma)]: v1 + (v0 - v1) * (1 - q)**gamma (gamma 1 = linear),
    so strong robots are strong in every respect.
    Entries of "uniform" are drawn independently of q."""
    out = {}
    for r, p in enumerate(profiles):
        rnd = p.get("randomize", {}) if randomize else {}
        uni = rnd.get("uniform", {})
        bq_all = rnd.get("by_quality", {}) if "quality" in rnd else {}
        bt = p["bumper_t_in"] * IN
        for e in range(n):
            q = rng.uniform(*rnd["quality"]) if "quality" in rnd else 1.0
            lerp = lambda v, q=q: v[1] + (v[0] - v[1]) * (1.0 - q) ** (v[2] if len(v) > 2 else 1.0)
            bq = bq_all

            def num(key, uni=uni, p=p, bq=bq, lerp=lerp):
                if key in bq:
                    return lerp(bq[key])
                return rng.uniform(*uni[key]) if key in uni else float(_get(p, key))

            frame = num("frame_in") * IN
            half = np.full(2, frame / 2 + bt)
            offs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * (frame / 2 - p["module_inset_m"])
            mass = num("mass_kg")
            dr, ctl, mech = p["drive"], p["control"], p["mechanism"]
            free = num("drive.free_speed_mps")
            mu_s = num("drive.mu_static")
            vals = {
                "mass": mass, "inertia": mass * ((2 * half) ** 2).sum() / 12.0, "half": half,
                "height": num("height_m"), "offsets": offs,
                "free_speed": free, "stall_force": dr["stall_force_N"], "stall_current": dr["stall_current_A"],
                "force_limit": dr["stall_force_N"] * num("drive.current_limit_A") / dr["stall_current_A"],
                "steer_rate": dr["steer_rate_radps"], "mu_s": mu_s,
                "mu_k": dr["mu_kinetic"] * mu_s / dr["mu_static"],
                "v_oc": p["battery"]["v_oc"], "r_int": p["battery"]["r_int_ohm"], "other_load": p["battery"]["other_load_A"],
                "max_v": min(num("control.max_speed_mps"), 0.95 * free), "max_w": num("control.max_omega_radps"),
                "deadband": ctl["deadband"],
                "station_side": 1.0 if mech["station_side"] == "front" else -1.0,
                "intake_depth": mech["intake_depth_m"], "intake_half_width": mech["intake_half_width_m"],
            }
            ts = lerp(bq["times_scale"]) if "times_scale" in bq else \
                (rng.uniform(*rnd["times_scale"]) if "times_scale" in rnd else 1.0)
            vals["times"] = np.array([mech["times_s"].get(g, 0.0) * ts for g in goals])
            vals["align_scale"] = lerp(bq["align_scale"]) if "align_scale" in bq else 1.0
            vals["quality"] = q
            sp = mech.get("success_p", {})
            if "success_p" in bq:
                vals["succ"] = np.array([lerp(bq["success_p"][g]) if g in bq["success_p"] else sp.get(g, 1.0) for g in goals])
            else:
                spr = rnd.get("success_p", {})
                vals["succ"] = np.array([rng.uniform(*spr[g]) if g in spr else sp.get(g, 1.0) for g in goals])
            cap = {k: lerp(v) for k, v in bq["capability_p"].items() if not k.startswith("_")} \
                if "capability_p" in bq else rnd.get("capability_p")
            if cap:
                lv = np.array([rng.random() < cap[l] for l in LEVELS])
                c = {k: rng.random() < cap[k] for k in ("station", "floor_coral", "floor_algae", "algae_reef", "processor", "net")}
                u = rng.random()
                climb = 2 if u < cap["climb_deep"] else (1 if u < cap["climb_deep"] + cap["climb_shallow"] else 0)
            else:
                lv = np.array([i + 1 in mech["levels"] for i in range(4)])
                c = {k: bool(mech[k]) for k in ("station", "floor_coral", "floor_algae", "algae_reef", "processor", "net")}
                climb = {"none": 0, "shallow": 1, "deep": 2}[mech["climb"]]
            vals.update(levels=lv, climb=climb, **c)
            for k, v in vals.items():
                out.setdefault(k, [[None] * R for _ in range(n)])[e][r] = v
    return RobotParams(**{k: np.array(v) for k, v in out.items()})
