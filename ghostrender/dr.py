"""Domain randomisation values. Sampled in Python (reproducible, logged with the data);
the engine only applies them."""
from __future__ import annotations

import numpy as np

from ghostsim.robots import ALLIANCE

BLUE, RED = np.array([0.10, 0.25, 0.80]), np.array([0.80, 0.10, 0.10])


def sample_appearance(rng, half, height, bumper_z, team_numbers=None):
    """half (R,2), height (R,) of one env -> list of 6 robot appearance dicts."""
    out = []
    for r in range(6):
        base = BLUE if ALLIANCE[r] == 0 else RED
        col = np.clip(base * rng.uniform(0.6, 1.25) + rng.uniform(-0.08, 0.08, 3), 0, 1)
        team = team_numbers[r] if team_numbers is not None else int(rng.integers(1, 11000))
        g = rng.uniform(0.15, 0.85)
        body = np.clip(g + rng.uniform(-0.1, 0.1, 3), 0, 1) if rng.random() < 0.7 else rng.uniform(0, 1, 3)
        out.append({"bumper_rgb": col.round(4).tolist(), "body_rgb": body.round(4).tolist(), "team": str(team),
                    "font": int(rng.integers(0, 10)), "size": [float(2 * half[r, 0]), float(2 * half[r, 1]), float(height[r])],
                    "bumper_z": list(map(float, bumper_z)), "seed": int(rng.integers(0, 2 ** 31))})
    return out


def sample_dr(rng):
    warm = rng.uniform(-0.15, 0.15)
    return {
        "sun_energy": float(rng.uniform(0.3, 2.0)), "sun_elev_deg": float(rng.uniform(25, 85)),
        "sun_az_deg": float(rng.uniform(0, 360)),
        "sun_rgb": [float(1 + warm), 1.0, float(1 - warm)],
        "ambient_energy": float(rng.uniform(0.2, 1.2)),
        "ambient_rgb": np.clip(1 + rng.uniform(-0.15, 0.15, 3), 0, 1).round(3).tolist(),
        "bg_rgb": rng.uniform(0.0, 0.35, 3).round(3).tolist(),
        "carpet_rgb": np.clip(rng.uniform(0.12, 0.35) + rng.uniform(-0.03, 0.03, 3), 0, 1).round(3).tolist(),
        "carpet_rough": float(rng.uniform(0.6, 1.0)),
        "crowd_seed": int(rng.integers(0, 2 ** 31)), "crowd_density": float(rng.uniform(0.3, 1.5)),
    }


def sample_camera_post(rng, n_cams):
    """Per camera, per episode: exposure gain, noise std, exposure time (s)."""
    return {"gain": rng.uniform(0.6, 1.5, n_cams), "noise": rng.uniform(0.0, 0.04, n_cams),
            "t_exp": rng.uniform(0.002, 0.015, n_cams)}


def blur_px(omega, fx_render, t_exp):
    """Ego-motion blur length in pixels from the robot yaw rate (dominant term)."""
    return np.abs(omega) * fx_render * t_exp
