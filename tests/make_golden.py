"""Regenerate tests/golden/motion_v1.json (the contract the Systemcore Java motion layer must reproduce)."""
import json
from pathlib import Path

import numpy as np

from ghostsim.motion import motion_layer

OFF = [[0.29, 0.29], [0.29, -0.29], [-0.29, 0.29], [-0.29, -0.29]]


def main():
    rng = np.random.default_rng(2025)
    params = {"max_v": 4.6, "max_w": 9.0, "deadband": 0.05, "offsets": OFF}
    p = {k: np.asarray(v) for k, v in params.items()}
    cases = []
    fixed = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0.03, -0.02, 0.01], [-0.7, 0.4, -0.9]]
    sticks = fixed + rng.uniform(-1, 1, (25, 3)).round(4).tolist()
    for i, s in enumerate(sticks):
        alliance = i % 2
        yaw = float(np.round(rng.uniform(-np.pi, np.pi), 4))
        cur = rng.uniform(-np.pi, np.pi, 4).round(4)
        prev = rng.uniform(-np.pi, np.pi, 4).round(4)
        spd, ang = motion_layer(np.array(s, float), np.array(alliance), np.array(yaw), cur, prev, p, 0.02)
        cases.append({"stick": s, "alliance": alliance, "yaw": yaw, "current_angle": cur.tolist(),
                      "prev_target_angle": prev.tolist(), "speed": spd.tolist(), "angle": ang.tolist()})
    out = {"version": 1, "dt": 0.02, "params": params, "units": "m/s, rad, robot frame, WPILib conventions",
           "cases": cases}
    path = Path(__file__).parent / "golden" / "motion_v1.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path, len(cases), "cases")


if __name__ == "__main__":
    main()
