"""Expected value per second of each scoring method, from game.json (points) and robots/ours.json (action times).

    uv run python tools/ev_table.py                  # defaults: 3 m travel per cycle, 4.6 m/s, our robot
    uv run python tools/ev_table.py --travel 5 --speed 3.5

Model (all numbers are assumptions unless marked as manual):
  points       teleop points from game.json (manual Table 6-2)
  cycle_s      acquire time + travel / speed + score time, for repeated actions (coral, algae->net)
  pts_per_s    points / cycle_s
  one-off      CAGE and PARK are scored once at the end of the match: shown as points, not per second.
Success probability is 1.0 for reef/net placement; the robot is assumed to find pieces without search time.
Damage and teammate-press penalties are NOT included (they are shown separately in the footer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GAME = Path(__file__).resolve().parents[1] / "game" / "2025" / "game.json"
OURS = Path(__file__).resolve().parents[1] / "robots" / "ours.json"


def table(travel_m: float, speed: float, teleop_only: bool = True):
    g = json.loads(GAME.read_text())
    ours = json.loads(OURS.read_text())
    t = ours["mechanism"]["times_s"]
    tp = lambda pts: pts[1] if teleop_only else pts[0]
    move = travel_m / speed
    rows = []
    for lv in ("L1", "L2", "L3", "L4"):
        pts = tp(g["points"]["coral"][lv])
        cyc = t["coral_station"] + move + t[f"score_{lv}"] + move
        rows.append((f"coral -> {lv}", pts, cyc, "reef branch / trough; uses a station cycle"))
    pts = tp(g["points"]["net"])
    cyc = t["algae_floor"] + move + t["net"] + move
    rows.append(("algae floor -> net", pts, cyc, "net; no processor, no reef-algae removal on our robot"))
    if ours["mechanism"].get("processor"):
        pts = tp(g["points"]["processor"])
        rows.append(("algae -> processor", pts, t["algae_floor"] + move + t["processor"] + move, ""))
    rows.sort(key=lambda r: -r[1] / r[2])
    return rows, g, ours


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--travel", type=float, default=3.0, help="metres travelled between pick-up and score, per cycle")
    ap.add_argument("--speed", type=float, default=4.6, help="m/s, nominal (ours.json control.max_speed_mps)")
    a = ap.parse_args(argv)
    rows, g, ours = table(a.travel, a.speed)
    print(f"Expected value per second, our robot (travel {a.travel} m at {a.speed} m/s, teleop points)")
    print(f"{'method':22s} {'pts':>4s} {'cycle s':>8s} {'pts/s':>7s}  note")
    for name, pts, cyc, note in rows:
        print(f"{name:22s} {pts:4.0f} {cyc:8.2f} {pts / cyc:7.2f}  {note}")
    print()
    print("One-off, end of match (not per second):")
    print(f"  PARK in own barge zone: {g['points']['park']} pts (any robot)")
    print(f"  CAGE shallow / deep:    {g['points']['cage']['shallow']} / {g['points']['cage']['deep']} pts "
          f"(our robot: climb={ours['mechanism']['climb']}, so not available)")
    m = g["model"]
    print()
    print(f"Penalties in training reward (not in the table): damage {m['damage_points']} pt per hit >= "
          f"{m['damage_speed_mps']} m/s (cooldown {m['damage_cooldown_s']} s); teammate contact beyond "
          f"{m['teammate_press_grace_s']} s costs {m.get('teammate_press_pen_per_s', 0)} pt/s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
