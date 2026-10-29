"""Mechanism animation as the live view plays it: the real JointTracker (ghostsim/mechanism_state.py) driven by a
scripted sequence of rule states (goal, dwell progress, held pieces), stepped at the video frame rate.

    uv run python tools/mech_anim.py                  # -> runs/mech/anim.mp4 (30 fps)
    uv run python tools/mech_anim.py --frames-png 8   # also dump every 8th frame as PNG into runs/mech/frames/

Story: back into the coral station (coral slides down the funnel), drive to the reef (nothing moves), dwell and score
L4 (elevator rises with the dwell, coral leaves at the end), algae off the floor with the front arm, drive, processor,
then climb.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ghostsim.mechanism import pose
from ghostsim.mechanism_state import JointTracker
from tools.mech_view import label, render

FPS = 30
GOALS = ["idle", "coral_station", "coral_floor", "algae_floor", "algae_reef", "score_L1", "score_L2", "score_L3",
         "score_L4", "processor", "net", "climb", "defend", "align_score"]

# (name, seconds, goal, dwell: None or (p0, p1), hold_coral, hold_algae, climbed); a dwell ends with the action firing
SCRIPT = [
    ("driving to the station (carry pose)", 0.8, "coral_station", None, False, False, False),
    ("backed into the station: coral slides down the funnel", 0.6, "coral_station", (0.0, 1.0), False, False, False),
    ("driving to the reef: nothing moves", 1.0, "score_L4", None, True, False, False),
    ("in place at the reef: elevator rises with the dwell", 0.9, "score_L4", (0.0, 1.0), True, False, False),
    ("coral released on L4, pose held, then back down", 1.4, "score_L4", None, False, False, False),
    ("algae arm down, intaking from the floor", 0.9, "algae_floor", None, False, False, False),
    ("driving with the algae cradled", 0.8, "processor", None, False, True, False),
    ("processor: arm lowers", 0.5, "processor", (0.0, 1.0), False, True, False),
    ("released into the processor", 0.8, "processor", None, False, False, False),
    ("climber reaches for the cage", 1.0, "climb", None, False, False, False),
    ("hanging", 1.0, "climb", None, False, False, True),
]


def timeline():
    return sum(s[1] for s in SCRIPT)


def frames():
    """Yield (t, segment name, solids, tracker) for every video frame."""
    tr = JointTracker(GOALS, 1)
    t = 0.0
    for name, dur, goal, dwell, hc, ha, cl in SCRIPT:
        n = int(round(dur * FPS))
        for i in range(n):
            p = 0.0 if dwell is None else dwell[0] + (dwell[1] - dwell[0]) * (i + 1) / n
            tr.update([GOALS.index(goal)], [min(p, 0.999)], [hc], [ha], [cl], dt=1.0 / FPS)
            h, w, a, c = tr.q[0]
            yield t, name, pose(h=h, wrist=w, arm=a, climb=c, hold=tr.hold[0], funnel=tr.funnel[0]), tr
            t += 1.0 / FPS


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mech/anim.mp4")
    ap.add_argument("--frames-png", type=int, default=0, help="if >0, also save every Nth frame as PNG")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (900, 700))
    if not wr.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4v")
    png_dir = out.parent / "frames"
    if a.frames_png:
        png_dir.mkdir(exist_ok=True)
    k = 0
    for t, name, solids, _ in frames():
        img = label(render(solids, eye=(2.6, -2.2, 2.0), target=(0.0, 0.0, 0.9)), f"{t:5.2f}s  {name}")
        wr.write(img)
        if a.frames_png and k % a.frames_png == 0:
            cv2.imwrite(str(png_dir / f"f{k:04d}.png"), img)
        k += 1
    wr.release()
    print("wrote", out, f"({k} frames, {timeline():.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
