"""Real 2025 FIRST Championship division data (frc-events public match pages) for calibrating robots/others.json.

    uv run python tools/champs_data.py fetch NEWTON HOPPER      # -> data/frc_events/<DIV>.jsonl (0.5 s between requests)
    uv run python tools/champs_data.py summary                  # alliance / robot / per-team (OPR) statistics

The match page shows the reef as icons (auto state, then final state), trough counts, processor / net algae, the
barge state per robot, auto / teleop / foul points. Every parsed match is checked against the official auto and
teleop points (a few mismatches are expected: pages where the reef icons and the points disagree).
"""
from __future__ import annotations

import collections
import glob
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "frc_events"
URL = "https://frc-events.firstinspires.org/2025/{div}/qualifications"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GhostADS calibration script)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def parse(s):
    """One match page -> dict (index 0 blue, 1 red)."""
    rows = collections.defaultdict(list)
    for name, body in re.findall(r'<tr>\s*<td>(?:<strong>)?([^<]+)(?:</strong>)?</td>(.*?)</tr>', s, flags=re.S):
        for _, td in re.findall(r'<td class="(info|danger)">(.*?)</td>', body, flags=re.S):
            rows[name.strip()].append(td)
    txt = [t.strip() for t in html.unescape(re.sub(r'<[^>]+>', '\n', re.sub(r'<script.*?</script>', '', s, flags=re.S)))
           .split('\n') if t.strip()]
    top = txt.index("Match Score Item")
    after = lambda label, n=2: txt[txt.index(label, top) + 1:txt.index(label, top) + 1 + n]
    words = lambda td: re.sub(r'<[^>]+>', ' ', td).split()
    num = lambda name, i: float(words(rows[name][i])[0].replace("+", "")) if words(rows[name][i]) else 0.0
    checks = lambda td: len(re.findall(r'fa-check', td))
    lv = ("High Branch", "Middle Branch", "Low Branch")                   # L4, L3, L2; each row: auto (2), final (2)
    return {
        "teams": [int(x) for x in after("Teams", 6)],
        "leave": [checks(rows["Leave"][i]) for i in (0, 1)],
        "auto_reef": [[checks(rows[n][i]) for n in lv] for i in (0, 1)],
        "final_reef": [[checks(rows[n][2 + i]) for n in lv] for i in (0, 1)],
        "auto_trough": [num("Trough", i) for i in (0, 1)],
        "tele_trough": [num("Trough", 2 + i) for i in (0, 1)],
        "processor": [num("Processor Algae", i) for i in (0, 1)],
        "net": [num("Net Algae", i) for i in (0, 1)],
        "barge": [words(rows["Barge"][i]) for i in (0, 1)],
        "auto_pts": [float(x) for x in after("Autonomous Points")],
        "tele_pts": [float(x) for x in after("Teleop Points")],
        "foul_pts": [float(x.replace("+", "")) for x in after("Foul Points")],
        "final": [float(x) for x in after("Final Score")],
    }


def fetch(divisions):
    OUT.mkdir(parents=True, exist_ok=True)
    for div in divisions:
        ids = sorted({int(x) for x in re.findall(rf'/2025/{div}/qualifications/(\d+)"', _get(URL.format(div=div)))})
        print(div, len(ids), "matches", flush=True)
        with open(OUT / f"{div}.jsonl", "w") as f:
            for i in ids:
                try:
                    m = parse(_get(URL.format(div=div) + f"/{i}"))
                    m["match"] = i
                    f.write(json.dumps(m) + "\n")
                except Exception as ex:                                    # page layout changes, timeouts
                    print("  match", i, "failed:", repr(ex)[:120], flush=True)
                time.sleep(0.5)


def load(path=OUT):
    return [json.loads(line) for f in sorted(glob.glob(str(Path(path) / "*.jsonl"))) for line in open(f)]


def alliance_rows(M):
    """Per alliance sample: metrics dict, its 3 teams, the 3 barge states; plus the number of point-check mismatches."""
    out, bad = [], 0
    for m in M:
        for a in (0, 1):
            ar, fr = np.array(m["auto_reef"][a]), np.array(m["final_reef"][a])
            at, tt = m["auto_trough"][a], m["tele_trough"][a]
            bpts = sum({"DeepCage": 12, "ShallowCage": 6, "Parked": 2}.get(b, 0) for b in m["barge"][a])
            auto = 3 * m["leave"][a] + ar @ [7, 6, 4] + 3 * at
            tele = (fr - ar) @ [5, 4, 3] + 2 * tt + 6 * m["processor"][a] + 4 * m["net"][a] + bpts
            bad += abs(auto - m["auto_pts"][a]) > 0.5 or abs(tele - m["tele_pts"][a]) > 0.5
            coral = fr.sum() + at + tt
            out.append(({"score": m["auto_pts"][a] + m["tele_pts"][a], "coral_n": coral, "L4": fr[0], "L3": fr[1], "L2": fr[2],
                         "L1": at + tt, "auto_coral": ar.sum() + at, "net": m["net"][a], "proc": m["processor"][a],
                         "barge": bpts, "pieces": coral + m["net"][a] + m["processor"][a]},
                        m["teams"][3 * a:3 * a + 3], m["barge"][a]))
    return out, bad


def summary(M):
    rows, bad = alliance_rows(M)
    teams = sorted({t for _, ts, _ in rows for t in ts})
    ti = {t: i for i, t in enumerate(teams)}
    print(f"{len(M)} matches, {len(teams)} teams, point-check mismatches {bad}")
    pct = lambda v: f"mean {np.mean(v):6.1f}  p10 {np.percentile(v, 10):5.1f}  p50 {np.percentile(v, 50):5.1f}  p90 {np.percentile(v, 90):5.1f}"
    print("per alliance (no fouls)")
    for k in rows[0][0]:
        print(f"  {k:10s} {pct([r[k] for r, _, _ in rows])}")
    ends = collections.Counter(b for _, _, bs in rows for b in bs)
    n = sum(ends.values())
    print("robot end state:", {k: round(v / n, 2) for k, v in ends.most_common()})
    X = np.zeros((len(rows), len(teams)))
    for j, (_, ts, _) in enumerate(rows):
        X[j, [ti[t] for t in ts]] = 1
    w = np.linalg.lstsq(X, np.array([r["pieces"] for r, _, _ in rows]), rcond=None)[0]
    qs = [10, 25, 50, 75, 90, 97]
    print("pieces per team (OPR):", dict(zip(qs, np.percentile(w, qs).round(1))))
    print("  relative to the 97th percentile team:", dict(zip(qs, (np.percentile(w, qs) / np.percentile(w, 97)).round(2))))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch(sys.argv[2:] or ["NEWTON", "HOPPER"])
    else:
        summary(load())
