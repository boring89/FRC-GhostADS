"""Evaluate a training run while it runs: every `--every` iterations, the new snapshot plays whole matches
(ghosttrain/arena.py: symmetric slots, deterministic) against fixed references and against the previously evaluated
snapshot. One line per evaluated snapshot in <run>/eval.jsonl; the dashboard plots it.

    uv run python -m ghosttrain.eval_watch runs/teacher8 --init runs/bc9/bc_net.pt [--every 250] [--episodes 96]

References: "baseline" (rule-based), "init" (the warm start, --init), "prev" (the snapshot evaluated before this one:
positive = still improving). Uses --n_procs sim processes next to the training (default 4); 96 matches against three
references take ~15-25 min on the P520 while the training runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

from .arena import play


def snapshots(run: Path, every: int):
    out = {}
    for p in run.glob("iter_*.pt"):
        m = re.match(r"iter_(\d+)\.pt$", p.name)
        if m and int(m.group(1)) % every == 0:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def done_iters(path: Path):
    if not path.exists():
        return set()
    return {json.loads(line)["it"] for line in path.read_text().splitlines() if line.strip()}


def keep_net(run: Path, it: int, snap: Path) -> Path:
    """Copy the net of an evaluated snapshot to <run>/eval_snaps (the run prunes old snapshots; 'prev' needs it)."""
    d = run / "eval_snaps"
    d.mkdir(exist_ok=True)
    dst = d / f"iter_{it:06d}_net.pt"
    if not dst.exists():
        sd = torch.load(snap, map_location="cpu", weights_only=False)
        torch.save({"net": sd["net"] if "net" in sd else sd}, dst)
    return dst


def evaluate_once(run: Path, it: int, snap: Path, refs: dict, episodes: int, n_procs: int, log=print, team="ours"):
    row = {"it": it, "t": time.time()}
    for name, ref in refs.items():
        t0 = time.time()
        r = play(str(snap), str(ref), episodes, n_procs=n_procs, envs_per_proc=max(1, episodes // (2 * n_procs)),
                 team=team)
        row[f"win_{name}"], row[f"diff_{name}"] = r["win"], r["diff"]
        row[f"diff_{name}_ci"], row[f"win_{name}_ci"] = r["diff_ci"], r["win_ci"]
        row[f"score_{name}"] = [r["score_a"], r["score_b"]]
        log(f"eval it {it} vs {name}: win {r['win']:.2f} diff {r['diff']:+.1f} "
            f"(CI {r['diff_ci'][0]:+.1f} to {r['diff_ci'][1]:+.1f}) [{time.time() - t0:.0f}s]")
    row["episodes"], row["team"] = episodes, team
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--init", default="", help="warm-start net (reference 'init')")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--n_procs", type=int, default=4)
    ap.add_argument("--start", type=int, default=0, help="skip snapshots before this iteration")
    ap.add_argument("--team", default="ours", choices=["ours", "all"], help="arena.py team mode (ours: our robot only)")
    ap.add_argument("--once", action="store_true", help="evaluate what is there, then exit")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out = run / "eval.jsonl"
    log = lambda m: print(time.strftime("%H:%M:%S ") + m, flush=True)
    while True:
        snaps = snapshots(run, a.every)
        todo = [i for i in snaps if i >= a.start and i not in done_iters(out)]
        for it in todo:
            evaluated = sorted(done_iters(out))
            refs = {"baseline": "baseline"}
            if a.init:
                refs["init"] = a.init
            prev = max([i for i in evaluated if i < it], default=None)
            kept = run / "eval_snaps" / f"iter_{prev:06d}_net.pt" if prev is not None else None
            if kept is not None and kept.exists():
                refs["prev"] = kept
            if not snaps[it].exists():                        # pruned meanwhile
                continue
            net = keep_net(run, it, snaps[it])
            row = evaluate_once(run, it, net, refs, a.episodes, a.n_procs, log, a.team)
            row["prev_it"] = prev
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if a.once:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
