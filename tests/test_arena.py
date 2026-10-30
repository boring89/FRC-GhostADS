"""Evaluation tools: ghosttrain/arena.py (whole matches, symmetric slots, A on both sides) and
ghosttrain/eval_watch.py (snapshots every N iterations against references, kept nets, eval.jsonl rows)."""
import json

import numpy as np
import torch

from ghostsim import SimCfg
from ghosttrain import arena, eval_watch
from ghosttrain.net import TeacherNet


def test_summary_win_rate_score_difference_and_sides():
    res = [(120, 100, 0), (90, 110, 0), (100, 100, 1), (150, 100, 1)]
    r = arena.summarise(res)
    assert r["episodes"] == 4 and abs(r["win"] - (1 + 0 + 0.5 + 1) / 4) < 1e-9
    assert abs(r["diff"] - (20 - 20 + 0 + 50) / 4) < 1e-9 and r["diff_ci"][0] < r["diff"] < r["diff_ci"][1]
    assert r["win_blue"] == 0.5 and r["win_red"] == 0.75 and r["win_ci"][0] < r["win"] < r["win_ci"][1]


def test_arena_plays_whole_matches_on_both_sides_with_symmetric_slots():
    env = arena.ArenaEnv(SimCfg(n_envs=2, sensors=False, seed=0, slots=arena.SYMMETRIC), 1, in_process=True)
    w = env.local[0]
    assert (w.trunc_at >= 10 ** 9).all()                                      # no desync cut
    assert w.sim.prm.quality[0, 0] == w.sim.prm.quality[0, 3]                 # our robot in both alliances
    net = TeacherNet(14, w.enc.G).eval()
    r = arena.play(net, "baseline", episodes=2, n_procs=1, envs_per_proc=2, device="cpu", in_process=True)
    assert r["episodes"] == 2 and "win_blue" in r and "win_red" in r            # one match per side
    assert r["score_b"] > r["score_a"]                                        # an untrained net loses to the baseline


def test_eval_watch_rows_and_kept_nets(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for it in (240, 250, 500):
        torch.save({"net": {"w": torch.ones(1) * it}, "opt": {}}, run / f"iter_{it:06d}.pt")
    assert list(eval_watch.snapshots(run, 250)) == [250, 500]
    calls = []

    def fake_play(a, b, episodes, **kw):
        calls.append((a, b))
        return {"win": 0.6, "diff": 5.0, "diff_ci": (1.0, 9.0), "win_ci": (0.5, 0.7), "score_a": 105.0, "score_b": 100.0}
    monkeypatch.setattr(eval_watch, "play", fake_play)
    eval_watch.main([str(run), "--once", "--init", "bc.pt"])
    rows = [json.loads(x) for x in (run / "eval.jsonl").read_text().splitlines()]
    assert [r["it"] for r in rows] == [250, 500] and rows[1]["prev_it"] == 250
    assert "diff_prev" not in rows[0] and rows[1]["diff_prev"] == 5.0 and rows[0]["diff_init"] == 5.0
    kept = run / "eval_snaps" / "iter_000250_net.pt"
    assert kept.exists() and set(torch.load(kept, weights_only=False)) == {"net"}
    assert any(b == str(kept) for _, b in calls)                              # 'prev' uses the kept copy
