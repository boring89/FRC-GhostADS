# Teacher training (phase 2)

Everything runs on the P520: 12 sim worker processes (CPU) + the learner on the GTX 1070
(PyTorch 2.14.1 + CUDA 12.6; the wheel ships sm_60 SASS, which runs on the 1070's sm_61 - checked with fp32/fp16 matmuls).

## Pipeline
```
uv run python -m ghosttrain.bc --rounds 120 --out runs/bc2           # 1. clone the rule-based baseline (~10 min)
uv run python -m ghosttrain.train --init runs/bc2/bc_net.pt --lr 1e-4 --ent_goal 0.003 \
        --iters 2000 --out runs/teacher --snapshot_every 25           # 2. MAPPO self-play + league
uv run python -m ghosttrain.evaluate runs/teacher/latest_net.pt --episodes 200   # 3. win rate vs baseline, both colours
uv run python -m ghosttrain.replay runs/teacher/latest_net.pt --matches 10 --out runs/teacher/replays   # 4. videos + audit
```
`uv run python -m ghosttrain.dashboard runs/teacher` serves live curves, an anomaly table and a live match at
http://localhost:8777 (`--no-live` for curves only, `--host 0.0.0.0` for other machines). Fullscreen viewer: http://localhost:8777/live
(F or double-click). Live view defaults to 1280x720, 4x MSAA, 25 fps (`--width/--height/--fps/--msaa/--jpeg-quality`;
fps = 50/k since physics runs at 50 Hz; every frame is a real physics substep, not an interpolation). While the
trainer uses the GPU at 100 % the viewer reached ~16 fps at 720p (it then plays slower than real time).
`--resume runs/teacher/iter_000500.pt` continues a run. `smoke/smoke_train.py` checks the pipeline in ~1 minute.

## Design
- **Actor**: entity-set attention over an ego-centric, alliance-canonical observation (other robots, floor pieces,
  reef faces, landmarks; `ghosttrain/obs.py`), masked by robot capabilities. Outputs: goal (13, re-decided once per
  second) + joystick [forward, left, rotate] every 0.1 s.
- **Critic**: centralised (global state of the alliance), reads the actor's embedding with `detach` so it can never
  disturb the actor.
- **Reward**: match-point difference x0.1 (fouls included) + win/loss +-1; shaping (annealed over 1500 iterations):
  acquiring a piece and potential-based distance to the baseline's target.
- **League**: per episode the red alliance is SELF (35 %), the baseline, or a frozen snapshot of the learner; PFSP
  weights (1-win)^2. Snapshots every 25 iterations, pool of 6.
- **Why BC first**: from scratch, 7 M agent-steps gave 0 % wins against the baseline (and the curve was flat), so the
  start-up cost would be days. BC alone reaches 17.7 % wins; PPO from there is shown below.

## Measured (this machine)
| item | value |
|---|---|
| training throughput | 10-12 k agent-steps/s (96-384 envs; lower with many pool opponents) ~ 120 k matches/day |
| baseline vs baseline win rate | 0.466 (95 % CI 0.40-0.54, 192 matches) - no colour bias |
| BC policy vs baseline | 0.177 (192 matches), score 45 vs 88 |
| BC + 150 MAPPO iterations (22 M steps, ~35 min) | **0.609** (95 % CI 0.54-0.68, 192 matches), score 92 vs 81 |
Target of the phase: > 80 %. Not reached yet; this is a ~35 minute run, the real run is hours to days.

## What the audit found so far (`ghosttrain/audit.py`, `runs/*/anomalies.jsonl`)
- Early in training the learner conceded up to 430 foul points per match (G421 one-defender and G427 zone contacts):
  inherited from the baseline, which ignores those rules. The reward penalises them; mean fouls fell from 23 to ~0-5
  per match within 30 iterations, with occasional spikes.
- Rollouts of the trained policy still show robots stuck at walls for 40-50 % of a match (`stuck_wall`).
- No conservation, out-of-bounds or score/state inconsistencies so far.

## Foul cooldown (`game.json` model.foul_cooldown_s = 3.0)
G427/G428 contact fouls are assessed at most once per robot pair per 3 s (modelling assumption, not a manual rule).
Motivation: late-training anomalies were matches where the opponent conceded 200-324 foul points. Check on
`runs/t1/iter_000150.pt` (stochastic policy, 192 matches): cooldown 0 -> win 0.586, cooldown 3 -> 0.612 (CIs overlap,
~0.52-0.68), so the 60.9 % was *not* noticeably inflated by foul farming; the cap still removes the unrealistic spikes.

## Teammate intent (ghosttrain/intent.py)
A separate predictor guesses each ally's current goal from what the teammate can observe (its ego-frame pose,
velocity, holding flags, capabilities, time left). Its 13-way probability vector is appended to the ally rows of
the teacher observation (`others`, now 36 features; opponents get zeros). Weights: `ghosttrain/intent_weights.npz`.
- Data: baseline vs baseline, 96 envs x 1500 ticks (train) and 32 envs (validation, other seed).
- Validation: accuracy 0.80 (majority class 0.25); recall for scoring-type goals 0.95, precision 0.94.
- Limit: trained on baseline teammates only. Learner teammates behave differently: retrain on learner rollouts
  before relying on it (collection currently supports the baseline only).
- Changing the observation size means the teacher must be trained from scratch.

## Live views (ghosttrain/broadcast.py, ghosttrain/dashboard.py)
`uv run python -m ghosttrain.dashboard runs/teacher3` then
- http://localhost:8777/live : broadcast 1280x720 (field view + two robot POVs, red/blue scoreboard, training step, 10 s result card)
- http://localhost:8777/robot?id=N : one robot POV (640x480) with the AprilTags the detector sees and the robot status

Rendering happens only while a page is open. Default parallel sim processes: 6 (`TrainCfg.n_procs`).
Architecture: the match simulation runs in its own process (`ghosttrain/broadcast_sim.py`, policy on CPU) and sends a
snapshot per policy tick; the dashboard process renders at 30 fps, interpolating robot poses between ticks, and encodes
JPEG in a separate thread. Measured on this machine (12 cores, GTX 1070): broadcast 29.9 fps alone and 29.9 fps with
the robot page open (robot page 14.9 fps). Pitfalls found: a sim thread in the same process starved the renderer
through the GIL (~29 fps cap); separate Godot clients need separate ports (default port depends on pid); policy
inference on GPU is slow while Godot renders on the same GPU.

## Auto-align (ghostsim/align.py) and KF fix (ghostsim/sensors.py)
- Policy trigger: goal `align_score` (index 13). Target: nearest free L2-L4 branch of the 12 (face x side) candidates,
  skipping branches with another robot within 1.0 m. Two-phase approach (waypoint 0.5 m out, then straight in).
  Dwell 0.7 s (ours.json times `align_score`) inside 5 cm / 5 deg. Stuck for 2 s -> that candidate is given up.
- Scoring rule: L2-L4 need the robot centred within 5 cm of the branch (model.branch_lateral_tol_m). L1 (trough) is
  unchanged (whole face). The old 0.4 m tolerance remains for the face itself.
- Observation: `self` gained aligned flag, dwell progress, has-target and target offset (S = 48). Without the
  aligned flag the cloned policy chose `align_score` forever (win 0.000); with it the policy fires scores.
- KF: wheel slip (robot pushing into walls/cages/robots) made the odometry report motion while stationary. Detected
  slip now sets translation speed to 0 and adds variance. Median error 1.7 m -> 0.09 m (with vision); without
  vision the reported sigma now covers the error (test_kf).
- Baseline: approaches with align (except L1) and fires the best free level on the aligned branch.
  Measured (16 matches, auto start): score ~105/100 (before align: ~121/130 after keep-out), 51 % of coral-holding
  ticks aligned. Align approaches are slower than the old direct approach: this is a cost, not yet tuned.
- BC (runs/bc5, 120 rounds, 14 goals, S=48): goal accuracy 0.980. Evaluation (96 episodes): win 0.042, score 52 vs 100.
- BC on the tuned align baseline (runs/bc6): goal accuracy 0.975, evaluation win 0.026, score 50 vs 102. BC only clones;
  MAPPO from it is the next step (not started).
  MAPPO from this BC is NOT run yet (needs a go-ahead). Intent predictor retrained (acc 0.82, scoring recall 0.71).

## Changes after teacher4 was stopped
- Broadcast match clock: the child process advanced the match by `sim.t + elapsed`, re-counting each tick's
  overshoot (about 3x real time). Now it advances by wall time only (measured 20.0 s match per 20.0 s wall).
  Broadcast 29.9 fps, unchanged.
- Auto-align is off by default (`game.json` model.auto_align = false): policy cannot pick `align_score`, baseline
  does not align, branch tolerance is the original 0.4 m. Tests that exercise align set the flag on.
  Measured sim throughput with align on/off differs by under 5 % (the slowdown in teacher4 came from 6 sim processes
  instead of 12 plus the align logic, not from align alone).
- Robot unified with FRC 254 (2025 binder): 29.5 in frame, 4.02 m/s free speed, 52 in elevator (1.32 m), deep cage
  climb (`climb: deep`), times for coral/algae/score updated (placeholders where the binder is silent).
  Mass, friction and the exact placement times are still placeholders (`verified: false`).
  The mechanism model follows the new frame and elevator. The climber is not drawn yet.
- teacher4 checkpoints (runs/teacher4/iter_0000{10,20,30}.pt) were trained on the old 26 in robot: not compatible
  with the 254 robot physics; BC and MAPPO must be redone.

## Known issues / next steps
- Align timing (tuned): time-to-aligned median 4.6 s -> 2.7 s (dwell 0.5 s, waypoint only for large offsets, give-up
  after 1 s stuck, V_MAX 3.0, V_GAIN 4.0, YAW_GAIN 10). Standalone 12 seeds: 12/12 aligned in 1.4-7.1 s.
- Align with sensors on: the controller uses the KF estimate, but scoring uses the true pose. With the estimate
  initialised correctly, 6/8 runs reached the estimated alignment and the true distance was 6-13 cm: the 5 cm branch
  tolerance is not reliably met from the estimate. Training runs with sensors off (true pose); the live broadcast
  (sensors on) therefore shows scoring misses that training does not have.
- Goal entropy rises during PPO (0.07 -> 1.5). `ent_goal` 0.01 is probably too high; use 0.003 and compare.
- The baseline's own fouls and stuck behaviour limit the BC start; league self-play should move past it.
- Shaping targets come from the baseline's plan (potential-based, so it does not change the optimal policy in theory).
