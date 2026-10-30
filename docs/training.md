# Teacher training (phase 2)

Everything runs on the P520: 12 sim worker processes (CPU) + the learner on the GTX 1070
(PyTorch 2.14.1 + CUDA 12.6; the wheel ships sm_60 SASS, which runs on the 1070's sm_61 - checked with fp32/fp16 matmuls).

## Pipeline
```
uv run python -m ghosttrain.bc --out runs/bc9                        # 1. clone the baseline with DAgger (~30 min)
sh tools/run_forever.sh runs/bc9/bc_net.pt runs/teacher8              # 2. MAPPO self-play + league, restarts on crashes
# (or one bounded run: uv run python -m ghosttrain.train --init runs/bc9/bc_net.pt --iters 2000 --out runs/teacher)
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

## Foul cooldown (`game.json` model.foul_cooldown_s = 3.0) - replaced, see "Fouls: referee model, pins, AUTO rules"
G427/G428 contact fouls were assessed at most once per robot pair per 3 s (modelling assumption, not a manual rule).
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
- http://localhost:8777/live : broadcast 1920x1080 (field camera + two follow cameras, official-style overlay, result card)
- http://localhost:8777/robot?id=N : one robot POV (640x480) with the AprilTags the detector sees and the robot status

Live view, current version (supersedes the 1280x720 numbers below):
- Video 1920x1080, every view rendered 1:1: field camera 1920x720 high behind the scoring table (54 deg lens, 14 m back,
  13 m up, 32 deg down; framing searched so the near edge fills ~90 % of the width), and two 960x360 follow cameras on
  tripods outside the field (scoring-table side, 4.2 m up) that pan to a BLUE / RED robot with a lag and zoom to keep
  ~3.6 m around it (re-picked every 10 s). MSAA 2x + FXAA (`fxaa` in the renderer configure), detailed field_hi.glb.
- No crowd, dark background, fixed lighting (`broadcast.live_appearance`); training cameras keep their randomisation.
- The scoreboard (alliance names, scores, clock, AUTO / TELEOP / ENDGAME, CORAL / ALGAE / BARGE points), tile labels
  (robot, goal, held piece, speed) and the result card (totals, LEAVE / CORAL / ALGAE / BARGE / PENALTY, winner, wins)
  are an HTML overlay (`ghosttrain/live_page.py`) fed by `/api/live` (`broadcast.live_state`), not pixels in the video.
- The two Godot instances render in parallel; the static part of each robot model is cached (mechanism 3 ms / frame).
  Measured (GTX 1070, no training running): 27.8 fps (MSAA 4x: 24.9, serial renders: 18.2). Per frame the field view
  costs ~12 ms of GPU render + readback in Godot and ~8 ms to move the 4 MB raw image to Python.

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
- teacher4 checkpoints (runs/teacher4/iter_0000{10,20,30}.pt) were trained on the old 26 in robot: not compatible
  with the 254 robot physics; BC and MAPPO must be redone.

## Live view: FRC 6328's robot from its CAD
- The live view draws our robot (and every other robot, scaled to its frame) from FRC 6328's 2025 CAD
  (Onshape export `6328-25B-0000.gltf`): `tools/prepare_robot.py` turns it into `game/2025/visual/robot.glb`
  (not committed, like the field; ~150 k triangles, 7 MB, ~3 min) with one mesh node per moving body: base
  (drivetrain, hopper, fixed elevator stage), stage2, stage3, carriage, effector, climber. Without robot.glb the
  renderer draws a box model of the same layout. Regenerate with:
  `uv run python tools/prepare_robot.py ~/下載/6328-25B-0000.gltf` (needs the dev dependency DracoPy).
- Layout (measured on the CAD, constants in `ghostsim/mechanism.py`): full-width 3-stage elevator (tubes at the frame
  edge) leaning 8 deg forward: the bottom stays where it is and the top comes out over the front bumper, so at L4 the
  coral is 0.23 m further forward than at the bottom (tip 6 cm past the bumper, over the branch). Effector on the
  carriage, pitched by a wrist about its pivot: coral passes through its channel (fed from the hopper belt at the back,
  out over the front), algae is held in front of it against its front wheel and top roller; there is no separate
  algae arm (floor algae: effector nose down at h = 0). Hopper at the back (the robot backs into the station, coral
  drops in and the belt feeds the effector). Deep-cage climber at the back. 0.94 m starting height (R104), 1.60 m
  elevator travel, every pose inside R105.
- Joints (`ghostsim/mechanism_state.py`): elevator h along the tilted axis (stage 2 h/3, stage 3 2h/3, carriage h),
  wrist, climber; joint zero = the CAD pose (also the carry pose: coral channel in line with the hopper belt).
  They move only while the rules dwell on the current goal (score pose at 70 % of the dwell, the piece leaves at
  100 %, pose held 0.35 s after a release), never while driving. Score poses put the coral centre at each level with
  the coral pitch of the branch; reef algae lifts the effector to the ball; net: full height, effector tips up; floor
  intake: nose down during the intake dwell. Rate limits: elevator 3 m/s, wrist 300 deg/s, climber 120 deg/s.
- Renderer: the "mech" array carries per robot either box-model solids or the six body transforms (kind 3 + body,
  full linear map incl. the frame scale) plus bumpers and held pieces (`mech_slots(..., cad=True)`); Godot loads
  robot.glb once (`robot="auto"` in `RenderClient.configure`, live view only; training cameras do not draw
  mechanisms). 64 slots per robot. `robots/ours.json held_offset_m` = the carry pose of the effector.
- Renders without a mechanism array (robot cameras in training, `ghostsim/env.py`, replays) draw the randomised box
  robots of the appearance (body, alliance bumpers with the team number, boxes on top) again; the mechanism hides
  them. From eb3a764 (first live-view mechanism) until this fix they were never built, so those images showed no
  robots at all. Regression test: `tests/test_render_robots.py` (runs when Godot is installed).
- `tools/mech_view.py` (box model, PNG poses) and `tools/mech_anim.py` (video of the real tracker on a scripted
  station -> L4 -> floor algae -> processor -> climb sequence).
- Earlier versions (before 2026-10-29 evening): a box model after 6328 pictures with a front algae arm, a funnel on
  posts and a vertical elevator in the middle.

## Result card fix
- The worker auto-resets when a match ends, so the card read 0-0. `_Worker.step` now also returns `final_cat` (category
  points before the reset) and the broadcast uses `final`/`final_cat`. Verified on a full match: 219-97.

## Simulator realism pass (world-championship level)
Checked against the 2025 manual (Game Rules V11, Game Details V13) and the 2025 Einstein finals (281-295 points).
- Floor ALGAE could never be picked up: the bumper pushed the ball (r 0.206 m) out of the 0.15 m intake. The intake
  reach is now measured to the near edge of the piece, and a piece inside the intake of a robot that is intaking it
  moves with the robot (rollers hold it). The baseline used to lose time chasing balls it could not take:
  baseline vs baseline (24 matches, auto start) 145 -> 238 points per alliance with this fix alone, floor-algae pickups
  0.6 -> 14 per match.
- Placing / receiving pieces needs the robot stopped (`model.stop_goals`, 0.3 m/s, 1 rad/s); before, L4 scored while
  driving past the reef at 0.85 m/s. Floor intakes and climbing still work on the move.
- auto_align off: the robot's own drive-to-pose time is charged as dwell (`model.virtual_align`: 0.15 s +
  distance / 1.0 m/s + heading / 3 rad/s; lateral distance to the nearest free branch of the level).
- Actions can fail (`robots/*.json success_p`, ESTIMATES: 254 L4 0.96, others L4 0.85-0.98, ...). A miss drops the piece
  on the floor next to the robot (coral that finds no free floor slot is counted in `coral_lost`).
- Together (baseline vs baseline): coral per alliance 26.9 -> 23.1, ~6 misses per match, total 229 per alliance.
- L1 trough holds 6 per face (2 layers of 3, manual 6.5.1 sets no count limit); render poses for trough coral and
  staged reef algae now follow the CAD (they floated in front of the reef / intersected the L4 branches).
- Baseline: also skips floor algae that lie inside an opponent-cage keep-out disc (more chasing near the barge
  since the intake fix raised opponent-cage contacts to ~3 per match).
- All earlier checkpoints (BC and teacher) were trained on the old rules and need retraining.

## Partners and opponents at Championship level (robots/others.json)
Real data: 2025 FIRST Championship NEWTON + HOPPER qualifications from the public frc-events match pages
(`tools/champs_data.py fetch` / `summary`; 250 matches, 150 teams; parsed points match the official auto / teleop
points in 246 of 250 matches). Per alliance: 180 points without fouls (p10 132, p90 231), 29 coral (L4 10.5 of 12),
net 4.5, processor 1.6, barge 20; robots end deep 48 %, parked 45 %, shallow 0 %, none 6 %. Pieces per team (OPR)
relative to the 97th-percentile team: p10 0.30, p25 0.45, p50 0.61, p75 0.75, p90 0.85.
- Before: others were drawn independently and could be faster than our robot (5.0 m/s, 0.7x its mechanism times);
  baseline robots of the others profile scored about as much as our robot.
- Now: one quality q ~ U(0, 0.95) per robot and reset; times, alignment skill (`align_scale`, multiplies the
  drive-to-pose time), effective top speed, capabilities and success_p all follow q (`by_quality`,
  [value at q=0, value at q=1, gamma]). q = 1 is our robot.
- Calibrated with `tools/calib_others.py solo`: one robot alone, full match, baseline driver, pieces relative to our
  robot. q 0.10 / 0.25 / 0.50 / 0.75 / 0.90 -> 0.29 / 0.41 / 0.57 / 0.76 / 0.95 (real 0.30 / 0.45 / 0.61 / 0.75 / 0.85).
- Full matches (`tools/calib_others.py match`, 96 baseline matches): net 4.5, processor 1.2, barge 19.9, deep 0.45,
  L4 10.0 per alliance (real 4.5, 1.6, 20, 0.48, 10.5); others' pieces relative to ours p50 0.61 (real 0.61).
  Absolute coral (15 vs 29) and score (128 vs 180) stay below real: the baseline is a mediocre driver (congestion at
  the same face / station, slow docking). The calibration is relative on purpose, so a better policy raises every
  robot together.
- Baseline strategy closer to the Champs teams: net before processor (`prefer_net`), algae biases 3.0 (floor) /
  1.5 (reef). Before: processor 9.4 per alliance and our robot spent its time on algae.
- Baseline bug: driving straight at a reef approach point from beside the reef wedged robots on a reef vertex for
  the rest of the match. Reef-face targets now go through a point 0.7 m in front of the face first; robot-time
  stalled for > 2 s fell from 18 % to 9 % (24 matches). Remaining stalls are two robots waiting for the same face or
  station (no coordination in the baseline).
- Also fixed: a robot's kinetic friction was computed from a second, independent draw of its static friction.

## Fouls: referee model, pins, AUTO rules (game/2025/rules.py `_fouls`, `_pins`)
Checked against the 2025 manual section 7 (V11).
- Referee model for contact rules (G403, G405/G418, G427, G428): a contact starts a new interaction only after the two
  robots (or robot and cage) were apart for `foul_sep_s` = 1 s; one foul when it starts, one more per `foul_repeat_s` =
  10 s of continued contact. The old 3 s per-pair cooldown still let flickering contact (chattering contact flags in a
  scrum) draw a foul every 3 s per pair: up to 2778 conceded points in one teacher7 self-play match. Tests: six separate
  rams = six fouls, contact flickering every 0.1 s for 8 s = one, 25 s of continuous contact = three.
- G425 pins (new): a robot is pinned while it touches an opponent, is held by the field (wall, reef, cage, barge
  support) or by another robot, and moves slower than 0.3 m/s, and the pinning robot is free of the field. The count
  runs while pinning and survives short breaks; it ends when the robots are 6 ft apart or either is 6 ft from where the
  pin started, for more than 3 s, or when the pinner gets pinned. MINOR past 3 s, then a MAJOR for every further 3 s.
- G403 (new): in AUTO, a robot fully across the barge zone (the G421 test) may not contact an opponent robot: MAJOR.
- G405 (new): in AUTO, opponent cage contact is a MAJOR foul too (G418 covers TELEOP).
- `State.foul_rule` (N, 2, 6): points conceded per alliance and rule (G403, G405/G418, G421, G425, G427, G428), sums to
  `fouls`. The worker returns it at the end of a match (`final_rule`) and `anomalies.jsonl` logs `fouls_by_rule`;
  the anomaly thresholds are now score > 350 or fouls > 60 (250 / 40 flagged almost every match).
- Not modelled: G435 (human player stores at most 4 algae), G419/G420 (anchors, net contact), G422-G424, G426.

## Command filter between the policy and the chassis (ghostsim/motion.py `filter_stick`)
- The policy's stick (10 Hz, sampled with exploration noise during training) went straight into the swerve motion
  layer, so the robots jittered. Now the robot code filters it every control period (50 Hz), like a WPILib LinearFilter
  + SlewRateLimiter: first-order low-pass (tau 0.15 s), then the change is limited to 8 m/s^2 (translation, as a
  vector) and 30 rad/s^2 (rotation). Parameters: `robots/ours.json control.cmd_*` (others inherit them).
- Auto-align (`align_score`) drives the robot with its own controller and bypasses the filter; the filter continues
  from the executed command, so there is no jump when the policy takes over again.
- The filter state is part of the observation (`self` gained 3 features, S = 48 -> 51; the student's sensor
  observation has `cmd`), so the problem stays Markov: the policy knows its command is still ramping.
  Checkpoints trained before this change do not load (input size).
- Java port contract: `tests/golden/cmd_filter_v1.json` (`tests/make_golden.py`; motion_v1.json is unchanged).
- Measured, steady half-speed command + exploration-size noise (std 0.11): acceleration jitter 2.65 -> 1.34 m/s^2,
  yaw-rate std 0.38 -> 0.19 rad/s; slow sideways drift (3.5 cm) is unchanged (low-frequency, not filtered).
  Baseline vs baseline score 127.5 -> 122.6 per alliance; the solo calibration of robots/others.json is unchanged
  within noise (0.30 / 0.36 / 0.60 / 0.75 / 0.94).

## Optimisation fixes for teacher8 (2026-10-29; teacher5-7 plateaued after ~iter 500)
Measured on teacher7 (iter 6060): the stick log std had collapsed -1.6 -> -4.2, ~25 % of PPO samples were clipped,
all by the stick ratio (the joint goal x stick ratio hid the goal head behind it), the actor gradient norm was ~11
against max_grad 0.5 (every update scaled by ~0.05), the goal entropy bonus was averaged over decide ticks while the
policy loss was averaged over all ticks (10x relative weight: goal entropy rose 0.33 -> 1.35), and the learner was
always blue = always with our robot (self-play blue won ~70 %, the baseline win rate was biased). Changes:
- Stick std: soft floor at 0.03 (`net.LOG_STD_MIN`, `TeacherNet.stick_log_std()`, identity well above it); a warm
  start begins at 0.05 (`init_log_std`). Not more: scoring needs the robot stopped (<= 0.3 m/s) and stick noise costs
  points. bc9 vs the baseline (96 matches each): deterministic 117-116, std 0.05 131-117, 0.10 107-120, 0.20 31-116
  (0.20 = the old BC start, which is why the first calibration run scored ~20 per match).
- PPO: one clipped surrogate per head (`clip` for the stick on every tick, `clip_goal` for the goal on decide ticks),
  each averaged over its own samples; goal entropy averaged over decide ticks too, `ent_goal` 0.005 per decision.
- Gradients: actor and critic clipped separately (`max_grad` 5, `max_grad_critic` 2, so only outliers are cut);
  the pre-clip norms are logged (`gn_actor`, `gn_critic`) and the dashboard plots them. With Adam a constant clip
  factor does not shrink the steps (it normalises per parameter), so "updates scaled by 0.05" overstated it; what
  the joint clip did do was let critic spikes shrink the actor's step in the same minibatch.
- Memory: gradients are accumulated over micro-batches of 2048 (`micro`, same update as one 8192 minibatch): the
  8192 forward pass ran out of GPU memory with the desktop apps holding ~2.7 GB.
- Sides: the learner's alliance is drawn per episode (`side_swap`); the baseline / pool snapshot plays the other one.
  Stats are from the learner's side (`score_learner`, `score_opp`) and the baseline win rate is split by whether our
  robot (slot 0, blue) was on the learner's side (`win_baseline_with_ours` / `_without_ours`).
- League: the warm start stays in the pool as an anchor (never evicted), pool 8, snapshot every 25 iterations.
- Learning rates per part: actor encoder / trunk / stick head 2e-5, goal head 1e-4, critic 1e-4. With the stick std
  at 0.05, one Adam step at 1e-4 moved the stick distribution by KL 0.1-0.5 (a 10x smaller rate on the stick head
  alone did not help: the shared trunk moves it); at 2e-5 the stick KL is ~0.007 per iteration. On top, the actor
  stops for the rest of an iteration once a minibatch exceeds 1.5 x `target_kl` (0.02); the critic keeps training
  (`actor_frac` = share of minibatches that moved the actor).
- `tools/run_forever.sh <bc_net.pt> <out>` starts a fresh run with `--init` and resumes from the newest snapshot after
  a crash; extra arguments go to `ghosttrain.train`.
- Behaviour cloning with DAgger (`ghosttrain/bc.py`, runs/bc9): bc8, cloned from baseline-driven data only, scored 24
  vs 112 with goal accuracy 0.996: the observation holds the filtered stick (chassis filter state) and the net learned
  to copy it (mean |stick| 0.07 vs the baseline's 0.23 on its own states; robots stalled 0.3-3 m short of targets).
  From round 30 on, a growing share of envs (up to 80 %) is driven by the net while the baseline labels every robot
  (`label_all` in the workers); the last 3 rounds are trained together. bc9: 117 vs 116 deterministic.

## teacher8 evaluation and the explained-variance bug (2026-10-30)
- iter 1540 (192-200 matches each): vs baseline win 0.83 (CI 0.77-0.88), 156 vs 117 (deterministic and stochastic
  alike; the bc9 start was 0.46, 117-116). Head-to-head: vs iter 500 0.53-0.56 (156 vs 148-150), vs iter 1000 0.54
  (159 vs 159): most of the gain came before iter ~500-1000. In head-to-heads the team with our robot (blue) wins
  ~0.8, so compare scores rather than win rates there.
- The logged critic explained variance (~0.155 all run) was wrong: `_value_of` evaluated only the first 16384 samples
  and counted the rest (~83 % at n ~ 100k) as 0. Measured correctly on fresh rollouts of iter 1700: 0.94-0.97. Fixed,
  with a test. The critic is not the bottleneck.
- What barely moves is the goal head: goal entropy 0.13-0.16 the whole run (one goal ~97 % of the time), goal KL
  ~3e-4 per iteration (stick ~8e-3). A/B from iter 1710 (interleaved 50-iteration blocks, same machine load): control
  vs `--ent_goal 0.02 --lr_goal 3e-4` (runs/teacher8_goal).

## Known issues / next steps
- Align timing (tuned): time-to-aligned median 4.6 s -> 2.7 s (dwell 0.5 s, waypoint only for large offsets, give-up
  after 1 s stuck, V_MAX 3.0, V_GAIN 4.0, YAW_GAIN 10). Standalone 12 seeds: 12/12 aligned in 1.4-7.1 s.
- Align with sensors on: the controller uses the KF estimate, but scoring uses the true pose. With the estimate
  initialised correctly, 6/8 runs reached the estimated alignment and the true distance was 6-13 cm: the 5 cm branch
  tolerance is not reliably met from the estimate. Training runs with sensors off (true pose); the live broadcast
  (sensors on) therefore shows scoring misses that training does not have.
- Goal entropy rose during PPO (0.07 -> 1.5) with the old per-tick averaging: see "Optimisation fixes for teacher8".
- The baseline's own fouls and stuck behaviour limit the BC start; league self-play should move past it.
- Shaping targets come from the baseline's plan (potential-based, so it does not change the optimal policy in theory).
