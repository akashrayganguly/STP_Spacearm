# CHECKLIST — tick each box before committing the phase

Targets come from `DESIGN.md` §8. Record your numbers in the "got" column.

## Phase 0 — Setup
- [x] `conda --version` >= 23.10 (or Miniforge/mamba available); git installed — cloud: uv + git
- [x] Project in a path without spaces, outside OneDrive; git repo initialised or cloned — cloud: `/home/user/STP_Spacearm` (clone)
- [x] `conda env create -f environment.yml` succeeded; `conda activate spacearm`; `pip install -e .` — cloud: `cloud/setup.sh` venv (Py 3.11.13) + SessionStart hook
- [x] PyBullet GUI window opens; `torch`, `gymnasium`, `onnxruntime` import — cloud: offscreen DIRECT + TinyRenderer render OK; imports OK
- [x] First commit made — `Phase 0: cloud environment and progress log`

## Phase 1 — World model
- [x] `pytest tests\test_p1_robot_sim.py tests\test_p1_kinematics.py` -> 18 passed
- [x] FK matches PyBullet (< 1e-5 m) — got: 1.2e-7 m
- [x] Free-floating: CoM drift < 2 mm, base rotates > 0.5° in the 2 s test — got: 0.02 mm / 2.8°
- [x] `view_robot.py`: sliders move the arm; clearance turns negative when folded onto the deck — cloud: `render_robot.py` offscreen (folded: d_body −36 cm); `view_robot.py` written for the workstation, not run here (no display)
- [x] Commit

## Phase 2 — Dataset
- [x] `pytest tests\test_p2_datagen.py` -> all passed (6 passed)
- [x] `data\kinematics_dataset.npz` (200k) and `data\kinematics_test.npz` (20k) exist
- [x] Collision share: ~12 % uniform, ~20 % with boundary samples — got: 13.0 % / 21.5 %
- [x] Commit (code only)

## Phase 3 — DistanceNet
- [x] Unit tests pass before training; acceptance test passes after training (7 passed + 1 skipped -> 8 passed)
- [x] MAE all <= 1.0 cm — got: 0.68 cm ; MAE |d|<10 cm <= 2.0 cm — got: 1.22 cm
- [x] Sign accuracy >= 97 % — got: 98.3 % ; false-safe @ 3 cm <= 0.5 % — got: 0.12 %
- [x] `models\distance_net.pt` + `runs\distance_net\metrics.json` saved — cloud: `reports/distance_net/metrics.json` (CLAUDE.md §5)
- [x] Commit

## Phase 4 — Level-1 planner
- [x] `pytest tests\test_p4_trajectory.py` -> all passed (acceptance test included after training) — 10 passed
- [x] Random set: full planner success >= 95 % — got: 100 % (baseline: 92 %)
- [x] Hard set: full planner success >= 90 % — got: 100 % (baseline: 0 % by construction)
- [x] Median plan time < 0.5 s — got: 99 ms (1 thread; 114 ms incl. PyBullet verification)
- [x] Open-loop free-floating check reported: inertial miss 17.2 cm median (33.3 max), base rotation 6.4° median (12.4° max)
- [x] `runs\level1\results.json` + table saved; commit — cloud: `reports/level1/results.{json,md}` (CLAUDE.md §5)

## Phase 5 — Level-2 environment
- [x] `pytest tests\test_p5_env.py` -> all passed (includes gymnasium `check_env`) — 15 passed
- [x] `play_env.py`: obstacles appear mid-episode; prior-only collides with some of them — cloud: offscreen MP4/PNG; obstacles appear (72 % of d = 1 episodes); the prior stalls in front of them, 0/40 collisions (spawn rule keeps obstacles from being born inside the arm)
- [x] Prior-only baseline at d = 0 and d = 1 recorded — got: 72.5 % success, 0 % collisions / 35.0 % success, 0 % collisions (40 episodes each)
- [x] Commit

## Phase 6 — PPO-Lagrangian
- [x] `pytest tests\test_p6_ppo_lag.py` -> all passed (8 fast)
- [x] `pytest -m slow tests\test_p6_ppo_lag.py` -> toy task solved (success >= 80 %, cost <= 1.5) — 100 % / 0.86 (after making the curriculum opt-in)
- [x] 200k-step smoke run finished without errors (7.5 min)
- [x] Full run finished; curves saved (success, cost, lambda, collision vs steps) — 3.0 M steps, `reports/ppo_lag/curves.png`
- [x] Periodic deterministic evaluation logged next to the prior-only score; best checkpoint kept in `models\ppo_lag.pt`
- [x] Best evaluation success 85 % (20-episode selection; 67 % on 100 unseen episodes) vs prior 45 % (47 % on the same 100) — RL beats the prior
- [x] Commit

## Phase 7 — Shield, evaluation, deployment
- [x] `pytest tests\test_p7_shield_export.py` -> all passed (9 fast)
- [x] Level-2 table S1–S4 x {prior, prior + shield, RL, RL + shield} saved — S4 RL + shield: success 59 % (prior: 45 %), collisions 2 % (target <= 3 %) — 100 episodes, `reports/level2/results.md`
- [x] Actor exported to `.npz` and `.onnx`; latency p50/p99 recorded — got: NumPy 0.012 / 0.029 ms, ONNX 0.008 / 0.024 ms, full step 4.5 / 7.3 ms
- [x] Demo video `videos\demo.mp4` shows re-routing around a surprise obstacle — cloud: `reports/demo.mp4` (seed 10017: prior times out, RL + shield arrives at 11.1 s)
- [x] `pytest -m "slow or not slow"` -> all passed or the failing targets are explained in the README — 76 passed, 1 failed (S4 acceptance on 40 episodes: 2/40 collisions at the 5 cm shield margin; explained, 6 cm option pending)
- [ ] README results section written; `git tag v1.0` — README done; tag `main` after the merge (shield-margin decision pending)

## Phase 8 — Stretch (optional)
- [ ] Dual-arm tests written first; env with `n_arms = 2`; MAPPO-Lagrangian trained; comparison table
