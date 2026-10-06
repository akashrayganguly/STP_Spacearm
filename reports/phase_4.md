## Phase 4 — Level-1 planner (TrajNet + refinement + IK polish + verification): DONE

Built:
* `src/spacearm/models/traj_net.py` — Bernstein basis, `squash`/`unsquash` (tanh into the joint limits), `make_control_points` (P0 = P1 = start, P6 = P7 = goal → rest-to-rest; every point inside the limits → the whole curve is), `curve`, `TrajNet` ([sin q, cos q, target] → 512×3 SiLU → u_goal + 4 offsets, last layer zero-initialised; limits stored as buffers).
* `src/spacearm/losses.py` — `reach_loss`, `collision_loss`, `smoothness_loss`, `limit_violation`, `level1_loss` (100·reach + 1000·collision + 0.1·smoothness).
* `src/spacearm/planner.py` — `free_pool`, `sample_queries`, `polish_ik` (DLS on the exact Jacobian), `Level1Planner.plan` (network → 30 Adam refine steps on the Bezier parameters → IK polish → 100 points), `verify_trajectory` (exact PyBullet), `plan_and_verify` (one retry with 100 refine steps), `ik_baseline` (PyBullet null-space IK + straight joint line), `execute_open_loop` (plays a path on the floating spacecraft).
* `scripts/train_traj_net.py` — self-supervised training on random (start, goal) pairs from the 124k collision-free training configs; network-only validation every 500 steps → `models/traj_net.pt`, `reports/traj_net/{progress.csv, summary.json, training.png}`.
* `scripts/eval_level1.py` — random + hard sets × 4 methods + open-loop free-floating check → `reports/level1/{results.json, results.md}`.

Commands I ran:
* `python -m pytest tests/test_p4_trajectory.py -v` (before training) — 9 passed, acceptance skipped — Windows: `pytest tests\test_p4_trajectory.py -v`
* `python scripts/train_traj_net.py --steps 100` — smoke run (20 ms/step → 10k steps ≈ 3.5 min, under the 10-min command cap, so no background run needed)
* `python scripts/train_traj_net.py` — 10k steps, 3 min 15 s on 4 threads — Windows: `python scripts\train_traj_net.py`
* `python -m pytest tests/test_p4_trajectory.py -v` (after training) — 10 passed, acceptance included
* `python scripts/eval_level1.py --n-random 10 --n-hard 3 --n-float 2` — 6 s smoke run of the evaluation
* `python scripts/eval_level1.py` — full evaluation, 1 min 45 s (planner timed on 1 thread) — Windows: `python scripts\eval_level1.py`
* `python -m pytest tests/test_p1_*.py tests/test_p2_datagen.py tests/test_p3_distance_net.py tests/test_p4_trajectory.py -q` — 42 passed

Tests: `test_p4_trajectory.py` **10 passed, 0 skipped** (after training); Phases 1–4 together 42 passed.

Results vs targets (DESIGN §8). Queries come from the unseen uniform test pool; every success is verified in PyBullet (clearance > 0 at 100 points, reach < 2 cm, within limits):
| Metric | Target | Validation build | Got |
|---|---|---|---|
| Full-planner success, random set (200) | ≥ 95 % | 100 % | **100 %** |
| Full-planner success, hard set (100, baseline 0 %) | ≥ 90 % | 95 % | **100 %** |
| Median plan time (1 CPU thread) | < 0.5 s | ~0.1 s | **99 ms** (114 ms incl. PyBullet verification) |
| Open-loop on the floating spacecraft: inertial miss, median / max | report | ~13 / 34 cm | **17.2 / 33.3 cm** |
| Open-loop base rotation, median / max | report | ~6° | **6.4° / 12.4°** |

Full table (`reports/level1/results.md`):
| Set | Method | Success | Collisions | Median reach error | Median plan time |
|---|---|---|---|---|---|
| random (200) | Baseline (PyBullet IK + line) | 92.0 % | 3.5 % | 0.14 cm | 1 ms |
| random (200) | Network only | 85.0 % | 1.5 % | 1.02 cm | 5 ms |
| random (200) | Network + IK polish | 98.0 % | 1.5 % | 0.00 cm | 8 ms |
| random (200) | Full planner | **100.0 %** | 0.0 % | 0.00 cm | 99 ms |
| hard (100) | Baseline | 0.0 % | 100.0 % | 0.21 cm | 1 ms |
| hard (100) | Network only | 66.0 % | 6.0 % | 1.32 cm | 2 ms |
| hard (100) | Network + IK polish | 93.0 % | 7.0 % | 0.00 cm | 28 ms* |
| hard (100) | Full planner | **100.0 %** | 0.0 % | 0.00 cm | 96 ms |

\* A transient: re-timed on 60 queries twice it is a steady 8 ms. Wall-clock timings on the shared cloud VM can jump.
Hard set: 100 baseline-colliding queries found in 2,170 random queries (4.6 %; lesson 10 said ~5 %). Retry needed: 0.5 % (random), 0 % (hard).

How to read it:
* **Each stage earns its place.** The network alone gets the hand to ~1 cm (85 %, failing mostly on the 2 cm reach test). The IK polish brings reach to < 1 mm. Refinement on the DistanceNet removes the remaining collisions (7 % → 0 % on the hard set). PyBullet verification makes every reported success real.
* **Free floating (motivates Level 2):** the plans are perfect in the body frame (tracking error 0.00 cm), but the spacecraft rotates 6.4° in reaction, so the hand misses an inertially fixed target by 17 cm (median). Level 2 closes this loop.

Decisions / deviations and why:
* **Three new config keys** (values from DESIGN / lessons, previously missing from the config): `traj.polish_damping: 0.01` (DESIGN §5.4), `traj.grad_clip: 1.0` (lesson 9), `traj.weight_decay: 0.01` (AdamW default, pinned). Additions only; no test changed.
* **Refinement uses `torch.autograd.grad`** with respect to the Bezier parameters only, so planning never touches the networks' `.grad`.
* After the polish, the goal control points are set to the polished goal exactly, and `Q[0]` to the exact float64 start (no float32 round-off at the ends).
* **`execute_open_loop` time-parameterisation:** each segment runs at the joint speed limit (slowest joint), joint-position feedback at 10 Hz, plus 2 s to settle. The open-loop median miss (17 cm) is above the validation build's ~13 cm; the max (33 cm) matches. It's a 20-plan sample and depends on path durations; it's reported, not targeted.
* Results go to `reports/level1/` and `reports/traj_net/` instead of `runs/` (CLAUDE.md §5). `models/traj_net.pt` (2.2 MB) is committed.

Next: Phase 5 — Level-2 environment (`control.py` reflex prior, `safety/shield.py` geometry helpers, `envs/space_reach_env.py`, `envs/toy_env.py`, `rl/vec_env.py`, `scripts/play_env.py`). Reply "go" to start.
