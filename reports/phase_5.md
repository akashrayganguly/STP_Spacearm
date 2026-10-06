## Phase 5 — Level-2 environment (obstacles, faults, free-floating base, reflex prior, vector envs): DONE

Built:
* `src/spacearm/control.py` — `reach_prior`: DLS reaching on the vision TCP→target vector (capped at 0.15 m/s) + null-space push away from the spacecraft + obstacle repulsion reflex; output scaled as a whole into [−1, 1]. Optional precomputed Jacobian.
* `src/spacearm/safety/shield.py` — `sphere_capsule_distance`, `obstacle_clearance` (the `SafetyShield` comes in Phase 7).
* `src/spacearm/envs/space_reach_env.py` — `SpaceReachEnv` (gymnasium):
  * free-floating base, target fixed in the inertial frame, residual command `clip(prior + 0.5·fade·a)`;
  * actor obs (1, 56) / critic obs (90,) exactly as in DESIGN §6.1 (`SL` slice table, `ACTOR_DIM`, `CRITIC_DIM`);
  * cost, termination, truncation, difficulty-scaled obstacles/faults/noise;
  * `place_obstacle`, `set_difficulty`, `set_shield` hook, offscreen `render()`; `shaped_reward`.
* `src/spacearm/envs/toy_env.py` — `PointHazardEnv`: 2D point, hazard disc on the straight line (straight path: cost 7.0/episode; scripted detour: cost 0 at slightly lower return).
* `src/spacearm/rl/vec_env.py` — `SyncVecEnv`, `SubprocVecEnv` ("spawn"): same-step auto-reset, episode statistics (`pop_episodes`), final observation in `infos[i]["final_obs"]` for truncation bootstrapping.
* `src/spacearm/sim.py` (+ `tcp_velocity`), `src/spacearm/kinematics.py` (+ `jacobian_from`, `capsules_from`: reuse one FK pass) — additive.
* `scripts/play_env.py` — prior-only episodes, per-episode print-out, summary JSON, offscreen MP4 + PNG frame strip (`--gui` for the workstation).

Commands I ran:
* `python -m pytest tests/test_p5_env.py -v` — 15 Phase 5 contracts incl. gymnasium `check_env` and subprocess-vs-sync equality — Windows: `pytest tests\test_p5_env.py -v`
* scripted-policy check of the toy task — straight line: success 100 %, cost 7.00; detour: success 100 %, cost 0.00
* `python scripts/play_env.py --difficulty 1.0 --episodes 3` — smoke run + first renders — Windows: `python scripts\play_env.py --difficulty 1.0 --episodes 3`
* `python scripts/play_env.py --difficulty 0 --episodes 40` and `... --difficulty 1 --episodes 40 --render 2` — prior-only baselines → `reports/level2/prior_d0.json`, `prior_d1.json`, `play_d*_ep*.mp4/.png`
* cProfile of `env.step`, 4-worker `SubprocVecEnv` throughput test, before/after comparison of the FK refactor (40/40 episodes identical)
* `python -m pytest tests/test_p1_*.py ... tests/test_p5_env.py -q` — 57 passed

Tests: `test_p5_env.py` **15 passed, 0 skipped**; Phases 1–5 together 57 passed.

Results (prior only = zero residual, 40 fixed-seed episodes each):
| | Validation build | Got |
|---|---|---|
| d = 0 (nominal): success / collisions / mean cost | ~80 % / — / — | **72.5 % / 0 % / 0.55** |
| d = 1 (full): success / collisions / mean cost | ~45 % / 3–8 % / — | **35.0 % / 0 % / 1.00** |
| d = 1: episodes with an obstacle; success when one appeared | — | 72 %; 17 % |
| Env step (1 process) / 4 subprocess workers | 1.5–2.5 ms (2 vCPU) / 300–440 steps/s with PPO | **3.6 ms / 726 env steps/s** (before PPO overhead) |

Both success rates are about one standard error (±7 %) below the validation build.

Why the prior fails (what Phase 6 must learn, and what it can't):
* **Nominal (d = 0), 11 failures:**
  * **8 are physically unreachable by the end.** The targets started near full stretch (0.98–1.07 m from the shoulder; max reach 1.09 m). As the arm reaches, the spacecraft recoils (6–12° rotation plus translation), and the inertially fixed target ends up 1.11–1.23 m away. No controller can win these as designed.
  * **The other 3 are reachable**, but joints get pinned at their limits (the DLS prior ignores limits). A residual policy can learn to fix this.
* **Full difficulty:** 24 of 26 failures are episodes with an obstacle. The repulsion reflex keeps the arm safely 8–12 cm away but hovers in front of it (see `play_d1_ep1.png`): the local minimum DESIGN D10 describes. Getting around is the residual policy's job.

Decisions / deviations and why:
* **Obstacle spawn rule** (DESIGN says only "between TCP and target ±5 cm"):
  * centre at 25–75 % of the TCP→target segment, ±5 cm per axis;
  * re-drawn (up to 20 tries) if it would start within 5 cm of the arm or cover the target; otherwise no obstacle that episode.
  * Why: an obstacle born inside the arm is an unavoidable collision and teaches nothing. Likely the reason for 0 % prior collisions vs the validation build's 3–8 %.
* **Encoder slip:** at a uniformly random step of the episode, ±3° on one random joint.
* **Model-based features use the measured angles only:** the obstacle centre is FK(q_meas) + vision offset; the self-clearance is the DistanceNet on q_meas. The dynamic sim is never teleported mid-episode, because `set_q` would zero velocities. In `self_clearance: "sim"` mode (tests only), the self-clearance is the PyBullet truth with a zero gradient.
* **Reward terms:** the action-rate and residual-size terms use the policy action (the residual). Progress uses the true TCP distance (training-only signal).
* **Success** uses the instantaneous TCP speed from PyBullet.
* **Observation scales:** gyro / 0.1 and clearances / 0.1 are module constants (part of the observation layout); "no obstacle" is encoded as clearance 0.30 m (the clip value) with a zero gradient.
* **Performance:** one FK pass per observation serves the prior's Jacobian and the obstacle features (+10 % throughput, identical episodes).

⚠️ Decision needed before Phase 6 — unreachable targets. About 20 % of nominal episodes can't be won (8/40 here), which caps success near 80 % and gives the learner episodes with no signal.
* (A) Keep DESIGN as is. RL vs prior stays fair (same seeds); absolute success numbers include the unwinnable episodes.
* (B) Sample targets only within a fraction of the arm's reach from the shoulder (new key `env.target_reach_frac`, e.g. 0.85). This means a DESIGN §6.1 + config change; no test changes.

My recommendation is (B) with 0.85, re-measuring the prior baselines before training. It makes success numbers mean "reached a reachable target" and makes the S4 stretch goal (≥ 70 %) attainable. But it departs from the validation build's setup, so it's your call.

Next: Phase 6 — PPO-Lagrangian (`rl/ppo_lag.py`, `scripts/train_ppo_lag.py`, resumable and time-boxed; toy slow test first). Reply "go" — with (A) or (B) for the target question.
