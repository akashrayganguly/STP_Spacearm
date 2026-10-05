## Phase 6 — PPO-Lagrangian residual policy: DONE

Built:
* **Target reachability, option (B) agreed:** new config key `env.target_reach_frac: 0.85`. Targets lie within 85 % of the arm's reach from the shoulder (joint 2). DESIGN §6.1 updated, plus a new test `test_targets_lie_within_the_reach_fraction`.
* `src/spacearm/rl/ppo_lag.py`:
  * Building blocks: `RunningMeanStd`, `compute_gae`, `LagrangeMultiplier`, `Actor` (56→64→64→7 tanh mean, log-std −1.6, 8.3k parameters), `Critic` (90→256→256→1, reward and cost).
  * `PPOLagAgent` (`act`, `save`, `load`), `evaluate`, plus `evaluate_vec` (parallel evaluation).
  * `train`: resumable via a `resume=` checkpoint dict and time-boxed via `max_minutes=`; `callback(ctx)` after every update.
* `scripts/train_ppo_lag.py`:
  * segments (`--resume`, `--max-minutes`), checkpoint `models/ppo_lag_ckpt.pt` every 10 updates and on exit;
  * deterministic eval every 25 updates on 20 fixed episodes at full difficulty (seeds 20,000+, deliberately not Phase 7's), against a prior-only reference;
  * best model `models/ppo_lag.pt` (success up without more collisions);
  * outputs: `reports/ppo_lag/{progress.csv, eval.csv, curves.png, summary.json, independent_check.json}`.

Commands I ran:
* `python scripts/play_env.py --difficulty {0,1} --episodes 40` — prior baselines re-measured with option (B)
* `python -m pytest tests/test_p6_ppo_lag.py -v` — 8 fast tests passed — Windows: `pytest tests\test_p6_ppo_lag.py -v`
* `python -m pytest -m slow tests/test_p6_ppo_lag.py -v` — toy task, the gate before arm training. **Failed first** (12 % success), fixed, then passed in 46 s — Windows: `pytest -m slow tests\test_p6_ppo_lag.py -v`
* `python scripts/train_ppo_lag.py --n-envs 4 --steps 200000 --tag smoke --fresh` — 200k-step smoke run, 7.5 min, no errors
* `python scripts/train_ppo_lag.py --n-envs 4 --resume --max-minutes 8` × 13 — full 3 M-step run, committed after every segment — Windows (one go, no time box needed): `python scripts\train_ppo_lag.py --n-envs 7` (use cores − 1)
* independent check: best, final and prior-only policies on 100 unseen episodes (seeds 30,000+) at full difficulty

Tests: `test_p6_ppo_lag.py` **9 passed** (8 fast + 1 slow); all fast tests of Phases 1–6: 66 passed.

Results vs targets (DESIGN §8 / CHECKLIST):
| Metric | Target | Validation build | Got |
|---|---|---|---|
| Toy task (PPO-Lag): success / cost per episode | ≥ 80 % / ≤ 1.5 | 100 % / 0.84 | **100 % / 0.86** (seed 0; seeds 1–2: 0.32, 0.56) |
| 200k smoke run | no errors | — | OK (best eval 70 % vs prior 45 %) |
| Full run | 3 M steps | 0.3 M (validation) | **3.0 M steps, 1,465 updates, ~1 h 45 min** (13 segments, ~500 steps/s) |
| Best eval success (selection, 20 ep, d = 1) vs prior | ≥ prior | — | 85 % / 5 % collisions vs 45 % / 15 % |
| **Independent check (100 unseen ep, d = 1)**: success | ≥ prior | 0.3 M: 55 % vs 47.5 % | **67 % (best), 68 % (final) vs 47 % (prior)** |
| Independent check: collisions / near-miss cost per episode | — | — | 6 % / 0.26 (best) vs 5 % / 1.21 (prior) |

How to read it:
* **The residual policy works:** +20 points of success (about 3 standard errors on 100 episodes), 4× fewer near-misses, and targets reached about 25 % faster (mean episode length 109 vs 149 steps). It learned what the reflex can't: getting around obstacles.
* **The best-checkpoint score (85 %) was inflated:** picking the best of 58 noisy 20-episode evals favours lucky ones. On unseen seeds it is 67 %, the same as the final model (68 %). Phase 7 evaluates on further fresh seeds.
* **Collisions did not drop (5–6 %):** a collision ends the episode after one cost step, so the per-episode near-miss limit (1.0) is rarely violated. λ stayed between 0 and 0.45 and almost never bound. Collision safety is therefore the Phase 7 shield's job, as DESIGN D15 intends; the target there is ≤ 3 %.

Decisions / deviations and why:
* **`train(curriculum=False)` is the default; the arm script passes `curriculum=True`.** With the curriculum on by default, the toy test failed (12 % success). The hazard grew late, cost rose while λ was still about 0, λ then wound up to about 7.5, and the over-weighted cost advantage destroyed the goal precision. Without the curriculum: 3/3 seeds at 100 % success, cost 0.32–0.86, λ peak 1.0–1.8 (matching the validation build's 0.84). No test was edited.
* **Learner threads:** 1 PyTorch thread during rollouts and 4 for updates. OpenMP spin-waits of a 4-thread learner stole CPU from the 4 env workers (280 → about 520 steps/s; identical learning numbers).
* **Batch size kept at 2048:** `n_steps = 2048 / n_envs` (512 with 4 envs, as CLAUDE.md §6).
* **Any episode end cuts the bootstrap** (the critic sees t/T, D14); the rollout boundary bootstraps with V(last obs).
* **Segment length 8 min, not 25:** `BASH_MAX_TIMEOUT_MS` is still unset, so commands are capped at 10 min.
* **Resume is statistically, not bitwise, identical:** the env RNG states live in the worker processes; on resume the envs are re-seeded with seed + 1000·update.
* **Option (B) re-measured prior baselines (40 episodes):** d = 0: 92.5 % success / 2.5 % collisions (was 72.5 % / 0 %); d = 1: 52.5 % / 2.5 % (was 35.0 % / 0 %).
* `models/ppo_lag.pt` (0.76 MB) and `models/ppo_lag_ckpt.pt` (2.6 MB) are committed. Smoke-run models were deleted; their logs are in `reports/ppo_lag_smoke/`.

Next: Phase 7 — Shield, Level-2 evaluation (S1–S4 × prior / prior + shield / RL / RL + shield, 100 episodes each), NumPy/ONNX export + latency, demo video. Reply "go" to start.
