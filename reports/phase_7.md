## Phase 7 — Shield, Level-2 evaluation, deployment, demo: DONE (one decision pending)

Built:
* `src/spacearm/safety/shield.py` — `SafetyShield`: 0.2 s lookahead at scales 1 / ½ / ¼ (one batched evaluation), h = min(DistanceNet clearance − 5 cm, obstacle clearance − 5 cm), escape along ∇h otherwise (via `autograd.grad`, no side effects on the DistanceNet). `shield_from_config`. The env counts interventions per episode.
* `src/spacearm/export.py` — `export_actor_numpy` / `NumpyActor` (3 matmuls, normaliser baked in), `export_actor_onnx` (opset 17, dynamic batch, legacy exporter).
* `scripts/eval_level2.py` — S1–S4 × {prior, prior + shield, RL, RL + shield} + reflex-off ablation:
  * 100 identical fixed-seed episodes per cell, 4 worker processes with explicit per-episode seeds;
  * per-scenario JSON written as it goes, plus `results.md`;
  * a `--sweep-margins` mode for the shield margin.
* `scripts/export_policy.py` — `models/actor.npz`, `models/actor.onnx`, 1-thread latency table → `reports/level2/latency.{json,md}`.
* `scripts/demo_video.py` — side-by-side prior vs RL + shield on the same S2 episode (auto-selected: the prior stalls, RL + shield gets around the obstacle) → `reports/demo.mp4`, `reports/figures/demo_strip.png`.
* `README.md` — all results tables from `reports/`, honest limits, Windows reproduction.

Commands I ran:
* `python -m pytest tests/test_p7_shield_export.py -v` — 9 fast tests passed — Windows: `pytest tests\test_p7_shield_export.py -v`
* `python -m pytest -m slow tests/test_p7_shield_export.py -v` — Level-2 acceptance (S4, 40 episodes) → **failed: 2/40 collisions**
* a replay of the 40 episodes logging the true vs estimated clearance at each collision (root cause below)
* `python scripts/eval_level2.py --sweep-margins 0.05 0.06 0.07 0.08 --seed-base 30000` — margin study on independent seeds
* `python scripts/eval_level2.py --scenarios S1 S2` and `--scenarios S3 S4` — official table (1,800 episodes, ~3 min) — Windows: `python scripts\eval_level2.py`
* `python scripts/eval_level2.py --sweep-margins 0.05 0.06 --seed-base 10000 --episodes 40` — margin check on the acceptance seeds
* `python scripts/export_policy.py` — exports + latency — Windows: `python scripts\export_policy.py`
* `python scripts/demo_video.py` — demo video — Windows: `python scripts\demo_video.py`
* `python -m pytest -m "slow or not slow" -q` — **76 passed, 1 failed** (the acceptance test above) — Windows: `pytest -m "slow or not slow"`

Tests: `test_p7_shield_export.py` 9 passed + 1 slow failed; whole suite 76 passed, 1 failed (explained in the README).

Results vs targets (DESIGN §8), 100 identical episodes per cell, seeds 10000–10099:
| Metric | Target | Validation build | Got |
|---|---|---|---|
| S4: RL + shield collisions | ≤ 3 % | 2.5 % (40 ep) | **2 %** |
| S4: RL + shield success vs prior alone | ≥ prior | 55 % vs 47.5 % | **59 % vs 45 %** |
| S4 stretch: success ≥ 70 % | stretch | — | 59 % (not met) |
| S1: RL vs prior success / near-miss cost | ≥ prior, fewer | 80 % = 80 %; 0.08 vs 0.63 | **94 % vs 88 %; RL + shield 0.00 vs 0.38** |
| Actor latency (NumPy, 1 thread) | < 1 ms | ~0.01 ms | **0.012 ms** (p99 0.029 ms); ONNX 0.008 ms |
| Slow acceptance test (first 40 of these episodes) | collisions ≤ 3 % | — | **5 % (2/40): FAIL** |

Across S1–S4:
* The RL residual adds 6–16 points of success and reaches targets ~30 % faster.
* The shield removes 20 of 21 prior collisions and 10 of 13 RL collisions, at a cost of 0–3 points of success, intervening on 7–12 % of steps.
* Reflex-off ablation: 59–60 % collisions in obstacle scenarios (validation ~50 %).
* Latency: full control step 4.5 ms p50, shield 0.18 ms (accepts) / 3.3 ms (escape), all inside the 100 ms period.

Why the acceptance test fails, and the decision needed:
* **What happened:** both collisions in the 100 S4 episodes are in the first 40, the ones the test uses. Both are grazes of the spacecraft. The shield held its own estimate at the 5 cm margin (estimated clearance 5.2–5.5 cm), but the truth was about 0 cm.
* **Why:** the estimate is the DistanceNet on *measured* angles. Seed 10015 carried up to 3.6° of encoder bias on joint 1 after a slip, which is ~6 cm at full arm extension. Seed 10025 combined net error with ~2° of bias. This is validation lesson 16 again.
* **Margin evidence:**

  | `shield.self_margin` | Acceptance seeds (40 ep): RL + shield | Independent seeds 30000+ (100 ep): RL + shield | Prior + shield (independent) |
  |---|---|---|---|
  | 5 cm (DESIGN) | 60 % / **5 %** coll. | 64 % / 0 % | 52 % / 0 % |
  | 6 cm | 60 % / **0 %** coll. | 62 % / 0 % | 50 % / 0 % |
  | 7 cm | — | 60 % / 0 % | 50 % / 0 % |
  | 8 cm | — | 55 % / 0 % | 39 % / 0 % |

* **(A) Keep 5 cm.** The S4 target is met on 100 episodes (2 %), and the failing 40-episode test is explained in the README (allowed by the checklist).
* **(B) 6 cm.** The test passes, and the margin covers the ~5.5 cm worst-case estimate error observed. It costs ~2 points of success on independent seeds. This means a DESIGN §6.4 + config change; the Level-2 table would be re-run (~3 min).
* **My recommendation is (B).** The margin choice is justified by the measured estimate error, not by the test seeds alone. But it changes a DESIGN number, so it's your call.

Decisions / deviations and why:
* **Explicit per-episode seeds** in `eval_level2.py`: the vector env's auto-reset would make episode sequences diverge between methods, so the comparison would no longer use identical episodes.
* **Seed hygiene:** model selection used seeds 20000+, the independent check and margin study 30000+, the final table 10000+ (the test's seeds).
* **Demo episode selection:** require that RL actually met the obstacle (episode ≥ 6 s). The first pick finished before the obstacle appeared and showed speed, not re-routing.
* **Not run:** the pure-RL ablation (it would need its own 3 M-step training); the validation build reported 0 % success after 130k steps.
* **Not done:** `git tag v1.0`. The work sits on a feature branch; tag `main` after the merge.

Next: decide (A)/(B) for the shield margin, merge the PR into `main`, tag v1.0. Phase 8 (dual arm) is the optional stretch.
