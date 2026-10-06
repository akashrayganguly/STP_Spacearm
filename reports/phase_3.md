## Phase 3 — DistanceNet (Level 1a): DONE

Built:
* `src/spacearm/models/distance_net.py` — `DistanceNet`: [sin q, cos q] (14) → 256×3 SiLU → (d_body, d_self); `clearance` = smooth min (−τ·logsumexp(−d/τ), τ = 1 cm, never above the smaller head); `from_config`; float64 inputs are cast to the weights' dtype and back. `distance_loss`: Huber on clipped targets, ×10 inside ±5 cm.
* `scripts/train_distance_net.py` — AdamW 2e-3 cosine → 0, batch 1024, 60 epochs, 4 threads. 5 % of the training file is held out for monitoring; final metrics on the 20k uniform test file. Writes `models/distance_net.pt`, `reports/distance_net/{metrics.json, progress.csv, training.png}`.

Commands I ran:
* `python -m pytest tests/test_p3_distance_net.py -v` (before training) — 7 passed, the acceptance test skipped (no model yet) — Windows: `pytest tests\test_p3_distance_net.py -v`
* `python scripts/train_distance_net.py --epochs 1` — 1 s smoke run of the whole pipeline
* `python scripts/train_distance_net.py` — full training, 53 s — Windows: `python scripts\train_distance_net.py`
* `python -m pytest tests/test_p3_distance_net.py -v` (after training) — 8 passed, acceptance included
* `python -m pytest tests/test_p1_*.py tests/test_p2_datagen.py tests/test_p3_distance_net.py -q` — 32 passed

Tests: `test_p3_distance_net.py` **8 passed, 0 skipped** (after training); Phases 1–3 together 32 passed.

Results vs targets (DESIGN §8), uniform test set, 20k samples:
| Metric | Target | Validation build | Got |
|---|---|---|---|
| MAE, all (both heads) | ≤ 1.0 cm | 0.8 cm | **0.68 cm** |
| MAE, \|d\| < 10 cm | ≤ 2.0 cm | 1.4 cm | **1.22 cm** |
| Sign accuracy (clearance vs true min) | ≥ 97 % | 97.8 % | **98.3 %** |
| False-safe @ 3 cm | ≤ 0.5 % | 0.17 % | **0.12 %** |
| Training time (4 vCPU) | ~1–2 min | — | 49 s (60 epochs) |
| Parameters | < 300k | 136k | 135,938 |

Per head: body MAE 1.21 cm, self MAE 0.15 cm. On the 5 % validation split (training distribution, 31 % near-surface, harder): MAE 0.66 cm, sign accuracy 95.3 %, false-safe 0.14 %.

What Phase 7's shield needs to know (clearance over-estimate = predicted − true, test set):
| True distance band | p99 over-estimate | Share over-estimated by > 5 cm |
|---|---|---|
| colliding (−5…0 cm, clipped) | 5.9 cm | 2.2 % |
| 0–5 cm | 4.5 cm | 0.8 % |
| 5–10 cm | 2.5 cm | 0.2 % |
| ≥ 10 cm | ≤ 1.9 cm | ≤ 0.02 % |

Of all configurations the net rates at ≥ 5 cm clearance, **0.05 % actually collide** (8 of 16k). Worst case: true −4.4 cm, predicted +9.0 cm (a body collision). So a 5 cm shield margin is not a guarantee. This matches DESIGN D15/§6.4 ("only as good as its models") and the validation build's 2.5 % S4 grazes. Phase 7 should report it, not hide it.

Decisions / deviations and why:
* **Two new config keys:** `distance_net.huber_beta: 0.02` (the value is in DESIGN §5.2 but was not in the config) and `distance_net.weight_decay: 0.01` (AdamW, pinned to PyTorch's current default so a library update can't change it silently). Additions only; no test changed.
* **The loss is a plain weighted mean, not normalised by the weights.** The Phase 3 test compares single-element batches; with normalisation the ×10 weight would cancel.
* **The arm–arm weak spot (Phase 2) shows up:** self-head MAE near its own surface is 4.1 cm on the test set (only 12 samples) and 1.5 cm on validation (16 samples). It doesn't affect acceptance, because self-collisions are 0.03 % of uniform configurations. If it matters later, the fix is boundary sampling on `|d_self|` (DESIGN §5.1 change).
* **Spread at the 18 cm `d_body` cap:** predictions there range from ~11 to 22 cm (visible as a vertical band in `training.png`). It's harmless, because the planner, prior and shield only act below 15 cm.
* Results go to `reports/distance_net/` instead of `runs/distance_net/` (CLAUDE.md §5). `models/distance_net.pt` (0.5 MB) is committed.

Next: Phase 4 — TrajNet + Level-1 planner + evaluation (`models/traj_net.py`, `losses.py`, `planner.py`, `train_traj_net.py`, `eval_level1.py`); targets random-set success ≥ 95 %, hard-set ≥ 90 %, median plan time < 0.5 s. Reply "go" to start.
