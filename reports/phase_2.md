## Phase 2 — Synthetic kinematics/collision dataset: DONE

Built:
* `src/spacearm/datagen.py` — `sample_configs` (uniform inside a fraction of each joint range), `label_configs` (PyBullet signed `d_body`/`d_self` + body-frame TCP, float32), `generate_distance_dataset` (uniform + boundary samples, shuffled, deterministic per seed), `save_dataset`/`load_dataset` (.npz).
* `scripts/gen_data.py` — 200k training + 20k uniform test samples into `data/` (git-ignored), stats JSON and a histogram in `reports/`.

How boundary sampling works: near-surface seeds (|min(d_body, d_self)| < 5 cm) are taken from the uniform part, perturbed with N(0, 0.05 rad) and clipped to 98 % of the range. Only perturbations that are still near the surface are kept (rejection, at most 50 rounds). Boundary share: 25 %.

Commands I ran:
* `python -m pytest tests/test_p2_datagen.py -v` — the 6 Phase 2 contracts — Windows: `pytest tests\test_p2_datagen.py -v`
* `python scripts/gen_data.py` — writes `data/kinematics_dataset.npz` (200k, 9.6 MB) and `data/kinematics_test.npz` (20k), `reports/phase2/dataset_stats.json`, `reports/figures/phase2_distance_hist.png` — Windows: `python scripts\gen_data.py`
* A reproducibility check — two fresh simulators, same seed → byte-identical 5k-sample datasets (same SHA-1)
* `python -m pytest tests/test_p1_*.py tests/test_p2_datagen.py -q` — all finished phases still green — Windows: `pytest tests\test_p1_robot_sim.py tests\test_p1_kinematics.py tests\test_p2_datagen.py -q`

Tests: `test_p2_datagen.py` **6 passed, 0 skipped** (0.7 s); Phases 1+2 together 24 passed.

Results vs targets (DESIGN §5.1 / CHECKLIST):
| Metric | Target | Got |
|---|---|---|
| Training set size / test set size | 200k / 20k | 200k / 20k |
| Collision share, uniform (test set) | ~12 % | **13.0 %** |
| Collision share, with boundary samples (train) | ~20–21 % | **21.5 %** |
| Generation time | ~1 min (~0.15 ms/sample) | **38 s train + 3.5 s test** (0.19 ms/sample incl. rejected boundary candidates) |
| Near-surface share (\|d\| < 5 cm) | — | 30.9 % train vs 7.7 % uniform (4x enrichment) |

Label ranges (train): `d_body` −0.48 … +0.18 m (median 0.16), `d_self` −0.10 … +0.24 m (1st percentile 8.5 cm); TCP x −0.79…1.39, y ±1.09, z −0.02…1.93 m.

Decisions / deviations and why:
* **Arm–arm collisions are almost absent: 0.02 % of training samples, 49 of 200k.** Only 265 samples lie within ±5 cm of the arm–arm surface, because the ACM removes the only close pairs and the remaining pairs need extreme folding. Consequences:
  * Phase 3 acceptance is unaffected (self-collisions are 0.03 % of the uniform test set).
  * The DistanceNet's `d_self` head will be uncertain inside ±5 cm of its own surface. The 10–24 cm range that the Level-2 null-space push uses (activation at 10 cm) is well covered: 20.7k samples below 10 cm.
  * Optional fix (not applied; it changes DESIGN §5.1): also seed boundary samples on `|d_self| < band`. Worth doing only if Phase 3 shows `d_self` errors near zero.
* **`d_body` piles up at its 18 cm cap** (link2–pedestal, see Phase 1): 50 % of uniform samples have exactly 0.18 m. That is easy to learn, and the label clip of 0.30 m never binds for `d_body`.
* `sample_configs` defaults to the full range (`frac=1.0`); the dataset passes `data.limit_frac` (0.98) explicitly, so no number is hard-coded in the module.
* Test set seed = training seed + 1 (constant `TEST_SEED_OFFSET` in the script).
* Data is not committed (CLAUDE.md §5); regenerate any time with `python scripts/gen_data.py` (~45 s, same bytes).

Next: Phase 3 — DistanceNet (Level 1a): `models/distance_net.py`, `scripts/train_distance_net.py`; targets MAE ≤ 1 cm, sign accuracy ≥ 97 %, false-safe ≤ 0.5 %. Reply "go" to start.
