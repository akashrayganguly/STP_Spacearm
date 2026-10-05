# Progress
| Phase | Status | Key results (vs DESIGN §8) | Date |
|---|---|---|---|
| 0 — Cloud environment and progress log | DONE | Py 3.11.13 venv, torch 2.14.1 (CPU use), PyBullet 3.2.7; headless render OK; `pytest --collect-only` = 7 collected / 7 errors (expected) | 2026-10-05 |
| 1 — World model (URDF, free-floating sim, exact FK) | DONE | 18/18 tests; FK err 1.2e-7 m (< 1e-5); CoM drift 0.02 mm (< 2 mm); base rotation 2.8° (> 0.5°); analytic capsules = PyBullet (5e-16 m) | 2026-10-05 |
| 2 — Synthetic kinematics/collision dataset | DONE | 6/6 tests; 200k train + 20k test in 42 s; collision share 13.0 % uniform (~12 %) / 21.5 % with boundary samples (~21 %); reproducible byte-for-byte | 2026-10-05 |
| 3 — DistanceNet | DONE | 8/8 tests (acceptance incl.); test MAE 0.68 cm (≤ 1.0), \|d\|<10 cm 1.22 cm (≤ 2.0), sign acc 98.3 % (≥ 97), false-safe 0.12 % (≤ 0.5); 49 s training | 2026-10-05 |
| 4 — Level-1 planner | DONE | 10/10 tests (acceptance incl.); full planner 100 % random (≥ 95) / 100 % hard (≥ 90, baseline 0 %); median plan 99 ms (< 0.5 s); open-loop floating: miss 17.2 cm, base 6.4°; TrajNet 10k steps in 3 min | 2026-10-05 |

## Decisions and deviations
* **Dotfiles were missing from the upload.** `.gitignore` and `.claude/` (the hook that CLAUDE.md §1 depends on) were not in the repo. Added both: `.claude/settings.json` + `.claude/hooks/session-start.sh` (puts `~/.venvs/spacearm/bin` on PATH, builds the venv via `cloud/setup.sh` if missing, `pip install -e .`), and a `.gitignore` for `data/ runs/ videos/` plus Python caches.
* **The environment's setup script had not run** (no venv at session start). Ran `bash cloud/setup.sh` by hand (42 s). The new hook now covers this automatically.
* **`download.pytorch.org` is blocked (HTTP 403)**, so `setup.sh` fell back to the PyPI torch build (`2.14.1+cu130`, CUDA libraries included). It runs fine on CPU but the venv is **6 GB** (of the ~30 GB disk). No action needed to continue.
* **GUI check replaced by an offscreen check:** `p.connect(p.DIRECT)` + `getCameraImage(..., renderer=ER_TINY_RENDERER)` gives a non-blank image (CLAUDE.md §4).
* Package versions are not pinned in `setup.sh`/`environment.yml`; this session got numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, gymnasium 1.4.0, onnx 1.23.1, onnxruntime 1.30.0, pytest 9.1.1.
* Phase 1: added config key `robot.arm.tcp_mass: 0.01` (value from DESIGN §4; needed so `total_mass` matches the URDF). Extra script `scripts/check_world_model.py` → `reports/phase1/world_model_check.json`.
* Phase 1: `d_body` is capped at 18 cm by the link2–pedestal pair (link2's base sphere is fixed above the pedestal). Harmless (all margins ≤ 15 cm).
* Pitfall: after dynamics the base is not at identity, so world ≠ body frame; compare body-frame FK with world obstacles only via `world_to_body` (matters in Phase 5).
* Phase 2: boundary samples = perturbed near-surface seeds kept only if still |d_min| < 5 cm (rejection). Test-set seed = seed + 1. `sample_configs` defaults to the full range; the dataset passes `data.limit_frac`.
* Phase 3: config keys `distance_net.huber_beta: 0.02` (DESIGN §5.2 value) and `distance_net.weight_decay: 0.01` (pinned AdamW default) added. Loss = plain weighted mean (the test needs it). Results in `reports/distance_net/`.
* Phase 4: config keys `traj.polish_damping: 0.01`, `traj.grad_clip: 1.0`, `traj.weight_decay: 0.01` added (DESIGN/lesson values). Results in `reports/level1/` and `reports/traj_net/`. Planner timings measured on 1 thread; cloud-VM timings can jump transiently (one 28 ms median re-measured at 8 ms).

## Open issues
* Environment settings (fix in the claude.ai cloud environment menu → Edit):
  * Network access: add `download.pytorch.org` to allowed domains (keep the default package-manager list) → 200 MB CPU torch instead of the 6 GB CUDA build.
  * Environment variables: `BASH_MAX_TIMEOUT_MS=1800000` and `BASH_DEFAULT_TIMEOUT_MS=300000` are **not set**, so a foreground command is capped at 10 min. That matters for the 25-min training segments in Phase 6 (else use ≤ 9-min segments or background runs).
  * Setup script: paste `cloud/setup.sh` so the venv is cached between sessions (the hook builds it anyway, at ~1 min per fresh VM).
* The Phase 0 PR is not merged into `main` yet, so Phases 1–4 are stacked on the same branch (`claude/friendly-newton-c17egd`).
* Arm–arm surface is thin in the data: 49 self-colliding and 265 near-surface (|d_self| < 5 cm) samples of 200k (boundary sampling follows d_min, i.e. the body). Phase 3 acceptance unaffected; if the `d_self` head is poor near 0, seed boundary samples on |d_self| < band too (DESIGN §5.1 change, needs agreement).
* For Phase 7: the DistanceNet over-estimates clearance by > 5 cm on 2.2 % of colliding test configs; 0.05 % of configs it rates ≥ 5 cm are actually colliding (worst: true −4.4 cm, predicted +9.0 cm). The 5 cm shield margin is not a guarantee; report it.
* Unpinned versions: if a later phase breaks on a library update, pin versions in `cloud/setup.sh` and `environment.yml` together.

## Next step
Phase 5 — Level-2 environment: `control.py` (DLS reflex prior), `safety/shield.py` (`sphere_capsule_distance`, `obstacle_clearance` only), `envs/space_reach_env.py`, `envs/toy_env.py`, `rl/vec_env.py`, `scripts/play_env.py` (offscreen GIF/MP4); target `tests/test_p5_env.py` all passed; record prior-only baseline at d = 0 and d = 1. Mind world vs body frame (Phase 1 pitfall).
