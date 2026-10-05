# Progress
| Phase | Status | Key results (vs DESIGN §8) | Date |
|---|---|---|---|
| 0 — Cloud environment and progress log | DONE | Py 3.11.13 venv, torch 2.14.1 (CPU use), PyBullet 3.2.7; headless render OK; `pytest --collect-only` = 7 collected / 7 errors (expected) | 2026-10-05 |
| 1 — World model (URDF, free-floating sim, exact FK) | DONE | 18/18 tests; FK err 1.2e-7 m (< 1e-5); CoM drift 0.02 mm (< 2 mm); base rotation 2.8° (> 0.5°); analytic capsules = PyBullet (5e-16 m) | 2026-10-05 |

## Decisions and deviations
* **Dotfiles were missing from the upload.** `.gitignore` and `.claude/` (the hook that CLAUDE.md §1 depends on) were not in the repo. Added both: `.claude/settings.json` + `.claude/hooks/session-start.sh` (puts `~/.venvs/spacearm/bin` on PATH, builds the venv via `cloud/setup.sh` if missing, `pip install -e .`), and a `.gitignore` for `data/ runs/ videos/` plus Python caches.
* **The environment's setup script had not run** (no venv at session start). Ran `bash cloud/setup.sh` by hand (42 s). The new hook now covers this automatically.
* **`download.pytorch.org` is blocked (HTTP 403)**, so `setup.sh` fell back to the PyPI torch build (`2.14.1+cu130`, CUDA libraries included). It runs fine on CPU but the venv is **6 GB** (of the ~30 GB disk). No action needed to continue.
* **GUI check replaced by an offscreen check:** `p.connect(p.DIRECT)` + `getCameraImage(..., renderer=ER_TINY_RENDERER)` gives a non-blank image (CLAUDE.md §4).
* Package versions are not pinned in `setup.sh`/`environment.yml`; this session got numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, gymnasium 1.4.0, onnx 1.23.1, onnxruntime 1.30.0, pytest 9.1.1.
* Phase 1: added config key `robot.arm.tcp_mass: 0.01` (value from DESIGN §4; needed so `total_mass` matches the URDF). Extra script `scripts/check_world_model.py` → `reports/phase1/world_model_check.json`.
* Phase 1: `d_body` is capped at 18 cm by the link2–pedestal pair (link2's base sphere is fixed above the pedestal). Harmless (all margins ≤ 15 cm).
* Pitfall: after dynamics the base is not at identity, so world ≠ body frame; compare body-frame FK with world obstacles only via `world_to_body` (matters in Phase 5).

## Open issues
* Environment settings (fix in the claude.ai cloud environment menu → Edit):
  * Network access: add `download.pytorch.org` to allowed domains (keep the default package-manager list) → 200 MB CPU torch instead of the 6 GB CUDA build.
  * Environment variables: `BASH_MAX_TIMEOUT_MS=1800000` and `BASH_DEFAULT_TIMEOUT_MS=300000` are **not set**, so a foreground command is capped at 10 min. That matters for the 25-min training segments in Phase 6 (else use ≤ 9-min segments or background runs).
  * Setup script: paste `cloud/setup.sh` so the venv is cached between sessions (the hook builds it anyway, at ~1 min per fresh VM).
* The Phase 0 PR is not merged into `main` yet, so Phase 1 is stacked on the same branch (`claude/friendly-newton-c17egd`).
* Uniform self-collision share is only 0.06 % (body 13.4 %; DESIGN expects ~12 % overall): few negative `d_self` labels for the DistanceNet. Check after Phase 2's boundary sampling / in Phase 3 metrics.
* Unpinned versions: if a later phase breaks on a library update, pin versions in `cloud/setup.sh` and `environment.yml` together.

## Next step
Phase 2 — Dataset: `datagen.py`, `scripts/gen_data.py` (200k + 20k samples into `data/`, git-ignored); target `tests/test_p2_datagen.py` all passed, collision share ~12 % uniform / ~20 % with boundary samples.
