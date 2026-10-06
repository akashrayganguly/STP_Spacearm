## Phase 0 — Cloud environment and progress log: DONE

Built:
* `.claude/settings.json`, `.claude/hooks/session-start.sh` — SessionStart hook: Python 3.11 venv on PATH for every command, `pip install -e .`, builds the venv if missing (was referenced by CLAUDE.md §1 but absent from the repo).
* `.gitignore` — keeps `data/`, `runs/`, `videos/` and Python caches out of git (also absent from the repo).
* `docs/PROGRESS.md` — cross-session memory.
* `reports/` (+ `reports/figures/`) — committed results folder.

Commands I ran:
* `bash cloud/setup.sh` — builds the 3.11 venv at `~/.venvs/spacearm` with uv and installs every dependency — Windows: `conda env create -f environment.yml`
* `CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh` — runs the new hook by hand to prove it works — Windows: not needed (`conda activate spacearm` + `pip install -e .`)
* `python -c "import pybullet, torch, gymnasium, onnxruntime"` — the key libraries import — Windows: same
* `python -c "...p.connect(p.DIRECT) ... getCameraImage(renderer=p.ER_TINY_RENDERER)"` — headless replacement for the GUI test; image is non-blank — Windows: `python -c "import pybullet as p, time; p.connect(p.GUI); time.sleep(5)"`
* `python -m pytest --collect-only -q` — lists tests without running them — Windows: `pytest --collect-only -q`
* `curl https://download.pytorch.org/whl/cpu/` — checks whether the CPU-torch index is reachable (it is not: 403)

Tests: `pytest --collect-only -q` → **7 tests collected, 7 errors** (expected: the 7 errors are the test files whose modules are not written yet; the 7 collected tests are in `test_p1_kinematics.py`, which imports no project module at the top).

Environment:
| Item | Got |
|---|---|
| OS / CPU / RAM / GPU | Ubuntu 24.04.4, 4 vCPU, 15 GB, none |
| Python (venv) | 3.11.13 |
| torch | 2.14.1+cu130 (PyPI build; CUDA unused, 4 threads) |
| pybullet / gymnasium / onnxruntime | 3.2.7 / 1.4.0 / 1.30.0 |
| Offscreen render (DIRECT + TinyRenderer) | OK |
| Venv size / free disk after | 6.0 GB / 24 GB |

Results vs targets (DESIGN §8): none for Phase 0.

Decisions / deviations and why:
* The upload dropped the dotfiles (`.gitignore`, `.claude/`); I recreated them, so the setup CLAUDE.md describes now actually exists.
* The environment's setup script had not run, and `download.pytorch.org` is blocked (403), so torch came from PyPI as the CUDA build (6 GB, works on CPU).
* `BASH_MAX_TIMEOUT_MS` is not set, so commands are capped at 10 min; this needs fixing before Phase 6 (see PROGRESS.md "Open issues").
* The Windows-only checklist items (conda, OneDrive path, GUI) are replaced by their cloud equivalents (CLAUDE.md §9).

Next: Phase 1 — World model (URDF, free-floating simulator, exact FK). Reply "go" to start.
