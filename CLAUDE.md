# CLAUDE.md — spacearm in a Claude Code cloud session (read first, every session)

This repo builds the project defined in `docs/DESIGN.md` (architecture, decisions, **module contracts §7**, **targets §8**), in the phase order of `docs/PLAN.md`, with acceptance boxes in `docs/CHECKLIST.md`, all numbers in `configs/default.yaml`, and executable contracts in `tests/`.
`docs/NEXT_CHAT_INSTRUCTIONS.md` still applies, **especially its §4 "Lessons from the validation build"**. This file adds the cloud-session rules on top of it. **Where the two conflict, this file wins.**

---

## 1. Start of every session (do this before anything else)
The environment's setup script (`cloud/setup.sh`) builds a Python 3.11 venv at `~/.venvs/spacearm`, and the SessionStart hook (`.claude/settings.json` -> `.claude/hooks/session-start.sh`) puts it on `PATH` for every command and re-installs this repo in editable mode. Check that it worked:
```bash
python --version                                                # must be 3.11.x (else: bash cloud/setup.sh, then
                                                                #  prefix commands with: source ~/.venvs/spacearm/bin/activate &&)
python -c "import spacearm, pybullet, torch; print('ok')"       # else: pip install -e .
cat docs/PROGRESS.md 2>/dev/null || echo "no progress log yet"  # cross-session memory: where are we?
git status && git log --oneline -5
```
Then run the tests of the phases already finished (e.g. `python -m pytest tests/test_p1_robot_sim.py tests/test_p1_kinematics.py -q`) to confirm the starting point is green. Running the whole suite before Phase 7 shows collection errors for unwritten modules; that is expected.

## 2. Who does what (changes to NEXT_CHAT_INSTRUCTIONS.md)
* **You write the files, run the commands, fix failures and commit.** The engineer reviews and steers. Do not hand Anaconda Prompt commands to the engineer; run the Linux equivalents yourself.
* **One phase per go-ahead.** Finish the phase (code → its tests → its script → checklist), write the phase report (§8), update `docs/PROGRESS.md` and tick `docs/CHECKLIST.md`, commit, push, then **stop and wait for "go"**. Never start the next phase unasked.
* Teach while doing: in the phase report, list the key commands you ran with a one-line explanation each, plus the Windows (Anaconda Prompt) equivalent when it differs. Keep chat messages short; details go in `reports/`.
* Implement the public API exactly as named in DESIGN §7. **Never edit a test just to make it pass.** If a design change is unavoidable, stop, explain it, and change DESIGN, config and test together only after the engineer agrees.

## 3. The cloud machine (Anthropic-hosted VM)
* Ubuntu 24.04 x86_64, about **4 vCPUs, 16 GB RAM, 30 GB disk**, **no GPU, no display**.
* Python **3.11** venv at `~/.venvs/spacearm` (PyBullet ships Linux wheels only up to Python 3.11; the system Python 3.12 would compile it from source). Do not install conda and do not change the Python version. `environment.yml` is for the engineer's Windows workstation: keep it, and keep it in sync if you add a dependency (also add the package to `cloud/setup.sh`).
* Network: package registries only (+ GitHub via proxy). If an install hits a 403/connect error, report the host; do not try to route around it.
* Command time: a foreground command can wait up to `BASH_MAX_TIMEOUT_MS` (set to 30 min in the environment; default 10 min). A command that times out moves to the background for up to 30 more minutes, then it is killed.
* **When the session goes idle for a few minutes the VM pauses, and it can later be reclaimed: running processes die and anything not pushed to git is lost.**

## 4. Headless rules
* Always `p.connect(p.DIRECT)` in code that runs here. A `--gui` flag is allowed only as an opt-in for the engineer's workstation.
* Phase 1: instead of (or in addition to) the GUI `view_robot.py`, write `scripts/render_robot.py`: offscreen snapshots (`getCameraImage(..., renderer=p.ER_TINY_RENDERER)`) of named configurations (zero, folded-into-deck, a few random ones) with the clearance printed in the title, saved to `reports/figures/`. **Open the PNGs with your Read tool** and sanity-check the geometry before continuing.
* `play_env.py` and `demo_video.py` render offscreen to GIF/MP4 with imageio (`imageio-ffmpeg` is installed). Matplotlib uses the Agg backend.
* Keep all code OS-independent (pathlib, `multiprocessing` "spawn", `if __name__ == "__main__":`, no bash-only logic in `src/` or `scripts/`), so the engineer can run everything later on Windows with `python scripts\<name>.py`.

## 5. Persistence: only git survives
* Commit and push at the end of every phase **and after every long-run segment**.
* **Commit:** code, tests, docs, `models/*.pt|*.npz|*.onnx` (all small), and `reports/` (metrics JSON, progress CSV, PNG plots, markdown tables, phase reports, demo MP4 < 20 MB).
* **Never commit:** `data/` (regenerate any time with `python scripts/gen_data.py`; ~1–2 min, same seed = same data) and `runs/` (scratch). Keep every file < 50 MB.
* Wherever PLAN.md writes results under `runs/...` (metrics JSON, progress CSV, plots), write them to `reports/...` instead (same sub-folder names), because `runs/` is git-ignored and would be lost.
* `docs/PROGRESS.md` is the memory between sessions. Create it in Phase 0 with this shape and update it at every phase end:
  ```
  # Progress
  | Phase | Status | Key results (vs DESIGN §8) | Date |
  ## Decisions and deviations
  ## Open issues
  ## Next step
  ```

## 6. Long jobs (Phase 6 training; anything else > 10 min)
* `scripts/train_ppo_lag.py` must be **resumable and time-boxed**: flags `--resume` and `--max-minutes N`; save `models/ppo_lag_ckpt.pt` (actor, both critics, optimiser, observation-normaliser stats, lambda, update index, RNG states, best eval score) every 10 updates and on exit. The best checkpoint by periodic deterministic evaluation stays in `models/ppo_lag.pt` (DESIGN §6.3). Extend `train()` only with **optional** arguments (e.g. a resume state and a time budget) so the existing tests keep passing.
* Run training in segments of **≤ 25 minutes**: `python scripts/train_ppo_lag.py --n-envs 4 --resume --max-minutes 25` (foreground with a 30-min timeout). After each segment: show the last lines of `reports/ppo_lag/progress.csv`, commit + push the checkpoint and CSV (`wip: training segment k`), then start the next segment right away. Staying busy keeps the VM from pausing.
* Budget on 4 vCPUs: `--n-envs 4` with `n_steps 512` (batch 2048) gives roughly 600–900 steps/s, so 3 M steps take about 1–1.5 h (3–4 segments). If time is short, stop at 2 M steps and say so in the report.
* The final Level-2 evaluation (S1–S4, 100 episodes each) can take 10–20 min: run it per scenario, saving results as you go.

## 7. Git flow
* Each cloud session works on its own branch. At the end of a phase, push and ask the engineer to open a pull request titled `Phase k: <title>` and merge it into `main` (or create it with `gh pr create` if available). The next session must start from the updated `main`, so the merged work is in the clone.
* Commit messages: `Phase k: <what>`, small and frequent. Never force-push and never rewrite `main`.

## 8. Phase report (post in chat at phase end; also save as `reports/phase_<k>.md`)
```
## Phase k — <title>: DONE | BLOCKED
Built: <file> — <one line>, ...
Commands I ran: <command> — <what it does> — Windows: <equivalent if different>
Tests: <file>: N passed, M skipped
Results vs targets (DESIGN §8): <metric> | <target> | <got>
Decisions / deviations and why: ...
Next: Phase k+1 — <title>. Reply "go" to start.
```

## 9. Phase-specific adjustments for the cloud
* **Phase 0:** skip the Windows/Anaconda steps. Verify the venv (`python -c "import pybullet, torch, gymnasium, onnxruntime"`), run `pytest --collect-only -q` (expect "7 tests collected, 7 errors"), create `docs/PROGRESS.md` and `reports/`, commit `Phase 0: cloud environment and progress log`.
* **Phase 1:** `render_robot.py` (offscreen) as above; look at the images.
* **Phases 2–4:** use all 4 cores for PyTorch (`torch.set_num_threads(4)`) in training scripts; env worker processes keep 1 thread. TrajNet training (10k steps) may exceed 10 min: run it time-boxed or in the background and poll.
* **Phase 6:** §6 protocol. Run the slow toy test (`python -m pytest -m slow tests/test_p6_ppo_lag.py`) before any arm training.
* **Phase 7:** demo video offscreen into `reports/demo.mp4`; the final README tables are filled from `reports/`.
* At the very end, add to the README how to reproduce on the Windows workstation (PLAN.md Phase 0 + `python scripts\...`), since the trained models are in the repo.
