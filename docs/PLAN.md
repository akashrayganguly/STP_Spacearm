# PLAN — from empty folder to a demo in ~8 working days

**Workflow for every phase:** write the module(s) → run that phase's tests → run the phase script → tick `CHECKLIST.md` → `git commit`.
**Rules:** the config is the single source of truth · do not edit a test just to make it pass (if a design decision changes, update `DESIGN.md`, the config and the test together) · keep each commit small.

All commands run in **Anaconda Prompt** (Windows `cmd`). `REM` lines are comments; they are safe to paste.

| Phase | What | Time | Tests |
|---|---|---|---|
| 0 | Tools, git, Python environment | ½ day | collection only |
| 1 | World model: URDF, simulator, exact FK | 1 day | `test_p1_*` |
| 2 | Synthetic kinematics/collision dataset | ½ day | `test_p2_*` |
| 3 | DistanceNet (Level 1a) | ½ day | `test_p3_*` |
| 4 | TrajNet + planner + Level-1 evaluation | 1½ days | `test_p4_*` |
| 5 | Level-2 environment, prior, vector envs | 1 day | `test_p5_*` |
| 6 | PPO-Lagrangian, toy check, training | 1½ days | `test_p6_*` |
| 7 | Shield, Level-2 evaluation, export, demo | 1 day | `test_p7_*`, `-m slow` |
| 8 | *(stretch)* two codependent arms | 2 days | new tests |

---

## Phase 0 — Setup (½ day)

**0.1 Check your tools**
```bat
REM conda version (>= 23.10 means the fast libmamba solver is built in)
conda --version
REM is the mamba command available? ("not recognized" is fine)
mamba --version
REM git for version control
git --version
REM NVIDIA GPU present? ("not recognized" is fine: we train on CPU)
nvidia-smi
REM number of logical CPU cores (decides how many parallel simulators to run)
echo %NUMBER_OF_PROCESSORS%
```
* conda >= 23.10 already uses the **libmamba solver** (the engine inside mamba), so `conda` is as fast as `mamba`. If `mamba --version` works, you may type `mamba` instead of `conda` for `env create`/`install`. **Do not** `conda install mamba` into Anaconda's base environment; if you want the `mamba` command, install Miniforge separately.
* No git? Install *Git for Windows* (git-scm.com), then reopen Anaconda Prompt.
* `nvidia-smi` failing is fine: this project trains on CPU.
* Note the processor count; you will use it for `n_envs` in Phase 6.

**0.2 Project folder and git** (use a path **without spaces and outside OneDrive**)

Option A — reuse the GitHub repo you created earlier:
```bat
REM create a parent folder and enter it (/d also switches drive)
mkdir C:\projects
cd /d C:\projects
REM copy your GitHub repo into C:\projects\spacearm (a local git repo linked to GitHub)
git clone https://github.com/<your-user>/<your-repo>.git spacearm
cd spacearm
```
Option B — local only:
```bat
mkdir C:\projects\spacearm
cd /d C:\projects\spacearm
REM start an empty local repository with a branch called main
git init -b main
```
Once per machine:
```bat
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**0.3 Unpack the starter kit** (it contains a top-level `spacearm\` folder)
```bat
REM Windows 10/11 ship tar.exe, which can unpack .zip files; -C sets the target folder
tar -xf "%USERPROFILE%\Downloads\spacearm_starter.zip" -C C:\projects
dir C:\projects\spacearm
```
You should see `configs\ docs\ src\ tests\ environment.yml pyproject.toml README.md`.

**0.4 Create and activate the environment**
```bat
cd /d C:\projects\spacearm
REM build the isolated Python environment described in environment.yml (name: spacearm)
conda env create -f environment.yml
REM switch this prompt into it (the prompt prefix changes to (spacearm))
conda activate spacearm
REM install this project in editable mode so "import spacearm" finds src\spacearm
pip install -e .
```
* `env create` installs Python 3.11 + PyBullet etc. from conda-forge, then torch/gymnasium/onnx with pip (5–10 min).
* `pip install -e .` installs *this project* in editable mode: `import spacearm` works everywhere and your edits apply immediately.
* Every new Anaconda Prompt: `conda activate spacearm` and `cd /d C:\projects\spacearm`.

**0.5 Verify**
```bat
REM all key libraries import; prints the torch version and whether CUDA is usable
python -c "import pybullet, torch, gymnasium, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
REM opens an empty PyBullet window for 5 seconds (checks OpenGL)
python -c "import pybullet as p, time; p.connect(p.GUI); time.sleep(5)"
REM lists the tests without running them
pytest --collect-only -q
```
* The second line must open a PyBullet window for 5 s (OpenGL works).
* The third line lists the tests. Expect "7 tests collected, 7 errors": the errors are the test files whose modules you have not written yet.

*Optional GPU build (not needed):* copy the exact command for "Windows / Pip / your CUDA" from pytorch.org, e.g.
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`.

**0.6 First commit**
```bat
REM stage every file (data/, runs/ etc. are excluded by .gitignore) and record a snapshot
git add .
git commit -m "Phase 0: starter kit, environment, design, tests"
```
(Option A users: `git push` to back up.)

---

## Phase 1 — World model (1 day)

Write: `src/spacearm/robot_model.py`, `sim.py`, `kinematics.py` (`config.py` is provided).
Scripts: `scripts/make_urdf.py` (writes `assets/space_robot.urdf`), `scripts/view_robot.py` (GUI with one slider per joint, prints `min_distances()` live).
```bat
pytest tests\test_p1_robot_sim.py tests\test_p1_kinematics.py -v
python scripts\make_urdf.py
python scripts\view_robot.py
git add . && git commit -m "Phase 1: robot model, free-floating sim, exact FK"
```
Done when: 18 tests pass; in the GUI the arm moves, clearance goes negative when you fold it into the deck.

## Phase 2 — Dataset (½ day)

Write: `src/spacearm/datagen.py`, `scripts/gen_data.py` (200k training samples + 20k uniform test samples, prints collision share and timing).
```bat
pytest tests\test_p2_datagen.py -v
python scripts\gen_data.py
git add . && git commit -m "Phase 2: synthetic kinematics/collision dataset"
```
Outputs: `data\kinematics_dataset.npz`, `data\kinematics_test.npz` (git-ignored; regenerate any time from the seed).

## Phase 3 — DistanceNet (½ day)

Write: `src/spacearm/models/distance_net.py`, `scripts/train_distance_net.py` (saves `models\distance_net.pt`, `runs\distance_net\metrics.json`, a loss plot).
```bat
pytest tests\test_p3_distance_net.py -v
python scripts\train_distance_net.py
pytest tests\test_p3_distance_net.py -v
git add . && git commit -m "Phase 3: neural collision-distance field"
```
The second test run includes the acceptance test on the trained model (it was skipped before training).

## Phase 4 — Level-1 planner (1½ days)

Write: `src/spacearm/models/traj_net.py`, `losses.py`, `planner.py`; `scripts/train_traj_net.py`, `scripts/eval_level1.py` (random + hard sets, baseline vs network vs full planner, open-loop free-floating drift; writes `runs\level1\results.json` and a markdown table).
```bat
pytest tests\test_p4_trajectory.py -v
python scripts\train_traj_net.py
python scripts\eval_level1.py
pytest tests\test_p4_trajectory.py -v
git add . && git commit -m "Phase 4: Level-1 kinematics-aware planner + evaluation"
```

## Phase 5 — Level-2 environment (1 day)

Write: `src/spacearm/control.py` (DLS prior), `safety/shield.py` (only `sphere_capsule_distance` and `obstacle_clearance` now), `envs/space_reach_env.py`, `envs/toy_env.py`, `rl/vec_env.py`; `scripts/play_env.py` (GUI; runs prior-only episodes at a chosen difficulty and prints the info dict).
```bat
pytest tests\test_p5_env.py -v
python scripts\play_env.py --difficulty 1.0 --episodes 3
git add . && git commit -m "Phase 5: Level-2 environment with obstacles, faults, residual prior"
```

## Phase 6 — PPO-Lagrangian (1½ days)

Write: `src/spacearm/rl/ppo_lag.py`; `scripts/train_ppo_lag.py` (`--n-envs`, `--steps`, `--vec subproc|sync`; logs `runs\ppo_lag\progress.csv`; a `callback` evaluates the deterministic policy every 25 updates on 20 fixed-seed episodes at full difficulty, logs it next to the prior-only score, and saves `models\ppo_lag.pt` only when it improves).
```bat
pytest tests\test_p6_ppo_lag.py -v
pytest -m slow tests\test_p6_ppo_lag.py -v
python scripts\train_ppo_lag.py --n-envs 8 --steps 200000
python scripts\train_ppo_lag.py --n-envs 8
git add . && git commit -m "Phase 6: PPO-Lagrangian residual policy"
```
* The slow test trains on the toy task (~1 min). **Do not start the arm run until it passes.**
* The 200k-step run is a smoke test (a few minutes); then the full 3 M-step run. Use `n-envs` = processor count − 1 (max 16).
* Watch `progress.csv`: lambda should rise when cost exceeds 1.0 and fall afterwards; the periodic deterministic evaluation is the number that matters. Early policies can be *worse* than the prior (seen in validation); best-checkpoint selection protects you, and DESIGN §9 lists the knobs if it stays flat.

## Phase 7 — Shield, evaluation, deployment (1 day)

Write: `SafetyShield` + `shield_from_config` in `safety/shield.py`, `src/spacearm/export.py`; `scripts/eval_level2.py` (S1–S4 x prior / prior+shield / RL / RL+shield, 100 episodes each), `scripts/export_policy.py` (`models\actor.npz`, `models\actor.onnx`, latency table), `scripts/demo_video.py` (GUI or offscreen frames -> `videos\demo.mp4` with a surprise obstacle).
```bat
pytest tests\test_p7_shield_export.py -v
python scripts\eval_level2.py
python scripts\export_policy.py
python scripts\demo_video.py
pytest -m "slow or not slow" -v
git add . && git commit -m "Phase 7: shield, evaluation, deployment exports, demo"
git tag v1.0
```
Finish by pasting the Level-1 and Level-2 result tables into `README.md`.

## Phase 8 — *(stretch)* two codependent arms (2 days)
Follow `DESIGN.md` §10: second arm in the URDF, `env.n_arms = 2`, per-arm observations + agent one-hot, arm–arm clearance in the cost, retrain DistanceNet, train MAPPO-Lagrangian with the same `train()`. Write `tests/test_p8_dual_arm.py` first.

---

## Windows pitfalls (read once)
* **Multiprocessing:** every script that starts worker processes needs `if __name__ == "__main__":` and picklable env factories (`functools.partial(make_env, cfg)`), or it crashes/hangs.
* **CPU threads:** workers call `torch.set_num_threads(1)`; otherwise 8 workers x N threads fight over the CPU.
* **Paths:** always `pathlib.Path`; never hard-code `\` or `/`.
* **Behind an institute proxy:** set `HTTPS_PROXY`/`HTTP_PROXY` before `conda`/`pip`, and `git config --global http.proxy ...`.
* **conda asks to accept "Terms of Service" for repo.anaconda.com:** that is the Anaconda `defaults` channel; this project only needs conda-forge. Follow your organisation's policy (accept, or remove `defaults` with `conda config --remove channels defaults`).
* **PyBullet GUI over Remote Desktop** may lack OpenGL: use `DIRECT` mode and save videos offscreen instead.
