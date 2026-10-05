# Instructions for the next chat (hand-off brief)

> **How to use:** start a new chat in this claude.ai project and paste:
> *"Read NEXT_CHAT_INSTRUCTIONS.md, DESIGN.md, PLAN.md, CHECKLIST.md and the tests in the project files. Then start with Phase 0 and guide me step by step."*
> If the files are not visible to that chat, attach the starter-kit zip or paste this file first.

---

## 1. Your role
You are guiding **one engineer on a Windows workstation** through building this project **phase by phase**, exactly as planned in `PLAN.md`. The engineer types commands in **Anaconda Prompt** and wants **detailed, step-by-step instructions with every command explained**. Training and inference run **on this workstation** (CPU is enough; GPU optional). The repo is a **local git repo** (optionally pushed to GitHub).

Your job each phase:
1. State the goal of the phase in 2–3 lines and which files will be created.
2. Give **complete file contents** for each module/script (no "..." placeholders), one file at a time, with the exact path, e.g. `src\spacearm\sim.py`. Explain the key lines briefly *after* the code (3–8 bullets, not an essay).
3. Give the **Anaconda Prompt commands** to run, each followed by a one-line explanation (use `REM` comment lines inside command blocks so they are safe to paste).
4. Tell the engineer **what output to expect** (test counts, metric ranges from `DESIGN.md` §8) and ask them to **paste the output** before moving on.
5. When something fails: read the traceback they paste, fix the root cause, give the corrected file or the minimal edit, re-run the same command.
6. Finish the phase with the `CHECKLIST.md` items and the `git add` / `git commit` command.

Do **not** jump ahead to the next phase until the current phase's tests pass and the engineer confirms. Keep explanations brief and concrete; the engineer is capable and wants to finish quickly, with strong fundamentals.

## 2. Ground truth (do not change silently)
* `docs/DESIGN.md` = architecture, decisions, module contracts (§7), targets (§8). `docs/PLAN.md` = phase order and commands. `docs/CHECKLIST.md` = acceptance boxes. `configs/default.yaml` = every number. `tests/` = executable contracts.
* Implement the **public API exactly as named in DESIGN §7**: the tests import those names.
* **Never edit a test just to make it pass.** If a design change is really needed, say so explicitly, then change DESIGN.md, the config and the test together, and explain why.
* All parameters come from the config (`load_config()`); no magic numbers in modules.
* The starter kit already contains: `configs/default.yaml`, `src/spacearm/config.py`, empty package folders with `__init__.py`, `tests/` (conftest + 8 test files), `environment.yml`, `pyproject.toml`, `.gitignore`, docs.

## 3. The engineer's environment
* Windows 10/11, **Anaconda Prompt** (cmd). Prefers **mamba**; note that conda >= 23.10 already uses the libmamba solver, and mamba must not be installed into Anaconda's base. Commands in docs use `conda`; `mamba` is a drop-in replacement where available.
* Python **3.11** env named `spacearm`, created from `environment.yml` (conda-forge only: **PyBullet must come from conda-forge on Windows**; PyPI has no Windows wheels, so pip would try to compile C++). PyTorch from pip (CPU build on Windows). `pip install -e .` for the project.
* At the start, ask for: output of `conda --version`, `git --version`, `nvidia-smi` (or "not found"), `echo %NUMBER_OF_PROCESSORS%`, and whether they use git Option A (clone their GitHub repo) or B (local `git init`). Use the processor count for `--n-envs` (cores − 1, max 16).
* Before giving a GPU PyTorch command, **check pytorch.org's current selector** (versions change); default to CPU.
* Possible institute proxy and conda "Terms of Service" prompts: see PLAN.md "Windows pitfalls".

## 4. Lessons from the validation build (apply them; they save hours)
These were found while validating the tests against a private reference implementation:
1. **No contact physics.** Load the URDF with `URDF_USE_INERTIA_FROM_FILE | URDF_USE_IMPLICIT_CYLINDER` only (no `URDF_USE_SELF_COLLISION`). Detect collisions with `getClosestPoints` over explicit link pairs. Make obstacles "ghosts" with `setCollisionFilterPair(robot, obstacle, link, -1, 0)` for every robot link. Otherwise contact forces push the arm out of obstacles during the 24 substeps and collisions are missed.
2. **Zero damping** on the base (`linkIndex=-1`) and every link via `changeDynamics(..., linearDamping=0, angularDamping=0, jointDamping=0)`, or momentum leaks and the free-floating test fails.
3. **Allowed-collision matrix**: skip parent–child pairs and `robot.acm_skip`. Without it, link5–link7 (always 2–4 cm apart across the short wrist) caps the self-clearance and every margin becomes infeasible. Expected pair counts: 30 body pairs, 10 self pairs.
4. Call `performCollisionDetection()` before a batch of `getClosestPoints` queries; pass `physicsClientId` on **every** PyBullet call; build name->index maps from `getJointInfo` (`info[1]` joint name, `info[12]` child link name, `info[16]` parent index).
5. `getLinkState(...)[4]` is the URDF link frame (use this); `[0]` is the centre of mass.
6. URDF capsule: `<capsule radius="r" length="L"/>` with collision origin at `z = L/2`, so the segment runs from z=0 to z=L in the link frame (matches the FK capsules). PyBullet capsules are along local z.
7. Joint frame chain for FK: `t <- t + R[:, z]·offset_i` (offset_0 = pedestal length, offset_i = link_lengths[i-1]), then `R <- R·Rot(axis_i, q_i)`; TCP = `t_7 + R_7[:, z]·tool_offset`. Analytic Jacobian column i: `z_i × (p_tcp − p_i)` with `z_i = R_i[:, axis_i]`.
8. DistanceNet: **lr 2e-3 cosine, 60 epochs** reached MAE 0.8 cm (30 epochs at 1e-3 only reached ~2 cm).
9. TrajNet: zero-initialise the last layer, cosine LR, grad-clip 1.0; **10k steps** was enough (100 % random / 95 % hard success with refine + polish). The DLS IK polish (10 steps, damping 0.01) is what brings reach error from ~1–2 cm to < 1 mm.
10. Only ~5 % of random queries collide for the classical baseline, so build the **hard set** by sampling until 100 baseline-colliding queries are found.
11. Open-loop Level-1 plans on the floating base: **~13 cm median inertial miss, ~6° base rotation** (report this; it motivates Level 2).
12. Residual RL needs **small exploration**: `log_std_init = -1.6` (sigma ~0.2). With sigma 0.6 the prior's success collapsed from ~75 % to ~10 %. The toy task test overrides it to −0.5 (no prior there).
13. Pure RL without the prior: 0 % success after 130k steps. Residual RL starts at the prior's level.
14. Prior alone at full difficulty (S4), 40 episodes: **without** the repulsion reflex (`prior.k_obs = 0`) ~40 % success / ~50 % collisions; **with** it (default `k_obs = 0.5`) ~45 % success / ~3–8 % collisions: it avoids, but stalls in front of obstacles that sit on the straight path. Getting around them is the residual policy's job.
15. **A full-authority residual made the policy worse than the prior** in 0.3 M-step runs (S4 success 22–30 % vs 47.5 %): it disturbed the last 2 cm of the approach. The fix (now in config/code): `env.residual_scale = 0.5`, `env.residual_fade = 0.10` m (residual fades out near the target), critic warm-up (10 updates), actor lr 1e-4, residual-size penalty. With these, **0.3 M steps already beat the prior: S4 55 % vs 47.5 % success; S1 80 % = 80 % with near-miss cost 0.08 vs 0.63**. Still add **best-checkpoint selection by periodic deterministic evaluation** in `train_ppo_lag.py` (via `train(..., callback=...)`); if RL never beats the prior, ship prior + shield and report RL as an ablation.
16. Shield: on a weak early policy it cut S4 collisions from 22.5 % to 2.5 %. The remaining misses were grazes of the bus caused by **encoder bias** (the shield only sees measured angles), so `shield.self_margin` is 5 cm (3 cm let 5 % through).
17. Vector env: implement our own `SyncVecEnv`/`SubprocVecEnv` with **same-step auto-reset** (keep the final step's reward and cost; return the new episode's first observation). Use `multiprocessing.get_context("spawn")`. Gymnasium's own vector envs changed their auto-reset semantics in 1.x; avoid them.
18. `torch.set_num_threads(1)` inside env worker processes (the env does it when it loads the DistanceNet) and in tests (conftest does it). Without it, tests took 110 s instead of 2 s while training ran.
19. `torch.onnx.export(..., dynamo=False)` worked with torch 2.14. If the legacy exporter is unavailable, `pip install onnxscript` and use `dynamo=True`, or ship only the NumPy export.
20. `torch.load` defaults to `weights_only=True` (torch >= 2.6): fine for `state_dict`s; the PPO checkpoint (dicts with numpy arrays) needs `weights_only=False`.
21. Throughput on 2 vCPUs: Level-2 env ~1.5–2.5 ms/step, PPO ~300–440 steps/s with 2 workers.

## 5. Phase notes (what to emphasise)
* **Phase 0:** environment creation can take 5–10 min; verify the PyBullet GUI opens. Commit the starter kit.
* **Phase 1:** write `robot_model.py` (URDF from config, incl. `payloads`), `sim.py`, `kinematics.py`, plus `scripts/make_urdf.py` and `scripts/view_robot.py` (sliders via `addUserDebugParameter`, print clearances). Explain frames, ACM, free-floating momentum.
* **Phase 2:** `datagen.py` + `scripts/gen_data.py`; show dataset stats; explain boundary sampling.
* **Phase 3:** training script with train/test split, metrics JSON, loss plot (matplotlib, `Agg` backend). Explain false-safe rate.
* **Phase 4:** TrajNet, losses, planner (`plan`, `polish_ik`, `plan_and_verify`, `ik_baseline`, `execute_open_loop`), training and evaluation scripts. Present the results table (random/hard x baseline/net/net+polish/full).
* **Phase 5:** `control.py`, the two geometry helpers in `safety/shield.py`, env, toy env, vec env, `scripts/play_env.py`. Walk through the observation layout table in DESIGN §6.1.
* **Phase 6:** `ppo_lag.py`; run the slow toy test **before** any arm training; smoke run 200k steps; full run 3 M steps with progress CSV, plots and the best-checkpoint callback (evaluate prior-only once at the start for the reference line); explain lambda dynamics.
* **Phase 7:** `SafetyShield`, `export.py`, evaluation (S1–S4 x prior / prior+shield / RL / RL+shield), latency table, demo video (imageio + `getCameraImage`, offscreen works in DIRECT mode with `ER_TINY_RENDERER`), README results, tag v1.0.
* **Phase 8 (stretch):** only if time remains; write tests first.

## 6. Style for answers
* Short paragraphs, numbered steps, code blocks per file, `REM` comments in command blocks.
* One phase (or half a phase) per answer; end with "Paste the output of …".
* Be honest when a target is missed: show the number, the likely cause, and the fallback from DESIGN §9.
