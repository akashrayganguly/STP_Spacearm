# spacearm — collision-free path planning for a 7-DOF arm on a free-floating spacecraft

**Level 1:** a kinematics-aware neural planner that turns *(current joint angles, target point)* into a smooth, collision-free joint trajectory and goal joint angles that never hit the spacecraft body.
**Level 2:** a lightweight constrained-RL policy (8k parameters, ~0.01 ms) that re-routes the arm in real time around surprise obstacles and through hardware faults, while the spacecraft floats and reacts to every arm motion.

Everything is simulated in PyBullet and runs on a CPU. The trained models are in the repo (`models/`), so every evaluation below can be re-run directly.

![demo: prior only (left) vs RL residual + shield (right)](reports/figures/demo_strip.png)
Demo video: [`reports/demo.mp4`](reports/demo.mp4). The same episode is shown twice. A surprise obstacle appears at about 4 s on the way to the target. The classical reflex (left) hovers safely beside it and times out 3.3 cm short. The RL residual with the safety shield (right) re-routes around it and arrives at 11.1 s.

**New in v1.1**
* 📖 **[Detailed writeup](docs/WRITEUP.md):** an illustrated, story-style tour of every model (world, dataset, DistanceNet, TrajNet, planner stages, Level-2 environment, RL policy, shield) with plots of every part and tables for every variant.
* 🖥️ **Interactive explorer:** `pip install "gradio>=6" plotly`, then `python scripts/gui.py`. Three tabs: a pose and clearance viewer (true `d_body`/`d_self` vs DistanceNet), a stage-by-stage planner, and a Level-2 mission simulator (choose the controller, start, target, obstacle, faults and sensor noise; compare all four controllers; replay any report episode).
* 🔬 **New analyses:** the planner stage by stage (refine alone *lowers* success; refine + polish is what reaches 100 %), and a 17-variant Level-2 study (obstacles drive the failures; faults and noise barely matter; the shield does not yet model moving obstacles).

## Results

### Level 1 — neural collision-distance field and planner

| DistanceNet (20k uniform test configurations) | Target | Got |
|---|---|---|
| MAE, all / \|d\| < 10 cm | ≤ 1.0 / 2.0 cm | **0.68 / 1.22 cm** |
| Sign accuracy / false-safe @ 3 cm | ≥ 97 % / ≤ 0.5 % | **98.3 % / 0.12 %** |

Planner, on queries from unseen test configurations. Every success is verified in PyBullet: clearance > 0 at 100 points, reach < 2 cm, within joint limits.

| Set | Method | Success | Collisions | Median reach error | Median plan time |
|---|---|---|---|---|---|
| random (200) | Baseline: PyBullet IK + straight joint line | 92 % | 3.5 % | 0.14 cm | 1 ms |
| random (200) | Network only | 85 % | 1.5 % | 1.02 cm | 5 ms |
| random (200) | Network + IK polish | 98 % | 1.5 % | < 0.01 cm | 8 ms |
| random (200) | **Full planner** (refine + polish + verify) | **100 %** | 0 % | < 0.01 cm | **99 ms** |
| hard (100) | Baseline | 0 % (by construction) | 100 % | 0.21 cm | 1 ms |
| hard (100) | Network only | 66 % | 6 % | 1.32 cm | 2 ms |
| hard (100) | Network + IK polish | 93 % | 7 % | < 0.01 cm | 28 ms¹ |
| hard (100) | **Full planner** | **100 %** | 0 % | < 0.01 cm | **96 ms** |

¹ A transient on the shared cloud VM; re-timed it is a steady 8 ms (the IK polish is a fixed 10 iterations).

Targets: random ≥ 95 %, hard ≥ 90 %, median plan time < 0.5 s; all met. Played open-loop on the free-floating spacecraft, these perfect body-frame plans miss an inertially fixed target by **17 cm** (median; max 33 cm), because the spacecraft rotates **6.4°** in reaction. That is why Level 2 closes the loop.

**Stage by stage** (same queries; [`reports/level1/stages.md`](reports/level1/stages.md)). Success on random / hard queries: TrajNet alone 85 / 66 %; + refine 56 / 54 % (it clears collisions but pulls the tip 1–2 cm off the target); + IK polish 98 / 93 %; refine + polish 99.5 / 100 %; full planner 100 / 100 %. Refine buys clearance, polish buys precision.

### Level 2 — residual PPO-Lagrangian policy + safety shield

100 identical fixed-seed episodes per cell (seeds 10000–10099, never used for training or model selection):

| Scenario | Method | Success | Collisions | Near-miss cost / ep. | Time to reach |
|---|---|---|---|---|---|
| S1 nominal | prior (reflex) | 88 % | 5 % | 0.38 | 9.7 s |
| S1 nominal | prior + shield | 83 % | 0 % | 0.00 | 9.9 s |
| S1 nominal | RL | 94 % | 2 % | 0.40 | 6.6 s |
| S1 nominal | **RL + shield** | **89 %** | **0 %** | 0.00 | 6.9 s |
| S2 obstacles only | prior | 58 % | 6 % | 0.66 | 10.4 s |
| S2 obstacles only | prior + shield | 53 % | 1 % | 0.30 | 10.2 s |
| S2 obstacles only | RL | 71 % | 4 % | 0.53 | 7.5 s |
| S2 obstacles only | **RL + shield** | **68 %** | **0 %** | 0.00 | 7.9 s |
| S2 obstacles only | ablation: prior, reflex off | 39 % | 59 % | 2.20 | 8.3 s |
| S3 faults + noise only | prior | 85 % | 3 % | 0.56 | 11.2 s |
| S3 faults + noise only | prior + shield | 77 % | 0 % | 0.00 | 11.0 s |
| S3 faults + noise only | RL | 92 % | 3 % | 0.38 | 7.8 s |
| S3 faults + noise only | **RL + shield** | **87 %** | **0 %** | 0.12 | 8.2 s |
| S4 everything | prior | 45 % | 7 % | 0.71 | 11.2 s |
| S4 everything | prior + shield | 41 % | 1 % | 0.05 | 10.9 s |
| S4 everything | RL | 61 % | 4 % | 0.38 | 8.0 s |
| S4 everything | **RL + shield** | **58 %** | **0 %** | 0.09 | 8.2 s |
| S4 everything | ablation: prior, reflex off | 34 % | 60 % | 2.67 | 9.5 s |

Full table with base rotation and shield intervention rates: [`reports/level2/results.md`](reports/level2/results.md).

* **The S4 target is met:** RL + shield has 0 % collisions (target ≤ 3 %) and 58 % success, against the prior's 45 % / 7 %. The stretch goal (≥ 70 % success) is not met.
* **S1:** success is not worse than the prior's (89 % vs 88 %) and there are no near-misses (0.00 vs 0.38 per episode).
* **The RL residual** adds 6–16 points of success in every scenario and reaches targets about 30 % faster.
* **The shield**, with a 6 cm self-margin, removes every RL collision in S1–S4 (13 → 0) and almost all prior collisions (21 → 2). The cost is 3–5 points of success for RL.
* **The reflex matters:** without it the prior collides in about 60 % of obstacle episodes.

Latency on one CPU thread (control period 100 ms):

| Component | p50 | p99 |
|---|---|---|
| Actor, NumPy (`models/actor.npz`) | 0.012 ms | 0.029 ms |
| Actor, ONNX Runtime (`models/actor.onnx`) | 0.008 ms | 0.024 ms |
| Reflex prior | 0.30 ms | 0.59 ms |
| Safety shield, command accepted / escape | 0.18 / 3.3 ms | 0.38 / 4.5 ms |
| Full control step incl. physics, features, policy, prior, shield | 4.5 ms | 7.3 ms |

**Variant study (v1.1)**: 17 variants × 4 controllers × 100 fresh episodes (seeds 40000+); [`reports/level2/variants.md`](reports/level2/variants.md) and [writeup §10](docs/WRITEUP.md#10-results). Success / collisions:

| Variant | Reflex | RL + shield |
|---|---|---|
| difficulty d = 0 → 1 | 89 % → 49 % | 89 % → 61 % (0 % collisions throughout) |
| static ball, 5–12 cm | 37 % / 3 % | 53 % / 0 % |
| drifting ball, ≤ 3 cm/s | 63 % / 3 % | 68 % / 1 % |
| faults only (bias, weak motors or slip) | 83–89 % / 1–2 % | 86–89 % / 0 % |
| sensor noise only (× 1 and × 3) | 89 % / 1 % | 89–90 % / 0 % |
| stress: ball drifting ≤ 6 cm/s | 68 % / 5 % | 76 % / 2 % |

### Honest limits
* **Shield margin raised from 5 to 6 cm (Phase 7, agreed).** At 5 cm, 2 of the 100 S4 episodes, both inside the 40 used by the acceptance test, grazed the spacecraft. The robot's own clearance estimate (DistanceNet on encoder angles carrying up to 3.6° of bias after a slip) was about 5.5 cm too optimistic. At 6 cm: 0 collisions in all four scenarios, for about 2 points of success (`reports/level2/margin_sweep_*.json`).
* **The shield is a model-based filter, not a proof:** it only knows the measured joint angles, the DistanceNet and the obstacle as seen by vision.
* **Targets are sampled within 85 % of the arm's reach.** With unrestricted targets, about 20 % of nominal episodes were unwinnable: the base recoil carried near-full-stretch targets out of reach.
* **The PPO-Lagrangian constraint (≤ 1 near-miss step per episode) rarely bound** (λ ≤ 0.45). Collision safety comes mostly from the reflex and the shield, not from the learned constraint.
* **The shield does not predict obstacle motion** (it looks 0.2 s ahead for the arm only), so drifting balls still cause 1–2 % collisions in the variant study, and 2–3 % at twice the trained speed.
* **A static ball on the straight path is the hardest case** (37 % success for the reflex, 53–54 % with RL).

## Reproduce on a Windows workstation (Anaconda Prompt)
```bat
cd /d C:\projects
git clone https://github.com/akashrayganguly/STP_Spacearm.git spacearm
cd spacearm
REM Python 3.11 + PyBullet from conda-forge (PLAN.md Phase 0), then this package in editable mode
conda env create -f environment.yml
conda activate spacearm
pip install -e .
REM fast tests (about 20 s); add -m "slow or not slow" for everything (about 3 min, 86 tests)
pytest
REM evaluations with the committed models
python scripts\gen_data.py
REM   (data\ is not in git: about 1 min, needed by eval_level1)
python scripts\eval_level1.py
python scripts\eval_level2.py
python scripts\export_policy.py
python scripts\demo_video.py
REM interactive explorer in the browser (pip install gradio plotly)
python scripts\gui.py
REM new v1.1 analyses (data\ needed for the stage study and the figures)
python scripts\eval_level1_stages.py
python scripts\eval_level2_variants.py
python scripts\make_writeup_figures.py
REM PyBullet GUI viewers (workstation only)
python scripts\view_robot.py
python scripts\play_env.py --gui --difficulty 1.0 --episodes 3
REM watch the trained policy behind the shield
python scripts\play_env.py --gui --policy models\ppo_lag.pt --shield --difficulty 1.0 --episodes 5
```
To retrain from scratch, run each phase's script in order (each one takes `--config` and `--seed`):
```bat
python scripts\gen_data.py
python scripts\train_distance_net.py
python scripts\train_traj_net.py
python scripts\train_ppo_lag.py --n-envs 7 --fresh
```
* `gen_data.py` regenerates `data\` (git-ignored) in about 1 min; the same seed gives the same data.
* `train_distance_net.py` takes about 1 min and `train_traj_net.py` about 3 min.
* `train_ppo_lag.py` runs 3 M steps. Use `--n-envs` = cores − 1; `--resume --max-minutes N` splits it into segments.

## Repository layout
```
configs/default.yaml     all parameters (single source of truth)
docs/                    WRITEUP (illustrated tour), DESIGN (architecture, contracts, targets), PLAN, CHECKLIST,
                         PROGRESS (status + decisions)
src/spacearm/            config, robot_model, sim, kinematics, datagen, models/ (distance_net, traj_net), losses,
                         planner, control, envs/ (space_reach_env, toy_env), rl/ (vec_env, ppo_lag), safety/shield, export,
                         gui/ (scene, backend, app: the interactive explorer)
scripts/                 one script per step: make_urdf, render_robot, view_robot, check_world_model, gen_data,
                         train_distance_net, train_traj_net, eval_level1, play_env, train_ppo_lag, eval_level2,
                         export_policy, demo_video; v1.1: gui, eval_level1_stages, eval_level2_variants,
                         make_writeup_figures
tests/                   executable contracts, one file per phase
models/                  trained weights (committed): distance_net, traj_net, ppo_lag (+ checkpoint), actor.npz/.onnx
reports/                 metrics JSON, progress CSVs, plots, tables, phase reports, demo video
cloud/, .claude/         Claude Code cloud setup (venv build script, SessionStart hook)
data/ runs/ videos/      generated, git-ignored
```

## Method in one paragraph
A neural collision-distance field (SiLU MLP on sin/cos of the joint angles) learns the clearance between the arm and the spacecraft from 200k simulated configurations. A trajectory network outputs Bezier control points that are squashed into the joint limits, so every path respects them by construction. It is trained without labels, on outcomes: exact differentiable forward kinematics for reaching, the distance field for clearance, and a smoothness term. At run time the proposal is refined, polished with damped-least-squares IK and verified in PyBullet. Level 2 adds a residual PPO-Lagrangian policy on top of a classical Jacobian controller: the constraint weight is learned by dual ascent, the critic sees privileged simulator truth while the actor only sees onboard sensors, and a model-based safety shield filters every command.
