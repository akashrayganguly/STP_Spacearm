# spacearm — collision-free path planning for a 7-DOF arm on a free-floating spacecraft

**Level 1:** a kinematics-aware neural planner that turns *(current joint angles, target point)* into a smooth, collision-free joint trajectory and goal joint angles that never hit the spacecraft body.
**Level 2:** a lightweight constrained-RL policy (8k parameters, ~0.01 ms) that re-routes the arm in real time around surprise obstacles and through hardware faults, while the spacecraft floats and reacts to every arm motion.

Everything is simulated in PyBullet and runs on a CPU. The trained models are in the repo (`models/`), so every evaluation below can be re-run directly.

![demo: prior only (left) vs RL residual + shield (right)](reports/figures/demo_strip.png)
Demo video: [`reports/demo.mp4`](reports/demo.mp4). The same episode is shown twice. A surprise obstacle appears at about 4 s on the way to the target. The classical reflex (left) hovers safely beside it and times out 3.3 cm short. The RL residual with the safety shield (right) re-routes around it and arrives at 11.1 s.

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

### Level 2 — residual PPO-Lagrangian policy + safety shield

100 identical fixed-seed episodes per cell (seeds 10000–10099, never used for training or model selection):

| Scenario | Method | Success | Collisions | Near-miss cost / ep. | Time to reach |
|---|---|---|---|---|---|
| S1 nominal | prior (reflex) | 88 % | 5 % | 0.38 | 9.7 s |
| S1 nominal | prior + shield | 87 % | 0 % | 0.04 | 10.0 s |
| S1 nominal | RL | 94 % | 2 % | 0.40 | 6.6 s |
| S1 nominal | **RL + shield** | **91 %** | **0 %** | **0.00** | 7.0 s |
| S2 obstacles only | prior | 58 % | 6 % | 0.66 | 10.4 s |
| S2 obstacles only | prior + shield | 56 % | 0 % | 0.41 | 10.6 s |
| S2 obstacles only | RL | 71 % | 4 % | 0.53 | 7.5 s |
| S2 obstacles only | **RL + shield** | **68 %** | **0 %** | **0.00** | 7.7 s |
| S2 obstacles only | ablation: prior, reflex off | 39 % | 59 % | 2.20 | 8.3 s |
| S3 faults + noise only | prior | 85 % | 3 % | 0.56 | 11.2 s |
| S3 faults + noise only | prior + shield | 84 % | 0 % | 0.00 | 11.5 s |
| S3 faults + noise only | RL | 92 % | 3 % | 0.38 | 7.8 s |
| S3 faults + noise only | **RL + shield** | **89 %** | **1 %** | 0.24 | 8.2 s |
| S4 everything | prior | 45 % | 7 % | 0.71 | 11.2 s |
| S4 everything | prior + shield | 45 % | 1 % | 0.05 | 11.5 s |
| S4 everything | RL | 61 % | 4 % | 0.38 | 8.0 s |
| S4 everything | **RL + shield** | **59 %** | **2 %** | 0.35 | 8.3 s |
| S4 everything | ablation: prior, reflex off | 34 % | 60 % | 2.67 | 9.5 s |

Full table with base rotation and shield intervention rates: [`reports/level2/results.md`](reports/level2/results.md).

* **The S4 target is met:** RL + shield has 2 % collisions (target ≤ 3 %) and 59 % success, against the prior's 45 %. The stretch goal (≥ 70 % success) is not met.
* **S1:** success is not worse than the prior's (91 % vs 88 %) and there are no near-misses (0.00 vs 0.38 per episode).
* **The RL residual** adds 6–16 points of success in every scenario and reaches targets about 30 % faster.
* **The shield** removes most collisions: across S1–S4, 20 of 21 for the prior and 10 of 13 for RL. The cost is 0–3 points of success.
* **The reflex matters:** without it the prior collides in about 60 % of obstacle episodes.

Latency on one CPU thread (control period 100 ms):

| Component | p50 | p99 |
|---|---|---|
| Actor, NumPy (`models/actor.npz`) | 0.012 ms | 0.029 ms |
| Actor, ONNX Runtime (`models/actor.onnx`) | 0.008 ms | 0.024 ms |
| Reflex prior | 0.30 ms | 0.59 ms |
| Safety shield, command accepted / escape | 0.18 / 3.3 ms | 0.38 / 4.5 ms |
| Full control step incl. physics, features, policy, prior, shield | 4.5 ms | 7.3 ms |

### Honest limits
* **The Level-2 acceptance test fails at DESIGN's 5 cm shield margin.** `test_trained_policy_is_safe_and_not_worse_than_the_prior` uses the first 40 of the 100 evaluation episodes, and both of the 2 S4 collisions fall inside them (2/40 = 5 % > 3 %). Both were spacecraft grazes: the robot's own clearance estimate (DistanceNet on encoder angles carrying up to 3.6° of bias after a slip) was about 5.5 cm too optimistic.
  * On 100 independent episodes (seeds 30000+), RL + shield has **0 %** collisions at 5 cm.
  * A 6 cm margin removes both collisions on the test seeds and costs about 2 points of success (`reports/level2/margin_sweep_*.json`).
  * Changing the margin is a DESIGN decision, pending the engineer.
* **The shield is a model-based filter, not a proof:** it only knows the measured joint angles, the DistanceNet and the obstacle as seen by vision.
* **Targets are sampled within 85 % of the arm's reach.** With unrestricted targets, about 20 % of nominal episodes were unwinnable: the base recoil carried near-full-stretch targets out of reach.
* **The PPO-Lagrangian constraint (≤ 1 near-miss step per episode) rarely bound** (λ ≤ 0.45). Collision safety comes mostly from the reflex and the shield, not from the learned constraint.

## Reproduce on a Windows workstation (Anaconda Prompt)
```bat
cd /d C:\projects
git clone https://github.com/akashrayganguly/STP_Spacearm.git spacearm
cd spacearm
REM Python 3.11 + PyBullet from conda-forge (PLAN.md Phase 0), then this package in editable mode
conda env create -f environment.yml
conda activate spacearm
pip install -e .
REM fast tests (about 15 s); add -m "slow or not slow" for everything (about 2 min)
pytest
REM evaluations with the committed models
python scripts\eval_level1.py
python scripts\eval_level2.py
python scripts\export_policy.py
python scripts\demo_video.py
REM GUI viewers (workstation only)
python scripts\view_robot.py
python scripts\play_env.py --gui --difficulty 1.0 --episodes 3
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
docs/                    DESIGN (architecture, contracts, targets), PLAN, CHECKLIST, PROGRESS (status + decisions)
src/spacearm/            config, robot_model, sim, kinematics, datagen, models/ (distance_net, traj_net), losses,
                         planner, control, envs/ (space_reach_env, toy_env), rl/ (vec_env, ppo_lag), safety/shield, export
scripts/                 one script per step: make_urdf, render_robot, view_robot, check_world_model, gen_data,
                         train_distance_net, train_traj_net, eval_level1, play_env, train_ppo_lag, eval_level2,
                         export_policy, demo_video
tests/                   executable contracts, one file per phase
models/                  trained weights (committed): distance_net, traj_net, ppo_lag (+ checkpoint), actor.npz/.onnx
reports/                 metrics JSON, progress CSVs, plots, tables, phase reports, demo video
cloud/, .claude/         Claude Code cloud setup (venv build script, SessionStart hook)
data/ runs/ videos/      generated, git-ignored
```

## Method in one paragraph
A neural collision-distance field (SiLU MLP on sin/cos of the joint angles) learns the clearance between the arm and the spacecraft from 200k simulated configurations. A trajectory network outputs Bezier control points that are squashed into the joint limits, so every path respects them by construction. It is trained without labels, on outcomes: exact differentiable forward kinematics for reaching, the distance field for clearance, and a smoothness term. At run time the proposal is refined, polished with damped-least-squares IK and verified in PyBullet. Level 2 adds a residual PPO-Lagrangian policy on top of a classical Jacobian controller: the constraint weight is learned by dual ascent, the critic sees privileged simulator truth while the actor only sees onboard sensors, and a model-based safety shield filters every command.
