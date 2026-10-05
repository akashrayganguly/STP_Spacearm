# spacearm — collision-free path planning for a 7-DOF arm on a free-floating spacecraft

**Level 1:** a kinematics-aware neural planner that turns *(current joint angles, target point)* into a smooth, collision-free joint trajectory and goal joint angles that never hit the spacecraft body.
**Level 2:** a lightweight constrained-RL policy (~8k parameters, < 1 ms) that re-routes the arm in real time around surprise obstacles and through hardware faults, while the spacecraft floats and reacts to every arm motion.

Everything is simulated in PyBullet and runs on a Windows workstation (CPU is enough).

## Quick start (Anaconda Prompt)
```bat
cd /d C:\projects\spacearm
conda env create -f environment.yml
conda activate spacearm
pip install -e .
pytest tests\test_p1_robot_sim.py -v
```
Then follow `docs/PLAN.md` phase by phase.

## Repository layout
```
configs/default.yaml     all parameters (single source of truth)
docs/DESIGN.md           architecture, decisions, module contracts, targets
docs/PLAN.md             phases with Anaconda Prompt commands
docs/CHECKLIST.md        acceptance boxes per phase
docs/NEXT_CHAT_INSTRUCTIONS.md   hand-off brief for the build chat
src/spacearm/            package (config, robot_model, sim, kinematics, datagen, models/, losses,
                         planner, control, envs/, rl/, safety/, export)
scripts/                 one script per phase step (data, training, evaluation, export, demo)
tests/                   unit + acceptance tests, one file per phase
data/ runs/ videos/      generated (git-ignored)      models/   trained weights (committed)
```

## Method in one paragraph
A neural collision-distance field (SiLU MLP on sin/cos of the joint angles) learns the clearance between the arm and the spacecraft from 200k simulated configurations. A trajectory network outputs Bezier control points that are squashed into the joint limits, so every path respects them by construction; it is trained without labels, on outcomes: exact differentiable forward kinematics for reaching, the distance field for clearance, and a smoothness term. At run time the proposal is refined, polished with damped-least-squares IK and verified in PyBullet. Level 2 adds a residual PPO-Lagrangian policy on top of a classical Jacobian controller: the constraint weight is learned by dual ascent, the critic sees privileged simulator truth while the actor only sees onboard sensors, and a model-based safety shield filters every command.

## Results
*(fill in after Phases 4 and 7: Level-1 table, Level-2 S1–S4 table, latency table, demo video link)*
