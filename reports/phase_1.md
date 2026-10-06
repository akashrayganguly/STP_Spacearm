## Phase 1 — World model (URDF, free-floating simulator, exact FK): DONE

Built:
* `src/spacearm/robot_model.py` — URDF generated from the config: bus (free-floating base), 2 panels, payload(s), pedestal, 7 capsule links, TCP frame; `joint_limits`, `spacecraft_links`, `total_mass`, `write_urdf`.
* `src/spacearm/sim.py` — `SpaceRobotSim`: zero gravity, zero damping, velocity motors, collision pairs (30 body + 10 self) with ACM, geometric distances (`getClosestPoints`), ghost sphere obstacles, base/frames/CoM helpers.
* `src/spacearm/kinematics.py` — `ArmKinematics`: batched differentiable FK (link1..7 + TCP), analytic Jacobian, capsule segments; `tcp_numpy`.
* `scripts/make_urdf.py` — writes `assets/space_robot.urdf` (for viewing; the simulator builds its own copy from the config).
* `scripts/render_robot.py` — offscreen snapshots → `reports/figures/robot_configs.png`, `robot_views.png`.
* `scripts/view_robot.py` — GUI sliders + live clearance, for the Windows workstation (not run here: no display).
* `scripts/check_world_model.py` — measures the CHECKLIST numbers → `reports/phase1/world_model_check.json`.

Commands I ran:
* `python -m pytest tests/test_p1_robot_sim.py tests/test_p1_kinematics.py -v` — the 18 Phase 1 contracts — Windows: `pytest tests\test_p1_robot_sim.py tests\test_p1_kinematics.py -v`
* `python scripts/make_urdf.py` — writes the URDF from the config — Windows: `python scripts\make_urdf.py`
* `python scripts/render_robot.py` — headless pictures of named configurations with their clearance — Windows: same with `\`; on the workstation also `python scripts\view_robot.py` (GUI sliders)
* `python scripts/check_world_model.py` — FK error, momentum, capsule match, ghost obstacles, ACM worst case, speed — Windows: `python scripts\check_world_model.py`
* `python -m pyflakes src/spacearm scripts` — static check for unused names / typos (pyflakes is only in the session venv, not a project dependency)
* `python -m pytest --collect-only -q` — 18 collected, 6 errors (the Phase 2–7 modules not written yet)

Tests: `test_p1_robot_sim.py` 11 passed, `test_p1_kinematics.py` 7 passed → **18 passed, 0 skipped** (1.2 s).

Results vs targets (DESIGN §8):
| Metric | Target | Got |
|---|---|---|
| FK vs PyBullet (500 random configs, all 8 frames) | < 1e-5 m | **1.2e-7 m** |
| CoM drift, 2 s motion | < 2 mm | **0.02 mm** |
| Base reaction, 2 s motion | > 0.5° | **2.8°** |
| Collision pairs body / self | 30 / 10 | 30 / 10 |
| Total mass | 212.01 kg (config) | 212.01 kg |

Extra checks (not covered by the tests):
| Check | Got | Why it matters |
|---|---|---|
| Analytic sphere–capsule vs PyBullet distance (200 cases) | 5e-16 m | Phase 7 shield relies on the analytic capsules (its test allows 2 mm) |
| Ghost obstacle swept through the arm | arm passes 16 cm deep, CoM drift 0.03 mm, joint still tracks | no contact forces hide collisions (DESIGN D4) |
| ACM pairs, worst distance over 20k configs | ≥ 2.1 cm (link5–link7), all > 0 | skipped pairs really cannot collide |
| `set_q` + `min_distances` | 0.16 ms/config | Phase 2 budget ~0.15 ms → 200k samples ≈ 35 s |
| Uniform collision share (98 % of range) | 13.4 % (body 13.4 %, self 0.06 %) | DESIGN expects ~12 %; checked again in Phase 2 |

Key ideas (short):
* **Frames:** each link frame sits on its joint; the capsule runs 0..L along the link's z. `getLinkState()[4]` is that frame, `[0]` is the centre of mass (10 cm away for link3, a test guards this).
* **ACM:** pairs that are always close but can never touch (e.g. link5–link7 across the short wrist, min 2.1 cm) are skipped; otherwise they would cap the clearance and every margin would be infeasible.
* **Free floating:** no external forces → total momentum stays zero → the system CoM cannot move, so the bus rotates against the arm (2.8°) and stops when the arm stops. Any damping would leak momentum, so it is zero everywhere.

Decisions / deviations and why:
* **New config key `robot.arm.tcp_mass: 0.01`.** DESIGN §4 gives the TCP mass (0.01 kg), but the config didn't have it, and `total_mass` must match the URDF. It's an addition only: no test or other value changed.
* **`d_body` never exceeds 18 cm.** Link 2's capsule starts in a sphere centred on joint 2, which never moves, so link2–pedestal clearance ≤ 0.84 − 0.06 − 0.60 = 0.18 m. This is harmless: all margins and activation distances are ≤ 15 cm, and the test's 30-pair count includes this pair on purpose.
* **Pitfall I hit while checking (worth remembering for Phase 5):** after a dynamics run the base is no longer at identity, so world ≠ body frame. My first capsule check compared world-frame PyBullet distances with body-frame FK and showed an 88 mm "error". Resetting the base gives 0.000 mm. Code that mixes FK (body frame) with world obstacles must transform with `world_to_body`.
* Uniform self-collisions are rare (0.06 %), so the DistanceNet's `d_self` head will see few negative labels. Phase 2's boundary sampling should help; watch the self-head numbers in Phase 3.
* `assets/space_robot.urdf` is committed for viewing; regenerate it with `make_urdf.py` after any geometry change.

Next: Phase 2 — Synthetic kinematics/collision dataset (`datagen.py`, `scripts/gen_data.py`). Reply "go" to start.
