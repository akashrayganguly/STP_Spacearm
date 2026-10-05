| Set | Method | Success | Collisions | Median reach error | Median plan time |
|---|---|---|---|---|---|
| random (200) | Baseline (PyBullet IK + line) | 92.0 % | 3.5 % | 0.14 cm | 1 ms |
| random (200) | Network only | 85.0 % | 1.5 % | 1.02 cm | 5 ms |
| random (200) | Network + IK polish | 98.0 % | 1.5 % | 0.00 cm | 8 ms |
| random (200) | Full planner (refine + polish + verify) | 100.0 % | 0.0 % | 0.00 cm | 99 ms |
| hard (100) | Baseline (PyBullet IK + line) | 0.0 % | 100.0 % | 0.21 cm | 1 ms |
| hard (100) | Network only | 66.0 % | 6.0 % | 1.32 cm | 2 ms |
| hard (100) | Network + IK polish | 93.0 % | 7.0 % | 0.00 cm | 28 ms |
| hard (100) | Full planner (refine + polish + verify) | 100.0 % | 0.0 % | 0.00 cm | 96 ms |

Full planner: retry rate 0.5 % (random), 0.0 % (hard); median time incl. PyBullet verification 114 ms (random).
Hard set: 100 baseline-colliding queries in 2170 random queries (4.6 %).

| Open-loop on the free-floating spacecraft (20 plans) | median | max |
|---|---|---|
| Inertial TCP miss | 17.2 cm | 33.3 cm |
| Base rotation | 6.4° | 12.4° |
| Body-frame reach error (tracking) | 0.00 cm | 0.00 cm |
