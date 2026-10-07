| Set | Stage | Success | Collisions | Reach < 2 cm | Median reach error | Median min. clearance | Median time |
|---|---|---|---|---|---|---|---|
| random (200) | Baseline: PyBullet IK + straight line | 92.0 % | 3.5 % | 95.5 % | 0.14 cm | 12.6 cm | 1 ms |
| random (200) | TrajNet alone | 85.0 % | 1.5 % | 85.5 % | 1.02 cm | 14.3 cm | 3 ms |
| random (200) | TrajNet + refine | 56.0 % | 0.5 % | 56.0 % | 1.85 cm | 13.9 cm | 140 ms |
| random (200) | TrajNet + IK polish | 98.0 % | 1.5 % | 99.5 % | 0.00 cm | 14.2 cm | 12 ms |
| random (200) | TrajNet + refine + polish | 99.5 % | 0.5 % | 100.0 % | 0.00 cm | 13.9 cm | 154 ms |
| random (200) | Full planner (+ PyBullet check, retry) | 100.0 % | 0.0 % | 100.0 % | 0.00 cm | 13.9 cm | 175 ms |
| hard (100) | Baseline: PyBullet IK + straight line | 0.0 % | 100.0 % | 82.0 % | 0.21 cm | -7.0 cm | 2 ms |
| hard (100) | TrajNet alone | 66.0 % | 6.0 % | 67.0 % | 1.32 cm | 11.2 cm | 4 ms |
| hard (100) | TrajNet + refine | 54.0 % | 1.0 % | 54.0 % | 1.95 cm | 10.7 cm | 144 ms |
| hard (100) | TrajNet + IK polish | 93.0 % | 7.0 % | 100.0 % | 0.00 cm | 10.6 cm | 13 ms |
| hard (100) | TrajNet + refine + polish | 100.0 % | 0.0 % | 100.0 % | 0.00 cm | 10.6 cm | 155 ms |
| hard (100) | Full planner (+ PyBullet check, retry) | 100.0 % | 0.0 % | 100.0 % | 0.00 cm | 10.6 cm | 164 ms |

Every result is checked in PyBullet: success = no contact at 100 path points, reach < 2 cm, within joint limits. Times on 1 CPU thread.
