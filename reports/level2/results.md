| Scenario | Method | Success | Collisions | Mean cost | Time to reach | Base rotation | Shield interventions |
|---|---|---|---|---|---|---|---|
| S1 nominal | prior | 88 % | 5 % | 0.38 | 9.7 s | 6.3° | — |
| S1 nominal | prior+shield | 83 % | 0 % | 0.00 | 9.9 s | 6.4° | 13.7 % of steps |
| S1 nominal | rl | 94 % | 2 % | 0.40 | 6.6 s | 6.6° | — |
| S1 nominal | rl+shield | 89 % | 0 % | 0.00 | 6.9 s | 6.6° | 14.6 % of steps |
| S2 obstacles only | prior | 58 % | 6 % | 0.66 | 10.4 s | 5.9° | — |
| S2 obstacles only | prior+shield | 53 % | 1 % | 0.30 | 10.2 s | 5.9° | 10.4 % of steps |
| S2 obstacles only | rl | 71 % | 4 % | 0.53 | 7.5 s | 6.4° | — |
| S2 obstacles only | rl+shield | 68 % | 0 % | 0.00 | 7.9 s | 6.4° | 11.6 % of steps |
| S2 obstacles only | prior (reflex off) | 39 % | 59 % | 2.20 | 8.3 s | 4.8° | — |
| S3 faults + noise only | prior | 85 % | 3 % | 0.56 | 11.2 s | 6.2° | — |
| S3 faults + noise only | prior+shield | 77 % | 0 % | 0.00 | 11.0 s | 6.2° | 12.6 % of steps |
| S3 faults + noise only | rl | 92 % | 3 % | 0.38 | 7.8 s | 6.5° | — |
| S3 faults + noise only | rl+shield | 87 % | 0 % | 0.12 | 8.2 s | 6.5° | 13.6 % of steps |
| S4 everything | prior | 45 % | 7 % | 0.71 | 11.2 s | 5.5° | — |
| S4 everything | prior+shield | 41 % | 1 % | 0.05 | 10.9 s | 5.5° | 9.1 % of steps |
| S4 everything | rl | 61 % | 4 % | 0.38 | 8.0 s | 6.3° | — |
| S4 everything | rl+shield | 58 % | 0 % | 0.09 | 8.2 s | 6.3° | 10.7 % of steps |
| S4 everything | prior (reflex off) | 34 % | 60 % | 2.67 | 9.5 s | 4.4° | — |

100 identical fixed-seed episodes per cell (seeds 10000+).
