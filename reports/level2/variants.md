Success / collisions, 100 identical fixed-seed episodes per cell (seeds 40000+).

| Group | Variant | prior | prior + shield | RL | RL + shield |
|---|---|---|---|---|---|
| difficulty | d = 0 (nominal) | 89 % / 1 % | 87 % / 0 % | 88 % / 2 % | 89 % / 0 % |
| difficulty | d = 0.25 | 84 % / 2 % | 81 % / 0 % | 84 % / 2 % | 85 % / 0 % |
| difficulty | d = 0.5 | 73 % / 3 % | 71 % / 0 % | 77 % / 3 % | 77 % / 0 % |
| difficulty | d = 0.75 | 63 % / 3 % | 62 % / 0 % | 69 % / 2 % | 71 % / 0 % |
| difficulty | d = 1 (everything) | 49 % / 3 % | 49 % / 0 % | 61 % / 2 % | 61 % / 0 % |
| obstacle | static ball, 5-12 cm | 37 % / 3 % | 37 % / 0 % | 54 % / 3 % | 53 % / 0 % |
| obstacle | drifting ball, up to 3 cm/s | 63 % / 3 % | 62 % / 2 % | 70 % / 3 % | 68 % / 1 % |
| obstacle | small ball, 5-7 cm | 51 % / 1 % | 50 % / 0 % | 62 % / 3 % | 61 % / 0 % |
| obstacle | large ball, 10-12 cm | 52 % / 1 % | 52 % / 1 % | 61 % / 2 % | 61 % / 0 % |
| faults | encoder bias, up to 2 deg | 88 % / 2 % | 87 % / 0 % | 87 % / 3 % | 88 % / 0 % |
| faults | weak motors, 70-100 % | 83 % / 2 % | 81 % / 0 % | 85 % / 3 % | 86 % / 0 % |
| faults | encoder slip, 3 deg (every episode) | 89 % / 1 % | 87 % / 0 % | 88 % / 2 % | 89 % / 0 % |
| noise | sensor noise (training level) | 89 % / 1 % | 87 % / 0 % | 89 % / 1 % | 89 % / 0 % |
| stress | ball 15-20 cm | 54 % / 2 % | 54 % / 0 % | 63 % / 1 % | 63 % / 0 % |
| stress | ball drifting up to 6 cm/s | 68 % / 5 % | 67 % / 3 % | 79 % / 4 % | 76 % / 2 % |
| stress | encoder bias up to 4 deg | 88 % / 2 % | 87 % / 0 % | 89 % / 3 % | 89 % / 1 % |
| stress | sensor noise x3 | 89 % / 1 % | 88 % / 0 % | 90 % / 1 % | 90 % / 0 % |
