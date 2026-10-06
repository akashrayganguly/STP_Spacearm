| Component (1 CPU thread) | p50 | p99 |
|---|---|---|
| Actor, NumPy (.npz) | 0.012 ms | 0.029 ms |
| Actor, ONNX Runtime | 0.008 ms | 0.024 ms |
| Actor, PyTorch | 0.044 ms | 0.101 ms |
| Reflex prior (DLS + null space + repulsion) | 0.303 ms | 0.588 ms |
| Safety shield, command accepted (3 candidates in one batch) | 0.175 ms | 0.384 ms |
| Safety shield, obstacle 15 cm above the TCP (escape) | 3.267 ms | 4.488 ms |
| Full control step incl. physics (24 substeps), features, policy, prior, shield | 4.549 ms | 7.283 ms |

Actor: 8270 parameters (deployed: 8263 weights + normaliser); exports match PyTorch to 7.5e-08. Control period: 100 ms.
