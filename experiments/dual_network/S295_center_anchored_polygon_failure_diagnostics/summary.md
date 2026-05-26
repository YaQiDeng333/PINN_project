# S295 Center-Anchored Polygon Failure Diagnostics

## Split Metrics

- test: IoU mean/min `0.084416` / `0.000000`, x/y bin acc `0.750000` / `0.100000`, zero-IoU rate `0.800000`, presence/type correctness `0.900000` / `0.900000`.
- train: IoU mean/min `0.989276` / `0.857143`, x/y bin acc `1.000000` / `1.000000`, zero-IoU rate `0.000000`, presence/type correctness `1.000000` / `1.000000`.
- val: IoU mean/min `0.072402` / `0.000000`, x/y bin acc `0.500000` / `0.200000`, zero-IoU rate `0.800000`, presence/type correctness `0.800000` / `0.500000`.

## Held-Out Failure Mechanism

- Held-out zero-IoU samples: `16` / `20`; with any y-bin wrong `16`, any x-bin wrong `8`, any bin wrong `16`.
- Held-out present components with correct bins have local vertex grid MAE mean `0.867101`; components with wrong bins have mean `2.487230`.
- Held-out samples with all center bins correct: `2` / `20`, mean IoU `0.468456`.

## Required Answers

1. Zero-IoU is primarily center-bin driven: `16` / `16` zero-IoU samples have at least one center-bin error.
2. Y-bin is the stronger bottleneck: zero-IoU samples with y-bin wrong `16`, x-bin wrong `8`.
3. Local shape is secondary here: correct-bin held-out components have lower local vertex grid MAE (`0.867101`) than wrong-bin components (`2.487230`).
4. Hardest hard_case by held-out mean IoU is `both_bins_wrong_like` with mean IoU `0.000000` and zero-IoU rate `1.000000`.
5. Hardest component slot by held-out mean IoU is slot `1` with mean IoU `0.000000`.
6. Rotated and multi-component groups are harder in held-out: rotated=False: zero-rate 0.400000, y-bin acc 0.400000; rotated=True: zero-rate 0.933333, y-bin acc 0.066667; multi=False: zero-rate 0.714286, y-bin acc 0.214286; multi=True: zero-rate 1.000000, y-bin acc 0.000000.
7. Worst samples share y-bin errors and sparse train-bin coverage; see `worst_heldout_samples.csv` and S296 coverage tables.

This diagnostic is read-only. It does not rerun training, change the model, or change the runner.
