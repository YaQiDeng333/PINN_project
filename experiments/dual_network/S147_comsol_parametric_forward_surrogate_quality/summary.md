# S147 COMSOL parametric forward surrogate quality gate

## 目的

S147 在 COMSOL V2 train / val / test 上训练 S146 learned forward surrogate，判断它是否足够作为 forward consistency referee。

## 配置

- train NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- targets: S113 raw parametric targets
- surrogate: MLP, `hidden_dim=256`, `num_layers=4`
- steps: `5000`
- signal normalization: train-only z-score

## 结果

| split | signal_nrmse_raw | signal_corr | peak_abs_nrmse |
| --- | ---: | ---: | ---: |
| train | 3.767854e-01 | 9.258671e-01 | 9.827892e-02 |
| val | 5.026852e-01 | 8.657639e-01 | 1.380844e-01 |
| test | 4.577952e-01 | 8.886174e-01 | 9.483848e-02 |

Per-channel corr 也保持在约 `0.865-0.891`，说明 surrogate 捕捉了主要 multi-height waveform。

## Gate 判断

S147 gate 通过：

- val/test `signal_corr >= 0.80`；
- val/test `signal_nrmse_raw < 1.0`。

因此可以继续 S148/S149，在 inverse training 中使用 in-memory frozen forward surrogate 做 consistency probe。

## 自评

- Surrogate train corr 高于 val/test，存在一定 train-side fit，但 held-out corr 仍满足当前 consistency referee 的最低门槛。
- 该 surrogate 只是 learned approximation，不等同 COMSOL forward solver。
- 后续 forward consistency 若无改善，需要区分 surrogate bias 与 inverse objective 本身的限制。
