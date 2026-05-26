# S65 synthetic multi-height Bz proxy probe

## Data source

- Original S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Original S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Original S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Proxy outputs:
  - `experiments/dual_network/S65_synthetic_multiheight_proxy_probe/data/train_proxy.npz`
  - `experiments/dual_network/S65_synthetic_multiheight_proxy_probe/data/val_proxy.npz`
  - `experiments/dual_network/S65_synthetic_multiheight_proxy_probe/data/test_proxy.npz`

## Proxy construction

This is a synthetic proxy, not physical COMSOL multi-height data.

The builder reads a single-channel `signals [N,L]` array and writes `signals [N,3,L]`:

- channel 0: raw Bz signal
- channel 1: moving-average smoothing with `window=3`, then decay factor `0.8`
- channel 2: moving-average smoothing with `window=7`, then decay factor `0.6`

The proxy files include metadata fields such as `source_signal_shape`, `proxy_signal_shape`, `signal_channels`, `signal_channel_names`, `source_type`, and `proxy_warning`.

Observed proxy shapes:

- train proxy: `signals shape (1000, 3, 20)`
- val proxy: `signals shape (200, 3, 20)`
- test proxy: `signals shape (200, 3, 20)`

## Configuration

Shared runner configuration:

- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `encoder_type=mlp`
- `conditioning_mode=concat`
- `signal_normalization=per_sample_zscore`
- `signal_feature_mode=raw`
- `point_signal_mode=none`
- `mask_head_mode=mu_threshold`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

Compared configs:

| config | data |
|---|---|
| `single_channel_reference` | original S55 single-channel train / val / test `.npz` |
| `synthetic_multiheight_proxy` | S65 proxy train / val / test `.npz` |

## S65 average metrics

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `single_channel_reference` | train | 9.032763e-01 | 6.758000e+00 | 3.090373e+03 | 5.138575e+00 |
| `single_channel_reference` | val | 9.708926e-02 | 5.930000e+00 | 5.136065e+04 | 5.453059e+01 |
| `single_channel_reference` | test | 9.556112e-02 | 6.435000e+00 | 5.259533e+04 | 5.573554e+01 |
| `synthetic_multiheight_proxy` | train | 5.832855e-01 | 5.784000e+00 | 1.385152e+04 | 2.264801e+01 |
| `synthetic_multiheight_proxy` | val | 1.055699e-01 | 4.905000e+00 | 4.319456e+04 | 5.193633e+01 |
| `synthetic_multiheight_proxy` | test | 1.116188e-01 | 5.180000e+00 | 4.297756e+04 | 5.214038e+01 |

All metrics are finite and train / eval / test metrics were generated for both configs.

## Comparison with S57 / S63

- S57 `per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.
- S63 `raw_reference`: train / val / test IoU = `8.758866e-01` / `9.413832e-02` / `1.040388e-01`.
- S63 `raw_abs_grad`: train / val / test IoU = `9.142336e-01` / `1.081610e-01` / `1.036962e-01`.
- S65 `synthetic_multiheight_proxy`: train / val / test IoU = `5.832855e-01` / `1.055699e-01` / `1.116188e-01`.

## Current judgment

1. The proxy multi-channel runner path is functional: `[N,3,20]` proxy data is read, flattened to `[B,60]`, and train / val / test metrics are produced.
2. The synthetic proxy improves held-out test IoU over the S65 single-channel reference and lowers held-out `mu_mse` / `mu_mae`.
3. The proxy train IoU is much lower than the single-channel reference, and the final training curve showed instability, so this is not a clean optimization improvement.
4. Because the proxy channels are only smoothed / decayed versions of the same single Bz signal, this result cannot be interpreted as evidence for real physical multi-height performance.
5. The result is a useful interface/proxy signal, but not a deployable or physics-level conclusion.

## Next-step recommendation

- Do not claim S65 proves COMSOL multi-height effectiveness.
- If continuing this direction, move to a real COMSOL / multi-height Bz dataset or a more physically grounded forward-data conversion path.
- If staying with synthetic proxy data, first add validation-aware selection or shorter / stabilized training to avoid selecting a degraded final checkpoint-free endpoint.
