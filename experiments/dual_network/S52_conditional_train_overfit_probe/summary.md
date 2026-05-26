# S52 conditional train-set overfit probe

## Data source

Reused S51 train data:

`experiments/dual_network/S51_conditional_supervised_small_data_probe/data/training_data_train.npz`

Samples:

`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`

S51 reference:

- avg `defect_iou = 5.228869e-01`
- avg `defect_area_pred = 5.050000e+00`
- avg `mu_mse = 3.178682e+04`
- avg `mu_mae = 1.286587e+02`

## Configurations

### longer_bce_dice

- `steps=1000`
- `lr=1e-3`
- `hidden_dim=64`
- `num_layers=3`
- `latent_dim=32`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

### longer_bce_dice_mu1e-4

- same as `longer_bce_dice`
- `lambda_mu_mse=1e-4`

### bigger_bce_dice

- `steps=1000`
- `lr=1e-3`
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

## Final train metrics

| config | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| `longer_bce_dice` | 8.211111e-01 | 5.600000e+00 | 6.942074e+03 | 2.143848e+01 |
| `longer_bce_dice_mu1e-4` | 7.919444e-01 | 5.500000e+00 | 5.815452e+03 | 1.315498e+01 |
| `bigger_bce_dice` | 9.375000e-01 | 5.950000e+00 | 2.941439e+03 | 1.233218e+01 |

All metrics are finite.

Best train IoU:

- `bigger_bce_dice`

Best continuous `mu_mse` / `mu_mae`:

- `bigger_bce_dice`

## Current judgment

1. `longer_bce_dice` improves average train IoU from S51 `0.523` to `0.821`, so S51 was partly limited by training duration.
2. Adding light `mu_mse` improves continuous `mu_mse` / `mu_mae` compared with `longer_bce_dice`, but slightly reduces train IoU in this run.
3. `bigger_bce_dice` gives the strongest result across IoU and continuous errors, suggesting current conditional train-set fit is capacity-sensitive.
4. S52 remains a train-set overfit probe only. It is not evidence of test-set generalization.

## Next-step recommendation

- Use `bigger_bce_dice` as the next conditional supervised training candidate.
- Build a train/val/test conditional runner before making generalization claims.
- Keep weak-form / physics loss as a later addition after the supervised conditional baseline has a stable validation protocol.
