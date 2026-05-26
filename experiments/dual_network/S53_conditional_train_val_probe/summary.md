# S53 conditional train/val generalization probe

## Data

Data generation command:

```powershell
python data_generator_v2.py --train-samples 80 --val-samples 20 --test-samples 0 --grid-x 20 --grid-y 10 --output-dir experiments/dual_network/S53_conditional_train_val_probe/data --seed 1053
```

Generated data:

- train npz: `experiments/dual_network/S53_conditional_train_val_probe/data/training_data_train.npz`
- val npz: `experiments/dual_network/S53_conditional_train_val_probe/data/training_data_val.npz`
- train samples: 80
- val samples: 20
- resolution: 20x10

## Configurations

### big_bce_dice

- `steps=2000`
- `lr=1e-3`
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

### big_bce_dice_mu1e-4

- same as `big_bce_dice`
- `lambda_mu_mse=1e-4`

## Results

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| `big_bce_dice` | train | 9.350000e-01 | 6.662500e+00 | 2.731668e+03 | 8.359778e+00 |
| `big_bce_dice` | val | 7.141148e-02 | 5.250000e+00 | 5.220703e+04 | 6.033646e+01 |
| `big_bce_dice_mu1e-4` | train | 0.000000e+00 | 0.000000e+00 | 3.511602e+04 | 3.516792e+01 |
| `big_bce_dice_mu1e-4` | val | 0.000000e+00 | 0.000000e+00 | 3.892078e+04 | 3.897649e+01 |

All train and val metrics are finite.

Train-val IoU gap:

- `big_bce_dice`: `8.635885e-01`
- `big_bce_dice_mu1e-4`: `0.000000e+00`, but this is because both train and val masks collapsed to empty predictions.

## Current judgment

1. `big_bce_dice` fits the train set strongly, but val IoU is very low. This indicates overfitting rather than reliable conditional generalization.
2. `big_bce_dice_mu1e-4` improves val continuous `mu_mse` / `mu_mae` relative to `big_bce_dice`, but it collapses mask prediction to empty masks on both train and val.
3. Current conditional supervised setup has not yet demonstrated useful train/val generalization at 20x10.
4. S53 is a small-scale train/val probe only and does not support main-baseline replacement claims.

## Next-step recommendation

- Do not move to main-baseline comparison yet.
- Diagnose why train overfit does not transfer to val: signal encoder capacity, lack of augmentation, loss imbalance, and dataset diversity are likely candidates.
- Consider validation-aware model selection, smaller capacity / regularization, or a better supervised objective before adding weak-form loss.
