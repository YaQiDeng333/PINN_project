# S25 80x40 10-Sample High-Resolution Feasibility Probe

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 10 --val-samples 0 --test-samples 0 --grid-x 80 --grid-y 40 --output-dir experiments\dual_network\S25_80x40_10sample_feasibility_probe\data --seed 1025
```

Generated train data:

- `experiments/dual_network/S25_80x40_10sample_feasibility_probe/data/training_data_train.npz`

## Runner Configuration

Both runs used:

- `grid_x=80`, `grid_y=40`
- `sample_indices=0,1,2,3,4,5,6,7,8,9`
- `outer_steps=20`
- `phi_steps=20`
- `mu_steps=20`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`

Compared runs:

- `baseline`: `lambda_mask_bce_prior=0.0`, `mask_prior_temperature=50.0`
- `temp25_lambda1`: `lambda_mask_bce_prior=1.0`, `mask_prior_temperature=25.0`

S25 is a high-resolution feasibility probe. It uses an `80x40` grid and `20/20/20` training steps, so it should not be treated as a strict numerical comparison against S24's `40x20` grid with `30/30/30` steps.

## Ten-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 6.824286e-02 | 1.418300e+03 | 3.996649e+05 | 4.761472e+02 |
| temp25_lambda1 | 5.102481e-01 | 2.611000e+02 | 1.519206e+05 | 3.708853e+02 |

## Per-Sample IoU

| sample | baseline | temp25_lambda1 |
| ---: | ---: | ---: |
| 0 | 4.629600e-02 | 3.420070e-01 |
| 1 | 4.856700e-02 | 4.935900e-01 |
| 2 | 4.724400e-02 | 4.800000e-01 |
| 3 | 5.457900e-02 | 6.627910e-01 |
| 4 | 3.947400e-02 | 5.785120e-01 |
| 5 | 5.871700e-02 | 7.191010e-01 |
| 6 | 1.183250e-01 | 2.772280e-01 |
| 7 | 6.213500e-02 | 4.810810e-01 |
| 8 | 9.793200e-02 | 8.864860e-01 |
| 9 | 1.091600e-01 | 1.816840e-01 |

## Stability Check

- Both 10-sample runner jobs completed successfully.
- No NaN/inf was observed in the aggregated metrics.
- `metrics.csv` was generated for both runs.
- No model checkpoints or weights were generated.
- `temp25_lambda1` improves IoU on all 10 samples.
- Two `temp25_lambda1` samples are weak: sample 6 (`0.277228`) and sample 9 (`0.181684`).

## Current Judgment

- `temp25_lambda1` still shows a clear semi-supervised BCE upper-bound improvement trend at `80x40`: average IoU improves from `6.824286e-02` to `5.102481e-01`, and predicted defect area drops from `1418.3` to `261.1`.
- The result is much weaker than the S24 `40x20` default validation, so `80x40` is not yet stable under the current `20/20/20` configuration.
- The high-resolution setting likely needs resolution-specific adaptation, such as more training steps, adjusted `test_radius`, more centers, stronger or sharper mask prior, or larger network capacity.
- These results remain semi-supervised / diagnostic upper-bound results because BCE and mask priors use `mu_label < 500`; they do not prove unsupervised weak-form inversion success.
