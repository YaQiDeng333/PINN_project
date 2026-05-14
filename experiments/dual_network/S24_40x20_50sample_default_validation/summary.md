# S24 40x20 50-Sample Default Candidate Validation

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 50 --val-samples 0 --test-samples 0 --grid-x 40 --grid-y 20 --output-dir experiments\dual_network\S24_40x20_50sample_default_validation\data --seed 1024
```

Generated train data:

- `experiments/dual_network/S24_40x20_50sample_default_validation/data/training_data_train.npz`

## Runner Configuration

Both runs used:

- `sample_indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49`
- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`

Compared runs:

- `baseline`: `lambda_mask_bce_prior=0.0`, `mask_prior_temperature=50.0`
- `temp25_lambda1`: `lambda_mask_bce_prior=1.0`, `mask_prior_temperature=25.0`

## Fifty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.426606e-01 | 2.353600e+02 | 2.503184e+05 | 2.932973e+02 |
| temp25_lambda1 | 9.203000e-01 | 3.340000e+01 | 3.598542e+04 | 1.498937e+02 |

## Per-Sample IoU

| sample | baseline | temp25_lambda1 |
| ---: | ---: | ---: |
| 0 | 1.036270e-01 | 9.500000e-01 |
| 1 | 9.282700e-02 | 1.000000e+00 |
| 2 | 8.921900e-02 | 8.461540e-01 |
| 3 | 1.021130e-01 | 7.878790e-01 |
| 4 | 1.339290e-01 | 1.000000e+00 |
| 5 | 7.373300e-02 | 9.000000e-01 |
| 6 | 2.758600e-02 | 9.600000e-01 |
| 7 | 1.218270e-01 | 1.000000e+00 |
| 8 | 1.036040e-01 | 1.000000e+00 |
| 9 | 1.787440e-01 | 9.729730e-01 |
| 10 | 1.335500e-01 | 9.523810e-01 |
| 11 | 8.438800e-02 | 6.800000e-01 |
| 12 | 3.207550e-01 | 9.298250e-01 |
| 13 | 1.084340e-01 | 1.000000e+00 |
| 14 | 3.194440e-01 | 1.000000e+00 |
| 15 | 7.491900e-02 | 7.500000e-01 |
| 16 | 8.270700e-02 | 8.400000e-01 |
| 17 | 1.835750e-01 | 9.736840e-01 |
| 18 | 3.939390e-01 | 1.000000e+00 |
| 19 | 1.307850e-01 | 1.000000e+00 |
| 20 | 8.058600e-02 | 8.800000e-01 |
| 21 | 1.224490e-01 | 9.666670e-01 |
| 22 | 8.490600e-02 | 9.473680e-01 |
| 23 | 1.272730e-01 | 1.000000e+00 |
| 24 | 1.165050e-01 | 1.000000e+00 |
| 25 | 5.263200e-02 | 8.125000e-01 |
| 26 | 8.560300e-02 | 9.583330e-01 |
| 27 | 9.701500e-02 | 7.812500e-01 |
| 28 | 3.101270e-01 | 1.000000e+00 |
| 29 | 2.038220e-01 | 9.411760e-01 |
| 30 | 2.022470e-01 | 5.714290e-01 |
| 31 | 2.380950e-01 | 9.183670e-01 |
| 32 | 1.542290e-01 | 9.677420e-01 |
| 33 | 2.531650e-01 | 1.000000e+00 |
| 34 | 1.229170e-01 | 1.000000e+00 |
| 35 | 1.704550e-01 | 9.666670e-01 |
| 36 | 7.971000e-02 | 9.200000e-01 |
| 37 | 1.032030e-01 | 9.354840e-01 |
| 38 | 1.121500e-01 | 7.931030e-01 |
| 39 | 8.910900e-02 | 9.642860e-01 |
| 40 | 9.729700e-02 | 1.000000e+00 |
| 41 | 1.626790e-01 | 9.189190e-01 |
| 42 | 2.943930e-01 | 1.000000e+00 |
| 43 | 1.504420e-01 | 8.888890e-01 |
| 44 | 5.471100e-02 | 5.897440e-01 |
| 45 | 1.012150e-01 | 9.600000e-01 |
| 46 | 1.529410e-01 | 9.285710e-01 |
| 47 | 1.373630e-01 | 1.000000e+00 |
| 48 | 1.494250e-01 | 8.928570e-01 |
| 49 | 1.666670e-01 | 9.687500e-01 |

## Stability Check

- Both 50-sample runner jobs completed successfully.
- No NaN/inf was observed in the aggregated metrics.
- `metrics.csv` was generated for both runs.
- No model checkpoints or weights were generated.
- `temp25_lambda1` improves IoU on all 50 samples.
- Three `temp25_lambda1` samples have IoU below `0.7`: sample 11 (`0.680000`), sample 30 (`0.571429`), and sample 44 (`0.589744`). These are weak samples but not run failures.

## Current Judgment

- `temp25_lambda1` is stable on this fresh 40x20 / 50-sample validation set and is suitable as the current 40x20 IoU-priority default candidate.
- It improves average IoU from `1.426606e-01` to `9.203000e-01`.
- It reduces average predicted defect area from `235.36` to `33.40`.
- It improves `mu_mse` and `mu_mae` compared with baseline, although S23 showed `temp50_lambda3` remains preferable when continuous `mu` error is the primary target.
- This remains a semi-supervised / diagnostic upper-bound result because the BCE and mask priors use `mu_label < 500`; it is not proof of unsupervised weak-form inversion success.
