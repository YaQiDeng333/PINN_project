# S28 80x40 50-Sample Default Candidate Validation

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 50 --val-samples 0 --test-samples 0 --grid-x 80 --grid-y 40 --output-dir experiments\dual_network\S28_80x40_50sample_default_validation\data --seed 1028
```

Generated train data:

- `experiments/dual_network/S28_80x40_50sample_default_validation/data/training_data_train.npz`

The generated train file contains 50 samples with `grid_x=80` and `grid_y=40`.

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
- `temp25_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=25.0`

## Fifty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.051006e-01 | 1.242440e+03 | 3.443720e+05 | 3.814918e+02 |
| temp25_lambda3 | 8.925113e-01 | 1.328200e+02 | 4.572207e+04 | 1.832775e+02 |

## Per-Sample IoU

| sample | baseline | temp25_lambda3 |
| ---: | ---: | ---: |
| 0 | 7.932692e-02 | 9.294118e-01 |
| 1 | 1.802031e-01 | 9.171975e-01 |
| 2 | 8.422665e-02 | 1.000000e+00 |
| 3 | 1.515487e-01 | 9.109589e-01 |
| 4 | 1.336453e-01 | 9.891697e-01 |
| 5 | 8.603145e-02 | 8.807340e-01 |
| 6 | 6.718346e-02 | 8.557692e-01 |
| 7 | 6.646526e-02 | 7.590361e-01 |
| 8 | 1.277014e-01 | 9.848485e-01 |
| 9 | 1.214224e-01 | 8.538012e-01 |
| 10 | 6.907545e-02 | 8.450704e-01 |
| 11 | 9.427609e-02 | 9.130435e-01 |
| 12 | 9.128205e-02 | 8.476191e-01 |
| 13 | 1.026455e-01 | 8.037383e-01 |
| 14 | 1.153101e-01 | 9.500000e-01 |
| 15 | 9.292503e-02 | 8.210526e-01 |
| 16 | 1.029289e-01 | 9.680000e-01 |
| 17 | 2.081281e-01 | 8.983957e-01 |
| 18 | 6.547931e-02 | 9.841270e-01 |
| 19 | 8.658009e-02 | 8.733333e-01 |
| 20 | 6.072289e-02 | 9.603174e-01 |
| 21 | 1.105372e-01 | 7.518519e-01 |
| 22 | 9.458776e-02 | 9.735450e-01 |
| 23 | 1.961116e-01 | 9.746835e-01 |
| 24 | 1.362694e-01 | 9.886364e-01 |
| 25 | 7.658834e-02 | 8.076923e-01 |
| 26 | 5.828517e-02 | 9.593496e-01 |
| 27 | 9.753788e-02 | 8.318584e-01 |
| 28 | 1.756602e-01 | 9.216868e-01 |
| 29 | 1.117705e-01 | 1.000000e+00 |
| 30 | 1.086729e-01 | 9.809524e-01 |
| 31 | 1.342812e-01 | 8.617021e-01 |
| 32 | 4.906205e-02 | 8.279570e-01 |
| 33 | 9.111618e-02 | 8.202247e-01 |
| 34 | 1.054054e-01 | 9.125000e-01 |
| 35 | 1.942809e-01 | 9.674796e-01 |
| 36 | 1.672862e-01 | 9.729730e-01 |
| 37 | 7.190161e-02 | 9.125000e-01 |
| 38 | 7.926267e-02 | 9.019608e-01 |
| 39 | 2.201108e-01 | 9.756944e-01 |
| 40 | 8.010013e-02 | 8.142857e-01 |
| 41 | 8.925144e-02 | 7.068965e-01 |
| 42 | 1.332065e-01 | 8.815789e-01 |
| 43 | 9.430256e-02 | 9.895833e-01 |
| 44 | 5.217391e-02 | 8.956522e-01 |
| 45 | 5.772812e-02 | 6.488550e-01 |
| 46 | 1.336956e-01 | 9.618320e-01 |
| 47 | 4.613734e-02 | 1.000000e+00 |
| 48 | 5.030488e-02 | 6.966292e-01 |
| 49 | 5.229456e-02 | 7.413793e-01 |

## Stability Check

- Both runner jobs completed successfully.
- Each `metrics.csv` contains 50 rows.
- No NaN/inf was observed in aggregated metrics.
- No model checkpoints or weights were generated.
- `temp25_lambda3` improves IoU over baseline on all 50 samples.
- Two `temp25_lambda3` samples remain below IoU `0.7`: sample 45 and sample 48. Sample 41 is borderline at `0.706897`.

## Current Judgment

- `temp25_lambda3` is stable over baseline on this 50-sample `80x40` validation set.
- Average IoU improves from `1.051006e-01` to `8.925113e-01`.
- Average predicted defect area drops from `1242.44` to `132.82`.
- Average `mu_mse` drops from `3.443720e+05` to `4.572207e+04`, and average `mu_mae` drops from `3.814918e+02` to `1.832775e+02`.
- `temp25_lambda3` can be treated as the current `80x40` comprehensive default candidate.
- The result remains a semi-supervised / diagnostic upper-bound result because BCE and mask priors use `mu_label < 500`; it does not prove unsupervised weak-form inversion success.
