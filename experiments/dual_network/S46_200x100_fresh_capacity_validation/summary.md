# S46 Fresh 200x100 Capacity Validation

## Data Generation

Command:

```text
C:/Users/19166/anaconda3/envs/pinn_mfl/python.exe data_generator_v2.py --train-samples 50 --val-samples 0 --test-samples 0 --grid-x 200 --grid-y 100 --output-dir experiments/dual_network/S46_200x100_fresh_capacity_validation/data --seed 1046
```

Train NPZ: `experiments/dual_network/S46_200x100_fresh_capacity_validation/data/training_data_train.npz`.

This run uses a fresh 200x100 / 50-sample dataset and does not reuse S42 sample outputs.

## Compared Configurations

Shared setup: `area3_bce7`, `outer_steps=60`, `phi_steps=30`, `mu_steps=30`, `test_radius=5.0`, `center_mode=three`, `lambda_area_prior=3.0`, `lambda_mask_prior=1.0`, `lambda_mask_bce_prior=7.0`, `area_prior_temperature=50.0`, `mask_prior_temperature=25.0`.

- `cap32_area3_bce7`: `hidden_dim=32`, `num_layers=2`.
- `cap128_area3_bce7`: `hidden_dim=128`, `num_layers=4`.

## Aggregate Metrics

| config | hidden_dim | num_layers | avg IoU | median IoU | avg area pred | avg mu_mse | avg mu_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cap32_area3_bce7` | 32 | 2 | 8.252710e-01 | 8.892318e-01 | 1.071520e+03 | 5.076185e+04 | 1.663466e+02 |
| `cap128_area3_bce7` | 128 | 4 | 8.628616e-01 | 9.016328e-01 | 8.390600e+02 | 4.923852e+04 | 1.764913e+02 |

## Per-Sample defect_iou Comparison

| sample | cap32 IoU | cap128 IoU | delta 128-32 |
| --- | --- | --- | --- |
| 0 | 8.807734e-01 | 7.899807e-01 | -9.079266e-02 |
| 1 | 8.647959e-01 | 8.404256e-01 | -2.437037e-02 |
| 2 | 8.695652e-01 | 8.694853e-01 | -7.987022e-05 |
| 3 | 9.692132e-01 | 9.324009e-01 | -3.681231e-02 |
| 4 | 3.074147e-01 | 9.439089e-01 | 6.364942e-01 |
| 5 | 8.714653e-01 | 9.006536e-01 | 2.918828e-02 |
| 6 | 9.354446e-01 | 7.599558e-01 | -1.754888e-01 |
| 7 | 9.325069e-01 | 7.809308e-01 | -1.515761e-01 |
| 8 | 8.936782e-01 | 9.865068e-01 | 9.282857e-02 |
| 9 | 9.933628e-01 | 9.512736e-01 | -4.208928e-02 |
| 10 | 1.986225e-01 | 9.669421e-01 | 7.683197e-01 |
| 11 | 9.021134e-01 | 7.026801e-01 | -1.994334e-01 |
| 12 | 9.560440e-01 | 9.561510e-01 | 1.070499e-04 |
| 13 | 9.055259e-01 | 9.494382e-01 | 4.391235e-02 |
| 14 | 9.352332e-01 | 8.427230e-01 | -9.251016e-02 |
| 15 | 9.724719e-01 | 9.563777e-01 | -1.609421e-02 |
| 16 | 8.807189e-01 | 9.026087e-01 | 2.188975e-02 |
| 17 | 4.700179e-01 | 8.701538e-01 | 4.001359e-01 |
| 18 | 8.678304e-01 | 8.450363e-01 | -2.279407e-02 |
| 19 | 5.350186e-01 | 9.506511e-01 | 4.156325e-01 |
| 20 | 7.178571e-01 | 6.885554e-01 | -2.930176e-02 |
| 21 | 7.502449e-01 | 9.047059e-01 | 1.544610e-01 |
| 22 | 8.055152e-01 | 3.106383e-01 | -4.948769e-01 |
| 23 | 9.256410e-01 | 9.802371e-01 | 5.459613e-02 |
| 24 | 8.451579e-01 | 9.731935e-01 | 1.280355e-01 |
| 25 | 9.617084e-01 | 9.140401e-01 | -4.766828e-02 |
| 26 | 6.225827e-01 | 9.006568e-01 | 2.780741e-01 |
| 27 | 8.879310e-01 | 8.044329e-01 | -8.349818e-02 |
| 28 | 9.895470e-01 | 9.233450e-01 | -6.620204e-02 |
| 29 | 9.694149e-01 | 9.471650e-01 | -2.224994e-02 |
| 30 | 8.260869e-01 | 9.557522e-01 | 1.296653e-01 |
| 31 | 9.619834e-01 | 8.804185e-01 | -8.156490e-02 |
| 32 | 8.549223e-01 | 7.505423e-01 | -1.043800e-01 |
| 33 | 9.738480e-01 | 8.938053e-01 | -8.004272e-02 |
| 34 | 9.644352e-01 | 9.029536e-01 | -6.148160e-02 |
| 35 | 8.968421e-01 | 8.034557e-01 | -9.338641e-02 |
| 36 | 9.841270e-01 | 9.664694e-01 | -1.765758e-02 |
| 37 | 7.358491e-01 | 5.832084e-01 | -1.526407e-01 |
| 38 | 2.437149e-01 | 9.662274e-01 | 7.225124e-01 |
| 39 | 5.739561e-01 | 9.210823e-01 | 3.471262e-01 |
| 40 | 9.974425e-01 | 9.961637e-01 | -1.278818e-03 |
| 41 | 8.056338e-01 | 5.552239e-01 | -2.504099e-01 |
| 42 | 8.905326e-01 | 9.437690e-01 | 5.323642e-02 |
| 43 | 9.063261e-01 | 8.505214e-01 | -5.580461e-02 |
| 44 | 8.451883e-01 | 8.423326e-01 | -2.855659e-03 |
| 45 | 9.557166e-01 | 6.096537e-01 | -3.460629e-01 |
| 46 | 9.938144e-01 | 9.775510e-01 | -1.626337e-02 |
| 47 | 7.987013e-01 | 8.655570e-01 | 6.685567e-02 |
| 48 | 9.909639e-01 | 8.930100e-01 | -9.795392e-02 |
| 49 | 4.460493e-01 | 9.401295e-01 | 4.940802e-01 |

## Stability and Failure Samples

- `cap128_area3_bce7` has higher average IoU than `cap32_area3_bce7`: `8.628616e-01` versus `8.252710e-01`.
- `cap128_area3_bce7` also lowers average predicted defect area and average `mu_mse`.
- `cap128_area3_bce7` has higher average `mu_mae` than `cap32_area3_bce7`: `1.764913e+02` versus `1.663466e+02`.
- Per-sample IoU comparison: `cap128` improves 19/50 samples, regresses 31/50 samples, and ties 0/50 samples.
- Obvious failure samples using IoU < 0.5: `cap32` = [4, 10, 17, 38, 49]; `cap128` = [22].

Largest `cap128` improvements over `cap32`:

| sample | cap32 IoU | cap128 IoU | delta |
| --- | --- | --- | --- |
| 10 | 1.986225e-01 | 9.669421e-01 | 7.683197e-01 |
| 38 | 2.437149e-01 | 9.662274e-01 | 7.225124e-01 |
| 4 | 3.074147e-01 | 9.439089e-01 | 6.364942e-01 |
| 49 | 4.460493e-01 | 9.401295e-01 | 4.940802e-01 |
| 19 | 5.350186e-01 | 9.506511e-01 | 4.156325e-01 |
| 17 | 4.700179e-01 | 8.701538e-01 | 4.001359e-01 |
| 39 | 5.739561e-01 | 9.210823e-01 | 3.471262e-01 |
| 26 | 6.225827e-01 | 9.006568e-01 | 2.780741e-01 |
| 21 | 7.502449e-01 | 9.047059e-01 | 1.544610e-01 |
| 30 | 8.260869e-01 | 9.557522e-01 | 1.296653e-01 |

Largest `cap128` regressions versus `cap32`:

| sample | cap32 IoU | cap128 IoU | delta |
| --- | --- | --- | --- |
| 22 | 8.055152e-01 | 3.106383e-01 | -4.948769e-01 |
| 45 | 9.557166e-01 | 6.096537e-01 | -3.460629e-01 |
| 41 | 8.056338e-01 | 5.552239e-01 | -2.504099e-01 |
| 11 | 9.021134e-01 | 7.026801e-01 | -1.994334e-01 |
| 6 | 9.354446e-01 | 7.599558e-01 | -1.754888e-01 |
| 37 | 7.358491e-01 | 5.832084e-01 | -1.526407e-01 |
| 7 | 9.325069e-01 | 7.809308e-01 | -1.515761e-01 |
| 32 | 8.549223e-01 | 7.505423e-01 | -1.043800e-01 |
| 48 | 9.909639e-01 | 8.930100e-01 | -9.795392e-02 |
| 35 | 8.968421e-01 | 8.034557e-01 | -9.338641e-02 |

## Current Conclusion

- `cap128_area3_bce7` is stronger on average and substantially fixes several severe `cap32` failures.
- It is not uniformly stable sample-by-sample: `cap128` regresses on more samples than it improves, even though the average IoU is higher because the improvements on weak samples are large.
- `cap128_area3_bce7` should be treated as the current 200x100 capacity candidate for further validation, but this S46 result alone is not enough to call it an unconditional default replacement for every sample.
- If the goal is average IoU and fewer severe failures, `cap128` is preferred. If the goal is per-sample stability, more validation or a hybrid/stability criterion is needed.
- Boundary unchanged: BCE remains a semi-supervised diagnostic upper bound, not unsupervised weak-form success.

## Next-Step Recommendation

- Run a follow-up diagnostics pass on the `cap128` regression samples, especially 22, 45, 41, 11, 6.
- Consider validating `cap128_area3_bce7` on 100 fresh samples before making it the formal 200x100 capacity default.
- Do not treat this as unsupervised weak-form success; it remains a semi-supervised BCE mask-prior result.
