# S45 200x100 Model Capacity Probe

## Data Source

- NPZ: `experiments/dual_network/S42_200x100_fresh_area3_bce7_validation/data/training_data_train.npz`.
- Samples: `0-29` from the S42 fresh 200x100 dataset.
- Loss/prior setup: `area3_bce7` with `outer_steps=60`, `phi_steps=30`, `mu_steps=30`, `test_radius=5.0`, `center_mode=three`, `lambda_area_prior=3.0`, `lambda_mask_prior=1.0`, `lambda_mask_bce_prior=7.0`, `area_prior_temperature=50.0`, `mask_prior_temperature=25.0`.
- Code change: `train_dual_variational.py` now exposes `--hidden-dim` and `--num-layers`; defaults remain `32` and `2`.

## Capacity Configurations

- `cap_32x2`: `hidden_dim=32`, `num_layers=2`.
- `cap_64x3`: `hidden_dim=64`, `num_layers=3`.
- `cap_128x4`: `hidden_dim=128`, `num_layers=4`.

## Aggregate Metrics

| config | hidden_dim | num_layers | avg IoU | avg area pred | avg mu_mse | avg mu_mae |
| --- | --- | --- | --- | --- | --- | --- |
| `cap_32x2` | 32 | 2 | 7.581612e-01 | 1.612800e+03 | 7.100698e+04 | 2.023675e+02 |
| `cap_64x3` | 64 | 3 | 7.008694e-01 | 1.687733e+03 | 7.751988e+04 | 2.043629e+02 |
| `cap_128x4` | 128 | 4 | 8.680831e-01 | 1.179267e+03 | 4.888211e+04 | 1.657460e+02 |

## Per-Sample defect_iou

| sample | cap_32x2 IoU | cap_64x3 IoU | cap_128x4 IoU |
| --- | --- | --- | --- |
| 0 | 9.768934e-01 | 9.455696e-01 | 8.954327e-01 |
| 1 | 8.834532e-01 | 2.852405e-01 | 9.056047e-01 |
| 2 | 9.068182e-01 | 9.101251e-01 | 4.825378e-01 |
| 3 | 9.723577e-01 | 8.935331e-01 | 9.747763e-01 |
| 4 | 7.333333e-01 | 9.284916e-01 | 9.247191e-01 |
| 5 | 2.542005e-01 | 5.654923e-01 | 3.745724e-01 |
| 6 | 9.125096e-01 | 5.954488e-01 | 9.910496e-01 |
| 7 | 5.025621e-01 | 2.944444e-01 | 9.608412e-01 |
| 8 | 3.944150e-01 | 3.680134e-01 | 9.732533e-01 |
| 9 | 9.878296e-01 | 9.469929e-01 | 9.644620e-01 |
| 10 | 9.113756e-01 | 8.419722e-01 | 7.995470e-01 |
| 11 | 2.971103e-01 | 9.940594e-01 | 9.854977e-01 |
| 12 | 8.440914e-01 | 9.424920e-01 | 8.902196e-01 |
| 13 | 8.670012e-01 | 6.782511e-01 | 9.507133e-01 |
| 14 | 9.227324e-01 | 7.432314e-01 | 9.051819e-01 |
| 15 | 9.919857e-01 | 4.493116e-01 | 9.964285e-01 |
| 16 | 8.865324e-01 | 4.204322e-01 | 8.492462e-01 |
| 17 | 9.929757e-01 | 9.689480e-01 | 9.847812e-01 |
| 18 | 8.888889e-01 | 9.214092e-01 | 9.499323e-01 |
| 19 | 7.897991e-01 | 7.975460e-01 | 8.429054e-01 |
| 20 | 9.540918e-01 | 8.759259e-01 | 8.656430e-01 |
| 21 | 3.065945e-01 | 1.728369e-01 | 9.177928e-01 |
| 22 | 8.587500e-01 | 4.413793e-01 | 8.184893e-01 |
| 23 | 2.890733e-01 | 9.932976e-01 | 9.744280e-01 |
| 24 | 7.969201e-01 | 7.500000e-01 | 7.195122e-01 |
| 25 | 9.961796e-01 | 9.701214e-01 | 9.802817e-01 |
| 26 | 9.232000e-01 | 9.036335e-01 | 9.079365e-01 |
| 27 | 4.885984e-01 | 9.807056e-01 | 3.629707e-01 |
| 28 | 2.323879e-01 | 2.276850e-01 | 9.637037e-01 |
| 29 | 9.821747e-01 | 2.194919e-01 | 9.300341e-01 |

## Weak-Sample Response

Weak samples are the S42/S43 weak samples that fall inside the S45 sample subset: 5, 8, 11, 21, 23, 27, 28.

| config | weak sample avg IoU | weak samples |
| --- | --- | --- |
| `cap_32x2` | 3.231971e-01 | 5, 8, 11, 21, 23, 27, 28 |
| `cap_64x3` | 6.145843e-01 | 5, 8, 11, 21, 23, 27, 28 |
| `cap_128x4` | 7.931741e-01 | 5, 8, 11, 21, 23, 27, 28 |

The strongest weak-sample improvement is `cap_128x4`, with weak-sample avg IoU `7.931741e-01`.

## Best Configurations

- Best average IoU: `cap_128x4` with avg `defect_iou=8.680831e-01`.
- Best average `mu_mse`: `cap_128x4` with avg `mu_mse=4.888211e+04`.
- Best average `mu_mae`: `cap_128x4` with avg `mu_mae=1.657460e+02`.
- NaN/inf check: none found in metrics.

## Current Judgment

- Larger capacity clearly improves this 30-sample 200x100 probe: `cap_128x4` improves average IoU from `cap_32x2` `7.581612e-01` to `8.680831e-01`, while also reducing avg area, `mu_mse`, and `mu_mae`.
- This supports the hypothesis that current 200x100 detail recovery is at least partly limited by PhiNet/MuNet expression capacity.
- `cap_64x3` is not a stable intermediate here: it underperforms `cap_32x2` on average and has several severe sample regressions.
- `cap_128x4` still has sample-level regressions relative to `cap_32x2` on a few samples, so it should be treated as the next candidate to validate, not as a final default yet.
- Boundary unchanged: the BCE mask prior remains a semi-supervised diagnostic upper bound, not unsupervised weak-form success.

## Next-Step Recommendation

- Continue with `cap_128x4` as the capacity candidate on a fresh or larger 200x100 validation set.
- Do not adopt `cap_64x3` as a default based on this probe.
- If `cap_128x4` remains better on fresh data, update the 200x100 default candidate to include larger capacity; otherwise keep `cap_32x2` as the stable baseline and focus on loss/structure changes.
