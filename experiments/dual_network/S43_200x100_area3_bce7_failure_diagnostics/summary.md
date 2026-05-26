# S43 200x100 area3_bce7 Failure Diagnostics

## 1. S42 Overall Conclusion

Source results: `experiments/dual_network/S42_200x100_fresh_area3_bce7_validation/`.

- `baseline`: avg `defect_iou=7.703941e-02`, avg `defect_area_pred=1.146581e+04`, avg `mu_mse=5.252954e+05`, avg `mu_mae=5.387451e+02`.
- `temp25_lambda5_outer60`: avg `defect_iou=7.398354e-01`, avg `defect_area_pred=1.429090e+03`, avg `mu_mse=6.927535e+04`, avg `mu_mae=1.894512e+02`.
- `area3_bce7`: avg `defect_iou=8.047192e-01`, avg `defect_area_pred=1.213090e+03`, avg `mu_mse=5.839230e+04`, avg `mu_mae=1.823810e+02`.
- Compared with `temp25_lambda5_outer60`, `area3_bce7` changes avg `defect_area_pred` by `-2.160000e+02`, avg `mu_mse` by `-1.088304e+04`, and avg `mu_mae` by `-7.070209e+00`.

`area3_bce7` is the best S42 config on average IoU, predicted defect area, `mu_mse`, and `mu_mae`, so it should replace `temp25_lambda5_outer60` as the current 200x100 semi-supervised BCE diagnostic candidate.

## 2. Successful Samples

Highest-IoU `area3_bce7` samples:

| sample | IoU | area pred/label | centroid offset |
| --- | --- | --- | --- |
| 47 | 9.962871e-01 | 807/806 | 0.0034 |
| 25 | 9.961796e-01 | 1046/1044 | 0.0016 |
| 85 | 9.940565e-01 | 671/671 | 0.0014 |
| 17 | 9.929757e-01 | 1561/1560 | 0.0064 |
| 15 | 9.919857e-01 | 1118/1119 | 0.0108 |

## 3. Weak Samples

Lowest-IoU `area3_bce7` samples and metric-derived failure tags:

| sample | IoU | area pred/label | centroid offset | failure type |
| --- | --- | --- | --- | --- |
| 28 | 2.323879e-01 | 5408/1327 | 2.7081 | low IoU; area too large; centroid shift; local false positive |
| 5 | 2.542005e-01 | 1649/665 | 3.3401 | low IoU; area too large; centroid shift; local false positive |
| 71 | 2.792664e-01 | 6156/1865 | 3.4480 | low IoU; area too large; centroid shift; local false positive |
| 23 | 2.890733e-01 | 5038/1486 | 3.4061 | low IoU; area too large; centroid shift; local false positive |
| 11 | 2.971103e-01 | 4861/1513 | 2.5944 | low IoU; area too large; centroid shift; local false positive |
| 21 | 3.065945e-01 | 1802/853 | 2.6476 | low IoU; area too large; centroid shift; local false positive |
| 8 | 3.944150e-01 | 4257/1785 | 2.2092 | low IoU; area too large; centroid shift; local false positive |
| 72 | 4.186533e-01 | 2276/1116 | 1.9933 | low IoU; area too large; centroid shift; local false positive |
| 99 | 4.314995e-01 | 3686/1622 | 2.0763 | low IoU; area too large; centroid shift; local false positive |
| 83 | 4.468653e-01 | 2200/1054 | 0.9635 | low IoU; area too large; centroid shift; local false positive |
| 82 | 4.622018e-01 | 3368/1719 | 1.7234 | low IoU; area too large; centroid shift; local false positive |
| 36 | 4.656413e-01 | 4002/1906 | 2.4655 | low IoU; area too large; centroid shift; local false positive |
| 27 | 4.885984e-01 | 3555/1798 | 2.1334 | low IoU; area too large; centroid shift; local false positive |
| 52 | 4.933628e-01 | 1720/980 | 1.0499 | low IoU; area too large; centroid shift; local false positive |
| 7 | 5.025621e-01 | 2487/1325 | 2.7190 | area too large; centroid shift; local false positive |

Additional diagnostic rankings:

- Largest area-error samples: 71, 28, 23, 11, 8, 36, 99, 27, 82, 7.
- Largest centroid-offset samples: 71, 23, 5, 7, 28, 21, 11, 36, 8, 27.
- Largest baseline-to-`area3_bce7` improvements: 47, 20, 31, 29, 89, 68, 46, 73, 0, 15.
- Smallest baseline-to-`area3_bce7` improvements: 71, 28, 11, 23, 5, 21, 8, 82, 99, 72.

## 4. Comparison With temp25_lambda5_outer60

Largest `temp25_lambda5_outer60` to `area3_bce7` IoU improvements:

| sample | area3 IoU | temp25 IoU | delta |
| --- | --- | --- | --- |
| 39 | 8.504348e-01 | 1.964286e-01 | 6.540062e-01 |
| 81 | 7.571428e-01 | 1.144662e-01 | 6.426767e-01 |
| 22 | 8.587500e-01 | 2.401884e-01 | 6.185616e-01 |
| 87 | 8.074656e-01 | 1.918376e-01 | 6.156280e-01 |
| 4 | 7.333333e-01 | 1.515695e-01 | 5.817638e-01 |
| 61 | 8.190104e-01 | 3.568649e-01 | 4.621455e-01 |
| 57 | 9.146342e-01 | 4.655259e-01 | 4.491082e-01 |
| 54 | 6.747405e-01 | 2.277637e-01 | 4.469768e-01 |
| 91 | 9.638554e-01 | 5.323637e-01 | 4.314918e-01 |
| 30 | 7.953488e-01 | 4.888203e-01 | 3.065285e-01 |

Largest `area3_bce7` regressions versus `temp25_lambda5_outer60`:

| sample | area3 IoU | temp25 IoU | delta |
| --- | --- | --- | --- |
| 82 | 4.622018e-01 | 9.087912e-01 | -4.465894e-01 |
| 5 | 2.542005e-01 | 6.046002e-01 | -3.503996e-01 |
| 64 | 5.376934e-01 | 8.858132e-01 | -3.481197e-01 |
| 40 | 7.191217e-01 | 9.618363e-01 | -2.427146e-01 |
| 23 | 2.890733e-01 | 4.028777e-01 | -1.138044e-01 |
| 45 | 8.304297e-01 | 9.258777e-01 | -9.544802e-02 |
| 72 | 4.186533e-01 | 4.843517e-01 | -6.569844e-02 |
| 96 | 7.090301e-01 | 7.647059e-01 | -5.567580e-02 |
| 28 | 2.323879e-01 | 2.807173e-01 | -4.832935e-02 |
| 79 | 5.729167e-01 | 6.097763e-01 | -3.685963e-02 |

The trade-off is favorable on average: `area3_bce7` improves IoU for 70/100 samples and reduces the average predicted area, `mu_mse`, and `mu_mae`. The remaining regressions are localized and mostly overlap with the weak-sample regime where shape detail, local false positives, centroid offset, or narrow/boundary geometry still dominate.

## 5. Current Judgment

- `area3_bce7` is the current 200x100 default candidate.
- The main residual problems are false positives, area overprediction, centroid shifts, and local shape-detail errors.
- This result remains a semi-supervised BCE mask-prior upper bound, not unsupervised weak-form success.

## 6. Next-Step Recommendation

- Do not continue blind area/BCE weight sweeps.
- If improving 200x100 further, prioritize weak-sample shape analysis and post-processing or structural false-positive control.
- Consider recording `area3_bce7` in `DUAL_NETWORK_RESULTS_REPORT` as the current 200x100 default.

## Artifacts

- `diagnostics_rankings.csv` contains the full per-sample numeric ranking table.
- `figures/` contains 25 representative `area3_bce7` `mu_pred_vs_label` PNGs copied from S42.
