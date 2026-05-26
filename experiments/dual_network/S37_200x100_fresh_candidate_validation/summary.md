# S37 200x100 fresh candidate validation

## 1. 数据生成

命令：

```powershell
python data_generator_v2.py --train-samples 30 --val-samples 0 --test-samples 0 --grid-x 200 --grid-y 100 --output-dir experiments/dual_network/S37_200x100_fresh_candidate_validation/data --seed 1037
```

使用的 `.npz`：

- `experiments/dual_network/S37_200x100_fresh_candidate_validation/data/training_data_train.npz`

## 2. 实验配置

三组均使用：

- `sample_indices=0..29`
- `outer_steps=60`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`

差异配置：

| config | lambda_mask_bce_prior | mask_prior_temperature | 用途 |
| --- | ---: | ---: | --- |
| baseline | 0.0 | 50.0 | 无 BCE 对照 |
| temp25_lambda5_outer60 | 5.0 | 25.0 | S36 综合候选 |
| temp20_lambda3_outer60 | 3.0 | 20.0 | S36/S26 风格的 IoU 优先候选 |

## 3. 平均指标

| config | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 6.917387e-02 | 1.126520e+04 | 5.193798e+05 | 5.264594e+02 |
| temp25_lambda5_outer60 | 8.093492e-01 | 1.127800e+03 | 6.106250e+04 | 1.837940e+02 |
| temp20_lambda3_outer60 | 6.485412e-01 | 1.882233e+03 | 9.386268e+04 | 2.070304e+02 |

## 4. 每个 sample 的 defect_iou 对比

| sample | baseline | temp25_lambda5_outer60 | temp20_lambda3_outer60 |
| ---: | ---: | ---: | ---: |
| 0 | 7.102070e-02 | 9.753289e-01 | 9.703460e-01 |
| 1 | 1.330008e-01 | 7.540541e-01 | 8.266476e-01 |
| 2 | 7.480711e-02 | 8.836735e-01 | 3.418704e-01 |
| 3 | 3.339327e-02 | 9.926740e-01 | 9.963303e-01 |
| 4 | 7.488427e-02 | 8.921892e-01 | 9.699880e-01 |
| 5 | 8.704756e-02 | 4.131539e-01 | 2.883731e-01 |
| 6 | 4.485351e-02 | 7.147193e-01 | 7.243976e-01 |
| 7 | 5.537459e-02 | 8.592965e-01 | 5.518832e-01 |
| 8 | 5.418719e-02 | 7.961905e-01 | 7.687970e-01 |
| 9 | 1.210358e-01 | 8.661182e-01 | 6.203424e-01 |
| 10 | 3.300019e-02 | 8.994709e-01 | 9.162162e-01 |
| 11 | 1.530984e-01 | 9.794313e-01 | 4.497074e-01 |
| 12 | 5.591847e-02 | 9.044502e-01 | 9.045752e-01 |
| 13 | 5.407344e-02 | 3.158585e-01 | 1.080074e-01 |
| 14 | 1.115579e-01 | 2.307250e-01 | 3.914449e-01 |
| 15 | 8.080428e-02 | 3.155952e-01 | 2.871287e-01 |
| 16 | 7.187655e-02 | 9.807475e-01 | 9.434590e-01 |
| 17 | 1.140456e-01 | 9.869338e-01 | 9.395230e-01 |
| 18 | 5.332910e-02 | 8.599168e-01 | 1.073837e-01 |
| 19 | 6.070371e-02 | 9.483961e-01 | 1.611894e-01 |
| 20 | 6.376389e-02 | 9.469274e-01 | 9.537167e-01 |
| 21 | 6.093438e-02 | 9.184442e-01 | 8.970223e-01 |
| 22 | 4.727469e-02 | 8.704883e-01 | 8.644068e-01 |
| 23 | 5.189782e-02 | 8.715891e-01 | 7.176558e-02 |
| 24 | 6.588604e-02 | 9.596083e-01 | 9.330086e-01 |
| 25 | 5.504889e-02 | 7.100372e-01 | 3.440678e-01 |
| 26 | 4.176180e-02 | 6.465753e-01 | 3.434727e-01 |
| 27 | 5.812850e-02 | 9.926254e-01 | 9.926144e-01 |
| 28 | 4.338354e-02 | 9.644352e-01 | 9.686847e-01 |
| 29 | 4.912407e-02 | 8.308208e-01 | 8.198653e-01 |

## 5. 失败样本

- baseline：30/30 个样本 `defect_iou < 0.5`，仍表现为大面积低 `mu` 扩散。
- `temp25_lambda5_outer60`：4 个样本 `defect_iou < 0.5`，包括 sample 5、13、14、15。
- `temp20_lambda3_outer60`：11 个样本 `defect_iou < 0.5`，包括 sample 2、5、11、13、14、15、18、19、23、25、26。

## 6. 当前判断

- `temp25_lambda5_outer60` 在新的 200x100 / 30-sample 数据上平均 IoU 最高，同时 `defect_area_pred`、`mu_mse` 和 `mu_mae` 也最低。
- `temp20_lambda3_outer60` 没有在 S37 中复现为 IoU 最优，且弱样本数量更多。
- IoU 优先、连续 `mu` 误差优先和综合考虑三种口径下，本轮都应选择 `temp25_lambda5_outer60` 作为当前 200x100 默认候选。
- 该结果仍是半监督 / 诊断上界，因为 BCE mask prior 使用 `mu_label < 500`，不能表述为无监督 weak-form 反演成功。

