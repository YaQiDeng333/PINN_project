# S54 conditional train/val/test generalization probe

## 数据来源

- 数据生成命令：
  `python data_generator_v2.py --train-samples 200 --val-samples 50 --test-samples 50 --grid-x 20 --grid-y 10 --output-dir experiments/dual_network/S54_conditional_train_val_test_probe/data --seed 1054`
- Train NPZ: `experiments/dual_network/S54_conditional_train_val_test_probe/data/training_data_train.npz`
- Val NPZ: `experiments/dual_network/S54_conditional_train_val_test_probe/data/training_data_val.npz`
- Test NPZ: `experiments/dual_network/S54_conditional_train_val_test_probe/data/training_data_test.npz`
- 样本数：train 200，val 50，test 50。
- 分辨率：20x10。

## 配置

两组都使用 supervised conditional runner，不接入 weak-form loss，不保存模型权重、checkpoint、`.npy` 或图片。

| config | steps | hidden_dim | num_layers | latent_dim | loss |
|---|---:|---:|---:|---:|---|
| `medium_bce_dice` | 3000 | 64 | 3 | 32 | BCE + Dice |
| `big_bce_dice` | 3000 | 128 | 4 | 64 | BCE + Dice |

## 平均指标

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `medium_bce_dice` | train | 8.627310e-01 | 6.645000e+00 | 6.321843e+03 | 1.538956e+01 |
| `medium_bce_dice` | val | 5.660991e-02 | 8.300000e+00 | 5.970452e+04 | 6.911570e+01 |
| `medium_bce_dice` | test | 8.526794e-02 | 7.320000e+00 | 5.634353e+04 | 6.527965e+01 |
| `big_bce_dice` | train | 9.397538e-01 | 6.990000e+00 | 2.139510e+03 | 6.837023e+00 |
| `big_bce_dice` | val | 7.397455e-02 | 4.840000e+00 | 4.569675e+04 | 5.130799e+01 |
| `big_bce_dice` | test | 7.558780e-02 | 5.160000e+00 | 5.081966e+04 | 5.732467e+01 |

All train / val / test metrics are finite.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `medium_bce_dice` | 8.061211e-01 | 7.774631e-01 |
| `big_bce_dice` | 8.657792e-01 | 8.641660e-01 |

`big_bce_dice` has the best train IoU and best val IoU, while `medium_bce_dice` has the best test IoU by a small margin. Both settings show a large generalization gap: train IoU is high, but held-out val/test IoU remains low.

## 当前判断

1. S54 confirms that the conditional supervised runner can fit 20x10 train samples, especially with the larger model.
2. S54 does not show useful held-out generalization: val/test IoU is far below train IoU for both configs.
3. `big_bce_dice` is still the best diagnostic candidate because it has stronger train fitting, better val IoU, lower val area, and better val continuous `mu` errors, but it is not ready for main-baseline comparison.
4. The main bottleneck is no longer the basic training closure; it is conditional generalization from `signals` to unseen masks.

## 下一步建议

- Do not compare against the main baseline yet.
- Before moving to higher resolution, diagnose conditional generalization: validation-aware model selection, regularization, better signal encoder design, augmentation, or revised loss balance.
- Keep the inference boundary explicit: deployable conditional inference may use `signals + coords`, but not `mu_label` or `label_mask`.
