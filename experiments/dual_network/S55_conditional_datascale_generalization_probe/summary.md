# S55 conditional data-scale generalization probe

## 数据来源

- 数据生成命令：
  `python data_generator_v2.py --train-samples 1000 --val-samples 200 --test-samples 200 --grid-x 20 --grid-y 10 --output-dir experiments/dual_network/S55_conditional_datascale_generalization_probe/data --seed 1055`
- Train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- 样本数：train 1000，val 200，test 200。
- 分辨率：20x10。

## 配置

| config | steps | hidden_dim | num_layers | latent_dim | loss |
|---|---:|---:|---:|---:|---|
| `medium_bce_dice_datascale` | 5000 | 64 | 3 | 32 | BCE + Dice |
| `big_bce_dice_datascale` | 5000 | 128 | 4 | 64 | BCE + Dice |

两组都使用 `train_conditional_dual.py`，不接入 weak-form loss，不保存模型权重、checkpoint、`.npy` 或图片。

## S55 平均指标

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `medium_bce_dice_datascale` | train | 6.568019e-01 | 6.304000e+00 | 1.340666e+04 | 2.864577e+01 |
| `medium_bce_dice_datascale` | val | 9.390210e-02 | 7.505000e+00 | 5.022028e+04 | 6.476759e+01 |
| `medium_bce_dice_datascale` | test | 8.759288e-02 | 6.415000e+00 | 4.712943e+04 | 6.086760e+01 |
| `big_bce_dice_datascale` | train | 9.130737e-01 | 6.823000e+00 | 2.989310e+03 | 6.954080e+00 |
| `big_bce_dice_datascale` | val | 8.848209e-02 | 6.295000e+00 | 5.213359e+04 | 5.738768e+01 |
| `big_bce_dice_datascale` | test | 7.708859e-02 | 5.990000e+00 | 5.202747e+04 | 5.674250e+01 |

All train / val / test metrics are finite.

## 与 S54 对比

S54 used 200 train / 50 val / 50 test samples.

| config family | S54 val IoU | S55 val IoU | S54 test IoU | S55 test IoU |
|---|---:|---:|---:|---:|
| medium | 5.660991e-02 | 9.390210e-02 | 8.526794e-02 | 8.759288e-02 |
| big | 7.397455e-02 | 8.848209e-02 | 7.558780e-02 | 7.708859e-02 |

Increasing train samples improves held-out IoU modestly, especially medium val IoU, but the absolute val/test IoU remains low.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `medium_bce_dice_datascale` | 5.629998e-01 | 5.692090e-01 |
| `big_bce_dice_datascale` | 8.245916e-01 | 8.359851e-01 |

## 当前判断

1. Data scale helps but does not solve conditional generalization: val/test IoU improves only modestly from S54.
2. `big_bce_dice_datascale` still fits train strongly, but the train-to-held-out gap is large, so the bigger model remains overfit-prone.
3. `medium_bce_dice_datascale` has slightly better held-out IoU and smaller gaps, making it the better next diagnostic path for generalization, although its train IoU is lower.
4. S55 suggests S54's weak generalization is not mainly explained by having only 200 train samples; signal-conditioned architecture, signal representation, regularization, or loss design still need work.

## 下一步建议

- Do not move to higher resolution yet.
- Prefer medium-capacity generalization diagnostics next: validation-aware early stopping, regularization, augmentation, or signal encoder changes.
- Keep the boundary unchanged: this is supervised conditional training with label-derived losses during training, not unsupervised weak-form success and not a main-baseline replacement.
