# S56 conditional signal ablation probe

## 数据来源

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.

## 配置

- Config: `big_bce_dice_signal_ablation`
- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`
- `--signal-ablation` enabled.

Signal ablation is forward-only after training. It does not change the training loss and does not save model weights, checkpoints, arrays, or images.

## Correct signal metrics

| split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---:|---:|---:|---:|
| train | 9.072941e-01 | 6.807000e+00 | 3.036859e+03 | 6.648404e+00 |
| val | 9.372990e-02 | 5.560000e+00 | 4.783744e+04 | 5.242300e+01 |
| test | 8.830051e-02 | 6.165000e+00 | 5.093861e+04 | 5.555969e+01 |

## Signal ablation IoU

| split | correct_signal IoU | zero_signal IoU | shuffled_signal IoU |
|---|---:|---:|---:|
| train | 9.072941e-01 | 1.577167e-02 | 6.966472e-02 |
| val | 9.372990e-02 | 1.992248e-02 | 6.911540e-02 |
| test | 8.830051e-02 | 2.013694e-02 | 5.805073e-02 |

## Correct vs ablated gaps

| split | correct - zero | correct - shuffled |
|---|---:|---:|
| train | 8.915224e-01 | 8.376294e-01 |
| val | 7.380742e-02 | 2.461450e-02 |
| test | 6.816357e-02 | 3.024978e-02 |

All correct / zero / shuffled metrics are finite.

## 当前判断

1. Train split: correct signals are far better than zero or shuffled signals, so the trained model does use `Bz signal` when fitting train samples.
2. Val/test split: correct signals remain better than zero/shuffled, but the absolute IoU is still low and the margin over shuffled signals is small.
3. This means the model is not purely coordinate-only, but the learned signal conditioning does not generalize strongly.
4. The remaining generalization failure is more likely tied to signal encoder / conditioning quality, normalization, representation, or loss design than to data quantity alone.

## 下一步建议

- Focus on signal conditioning before scaling resolution: signal normalization, stronger or structured `BzEncoder`, conditioning injection strategy, and validation-aware regularization.
- Do not compare against the main baseline yet.
- Keep the boundary unchanged: this is supervised conditional training with label-derived training losses, not unsupervised weak-form success.
