# S51 conditional supervised small-data probe

## 数据来源

数据生成命令：

```powershell
python data_generator_v2.py --train-samples 20 --val-samples 0 --test-samples 0 --grid-x 20 --grid-y 10 --output-dir experiments/dual_network/S51_conditional_supervised_small_data_probe/data --seed 1051
```

使用的 train `.npz`：

`experiments/dual_network/S51_conditional_supervised_small_data_probe/data/training_data_train.npz`

样本数与分辨率：

- train samples: 20
- grid_x: 20
- grid_y: 10

## 训练配置

运行配置：

```powershell
python train_conditional_dual.py --npz-path experiments/dual_network/S51_conditional_supervised_small_data_probe/data/training_data_train.npz --output-dir experiments/dual_network/S51_conditional_supervised_small_data_probe/train20_bce_dice --sample-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --steps 300 --lr 1e-3 --hidden-dim 64 --num-layers 3 --latent-dim 32 --lambda-mask-bce 1.0 --lambda-mask-dice 1.0 --lambda-mu-mse 0.0 --mask-temperature 50.0
```

说明：

- 使用 `ConditionalDualNet`。
- 输入为 `signals + coords`。
- 训练监督使用 `mask_label`。
- 未接入 weak-form / physics loss。
- 未保存模型权重、checkpoint、`.npy` 或图片。

## Final train metrics

`train20_bce_dice/metrics.csv` 包含 20 个训练样本，所有 metrics 均为 finite。

- final avg `defect_iou = 5.228869e-01`
- final avg `defect_area_pred = 5.050000e+00`
- final avg `mu_mse = 3.178682e+04`
- final avg `mu_mae = 1.286587e+02`
- min `defect_iou = 0.000000e+00`
- max `defect_iou = 1.000000e+00`

## 当前判断

1. 300 step 后 train-set 平均 IoU 达到约 `0.523`，说明 conditional supervised runner 的训练闭环可用，并且 signal-conditioned model 已经能从真实小型 `.npz` 数据中学习到部分 mask 信号。
2. 结果还不算强，且存在 IoU 为 0 的训练样本，说明当前 encoder / conditional MLP / loss 设置仍需要调整。
3. S51 只是 train-set smoke/probe，不代表泛化性能。
4. 当前结果不能与主线 baseline 直接比较，因为还没有 train/val/test conditional protocol。

## 下一步建议

S52 或后续阶段应建立 train/val/test conditional runner：

- 保留 `signals + coords` 推理接口；
- 增加 validation split；
- 记录 train / val metrics；
- 再考虑 weak-form / physics loss 的接入；
- 避免把 S51 train-set probe 写成泛化结论。
