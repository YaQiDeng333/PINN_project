# S72 real COMSOL pilot conditional sanity probe

## 目的

S72 使用 S71 converted NPZ 做极小规模 conditional supervised sanity training，确认真实 COMSOL multi-height Bz pilot 数据可以被 `train_conditional_dual.py` 正常读取、训练并输出 metrics。

## 数据

- converted NPZ：`experiments/dual_network/S71_comsol_pilot_ingest/converted/comsol_multiheight_pilot.npz`
- samples：`0,1,2,3,4`
- `signals shape = [5, 3, 200]`
- flattened signal length：`600`
- target fields：`mu_maps`, `masks`

## 训练配置

```powershell
python train_conditional_dual.py --npz-path experiments/dual_network/S71_comsol_pilot_ingest/converted/comsol_multiheight_pilot.npz --output-dir experiments/dual_network/S72_comsol_pilot_conditional_sanity_probe/train5_multichannel --sample-indices 0,1,2,3,4 --steps 300 --lr 1e-3 --hidden-dim 64 --num-layers 3 --latent-dim 32 --lambda-mask-bce 1.0 --lambda-mask-dice 1.0 --lambda-mu-mse 0.0 --mask-temperature 50.0 --signal-normalization per_sample_zscore --signal-feature-mode raw --mask-head-mode mu_threshold
```

## 输出

- `train5_multichannel/metrics.csv`
- `train5_multichannel/run_summary.md`

未保存模型权重、checkpoint、`.npy` 或图片。

## 结果

final average metrics：
- `avg defect_iou = 4.767923e-01`
- `avg defect_area_pred = 4.019000e+03`
- `avg mu_mse = 9.244401e+04`
- `avg mu_mae = 2.460563e+02`

`metrics.csv` 共 5 行，所有 metrics 均为 finite。

## 当前边界

- 这是 sanity probe，不是正式训练。
- 样本数只有 5。
- 第一批 pilot 固定仿体，只改动磁性参数。
- 因此 S72 不代表泛化性能，也不代表 conditional model 已能学习几何变化。
- S72 只说明真实 COMSOL pilot converted NPZ 可以进入 conditional runner 并完成最小训练闭环。

## 下一步

S73 应准备下一批几何变化 COMSOL 数据请求，使后续数据真正覆盖 defect shape / location / size 变化。
