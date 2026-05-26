# S71 real COMSOL pilot multi-height Bz ingest

## 目的

S71 将 `comsol_pilot_exports/` 中的第一批真实 COMSOL-style multi-height Bz pilot 数据接入 dual-network 支线，验证 S67 CSV -> NPZ converter、S66 validator、conditional data utils 和 conditional model forward 链路是否能处理真实 pilot 数据。

## 数据来源

- 原始目录：`comsol_pilot_exports/`
- 接入目录：`experiments/dual_network/S71_comsol_pilot_ingest/`

raw 文件：
- `raw/signals_multiheight.csv`
- `raw/targets.npz`
- `raw/README.md`

converted NPZ：
- `converted/comsol_multiheight_pilot.npz`

## 转换结果

使用命令：

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv experiments/dual_network/S71_comsol_pilot_ingest/raw/signals_multiheight.csv --target-npz experiments/dual_network/S71_comsol_pilot_ingest/raw/targets.npz --output-npz experiments/dual_network/S71_comsol_pilot_ingest/converted/comsol_multiheight_pilot.npz
```

转换后的关键 shape：
- `signals shape = [5, 3, 200]`
- `num_samples = 5`
- `num_channels = 3`
- `signal_len = 200`
- `target fields = mu_maps, masks`

## 验证结果

`comsol_multiheight_npz_utils.py` validator 通过，summary 显示：
- `num_samples = 5`
- `num_channels = 3`
- `signal_len = 200`
- `has_mu_maps = True`
- `has_masks = True`
- `has_x_y = True`
- `channel_names = ["Bz_liftoff_0p5", "Bz_liftoff_1p0", "Bz_liftoff_2p0"]`
- `lift_off_values = [0.5, 1.0, 2.0]`
- `field_components = ["Bz", "Bz", "Bz"]`

Conditional batch 检查通过：
- `get_conditional_batch(dataset, [0,1,2])` 输出 `signals shape = [3, 600]`
- `coords shape = [20000, 2]`
- `infer_signal_len(dataset) = 600`
- flatten 规则为 channels-first，即 `3 * 200 = 600`

Model forward 检查通过：
- `ConditionalDualNet(signal_len=600, latent_dim=16, hidden_dim=32, num_layers=2)`
- `mu shape = [3, 20000, 1]`
- `phi shape = [3, 20000, 1]`

## 当前边界

- 这是真实 COMSOL pilot 数据，不是 synthetic proxy。
- 当前样本数只有 5。
- 当前 pilot 固定仿体，只改动磁性参数。
- 因此 S71 主要用于接口验证，不代表形状泛化能力，也不代表正式训练集质量。

## 下一步

S72 使用 converted NPZ 做极小规模 conditional runner sanity probe，确认真实 COMSOL multi-height Bz pilot 可以进入 training runner 并输出 metrics。
