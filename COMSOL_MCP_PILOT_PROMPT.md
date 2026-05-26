# COMSOL_MCP_PILOT_PROMPT

## 任务目标

请生成 5-10 个真实 COMSOL-style multi-height Bz pilot 样本，用于后续 dual-network 支线验证真实多高度 Bz 数据的转换、校验和 conditional model 接入链路。

## 输出文件

请输出以下文件：

- `signals_multiheight.csv`
- `targets.npz`
- `README.md`

## signals CSV schema

`signals_multiheight.csv` 必须使用 long-table schema，并包含以下列：

- sample_index
- channel_index
- channel_name
- lift_off
- field_component
- x_index
- x
- value

每一行对应一个 `(sample_index, channel_index, x_index)` 的 Bz value。

## target NPZ schema

`targets.npz` 至少包含：

- `mu_maps` 或 `masks`；
- `x`；
- `y`；
- metadata 可选。

`mu_maps` 或 `masks` 的第一维必须等于 samples。

## 推荐参数

- samples = 5 到 10；
- grid_x = 200；
- grid_y = 100；
- probe x points = 200；
- lift_off_values = [0.5, 1.0, 2.0]；
- field_components = ["Bz", "Bz", "Bz"]。

建议 channel 设计：

- channel_index = 0，channel_name = `Bz_liftoff_0p5`，lift_off = 0.5，field_component = `Bz`；
- channel_index = 1，channel_name = `Bz_liftoff_1p0`，lift_off = 1.0，field_component = `Bz`；
- channel_index = 2，channel_name = `Bz_liftoff_2p0`，lift_off = 2.0，field_component = `Bz`。

## 验收标准

- CSV 行数 = samples * channels * signal_len；
- 每个 sample / channel 的 `x_index` 完整；
- target 第一维与 samples 一致；
- 可以用本支线 converter 转为 NPZ；
- 可以用 validator 通过。

## 回到 dual-network 支线后的命令

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv signals_multiheight.csv --target-npz targets.npz --output-npz comsol_multiheight_pilot.npz
```

```powershell
python comsol_multiheight_npz_utils.py --npz-path comsol_multiheight_pilot.npz
```

## 边界

- 不要生成大规模数据；
- 不要先做训练；
- 先验证数据导出和转换链路；
- 这一步只需要 pilot 数据，不需要声称模型性能。
