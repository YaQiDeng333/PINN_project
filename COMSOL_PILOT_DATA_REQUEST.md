# COMSOL_PILOT_DATA_REQUEST

## 1. 目标

生成第一批真实 COMSOL-style multi-height Bz pilot 数据，用于验证当前 conditional dual-network 是否能从真实多高度 Bz 中获得比单条 synthetic Bz 更强的泛化信息。

## 2. pilot 数据规模

第一版建议：

- samples = 5 到 10；
- grid_x = 200；
- grid_y = 100；
- probe x points = 200；
- channels = 3；
- lift_off_values = [0.5, 1.0, 2.0]，单位按 COMSOL 模型定义；
- field_components = ["Bz", "Bz", "Bz"]。

## 3. 每个样本需要导出的内容

### A. signals long CSV

必须列：

- sample_index
- channel_index
- channel_name
- lift_off
- field_component
- x_index
- x
- value

### B. target NPZ 或等价 target 文件

至少包含：

- mu_maps 或 masks；
- x；
- y；
- 可选 metadata。

## 4. 输出目录建议

建议 COMSOL 侧输出为：

```text
comsol_pilot_exports/
  signals_multiheight.csv
  targets.npz
  README.md
```

## 5. 与本支线的接入方式

后续回到本支线后运行：

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv signals_multiheight.csv --target-npz targets.npz --output-npz comsol_multiheight_pilot.npz
```

然后验证：

```powershell
python comsol_multiheight_npz_utils.py --npz-path comsol_multiheight_pilot.npz
```

## 6. 验收标准

- CSV 行数 = samples * channels * signal_len；
- 每个 sample / channel 的 x_index 完整；
- signals 转换后 shape = [samples, channels, signal_len]；
- mu_maps 或 masks 第一维 = samples；
- validator 通过；
- 不需要训练模型即可先完成接口验收。

## 7. 边界说明

- 这是 pilot 数据，不是正式训练集；
- 当前只建议 5-10 个样本；
- 不要一开始生成大规模 COMSOL 数据；
- 先验证导出字段和转换流程，再扩大样本数。
