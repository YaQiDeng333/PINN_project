# COMSOL_GEOMETRY_VARIATION_DATA_REQUEST

## 1. 目标

生成第二批真实 COMSOL-style multi-height Bz 数据，用于 conditional dual-network 训练和验证。与第一批 pilot 不同，本批必须变化缺陷几何，而不是只改磁性参数。

第一批 pilot 已经证明 `signals_multiheight.csv` + `targets.npz` 可以通过 S67 converter 转成支线可读 multi-channel NPZ，并进入 conditional runner。它的主要限制是固定仿体，因此不能判断 shape generalization。第二批数据需要让模型看到 Bz signal 与 defect shape / location / size 之间的变化关系。

## 2. 建议数据规模

第一版建议：
- train samples = 50 到 100；
- val samples = 10 到 20；
- test samples = 10 到 20；
- grid_x = 200；
- grid_y = 100；
- probe x points = 200；
- channels = 3；
- lift_off_values = [0.5, 1.0, 2.0]；
- field_components = ["Bz", "Bz", "Bz"]。

如果 COMSOL 运行成本过高，可以先做：
- train = 20；
- val = 5；
- test = 5。

## 3. 必须变化的缺陷参数

建议至少变化：
- defect center x；
- defect center y；
- defect radius / width；
- defect depth 或等效形状参数；
- defect permeability / mu；
- 可选：椭圆长短轴、旋转角、边界不规则度。

建议在 metadata 中记录每个样本的缺陷参数，便于后续排查模型是否只记住某一类几何或磁性范围。

## 4. 数据输出

每个 split 输出：

```text
train/
  signals_multiheight.csv
  targets.npz
  README.md

val/
  signals_multiheight.csv
  targets.npz
  README.md

test/
  signals_multiheight.csv
  targets.npz
  README.md
```

也可以输出到一个总目录，并用 `sample_index` / `split` metadata 区分。但第一版更推荐 split 目录，便于直接转换为 train / val / test NPZ。

## 5. signals CSV schema

保持 S67 schema：
- sample_index
- channel_index
- channel_name
- lift_off
- field_component
- x_index
- x
- value

要求：
- `sample_index` 从 0 开始，并在每个 split 内连续编号；
- `channel_index` 按 lift-off / component 顺序从 0 开始；
- 每个 sample / channel 都必须包含完整 `x_index = 0..signal_len-1`；
- `value` 不应包含 NaN 或 Inf。

## 6. targets NPZ schema

至少包含：
- `mu_maps` 或 `masks`；
- `x`；
- `y`；
- metadata，建议记录每个样本的 defect parameters。

推荐：
- `mu_maps shape = [num_samples, grid_y, grid_x]`
- `masks shape = [num_samples, grid_y, grid_x]`
- `x shape = [grid_x]`
- `y shape = [grid_y]`

如果能同时输出 `mu_maps` 和 `masks`，优先同时输出。

## 7. 验收标准

- 每个 split 的 CSV 行数正确；
- 每个 sample / channel 的 `x_index` 完整；
- target 第一维与样本数一致；
- S66 validator 通过；
- 无 NaN / Inf；
- metadata 中能追踪缺陷参数分布；
- `x` / `y` 与 target 网格一致；
- `signals` 转换后 shape 应为 `[num_samples, 3, 200]`。

## 8. 回到本支线后的下一步

用 S67 converter 分别转换 train / val / test：

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv train/signals_multiheight.csv --target-npz train/targets.npz --output-npz train_comsol_multiheight.npz

python convert_comsol_multiheight_csv_to_npz.py --signals-csv val/signals_multiheight.csv --target-npz val/targets.npz --output-npz val_comsol_multiheight.npz

python convert_comsol_multiheight_csv_to_npz.py --signals-csv test/signals_multiheight.csv --target-npz test/targets.npz --output-npz test_comsol_multiheight.npz
```

然后用 validator 分别检查：

```powershell
python comsol_multiheight_npz_utils.py --npz-path train_comsol_multiheight.npz
python comsol_multiheight_npz_utils.py --npz-path val_comsol_multiheight.npz
python comsol_multiheight_npz_utils.py --npz-path test_comsol_multiheight.npz
```

通过后再用 `train_conditional_dual.py` 做真实 COMSOL train / val / test probe。

## 9. 边界说明

- 这才是用于 conditional model 泛化测试的第一批有意义数据。
- 不建议一开始做太大。
- 先做字段正确、几何变化明确、样本量适中的版本。
- 如果几何变化范围过窄，仍然只能验证接口或局部插值能力，不能证明模型具备可靠 shape generalization。
