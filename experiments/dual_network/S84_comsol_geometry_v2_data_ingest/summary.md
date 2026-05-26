# S84 COMSOL geometry V2 data ingest

## 目的

S84 将 `comsol_geometry_variation_v2_exports/` 中的真实 COMSOL geometry V2 fallback 数据接入 dual-network 支线，复制 raw train / val / test 文件，使用 S67 converter 转成支线可读的 multi-channel NPZ，并验证 validator、conditional loader 和 model forward 链路。

## 数据来源

- 原始目录：`comsol_geometry_variation_v2_exports/`
- raw train：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/raw/train/`
- raw val：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/raw/val/`
- raw test：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/raw/test/`

每个 split 包含：

- `signals_multiheight.csv`
- `targets.npz`
- `README.md`
- `defect_params.csv`

## 转换输出

- train：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`

## schema 验证

S66 validator 通过：

- train signals shape：`[100, 3, 200]`
- val signals shape：`[20, 3, 200]`
- test signals shape：`[20, 3, 200]`
- channels：`3`
- signal_len：`200`
- flattened signal length：`600`
- target 字段包含 `mu_maps` 和 `masks`
- `x` length = `200`
- `y` length = `100`

conditional data utils 检查通过：

- `get_conditional_batch(train, [0,1,2])` 输出 `signals [3,600]`
- `infer_signal_len(train) = 600`
- `ConditionalDualNet(signal_len=600, latent_dim=16, hidden_dim=32, num_layers=2)` forward 通过
- `mu` / `phi` shape 均为 `[3,20000,1]`

## 数据质量摘要

详见 `data_quality_summary.csv`。

- train mean label area = `5.355851e-02`，min / max = `4.595000e-02` / `6.400000e-02`
- val mean label area = `5.383750e-02`，min / max = `4.755000e-02` / `6.325000e-02`
- test mean label area = `5.350000e-02`，min / max = `4.595000e-02` / `6.400000e-02`
- signals 无 NaN / Inf
- targets 无 NaN / Inf

## defect parameter 摘要

详见 `defect_param_summary.md`。

- `defect_type` 覆盖四类 multi_defect 组合：
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`
- `rotation_angle` 范围：`0` 到 `30` degree
- `boundary_irregularity_proxy` 覆盖：`near` / `medium` / `far`
- `defect_center_x/y/z`、`defect_axis_x/y/z` 和 `defect_depth_or_shape_param` 均有变化
- magnetic parameters 当前固定：`defect_mu=1.0`、`c_magn=0.0`、`mur_magn=1.0`、`Mr_magn_A_per_m=0.0`

## 当前边界

- 这是真实 COMSOL geometry V2 fallback 数据接入，不是最终正式大规模数据。
- 使用 fallback 规模：train = `100`，val = `20`，test = `20`。
- 包含 `rectangular_notch` / `rotated_rect` multi_defect 和 rotation variation。
- 不包含 `ellipsoid`，README 中已说明 source pack 不支持。
- `boundary_irregularity` 是由 component distance bin 派生的 proxy，不是真实 free-form roughness。
- magnetic parameters 当前固定，没有逐样本变化。

## 下一步

使用 S84 converted NPZ 运行 S85 conditional train / val / test probe，判断 V2 更丰富几何变化是否相对 V1 S75 显示更好的 held-out 泛化潜力。
