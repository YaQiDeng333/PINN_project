# S74 COMSOL geometry-variation data ingest

## 目的

S74 将 COMSOL 侧已经生成的真实 geometry-variation multi-height Bz train / val / test 数据接入 dual-network 支线，验证 S67 converter、S66 validator、conditional data utils 和 conditional model forward 链路是否能处理真实多高度 Bz 数据。

## 数据来源

原始数据来自工作区根目录的 `comsol_geometry_variation_exports/`：

- `train/signals_multiheight.csv`
- `train/targets.npz`
- `train/README.md`
- `val/signals_multiheight.csv`
- `val/targets.npz`
- `val/README.md`
- `test/signals_multiheight.csv`
- `test/targets.npz`
- `test/README.md`

S74 将这些文件复制到：

- `experiments/dual_network/S74_comsol_geometry_data_ingest/raw/train/`
- `experiments/dual_network/S74_comsol_geometry_data_ingest/raw/val/`
- `experiments/dual_network/S74_comsol_geometry_data_ingest/raw/test/`

## 转换输出

S67 converter 输出：

- `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`
- `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/val_comsol_multiheight.npz`
- `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/test_comsol_multiheight.npz`

converted NPZ shape：

- train signals: `[50, 3, 200]`
- val signals: `[10, 3, 200]`
- test signals: `[10, 3, 200]`

channels = 3，signal_len = 200，channels-first flatten 后 encoder input length = 600。

## 验证结果

`comsol_multiheight_npz_utils.py` validator 对 train / val / test 均通过：

- `num_channels = 3`
- `signal_len = 200`
- `has_mu_maps = True`
- `has_masks = True`
- `has_x_y = True`
- `signal_channel_names = ["Bz_liftoff_0p5", "Bz_liftoff_1p0", "Bz_liftoff_2p0"]`
- `lift_off_values = [0.5, 1.0, 2.0]`
- `field_components = ["Bz", "Bz", "Bz"]`

conditional data utils 检查：

- `infer_signal_len(dataset) = 600`
- `get_conditional_batch(train, [0,1,2])` 输出 `signals shape = [3,600]`
- `coords shape = [20000,2]`
- `ConditionalDualNet(signal_len=600, latent_dim=16, hidden_dim=32, num_layers=2)` forward 通过
- `mu` / `phi` 输出 shape 均为 `[3,20000,1]`

## 数据质量摘要

详见 `data_quality_summary.csv`。

| split | samples | label area mean | label area min | label area max | signals finite | targets finite |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| train | 50 | 2344.18 | 1074 | 4305 | yes | yes |
| val | 10 | 1988.90 | 1230 | 3098 | yes | yes |
| test | 10 | 2398.40 | 1421 | 4198 | yes | yes |

`defect_param_summary.md` 记录了实际变化的缺陷参数。当前已变化 center、axis / width、depth / shape parameter、`c_magn`、`mur_magn`、`Mr_magn_A_per_m` 和 `defect_mu`；`defect_type` 固定为 `ellipsoid`。

## 边界

这是第一批真实 COMSOL geometry-variation pilot 数据，样本量仍小；缺陷类型仍固定为 ellipsoid，未变化旋转角或边界不规则度。S74 只证明数据接入、转换、验证和 model forward 链路可用，不代表最终正式大规模训练结论。
