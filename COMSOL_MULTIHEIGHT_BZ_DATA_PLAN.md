# COMSOL / Multi-height Bz Data Plan

## 1. 阶段动机

S56-S63 表明 conditional model 确实会使用 `Bz signal`：zero / shuffled signal ablation 会明显降低性能，说明模型不是只依赖坐标和数据分布。但 held-out val/test IoU 仍然很弱。

S57 的 normalization、S58 的 FiLM conditioning、S59 的 CNN encoder、S60 的 local signal feature、S61-S62 的 direct mask head / multi-task loss，以及 S63 的 `raw_abs_grad` 单条 Bz 派生特征，都没有稳定解决泛化问题。S65 的 synthetic multi-height proxy 也只能说明多通道接口能跑通，不能代表真实物理多高度信息。

因此下一阶段应增加输入物理信息，而不是继续只改单条 Bz 表示。COMSOL multi-height / multi-channel Bz 是后续数据方向：让模型看到多个 lift-off 高度、多个 field component 或多条 probe line 的磁场响应，以减少单条 Bz 的 signal-to-shape ambiguity。

## 2. 推荐数据 schema

必须字段：

- `signals`
- `mu_maps` 或 `masks`
- `x`
- `y`

`signals` 推荐形状：

```text
[num_samples, num_channels, signal_len]
```

推荐 metadata 字段：

- `source_type = "comsol_multiheight"`
- `signal_channels`
- `signal_channel_names`
- `lift_off_values`
- `field_components`
- `probe_line_y_values`
- `signal_flatten_order = "channels_first"`
- `geometry_units`
- `field_units`

## 3. 推荐通道设计

第一版推荐：

- `Bz` at low lift-off
- `Bz` at medium lift-off
- `Bz` at high lift-off

后续扩展：

- `Bx + Bz`
- multiple probe lines
- multiple lift-off heights
- COMSOL 2D / 3D metadata

## 4. 与现有 conditional runner 的兼容方式

`conditional_dual_data_utils.py` 已支持 3D signals：`[B,C,L]` 会按 channels-first 规则 flatten 成 `[B,C*L]`。

`train_conditional_dual.py` 使用 `infer_signal_len(dataset)` 得到 flatten 后的 encoder input length，并在 `run_summary.md` 中记录原始 signal shape、channel count、per-channel length 和 flattened signal length。

因此第一版 COMSOL-style `.npz` 可以直接接入现有 MLP `BzEncoder`。后续如果需要保留 channel structure，可以再引入真正的 multi-channel `Conv1d` encoder 或 channel-aware signal encoder。

## 5. 当前 S66 范围

S66 只做：

- NPZ schema validator；
- mock COMSOL-style smoke test；
- 文档和 artifact index 更新。

S66 不调用 COMSOL，不生成真实 COMSOL 数据，不训练模型，不保存 checkpoint，也不声称模型性能。

## 6. S67 long CSV schema

S67 增加一个面向 COMSOL-style signal 导出的标准 long-table CSV 约定。推荐列为：

- `sample_index`
- `channel_index`
- `channel_name`
- `lift_off`
- `field_component`
- `x_index`
- `x`
- `value`

推荐的 COMSOL 导出格式是每个 `(sample_index, channel_index, x_index)` 对应一行。

`sample_index` 表示 defect / simulation sample。`channel_index` 表示 lift-off height、field component、probe line 或其他 signal channel。`x_index` 表示 probe line 上的位置点。`value` 保存该行的 magnetic field value。

## 7. CSV -> NPZ converter entrypoint

S67 增加 `convert_comsol_multiheight_csv_to_npz.py`。

输入：

- `--signals-csv`: long-table COMSOL-style signals CSV.
- `--target-npz`: target arrays containing `mu_maps` or `masks`, plus `x/y` or `coords`.
- `--output-npz`: converted multi-channel `.npz`.

converter 会按 `sample_index`、`channel_index`、`x_index` 排序，构造 `signals [num_samples, num_channels, signal_len]`，复制 target arrays，并写入 `signal_channel_names`、`lift_off_values`、`field_components`、`source_type`、`signal_flatten_order` 等 metadata。

该 converter 不调用 COMSOL，也不修改输入文件。
## 8. S68 pilot handoff

S68 增加 `COMSOL_PILOT_DATA_REQUEST.md`，用于把真实 COMSOL pilot 数据需求交给 COMSOL MCP / COMSOL 项目。

第一批真实 pilot 建议先保持小规模：samples = 5 到 10，grid_x = 200，grid_y = 100，probe x points = 200，channels = 3，`lift_off_values = [0.5, 1.0, 2.0]`，`field_components = ["Bz", "Bz", "Bz"]`。

COMSOL 侧建议输出：

- `signals_multiheight.csv`
- `targets.npz`
- `README.md`

回到本支线后，使用 S67 converter 接入：

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv signals_multiheight.csv --target-npz targets.npz --output-npz comsol_multiheight_pilot.npz
python comsol_multiheight_npz_utils.py --npz-path comsol_multiheight_pilot.npz
```

S68 仍然只是 handoff 文档阶段，不调用 COMSOL，不生成真实 COMSOL 数据，也不运行训练。
