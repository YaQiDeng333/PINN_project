# S67 COMSOL multi-height CSV to NPZ converter

## 目的

S67 创建一个标准转换入口，用于后续 COMSOL-style multi-height / multi-channel Bz data。它把 long-table signal CSV 和 target `.npz` 转换成 S66 multi-channel NPZ schema。

S67 不调用 COMSOL，不生成真实 COMSOL 数据，也不运行正式训练。

## Converter 输入格式

`convert_comsol_multiheight_csv_to_npz.py` 接收：

- `--signals-csv`
- `--target-npz`
- `--output-npz`

long CSV 必须包含：

- `sample_index`
- `channel_index`
- `channel_name`
- `lift_off`
- `field_component`
- `x_index`
- `x`
- `value`

target NPZ 必须包含：

- `mu_maps` or `masks`
- `x/y` or `coords`

## Converter 输出 NPZ schema

转换后的 `.npz` 包含：

- `signals`，shape 为 `[num_samples, num_channels, signal_len]`
- `mu_maps` or `masks`
- `x/y` or `coords`
- `signal_channel_names`
- `lift_off_values`
- `field_components`
- `source_type = "converted_comsol_multiheight_csv"`
- `signal_flatten_order = "channels_first"`
- `converter_note`
- `csv_sample_indices`
- `csv_channel_indices`
- `csv_x_indices`

`signals` 保存为 `float32`。target arrays 从 `target_npz` 复制，shape 不做修改。

## Smoke test 覆盖范围

`smoke_test_convert_comsol_multiheight_csv_to_npz.py` 使用 tempfile 创建 target NPZ，其中包含 `mu_maps [3,10,20]`、`x [20]`、`y [10]`；同时创建包含 3 个 samples、3 个 channels、20 个 x points 的 mock long CSV。

smoke test 检查：

- converter 能创建输出 `.npz`；
- 输出 `signals` shape 为 `[3,3,20]`；
- `mu_maps`、`x`、`y` 被保留；
- `signal_channel_names`、`lift_off_values`、`field_components` 存在；
- `comsol_multiheight_npz_utils.validate_comsol_multiheight_npz` 能接受转换后的文件；
- `conditional_dual_data_utils.get_conditional_batch(..., [0,1])` 返回 `signals [2,60]`；
- `infer_signal_len` 返回 `60`；
- `ConditionalDualNet(signal_len=60)` forward 能运行；
- 缺少 CSV 必需列时会失败；
- sample/channel/x 覆盖不完整时会失败。
- signal `value` 出现 NaN / Inf 时会失败。

## 当前边界

这是用 mock CSV data 验证的 converter skeleton。它不是真实 COMSOL export result，也不包含任何训练结果。

## 下一步建议

S68 可以准备 COMSOL exporter 侧 CSV 约定，或使用 COMSOL MCP 项目生成第一批真实 multi-height Bz pilot dataset。训练前应先用 S66/S67 工具验证数据。
