# S69 COMSOL pilot handoff end-to-end dry-run

## 目的

S69 用 tempfile 模拟 COMSOL 侧输出，并验证当前支线的 handoff 链路是否能从 `signals_multiheight.csv` 和 `targets.npz` 走到 converted NPZ、validator、conditional batch 和 model forward。

## dry-run 模拟文件

- `signals_multiheight.csv`：long schema，包含 `sample_index`、`channel_index`、`channel_name`、`lift_off`、`field_component`、`x_index`、`x`、`value`。
- `targets.npz`：包含 `mu_maps [4,10,20]`、`x [20]`、`y [10]`。

mock 数据规模为 samples = 4、channels = 3、signal_len = 20。channels 为 `Bz_liftoff_0p5`、`Bz_liftoff_1p0`、`Bz_liftoff_2p0`。

## 验证链路

`smoke_test_comsol_pilot_handoff_end_to_end.py` 覆盖：

- 调用 S67 `convert_comsol_multiheight_csv_to_npz.py`；
- 用 S66 `validate_comsol_multiheight_npz` 验证转换后的 `.npz`；
- 用 `load_conditional_npz` / `get_conditional_batch` 读取 batch；
- 检查 batch signals shape 为 `[3,60]`；
- 检查 `infer_signal_len` 返回 `60`；
- 运行 `ConditionalDualNet(signal_len=60, latent_dim=16, hidden_dim=32, num_layers=2)` forward；
- 检查 `mu` / `phi` shape。

## 当前边界

S69 仍然不是真实 COMSOL 数据，不调用 COMSOL，不训练模型，不保存 checkpoint / 权重 / 图片。

## 下一步建议

下一步应切到 COMSOL MCP 项目或相关 COMSOL 对话，按 S68 / S70 的请求生成真实 pilot 数据。
