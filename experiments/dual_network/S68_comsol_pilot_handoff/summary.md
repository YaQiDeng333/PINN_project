# S68 COMSOL pilot data handoff

## 目的

S68 将支线从 COMSOL-style multi-height Bz interface skeleton 推进到真实 COMSOL pilot 数据交接准备阶段。该阶段只准备请求文档和接入说明，不调用 COMSOL，也不生成真实 COMSOL 数据。

## 新增产物

- `COMSOL_PILOT_DATA_REQUEST.md`：面向 COMSOL MCP / COMSOL 项目的 pilot 数据请求说明。

## pilot 数据建议规模

- samples = 5 到 10；
- grid_x = 200；
- grid_y = 100；
- probe x points = 200；
- channels = 3；
- lift_off_values = [0.5, 1.0, 2.0]；
- field_components = ["Bz", "Bz", "Bz"]。

## 后续接入命令

COMSOL 侧生成 `signals_multiheight.csv` 和 `targets.npz` 后，回到本支线运行：

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv signals_multiheight.csv --target-npz targets.npz --output-npz comsol_multiheight_pilot.npz
python comsol_multiheight_npz_utils.py --npz-path comsol_multiheight_pilot.npz
```

## 当前边界

S68 没有调用 COMSOL，没有生成真实 COMSOL 数据，没有训练模型，也不声称任何模型效果。

## 下一步建议

下一步应在 COMSOL MCP 项目或相关对话中生成 5-10 个真实 pilot samples，然后回到本支线用 S67 converter 转换，并用 S66 validator 验证。
