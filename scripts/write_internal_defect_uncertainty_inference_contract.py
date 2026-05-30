#!/usr/bin/env python
"""写出 internal defect uncertainty / abstention 推理合同。"""

from __future__ import annotations

from pathlib import Path

from load_internal_defect_pilot_dataset import ROOT


CONTRACT_PATH = ROOT / "results/summaries/internal_defect_uncertainty_inference_contract.md"


def main() -> int:
    text = """# Internal defect uncertainty inference contract

当前 internal / buried defect 模型只能称为 benchmark candidate，不能称为 stable inference model，也不能替代 surface / near-surface `CURRENT_BASELINE.md`。

## 必填输入

- `delta_b` 或等价的 `Bx/By/Bz` 三轴磁场差分，轴顺序必须为 `[Bx, By, Bz]`。
- `sensor_x_m`、`scan_line_y_m`、`sensor_z_m`、单位、坐标系、no-defect reference 元数据。
- 数据必须通过 manifest / registry 指定 dataset 或真实样本 manifest，不允许 latest/newest 自动扫描。

## 推理输出

推理必须输出两层结果：

1. internal model prediction：`L/W/D`、`burial_depth_m`、`center_xyz_m`、`shape_type`。
2. risk gate prediction：`risk_score`、`risk_route`、`risk_reason`。

`risk_route` 的含义：

- `accept`：可以报告模型预测，但仍标注为 internal benchmark-domain prediction。
- `abstain_need_review`：只能报告高风险，不给出确定的 center/burial 结论。

## 高风险行为

高风险样本不得被写成稳定推理结果。若 risk gate 标记为高风险，应输出：

- `risk_score`
- 触发的主要风险信号，例如跨模型 shape disagreement、center/burial disagreement、预测范围异常或 delta_b feature anomaly
- 建议人工复核或追加采集/仿真

## 适用范围

当前合同只覆盖 `comsol_internal_defect_pilot_pack_v3_hardcase` 的 COMSOL internal defect 仿真域。真实 internal sample inference 之前仍需要真实样本 metadata/schema dry run；若传感器、材料、坐标、no-defect reference 或 liftoff 条件不匹配，应先停止。
"""
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(text, encoding="utf-8")
    print(CONTRACT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
