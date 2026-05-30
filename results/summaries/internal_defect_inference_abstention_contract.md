# Internal defect inference with abstention contract

internal defect 推理当前不是 stable all-sample predictor。22.7 的唯一允许口径是：模型先给出 raw internal defect prediction，再由 risk gate 输出 `risk_score` 和 `inference_status`。

## 输出字段

- `risk_score`：22.6 validation-selected risk gate 的高风险分数。
- `inference_status`：只能是 `accepted_prediction` 或 `abstain_need_review`。
- `accepted_prediction`：可以报告 `L/W/D`、`burial_depth_m`、`center_xyz_m`、`shape_type`，但仍属于 COMSOL internal benchmark domain prediction。
- `abstain_need_review`：只能保存 raw prediction、risk_score 和 warning；不得给出稳定可靠的 center/burial 结论。

## 高风险样本处理

当 `risk_score >= 0.07046389` 时，样本必须标记为 `abstain_need_review`。这类样本需要人工复核、更多观测、重新采集或后续 COMSOL hard-case/richer-observation 计划，不能被自动写入确定几何结果。

## 适用边界

当前合同只适用于 `comsol_internal_defect_pilot_pack_v3_hardcase` 的 COMSOL internal / buried defect domain。真实样品推理前必须先完成 metadata/schema validation，包括 Bx/By/Bz、no-defect reference、sensor_z_m、坐标系、单位、扫描线、sensor_x 对齐、gain 状态和 ground truth 记录。

## 禁止口径

- 不把 internal branch 写成 `CURRENT_BASELINE.md`。
- 不把 `abstain_need_review` 样本解释成稳定推理。
- 不用 test split 重新选择 risk threshold。
- 不用 true label、shape/bin、sample_id 作为 inference input。
