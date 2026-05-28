# True 3D RBC 真实数据接入 Schema

阶段：20.97 real-data schema intake / acquisition metadata contract。

本文档定义真实实验 MFL 数据进入当前 true 3D RBC 推理链路之前必须满足的数组格式、metadata 合同和阻断条件。它不是新的 baseline，也不修改 `CURRENT_BASELINE.md`。

当前推理链路：

`delta_b + sensor_z_m -> 20.85 baseline 或 20.85 baseline + A2 adapter -> L_m, W_m, D_m, wLD, wWD, wLW -> RBC-style profile/depth -> projected mask`

`sensor_z_m` 必须提供。名义 liftoff `0.008 m` 默认走 frozen 20.85 baseline；`[0.006, 0.012]` 内的非名义 liftoff 默认走 A2 liftoff companion adapter。缺少 `sensor_z_m` 是 blocker；超出范围的 `sensor_z_m` 必须标记为 `out_of_range`，不能当作已验证的生产推理输入。

## 格式 1：Prepared Delta-B Format

这是推荐接入格式。

必需数组：

- `delta_b`：批量数据 shape 为 `(N, 3, 3, 201)`，单样本 shape 为 `(3, 3, 201)`。

必需 metadata：

- `axis_order`：必须是 `["Bx", "By", "Bz"]`。
- `scan_line_y_m`：必须映射到三条 scan line，推荐 `[-0.001, 0.0, 0.001]`。
- `sensor_x_m`：长度必须为 `201`，并按扫描方向排序。
- `sensor_z_m`：单位为 m，表示 liftoff，每个样本必须提供。
- `delta_b_unit`：必须是 `Tesla`。
- `sample_id`：每个样本必须提供。
- `specimen_id`：每个样本必须提供。
- `no_defect_reference_id`：每个样本必须提供。
- `coordinate_system`：必须提供，并说明扫描方向、y-line 约定和 liftoff 方向。
- `no_defect_reference_method`：必须提供，并说明 no-defect reference 如何匹配采集。
- `sensor_alignment_status`：必须提供，并说明 `Bx/By/Bz` 是否完成空间对齐。
- `gain_calibration_status`：必须提供；gain/amplitude calibration 只作为 diagnostic，不替换 baseline。
- `material`：已知时必须记录。
- `specimen_info`：已知时必须记录。
- `magnetization_setup`：必须提供。

可选 metadata：

- `split_tag`：可选，仅用于报告，不允许作为模型输入。
- `acquisition_date`：可选。
- `operator`：可选。
- `ground_truth_LWD`：可选，但建议提供，用于评价 L/W/D。
- `profile_depth_ground_truth`：可选，用于评价 profile/depth。

## 格式 2：Raw Defect Plus No-Defect Format

当原始 defect scan 和匹配的 no-defect reference 分开提供时，使用此格式。

必需数组：

- `b_defect`：shape 为 `(N, 3, 3, 201)` 或 `(3, 3, 201)`。
- `b_no_defect`：shape 为 `(N, 3, 3, 201)` 或 `(3, 3, 201)`。
- 推理前必须计算 `delta_b = b_defect - b_no_defect`。

格式 1 的所有 metadata 要求仍然适用。`no_defect_reference_id` 和 `no_defect_reference_method` 是强制字段。

## Blockers

以下情况会阻止真实数据推理，必须先修复：

- 缺少 `sensor_z_m`。
- `delta_b` 不可信，且缺少可匹配的 no-defect reference。
- 只有 `Bz`，没有 `Bx/By`；当前 true 3D RBC 模型需要 `Bx`、`By`、`Bz` 三轴输入。
- 轴顺序未知。
- 磁场单位未知。
- `sensor_x_m` 无法重采样到长度 `201`。
- `scan_line_y_m` 无法映射到三条 scan line。
- `sensor_z_m` 超出 `[0.006, 0.012]`，且没有明确的 out-of-range 重新训练或验证计划。
- internal / buried defect 被混入当前 surface-breaking RBC-style schema。

## Warnings

以下情况不一定阻止接入，但必须在报告中标记：

- gain 或 amplitude calibration 状态未知。
- sensor alignment 未验证。
- 没有 L/W/D 或 profile/depth ground truth；模型可以推理，但无法完整评分。
- 真实试件材料或 magnetization setup 与 COMSOL 仿真域不一致。

## 真实数据推理边界

当前分支仍然是 `exact_piao_rbc=False` 和 `rbc_style_approximation=True`。它验证的是 COMSOL-derived surface-breaking RBC-style defect，不覆盖 arbitrary free-form、buried/internal 或 multi-defect real data。
