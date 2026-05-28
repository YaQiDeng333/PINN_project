# Internal / Buried Defect Feasibility Schema

阶段：20.99 internal / buried defect feasibility schema for true 3D MFL。

本 schema 是独立分支，不替换 `CURRENT_BASELINE.md`。当前 `CURRENT_BASELINE` 仍是 20.85 true 3D RBC surface / near-surface profile-depth baseline；A2 仍只是 liftoff robustness companion module。内部/埋藏缺陷不能直接混入当前 surface RBC schema，因为它的标签语义、可观测性和评价目标都不同。

## internal defect 与 surface defect 的分界

surface / near-surface defect = 缺陷与扫描表面相交或足够接近表面，当前 RBC-style baseline 用 `L_m, W_m, D_m, wLD, wWD, wLW` 表示表面缺陷 profile，并生成 surface depth/profile 与 projected mask。

internal / buried defect = 缺陷不直接暴露在扫描表面，磁场扰动来自埋藏空腔或材料不连续体。此时 `D_m` 不再等价于表面缺陷深度，必须额外定义 `burial_depth_m` 或 `depth_to_surface_m`。当前 surface RBC 的 profile/depth/mask 语义不能直接复用。

## required inputs

真实采集或 COMSOL smoke pack 必须提供：

- `Bx/By/Bz` 三轴 MFL 输入；如果只有 `Bz`，必须进入低能力 `Bz-only` 分支，不能使用当前 true 3D RBC 主分支。
- `sensor_z_m`，单位 m，记录传感器 liftoff。
- `no_defect_reference_id` 和 `no_defect_reference_method`，用于计算可信 `delta_b = b_defect - b_no_defect`。
- `axis_order`，推荐固定为 `["Bx", "By", "Bz"]`。
- `scan_line_y_m`，至少三条扫描线；若真实实验只有单线，需要单独定义低能力分支。
- `sensor_x_m`，需要能重采样到项目标准长度 `201`。
- `unit`，磁场单位必须可换算到 `Tesla`。
- `material` 与 `specimen_geometry`，包括试件长宽厚、扫描面定义、材料牌号或磁性参数来源。
- `coordinate_system`，必须定义 x 扫描方向、y 横向线方向、z liftoff / depth 方向。
- `sensor_alignment_status`，记录三轴是否完成空间对齐。
- `gain_calibration_status`，记录幅值/gain 是否可追溯。

## required labels

internal / buried defect 至少需要这些标签；没有这些标签时只能做采集 schema 检查，不能做监督训练或定量 benchmark。

- `L_m`：缺陷沿 x 或主扫描方向的长度。
- `W_m`：缺陷沿 y 或横向方向的宽度。
- `D_m` 或 `cavity_size_m`：空腔尺寸或缺陷体在 z 方向的厚度。注意它不是 surface RBC 的表面深度。
- `burial_depth_m`：从扫描表面到缺陷上表面或最近点的深度。
- `depth_to_surface_m`：同 `burial_depth_m`，若定义不同必须写清参考点。
- `defect_center_xyz_m`：缺陷中心在统一坐标系下的位置。
- `shape_type`：例如 `internal_ellipsoid`、`internal_cuboid`、`sphere_like`。
- `profile_descriptor` 或 `cavity_mask`：用于描述缺陷体形状的参数、体素 mask 或可投影描述。
- `ground_truth_method`：标签来源，例如机械加工设计值、CT、切片测量、CAD、COMSOL 参数表。

## output representation candidates

推荐候选按优先级从简单到复杂排列：

1. `shape_type + L/W/D + burial_depth + center_xyz`
   - 最小可执行表示，适合 6-12 个样本的 COMSOL smoke。
   - 风险是只覆盖规则几何，不能表示自由形状。

2. `internal_ellipsoid_params`
   - 输出椭球中心、三轴半径、埋深和姿态。
   - 适合模拟球状/椭球状内孔，标签稳定。

3. `internal_cuboid_params`
   - 输出长方体空腔中心、尺寸、埋深和姿态。
   - 与加工块状缺陷更接近，但边角导致场响应更尖锐。

4. `3D occupancy / cavity mask`
   - 输出体素化空腔 mask。
   - 表达力强，但需要更多数据和明确体素坐标系，不能作为第一轮 smoke 的主目标。

5. `surface_equivalent_projected_profile`
   - 把内部缺陷投影成表面等效 footprint 或等效 profile。
   - 只能作为 QA / comparator，不应作为唯一监督目标，因为它会丢失 `burial_depth_m`。

## blockers

以下任一条件成立时，不能进入 internal defect supervised training gate：

- 无 `burial_depth_m` 或 `depth_to_surface_m`。
- 无可信 `no_defect_reference`，无法构造稳定 `delta_b`。
- 只有 `Bz`，且没有明确的 `Bz-only` 低能力分支。
- 不知道缺陷相对扫描面的坐标或坐标系。
- 没有 ground truth，或 ground truth 来源不明。
- 不知道 `sensor_z_m`。
- 不知道材料、试件尺寸、励磁设置或传感器对齐/gain 状态。

## 为什么不能直接用当前 surface RBC baseline

当前 surface RBC baseline 学的是：

`delta_b + sensor_z_m -> surface RBC six params -> surface profile/depth -> projected mask`

internal / buried defect 的真实问题是：

`delta_b + sensor_z_m + specimen/material context -> buried cavity geometry + burial depth + center -> volumetric or equivalent response`

两者的关键差异在 `burial_depth_m` 和缺陷体积语义。内部缺陷的磁场幅值和形状会同时受尺寸、埋深、材料路径、liftoff 和传感器对齐影响；如果直接套 surface RBC 六参数，模型会把埋深变化误解释为表面 profile 或 curvature 变化，得到物理含义错误的输出。

因此 20.99 的结论是：internal / buried defect 必须先走独立 feasibility schema 和 COMSOL smoke pack，再决定是否训练 internal-specific model。
