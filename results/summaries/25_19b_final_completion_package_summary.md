# 25.19b Final MFL Surface Inversion Completion Package

## Executive Conclusion

当前 `PINN_project` 可以阶段性交付为：基于 COMSOL 仿真与深度学习的 MFL 表面 RBC-like 缺陷反演方法研究，含 liftoff companion、surface forward-refinement companion 与 multi-pit 扩展边界分析。

这个版本的核心完成状态不是“所有表面缺陷都稳定推理”，而是形成了一个边界清楚的交付包：20.85 nominal true 3D RBC profile-depth baseline 保持稳定；liftoff robustness 有 companion-level 处理；surface forward-refinement 对部分 RBC-representable 但 frozen inverse 失败的 single-component surface defects 有明显改善；multi-pit 已完成数据、训练探索、失败归因和 geometry-primary 后续路线设计。

## Current Baseline Status

`CURRENT_BASELINE.md` 保持不变，当前权威 baseline 仍是 20.85 / 20.86 后的 true 3D RBC-style profile-depth reconstruction。它使用 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，输入为 Bx/By/Bz `delta_b`，输出 `L_m/W_m/D_m/wLD/wWD/wLW` 并生成 RBC-style 3D profile/depth 和 projected mask。

关键 test 指标保持原口径：normalized MAE `0.678014`，profile depth RMSE `0.000387737 m`，Er-like profile error `0.340544`，projected mask Dice `0.847727`。本轮没有 baseline transition，也没有修改 `CURRENT_BASELINE.md`。

## Liftoff Companion Status

liftoff-conditioned companion 已打通。20.95 formal benchmark 选择 `A2_latent_residual_adapter` seed `2026`，它保持 nominal `0.008 m` operating point，同时把 non-nominal liftoff profile RMSE 从 frozen baseline 的 `0.000874310 m` 降到 `0.000437214 m`，non-nominal Dice 为 `0.842378`。

20.96 smoke 已形成 routing contract：nominal `sensor_z_m≈0.008` 使用 frozen 20.85 baseline，non-nominal liftoff 使用 baseline + A2；`sensor_z_m` 缺失是 hard error，不允许猜测。A2 是 robustness companion，不是 `CURRENT_BASELINE` replacement。

## Surface Forward-Refinement Companion Status

surface forward-refinement companion 已收口。25.7 / 25.8 固定 runner：frozen 20.85 prediction -> observed `delta_b` feature extraction -> exported `ridge_param_only_linear_alpha_10` artifact -> post-hoc refine `L_m/W_m/D_m/wLD/wWD/wLW`。

在 `82` 条 `rbc_representable_but_model_fail` rows 上，profile RMSE 从 `0.000509518351056 m` 改善到 `0.000220386413188 m`，Er-like 从 `2.80015739379` 降到 `0.909941363416`，Dice 从 `0.480524080842` 升到 `0.709451842351`。这个 runner 是 20.85 上的 companion/post-hoc repair layer，不是 baseline replacement；multi-pit / component-set / multi-component rows 必须标记为不适用。

## Surface RBC / RBC-Representable Capability

可以声称：RBC-like / RBC-representable single-component surface defect inversion pipeline 已形成 stable baseline + companion 体系。主线是 20.85 nominal true 3D RBC profile-depth baseline；liftoff variation 由 A2 companion 处理；部分 non-canonical but RBC-representable single-component surface defects 可由 forward-refinement companion 明显改善。

这仍是 COMSOL 仿真域内、RBC-style / Piao-inspired approximation，不是 exact Piao 2019 reproduction，也不是真实实验部署级结论。

## Multi-Pit Component-Set Data And Exploration Status

multi-pit component-set pilot dataset 已完成。`comsol_surface_multipit_component_set_pilot_v1` 有 `112` 行，split 为 `72/20/20`，`K_max=3`，component count 为 `2=100`、`3=12`，separation 覆盖 `separated=40`、`close=24`、`touching=24`、`partially_overlapping=24`。manifest 标记 `train_ready_candidate=true`、`baseline_ready=false`、`auto_discovery_allowed=false`。

训练探索也已完成到足够判停的程度：25.10 component-set gate 是 `PARTIAL`，25.17 label-v3b gate 仍是 `PARTIAL`，test merged rate 保持 `1.000000`，component Dice 只有 `0.034007`，union Dice `0.060961`。这些结果支持失败归因和下一路线设计，但不支持把 multi-pit 写成 stable inference model。

## Raster-Target Mainline Stop Reason

25.18 主结论是 `STOP_RASTER_TARGET_MAINLINE`。停止原因是路线级模式，而不是单个 target bug：25.13 target-v2 清理 ownership 后变成 near-empty mask collapse；25.15 label-v3 通过扩大 soft support 缓解 sparsity，却变成 union-like merged collapse；25.17 label-v3b 加 hard-core / halo / SDF 后，merged rate 仍为 `1.000000`。

因此停止的不是 multi-pit 方向，而是“继续把 per-component raster ownership 当作主监督继续调 target / loss”的主线。

## Geometry-Primary Future Route

25.19 已明确下一阶段路线：geometry-primary component-set。`K=3` 表示最大 component slots，不是 Piao kernel。每个 slot 输出 `existence_prob`、`center_x_m`、`center_y_m`、`L_m`、`W_m`、`D_m`、`rotation_angle`、`shape_family` 和 `compact_shape_parameters`。

mask/depth 改为从 geometry slots 派生：`derived_component_mask`、`derived_union_mask`、`derived_component_depth`、`derived_union_depth`。后续 forward consistency 路径是 `geometry slots -> derived profile/mask/depth -> lightweight forward surrogate or feature-space residual -> Bx/By/Bz residual`，先作为 evaluator / refinement referee，再考虑进入训练 loss。

## Final Deliverable Checklist

- 20.85 stable `CURRENT_BASELINE` 保持不变：完成。
- liftoff-conditioned companion 已打通：完成。
- surface forward-refinement companion 已收口：完成。
- RBC-like / RBC-representable single-component surface defects 有 stable baseline + companion 体系：完成。
- multi-pit component-set pilot dataset 已完成：完成。
- multi-pit raster-target 主线已完成探索并判停：完成。
- geometry-primary + forward consistency 后续路线已明确：完成。

## Limitations / Not Claimed

不能声称 multi-pit stable inference model 已完成；不能声称 touching / overlap / three-component 稳定分离已解决；不能声称 arbitrary non-RBC stable inference 已解决；不能声称工程部署级鲁棒性或真实实验 MFL 验证完成；不能把 multi-pit 写成 baseline；不能更新 `CURRENT_BASELINE.md`。

## Next Optional Route

若继续当前 completion package 之外的研究，下一步才是：`25.20 separated/close two-component geometry-primary training gate; optional beyond current completion package; no baseline transition.`

25.20 不是当前收口必需项，也不是 baseline transition。它只应先测试 separated / close two-component 子集，不进入 touching / partially_overlapping / three-component。

## Baseline Transition Statement

No baseline transition. `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC profile-depth baseline.
