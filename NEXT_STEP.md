# NEXT_STEP

## 2026-05-26 after Stage 20.88 preflight blocker

Next step: **recover or export the frozen 20.77/20.85 baseline model artifact before robustness evaluation**.

20.88 stopped at preflight. Registry/manifest/schema checks passed for `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`, but the required frozen-model artifact was not available locally: no selected seed=42 true 3D RBC checkpoint and no sufficient raw prediction artifact that can rerun the model on perturbed `delta_b`.

Do not treat the clean per-sample metrics as a robustness result. The next step is a separate artifact recovery/export stage: first try to recover the 20.77/20.85 selected checkpoint; if that is impossible, explicitly approve a fixed 20.85 artifact-export rerun that saves a checkpoint/prediction artifact. Only after that should 20.88 observation perturbation robustness be rerun. Do not train inside 20.88.

## 2026-05-26 after Stage 20.87

Next step: **20.88 observation perturbation robustness audit on the current true 3D RBC baseline**.

Stage 20.87 was design-only: no COMSOL, no training, no data/NPZ generation, and no `CURRENT_BASELINE.md` change. The next actionable step is to perturb the existing v3_240 `delta_b` observations through explicit dataset_id / manifest loading and evaluate the frozen 20.86 baseline under observation-space stress only.

20.88 should start with additive noise `0/5/10/15/20%`, amplitude scaling / sensor gain error, baseline zero drift, no-defect reference subtraction error, channel dropout, and `sensor_x_resampling_jitter` as a diagnostic-only interpolation perturbation. Formal spatial-sampling, liftoff, scan-line offset, Bx/By/Bz misalignment, source-strength, and material/B-H claims require the later 20.89 COMSOL diagnostic pack. Internal/buried defects stay out of the current surface RBC baseline and should wait for 20.91 label/schema design.

## 2026-05-26 after Stage 20.86

Next step: **benchmark documentation and real-data alignment planning around the new true 3D RBC profile-depth baseline**.

Stage 20.86 promoted the 20.77/20.85 formal rerun candidate to `CURRENT_BASELINE`. The project baseline has transitioned from old v3_complex 2D mask/boundary prediction to true 3D RBC-style profile-depth reconstruction. The old 2D baseline remains an archived comparator; it was not deleted.

The new current baseline is fixed to `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`, with Bx/By/Bz `delta_b` input `(N,3,3,201)`, Conv1D input `(N,9,201)`, and six RBC-style outputs `L_m/W_m/D_m/wLD/wWD/wLW`. Its headline metrics are profile/depth metrics: `profile_depth_rmse_m=0.000387737`, Er-like profile error `0.340544`, and L/W/D MAE `1.892/2.186/0.800 mm`. Projected mask Dice `0.847727` remains QA; wMAE `0.201076` remains auxiliary diagnostic.

Immediate next work should not be another training tweak. The useful next step is to prepare a concise benchmark/report narrative and plan real-data alignment or exact-Piao/representation follow-up under the new baseline scope. Keep `exact_piao_rbc=False` and do not claim real experimental deployment readiness.

## 2026-05-26 after Stage 20.85

Next step: **prepare paper/report display around the formal 20.77-profile benchmark candidate**.

Stage 20.85 reran the 20.77 neural candidate on `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` with the original architecture, loss, and validation-only selection protocol. Seeds `42/123/2026` completed; seed `42` was selected. The rerun exactly reproduced the 20.77 profile/depth result: test normalized MAE `0.678014`, L/W/D MAE `1.892/2.186/0.800 mm`, profile depth RMSE `0.000387737 m`, Er-like profile error `0.340544`, projected mask IoU/Dice `0.750650/0.847727`, and auxiliary wMAE `0.201076`.

The role split remains fixed: 20.77/formal rerun is the profile/depth benchmark candidate; 20.81 remains the projected-mask / visual comparator because its Dice is higher but profile RMSE is worse; 20.83 remains negative evidence for the tested profile-primary loss. This is still not a baseline replacement, and `CURRENT_BASELINE.md` must remain unchanged.

## 2026-05-26 after Stage 20.84

Next step: **A. keep 20.77 as profile/depth benchmark candidate for formal rerun**.

Stage 20.84 consolidated the existing 20.77 / 20.81 / 20.83 candidates without retraining, COMSOL, new data, NPZ changes, or baseline updates. The role split is now fixed:

- 20.77 remains the profile/depth main candidate: `profile_depth_rmse_m=0.000387737`, projected mask Dice `0.847727`.
- 20.81 remains the non-negative projected-mask / visual reference: Dice `0.866573`, but profile RMSE `0.000445297` is worse than 20.77.
- 20.83 remains negative evidence for the current R1 profile-primary loss path: Dice `0.868042` is numerically high, but profile RMSE `0.000409718` is worse than 20.77, so it cannot replace the profile/depth candidate.

The prediction gallery audit supports the same split: best-profile samples are genuinely low profile-error cases, but worst-profile and high-Dice/high-profile-error samples show that 2D projected mask quality is not enough to judge the 3D profile. Do not continue small tweaks to the current 20.83 profile-primary loss. If the route continues, run a formal benchmark rerun around 20.77 as the profile/depth candidate and keep 20.81 only as the visual/mask comparator. Do not update `CURRENT_BASELINE.md`.

## 2026-05-26 after Stage 20.83

Next step: **B. keep 20.77/20.81 candidate**.

Stage 20.83 tested `R1_six_params_profile_primary_loss` on `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`. The result is a negative training-gate result: the selected profile-primary candidate improved projected mask Dice (`0.868042`) but did not improve the primary 3D profile metric (`profile_depth_rmse_m=0.000409718` vs 20.77 `0.000387737`). Multi-seed was correctly skipped because the candidate screen gate failed.

Keep 20.77 as the profile reconstruction reference and 20.81 as the visual/mask comparator. Do not update baseline docs or `CURRENT_BASELINE.md`.

Immediate boundary for the next stage: no COMSOL or data expansion is justified by 20.83 alone. If continuing the true 3D route, the next useful work is a cleaner profile-native representation experiment, not another small loss-weight tweak.

## 2026-05-25 更新：第 20.82 后的下一步

第 20.82 已完成 true 3D RBC curvature label / output representation audit。本轮没有运行 COMSOL，没有生成或修改 data / NPZ，没有重新训练模型，没有建立 baseline，也没有修改 `CURRENT_BASELINE.md`。审计边界很明确：20.77 / 20.81 有逐样本 profile/error artifacts；20.80 只有 aggregate/group/failure-case artifacts；当前没有 raw `pred_params` 或 predicted profile arrays，因此不做 prediction reconstruction。

核心判断是：不要继续把 `wLD/wWD/wLW` 逐项 MAE 当作 true 3D branch 的主评价。它们仍是有用的 curvature diagnostic，但 Piao-style 路线真正要评价的是六参数生成的 3D profile 是否准确。20.77 test 的 curvature-vs-profile RMSE correlation 只有 `0.358243`；20.81 虽然 Dice 更高，但 profile depth RMSE 更差，说明 projected mask 也不能替代 3D profile 指标。

下一步唯一建议：**20.83 做 `R1_six_params_profile_primary_loss`**。继续输出 `L/W/D/wLD/wWD/wLW`，但把 validation / loss 主目标改成 profile-level reconstruction，例如 `profile_depth_rmse_m` 或 Er-like depth/profile error；`wLD/wWD/wLW` 降为 auxiliary diagnostics；不需要新 COMSOL 数据，不需要扩到 480，不做 baseline replacement。
## 2026-05-25 更新：第 20.81 后的下一步

第 20.81 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 feature-fusion neural model diagnostic。本轮没有运行 COMSOL，没有生成或修改 NPZ/data，没有创建 baseline，也没有更新 `CURRENT_BASELINE.md`；所有输入仍通过 registry/manifest 显式 dataset_id 加载，禁止 latest/newest 自动扫描。

核心判断是：feature-fusion 改善了整体拟合和 projected mask，但没有解决 curvature 风险。validation-only selection 选中 `H3_curv_fusion_F0F1F2_w1p0`，multi-seed 后 selected seed `2026` 的 test total MAE 为 `0.667888`，优于 20.77 neural 的 `0.678014`；projected mask Dice 为 `0.866573`，也优于 20.77 的 `0.847727`。但是 curvature MAE 只是从 `0.201076` 降到 `0.194483`，改善 `0.006592`，未达到本轮 `>=0.01` 实质改善门槛；wLD 从 `0.209439` 退到 `0.217079`，且 curvature 仍弱于 20.80 feature-only 的 `0.190304`。

下一步唯一建议：**redefine curvature labels / output representation**。真正的分界点不是再往 neural head 里塞更多 feature，而是 `wLD/wWD/wLW` 这组三维 profile curvature label 是否以当前形式可辨识、可学习、可评价。下一轮应先审计 curvature 参数定义、替代输出表示、profile/depth loss 口径和与 projected mask 的脱钩关系；不要直接 formal benchmark rerun，不要 baseline replacement，也不要先扩到 480。

## 2026-05-25 更新：第 20.80 后的下一步

第 20.80 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 Piao/NLS-inspired feature diagnostic。本轮没有运行 COMSOL，没有生成或修改 NPZ，没有训练 neural model，没有创建 baseline，也没有更新 `CURRENT_BASELINE.md`；所有输入仍通过 registry/manifest 显式 dataset_id 加载，禁止 latest/newest NPZ 自动扫描。

核心判断是：F0+F1+F2 physical features 对 curvature 有真实但有限的帮助。validation 选中 `F0_F1_F2_basic_physical + svr_rbf_C10_eps0.03`，test total MAE 为 `0.695724`，curvature MAE 为 `0.190304`，wLD/wWD/wLW 为 `0.209649/0.194797/0.166465`，projected mask Dice 为 `0.826272`。它优于 20.77 feature baseline 的 total `0.715395` / curvature `0.195046`，也比 20.79 refined model 更好；但仍弱于 20.77 neural 的 total `0.678014` 和 Dice `0.847727`，且 wLD 没有改善。F4 NLS proxy 提取稳定，但没有被 validation 选中，不能把本轮写成 exact Piao/NLS 成功。

下一步唯一建议：做 **feature-fusion / hybrid neural model**，保留 20.77 neural path 负责 L/W/D 与 mask/profile，把 F1/F2 这类稳定物理特征作为 curvature 辅助输入或辅助 head；不要做 baseline replacement，不要声称完整 Piao 复现，也不要直接扩到 480。

## 2026-05-25 更新：第 20.79 后的下一步

第 20.79 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 curvature-aware model refinement。本轮没有运行 COMSOL，没有生成或修改 NPZ，没有更新 `CURRENT_BASELINE.md`，也没有把 refined model 写成 baseline。

核心判断是：`C1_split_heads` 被 validation-only selection 选中，但 test 指标相对 20.77 reference 退化。20.77 reference test normalized MAE 为 `0.678014`、curvature MAE 为 `0.201076`、projected mask Dice 为 `0.847727`；20.79 selected refined model test normalized MAE 为 `0.753387`、curvature MAE 为 `0.211584`、projected mask Dice 为 `0.834597`。`wLW` 和 `W_m` 有轻微改善，但 `L_m`、`D_m`、`wLD`、`wWD` 和 profile depth RMSE 退化，因此不能升级 benchmark candidate。

下一步唯一建议：保留第 20.77 的 v3_240 benchmark candidate，不采用 20.79 refined model；优先做 **exact Piao / NLS-inspired feature pipeline** 作为 curvature 诊断和 comparator，其次再考虑 curvature-targeted data top-up。不要把本轮结果写成 baseline replacement。

## 2026-05-25 更新：第 20.78 后的下一步

第 20.78 已完成 formal true 3D RBC benchmark candidate audit。本轮没有运行 COMSOL、没有生成或修改 NPZ、没有重新训练模型，也没有更新 `CURRENT_BASELINE.md`。审计结论是：`comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 可以进入 **formal benchmark candidate**，但必须带 curvature risk，且明确不是 baseline。

核心分界点是 curvature：v3_240 的 neural test normalized MAE `0.678014` 优于 feature comparator `0.715395`，L/W/D MAE 为 `1.892/2.186/0.800 mm`，D_m、projected mask Dice 和 profile depth RMSE 都较 N=112 改善；但 `wLD/wWD/wLW` 仍不稳定，boxy / sharp 最差，且出现 Dice `0.956750` 但 curvature error `0.364948` 的样本，说明 2D projected mask 已不足以评价 true 3D profile curvature。

下一步唯一建议：进入 **model refinement for formal benchmark candidate**，先做 curvature-aware model/head/loss、stronger Bx/By/Bz sequence encoder，以及 exact Piao / NLS-inspired feature diagnostic。curvature-targeted data top-up 是第二选择；不要把第一步直接设成扩到 480，也不要做 baseline replacement。

## 2026-05-25 更新：第 20.77 后的下一步

第 20.77 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 的 true 3D RBC training gate。本轮通过 `dataset_id + COMSOL_DATA_REGISTRY.md + manifest` 显式加载 v3_240，没有 latest/newest NPZ 自动扫描，没有运行 COMSOL，没有生成或修改数据，也没有更新 `CURRENT_BASELINE.md`。输入为 `delta_b=(240,3,3,201)`，Conv1D 输入为 `(240,9,201)`，split=train/val/test 162/39/39。

结果是 **promising benchmark candidate, not baseline**：feature sanity comparator selected `svr_rbf_C10`，test normalized MAE 为 `0.7154`；neural gate 三个 seed 完成，validation 选择 seed `42`，test normalized MAE 为 `0.6780`，优于 mean baseline `0.9127` 和 feature comparator `0.7154`。相对第 20.75 N=112，L/W/D MAE 改善为 `1.89/2.19/0.80 mm`，D_m 从 `1.106 mm` 改善到 `0.800 mm`，projected mask Dice 从 `0.8364` 到 `0.8477`，profile depth RMSE 从 `0.000548 m` 到 `0.000388 m`。但 curvature 参数 `wLD/wWD/wLW` 仍不可稳定学习，curvature MAE 相对 N=112 从 `0.1905` 退到 `0.2011`。

下一步唯一建议：进入 **formal true 3D RBC benchmark candidate / model refinement**，但继续把 dense mask baseline 只作为 comparator，不做 baseline replacement。下一阶段应固定 registry/manifest gate，优先处理 curvature learnability：可以比较更强的 Conv1D/Transformer-style sequence encoder、curvature-aware loss 或 Piao-inspired NLS/LS-SVM 特征；不要直接把 v3_240 模型写入 `CURRENT_BASELINE.md`。

## 2026-05-25 更新：第 20.75 后的下一步

第 20.75 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v2_120` 的 true 3D RBC training gate。本轮只用 registry + manifest 显式加载数据，不运行 COMSOL、不生成或修改 NPZ、不训练正式 baseline、不更新 `CURRENT_BASELINE.md`。v2_120 的输入为 `delta_b=(112,3,3,201)`，Conv1D 输入为 `(112,9,201)`，split=train/val/test 76/18/18。

结果是 promising but not baseline：feature sanity validation 选择 `svr_rbf_C10`，test normalized MAE 为 `0.7677`；neural gate 三个 seed 全部完成，validation 选择 seed `42`，test normalized MAE 为 `0.7039`，优于 mean baseline `0.8803` 和 feature baseline `0.7677`。相对 20.73 N=56，neural test MAE 从 `0.7601` 降到 `0.7039`，L/W/D MAE 改善到 `2.51/2.59/1.11 mm`，curvature MAE 从 `0.2095` 降到 `0.1905`，projected mask Dice 从 `0.8347` 到 `0.8364`。但 `wLD/wWD/wLW` 仍不稳定，N=112 仍不足以写成 baseline。

下一步唯一建议：扩展 true 3D RBC dataset 到 240，再用同一套 `dataset_id + manifest + registry` gate 重跑 training gate。不要先更新 baseline，也不要把 v2_120 自动接入 mainline training；dense mask baseline 继续只作为 comparator。

## 2026-05-25 更新：第 20.74 后的下一步

第 20.74 已把 true 3D RBC imported-watertight 数据集从 v1 assembled N=56 扩展到 `comsol_true_3d_rbc_imported_watertight_pilot_v2_120`。实际 assembled N=112，split=train/val/test 76/18/18，curvature coverage 为 sharp=22、round=23、boxy=23、LD_dominant=24、WD_dominant=20；NPZ/schema validation、registry validation、manifest 和 Claude Code review 均通过。状态是 `pilot_generated`、`train_ready_candidate=True`、`baseline_ready=False`，不是 baseline。

下一步唯一建议：进入 **true 3D training gate on v2_120**。训练/评估必须通过 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v2_120`、`COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json` 显式加载，禁止 latest/newest NPZ 自动扫描；不更新 `CURRENT_BASELINE.md`，dense mask baseline 继续只作为 comparator。如果 v2_120 训练后 WD_dominant、deep/elongated 或 curvature 参数仍不稳，再做第二波 targeted top-up。

## 2026-05-24 更新：第 20.72 后的下一步

第 20.72 已把 20.71 的 partial pilot 补齐为 assembled true 3D RBC-style pilot pack candidate：assembled dataset_id 为 `comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`，N=56，split = train/val/test 36/10/10，curvature coverage 包含 sharp、round、boxy、`LD_dominant`、`WD_dominant`，NPZ/schema validation、registry validation 和 Claude Code review 均通过。

下一步唯一建议：进入 **true 3D training gate**，但这仍是 pilot training，不是 baseline replacement。训练/评估脚本必须通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json` 显式读取 dataset_id，禁止 latest/newest 自动扫描；`CURRENT_BASELINE.md` 不更新，dense mask baseline 继续只作为 comparator。

## 2026-05-24 更新：第 20.71 后的下一步

第 20.71 已生成 true 3D RBC-style imported-watertight pilot pack，但状态是 `partial_pilot_generated`。当前有效样本为 30，split = train/val/test 20/5/5；所有成功样本使用 `imported_watertight_mesh_solid`、20.70 material/domain fix、default solver、`mesh_auto_size=5`、`Jscale=1.0`，真实导出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b` 和 NPZ/schema validation 均通过。

当前不能进入 training gate：inventory 虽已完整覆盖 60 行，但只有 30 pass、2 fail、28 not_attempted；`LD_dominant` 和 `WD_dominant` 两个 curvature family 尚无成功样本。Registry / manifest 已建立，`allowed_use=schema_validation, explicit_pilot_training_gate`，`forbidden_use=automatic_mainline_training, baseline_update, current_baseline_replacement`；该 pack 不是 baseline，`CURRENT_BASELINE.md` 不更新。

下一步唯一建议：做 20.71 top-up generation，优先补齐 `LD_dominant` / `WD_dominant` not_attempted 样本，并复查 `deep_elongated` timeout 样本。top-up 通过后再重新验证 manifest / registry / split / curvature coverage，才讨论 explicit true 3D training gate。

## 2026-05-24 更新：第 20.70 后的下一步

第 20.70 已把 20.69 的 imported watertight solid 路线从 geometry gate pass 推进到 full-source forward smoke pass。原始 blocker 不是 Python mesh 或 Boolean subtract，而是 imported solid 后的 domain/material selection：COMSOL 未暴露稳定 cavity domain selection，且原始 air selection 与 steel selection 重叠。采用最小 selection/material fix 后，`material_domain_fixed` 在 `mesh_auto_size=5`、default solver、`Jscale=1.0` 下通过 defect stationary solve。

当前 forward-ready 证据是：`medium_round` one-sample imported watertight solid 已真实导出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验误差为 `0.0`，NPZ/schema validation 通过，`selected_solver_protocol=default`，没有 direct solver 依赖，也没有使用 high-layer fallback。

下一步唯一建议：进入 **smooth/mesh-based true 3D RBC pilot generation design**，先做小规模 pilot 计划与 mesh/material selection QA，而不是回到 high-layer approximation、2D profile-forward 小修或训练 surrogate。`CURRENT_BASELINE.md` 仍不更新，dense mask baseline 继续只作为 comparator；generated NPZ、temp STL、raw CSV、`.mph` 等仍不能提交。

## 2026-05-24 更新：第 20.69 后的下一步

第 20.69 已完成 watertight imported solid builder hardening。本轮不训练模型、不进入 pilot、不更新 `CURRENT_BASELINE.md`，也不创建或修改 COMSOL baseline 文档。`medium_round` 的 RBC-style depth map 已由 pure NumPy 生成 watertight closed STL：`mesh_units=m`，top cap 位于 `z=0`，bottom surface 使用 `z=-depth`，defect void 嵌入 steel 且与 steel surface 相交；mesh validation 通过，`is_watertight=True`，`nonmanifold_edges_count=0`，`zero_area_triangles_count=0`，`volume_m3=1.2918e-07`，`max_depth_m=0.0025`。

COMSOL 侧结果比 20.68 更进一步：known prism sanity probe 通过；RBC watertight STL 的 `import_success=True`、`repair_success=True`、`form_solid_success=True`、`imported_domain_count=1`、`boolean_subtract_success=True`、`steel_notched_domain_count=1`、`mesh_precheck_success=True`。这说明 20.68 的 imported mesh Boolean empty steel domain blocker 已被推进到 geometry gate 通过。

但 one-sample forward smoke 只完成到 no-defect solve；defect model 的 stationary solver 不收敛，未生成 `true_3d_imported_watertight_forward_smoke_v1.npz`，因此没有运行 NPZ/schema validator，也没有 `delta_b`。当前 route decision 是 `C_import_boolean_pass_forward_not_run_or_failed`：imported watertight solid geometry route 技术上可行，但尚不是 pilot-ready。下一步唯一建议是先修 COMSOL imported solid 的 solve / mesh-quality / solver robustness，再考虑 smooth/mesh-based true 3D RBC pilot；不要扩样、不要训练、不要回退到 2D profile-forward 小修。dense mask baseline 仍只作 comparator。

## 2026-05-24 更新：第 20.68 后的下一步

第 20.68 已完成 smooth / near-smooth true 3D variable-depth builder completion feasibility。本轮没有训练模型、没有进入 pilot、没有生成 20.68 forward NPZ、没有更新 `CURRENT_BASELINE.md`，也没有创建或修改 COMSOL baseline 文档。bounded geometry probe 的结论是：`lofted_contour_solid`、`stacked_workplane_contour_loft`、`interpolated_surface_solid`、`imported_closed_mesh_solid` 均未形成可进入 forward 的 smooth / near-smooth candidate；唯一通过的是 `high_layer_control_24`。

当前状态必须写成 `high_layer_control_pass`，不能写成 `variable_depth_pass` 或 `near_smooth_pass`。`high_layer_control_24` 比 20.67 的 12-layer control 更进一步，记录了 24 个 depth levels，且 `closed_body_success=True`、`boolean_subtract_success=True`、`mesh_precheck_success=True`、`spatial_depth_variation=True`、`is_constant_depth=False`；但它仍是 stepped/high-layer control，不是 smooth RBC surface，也不是 exact Piao RBC geometry。

下一步唯一需要人工确认：是否接受 high-layer approximation 作为后续 pilot 口径。如果接受，可以设计 true 3D RBC pilot，但所有文件必须显式标注 `high_layer_approximation`，不能写成 smooth / near-smooth；如果不接受，则继续修 COMSOL smooth / closed-surface builder，优先诊断 imported mesh Boolean empty steel domain 或寻找可用的 closed-surface / convert-to-solid 路径。dense mask baseline 仍只作 comparator。

## 2026-05-23 更新：第 20.66 后的下一步

第 20.66 已完成 true 3D RBC-style smoke pack generation。本轮只做 smoke pack generation 和 schema validation：没有训练 forward surrogate 或 inverse model，没有做 refinement，没有更新 `CURRENT_BASELINE.md`，也没有创建或修改 COMSOL baseline 文档。Claude Code review 完成且无 must-fix。

当前通过状态是 `stepped_depth_smoke_pass`。Stage A 生成 6 个 RBC-style single-defect samples，`L_m=0.010-0.030`、`W_m=0.006-0.020`、`D_m=0.001-0.006`，pure-Python depth/profile/mask validation 6/6 通过；Stage B 真实 COMSOL forward 6/6 通过，输出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验通过；Stage C NPZ schema validation 6/6 通过。

边界必须写清：本轮没有通过 smooth true variable-depth RBC solid。COMSOL 几何实现是 5 层 `stepped_depth_layered_approximation`，`smooth_variable_depth_solid_verified=False`，`stepped_depth_approximation=True`，`constant_depth_extrusion_used_as_success=False`。RBC generator 也标记为 `exact_piao_rbc=False`，属于 RBC-style / RBC-inspired engineering approximation，不是完整复现 Piao 2019。

下一步唯一建议：先做路线决策，不要直接把 20.66 写成 smooth 3D pilot ready。需要在两个选项中选择：继续改 COMSOL smooth variable-depth geometry，或明确接受 stepped-depth 作为 20.67 pilot approximation 后再设计 60-sample pilot。dense mask baseline 仍只作为 comparator。

## 2026-05-23 更新：第 20.65 后的下一步

第 20.65 已完成 true 3D / Piao-style geometry profile feasibility design。本轮是 design-only：没有运行 COMSOL、没有生成数据、没有训练 surrogate / inverse model、没有做 refinement，也没有更新 `CURRENT_BASELINE.md` 或任何 COMSOL baseline 文档。Claude Code review 通过且无 must-fix。

当前判断是：20.61-20.64 已足以暂停 2D profile-forward 小修。single-height Bz、multi-height Bz、same-direction Bx/By/Bz、multi-direction excitation 都没有让真实 COMSOL oracle residual 稳定排序 profile quality；继续训练 2D profile surrogate 或继续 refinement config 微调没有主线价值。下一步主线切换为 **true 3D / Piao-style geometry profile**，dense mask baseline 只作为 comparator。

第 20.66 的唯一推荐任务是 small smoke，不是正式数据集：验证 `RBC params -> depth map -> COMSOL variable-depth defect solid -> same-source projected mask -> Bx/By/Bz @ sensor_z_m=0.008 -> delta_B = B_defect - B_no_defect`。当前只能写成 COMSOL 支持真实 3D volume solve；RBC / variable-depth true 3D profile generation 尚未验证，是 20.66 的核心 blocker。不要把 20.66 smoke 扩成 multi-height，`0.012m` 只作为后续 pilot / ablation 设计保留。

如果 20.66 无法构建 variable-depth true 3D solid，应暂停 geometry-forward route，先解决 COMSOL geometry blocker；如果 smoke 通过，再进入小规模 3D pilot。未来 20.67 的 IoU/Dice/profile-error 阈值目前只是 preliminary acceptance guidance，不是已验证硬标准。

## 2026-05-23 更新：第 20.64 后的下一步

第 20.64 已完成 multi-direction excitation profile perturbation oracle ordering feasibility POC。本轮只做 oracle residual audit，不训练 surrogate、不做 refinement、不更新 baseline。COMSOL pack 覆盖 12 base samples / 96 profile rows / 3 directions / 3 axes；`direction_0` 复用同 geometry 的 20.63 default-direction rows，`direction_45` 和 `direction_90` 使用真实 COMSOL forward，并通过 `ExternalCurrentDensity.Je` 设置真实改变 excitation / magnetization direction。direction probe 显示 `direction_90` 相对 `direction_0` 的 no-defect / defect response NRMSE 为 `1.6479 / 1.7981`，因此不是数组旋转或信号伪造。

结果没有通过 gate。same-pack test 中，`direction_0` Bz-only ordering = `0.4545`，`direction_90` Bz-only = `0.5273`，multi-direction Bz train-std normalized = `0.5636`，说明 Bz-only multi-direction 有边际正信号；但 all-axis normalized ordering 只有 `0.3455`，mismatch_rate = `0.6545`，residual-error correlation = `-0.8028`，明显劣于 same-pack default-direction Bz。Claude Code review 通过且无 must-fix，同时指出 Bz-only 正信号仍不稳定，test base 太少，不能支撑 surrogate 训练。

当前下一步唯一优先级：**true 3D profile / Piao-style route**。不要进入 20.64 的 multi-direction surrogate training，也不要回到 profile-forward refinement；第 20.64 只说明改变 excitation direction 比同方向 multi-axis / multi-height 更有一点信号，但仍未证明 richer direction observation 可以稳定缓解 profile residual non-identifiability。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.63 后的下一步

第 20.63 已完成 multi-axis MFL profile perturbation oracle ordering feasibility POC。本轮只做 oracle residual audit，不训练 surrogate、不做 refinement、不更新 baseline。multi-axis pack 覆盖 24 base samples、192 profile rows、3 个 field axes `[Bx, By, Bz]`，共 576 个 axis observations；所有行包括 `true_reference` 均由真实 COMSOL forward 生成，未复用旧 Bz-only 数组。实际 COMSOL expressions 为 `[mf.Bx, mf.By, mf.Bz]`，expression probe 通过，`delta_B = B_defect - B_no_defect` 三轴校验通过。

结果是否定性的：test Bx-only oracle ordering = `0.4505`，By-only = `0.4955`，Bz-only = `0.4505`，Bx+By+Bz train-std normalized ordering = `0.4505`，mismatch_rate = `0.5495`，residual-error correlation = `0.0242`。multi-axis 没有超过同 pack 的 Bz-only，也没有超过 20.61 single-height Bz oracle test reference `0.5030`，未达到 `>0.65` 或 `+0.10` improvement gate。

当前下一步唯一优先级：**multi-direction excitation / richer scan geometry**。不要训练 multi-axis profile surrogate，也不要回到 profile-forward refinement retry；因为真实 COMSOL oracle residual 在 same-liftoff Bx/By/Bz 下仍不能稳定排序 profile quality。若继续 forward-guided route，应改变激励方向、扫描方向或 scan geometry；若不扩展观测物理，则应暂停 profile-forward residual route。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.62 后的下一步

第 20.62 已完成 multi-height Bz profile perturbation oracle ordering feasibility POC。本轮只做 oracle residual audit，不训练 surrogate、不做 refinement、不更新 baseline。multi-height pack 覆盖 12 base samples、96 profile rows、3 个 sensor_z heights `[0.004, 0.008, 0.012]`，共 288 个 height observations；其中 0.008m 的 96 个 observation 复用第 20.61 exact rows，0.004m 和 0.012m 共 192 个 observation 由真实 COMSOL forward 生成。profile polygon geometry 有效，`delta_bz = bz_defect - bz_no_defect` 校验通过。

结果是否定性的：test single-height 0.008m oracle ordering = `0.4909`，0.004m = `0.4364`，0.012m = `0.4545`，multi-height train-std normalized ordering = `0.4545`，mismatch_rate = `0.5455`，residual-error correlation = `-0.5920`。multi-height 没有超过 20.61 single-height oracle test reference `0.5030`，也没有达到 `>0.65` 或 `+0.10` improvement gate。

当前下一步唯一优先级：**multi-axis / multi-direction observation**。不要训练 multi-height profile surrogate，也不要回到 profile-forward refinement retry；因为真实 COMSOL oracle residual 在 multi-liftoff Bz 下仍不能稳定排序 profile quality。若继续 forward-guided route，应改变观测信息维度，例如不同扫描方向、横向 scan lines / components 或更丰富观测，而不是继续小调当前 profile surrogate、refinement loss 或单纯扩大同类 lift-off 数据。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.61 后的下一步

第 20.61 已完成 expanded profile perturbation forward pack + profile-compatible surrogate calibration POC。expanded pack 达到 target：36 base samples、288 rows，其中 `reused_original_rows=36`、`reused_from_20_60_rows=0`、`real_comsol_forward_rows=252`；split = 192/48/48，rect/rot = 144/144，8 类 profile variant 各 36 行。COMSOL 侧 profile polygon geometry 有效，`delta_bz = bz_defect - bz_no_defect` 校验通过。

selected surrogate 为 `EPPF1_profile_station_mlp`，waveform val/test NRMSE/correlation = `0.3314 / 0.9435` 和 `0.3735 / 0.9299`。相比 20.60，test surrogate ordering 从 `0.2143` 提升到 `0.5569`，mismatch_rate 从 `0.7857` 降到 `0.4431`，说明扩大 profile perturbation data 缓解了最严重的 test collapse；但 val/test surrogate ordering 仍低于可用 gate，且 COMSOL oracle residual ordering train/val/test 仅 `0.4471 / 0.5120 / 0.5030`，接近随机。

当前下一步唯一优先级：**richer observations / multi-height / multi-axis 或 non-identifiability audit**。不要直接回到 profile-forward refinement retry，也不要继续小调当前 profile surrogate architecture / loss；因为 oracle residual 本身不能可靠排序 profile quality，继续优化 surrogate 很难把不可辨识的 residual objective 变成可用 refinement 梯度。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.60 后的下一步

第 20.60 已完成 profile perturbation forward pack + profile-compatible surrogate calibration POC。按修正版 row-count gate，COMSOL 侧生成 minimum partial pack：`total_rows=96`，`reused_original_rows=12`，`real_comsol_forward_rows=84`，represented base samples = 12，split = 64/16/16，rect/rot = 48/48，8 类 profile variant 各 12 行。`true_reference` 行只作为 residual ordering anchor 复用 pilot_v9 原始数组，不计入真实 COMSOL forward rows；真实生成行使用 profile polygon geometry，delta check 通过。

profile-native surrogate 的 waveform fit 可以接受但 residual ordering 不足。validation 选中 `PPF1_profile_station_mlp`，val/test NRMSE/correlation 为 `0.4396 / 0.8990` 和 `0.3758 / 0.9274`；但 oracle residual ordering val/test 只有 `0.6786 / 0.5357`，selected surrogate residual ordering 为 `0.6607 / 0.2143`，mismatch_rate 为 `0.3393 / 0.7857`。因此第 20.60 没有进入 profile-forward refinement，也不更新任何 baseline。

当前下一步唯一优先级：**扩 profile perturbation data**。需要优先增加 base sample 覆盖，尤其是 val/test base 数，重新检查 oracle residual 是否能稳定排序 profile quality；若 oracle ordering 仍弱，则应转向 richer observations / multi-axis / multi-height 或保留 no-forward profile basis，不应继续对当前 profile-forward surrogate 做小幅架构或 loss 微调。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.58 后的下一步

第 20.58 已完成 mask/profile basis refinement POC。profile extraction 从 predicted dense mask/probability 中提取 K=8 profile 表示，validation 选择 `P1_hardmask_profile`；profile-extracted test IoU/Dice/area_error 为 `0.6589 / 0.7921 / 0.2170`。no-forward profile refinement 只拟合 dense initial probability 并加 smoothness / area / bounds prior，test 提升到 `0.6697 / 0.8002 / 0.2196`，说明 profile basis 相比第 20.57 的 single rotated-box refinement 更稳，但没有稳定超过第 20.54 extracted rotated-box proposal `0.6726 / 0.8017 / 0.1945`。

forward profile refinement 已执行受控 sweep，但 validation 选择 `lambda_forward=0.0`，test 为 `0.6620 / 0.7938 / 0.2243`。这说明当前第 20.56/20.57 的 S1 surrogate 通过 lossy profile-to-rect summary 接入后，不能作为可靠的 profile-space forward consistency 约束。Claude Code review 通过且无必须修复；审查结论是不建议继续在当前 surrogate-dependent profile refinement 上小调。

当前下一步唯一优先级：**改进 profile-compatible forward surrogate**。如果继续 profile/basis 路线，应先让 forward model 直接接受 profile/basis 或 rasterized-profile derived features，而不是把 profile 压回单个 rect/rot summary；否则应暂停 geometry/refinement route，等待更丰富观测或更强 forward data。仍不更新 `CURRENT_BASELINE.md`，也不创建新的 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.57 后的下一步

第 20.57 已完成 perturbation-calibrated surrogate 的受控 Priewald-style refinement retry。`S1_perturb_geom_mlp` 按第 20.56 protocol 重训于内存中，recovery 指标与 20.56 对齐：val/test waveform NRMSE 为 `0.3666 / 0.4289`，residual ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`。

但是连续低维 refinement 没有通过 gate。validation 上 8 个 config 全部导致 mask 指标退化或 mismatch 过高，最终仅选最高分 config 做 diagnostic：`steps=50, lr=0.003, lambda_prior=0.10`。test geometry-raster IoU/Dice/area_error 从 `0.6726 / 0.8017 / 0.1945` 变为 `0.6492 / 0.7829 / 0.2417`；forward NRMSE 下降 `0.0713`，但 mismatch_rate 为 `0.6212`，residual reduction 与 IoU/Dice delta 相关性为 `-0.1824 / -0.2250`。

当前判断：20.56 的 pairwise residual ordering 改善没有转化为可用的连续 geometry optimization 梯度。不要继续在当前 rect/rot low-dimensional refinement objective 上小调 steps / lr / prior；也不要回到 direct geometry head 或 dense baseline patch。最近下一步优先转向 **mask/profile basis refinement**，降低对 single rect/rot parameter residual landscape 的依赖。若未来重新尝试 Priewald-style refinement，应先扩大 perturbation pack 或加入 richer observations，再重新验证 residual landscape。

## 2026-05-22 更新：第 20.56 后的下一步

第 20.56 已生成小规模 local geometry perturbation forward-calibration pack，并完成 surrogate residual ordering audit。实际 COMSOL pack 是 96 行 partial pack（12 个 base，train/val/test = 64/16/16，rect/rot = 48/48），84 行为真实 COMSOL forward，12 行 true reference 复用原始 NPZ；`delta_bz = bz_defect - bz_no_defect` 校验通过。

关键结论是：COMSOL oracle residual 的 val/test ordering accuracy 为 `0.6607 / 0.8393`，选中的 `S1_perturb_geom_mlp` surrogate 的 val/test ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`，较 20.55 明显改善。这说明 perturbation forward data 对 surrogate mismatch 有帮助，下一步可以回到 **controlled Priewald-style refinement retry**，但必须继续把它作为 POC/candidate，不更新 baseline。

限制也很明确：当前 pack 只有 96/192 行，且 selected surrogate 的 test residual-error correlation 仍为负（`-0.0462`）。因此下一步不要直接扩大为正式路线，也不要继续训练新的 direct geometry head；应先用 perturbation-calibrated surrogate 做一次受控 refinement retry，观察 residual ordering 是否能转化为 mask / geometry 改善。如果 retry 仍出现 residual 下降但 mask 退化，则优先扩 perturbation data 或转向 mask/profile basis refinement。

## 当前状态

`CURRENT_BASELINE` 仍以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准：

- v3_complex mask-only grid decoder + forward consistency
- `lambda_forward = 0.10`
- validation-selected probability threshold = `0.80`
- forward surrogate = `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`

该 baseline 是当前 v3_complex 上最强 boundary-oriented baseline，但 polygon / rotated_rect 精细边界圆斑化、small / low-signal、多缺陷仍未根本解决。

内部 decoder / loss / threshold / geometry / basis / refinement 小修补已经基本收口。当前第 20 阶段已经从单纯 COMSOL 数据包验证，进一步进入 geometry-aware / forward-consistent inverse reconstruction 方法验证。当前这些工作仍是 COMSOL data-domain POC / candidate，不更新 v3_complex `CURRENT_BASELINE`。

## 最近下一步

当前不要继续普通 dense decoder patch，也不要继续单独 geometry head 小修补。第 20.48-20.55 已证明：

1. geometry labels 和 differentiable rotated-rectangle rasterizer 没有 blocker；
2. direct delta_bz-only geometry head 的 type / angle 学习不足；
3. controlled architecture sweep 没有找到有效 head 结构；
4. feature-assisted geometry head + lightweight forward consistency 只带来边际 mask / angle 改善，type confusion 仍是主因；
5. Priewald-style low-dimensional refinement 能降低 forward residual，并可小幅改善 geometry-raster mask，但 initializer / proposal 质量决定上限；
6. dense/coarse mask initializer + PCA bbox extraction 在 20.53 中没有超过 20.51 geometry-head proposal，type / angle proposal 仍弱；
7. 第 20.54 的 strong dense initializer 和 improved proposal extraction 已把 rect/rot geometry proposal 提到 test IoU/Dice `0.6726 / 0.8017`，但 Priewald-style refinement 让 test IoU/Dice 回落到 `0.6646 / 0.7958`，同时 forward NRMSE 下降，说明当前主要 blocker 已从 proposal quality 转为 forward surrogate mismatch；
8. 第 20.55 的 calibrated surrogate sweep 没有找到可用 residual objective：S2 的 waveform NRMSE 最好，但 val residual-error correlation 为负，S3 的正相关也只有 `0.0215`，未过 gate，因此 calibrated refinement 被正确跳过。

因此最近下一步优先转向：

1. **生成 synthetic perturbation forward data / 局部扰动校准数据**：当前缺少同一 geometry 附近的已知扰动与 forward response，surrogate 学不到“几何越差 residual 越高”的局部单调关系。
2. 如果不能生成扰动 forward 数据，则转向 **mask/profile basis refinement**，减少对当前低维 rect/rot geometry residual objective 的依赖。
3. 暂停继续对现有 surrogate loss、peak weighting 或 refinement objective 做小调；20.55 已说明 waveform 拟合不等于 residual 可用于 geometry refinement。
4. 继续保持 train-only normalization、validation checkpoint / threshold selection、test-only final evaluation。

如果后续 refinement 不能在更强 proposal 上稳定改善 mask / geometry，再暂停 rect/rot geometry route，等待更丰富观测、更多通道或更强 forward surrogate；当前不建立新 baseline。

## 当前不要继续的方向

不要继续围绕现有 v3_complex grid decoder 做 selection metric、ensemble、threshold trick、loss trick、decoder head、SDF / boundary head、coordinate refinement、hand-crafted Bz features、U-Net-like decoder、shape-type conditional、star-convex、retrieval、box / quad / basis / profile 或 mask-logit refinement 小修补。

也不要继续单独调 rect/rot neural geometry head。新的实验必须回答：显式 geometry representation、differentiable rasterization 和 forward residual 是否能稳定提高边界反演可辨识性，而不是只带来局部指标波动。
## 2026-05-22 更新：第 20.59 后的下一步

第 20.59 已完成 profile-compatible forward surrogate POC。preflight 结论是该方向符合 Priewald-style forward-model-based inversion / refinement 路线，但必须先证明 profile-native residual 能在 validation 上稳定排序 geometry/profile quality，不能再使用把 profile 压缩为单个 rotated box 的旧 surrogate bridge。

本轮构建 original profile-forward dataset（rect/rot N=400，split=268/66/66）和 perturb profile-forward dataset（20.56 partial pack N=96，split=64/16/16），并训练 3 个 profile-compatible surrogate。validation 选中 `PFS3_profile_station_sequence`，其 waveform val/test NRMSE 为 `0.3841 / 0.3995`，但 validation ordering accuracy 仅 `0.6607`，mismatch_rate 为 `0.3393`，未通过 usable-surrogate gate。因此 profile-forward refinement retry 被跳过。

当前下一步唯一优先级：**扩展 profile perturbation data**。如果扩展后的 validation ordering / mismatch gate 仍不通过，则不再继续 forward-guided profile refinement 小调，改回 no-forward profile basis 或等待 richer observations / multi-axis data。当前仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

---

## 第 20.67 后下一步

20.67 的结论是 `high_layer_pass`：`medium_round` 的 12-layer high-layer approximation 通过了 geometry-only gate、Bx/By/Bz one-sample forward 和 NPZ/schema validation，并且明确区别于 20.66 的 5-layer stepped-depth smoke。

但本轮没有证明 smooth variable-depth true 3D geometry 可行：limited smooth / loft / imported closed-surface probe 没有形成 verified closed defect body，最终不是 `variable_depth_pass`，也不是 `near_smooth_pass`。因此下一步不能直接进入 60-sample true 3D RBC pilot，除非人工确认接受 high-layer approximation 作为 pilot approximation。

唯一推荐下一步：
- 如果接受 high-layer approximation：进入小规模 true 3D RBC pilot plan/generation，但所有文件和结论必须标注 `high_layer_approximation`，不得写成 smooth RBC。
- 如果不接受 high-layer approximation：继续修 smooth / closed-surface COMSOL geometry builder，不扩样、不训练。

仍然不更新 `CURRENT_BASELINE.md`；dense mask baseline 只作为 comparator。
## 2026-05-24 更新：第 20.73 后的下一步

第 20.73 已完成 true 3D RBC pilot training gate。它不是 baseline，也不更新 `CURRENT_BASELINE.md`；所有训练/评估都通过 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`、`COMSOL_DATA_REGISTRY.md` 和 manifest 显式加载，没有 latest/newest NPZ 自动扫描，也没有运行 COMSOL 或生成新数据。

训练 gate 的核心结论是：模型能拟合 train，但 N=56 的泛化证据不足。小型 Conv1D 在完整训练轨迹中可把 train normalized MAE 降到 `0.0012`，说明链路能学习训练样本；但 validation 选择的 checkpoint 在 test 上 normalized MAE 为 `0.7601`，只优于 mean baseline `0.8598`，没有超过 Piao-inspired feature baseline `0.7564`。当前可学习信号主要在 `L_m`、`W_m`，`D_m` 边缘，`wLD/wWD/wLW` 三个 curvature 参数仍不可稳定辨识。

唯一下一步建议：扩展 true 3D RBC dataset 到 120/240 量级，并把 validation set 扩到至少 20-30 个样本，再重新跑同一套 registry/manifest-gated training gate。不要先调大模型、不要更新 baseline、不要回到 dense mask 主线；dense mask baseline 继续只作 comparator。
## 2026-05-25 更新：第 20.76 后的下一步

第 20.76 已把 true 3D RBC imported-watertight dataset 从 v2_120 扩展到 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`。本轮没有训练、没有 baseline、没有更新 `CURRENT_BASELINE.md`；v2_120 source pack 未覆盖，v3 top-up 和 assembled NPZ 仍是 generated data，不提交。

当前 v3_240 状态是 `pilot_generated` 且 `train_ready_candidate=True`：N=240，split=162/39/39，curvature coverage=sharp 48 / round 49 / boxy 47 / LD_dominant 46 / WD_dominant 50，schema/registry/manifest validation 全部通过，baseline_ready=False。下一步唯一建议是执行 true 3D training gate on v3_240，必须通过 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` + manifest + `COMSOL_DATA_REGISTRY.md` 显式加载，禁止 latest/newest NPZ 自动扫描；重点检查 `D_m` 和 `wLD/wWD/wLW` 是否相对 v2_120 进一步稳定。

