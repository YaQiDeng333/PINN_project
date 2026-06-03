# NEXT_STEP

## 2026-06-03 after Stage 25.18 raster-target route reset

唯一下一步：**进入 25.19 geometry-primary component-set 设计与标签派生计划；不训练**。

25.18 已停止 per-component raster-target 主线，不再把 25.17b 作为主路线继续推进。证据已经足够：25.13 target-v2 产生 near-empty collapse，25.15 label-v3 产生 union-like merged collapse，25.17 label-v3b 的 merged rate 仍为 `1.000000`。这说明继续 label-v4 / loss-v5 / raster-target training 只是在同一个 target 层问题上打转。

下一阶段保留 component-set 方向、`K=3` slots、geometry prediction、raw labels 和后续 forward consistency，但主监督改为 geometry-primary slot 输出。mask/depth 只由几何 slot 派生，用于评价、弱监督或一致性检查，不能继续作为 per-component raster ownership 的主线。

边界：不训练，不运行 COMSOL，不生成或修改 data/NPZ，不更新 `CURRENT_BASELINE.md`，不做 baseline transition。