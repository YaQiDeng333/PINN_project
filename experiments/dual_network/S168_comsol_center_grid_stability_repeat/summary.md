# S168 COMSOL center-grid stability repeat

## 目的

验证 S164 的 `lambda_center_grid=0.1` 是否是可重复的 center-localization improvement，而不是单次随机初始化波动。本阶段不新增 param-only training run，不测试其他 lambda，不重跑 `center_axis_relative`。

## Runs

| run_id | seed label | source | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
|---|---|---|---:|---:|---:|---:|
| existing_unrecorded | existing_unrecorded | S164 full probe, no CLI seed recorded | 0.469423 | 0.498874 | 5.996350 | 5.546025 |
| center_grid_seed1 | 1 | new S168 run | 0.485716 | 0.505590 | 5.443171 | 4.931658 |
| center_grid_seed2 | 2 | new S168 run | 0.446966 | 0.503713 | 6.732050 | 4.872537 |

## Early-stop handling

Seed1 did not trigger the early stop. Its val/test IoU were both above the historical param-only baseline, so seed2 was executed.

## 自评

S168 completed the planned two new seeds. Both new seeds preserved positive val/test IoU deltas and lower center grid error relative to S161, so the stage can proceed to S169 aggregate acceptance.
