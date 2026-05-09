# 变更日志

本文档记录项目文档和代码的重要变更。

---

## 2026-05-09：术语说明补充

**修改文件**：`术语说明.md`

**变更类型**：内容扩展

**变更内容**：新增 4 个章节，约 50 个术语，从 372 行扩展到约 590 行。

### 新增章节

| 章节 | 术语数量 | 主要内容 |
|---|---|---|
| 11. v3 / v4 复杂缺陷数据集相关术语 | ~25 | `v3_complex`、`v4_balanced_complex`、`rotated_rect`、`polygon`、`multi_defect`、`complexity_level`、`num_defects`、`num_vertices`、`mask_pixels`、`signal_snr`、`area_bin`、`seed`、`--dataset`、`--loss-type` 等 |
| 12. 进阶 Loss 相关术语 | ~10 | `weighted_mse` / `defect_weight`、`soft Dice Loss` / `lambda_dice`、`area-aware Loss` / `lambda_area`、`symmetric` / `over_only`、`--lambda-tv`、`--lambda-phy` |
| 13. 训练策略和实验概念 | ~12 | `漏检` / `pred_area=0`、`过分割`、`area_ratio`、`延长训练`、`oversampling`、`focal loss`、`forward model`、`Bz_pred`、`per-sample evaluation`、`seed` / `repeat`、`.pt` |
| 14. 最新推荐模型速查 | 7 条 | simple baseline、v3_complex baseline、v4 各专项候选模型的路径与选择原则 |

### 覆盖范围

新术语从以下文档中提取，用于填补此前术语说明的空白：

- `README.md`（第 7.5 到 7.17 步的记录）
- `EXPERIMENT_LOG.md`（第 7 步以后的实验记录）
- `PINN优化路线.md`（优化路线补充记录）
- `NEXT_STEP.md`（当前状态和下一步建议）
- `CURRENT_BASELINE.md`（baseline 记录和 v4 专项候选）
