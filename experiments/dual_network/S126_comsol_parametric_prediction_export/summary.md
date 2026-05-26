# S126 COMSOL parametric prediction export

## 目的

S126 为 `train_comsol_parametric_inverse.py` 增加 `--export-predictions`，用于导出每个 split、每个 sample、每个 component slot 的 prediction 与 target。该阶段用于支撑 S127 的 grouped diagnostics，并避免只看 aggregate metrics。

## 产物

- `s115_raw_mlp_export/train_predictions.csv`
- `s115_raw_mlp_export/val_predictions.csv`
- `s115_raw_mlp_export/test_predictions.csv`
- `s115_raw_mlp_export/train_prediction_mask_metrics.csv`
- `s115_raw_mlp_export/val_prediction_mask_metrics.csv`
- `s115_raw_mlp_export/test_prediction_mask_metrics.csv`

component-level CSV 包含 `presence_true`、`presence_prob`、`presence_pred`、`type_true`、`type_pred`、`center_x/y`、`axis_x/y`、`depth`、`rotation` 和对应误差。sample-level mask metrics 包含 `pred_mask_iou`、`pred_dice`、`oracle_mask_iou`、`oracle_gap`、`target_area`、`pred_area` 和 type sequence。

## S115 raw MLP export 复现指标

- train: presence accuracy = `1.000000e+00`, type accuracy = `1.000000e+00`, rotation MAE = `1.854687e-01`, mask IoU = `6.980716e-01`
- val: presence accuracy = `1.000000e+00`, type accuracy = `6.500000e-01`, rotation MAE = `7.731843e+00`, mask IoU = `3.699078e-01`
- test: presence accuracy = `1.000000e+00`, type accuracy = `6.666667e-01`, rotation MAE = `7.740397e+00`, mask IoU = `4.244624e-01`

这些结果与 S115 raw MLP baseline 对齐，说明新增 export 没有改变默认 fixed-order 训练行为。

## 自评

- prediction export 成功。
- 未生成模型权重、checkpoint 或图片。
- S128 review 指出 `permutation_min` 下 export 需要 matched slot；已修复为导出时按 matching mode 选择 target slot，并用 `matched_slot` 记录。
