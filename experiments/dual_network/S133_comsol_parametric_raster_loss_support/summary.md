# S133 COMSOL parametric raster mask loss support

## 目的

S133 将 S132 differentiable soft rasterizer 接入 `train_comsol_parametric_inverse.py`，使 parametric inverse model 可以在参数监督之外可选接受 mask-level supervision。

## 新增参数

- `--lambda-raster-bce`，默认 `0.0`
- `--lambda-raster-dice`，默认 `0.0`
- `--raster-softness-cells`，默认 `1.0`
- `--raster-target-source masks|mu_threshold`，默认 `masks`

当 `lambda_raster_bce=0` 且 `lambda_raster_dice=0` 时不计算 raster loss，旧行为保持不变。

## raster loss 计算

启用 raster loss 时：

1. 使用模型输出的 `continuous`、`presence_prob` 和 `type_logits`；
2. 调用 `soft_rasterize_components` 得到 `soft_mask`；
3. target mask 来自 `masks > 0.5` 或 `mu_maps < 500`；
4. 计算 `soft_bce_loss` 和 `soft_dice_loss`；
5. 将 `lambda_raster_bce * raster_bce + lambda_raster_dice * raster_dice` 加到总 loss。

`training_history.csv` 记录 `raster_bce_loss`、`raster_dice_loss`、`raster_soft_iou` 和 `raster_soft_dice`。

## 当前边界

- raster loss 是 parametric mask supervision，不是 COMSOL forward consistency。
- 第一版仍把 `rectangular_notch` / `rotated_rect` 都按 rotated rectangle approximation rasterize。
- 默认关闭，避免影响 S115 raw baseline 兼容性。

## 自评

- raster loss smoke test 已覆盖 BCE + Dice enabled path。
- 默认参数保持旧行为。
- 下一步 S134 需要验证 raster supervision 是否改善 val/test mask IoU。
