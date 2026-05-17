import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import (  # noqa: E402
    BzEncoder,
    MFLDataset,
    project_path,
    set_seed,
    signal_shape_info,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    MaskBoundaryGridModel,
    aggregate_seed_rows,
    build_sample_rows,
    compute_pos_weight,
    get_area_edges,
    load_low_signal_indices,
    make_loader,
    mask_loss,
    predict_prob_maps,
    safe_nanmean,
    safe_nanstd,
    sample_mean_metrics,
    select_preview_samples,
    summarize_candidate,
    summarize_samples,
    threshold_matches,
)


TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

CURRENT_BASELINE_CHECKPOINTS = {
    42: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed42.pt',
    123: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed123.pt',
    2026: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed2026.pt',
}

SEEDS = [42, 123, 2026]
SCREENING_SEED = 42
DECODER_VARIANTS = ['unet_light', 'unet_residual']
EPOCHS = 50
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 1e-3
LATENT_DIM = 64
MASK_THRESHOLD_NORM = 0.5
TRAIN_SELECTION_THRESHOLD = 0.5
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
CURRENT_BASELINE_THRESHOLD = 0.90
POS_WEIGHT_CAP = 8.0
GRID_LOW_SHAPE = (10, 20)
LIGHT_BASE_CHANNELS = 48
RESIDUAL_BASE_CHANNELS = 64
POSITIVE_SIGNAL_AREA_TOLERANCE = 0.02
TYPE_METRIC_DROP_TOLERANCE = 0.02
TYPE_AREA_TOLERANCE = 0.05
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_boundary_unet_decoder_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_unet_decoder_candidate_metrics.csv'
SCREENING_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_unet_decoder_screening.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_boundary_unet_decoder_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'mask_boundary_unet_decoder_candidate'

METRIC_KEYS = [
    'iou',
    'dice',
    'area_error',
    'center_error',
    'pred_area_zero',
    'pred_area_lt_true',
    'pred_area_gt_true',
    'composite',
    'macro_area_composite',
]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels >= 32 else 4
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x):
        return F.silu(x + self.block(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        groups = 8 if out_channels >= 32 else 4
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        ]
        if residual:
            layers.append(ResidualBlock(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MaskBoundaryUNetDecoderModel(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channels=1,
        latent_dim=64,
        out_shape=(100, 200),
        variant='unet_light',
        low_shape=GRID_LOW_SHAPE,
    ):
        super().__init__()
        if variant not in DECODER_VARIANTS:
            raise ValueError(f'Unsupported decoder variant: {variant}')
        self.out_shape = tuple(out_shape)
        self.low_shape = tuple(low_shape)
        self.variant = variant
        self.base_channels = RESIDUAL_BASE_CHANNELS if variant == 'unet_residual' else LIGHT_BASE_CHANNELS
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        low_h, low_w = self.low_shape
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.base_channels * low_h * low_w),
            nn.GELU(),
        )

        if variant == 'unet_light':
            self.stem = nn.Sequential(
                nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, self.base_channels),
                nn.SiLU(),
            )
            self.up1 = UpBlock(self.base_channels, 48, residual=False)
            self.up2 = UpBlock(48, 24, residual=False)
            self.up3 = UpBlock(24, 16, residual=False)
            self.head = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 1, kernel_size=1),
            )
        else:
            self.stem = nn.Sequential(
                ResidualBlock(self.base_channels),
                ResidualBlock(self.base_channels),
            )
            self.up1 = UpBlock(self.base_channels, 64, residual=True)
            self.up2 = UpBlock(64, 32, residual=True)
            self.up3 = UpBlock(32, 16, residual=True)
            self.head = nn.Sequential(
                nn.Conv2d(18, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                ResidualBlock(32),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 1, kernel_size=1),
            )

    def coordinate_channels(self, batch_size, height, width, device):
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return coords.expand(batch_size, -1, -1, -1)

    def forward(self, bz_signal, coords=None):
        batch_size = bz_signal.shape[0]
        latent = self.bz_encoder(bz_signal)
        low_h, low_w = self.low_shape
        features = self.project(latent).view(batch_size, self.base_channels, low_h, low_w)
        x = self.stem(features)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        if tuple(x.shape[-2:]) != self.out_shape:
            x = F.interpolate(x, size=self.out_shape, mode='bilinear', align_corners=False)
        if self.variant == 'unet_residual':
            coord_channels = self.coordinate_channels(batch_size, x.shape[-2], x.shape[-1], x.device)
            x = torch.cat([x, coord_channels], dim=1)
        logits = self.head(x)
        return logits[:, 0].reshape(batch_size, -1)


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREENING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_current_baseline_checkpoints():
    missing = [path for path in CURRENT_BASELINE_CHECKPOINTS.values() if not Path(project_path(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing current mask boundary grid checkpoints: ' + ', '.join(missing))


def dataset_low_signal_indices(data_path):
    dataset = MFLDataset(data_path)
    signals = np.asarray(dataset.signals, dtype=np.float32)
    if signals.ndim == 2:
        flat = signals
    else:
        flat = signals.reshape(signals.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    threshold = np.quantile(max_abs, 1 / 3)
    return {int(idx) for idx, value in enumerate(max_abs) if float(value) <= float(threshold)}


def load_grid_checkpoint(path, signal_length, signal_channels, out_shape, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundaryGridModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
        out_shape=tuple(args.get('out_shape', out_shape)),
        low_shape=tuple(args.get('low_shape', GRID_LOW_SHAPE)),
        base_channels=int(args.get('base_channels', RESIDUAL_BASE_CHANNELS)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_unet_checkpoint(path, signal_length, signal_channels, out_shape, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundaryUNetDecoderModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
        out_shape=tuple(args.get('out_shape', out_shape)),
        variant=args.get('variant', 'unet_light'),
        low_shape=tuple(args.get('low_shape', GRID_LOW_SHAPE)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_model_for_selection(model, dataset, device, area_edges):
    prob_maps, true_masks = predict_prob_maps(model, dataset, coords=None, device=device)
    rows = build_sample_rows(
        candidate='selection',
        seed='selection',
        split='val',
        threshold=TRAIN_SELECTION_THRESHOLD,
        prob_maps=prob_maps,
        true_masks=true_masks,
        dataset=dataset,
        area_edges=area_edges,
        low_signal_indices=set(),
    )
    return summarize_samples(rows)


def train_one_seed(seed, variant, device, pos_weight_value):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    out_shape = tuple(train_dataset.mu_maps.shape[1:])
    model = MaskBoundaryUNetDecoderModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
        out_shape=out_shape,
        variant=variant,
    ).to(device)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_mask_boundary_unet_{variant}_seed{seed}.pt'

    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu')
        info = checkpoint.get('val_metrics')
        if info is None:
            info = {
                'seed': seed,
                'variant': variant,
                'epoch': int(checkpoint.get('epoch', 0)),
                'selection_score': float(checkpoint.get('selection_score', float('nan'))),
                'val_iou': float('nan'),
                'val_dice': float('nan'),
                'val_area_error': float('nan'),
                'val_center_error': float('nan'),
                'val_pred_area_zero': float('nan'),
            }
        print(f'Reusing existing checkpoint for variant={variant} seed={seed}: {best_path}')
        return best_path, info

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_samples = 0
        for signals, mu_targets, indices in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            mask_logits = model(signals)
            loss, bce, dice = mask_loss(mask_logits, target_mask, pos_weight)
            loss.backward()
            optimizer.step()

            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
            total_samples += batch_size

        val_summary = evaluate_model_for_selection(model, val_dataset, device, val_area_edges)
        selection_score = val_summary['composite']
        if selection_score > best_score:
            best_score = selection_score
            best_info = {
                'seed': seed,
                'variant': variant,
                'epoch': epoch,
                'selection_score': float(selection_score),
                'val_iou': val_summary['iou'],
                'val_dice': val_summary['dice'],
                'val_area_error': val_summary['area_error'],
                'val_center_error': val_summary['center_error'],
                'val_pred_area_zero': val_summary['pred_area_zero'],
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': {
                    'model': 'mask_boundary_unet_decoder_model',
                    'dataset': 'v3_complex',
                    'variant': variant,
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'loss': 'BCEWithLogits + soft Dice',
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'decoder': 'latent projection to low-resolution feature map plus U-Net-like multi-scale spatial decoder',
                    'out_shape': out_shape,
                    'low_shape': GRID_LOW_SHAPE,
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
                    'signal_channels': signal_channels,
                },
                'signal_mean': float(train_dataset.signal_mean),
                'signal_std': float(train_dataset.signal_std),
                'epoch': epoch,
                'selection_score': float(selection_score),
                'val_metrics': best_info,
            }, best_path)

        print(
            f"variant={variant} seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"loss={total_loss / total_samples:.6e} | "
            f"bce={total_bce / total_samples:.6e} | "
            f"dice_loss={total_dice / total_samples:.6e} | "
            f"val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | "
            f"score={selection_score:.6e}"
        )

    return best_path, best_info


def evaluate_checkpoint_family(checkpoints, model_type, candidate, split, data_path, thresholds, device, area_edges, low_signal_indices):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    prob_cache = {}

    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu')
        dataset = MFLDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        if model_type == 'grid':
            model, _ = load_grid_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, out_shape, device)
        elif model_type == 'unet':
            model, _ = load_unet_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, out_shape, device)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
        prob_maps, true_masks = predict_prob_maps(model, dataset, coords=None, device=device)
        prob_cache[seed] = (prob_maps, true_masks, dataset)
        for threshold in thresholds:
            sample_rows = build_sample_rows(
                candidate=candidate,
                seed=seed,
                split=split,
                threshold=threshold,
                prob_maps=prob_maps,
                true_masks=true_masks,
                dataset=dataset,
                area_edges=area_edges,
                low_signal_indices=low_signal_indices,
            )
            sample_rows_by_seed_threshold[(seed, threshold)] = sample_rows
            metric_rows.extend(summarize_candidate(sample_rows, candidate, seed, split, threshold))

    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows(metric_rows, candidate, split, threshold))
    return metric_rows, sample_rows_by_seed_threshold, prob_cache


def find_row(rows, candidate, group_type='overall', group='all', split='test', threshold=None):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['group_type'] == group_type
        and row['group'] == group
        and row['split'] == split
    ]
    if threshold is not None:
        selected = [row for row in selected if threshold_matches(row['threshold'], threshold)]
    if not selected:
        raise KeyError(f'Missing row candidate={candidate}, split={split}, group_type={group_type}, group={group}, threshold={threshold}')
    return selected[0]


def get_overall_mean(rows, candidate, threshold, split='test'):
    return find_row(rows, f'{candidate}_mean', 'overall', 'all', split=split, threshold=threshold)


def select_threshold(rows, baseline_overall):
    validation_means = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_unet_decoder_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    eligible = [
        row for row in validation_means
        if float(row['iou']) > float(baseline_overall['iou'])
        and float(row['dice']) > float(baseline_overall['dice'])
    ]
    if eligible:
        return min(
            eligible,
            key=lambda row: (
                float(row['area_error']),
                -float(row['composite']),
                float(row['pred_area_zero']),
            ),
        )
    return max(validation_means, key=lambda row: float(row['composite']))


def write_metrics(rows):
    fieldnames = [
        'candidate',
        'seed',
        'split',
        'group_type',
        'group',
        'threshold',
        'n',
        'iou',
        'dice',
        'area_error',
        'center_error',
        'pred_area_zero',
        'pred_area_lt_true',
        'pred_area_gt_true',
        'composite',
        'macro_area_composite',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_screening_metrics(screening_records):
    fieldnames = [
        'decoder_variant',
        'seed',
        'best_epoch',
        'best_threshold',
        'val_score',
        'val_iou',
        'val_dice',
        'val_area_error',
        'val_center_error',
        'val_pred_area_zero',
        'small_iou',
        'small_dice',
        'small_area_error',
        'small_pred_area_zero',
        'low_signal_iou',
        'low_signal_dice',
        'low_signal_area_error',
        'low_signal_pred_area_zero',
        'polygon_iou',
        'polygon_dice',
        'polygon_area_error',
        'rotated_rect_iou',
        'rotated_rect_dice',
        'rotated_rect_area_error',
        'positive_signal',
        'checkpoint',
    ]
    with open(SCREENING_METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in screening_records:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def positive_signal(candidate_overall, candidate_polygon, candidate_rotated, baseline_overall, baseline_polygon, baseline_rotated):
    checks = {
        'overall_iou_not_down': float(candidate_overall['iou']) >= float(baseline_overall['iou']) - 1e-6,
        'overall_dice_not_down': float(candidate_overall['dice']) >= float(baseline_overall['dice']) - 1e-6,
        'overall_area_close': float(candidate_overall['area_error']) <= float(baseline_overall['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE,
        'overall_pred_zero_not_up': float(candidate_overall['pred_area_zero']) <= float(baseline_overall['pred_area_zero']) + 1e-6,
        'polygon_iou_not_much_down': float(candidate_polygon['iou']) >= float(baseline_polygon['iou']) - TYPE_METRIC_DROP_TOLERANCE,
        'polygon_dice_not_much_down': float(candidate_polygon['dice']) >= float(baseline_polygon['dice']) - TYPE_METRIC_DROP_TOLERANCE,
        'polygon_area_close': float(candidate_polygon['area_error']) <= float(baseline_polygon['area_error']) + TYPE_AREA_TOLERANCE,
        'rotated_iou_not_much_down': float(candidate_rotated['iou']) >= float(baseline_rotated['iou']) - TYPE_METRIC_DROP_TOLERANCE,
        'rotated_dice_not_much_down': float(candidate_rotated['dice']) >= float(baseline_rotated['dice']) - TYPE_METRIC_DROP_TOLERANCE,
        'rotated_area_close': float(candidate_rotated['area_error']) <= float(baseline_rotated['area_error']) + TYPE_AREA_TOLERANCE,
    }
    return bool(all(checks.values())), checks


def fmt(value, metric):
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_screening_table(screening_records):
    lines = [
        '| decoder_variant | best_epoch | best_threshold | val_score | val_IoU | val_Dice | val_area_error | val_pred_area=0 | small_IoU | low_signal_IoU | polygon_IoU | rotated_rect_IoU | positive_signal | checkpoint |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in screening_records:
        lines.append(
            f"| {row['decoder_variant']} | {row['best_epoch']} | {row['best_threshold']:.2f} | {row['val_score']:.6e} | "
            f"{row['val_iou']:.4f} | {row['val_dice']:.4f} | {row['val_area_error']:.4f} | "
            f"{row['val_pred_area_zero']:.2f} | {row['small_iou']:.4f} | {row['low_signal_iou']:.4f} | "
            f"{row['polygon_iou']:.4f} | {row['rotated_rect_iou']:.4f} | "
            f"{row['positive_signal']} | {row['checkpoint']} |"
        )
    return '\n'.join(lines)


def format_val_scan(rows, baseline_overall):
    lines = [
        '| threshold | val IoU | val Dice | val area_error | val center_error | val pred_area=0 | selected eligible |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    selected = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_unet_decoder_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    for row in sorted(selected, key=lambda item: float(item['threshold'])):
        eligible = float(row['iou']) > float(baseline_overall['iou']) and float(row['dice']) > float(baseline_overall['dice'])
        lines.append(
            f"| {float(row['threshold']):.2f} | {float(row['iou']):.4f} | "
            f"{float(row['dice']):.4f} | {float(row['area_error']):.4f} | "
            f"{float(row['center_error']):.4f} | {float(row['pred_area_zero']):.2f} | {eligible} |"
        )
    return '\n'.join(lines)


def format_comparison_table(rows, group_type, groups, selected_threshold):
    lines = [
        '| group | candidate | threshold | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group in groups:
        base_mean = find_row(rows, 'current_mask_boundary_baseline_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        base_std = find_row(rows, 'current_mask_boundary_baseline_std', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        candidate_mean = find_row(rows, 'mask_boundary_unet_decoder_test_mean', group_type, group, threshold=selected_threshold)
        candidate_std = find_row(rows, 'mask_boundary_unet_decoder_test_std', group_type, group, threshold=selected_threshold)
        lines.append(
            f"| {group} | current grid decoder baseline | {CURRENT_BASELINE_THRESHOLD:.2f} | "
            f"{metric_with_std(base_mean, base_std, 'iou')} | "
            f"{metric_with_std(base_mean, base_std, 'dice')} | "
            f"{metric_with_std(base_mean, base_std, 'area_error')} | "
            f"{metric_with_std(base_mean, base_std, 'center_error')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | U-Net-like spatial decoder | {selected_threshold:.2f} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'iou')} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'dice')} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'area_error')} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'center_error')} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'pred_area_zero')} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(candidate_mean, candidate_std, 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group, selected_threshold):
    baseline = find_row(rows, 'current_mask_boundary_baseline_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
    candidate = find_row(rows, 'mask_boundary_unet_decoder_test_mean', group_type, group, threshold=selected_threshold)
    return {
        'iou_not_down': float(candidate['iou']) >= float(baseline['iou']) - 1e-6,
        'dice_not_down': float(candidate['dice']) >= float(baseline['dice']) - 1e-6,
        'area_error_close': float(candidate['area_error']) <= float(baseline['area_error']) + 0.02,
        'pred_area_zero_not_up': float(candidate['pred_area_zero']) <= float(baseline['pred_area_zero']) + 1e-6,
    }


def write_summary(
    rows,
    best_infos,
    checkpoint_paths,
    selected_threshold,
    pos_weight,
    mask_fraction,
    screening_records,
    best_variant,
    entered_stage_b,
):
    screening_section = format_screening_table(screening_records)
    if not entered_stage_b:
        summary = f"""# v3_complex mask-only conditional U-Net-like spatial decoder candidate

This RESULT_DRIVEN_EXPERIMENT tests fixed U-Net-like spatial decoder variants without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Stage A: seed=42 decoder screening

The model keeps the direct Bz -> mask setup and changes only the spatial decoder capacity. Training loss remains BCEWithLogits + soft Dice. No SDF, boundary head/loss, handcrafted Bz feature augmentation, adaptive threshold, post-processing, retrieval, or star-convex parameterization is used.

Decoder variants tested: {', '.join(DECODER_VARIANTS)}.

Threshold calibration rescue check is included in Stage A. Each seed=42 checkpoint is evaluated on validation thresholds {', '.join(f'{value:.2f}' for value in THRESHOLDS)}, and the best threshold is selected by IoU + Dice - area_error before applying the positive-signal gate.

{screening_section}

Best decoder variant by validation gate/score: {best_variant}

No decoder variant passed the validation positive-signal gate after threshold calibration. Stage B was not run, and U-Net v2/channel/block/threshold-variant tuning is not continued.

Stage B entered: False
Selected probability threshold: N/A
Accepted by metric gate: False

Output metrics for Stage B are not available because Stage B was not run. Screening CSV is written to `{SCREENING_METRICS_PATH.relative_to(ROOT)}`.
"""
        SUMMARY_PATH.write_text(summary, encoding='utf-8')
        return {'accepted': False}

    overall = improvement_status(rows, 'overall', 'all', selected_threshold)
    small = improvement_status(rows, 'area_bin', 'small', selected_threshold)
    medium = improvement_status(rows, 'area_bin', 'medium', selected_threshold)
    large = improvement_status(rows, 'area_bin', 'large', selected_threshold)
    low = improvement_status(rows, 'signal_bin', 'low_signal', selected_threshold)
    polygon = improvement_status(rows, 'defect_type', 'polygon', selected_threshold)
    rotated = improvement_status(rows, 'defect_type', 'rotated_rect', selected_threshold)
    accepted = bool(
        overall['iou_not_down']
        and overall['dice_not_down']
        and overall['area_error_close']
        and overall['pred_area_zero_not_up']
        and small['iou_not_down']
        and small['dice_not_down']
        and low['iou_not_down']
        and low['dice_not_down']
        and small['pred_area_zero_not_up']
        and low['pred_area_zero_not_up']
        and polygon['iou_not_down']
        and rotated['iou_not_down']
    )

    best_lines = [
        '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | checkpoint |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    for info, checkpoint_path in zip(best_infos, checkpoint_paths):
        best_lines.append(
            f"| {info['seed']} | {info['epoch']} | {info['selection_score']:.6e} | "
            f"{info['val_iou']:.4f} | {info['val_dice']:.4f} | {info['val_area_error']:.4f} | "
            f"{checkpoint_path.relative_to(ROOT)} |"
        )

    status_lines = [
        f"* overall: IoU not down={overall['iou_not_down']}, Dice not down={overall['dice_not_down']}, area_error close={overall['area_error_close']}, pred_area=0 not up={overall['pred_area_zero_not_up']}",
        f"* small: IoU not down={small['iou_not_down']}, Dice not down={small['dice_not_down']}, area_error close={small['area_error_close']}, pred_area=0 not up={small['pred_area_zero_not_up']}",
        f"* medium: IoU not down={medium['iou_not_down']}, Dice not down={medium['dice_not_down']}, area_error close={medium['area_error_close']}, pred_area=0 not up={medium['pred_area_zero_not_up']}",
        f"* large: IoU not down={large['iou_not_down']}, Dice not down={large['dice_not_down']}, area_error close={large['area_error_close']}, pred_area=0 not up={large['pred_area_zero_not_up']}",
        f"* low_signal: IoU not down={low['iou_not_down']}, Dice not down={low['dice_not_down']}, area_error close={low['area_error_close']}, pred_area=0 not up={low['pred_area_zero_not_up']}",
        f"* polygon: IoU not down={polygon['iou_not_down']}, Dice not down={polygon['dice_not_down']}, area_error close={polygon['area_error_close']}, pred_area=0 not up={polygon['pred_area_zero_not_up']}",
        f"* rotated_rect: IoU not down={rotated['iou_not_down']}, Dice not down={rotated['dice_not_down']}, area_error close={rotated['area_error_close']}, pred_area=0 not up={rotated['pred_area_zero_not_up']}",
    ]

    baseline_overall = find_row(rows, 'current_mask_boundary_baseline_mean', 'overall', 'all', threshold=CURRENT_BASELINE_THRESHOLD)
    summary = f"""# v3_complex mask-only conditional U-Net-like spatial decoder candidate

This RESULT_DRIVEN_EXPERIMENT tests fixed U-Net-like spatial decoder variants without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Stage A: seed=42 decoder screening

The model keeps the direct Bz -> mask setup and changes only the spatial decoder capacity. Decoder variants tested: {', '.join(DECODER_VARIANTS)}.

Stage A includes threshold calibration rescue screening. Each seed=42 checkpoint is evaluated on validation thresholds {', '.join(f'{value:.2f}' for value in THRESHOLDS)}, and the best threshold is selected by IoU + Dice - area_error before applying the positive-signal gate.

{screening_section}

Best decoder variant by validation gate/score: {best_variant}

Stage B entered: {entered_stage_b}

## Model and loss

The Stage B model uses BzEncoder, latent projection to a low-resolution 2D feature map, and a multi-scale U-Net-like spatial decoder to produce mask logits.

* BzEncoder latent dimension: {LATENT_DIM}
* low-resolution feature map: {GRID_LOW_SHAPE[0]} x {GRID_LOW_SHAPE[1]}
* best decoder variant: {best_variant}
* loss: BCEWithLogits + soft Dice

No SDF/boundary loss, loss-weight sweep, adaptive threshold, post-processing, retrieval, or star-convex parameterization is used.

Train mask positive fraction: {mask_fraction:.6f}. BCE pos_weight uses sqrt(neg/pos) capped at {POS_WEIGHT_CAP:.1f}; value used: {pos_weight:.6f}.

## Selected checkpoints

{chr(10).join(best_lines)}

## Validation threshold calibration

The three seeds share one validation-selected probability threshold. Threshold candidates: {', '.join(f'{value:.2f}' for value in THRESHOLDS)}.

Current baseline reference for threshold eligibility: mask-only grid decoder CURRENT_BASELINE test mean at threshold={CURRENT_BASELINE_THRESHOLD:.2f}, IoU={float(baseline_overall['iou']):.4f}, Dice={float(baseline_overall['dice']):.4f}.

Selected threshold: {selected_threshold:.2f}

{format_val_scan(rows, baseline_overall)}

## Overall test comparison

{format_comparison_table(rows, 'overall', ['all'], selected_threshold)}

## Area-bin test comparison

{format_comparison_table(rows, 'area_bin', ['small', 'medium', 'large'], selected_threshold)}

## Low-signal test comparison

{format_comparison_table(rows, 'signal_bin', ['low_signal', 'non_low_signal'], selected_threshold)}

## Defect-type test comparison

{format_comparison_table(rows, 'defect_type', ['polygon', 'rotated_rect', 'multi_defect'], selected_threshold)}

## Gate checks

{chr(10).join(status_lines)}

Accepted by metric gate: {accepted}

Preview PNGs are written to `{PREVIEW_DIR.relative_to(ROOT)}` for visual inspection if Stage B is reached.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {'accepted': accepted}


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def generate_previews(selected, prob_cache, sample_rows_by_seed_threshold, selected_threshold):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    written = []
    for item in selected:
        idx = int(item['index'])
        seed_probs = []
        seed_rows = []
        dataset = None
        for seed in SEEDS:
            prob_maps, true_masks, ds = prob_cache[seed]
            dataset = ds
            seed_probs.append(prob_maps[idx])
            seed_rows.append(next(row for row in sample_rows_by_seed_threshold[(seed, selected_threshold)] if row['sample_index'] == idx))
        prob_map = np.mean(seed_probs, axis=0)
        true_mask = true_masks[idx]
        pred_mask = prob_map >= selected_threshold
        true_edge = true_mask ^ (
            np.pad(true_mask[1:-1, 1:-1], ((1, 1), (1, 1)), mode='constant')
            if min(true_mask.shape) > 2 else true_mask
        )
        pred_edge = pred_mask ^ (
            np.pad(pred_mask[1:-1, 1:-1], ((1, 1), (1, 1)), mode='constant')
            if min(pred_mask.shape) > 2 else pred_mask
        )
        overlay = np.zeros((*true_mask.shape, 3), dtype=np.float32)
        overlay[..., 0] = pred_edge.astype(np.float32)
        overlay[..., 1] = true_edge.astype(np.float32)
        overlay[..., 2] = pred_mask.astype(np.float32) * 0.25
        mean_iou = safe_nanmean([float(row['iou']) for row in seed_rows])
        mean_dice = safe_nanmean([float(row['dice']) for row in seed_rows])
        mean_area_error = safe_nanmean([float(row['area_error']) for row in seed_rows])

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
        axes[0].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('true mask')
        im = axes[1].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
        axes[1].set_title('probability')
        axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'pred >= {selected_threshold:.2f}')
        axes[3].imshow(overlay)
        axes[3].set_title('overlay G=true R=pred')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{item['category']} | sample={idx} | {dataset.defect_types[idx]} | "
            f"IoU={mean_iou:.3f} Dice={mean_dice:.3f} area_error={mean_area_error:.3f}",
            fontsize=9,
        )
        filename = f"{safe_name(item['category'])}_sample{idx:03d}_{safe_name(dataset.defect_types[idx])}.png"
        path = PREVIEW_DIR / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def main():
    ensure_outputs()
    check_current_baseline_checkpoints()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset_for_weight = MFLDataset(TRAIN_DATA)
    pos_weight, mask_fraction = compute_pos_weight(train_dataset_for_weight)
    print(f'mask positive fraction: {mask_fraction:.6f}')
    print(f'pos_weight: {pos_weight:.6f}')

    val_area_edges = get_area_edges(MFLDataset(VAL_DATA))
    test_dataset_for_edges = MFLDataset(TEST_DATA)
    test_area_edges = get_area_edges(test_dataset_for_edges)
    low_signal_indices = load_low_signal_indices()
    val_low_signal_indices = dataset_low_signal_indices(VAL_DATA)

    baseline_val_rows, _, _ = evaluate_checkpoint_family(
        checkpoints=CURRENT_BASELINE_CHECKPOINTS,
        model_type='grid',
        candidate='current_mask_boundary_baseline_val_reference',
        split='val',
        data_path=VAL_DATA,
        thresholds=[CURRENT_BASELINE_THRESHOLD],
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=val_low_signal_indices,
    )
    baseline_val_overall = find_row(
        baseline_val_rows,
        'current_mask_boundary_baseline_val_reference_mean',
        'overall',
        'all',
        split='val',
        threshold=CURRENT_BASELINE_THRESHOLD,
    )
    baseline_val_polygon = find_row(
        baseline_val_rows,
        'current_mask_boundary_baseline_val_reference_mean',
        'defect_type',
        'polygon',
        split='val',
        threshold=CURRENT_BASELINE_THRESHOLD,
    )
    baseline_val_rotated = find_row(
        baseline_val_rows,
        'current_mask_boundary_baseline_val_reference_mean',
        'defect_type',
        'rotated_rect',
        split='val',
        threshold=CURRENT_BASELINE_THRESHOLD,
    )

    screening_records = []
    screening_metric_rows = list(baseline_val_rows)
    screening_checkpoints = {}
    screening_best_infos = {}
    for variant in DECODER_VARIANTS:
        print(f'Stage A screening decoder_variant={variant} seed={SCREENING_SEED}')
        checkpoint_path, best_info = train_one_seed(SCREENING_SEED, variant, device, pos_weight)
        screening_checkpoints[variant] = checkpoint_path
        screening_best_infos[variant] = best_info
        candidate_name = f'unet_decoder_screening_{variant}'
        variant_rows, _, _ = evaluate_checkpoint_family(
            checkpoints={SCREENING_SEED: str(checkpoint_path.relative_to(ROOT))},
            model_type='unet',
            candidate=candidate_name,
            split='val',
            data_path=VAL_DATA,
            thresholds=THRESHOLDS,
            device=device,
            area_edges=val_area_edges,
            low_signal_indices=val_low_signal_indices,
        )
        screening_metric_rows.extend(variant_rows)
        overall_candidates = [
            row for row in variant_rows
            if row['candidate'] == f'{candidate_name}_mean'
            and row['split'] == 'val'
            and row['group_type'] == 'overall'
            and row['group'] == 'all'
        ]
        overall = max(overall_candidates, key=lambda row: float(row['composite']))
        best_threshold = float(overall['threshold'])
        small = find_row(variant_rows, f'{candidate_name}_mean', 'area_bin', 'small', split='val', threshold=best_threshold)
        low = find_row(variant_rows, f'{candidate_name}_mean', 'signal_bin', 'low_signal', split='val', threshold=best_threshold)
        polygon = find_row(variant_rows, f'{candidate_name}_mean', 'defect_type', 'polygon', split='val', threshold=best_threshold)
        rotated = find_row(variant_rows, f'{candidate_name}_mean', 'defect_type', 'rotated_rect', split='val', threshold=best_threshold)
        positive, checks = positive_signal(overall, polygon, rotated, baseline_val_overall, baseline_val_polygon, baseline_val_rotated)
        screening_records.append({
            'decoder_variant': variant,
            'seed': SCREENING_SEED,
            'best_epoch': best_info['epoch'],
            'best_threshold': best_threshold,
            'val_score': float(overall['composite']),
            'val_iou': float(overall['iou']),
            'val_dice': float(overall['dice']),
            'val_area_error': float(overall['area_error']),
            'val_center_error': float(overall['center_error']),
            'val_pred_area_zero': float(overall['pred_area_zero']),
            'small_iou': float(small['iou']),
            'small_dice': float(small['dice']),
            'small_area_error': float(small['area_error']),
            'small_pred_area_zero': float(small['pred_area_zero']),
            'low_signal_iou': float(low['iou']),
            'low_signal_dice': float(low['dice']),
            'low_signal_area_error': float(low['area_error']),
            'low_signal_pred_area_zero': float(low['pred_area_zero']),
            'polygon_iou': float(polygon['iou']),
            'polygon_dice': float(polygon['dice']),
            'polygon_area_error': float(polygon['area_error']),
            'rotated_rect_iou': float(rotated['iou']),
            'rotated_rect_dice': float(rotated['dice']),
            'rotated_rect_area_error': float(rotated['area_error']),
            'positive_signal': positive,
            'positive_signal_checks': checks,
            'checkpoint': str(checkpoint_path.relative_to(ROOT)),
        })

    write_screening_metrics(screening_records)
    positive_records = [row for row in screening_records if row['positive_signal']]
    if positive_records:
        best_screening = max(positive_records, key=lambda row: row['val_score'])
        entered_stage_b = True
    else:
        best_screening = max(screening_records, key=lambda row: row['val_score'])
        entered_stage_b = False
    best_variant = best_screening['decoder_variant']
    print(f'Best decoder variant: {best_variant}; entered Stage B: {entered_stage_b}')

    baseline_rows, baseline_sample_rows, _ = evaluate_checkpoint_family(
        checkpoints=CURRENT_BASELINE_CHECKPOINTS,
        model_type='grid',
        candidate='current_mask_boundary_baseline',
        split='test',
        data_path=TEST_DATA,
        thresholds=[CURRENT_BASELINE_THRESHOLD],
        device=device,
        area_edges=test_area_edges,
        low_signal_indices=low_signal_indices,
    )
    baseline_overall = get_overall_mean(
        baseline_rows,
        candidate='current_mask_boundary_baseline',
        threshold=CURRENT_BASELINE_THRESHOLD,
        split='test',
    )

    if not entered_stage_b:
        write_metrics(screening_metric_rows + baseline_rows)
        judgment = write_summary(
            rows=screening_metric_rows + baseline_rows,
            best_infos=[],
            checkpoint_paths=[],
            selected_threshold=None,
            pos_weight=pos_weight,
            mask_fraction=mask_fraction,
            screening_records=screening_records,
            best_variant=best_variant,
            entered_stage_b=False,
        )
        print(f'Wrote screening metrics: {SCREENING_METRICS_PATH}')
        print(f'Wrote metrics: {METRICS_PATH}')
        print(f'Wrote summary: {SUMMARY_PATH}')
        print(f"Accepted by metric gate: {judgment['accepted']}")
        return

    checkpoint_paths = [screening_checkpoints[best_variant]]
    best_infos = [screening_best_infos[best_variant]]
    for seed in [seed for seed in SEEDS if seed != SCREENING_SEED]:
        print(f'Stage B training decoder_variant={best_variant} seed={seed}')
        checkpoint_path, best_info = train_one_seed(seed, best_variant, device, pos_weight)
        checkpoint_paths.append(checkpoint_path)
        best_infos.append(best_info)

    candidate_checkpoints = {seed: str(path.relative_to(ROOT)) for seed, path in zip(SEEDS, checkpoint_paths)}
    val_rows, _, _ = evaluate_checkpoint_family(
        checkpoints=candidate_checkpoints,
        model_type='unet',
        candidate='mask_boundary_unet_decoder_val_scan',
        split='val',
        data_path=VAL_DATA,
        thresholds=THRESHOLDS,
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=set(),
    )
    selected_row = select_threshold(val_rows, baseline_overall)
    selected_threshold = float(selected_row['threshold'])
    print(f'Selected threshold: {selected_threshold:.2f}')

    test_rows, candidate_sample_rows, candidate_prob_cache = evaluate_checkpoint_family(
        checkpoints=candidate_checkpoints,
        model_type='unet',
        candidate='mask_boundary_unet_decoder_test',
        split='test',
        data_path=TEST_DATA,
        thresholds=[selected_threshold],
        device=device,
        area_edges=test_area_edges,
        low_signal_indices=low_signal_indices,
    )

    all_rows = baseline_rows + val_rows + test_rows
    write_metrics(all_rows)
    judgment = write_summary(
        rows=all_rows,
        best_infos=best_infos,
        checkpoint_paths=checkpoint_paths,
        selected_threshold=selected_threshold,
        pos_weight=pos_weight,
        mask_fraction=mask_fraction,
        screening_records=screening_records,
        best_variant=best_variant,
        entered_stage_b=True,
    )

    candidate_means = sample_mean_metrics(candidate_sample_rows, SEEDS, selected_threshold)
    baseline_means = sample_mean_metrics(baseline_sample_rows, SEEDS, CURRENT_BASELINE_THRESHOLD)
    selected_samples = select_preview_samples(candidate_means, baseline_means)
    preview_paths = generate_previews(selected_samples, candidate_prob_cache, candidate_sample_rows, selected_threshold)

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote previews: {PREVIEW_DIR} ({len(preview_paths)} png)')
    print(f"Accepted by metric gate: {judgment['accepted']}")


if __name__ == '__main__':
    main()
