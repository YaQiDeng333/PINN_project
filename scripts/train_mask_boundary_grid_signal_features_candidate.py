import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import (  # noqa: E402
    BzEncoder,
    MFLDataset,
    MU_SCALE,
    build_coord_grid,
    feature_mapping,
    project_path,
    set_seed,
    signal_shape_info,
)


TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

CURRENT_BASELINE_CHECKPOINTS = {
    42: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed42.pt',
    123: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed123.pt',
    2026: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed2026.pt',
}
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'

SEEDS = [42, 123, 2026]
SCREENING_SEED = 42
FEATURE_MODES = ['raw_norm', 'raw_deriv', 'raw_norm_deriv', 'raw_norm_deriv_stats']
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
GRID_BASE_CHANNELS = 64
GRID_LOW_SHAPE = (10, 20)
POSITIVE_SIGNAL_AREA_TOLERANCE = 0.02

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_boundary_grid_signal_features_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_grid_signal_features_candidate_metrics.csv'
SCREENING_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_grid_signal_features_candidate_screening.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_boundary_grid_signal_features_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'mask_boundary_grid_signal_features_candidate'

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


def safe_nanmean(values):
    arr = np.array(values, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


def safe_nanstd(values):
    arr = np.array(values, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    if finite.size <= 1:
        return 0.0
    return float(finite.std(ddof=1))


def as_channel_first(signals):
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 2:
        return signals[:, None, :]
    if signals.ndim == 3:
        return signals
    raise ValueError(f'Unsupported raw signal shape: {signals.shape}')


def normalize_per_sample(signals):
    denom = np.max(np.abs(signals), axis=-1, keepdims=True) + 1e-8
    return signals / denom


def derivative_features(signals):
    first = np.gradient(signals, axis=-1, edge_order=2).astype(np.float32)
    second = np.gradient(first, axis=-1, edge_order=2).astype(np.float32)
    return first, second


def signal_stats(signals):
    flat = signals.reshape(signals.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    peak_to_peak = np.ptp(flat, axis=1)
    l2_energy = np.sqrt(np.mean(np.square(flat), axis=1))
    return np.stack([max_abs, peak_to_peak, l2_energy], axis=1).astype(np.float32)


def build_signal_features(raw_signals, feature_mode):
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f'Unsupported feature mode: {feature_mode}')
    raw = as_channel_first(raw_signals)
    norm = normalize_per_sample(raw)
    first, second = derivative_features(raw)

    channels = []
    names = []
    if feature_mode in ('raw_norm', 'raw_deriv', 'raw_norm_deriv', 'raw_norm_deriv_stats'):
        channels.append(raw)
        names.append('raw_bz')
    if feature_mode in ('raw_norm', 'raw_norm_deriv', 'raw_norm_deriv_stats'):
        channels.append(norm)
        names.append('per_sample_norm_bz')
    if feature_mode in ('raw_deriv', 'raw_norm_deriv', 'raw_norm_deriv_stats'):
        channels.extend([first, second])
        names.extend(['d_bz_dx', 'd2_bz_dx2'])

    features = np.concatenate(channels, axis=1).astype(np.float32)
    stats = np.zeros((features.shape[0], 0), dtype=np.float32)
    stat_names = []
    if feature_mode == 'raw_norm_deriv_stats':
        stats = signal_stats(raw)
        stat_names = ['max_abs_bz', 'peak_to_peak_bz', 'l2_energy_bz']
    return features, stats, names, stat_names


class SignalFeatureDataset(Dataset):
    def __init__(
        self,
        npz_path,
        feature_mode,
        feature_mean=None,
        feature_std=None,
        stat_mean=None,
        stat_std=None,
    ):
        data = np.load(project_path(npz_path), allow_pickle=False)
        raw_signals = data['signals'].astype(np.float32)
        features, stats, feature_names, stat_names = build_signal_features(raw_signals, feature_mode)
        if feature_mean is None:
            feature_mean = features.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        if feature_std is None:
            feature_std = (features.std(axis=(0, 2), dtype=np.float64) + 1e-8).astype(np.float32)
        feature_mean = np.asarray(feature_mean, dtype=np.float32)
        feature_std = np.asarray(feature_std, dtype=np.float32)
        self.signals = ((features - feature_mean[None, :, None]) / feature_std[None, :, None]).astype(np.float32)

        if stats.shape[1] > 0:
            if stat_mean is None:
                stat_mean = stats.mean(axis=0, dtype=np.float64).astype(np.float32)
            if stat_std is None:
                stat_std = (stats.std(axis=0, dtype=np.float64) + 1e-8).astype(np.float32)
            stat_mean = np.asarray(stat_mean, dtype=np.float32)
            stat_std = np.asarray(stat_std, dtype=np.float32)
            self.stat_features = ((stats - stat_mean[None, :]) / stat_std[None, :]).astype(np.float32)
        else:
            stat_mean = np.zeros((0,), dtype=np.float32)
            stat_std = np.ones((0,), dtype=np.float32)
            self.stat_features = stats.astype(np.float32)

        self.feature_mode = feature_mode
        self.feature_names = feature_names
        self.stat_names = stat_names
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.stat_mean = np.asarray(stat_mean, dtype=np.float32)
        self.stat_std = np.asarray(stat_std, dtype=np.float32)
        self.mu_maps = data['mu_maps'].astype(np.float32) / MU_SCALE
        self.defect_types = data['defect_types']
        self.metadata = data['metadata']
        self.metadata_keys = data['metadata_keys'] if 'metadata_keys' in data.files else None
        self.x = data['x'].astype(np.float32)
        self.y = data['y'].astype(np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.stat_features[idx]),
            torch.from_numpy(self.mu_maps[idx].reshape(-1)),
            idx,
        )


def load_signal_normalization(args):
    return {
        'feature_mean': np.asarray(args['feature_mean'], dtype=np.float32),
        'feature_std': np.asarray(args['feature_std'], dtype=np.float32),
        'stat_mean': np.asarray(args.get('stat_mean', []), dtype=np.float32),
        'stat_std': np.asarray(args.get('stat_std', []), dtype=np.float32),
    }


def val_low_signal_indices(data_path):
    data = np.load(project_path(data_path), allow_pickle=False)
    raw = as_channel_first(data['signals'].astype(np.float32))
    values = np.max(np.abs(raw.reshape(raw.shape[0], -1)), axis=1)
    threshold = np.quantile(values, 1 / 3)
    return {int(idx) for idx, value in enumerate(values) if float(value) <= float(threshold)}


class MaskOnlyModel(nn.Module):
    def __init__(self, signal_length, signal_channels=1, latent_dim=64, coord_feature_dim=84):
        super().__init__()
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        input_dim = coord_feature_dim + latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, bz_signal, coords):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(bz_signal.shape[0], -1, -1)
        bz_latent = self.bz_encoder(bz_signal)
        coord_features = feature_mapping(coords)
        bz_features = bz_latent.unsqueeze(1).expand(-1, coord_features.shape[1], -1)
        features = torch.cat([bz_features, coord_features], dim=-1)
        return self.decoder(features).squeeze(-1)


class MaskBoundaryGridModel(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channels=1,
        latent_dim=64,
        out_shape=(100, 200),
        low_shape=GRID_LOW_SHAPE,
        base_channels=GRID_BASE_CHANNELS,
    ):
        super().__init__()
        self.out_shape = tuple(out_shape)
        self.low_shape = tuple(low_shape)
        self.base_channels = int(base_channels)
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        low_h, low_w = self.low_shape
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, self.base_channels * low_h * low_w),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, bz_signal, coords=None):
        batch_size = bz_signal.shape[0]
        bz_latent = self.bz_encoder(bz_signal)
        low_h, low_w = self.low_shape
        features = self.project(bz_latent).view(batch_size, self.base_channels, low_h, low_w)
        logits = self.decoder(features)
        if tuple(logits.shape[-2:]) != self.out_shape:
            logits = F.interpolate(logits, size=self.out_shape, mode='bilinear', align_corners=False)
        return logits[:, 0].reshape(batch_size, -1)


class MaskBoundaryGridSignalFeatureModel(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channels=1,
        stats_dim=0,
        latent_dim=64,
        out_shape=(100, 200),
        low_shape=GRID_LOW_SHAPE,
        base_channels=GRID_BASE_CHANNELS,
    ):
        super().__init__()
        self.out_shape = tuple(out_shape)
        self.low_shape = tuple(low_shape)
        self.base_channels = int(base_channels)
        self.stats_dim = int(stats_dim)
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        if self.stats_dim > 0:
            self.stat_fusion = nn.Sequential(
                nn.Linear(latent_dim + self.stats_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
            )
        else:
            self.stat_fusion = None
        low_h, low_w = self.low_shape
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, self.base_channels * low_h * low_w),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, bz_signal, coords=None, stat_features=None):
        batch_size = bz_signal.shape[0]
        bz_latent = self.bz_encoder(bz_signal)
        if self.stat_fusion is not None:
            if stat_features is None:
                raise ValueError('stat_features are required for this feature mode')
            bz_latent = self.stat_fusion(torch.cat([bz_latent, stat_features], dim=1))
        low_h, low_w = self.low_shape
        features = self.project(bz_latent).view(batch_size, self.base_channels, low_h, low_w)
        logits = self.decoder(features)
        if tuple(logits.shape[-2:]) != self.out_shape:
            logits = F.interpolate(logits, size=self.out_shape, mode='bilinear', align_corners=False)
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
        raise FileNotFoundError('Missing current mask boundary checkpoints: ' + ', '.join(missing))


def make_loader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def compute_pos_weight(dataset):
    masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    pos = float(masks.sum())
    neg = float(masks.size - masks.sum())
    raw = np.sqrt(neg / max(pos, 1.0))
    return float(min(raw, POS_WEIGHT_CAP)), float(pos / masks.size)


def soft_dice_loss(logits, target_mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target_mask.reshape(target_mask.shape[0], -1)
    intersection = torch.sum(probs_flat * target_flat, dim=1)
    pred_sum = torch.sum(probs_flat, dim=1)
    target_sum = torch.sum(target_flat, dim=1)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return torch.mean(1.0 - dice)


def mask_loss(mask_logits, target_mask, pos_weight):
    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask, pos_weight=pos_weight)
    dice = soft_dice_loss(mask_logits, target_mask)
    return bce + dice, bce, dice


def mask_center(mask, x_grid, y_grid):
    if not np.any(mask):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([
        float(x_grid[mask].mean()),
        float(y_grid[mask].mean()),
    ], dtype=np.float32)


def compute_mask_metrics(pred_mask, true_mask, x_grid, y_grid):
    intersection = int(np.logical_and(pred_mask, true_mask).sum())
    union = int(np.logical_or(pred_mask, true_mask).sum())
    pred_area_pixels = int(pred_mask.sum())
    true_area_pixels = int(true_mask.sum())

    iou = 1.0 if union == 0 else intersection / union
    dice_denominator = pred_area_pixels + true_area_pixels
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator

    dx = float(abs(x_grid[0, 1] - x_grid[0, 0])) if x_grid.shape[1] > 1 else 1.0
    dy = float(abs(y_grid[1, 0] - y_grid[0, 0])) if y_grid.shape[0] > 1 else 1.0
    cell_area = dx * dy
    pred_area = pred_area_pixels * cell_area
    true_area = true_area_pixels * cell_area
    area_error = np.nan if true_area == 0 else abs(pred_area - true_area) / true_area

    pred_center = mask_center(pred_mask, x_grid, y_grid)
    true_center = mask_center(true_mask, x_grid, y_grid)
    center_error = float(np.linalg.norm(pred_center - true_center))

    return {
        'iou': float(iou),
        'dice': float(dice),
        'area_error': float(area_error),
        'center_error': center_error,
        'pred_area': float(pred_area),
        'true_area': float(true_area),
    }


def get_area_edges(dataset):
    masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    dx = float(abs(dataset.x[1] - dataset.x[0])) if len(dataset.x) > 1 else 1.0
    dy = float(abs(dataset.y[1] - dataset.y[0])) if len(dataset.y) > 1 else 1.0
    true_areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64) * dx * dy
    return np.quantile(true_areas, [1 / 3, 2 / 3])


def area_bin(true_area, edges):
    if true_area <= edges[0]:
        return 'small'
    if true_area <= edges[1]:
        return 'medium'
    return 'large'


def load_low_signal_indices():
    if not SIGNAL_AUDIT_PATH.exists():
        return set()
    with open(SIGNAL_AUDIT_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return set()
    values = sorted(float(row['max_abs_bz']) for row in rows)
    threshold = values[min(66, len(values) - 1)]
    return {int(row['sample_index']) for row in rows if float(row['max_abs_bz']) <= threshold}


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
        })
        return summary

    for key in ['iou', 'dice', 'area_error', 'center_error']:
        summary[key] = safe_nanmean([float(row[key]) for row in rows])
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    return summary


def metric_row(candidate, seed, split, threshold, group_type, group, summary, macro_area_composite):
    row = {
        'candidate': candidate,
        'seed': seed,
        'split': split,
        'group_type': group_type,
        'group': group,
        'threshold': threshold,
        'n': summary['n'],
    }
    for key in [
        'iou',
        'dice',
        'area_error',
        'center_error',
        'pred_area_zero',
        'pred_area_lt_true',
        'pred_area_gt_true',
        'composite',
    ]:
        row[key] = summary[key]
    row['macro_area_composite'] = macro_area_composite
    return row


def summarize_candidate(sample_rows, candidate, seed, split, threshold):
    rows = []
    overall = summarize_samples(sample_rows)
    area_summaries = {
        group: summarize_samples([row for row in sample_rows if row['area_bin'] == group])
        for group in ['small', 'medium', 'large']
    }
    macro_area_composite = safe_nanmean([
        area_summaries[group]['composite']
        for group in ['small', 'medium', 'large']
    ])

    rows.append(metric_row(candidate, seed, split, threshold, 'overall', 'all', overall, macro_area_composite))
    for group in ['small', 'medium', 'large']:
        rows.append(metric_row(candidate, seed, split, threshold, 'area_bin', group, area_summaries[group], macro_area_composite))
    for group in ['low_signal', 'non_low_signal']:
        selected = [row for row in sample_rows if row['signal_bin'] == group]
        rows.append(metric_row(candidate, seed, split, threshold, 'signal_bin', group, summarize_samples(selected), macro_area_composite))
    for defect_type in sorted({row['defect_type'] for row in sample_rows}):
        selected = [row for row in sample_rows if row['defect_type'] == defect_type]
        rows.append(metric_row(candidate, seed, split, threshold, 'defect_type', defect_type, summarize_samples(selected), macro_area_composite))
    return rows


def threshold_matches(left, right):
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return str(left) == str(right)


def aggregate_seed_rows(metric_rows, source_candidate, split, threshold):
    aggregate_rows = []
    groups = sorted({
        (row['group_type'], row['group'])
        for row in metric_rows
        if row['candidate'] == source_candidate
        and row['split'] == split
        and threshold_matches(row['threshold'], threshold)
    })
    for group_type, group in groups:
        selected = [
            row for row in metric_rows
            if row['candidate'] == source_candidate
            and row['split'] == split
            and threshold_matches(row['threshold'], threshold)
            and row['group_type'] == group_type
            and row['group'] == group
        ]
        mean_row = {
            'candidate': f'{source_candidate}_mean',
            'seed': 'mean',
            'split': split,
            'group_type': group_type,
            'group': group,
            'threshold': threshold,
            'n': selected[0]['n'] if selected else 0,
        }
        std_row = dict(mean_row)
        std_row['candidate'] = f'{source_candidate}_std'
        std_row['seed'] = 'sample_std'
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in selected]
            mean_row[key] = safe_nanmean(values)
            std_row[key] = safe_nanstd(values)
        aggregate_rows.extend([mean_row, std_row])
    return aggregate_rows


def build_sample_rows(candidate, seed, split, threshold, prob_maps, true_masks, dataset, area_edges, low_signal_indices):
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    rows = []
    for sample_idx in range(len(dataset)):
        pred_mask = prob_maps[sample_idx] >= threshold
        metrics = compute_mask_metrics(pred_mask, true_masks[sample_idx], x_grid, y_grid)
        metrics.update({
            'candidate': candidate,
            'seed': seed,
            'split': split,
            'threshold': threshold,
            'sample_index': sample_idx,
            'defect_type': str(dataset.defect_types[sample_idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if sample_idx in low_signal_indices else 'non_low_signal',
        })
        rows.append(metrics)
    return rows


@torch.no_grad()
def predict_prob_maps(model, dataset, coords, device):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    model.eval()
    for batch in loader:
        if len(batch) == 4:
            signals, stat_features, mu_targets, indices = batch
            stat_features = stat_features.to(device)
        else:
            signals, mu_targets, indices = batch
            stat_features = None
        signals = signals.to(device)
        output = model(signals, coords, stat_features) if stat_features is not None else model(signals, coords)
        logits = output[0] if isinstance(output, tuple) else output
        probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = probs[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
    return prob_maps, true_masks


def load_grid_checkpoint(path, signal_length, signal_channels, out_shape, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundaryGridModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
        out_shape=tuple(args.get('out_shape', out_shape)),
        low_shape=tuple(args.get('low_shape', GRID_LOW_SHAPE)),
        base_channels=int(args.get('base_channels', GRID_BASE_CHANNELS)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_signal_feature_checkpoint(path, signal_length, signal_channels, out_shape, stats_dim, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundaryGridSignalFeatureModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        stats_dim=int(args.get('stats_dim', stats_dim)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
        out_shape=tuple(args.get('out_shape', out_shape)),
        low_shape=tuple(args.get('low_shape', GRID_LOW_SHAPE)),
        base_channels=int(args.get('base_channels', GRID_BASE_CHANNELS)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_mask_only_checkpoint(path, signal_length, signal_channels, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskOnlyModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_model_for_selection(model, dataset, coords, device, area_edges):
    prob_maps, true_masks = predict_prob_maps(model, dataset, coords, device)
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


def train_one_seed(seed, device, pos_weight_value, feature_mode):
    set_seed(seed)
    train_dataset = SignalFeatureDataset(TRAIN_DATA, feature_mode)
    val_dataset = SignalFeatureDataset(
        VAL_DATA,
        feature_mode,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
        stat_mean=train_dataset.stat_mean,
        stat_std=train_dataset.stat_std,
    )
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    stats_dim = int(train_dataset.stat_features.shape[1])
    out_shape = tuple(train_dataset.mu_maps.shape[1:])
    model = MaskBoundaryGridSignalFeatureModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        stats_dim=stats_dim,
        latent_dim=LATENT_DIM,
        out_shape=out_shape,
    ).to(device)
    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_mask_boundary_grid_signal_features_{feature_mode}_seed{seed}.pt'

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_samples = 0
        for signals, stat_features, mu_targets, indices in train_loader:
            signals = signals.to(device)
            stat_features = stat_features.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            mask_logits = model(signals, coords, stat_features)
            loss, bce, dice = mask_loss(mask_logits, target_mask, pos_weight)
            loss.backward()
            optimizer.step()

            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
            total_samples += batch_size

        val_summary = evaluate_model_for_selection(model, val_dataset, coords, device, val_area_edges)
        selection_score = val_summary['composite']
        if selection_score > best_score:
            best_score = selection_score
            best_info = {
                'seed': seed,
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
                    'model': 'mask_boundary_grid_signal_features_model',
                    'dataset': 'v3_complex',
                    'seed': seed,
                    'feature_mode': feature_mode,
                    'feature_names': train_dataset.feature_names,
                    'stat_names': train_dataset.stat_names,
                    'feature_mean': train_dataset.feature_mean.tolist(),
                    'feature_std': train_dataset.feature_std.tolist(),
                    'stat_mean': train_dataset.stat_mean.tolist(),
                    'stat_std': train_dataset.stat_std.tolist(),
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'stats_dim': stats_dim,
                    'loss': 'BCEWithLogits + soft Dice',
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'decoder': 'mask-only grid decoder; only BzEncoder input features are changed',
                    'out_shape': out_shape,
                    'low_shape': GRID_LOW_SHAPE,
                    'base_channels': GRID_BASE_CHANNELS,
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
                    'signal_channels': signal_channels,
                },
                'signal_mean': 0.0,
                'signal_std': 1.0,
                'epoch': epoch,
                'selection_score': float(selection_score),
                'val_metrics': best_info,
            }, best_path)

        print(
            f"mode={feature_mode} seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
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
        args = checkpoint.get('args', {})
        if model_type == 'signal_features':
            norm = load_signal_normalization(args)
            dataset = SignalFeatureDataset(
                data_path,
                feature_mode=args['feature_mode'],
                feature_mean=norm['feature_mean'],
                feature_std=norm['feature_std'],
                stat_mean=norm['stat_mean'],
                stat_std=norm['stat_std'],
            )
        else:
            dataset = MFLDataset(
                data_path,
                signal_mean=float(checkpoint['signal_mean']),
                signal_std=float(checkpoint['signal_std']),
            )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        if model_type == 'grid':
            model, _ = load_grid_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, out_shape, device)
        elif model_type == 'signal_features':
            stats_dim = int(dataset.stat_features.shape[1])
            model, _ = load_signal_feature_checkpoint(
                Path(project_path(checkpoint_path)),
                signal_length,
                signal_channels,
                out_shape,
                stats_dim,
                device,
            )
        else:
            model, _ = load_mask_only_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, device)
        coords = build_coord_grid(dataset.x, dataset.y).to(device)
        prob_maps, true_masks = predict_prob_maps(model, dataset, coords, device)
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


def get_overall_mean(rows, candidate, threshold, split='test'):
    return next(
        row for row in rows
        if row['candidate'] == f'{candidate}_mean'
        and row['split'] == split
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
        and threshold_matches(row['threshold'], threshold)
    )


def select_threshold(rows, baseline_overall):
    validation_means = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_grid_signal_features_val_scan_mean'
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
    return selected[0]


def fmt(value, metric):
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_screening_table(screening_records):
    lines = [
        '| feature_mode | best_epoch | val_score | val_IoU | val_Dice | val_area_error | val_center_error | val_pred_area=0 | small_IoU | small_Dice | small_area_error | low_signal_IoU | low_signal_Dice | low_signal_area_error | positive_signal | checkpoint |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in screening_records:
        lines.append(
            f"| {row['feature_mode']} | {row['best_epoch']} | {row['val_score']:.6e} | "
            f"{row['val_iou']:.4f} | {row['val_dice']:.4f} | {row['val_area_error']:.4f} | "
            f"{row['val_center_error']:.4f} | {row['val_pred_area_zero']:.2f} | "
            f"{row['small_iou']:.4f} | {row['small_dice']:.4f} | {row['small_area_error']:.4f} | "
            f"{row['low_signal_iou']:.4f} | {row['low_signal_dice']:.4f} | {row['low_signal_area_error']:.4f} | "
            f"{row['positive_signal']} | {row['checkpoint']} |"
        )
    return '\n'.join(lines)


def write_screening_metrics(screening_records):
    fieldnames = [
        'feature_mode',
        'seed',
        'best_epoch',
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
        'positive_signal',
        'checkpoint',
    ]
    with open(SCREENING_METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in screening_records:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def positive_signal(candidate_overall, candidate_small, candidate_low, baseline_overall, baseline_small, baseline_low):
    checks = {
        'overall_iou_not_down': float(candidate_overall['iou']) >= float(baseline_overall['iou']) - 1e-6,
        'overall_dice_not_down': float(candidate_overall['dice']) >= float(baseline_overall['dice']) - 1e-6,
        'overall_area_close': float(candidate_overall['area_error']) <= float(baseline_overall['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE,
        'overall_pred_zero_not_up': float(candidate_overall['pred_area_zero']) <= float(baseline_overall['pred_area_zero']) + 1e-6,
        'small_iou_not_down': float(candidate_small['iou']) >= float(baseline_small['iou']) - 1e-6,
        'small_dice_not_down': float(candidate_small['dice']) >= float(baseline_small['dice']) - 1e-6,
        'small_area_close': float(candidate_small['area_error']) <= float(baseline_small['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE,
        'small_pred_zero_not_up': float(candidate_small['pred_area_zero']) <= float(baseline_small['pred_area_zero']) + 1e-6,
        'low_signal_iou_not_down': float(candidate_low['iou']) >= float(baseline_low['iou']) - 1e-6,
        'low_signal_dice_not_down': float(candidate_low['dice']) >= float(baseline_low['dice']) - 1e-6,
        'low_signal_area_close': float(candidate_low['area_error']) <= float(baseline_low['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE,
        'low_signal_pred_zero_not_up': float(candidate_low['pred_area_zero']) <= float(baseline_low['pred_area_zero']) + 1e-6,
    }
    return bool(all(checks.values())), checks


def format_val_scan(rows, baseline_overall):
    lines = [
        '| threshold | val IoU | val Dice | val area_error | val center_error | val pred_area=0 | selected eligible |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    selected = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_grid_signal_features_val_scan_mean'
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
        grid_mean = find_row(rows, 'mask_boundary_grid_signal_features_test_mean', group_type, group, threshold=selected_threshold)
        grid_std = find_row(rows, 'mask_boundary_grid_signal_features_test_std', group_type, group, threshold=selected_threshold)
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
            f"| {group} | best signal-feature grid decoder | {selected_threshold:.2f} | "
            f"{metric_with_std(grid_mean, grid_std, 'iou')} | "
            f"{metric_with_std(grid_mean, grid_std, 'dice')} | "
            f"{metric_with_std(grid_mean, grid_std, 'area_error')} | "
            f"{metric_with_std(grid_mean, grid_std, 'center_error')} | "
            f"{metric_with_std(grid_mean, grid_std, 'pred_area_zero')} | "
            f"{metric_with_std(grid_mean, grid_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(grid_mean, grid_std, 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group, selected_threshold):
    baseline = find_row(rows, 'current_mask_boundary_baseline_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
    candidate = find_row(rows, 'mask_boundary_grid_signal_features_test_mean', group_type, group, threshold=selected_threshold)
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
    best_feature_mode,
    entered_stage_b,
):
    screening_section = format_screening_table(screening_records)
    if not entered_stage_b:
        best_record = next(row for row in screening_records if row['feature_mode'] == best_feature_mode)
        summary = f"""# v3_complex mask-only grid decoder + Bz signal feature augmentation pack

This RESULT_DRIVEN_EXPERIMENT_PACK tests fixed Bz signal feature modes without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Stage A: seed=42 feature-mode screening

The model keeps the current mask-only grid decoder body and only changes the BzEncoder input channels. Training loss remains BCEWithLogits + soft Dice. No SDF, boundary head/loss, decoder change, adaptive threshold, post-processing, or ensemble is used.

Feature modes tested: {', '.join(FEATURE_MODES)}.

{screening_section}

Best feature mode by validation score: {best_feature_mode}

The best feature mode did not pass the validation positive-signal gate. It is therefore not expanded to Stage B. No derivative v2 or additional handcrafted Bz features are continued.

Stage B entered: False
Selected probability threshold: N/A
Accepted by metric gate: False

Output metrics for Stage B are not available because Stage B was not run. Screening CSV is written to `{SCREENING_METRICS_PATH.relative_to(ROOT)}`.
"""
        SUMMARY_PATH.write_text(summary, encoding='utf-8')
        return {
            'accepted': False,
            'overall': None,
            'small': None,
            'medium': None,
            'large': None,
            'low_signal': None,
            'polygon': None,
            'rotated_rect': None,
            'best_record': best_record,
        }

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
    summary = f"""# v3_complex mask-only grid decoder + Bz signal feature augmentation pack

This RESULT_DRIVEN_EXPERIMENT_PACK tests fixed Bz signal feature modes without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Stage A: seed=42 feature-mode screening

The model keeps the current mask-only grid decoder body and only changes the BzEncoder input channels. Feature modes tested: {', '.join(FEATURE_MODES)}.

{screening_section}

Best feature mode by validation score: {best_feature_mode}

Stage B entered: {entered_stage_b}

## Model and loss

The Stage B model keeps the direct Bz -> mask setup. It reuses BzEncoder, projects the latent vector into a low-resolution 2D feature map, then upsamples it with ConvTranspose2d blocks to produce full-grid mask logits.

* BzEncoder latent dimension: {LATENT_DIM}
* low-resolution feature map: {GRID_BASE_CHANNELS} x {GRID_LOW_SHAPE[0]} x {GRID_LOW_SHAPE[1]}
* best feature mode: {best_feature_mode}
* decoder: unchanged ConvTranspose2d grid decoder followed by 2D convolution mask-logit head

Training loss is BCEWithLogits + soft Dice. No SDF/boundary loss, loss-weight sweep, adaptive threshold, post-processing, or ensemble is used.

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

Preview PNGs are written to `{PREVIEW_DIR.relative_to(ROOT)}` for visual inspection of boundary shape.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'accepted': accepted,
        'overall': overall,
        'small': small,
        'medium': medium,
        'large': large,
        'low_signal': low,
        'polygon': polygon,
        'rotated_rect': rotated,
    }


def sample_mean_metrics(sample_rows_by_seed_threshold, seeds, threshold):
    by_index = {}
    for seed in seeds:
        for row in sample_rows_by_seed_threshold[(seed, threshold)]:
            by_index.setdefault(row['sample_index'], []).append(row)
    means = []
    for sample_idx, rows in by_index.items():
        mean_iou = safe_nanmean([float(row['iou']) for row in rows])
        mean_dice = safe_nanmean([float(row['dice']) for row in rows])
        mean_area_error = safe_nanmean([float(row['area_error']) for row in rows])
        first = rows[0]
        means.append({
            'index': sample_idx,
            'defect_type': first['defect_type'],
            'area_bin': first['area_bin'],
            'signal_bin': first['signal_bin'],
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'mean_area_error': mean_area_error,
            'mask_score': mean_iou + mean_dice - mean_area_error,
        })
    return means


def select_preview_samples(grid_means, baseline_means):
    baseline_by_index = {row['index']: row for row in baseline_means}
    rows = []
    for row in grid_means:
        baseline = baseline_by_index[row['index']]
        improvement = (
            row['mean_iou'] - baseline['mean_iou']
            + row['mean_dice'] - baseline['mean_dice']
            + baseline['mean_area_error'] - row['mean_area_error']
        )
        item = dict(row)
        item['improvement'] = improvement
        rows.append(item)

    selected = []
    selected_indices = set()

    def take(category, candidates, n=3):
        out = []
        for item in candidates:
            if item['index'] in selected_indices:
                continue
            selected_indices.add(item['index'])
            copy = dict(item)
            copy['category'] = category
            selected.append(copy)
            out.append(copy)
            if len(out) == n:
                break
        return out

    small_polygon = [
        row for row in rows
        if row['defect_type'] == 'polygon'
        and row['area_bin'] == 'small'
        and row['improvement'] > 0
    ]
    small_polygon.sort(key=lambda row: row['improvement'], reverse=True)
    take('small_polygon_improved', small_polygon, 3)

    low_signal = [
        row for row in rows
        if row['signal_bin'] == 'low_signal'
        and row['improvement'] > 0
    ]
    low_signal.sort(key=lambda row: row['improvement'], reverse=True)
    take('low_signal_improved', low_signal, 3)

    failures = [row for row in rows if row['index'] not in selected_indices]
    failures.sort(key=lambda row: (row['mask_score'], row['mean_iou'], -row['mean_area_error']))
    take('mask_boundary_grid_signal_features_failure', failures, 3)

    remaining = [row for row in rows if row['index'] not in selected_indices]
    median_score = float(np.median([row['mask_score'] for row in remaining]))
    ordinary = [row for row in remaining if row['area_bin'] == 'medium'] or remaining
    ordinary.sort(key=lambda row: abs(row['mask_score'] - median_score))
    take('ordinary_medium', ordinary, 3)
    return selected


def safe_name(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('_')


def generate_previews(selected, prob_cache, sample_rows_by_seed_threshold, selected_threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    first_dataset = next(iter(prob_cache.values()))[2]
    x_grid, y_grid = np.meshgrid(first_dataset.x, first_dataset.y)
    extent = [
        float(first_dataset.x.min()),
        float(first_dataset.x.max()),
        float(first_dataset.y.min()),
        float(first_dataset.y.max()),
    ]
    rows_by_seed_index = {
        (seed, row['sample_index']): row
        for seed in SEEDS
        for row in sample_rows_by_seed_threshold[(seed, selected_threshold)]
    }

    written = []
    for item in selected:
        idx = item['index']
        fig, axes = plt.subplots(len(SEEDS), 4, figsize=(13.5, 9.0), constrained_layout=True)
        fig.suptitle(
            f"{item['category']} | sample {idx} | type={item['defect_type']} | area_bin={item['area_bin']} | "
            f"mean IoU={item['mean_iou']:.3f}, Dice={item['mean_dice']:.3f}, area_err={item['mean_area_error']:.3f}",
            fontsize=11,
        )
        im = None
        for row_idx, seed in enumerate(SEEDS):
            prob_maps, true_masks, _ = prob_cache[seed]
            prob = prob_maps[idx]
            true = true_masks[idx]
            pred = prob >= selected_threshold
            metrics = rows_by_seed_index[(seed, idx)]
            row_axes = axes[row_idx]
            row_axes[0].imshow(true, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
            row_axes[0].set_title('true mask')
            im = row_axes[1].imshow(prob, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
            row_axes[1].set_title('mask probability')
            row_axes[2].imshow(pred, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
            row_axes[2].set_title(f'pred mask >= {selected_threshold:.2f}')
            row_axes[3].imshow(true, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1, alpha=0.25)
            row_axes[3].contour(x_grid, y_grid, true.astype(float), levels=[0.5], colors=['lime'], linewidths=1.2)
            if pred.any():
                row_axes[3].contour(x_grid, y_grid, pred.astype(float), levels=[0.5], colors=['red'], linewidths=1.2)
            row_axes[3].set_title('overlay: true green, pred red')
            row_axes[0].set_ylabel(
                f"seed {seed}\nIoU={float(metrics['iou']):.3f}\n"
                f"Dice={float(metrics['dice']):.3f}\narea={float(metrics['area_error']):.3f}",
                fontsize=9,
            )
            for ax in row_axes:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(im, ax=axes[:, 1], fraction=0.035, pad=0.02)
        filename = f"{safe_name(item['category'])}_sample{idx:03d}_{safe_name(item['defect_type'])}.png"
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

    train_dataset_for_weight = SignalFeatureDataset(TRAIN_DATA, FEATURE_MODES[0])
    pos_weight, mask_fraction = compute_pos_weight(train_dataset_for_weight)
    print(f'mask positive fraction: {mask_fraction:.6f}')
    print(f'pos_weight: {pos_weight:.6f}')

    val_area_edges = get_area_edges(MFLDataset(VAL_DATA))
    test_dataset_for_edges = MFLDataset(TEST_DATA)
    test_area_edges = get_area_edges(test_dataset_for_edges)
    low_signal_indices = load_low_signal_indices()
    val_low_signal = val_low_signal_indices(VAL_DATA)

    baseline_val_rows, _, _ = evaluate_checkpoint_family(
        checkpoints=CURRENT_BASELINE_CHECKPOINTS,
        model_type='grid',
        candidate='current_mask_boundary_baseline_val_reference',
        split='val',
        data_path=VAL_DATA,
        thresholds=[CURRENT_BASELINE_THRESHOLD],
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=val_low_signal,
    )
    baseline_val_overall = find_row(
        baseline_val_rows,
        'current_mask_boundary_baseline_val_reference_mean',
        'overall',
        'all',
        split='val',
        threshold=CURRENT_BASELINE_THRESHOLD,
    )
    baseline_val_small = find_row(
        baseline_val_rows,
        'current_mask_boundary_baseline_val_reference_mean',
        'area_bin',
        'small',
        split='val',
        threshold=CURRENT_BASELINE_THRESHOLD,
    )
    baseline_val_low = find_row(
        baseline_val_rows,
        'current_mask_boundary_baseline_val_reference_mean',
        'signal_bin',
        'low_signal',
        split='val',
        threshold=CURRENT_BASELINE_THRESHOLD,
    )

    screening_records = []
    screening_metric_rows = list(baseline_val_rows)
    screening_checkpoints = {}
    for feature_mode in FEATURE_MODES:
        print(f'Stage A screening feature_mode={feature_mode} seed={SCREENING_SEED}')
        checkpoint_path, best_info = train_one_seed(SCREENING_SEED, device, pos_weight, feature_mode)
        screening_checkpoints[feature_mode] = checkpoint_path
        candidate_name = f'signal_features_screening_{feature_mode}'
        mode_rows, _, _ = evaluate_checkpoint_family(
            checkpoints={SCREENING_SEED: str(checkpoint_path.relative_to(ROOT))},
            model_type='signal_features',
            candidate=candidate_name,
            split='val',
            data_path=VAL_DATA,
            thresholds=[TRAIN_SELECTION_THRESHOLD],
            device=device,
            area_edges=val_area_edges,
            low_signal_indices=val_low_signal,
        )
        screening_metric_rows.extend(mode_rows)
        overall = find_row(
            mode_rows,
            f'{candidate_name}_mean',
            'overall',
            'all',
            split='val',
            threshold=TRAIN_SELECTION_THRESHOLD,
        )
        small = find_row(
            mode_rows,
            f'{candidate_name}_mean',
            'area_bin',
            'small',
            split='val',
            threshold=TRAIN_SELECTION_THRESHOLD,
        )
        low = find_row(
            mode_rows,
            f'{candidate_name}_mean',
            'signal_bin',
            'low_signal',
            split='val',
            threshold=TRAIN_SELECTION_THRESHOLD,
        )
        positive, checks = positive_signal(
            overall,
            small,
            low,
            baseline_val_overall,
            baseline_val_small,
            baseline_val_low,
        )
        screening_records.append({
            'feature_mode': feature_mode,
            'seed': SCREENING_SEED,
            'best_epoch': best_info['epoch'],
            'val_score': float(best_info['selection_score']),
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
            'positive_signal': positive,
            'positive_signal_checks': checks,
            'checkpoint': str(checkpoint_path.relative_to(ROOT)),
        })

    write_screening_metrics(screening_records)
    best_screening = max(screening_records, key=lambda row: row['val_score'])
    best_feature_mode = best_screening['feature_mode']
    entered_stage_b = bool(best_screening['positive_signal'])
    print(f'Best feature mode: {best_feature_mode}; entered Stage B: {entered_stage_b}')

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
            best_feature_mode=best_feature_mode,
            entered_stage_b=False,
        )
        print(f'Wrote screening metrics: {SCREENING_METRICS_PATH}')
        print(f'Wrote metrics: {METRICS_PATH}')
        print(f'Wrote summary: {SUMMARY_PATH}')
        print(f"Accepted by metric gate: {judgment['accepted']}")
        return

    checkpoint_paths = [screening_checkpoints[best_feature_mode]]
    best_infos = [next(
        {
            'seed': SCREENING_SEED,
            'epoch': row['best_epoch'],
            'selection_score': row['val_score'],
            'val_iou': row['val_iou'],
            'val_dice': row['val_dice'],
            'val_area_error': row['val_area_error'],
            'val_center_error': row['val_center_error'],
            'val_pred_area_zero': row['val_pred_area_zero'],
        }
        for row in screening_records
        if row['feature_mode'] == best_feature_mode
    )]
    for seed in [seed for seed in SEEDS if seed != SCREENING_SEED]:
        print(f'Stage B training best_feature_mode={best_feature_mode} seed={seed}')
        checkpoint_path, best_info = train_one_seed(seed, device, pos_weight, best_feature_mode)
        checkpoint_paths.append(checkpoint_path)
        best_infos.append(best_info)

    grid_checkpoints = {seed: str(path.relative_to(ROOT)) for seed, path in zip(SEEDS, checkpoint_paths)}
    val_rows, _, _ = evaluate_checkpoint_family(
        checkpoints=grid_checkpoints,
        model_type='signal_features',
        candidate='mask_boundary_grid_signal_features_val_scan',
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

    test_rows, grid_sample_rows, grid_prob_cache = evaluate_checkpoint_family(
        checkpoints=grid_checkpoints,
        model_type='signal_features',
        candidate='mask_boundary_grid_signal_features_test',
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
        best_feature_mode=best_feature_mode,
        entered_stage_b=True,
    )

    grid_means = sample_mean_metrics(grid_sample_rows, SEEDS, selected_threshold)
    baseline_means = sample_mean_metrics(baseline_sample_rows, SEEDS, CURRENT_BASELINE_THRESHOLD)
    selected_samples = select_preview_samples(grid_means, baseline_means)
    preview_paths = generate_previews(selected_samples, grid_prob_cache, grid_sample_rows, selected_threshold)

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote previews: {PREVIEW_DIR} ({len(preview_paths)} png)')
    print(f"Accepted by metric gate: {judgment['accepted']}")


if __name__ == '__main__':
    main()
