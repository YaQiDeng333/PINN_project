import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import BzEncoder, MFLDataset, project_path, set_seed, signal_shape_info  # noqa: E402


TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'
CURRENT_BASELINE_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_grid_candidate_metrics.csv'
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'starconvex_radial_shape_candidate'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_starconvex_radial_shape_candidate_summary.txt'
ORACLE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_starconvex_radial_shape_oracle_metrics.csv'
SCREENING_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_starconvex_radial_shape_screening.csv'
CANDIDATE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_starconvex_radial_shape_candidate_metrics.csv'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'starconvex_radial_shape_candidate'

SEEDS = [42, 123, 2026]
SCREENING_SEED = 42
K_VALUES = [16, 32]
EPOCHS = 50
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 1e-3
LATENT_DIM = 64
MASK_THRESHOLD_NORM = 0.5
TRAIN_SELECTION_THRESHOLD = 0.5
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
TEMPERATURE = 0.06
MAX_RADIUS = 1.75
CURRENT_BASELINE_THRESHOLD = 0.90
POS_WEIGHT_CAP = 8.0

METRIC_KEYS = [
    'iou',
    'dice',
    'area_error',
    'center_error',
    'pred_area_zero',
    'pred_area_lt_true',
    'pred_area_gt_true',
    'composite',
]


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORACLE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_npz(path):
    data = np.load(project_path(path), allow_pickle=False)
    raw_mu = data['mu_maps'].astype(np.float32)
    if float(np.nanmax(raw_mu)) > 2.0:
        mu_norm = raw_mu / 1000.0
        mask_rule = 'target_mu_raw < 500'
    else:
        mu_norm = raw_mu
        mask_rule = 'target_mu_norm < 0.5'
    return {
        'signals': data['signals'].astype(np.float32),
        'mu_norm': mu_norm.astype(np.float32),
        'masks': mu_norm < MASK_THRESHOLD_NORM,
        'defect_types': data['defect_types'].astype(str),
        'x': data['x'].astype(np.float32),
        'y': data['y'].astype(np.float32),
        'mask_rule': mask_rule,
    }


def normalized_grids(x, y):
    x_norm = 2.0 * (x - float(x.min())) / (float(x.max()) - float(x.min())) - 1.0
    y_norm = 2.0 * (y - float(y.min())) / (float(y.max()) - float(y.min())) - 1.0
    xx, yy = np.meshgrid(x_norm.astype(np.float32), y_norm.astype(np.float32))
    return xx.astype(np.float32), yy.astype(np.float32)


def torch_grid(x, y, device):
    xx, yy = normalized_grids(x, y)
    return torch.from_numpy(xx).to(device), torch.from_numpy(yy).to(device)


def safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


def safe_nanstd(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    if finite.size <= 1:
        return 0.0
    return float(finite.std(ddof=1))


def mask_center(mask, x_grid, y_grid):
    if not np.any(mask):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([float(x_grid[mask].mean()), float(y_grid[mask].mean())], dtype=np.float32)


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


def get_area_edges(masks, x, y):
    dx = float(abs(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = float(abs(y[1] - y[0])) if len(y) > 1 else 1.0
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64) * dx * dy
    return np.quantile(areas, [1 / 3, 2 / 3])


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


def low_signal_indices_from_raw(signals):
    raw = signals.reshape(signals.shape[0], -1)
    values = np.max(np.abs(raw), axis=1)
    threshold = np.quantile(values, 1 / 3)
    return {int(idx) for idx, value in enumerate(values) if float(value) <= float(threshold)}


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error']:
            summary[key] = float('nan')
        summary.update({'pred_area_zero': 0, 'pred_area_lt_true': 0, 'pred_area_gt_true': 0, 'composite': float('nan')})
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error']:
        summary[key] = safe_nanmean([float(row[key]) for row in rows])
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    return summary


def metric_row(candidate, seed, split, group_type, group, k_value, threshold, summary):
    row = {
        'candidate': candidate,
        'seed': seed,
        'split': split,
        'group_type': group_type,
        'group': group,
        'K': k_value,
        'threshold': threshold,
        'n': summary['n'],
    }
    for key in METRIC_KEYS:
        row[key] = summary[key]
    return row


def summarize_candidate(sample_rows, candidate, seed, split, k_value, threshold):
    rows = []
    groups = [('overall', 'all', sample_rows)]
    for group in ['small', 'medium', 'large']:
        groups.append(('area_bin', group, [row for row in sample_rows if row['area_bin'] == group]))
    for group in ['low_signal', 'non_low_signal']:
        groups.append(('signal_bin', group, [row for row in sample_rows if row['signal_bin'] == group]))
    for defect_type in sorted({row['defect_type'] for row in sample_rows}):
        groups.append(('defect_type', defect_type, [row for row in sample_rows if row['defect_type'] == defect_type]))
    for group_type, group, selected in groups:
        rows.append(metric_row(candidate, seed, split, group_type, group, k_value, threshold, summarize_samples(selected)))
    return rows


def build_sample_rows(candidate, seed, split, pred_masks, true_masks, data, area_edges, low_signal_indices, k_value, threshold):
    x_grid, y_grid = np.meshgrid(data['x'], data['y'])
    rows = []
    for idx in range(len(true_masks)):
        metrics = compute_mask_metrics(pred_masks[idx], true_masks[idx], x_grid, y_grid)
        metrics.update({
            'candidate': candidate,
            'seed': seed,
            'split': split,
            'sample_index': idx,
            'defect_type': str(data['defect_types'][idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if idx in low_signal_indices else 'non_low_signal',
            'K': k_value,
            'threshold': threshold,
        })
        rows.append(metrics)
    return rows


def aggregate_seed_rows(rows, candidate, split, k_value, threshold):
    out = []
    groups = sorted({
        (row['group_type'], row['group'])
        for row in rows
        if row['candidate'] == candidate and row['split'] == split and row['K'] == k_value and str(row['threshold']) == str(threshold)
    })
    for group_type, group in groups:
        selected = [
            row for row in rows
            if row['candidate'] == candidate and row['split'] == split and row['K'] == k_value
            and str(row['threshold']) == str(threshold) and row['group_type'] == group_type and row['group'] == group
        ]
        mean_row = {
            'candidate': f'{candidate}_mean',
            'seed': 'mean',
            'split': split,
            'group_type': group_type,
            'group': group,
            'K': k_value,
            'threshold': threshold,
            'n': selected[0]['n'] if selected else 0,
        }
        std_row = dict(mean_row)
        std_row['candidate'] = f'{candidate}_std'
        std_row['seed'] = 'sample_std'
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in selected]
            mean_row[key] = safe_nanmean(values)
            std_row[key] = safe_nanstd(values)
        out.extend([mean_row, std_row])
    return out


def find_row(rows, candidate, group_type='overall', group='all', split=None, k_value=None, threshold=None):
    selected = [
        row for row in rows
        if row['candidate'] == candidate and row['group_type'] == group_type and row['group'] == group
    ]
    if split is not None:
        selected = [row for row in selected if row['split'] == split]
    if k_value is not None:
        selected = [row for row in selected if str(row['K']) == str(k_value)]
    if threshold is not None:
        selected = [row for row in selected if str(row['threshold']) == str(threshold)]
    if not selected:
        raise KeyError((candidate, group_type, group, split, k_value, threshold))
    return selected[0]


def current_baseline_rows():
    with open(CURRENT_BASELINE_METRICS, newline='', encoding='utf-8') as f:
        source_rows = list(csv.DictReader(f))
    rows = []
    for row in source_rows:
        if row['candidate'] not in ('mask_boundary_grid_test_mean', 'mask_boundary_grid_test_std'):
            continue
        if row['split'] != 'test':
            continue
        out = {
            'candidate': 'current_grid_decoder_baseline_mean' if row['candidate'].endswith('_mean') else 'current_grid_decoder_baseline_std',
            'seed': row['seed'],
            'split': 'test',
            'group_type': row['group_type'],
            'group': row['group'],
            'K': 'current_baseline',
            'threshold': CURRENT_BASELINE_THRESHOLD,
            'n': row['n'],
        }
        for key in METRIC_KEYS:
            out[key] = row[key]
        rows.append(out)
    return rows


def read_baseline_val_reference():
    with open(CURRENT_BASELINE_METRICS, newline='', encoding='utf-8') as f:
        source_rows = list(csv.DictReader(f))
    rows = []
    for row in source_rows:
        if row['candidate'] != 'mask_boundary_grid_val_scan_mean':
            continue
        if row['split'] == 'val' and abs(float(row['threshold']) - CURRENT_BASELINE_THRESHOLD) < 1e-9:
            out = dict(row)
            out['K'] = 'current_baseline'
            rows.append(out)
    return rows


def oracle_radial_for_mask(mask, x_norm, y_norm, k_value):
    if not np.any(mask):
        return np.array([0.0, 0.0], dtype=np.float32), np.zeros((k_value,), dtype=np.float32)
    cx = float(x_norm[mask].mean())
    cy = float(y_norm[mask].mean())
    dx = x_norm[mask] - cx
    dy = y_norm[mask] - cy
    radii_pixels = np.sqrt(dx * dx + dy * dy)
    angles = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    bins = np.floor((angles / (2.0 * np.pi)) * k_value + 0.5).astype(np.int64) % k_value
    radii = np.zeros((k_value,), dtype=np.float32)
    has_bin = np.zeros((k_value,), dtype=bool)
    for b, r in zip(bins, radii_pixels):
        if r > radii[b]:
            radii[b] = float(r)
            has_bin[b] = True
    if not np.all(has_bin):
        known = np.flatnonzero(has_bin)
        if known.size == 0:
            radii[:] = 0.0
        elif known.size == 1:
            radii[:] = radii[known[0]]
        else:
            known_ext = np.concatenate([known - k_value, known, known + k_value])
            radius_ext = np.concatenate([radii[known], radii[known], radii[known]])
            radii = np.interp(np.arange(k_value), known_ext, radius_ext).astype(np.float32)
    return np.array([cx, cy], dtype=np.float32), radii.astype(np.float32)


def rasterize_numpy(center, radii, x_norm, y_norm):
    k_value = len(radii)
    dx = x_norm - float(center[0])
    dy = y_norm - float(center[1])
    dist = np.sqrt(dx * dx + dy * dy)
    angles = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    pos = angles / (2.0 * np.pi) * k_value
    i0 = np.floor(pos).astype(np.int64) % k_value
    frac = pos - np.floor(pos)
    i1 = (i0 + 1) % k_value
    radius_theta = (1.0 - frac) * radii[i0] + frac * radii[i1]
    return dist <= radius_theta


def oracle_masks(data, k_value):
    x_norm, y_norm = normalized_grids(data['x'], data['y'])
    preds = np.zeros_like(data['masks'], dtype=bool)
    centers = np.zeros((len(preds), 2), dtype=np.float32)
    radii_all = np.zeros((len(preds), k_value), dtype=np.float32)
    for idx, mask in enumerate(data['masks']):
        center, radii = oracle_radial_for_mask(mask, x_norm, y_norm, k_value)
        centers[idx] = center
        radii_all[idx] = radii
        preds[idx] = rasterize_numpy(center, radii, x_norm, y_norm)
    return preds, centers, radii_all


def evaluate_oracle_split(data, split, k_value, low_signal_indices):
    pred_masks, _, _ = oracle_masks(data, k_value)
    area_edges = get_area_edges(data['masks'], data['x'], data['y'])
    sample_rows = build_sample_rows('oracle_starconvex_radial', 'oracle', split, pred_masks, data['masks'], data, area_edges, low_signal_indices, k_value, 'binary')
    return summarize_candidate(sample_rows, 'oracle_starconvex_radial', 'oracle', split, k_value, 'binary'), sample_rows


class StarConvexRadialModel(nn.Module):
    def __init__(self, signal_length, signal_channels=1, latent_dim=64, k_value=16, max_radius=MAX_RADIUS):
        super().__init__()
        self.k_value = int(k_value)
        self.max_radius = float(max_radius)
        self.encoder = BzEncoder(signal_length=signal_length, signal_channels=signal_channels, latent_dim=latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2 + self.k_value),
        )

    def forward(self, signals, x_norm_grid, y_norm_grid):
        latent = self.encoder(signals)
        raw = self.head(latent)
        center = torch.tanh(raw[:, :2])
        radii = 0.02 + self.max_radius * torch.sigmoid(raw[:, 2:])
        return rasterize_torch(center, radii, x_norm_grid, y_norm_grid, TEMPERATURE), center, radii


def rasterize_torch(center, radii, x_norm_grid, y_norm_grid, temperature):
    batch_size, k_value = radii.shape
    x_flat = x_norm_grid.reshape(-1)
    y_flat = y_norm_grid.reshape(-1)
    dx = x_flat.unsqueeze(0) - center[:, 0:1]
    dy = y_flat.unsqueeze(0) - center[:, 1:2]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    angles = torch.remainder(torch.atan2(dy, dx) + 2.0 * math.pi, 2.0 * math.pi)
    pos = angles / (2.0 * math.pi) * k_value
    i0 = torch.floor(pos).long() % k_value
    frac = pos - torch.floor(pos)
    i1 = (i0 + 1) % k_value
    r0 = torch.gather(radii, 1, i0)
    r1 = torch.gather(radii, 1, i1)
    radius_theta = (1.0 - frac) * r0 + frac * r1
    return (radius_theta - dist) / temperature


def soft_dice_loss(logits, target_mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target_mask.reshape(target_mask.shape[0], -1)
    intersection = torch.sum(probs_flat * target_flat, dim=1)
    pred_sum = torch.sum(probs_flat, dim=1)
    target_sum = torch.sum(target_flat, dim=1)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return torch.mean(1.0 - dice)


def mask_loss(logits, target_mask, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target_mask, pos_weight=pos_weight)
    dice = soft_dice_loss(logits, target_mask)
    return bce + dice, bce, dice


def make_loader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator if shuffle else None)


def compute_pos_weight(dataset):
    masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    pos = float(masks.sum())
    neg = float(masks.size - masks.sum())
    raw = np.sqrt(neg / max(pos, 1.0))
    return float(min(raw, POS_WEIGHT_CAP)), float(pos / masks.size)


@torch.no_grad()
def predict_prob_maps(model, dataset, x_grid_t, y_grid_t, device):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    probs = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        logits, _, _ = model(signals, x_grid_t, y_grid_t)
        batch_probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, sample_idx_tensor in enumerate(indices):
            idx = int(sample_idx_tensor.item())
            probs[idx] = batch_probs[batch_pos]
            true_masks[idx] = batch_true[batch_pos]
    return probs, true_masks


def evaluate_model_threshold(model, dataset, raw_data, x_grid_t, y_grid_t, device, area_edges, low_signal_indices, k_value, threshold, seed, split, candidate):
    prob_maps, true_masks = predict_prob_maps(model, dataset, x_grid_t, y_grid_t, device)
    pred_masks = prob_maps >= float(threshold)
    sample_rows = build_sample_rows(candidate, seed, split, pred_masks, true_masks, raw_data, area_edges, low_signal_indices, k_value, threshold)
    return summarize_candidate(sample_rows, candidate, seed, split, k_value, threshold), sample_rows, prob_maps


def evaluate_model_selection(model, dataset, raw_data, x_grid_t, y_grid_t, device, area_edges, k_value):
    rows, _, _ = evaluate_model_threshold(model, dataset, raw_data, x_grid_t, y_grid_t, device, area_edges, set(), k_value, TRAIN_SELECTION_THRESHOLD, 'selection', 'val', 'selection')
    return find_row(rows, 'selection', 'overall', 'all', split='val', k_value=k_value, threshold=TRAIN_SELECTION_THRESHOLD)


def train_one_seed(seed, k_value, device, pos_weight_value):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    val_raw = load_raw_npz(VAL_DATA)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    model = StarConvexRadialModel(signal_length, signal_channels, LATENT_DIM, k_value).to(device)
    x_grid_t, y_grid_t = torch_grid(train_dataset.x, train_dataset.y, device)
    val_area_edges = get_area_edges(val_raw['masks'], val_raw['x'], val_raw['y'])
    loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_starconvex_radial_K{k_value}_seed{seed}.pt'
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_bce = total_dice = total_samples = 0.0
        for signals, mu_targets, _indices in loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(signals, x_grid_t, y_grid_t)
            loss, bce, dice = mask_loss(logits, target_mask, pos_weight)
            loss.backward()
            optimizer.step()
            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
            total_samples += batch_size
        val_summary = evaluate_model_selection(model, val_dataset, val_raw, x_grid_t, y_grid_t, device, val_area_edges, k_value)
        score = float(val_summary['composite'])
        if score > best_score:
            best_score = score
            best_info = {
                'seed': seed,
                'K': k_value,
                'epoch': epoch,
                'selection_score': score,
                'val_iou': float(val_summary['iou']),
                'val_dice': float(val_summary['dice']),
                'val_area_error': float(val_summary['area_error']),
                'val_center_error': float(val_summary['center_error']),
                'val_pred_area_zero': float(val_summary['pred_area_zero']),
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': {
                    'model': 'starconvex_radial_shape_model',
                    'dataset': 'v3_complex',
                    'seed': seed,
                    'K': k_value,
                    'temperature': TEMPERATURE,
                    'max_radius': MAX_RADIUS,
                    'latent_dim': LATENT_DIM,
                    'epochs': EPOCHS,
                    'loss': 'BCEWithLogits + soft Dice',
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
                    'signal_channels': signal_channels,
                },
                'signal_mean': float(train_dataset.signal_mean),
                'signal_std': float(train_dataset.signal_std),
                'epoch': epoch,
                'selection_score': score,
                'val_metrics': best_info,
            }, best_path)
        print(
            f"K={k_value} seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"loss={total_loss / total_samples:.6e} | bce={total_bce / total_samples:.6e} | "
            f"dice_loss={total_dice / total_samples:.6e} | val_iou={float(val_summary['iou']):.6e} | "
            f"val_dice={float(val_summary['dice']):.6e} | val_area_error={float(val_summary['area_error']):.6e} | score={score:.6e}"
        )
    return best_path, best_info


def load_starconvex_checkpoint(path, signal_length, signal_channels, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint['args']
    model = StarConvexRadialModel(signal_length, signal_channels, LATENT_DIM, int(args['K']), float(args.get('max_radius', MAX_RADIUS))).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_checkpoints(checkpoints, k_value, split, data_path, thresholds, device, area_edges, low_signal_indices, candidate):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    prob_cache = {}
    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu')
        dataset = MFLDataset(data_path, signal_mean=float(checkpoint['signal_mean']), signal_std=float(checkpoint['signal_std']))
        raw_data = load_raw_npz(data_path)
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        model, _ = load_starconvex_checkpoint(project_path(checkpoint_path), signal_length, signal_channels, device)
        x_grid_t, y_grid_t = torch_grid(dataset.x, dataset.y, device)
        prob_maps, true_masks = predict_prob_maps(model, dataset, x_grid_t, y_grid_t, device)
        prob_cache[seed] = (prob_maps, true_masks, raw_data)
        for threshold in thresholds:
            pred_masks = prob_maps >= float(threshold)
            sample_rows = build_sample_rows(candidate, seed, split, pred_masks, true_masks, raw_data, area_edges, low_signal_indices, k_value, threshold)
            sample_rows_by_seed_threshold[(seed, threshold)] = sample_rows
            metric_rows.extend(summarize_candidate(sample_rows, candidate, seed, split, k_value, threshold))
    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows(metric_rows, candidate, split, k_value, threshold))
    return metric_rows, sample_rows_by_seed_threshold, prob_cache


def write_rows(path, rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'K', 'threshold', 'n', *METRIC_KEYS]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def oracle_passes(oracle_rows, baseline_rows):
    for k_value in K_VALUES:
        polygon = find_row(oracle_rows, 'oracle_starconvex_radial', 'defect_type', 'polygon', split='test', k_value=k_value, threshold='binary')
        rotated = find_row(oracle_rows, 'oracle_starconvex_radial', 'defect_type', 'rotated_rect', split='test', k_value=k_value, threshold='binary')
        base_polygon = find_row(baseline_rows, 'current_grid_decoder_baseline_mean', 'defect_type', 'polygon')
        base_rotated = find_row(baseline_rows, 'current_grid_decoder_baseline_mean', 'defect_type', 'rotated_rect')
        if (
            float(polygon['iou']) >= float(base_polygon['iou'])
            and float(rotated['iou']) >= float(base_rotated['iou'])
            and float(polygon['dice']) >= float(base_polygon['dice'])
            and float(rotated['dice']) >= float(base_rotated['dice'])
        ):
            return True
    return False


def screening_positive(screening_rows, best_k):
    baseline_val = read_baseline_val_reference()
    base_overall = find_row(baseline_val, 'mask_boundary_grid_val_scan_mean', 'overall', 'all', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    base_polygon = find_row(baseline_val, 'mask_boundary_grid_val_scan_mean', 'defect_type', 'polygon', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    base_rotated = find_row(baseline_val, 'mask_boundary_grid_val_scan_mean', 'defect_type', 'rotated_rect', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    cand_overall = find_row(screening_rows, 'starconvex_radial_screening', 'overall', 'all', split='val', k_value=best_k, threshold=TRAIN_SELECTION_THRESHOLD)
    cand_polygon = find_row(screening_rows, 'starconvex_radial_screening', 'defect_type', 'polygon', split='val', k_value=best_k, threshold=TRAIN_SELECTION_THRESHOLD)
    cand_rotated = find_row(screening_rows, 'starconvex_radial_screening', 'defect_type', 'rotated_rect', split='val', k_value=best_k, threshold=TRAIN_SELECTION_THRESHOLD)
    return bool(
        float(cand_overall['iou']) >= float(base_overall['iou'])
        and float(cand_overall['dice']) >= float(base_overall['dice'])
        and float(cand_overall['area_error']) <= float(base_overall['area_error']) + 0.02
        and float(cand_polygon['iou']) >= float(base_polygon['iou'])
        and float(cand_rotated['iou']) >= float(base_rotated['iou'])
    )


def select_threshold(rows, k_value):
    baseline = find_row(current_baseline_rows(), 'current_grid_decoder_baseline_mean', 'overall', 'all')
    means = [
        row for row in rows
        if row['candidate'] == 'starconvex_radial_candidate_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
        and str(row['K']) == str(k_value)
    ]
    eligible = [
        row for row in means
        if float(row['iou']) >= float(baseline['iou']) and float(row['dice']) >= float(baseline['dice'])
    ]
    if eligible:
        return min(eligible, key=lambda row: (float(row['area_error']), -float(row['composite']), float(row['pred_area_zero'])))
    return max(means, key=lambda row: float(row['composite']))


def metric_with_std(rows, candidate, group_type, group, metric):
    mean = find_row(rows, f'{candidate}_mean', group_type, group)
    std = find_row(rows, f'{candidate}_std', group_type, group)
    return f"{float(mean[metric]):.4f} +/- {float(std[metric]):.4f}"


def format_comparison(rows, candidate):
    lines = [
        '| group | baseline IoU | baseline Dice | baseline area_error | baseline pred_area=0 | candidate IoU | candidate Dice | candidate area_error | candidate pred_area=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group_type, groups in [
        ('overall', ['all']),
        ('area_bin', ['small', 'medium', 'large']),
        ('signal_bin', ['low_signal', 'non_low_signal']),
        ('defect_type', ['polygon', 'rotated_rect', 'multi_defect']),
    ]:
        for group in groups:
            base = find_row(rows, 'current_grid_decoder_baseline_mean', group_type, group)
            cand = find_row(rows, f'{candidate}_mean', group_type, group)
            lines.append(
                f"| {group} | {float(base['iou']):.4f} | {float(base['dice']):.4f} | {float(base['area_error']):.4f} | {float(base['pred_area_zero']):.2f} | "
                f"{float(cand['iou']):.4f} | {float(cand['dice']):.4f} | {float(cand['area_error']):.4f} | {float(cand['pred_area_zero']):.2f} |"
            )
    return '\n'.join(lines)


def write_summary(oracle_rows, screening_rows, candidate_rows, oracle_ok, entered_screening, entered_three_seed, best_k, selected_threshold, accepted, mask_rule, preview_count):
    def row_text(rows, candidate, split, k_value):
        try:
            row = find_row(rows, candidate, 'overall', 'all', split=split, k_value=k_value)
            poly = find_row(rows, candidate, 'defect_type', 'polygon', split=split, k_value=k_value)
            rot = find_row(rows, candidate, 'defect_type', 'rotated_rect', split=split, k_value=k_value)
            return (
                f"K={k_value}: overall IoU={float(row['iou']):.4f}, Dice={float(row['dice']):.4f}, area_error={float(row['area_error']):.4f}; "
                f"polygon IoU={float(poly['iou']):.4f}, rotated_rect IoU={float(rot['iou']):.4f}"
            )
        except Exception:
            return f"K={k_value}: not available"

    oracle_lines = [row_text(oracle_rows, 'oracle_starconvex_radial', 'test', k) for k in K_VALUES]
    screening_lines = [row_text(screening_rows, 'starconvex_radial_screening', 'val', k) for k in K_VALUES] if screening_rows else ['screening not run']
    candidate_section = '3-seed stage not run.'
    if entered_three_seed and candidate_rows:
        candidate_section = format_comparison(candidate_rows, 'starconvex_radial_candidate_test')

    baseline = find_row(current_baseline_rows(), 'current_grid_decoder_baseline_mean', 'overall', 'all')
    summary = f"""# v3_complex star-convex radial shape candidate

This RESULT_DRIVEN_EXPERIMENT tests a geometric star-convex radial shape model without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

Target mask rule used: {mask_rule}.

## Stage A: oracle shape capacity

Oracle capacity passed: {oracle_ok}

{chr(10).join('* ' + line for line in oracle_lines)}

## Stage B: seed=42 training screening

Entered seed=42 screening: {entered_screening}

{chr(10).join('* ' + line for line in screening_lines)}

best_K: {best_k}

## Stage C: 3-seed candidate

Entered 3-seed stage: {entered_three_seed}
Selected threshold: {selected_threshold}

CURRENT_BASELINE overall: IoU={float(baseline['iou']):.4f}, Dice={float(baseline['dice']):.4f}, area_error={float(baseline['area_error']):.4f}, pred_area=0={float(baseline['pred_area_zero']):.2f}

{candidate_section}

Accepted by metric gate: {accepted}

Preview PNG count: {preview_count}. Preview PNGs are written to `{PREVIEW_DIR.relative_to(ROOT)}` only if the 3-seed stage is reached.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')


def sample_mean_metrics(sample_rows_by_seed_threshold, seeds, threshold):
    by_index = {}
    for seed in seeds:
        for row in sample_rows_by_seed_threshold[(seed, threshold)]:
            by_index.setdefault(row['sample_index'], []).append(row)
    means = []
    for idx, rows in by_index.items():
        first = rows[0]
        means.append({
            'index': idx,
            'defect_type': first['defect_type'],
            'area_bin': first['area_bin'],
            'signal_bin': first['signal_bin'],
            'mean_iou': safe_nanmean([float(row['iou']) for row in rows]),
            'mean_dice': safe_nanmean([float(row['dice']) for row in rows]),
            'mean_area_error': safe_nanmean([float(row['area_error']) for row in rows]),
        })
    for row in means:
        row['mask_score'] = row['mean_iou'] + row['mean_dice'] - row['mean_area_error']
    return means


def select_preview_samples(means):
    selected = []
    used = set()
    def take(category, rows, n=3):
        count = 0
        for item in rows:
            if item['index'] in used:
                continue
            used.add(item['index'])
            out = dict(item)
            out['category'] = category
            selected.append(out)
            count += 1
            if count == n:
                break
    small_polygon = [row for row in means if row['area_bin'] == 'small' and row['defect_type'] == 'polygon']
    small_polygon.sort(key=lambda row: row['mask_score'], reverse=True)
    take('small_polygon_best', small_polygon)
    low_signal = [row for row in means if row['signal_bin'] == 'low_signal']
    low_signal.sort(key=lambda row: row['mask_score'], reverse=True)
    take('low_signal_best', low_signal)
    failures = [row for row in means if row['index'] not in used]
    failures.sort(key=lambda row: (row['mask_score'], row['mean_iou'], -row['mean_area_error']))
    take('starconvex_failure', failures)
    remaining = [row for row in means if row['index'] not in used]
    if remaining:
        median = float(np.median([row['mask_score'] for row in remaining]))
        ordinary = [row for row in remaining if row['area_bin'] == 'medium'] or remaining
        ordinary.sort(key=lambda row: abs(row['mask_score'] - median))
        take('ordinary_medium', ordinary)
    return selected


def safe_name(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('_')


def generate_previews(selected, prob_cache, sample_rows_by_seed_threshold, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    first_data = next(iter(prob_cache.values()))[2]
    x_grid, y_grid = np.meshgrid(first_data['x'], first_data['y'])
    extent = [float(first_data['x'].min()), float(first_data['x'].max()), float(first_data['y'].min()), float(first_data['y'].max())]
    rows_by_seed_index = {
        (seed, row['sample_index']): row
        for seed in SEEDS
        for row in sample_rows_by_seed_threshold[(seed, threshold)]
    }
    written = []
    for item in selected:
        idx = int(item['index'])
        fig, axes = plt.subplots(len(SEEDS), 4, figsize=(13.5, 9.0), constrained_layout=True)
        fig.suptitle(
            f"{item['category']} | sample {idx} | type={item['defect_type']} | "
            f"mean IoU={item['mean_iou']:.3f}, Dice={item['mean_dice']:.3f}, area_err={item['mean_area_error']:.3f}",
            fontsize=11,
        )
        im = None
        for row_idx, seed in enumerate(SEEDS):
            prob_maps, true_masks, data = prob_cache[seed]
            prob = prob_maps[idx]
            true = true_masks[idx]
            pred = prob >= threshold
            metrics = rows_by_seed_index[(seed, idx)]
            row_axes = axes[row_idx]
            row_axes[0].imshow(true, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
            row_axes[0].set_title('true mask')
            im = row_axes[1].imshow(prob, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
            row_axes[1].set_title('mask probability')
            row_axes[2].imshow(pred, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
            row_axes[2].set_title(f'pred mask >= {threshold:.2f}')
            row_axes[3].imshow(true, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1, alpha=0.25)
            row_axes[3].contour(x_grid, y_grid, true.astype(float), levels=[0.5], colors=['lime'], linewidths=1.2)
            if pred.any():
                row_axes[3].contour(x_grid, y_grid, pred.astype(float), levels=[0.5], colors=['red'], linewidths=1.2)
            row_axes[3].set_title('overlay: true green, pred red')
            row_axes[0].set_ylabel(f"seed {seed}\nIoU={float(metrics['iou']):.3f}\nDice={float(metrics['dice']):.3f}\narea={float(metrics['area_error']):.3f}", fontsize=9)
            for ax in row_axes:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(im, ax=axes[:, 1], fraction=0.035, pad=0.02)
        path = PREVIEW_DIR / f"{safe_name(item['category'])}_sample{idx:03d}_{safe_name(item['defect_type'])}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def candidate_accepts(candidate_rows):
    overall = improvement(candidate_rows, 'overall', 'all')
    small = improvement(candidate_rows, 'area_bin', 'small')
    low = improvement(candidate_rows, 'signal_bin', 'low_signal')
    polygon = improvement(candidate_rows, 'defect_type', 'polygon')
    rotated = improvement(candidate_rows, 'defect_type', 'rotated_rect')
    return bool(
        overall['iou_not_down'] and overall['dice_not_down'] and overall['area_error_close'] and overall['pred_area_zero_not_up']
        and small['iou_not_down'] and low['iou_not_down'] and polygon['iou_not_down'] and rotated['iou_not_down']
    )


def improvement(rows, group_type, group):
    base = find_row(rows, 'current_grid_decoder_baseline_mean', group_type, group)
    cand = find_row(rows, 'starconvex_radial_candidate_test_mean', group_type, group)
    return {
        'iou_not_down': float(cand['iou']) >= float(base['iou']) - 1e-6,
        'dice_not_down': float(cand['dice']) >= float(base['dice']) - 1e-6,
        'area_error_close': float(cand['area_error']) <= float(base['area_error']) + 0.02,
        'pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
    }


def main():
    ensure_outputs()
    train_raw = load_raw_npz(TRAIN_DATA)
    val_raw = load_raw_npz(VAL_DATA)
    test_raw = load_raw_npz(TEST_DATA)
    mask_rule = train_raw['mask_rule']
    if val_raw['mask_rule'] != mask_rule or test_raw['mask_rule'] != mask_rule:
        raise ValueError('Inconsistent mask target rule across splits')
    baseline_rows = current_baseline_rows()

    oracle_rows = []
    for k_value in K_VALUES:
        val_low = low_signal_indices_from_raw(val_raw['signals'])
        test_low = load_low_signal_indices()
        val_rows, _ = evaluate_oracle_split(val_raw, 'val', k_value, val_low)
        test_rows, _ = evaluate_oracle_split(test_raw, 'test', k_value, test_low)
        oracle_rows.extend(val_rows + test_rows)
    write_rows(ORACLE_METRICS_PATH, oracle_rows)
    oracle_ok = oracle_passes(oracle_rows, baseline_rows)
    print(f'Oracle capacity passed: {oracle_ok}')

    screening_rows = []
    candidate_rows = []
    best_k = None
    selected_threshold = 'N/A'
    accepted = False
    preview_count = 0
    entered_screening = bool(oracle_ok)
    entered_three_seed = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset_for_weight = MFLDataset(TRAIN_DATA)
    pos_weight, mask_fraction = compute_pos_weight(train_dataset_for_weight)
    print(f'Using device: {device}; pos_weight={pos_weight:.6f}; mask_fraction={mask_fraction:.6f}')

    checkpoint_paths = {}
    screening_infos = {}
    if entered_screening:
        val_data_raw = load_raw_npz(VAL_DATA)
        test_data_raw = load_raw_npz(TEST_DATA)
        val_edges = get_area_edges(val_data_raw['masks'], val_data_raw['x'], val_data_raw['y'])
        test_edges = get_area_edges(test_data_raw['masks'], test_data_raw['x'], test_data_raw['y'])
        val_low = low_signal_indices_from_raw(val_data_raw['signals'])
        test_low = load_low_signal_indices()
        for k_value in K_VALUES:
            path, info = train_one_seed(SCREENING_SEED, k_value, device, pos_weight)
            checkpoint_paths[(k_value, SCREENING_SEED)] = path
            screening_infos[k_value] = info
            ckpt = {SCREENING_SEED: str(path.relative_to(ROOT))}
            val_rows, _, _ = evaluate_checkpoints(ckpt, k_value, 'val', VAL_DATA, [TRAIN_SELECTION_THRESHOLD], device, val_edges, val_low, 'starconvex_radial_screening')
            test_rows, _, _ = evaluate_checkpoints(ckpt, k_value, 'test', TEST_DATA, [TRAIN_SELECTION_THRESHOLD], device, test_edges, test_low, 'starconvex_radial_screening')
            screening_rows.extend(val_rows + test_rows)
        write_rows(SCREENING_METRICS_PATH, screening_rows)
        best_row = max(
            [row for row in screening_rows if row['candidate'] == 'starconvex_radial_screening' and row['split'] == 'val' and row['group_type'] == 'overall' and row['group'] == 'all'],
            key=lambda row: float(row['composite']),
        )
        best_k = int(best_row['K'])
        entered_three_seed = screening_positive(screening_rows, best_k)
        print(f'Best K={best_k}; entered 3-seed: {entered_three_seed}')
    else:
        write_rows(SCREENING_METRICS_PATH, screening_rows)

    if entered_three_seed:
        for seed in [seed for seed in SEEDS if seed != SCREENING_SEED]:
            path, info = train_one_seed(seed, best_k, device, pos_weight)
            checkpoint_paths[(best_k, seed)] = path
        final_checkpoints = {seed: str(checkpoint_paths[(best_k, seed)].relative_to(ROOT)) for seed in SEEDS}
        val_edges = get_area_edges(val_raw['masks'], val_raw['x'], val_raw['y'])
        test_edges = get_area_edges(test_raw['masks'], test_raw['x'], test_raw['y'])
        val_low = low_signal_indices_from_raw(val_raw['signals'])
        test_low = load_low_signal_indices()
        val_rows, _, _ = evaluate_checkpoints(final_checkpoints, best_k, 'val', VAL_DATA, THRESHOLDS, device, val_edges, val_low, 'starconvex_radial_candidate_val_scan')
        selected = select_threshold(val_rows, best_k)
        selected_threshold = float(selected['threshold'])
        test_rows, sample_rows_by_seed_threshold, prob_cache = evaluate_checkpoints(final_checkpoints, best_k, 'test', TEST_DATA, [selected_threshold], device, test_edges, test_low, 'starconvex_radial_candidate_test')
        candidate_rows = baseline_rows + val_rows + test_rows
        accepted = candidate_accepts(candidate_rows)
        means = sample_mean_metrics(sample_rows_by_seed_threshold, SEEDS, selected_threshold)
        preview_paths = generate_previews(select_preview_samples(means), prob_cache, sample_rows_by_seed_threshold, selected_threshold)
        preview_count = len(preview_paths)
    else:
        candidate_rows = baseline_rows

    write_rows(CANDIDATE_METRICS_PATH, candidate_rows)
    write_summary(oracle_rows, screening_rows, candidate_rows, oracle_ok, entered_screening, entered_three_seed, best_k, selected_threshold, accepted, mask_rule, preview_count)
    print(f'Wrote oracle metrics: {ORACLE_METRICS_PATH}')
    print(f'Wrote screening metrics: {SCREENING_METRICS_PATH}')
    print(f'Wrote candidate metrics: {CANDIDATE_METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Accepted by metric gate: {accepted}')


if __name__ == '__main__':
    main()
