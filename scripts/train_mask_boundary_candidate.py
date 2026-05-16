import csv
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

from train_pinn import (
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
BASELINE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_macro_area_selection_audit_metrics.csv'
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'

SEEDS = [42, 123, 2026]
EPOCHS = 50
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 1e-3
LATENT_DIM = 64
MASK_THRESHOLD_NORM = 0.5
MASK_PROB_THRESHOLD = 0.5
POS_WEIGHT_CAP = 8.0

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_boundary_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_candidate_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_boundary_candidate_summary.txt'

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


class MaskBoundaryModel(nn.Module):
    def __init__(self, signal_length, signal_channels=1, latent_dim=64, coord_feature_dim=84):
        super().__init__()
        self.signal_channels = signal_channels
        self.latent_dim = latent_dim
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


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_loader(dataset, batch_size, shuffle=False, seed=42):
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


def mask_loss(logits, target_mask, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target_mask, pos_weight=pos_weight)
    dice = soft_dice_loss(logits, target_mask)
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
        summary[key] = float(np.nanmean([float(row[key]) for row in rows]))
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    return summary


@torch.no_grad()
def evaluate_model(model, dataset, coords, device, area_edges, low_signal_indices):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    rows = []
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        logits = model(signals, coords)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        true_masks = (mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM)
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            pred_mask = probs[batch_pos] >= MASK_PROB_THRESHOLD
            metrics = compute_mask_metrics(pred_mask, true_masks[batch_pos], x_grid, y_grid)
            metrics.update({
                'sample_index': sample_idx,
                'defect_type': str(dataset.defect_types[sample_idx]),
                'area_bin': area_bin(float(metrics['true_area']), area_edges),
                'signal_bin': 'low_signal' if sample_idx in low_signal_indices else 'non_low_signal',
            })
            rows.append(metrics)
    rows.sort(key=lambda row: row['sample_index'])
    return rows


def summarize_candidate(sample_rows, candidate, seed):
    rows = []
    overall = summarize_samples(sample_rows)
    area_summaries = {
        group: summarize_samples([row for row in sample_rows if row['area_bin'] == group])
        for group in ['small', 'medium', 'large']
    }
    macro_area_composite = float(np.nanmean([
        area_summaries[group]['composite']
        for group in ['small', 'medium', 'large']
    ]))
    rows.append(metric_row(candidate, seed, 'overall', 'all', overall, macro_area_composite))
    for group in ['small', 'medium', 'large']:
        rows.append(metric_row(candidate, seed, 'area_bin', group, area_summaries[group], macro_area_composite))
    for group in ['low_signal', 'non_low_signal']:
        selected = [row for row in sample_rows if row['signal_bin'] == group]
        rows.append(metric_row(candidate, seed, 'signal_bin', group, summarize_samples(selected), macro_area_composite))
    return rows


def metric_row(candidate, seed, group_type, group, summary, macro_area_composite):
    row = {
        'candidate': candidate,
        'seed': seed,
        'group_type': group_type,
        'group': group,
        'threshold': f'mask_prob>={MASK_PROB_THRESHOLD}',
        'n': summary['n'],
        'mse': 'N/A',
        'mae': 'N/A',
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


def train_one_seed(seed, device, pos_weight_value):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    val_area_edges = get_area_edges(val_dataset)
    low_signal_indices = set()
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    model = MaskBoundaryModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
    ).to(device)
    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_mask_boundary_seed{seed}.pt'

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_samples = 0
        for signals, mu_targets, _ in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals, coords)
            loss, bce, dice = mask_loss(logits, target_mask, pos_weight)
            loss.backward()
            optimizer.step()
            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
            total_samples += batch_size

        val_rows = evaluate_model(model, val_dataset, coords, device, val_area_edges, low_signal_indices)
        val_summary = summarize_samples(val_rows)
        selection_score = val_summary['composite']
        if selection_score > best_score:
            best_score = selection_score
            best_info = {
                'seed': seed,
                'epoch': epoch,
                'selection_score': selection_score,
                'val_iou': val_summary['iou'],
                'val_dice': val_summary['dice'],
                'val_area_error': val_summary['area_error'],
                'val_center_error': val_summary['center_error'],
                'val_pred_area_zero': val_summary['pred_area_zero'],
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': {
                    'model': 'mask_boundary_model',
                    'dataset': 'v3_complex',
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'loss': 'BCEWithLogits + soft Dice',
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'selection_metric': 'val_iou + val_dice - val_area_error',
                    'threshold': MASK_PROB_THRESHOLD,
                    'signal_channels': signal_channels,
                },
                'signal_mean': float(train_dataset.signal_mean),
                'signal_std': float(train_dataset.signal_std),
                'epoch': epoch,
                'selection_score': float(selection_score),
                'val_metrics': best_info,
            }, best_path)

        print(
            f"seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"loss={total_loss / total_samples:.6e} | "
            f"bce={total_bce / total_samples:.6e} | "
            f"dice_loss={total_dice / total_samples:.6e} | "
            f"val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | "
            f"score={selection_score:.6e}"
        )

    return best_path, best_info


def load_mask_checkpoint(path, signal_length, signal_channels, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundaryModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(path, seed, device, test_area_edges, low_signal_indices):
    checkpoint = torch.load(path, map_location=device)
    train_signal_mean = float(checkpoint['signal_mean'])
    train_signal_std = float(checkpoint['signal_std'])
    dataset = MFLDataset(TEST_DATA, signal_mean=train_signal_mean, signal_std=train_signal_std)
    signal_length, signal_channels = signal_shape_info(dataset.signals)
    model, _ = load_mask_checkpoint(path, signal_length, signal_channels, device)
    coords = build_coord_grid(dataset.x, dataset.y).to(device)
    sample_rows = evaluate_model(model, dataset, coords, device, test_area_edges, low_signal_indices)
    return summarize_candidate(sample_rows, 'mask_boundary_candidate', seed)


def read_baseline_rows():
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(f'Missing baseline metrics: {BASELINE_METRICS_PATH}')
    with open(BASELINE_METRICS_PATH, newline='', encoding='utf-8') as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for row in raw_rows:
        if row['candidate'] != 'current_baseline_composite':
            continue
        out = {
            'candidate': 'current_baseline_composite',
            'seed': int(row['seed']),
            'group_type': row['group_type'],
            'group': row['group'],
            'threshold': 'mu<500',
            'n': int(row['n']),
            'mse': row['mse'],
            'mae': row['mae'],
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
            'macro_area_composite',
        ]:
            out[key] = float(row[key])
        rows.append(out)
    return rows


def aggregate_seed_rows(metric_rows, source_candidate):
    aggregate_rows = []
    groups = sorted({
        (row['group_type'], row['group'])
        for row in metric_rows
        if row['candidate'] == source_candidate
    })
    for group_type, group in groups:
        selected = [
            row for row in metric_rows
            if row['candidate'] == source_candidate and row['group_type'] == group_type and row['group'] == group
        ]
        mean_row = {
            'candidate': f'{source_candidate}_mean',
            'seed': 'mean',
            'group_type': group_type,
            'group': group,
            'threshold': selected[0]['threshold'] if selected else '',
            'n': selected[0]['n'] if selected else 0,
            'mse': 'N/A',
            'mae': 'N/A',
        }
        std_row = dict(mean_row)
        std_row['candidate'] = f'{source_candidate}_std'
        std_row['seed'] = 'sample_std'
        for key in METRIC_KEYS:
            values = np.array([float(row[key]) for row in selected], dtype=np.float64)
            mean_row[key] = float(np.nanmean(values))
            std_row[key] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.extend([mean_row, std_row])
    return aggregate_rows


def write_metrics(rows):
    fieldnames = [
        'candidate',
        'seed',
        'group_type',
        'group',
        'threshold',
        'n',
        'mse',
        'mae',
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


def find_row(rows, candidate, group_type='overall', group='all'):
    return next(
        row for row in rows
        if row['candidate'] == candidate and row['group_type'] == group_type and row['group'] == group
    )


def fmt(value, metric):
    if isinstance(value, str):
        return value
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_table(rows, group_type, groups):
    lines = [
        '| group | candidate | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group in groups:
        base_mean = find_row(rows, 'current_baseline_composite_mean', group_type, group)
        base_std = find_row(rows, 'current_baseline_composite_std', group_type, group)
        mask_mean = find_row(rows, 'mask_boundary_candidate_mean', group_type, group)
        mask_std = find_row(rows, 'mask_boundary_candidate_std', group_type, group)
        lines.append(
            f"| {group} | current composite baseline | "
            f"{metric_with_std(base_mean, base_std, 'iou')} | "
            f"{metric_with_std(base_mean, base_std, 'dice')} | "
            f"{metric_with_std(base_mean, base_std, 'area_error')} | "
            f"{metric_with_std(base_mean, base_std, 'center_error')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | mask-only boundary | "
            f"{metric_with_std(mask_mean, mask_std, 'iou')} | "
            f"{metric_with_std(mask_mean, mask_std, 'dice')} | "
            f"{metric_with_std(mask_mean, mask_std, 'area_error')} | "
            f"{metric_with_std(mask_mean, mask_std, 'center_error')} | "
            f"{metric_with_std(mask_mean, mask_std, 'pred_area_zero')} | "
            f"{metric_with_std(mask_mean, mask_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(mask_mean, mask_std, 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group):
    baseline = find_row(rows, 'current_baseline_composite_mean', group_type, group)
    candidate = find_row(rows, 'mask_boundary_candidate_mean', group_type, group)
    return {
        'iou_up': float(candidate['iou']) > float(baseline['iou']),
        'dice_up': float(candidate['dice']) > float(baseline['dice']),
        'area_error_not_worse': float(candidate['area_error']) <= float(baseline['area_error']) + 0.02,
        'pred_area_zero_down': float(candidate['pred_area_zero']) < float(baseline['pred_area_zero']),
    }


def write_summary(rows, best_infos, checkpoint_paths, pos_weight, mask_fraction, target_check):
    overall = improvement_status(rows, 'overall', 'all')
    small = improvement_status(rows, 'area_bin', 'small')
    medium = improvement_status(rows, 'area_bin', 'medium')
    large = improvement_status(rows, 'area_bin', 'large')
    low = improvement_status(rows, 'signal_bin', 'low_signal')
    small_or_low = (
        (small['iou_up'] and small['dice_up'])
        or (low['iou_up'] and low['dice_up'])
    )
    accepted = bool(
        overall['iou_up']
        and overall['dice_up']
        and overall['pred_area_zero_down']
        and overall['area_error_not_worse']
        and small_or_low
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
        f"* overall: IoU up={overall['iou_up']}, Dice up={overall['dice_up']}, "
        f"area_error not materially worse={overall['area_error_not_worse']}, "
        f"pred_area=0 down={overall['pred_area_zero_down']}",
        f"* small: IoU up={small['iou_up']}, Dice up={small['dice_up']}, "
        f"area_error not materially worse={small['area_error_not_worse']}, "
        f"pred_area=0 down={small['pred_area_zero_down']}",
        f"* medium: IoU up={medium['iou_up']}, Dice up={medium['dice_up']}, "
        f"area_error not materially worse={medium['area_error_not_worse']}, "
        f"pred_area=0 down={medium['pred_area_zero_down']}",
        f"* large: IoU up={large['iou_up']}, Dice up={large['dice_up']}, "
        f"area_error not materially worse={large['area_error_not_worse']}, "
        f"pred_area=0 down={large['pred_area_zero_down']}",
        f"* low_signal: IoU up={low['iou_up']}, Dice up={low['dice_up']}, "
        f"area_error not materially worse={low['area_error_not_worse']}, "
        f"pred_area=0 down={low['pred_area_zero_down']}",
    ]

    summary = f"""# v3_complex mask-only boundary model candidate

This RESULT_DRIVEN_EXPERIMENT trains an independent mask-only boundary model for seeds 42, 123, and 2026. It does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Target check

Raw v3_complex mu range observed before training: min={target_check['raw_mu_min']:.1f}, max={target_check['raw_mu_max']:.1f}. MFLDataset normalizes mu by MU_SCALE=1000, so the training target is `target_mu_norm < 0.5`, equivalent to raw `target_mu < 500`.

Train mask positive fraction: {mask_fraction:.6f}. BCE pos_weight uses sqrt(neg/pos) capped at {POS_WEIGHT_CAP:.1f}; value used: {pos_weight:.6f}.

## Model and loss

The candidate reuses the existing BzEncoder and Fourier coordinate features, then decodes to one mask logit per grid point. It predicts mask probability only and does not predict mu. Training loss is BCEWithLogits + soft Dice. Checkpoint selection uses validation IoU + Dice - area_error. Test evaluation uses fixed mask_prob >= 0.5.

MSE / MAE are not applicable to this mask-only candidate and are recorded as N/A in the metrics CSV.

## Selected checkpoints

{chr(10).join(best_lines)}

## Overall test comparison

{format_table(rows, 'overall', ['all'])}

## Area-bin test comparison

{format_table(rows, 'area_bin', ['small', 'medium', 'large'])}

## Low-signal test comparison

{format_table(rows, 'signal_bin', ['low_signal', 'non_low_signal'])}

## Gate checks

{chr(10).join(status_lines)}

Accepted as formal mask-only candidate direction: {accepted}

If not accepted, no further mask-only structure, loss-weight, threshold, ensemble, or post-processing variant is recommended from this gate.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'accepted': accepted,
        'overall': overall,
        'small': small,
        'medium': medium,
        'large': large,
        'low_signal': low,
    }


def target_check():
    data = np.load(project_path(TRAIN_DATA), allow_pickle=False)
    mu = data['mu_maps']
    return {
        'raw_mu_min': float(mu.min()),
        'raw_mu_max': float(mu.max()),
    }


def main():
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset_for_weight = MFLDataset(TRAIN_DATA)
    pos_weight, mask_fraction = compute_pos_weight(train_dataset_for_weight)
    print(f'mask positive fraction: {mask_fraction:.6f}')
    print(f'pos_weight: {pos_weight:.6f}')

    checkpoint_paths = []
    best_infos = []
    for seed in SEEDS:
        print(f'Training mask boundary candidate seed={seed}')
        checkpoint_path, best_info = train_one_seed(seed, device, pos_weight)
        checkpoint_paths.append(checkpoint_path)
        best_infos.append(best_info)

    test_dataset_for_edges = MFLDataset(TEST_DATA)
    test_area_edges = get_area_edges(test_dataset_for_edges)
    low_signal_indices = load_low_signal_indices()

    candidate_rows = []
    for seed, checkpoint_path in zip(SEEDS, checkpoint_paths):
        candidate_rows.extend(evaluate_checkpoint(checkpoint_path, seed, device, test_area_edges, low_signal_indices))

    baseline_seed_rows = read_baseline_rows()
    all_rows = (
        baseline_seed_rows
        + aggregate_seed_rows(baseline_seed_rows, 'current_baseline_composite')
        + candidate_rows
        + aggregate_seed_rows(candidate_rows, 'mask_boundary_candidate')
    )
    write_metrics(all_rows)
    judgment = write_summary(
        rows=all_rows,
        best_infos=best_infos,
        checkpoint_paths=checkpoint_paths,
        pos_weight=pos_weight,
        mask_fraction=mask_fraction,
        target_check=target_check(),
    )

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f"Accepted: {judgment['accepted']}")


if __name__ == '__main__':
    main()
