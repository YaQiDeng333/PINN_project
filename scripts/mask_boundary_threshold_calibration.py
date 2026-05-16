import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import (
    BzEncoder,
    MFLDataset,
    build_coord_grid,
    feature_mapping,
    project_path,
    signal_shape_info,
)


VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'
BASELINE_METRICS_CANDIDATES = [
    ROOT / 'results' / 'metrics' / 'v3_complex_composite_candidate_per_sample_audit.csv',
    ROOT / 'results' / 'metrics' / 'v3_complex_composite_selection_candidate_metrics.csv',
    ROOT / 'results' / 'metrics' / 'v3_complex_macro_area_selection_audit_metrics.csv',
]
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'

CHECKPOINTS = {
    42: 'checkpoints/mask_boundary_candidate/best_mask_boundary_seed42.pt',
    123: 'checkpoints/mask_boundary_candidate/best_mask_boundary_seed123.pt',
    2026: 'checkpoints/mask_boundary_candidate/best_mask_boundary_seed2026.pt',
}

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
BATCH_SIZE = 8
LATENT_DIM = 64

METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_threshold_calibration_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_boundary_threshold_calibration_summary.txt'

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


def ensure_outputs():
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def check_checkpoints():
    missing = [path for path in CHECKPOINTS.values() if not Path(project_path(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing mask boundary checkpoints: ' + ', '.join(missing))


def find_baseline_metrics_path():
    for path in BASELINE_METRICS_CANDIDATES:
        if path.exists():
            return path
    candidates = ', '.join(str(path) for path in BASELINE_METRICS_CANDIDATES)
    raise FileNotFoundError(f'No baseline metrics CSV found. Checked: {candidates}')


def load_current_baseline_overall_metrics():
    path = find_baseline_metrics_path()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if {'composite_mean_iou', 'composite_mean_dice', 'composite_mean_area_error'}.issubset(fieldnames):
        return {
            'source': str(path.relative_to(ROOT)),
            'iou': safe_nanmean([float(row['composite_mean_iou']) for row in rows]),
            'dice': safe_nanmean([float(row['composite_mean_dice']) for row in rows]),
            'area_error': safe_nanmean([float(row['composite_mean_area_error']) for row in rows]),
        }

    if {'row_type', 'model', 'threshold', 'area_size_bin', 'iou', 'dice', 'area_error'}.issubset(fieldnames):
        selected = [
            row for row in rows
            if row['row_type'] == 'overall'
            and row['area_size_bin'] == 'all'
            and str(row['threshold']) == '500'
            and row['model'] in ('composite_candidate', 'current_baseline_composite')
        ]
        if selected:
            return {
                'source': str(path.relative_to(ROOT)),
                'iou': safe_nanmean([float(row['iou']) for row in selected]),
                'dice': safe_nanmean([float(row['dice']) for row in selected]),
                'area_error': safe_nanmean([float(row['area_error']) for row in selected]),
            }

    if {'candidate', 'group_type', 'group', 'iou', 'dice', 'area_error'}.issubset(fieldnames):
        selected = [
            row for row in rows
            if row['candidate'] == 'current_baseline_composite'
            and row['group_type'] == 'overall'
            and row['group'] == 'all'
        ]
        if selected:
            return {
                'source': str(path.relative_to(ROOT)),
                'iou': safe_nanmean([float(row['iou']) for row in selected]),
                'dice': safe_nanmean([float(row['dice']) for row in selected]),
                'area_error': safe_nanmean([float(row['area_error']) for row in selected]),
            }

    raise ValueError(f'Could not read current composite baseline metrics from {path}')


def find_baseline_comparison_metrics_path():
    required = {
        'seed',
        'candidate',
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
    }
    for path in BASELINE_METRICS_CANDIDATES:
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if not required.issubset(fieldnames):
                continue
            if any(row.get('candidate') == 'current_baseline_composite' for row in reader):
                return path
    candidates = ', '.join(str(path) for path in BASELINE_METRICS_CANDIDATES)
    raise FileNotFoundError(
        'No grouped current_baseline_composite metrics CSV found. '
        f'Checked: {candidates}'
    )


def make_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


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
    masks = dataset.mu_maps < 0.5
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


def load_model(path, signal_length, signal_channels, device):
    checkpoint = torch.load(project_path(path), map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundaryModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_prob_maps(checkpoint_path, dataset_path, device):
    checkpoint_raw = torch.load(project_path(checkpoint_path), map_location='cpu')
    signal_mean = float(checkpoint_raw['signal_mean'])
    signal_std = float(checkpoint_raw['signal_std'])
    dataset = MFLDataset(dataset_path, signal_mean=signal_mean, signal_std=signal_std)
    signal_length, signal_channels = signal_shape_info(dataset.signals)
    model, _ = load_model(checkpoint_path, signal_length, signal_channels, device)
    coords = build_coord_grid(dataset.x, dataset.y).to(device)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)

    for signals, mu_targets, indices in make_loader(dataset):
        logits = model(signals.to(device), coords)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < 0.5
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = probs[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
    return prob_maps, true_masks, dataset


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
    macro_area_composite = float(np.nanmean([
        area_summaries[group]['composite']
        for group in ['small', 'medium', 'large']
    ]))
    rows.append(metric_row(candidate, seed, split, threshold, 'overall', 'all', overall, macro_area_composite))
    for group in ['small', 'medium', 'large']:
        rows.append(metric_row(candidate, seed, split, threshold, 'area_bin', group, area_summaries[group], macro_area_composite))
    for group in ['low_signal', 'non_low_signal']:
        selected = [row for row in sample_rows if row['signal_bin'] == group]
        rows.append(metric_row(candidate, seed, split, threshold, 'signal_bin', group, summarize_samples(selected), macro_area_composite))
    return rows


def aggregate_seed_rows(metric_rows, source_candidate, split, threshold):
    aggregate_rows = []
    groups = sorted({
        (row['group_type'], row['group'])
        for row in metric_rows
        if row['candidate'] == source_candidate and row['split'] == split and float(row['threshold']) == float(threshold)
    })
    for group_type, group in groups:
        selected = [
            row for row in metric_rows
            if row['candidate'] == source_candidate
            and row['split'] == split
            and float(row['threshold']) == float(threshold)
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


def read_baseline_rows():
    baseline_comparison_metrics_path = find_baseline_comparison_metrics_path()
    with open(baseline_comparison_metrics_path, newline='', encoding='utf-8') as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for row in raw_rows:
        if row['candidate'] != 'current_baseline_composite':
            continue
        out = {
            'candidate': 'current_baseline_composite',
            'seed': int(row['seed']),
            'split': 'test',
            'group_type': row['group_type'],
            'group': row['group'],
            'threshold': 'mu<500',
            'n': int(row['n']),
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
    aggregate = aggregate_seed_rows_for_baseline(rows)
    return rows + aggregate


def aggregate_seed_rows_for_baseline(rows):
    aggregate_rows = []
    groups = sorted({(row['group_type'], row['group']) for row in rows})
    for group_type, group in groups:
        selected = [row for row in rows if row['group_type'] == group_type and row['group'] == group]
        mean_row = {
            'candidate': 'current_baseline_composite_mean',
            'seed': 'mean',
            'split': 'test',
            'group_type': group_type,
            'group': group,
            'threshold': 'mu<500',
            'n': selected[0]['n'],
        }
        std_row = dict(mean_row)
        std_row['candidate'] = 'current_baseline_composite_std'
        std_row['seed'] = 'sample_std'
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in selected]
            mean_row[key] = safe_nanmean(values)
            std_row[key] = safe_nanstd(values)
        aggregate_rows.extend([mean_row, std_row])
    return aggregate_rows


def select_threshold(rows, baseline_metrics):
    validation_means = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    eligible = [
        row for row in validation_means
        if float(row['iou']) > baseline_metrics['iou'] and float(row['dice']) > baseline_metrics['dice']
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
        selected = [row for row in selected if str(row['threshold']) == str(threshold)]
    return selected[0]


def fmt(value, metric):
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_comparison_table(rows, group_type, groups, selected_threshold):
    lines = [
        '| group | candidate | threshold | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group in groups:
        base_mean = find_row(rows, 'current_baseline_composite_mean', group_type, group)
        base_std = find_row(rows, 'current_baseline_composite_std', group_type, group)
        mask05_mean = find_row(rows, 'mask_boundary_test_mean', group_type, group, threshold='0.5')
        mask05_std = find_row(rows, 'mask_boundary_test_std', group_type, group, threshold='0.5')
        selected_mean = find_row(rows, 'mask_boundary_test_mean', group_type, group, threshold=str(selected_threshold))
        selected_std = find_row(rows, 'mask_boundary_test_std', group_type, group, threshold=str(selected_threshold))
        lines.append(
            f"| {group} | current composite baseline | mu<500 | "
            f"{metric_with_std(base_mean, base_std, 'iou')} | "
            f"{metric_with_std(base_mean, base_std, 'dice')} | "
            f"{metric_with_std(base_mean, base_std, 'area_error')} | "
            f"{metric_with_std(base_mean, base_std, 'center_error')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | mask-only | 0.50 | "
            f"{metric_with_std(mask05_mean, mask05_std, 'iou')} | "
            f"{metric_with_std(mask05_mean, mask05_std, 'dice')} | "
            f"{metric_with_std(mask05_mean, mask05_std, 'area_error')} | "
            f"{metric_with_std(mask05_mean, mask05_std, 'center_error')} | "
            f"{metric_with_std(mask05_mean, mask05_std, 'pred_area_zero')} | "
            f"{metric_with_std(mask05_mean, mask05_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(mask05_mean, mask05_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | mask-only selected | {selected_threshold:.2f} | "
            f"{metric_with_std(selected_mean, selected_std, 'iou')} | "
            f"{metric_with_std(selected_mean, selected_std, 'dice')} | "
            f"{metric_with_std(selected_mean, selected_std, 'area_error')} | "
            f"{metric_with_std(selected_mean, selected_std, 'center_error')} | "
            f"{metric_with_std(selected_mean, selected_std, 'pred_area_zero')} | "
            f"{metric_with_std(selected_mean, selected_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(selected_mean, selected_std, 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def format_val_scan(rows, baseline_metrics):
    lines = [
        '| threshold | val IoU | val Dice | val area_error | val center_error | val pred_area=0 | selected eligible |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    selected = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
    ]
    for row in sorted(selected, key=lambda item: float(item['threshold'])):
        eligible = float(row['iou']) > baseline_metrics['iou'] and float(row['dice']) > baseline_metrics['dice']
        lines.append(
            f"| {float(row['threshold']):.2f} | {float(row['iou']):.4f} | "
            f"{float(row['dice']):.4f} | {float(row['area_error']):.4f} | "
            f"{float(row['center_error']):.4f} | {float(row['pred_area_zero']):.2f} | {eligible} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group, selected_threshold):
    baseline = find_row(rows, 'current_baseline_composite_mean', group_type, group)
    selected = find_row(rows, 'mask_boundary_test_mean', group_type, group, threshold=str(selected_threshold))
    return {
        'iou_up': float(selected['iou']) > float(baseline['iou']),
        'dice_up': float(selected['dice']) > float(baseline['dice']),
        'area_error_close': float(selected['area_error']) <= float(baseline['area_error']) + 0.05,
        'pred_area_zero_not_much_worse': float(selected['pred_area_zero']) <= float(baseline['pred_area_zero']) + 2.0,
    }


def write_summary(rows, selected_threshold, baseline_metrics):
    overall = improvement_status(rows, 'overall', 'all', selected_threshold)
    small = improvement_status(rows, 'area_bin', 'small', selected_threshold)
    medium = improvement_status(rows, 'area_bin', 'medium', selected_threshold)
    large = improvement_status(rows, 'area_bin', 'large', selected_threshold)
    low_signal = improvement_status(rows, 'signal_bin', 'low_signal', selected_threshold)
    accepted = bool(
        overall['iou_up']
        and overall['dice_up']
        and overall['area_error_close']
        and overall['pred_area_zero_not_much_worse']
        and small['iou_up']
        and small['dice_up']
        and low_signal['iou_up']
        and low_signal['dice_up']
        and small['pred_area_zero_not_much_worse']
        and low_signal['pred_area_zero_not_much_worse']
    )
    status_lines = [
        f"* overall: IoU up={overall['iou_up']}, Dice up={overall['dice_up']}, area_error close={overall['area_error_close']}, pred_area=0 not much worse={overall['pred_area_zero_not_much_worse']}",
        f"* small: IoU up={small['iou_up']}, Dice up={small['dice_up']}, area_error close={small['area_error_close']}, pred_area=0 not much worse={small['pred_area_zero_not_much_worse']}",
        f"* medium: IoU up={medium['iou_up']}, Dice up={medium['dice_up']}, area_error close={medium['area_error_close']}, pred_area=0 not much worse={medium['pred_area_zero_not_much_worse']}",
        f"* large: IoU up={large['iou_up']}, Dice up={large['dice_up']}, area_error close={large['area_error_close']}, pred_area=0 not much worse={large['pred_area_zero_not_much_worse']}",
        f"* low_signal: IoU up={low_signal['iou_up']}, Dice up={low_signal['dice_up']}, area_error close={low_signal['area_error_close']}, pred_area=0 not much worse={low_signal['pred_area_zero_not_much_worse']}",
    ]

    summary = f"""# v3_complex mask-only boundary probability threshold calibration

This FAST_CALIBRATION_GATE loads the three mask-only boundary checkpoints from step 15.1. It performs no training, no model changes, no adaptive threshold, no post-processing, and no ensemble.

Threshold candidates: {', '.join(f'{value:.2f}' for value in THRESHOLDS)}

Selection rule: choose one global validation threshold shared by all seeds. Among thresholds whose validation IoU and Dice remain above the current composite-selection CURRENT_BASELINE test means, choose the threshold with the lowest validation area_error.

Composite baseline metrics source for validation threshold eligibility: {baseline_metrics['source']}

Selected threshold: {selected_threshold:.2f}

## Validation threshold scan

{format_val_scan(rows, baseline_metrics)}

## Overall test comparison

{format_comparison_table(rows, 'overall', ['all'], selected_threshold)}

## Area-bin test comparison

{format_comparison_table(rows, 'area_bin', ['small', 'medium', 'large'], selected_threshold)}

## Low-signal test comparison

{format_comparison_table(rows, 'signal_bin', ['low_signal', 'non_low_signal'], selected_threshold)}

## Gate checks

{chr(10).join(status_lines)}

Accepted for formal mask-only candidate discussion: {accepted}

If not accepted, stop mask-only threshold, loss, structure, post-processing, and ensemble variants.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'accepted': accepted,
        'overall': overall,
        'small': small,
        'medium': medium,
        'large': large,
        'low_signal': low_signal,
    }


def main():
    ensure_outputs()
    check_checkpoints()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    baseline_metrics = load_current_baseline_overall_metrics()
    print(f"Loaded composite baseline metrics from: {baseline_metrics['source']}")

    val_area_edges = get_area_edges(MFLDataset(VAL_DATA))
    test_area_edges = get_area_edges(MFLDataset(TEST_DATA))
    low_signal_indices = load_low_signal_indices()

    val_metric_rows = []
    val_prob_cache = {}
    for seed, checkpoint in CHECKPOINTS.items():
        print(f'Loading validation probabilities seed={seed}: {checkpoint}')
        prob_maps, true_masks, dataset = predict_prob_maps(checkpoint, VAL_DATA, device)
        val_prob_cache[seed] = (prob_maps, true_masks, dataset)
        for threshold in THRESHOLDS:
            sample_rows = build_sample_rows(
                candidate='mask_boundary_val_scan',
                seed=seed,
                split='val',
                threshold=threshold,
                prob_maps=prob_maps,
                true_masks=true_masks,
                dataset=dataset,
                area_edges=val_area_edges,
                low_signal_indices=set(),
            )
            val_metric_rows.extend(summarize_candidate(sample_rows, 'mask_boundary_val_scan', seed, 'val', threshold))
    aggregate_val_rows = []
    for threshold in THRESHOLDS:
        aggregate_val_rows.extend(aggregate_seed_rows(val_metric_rows, 'mask_boundary_val_scan', 'val', threshold))
    for row in aggregate_val_rows:
        if row['candidate'] == 'mask_boundary_val_scan_mean':
            row['candidate'] = 'mask_boundary_val_scan_mean'
        elif row['candidate'] == 'mask_boundary_val_scan_std':
            row['candidate'] = 'mask_boundary_val_scan_std'
    all_rows = val_metric_rows + aggregate_val_rows
    selected_row = select_threshold(all_rows, baseline_metrics)
    selected_threshold = float(selected_row['threshold'])
    print(f'Selected threshold: {selected_threshold:.2f}')

    test_metric_rows = []
    for seed, checkpoint in CHECKPOINTS.items():
        print(f'Loading test probabilities seed={seed}: {checkpoint}')
        prob_maps, true_masks, dataset = predict_prob_maps(checkpoint, TEST_DATA, device)
        for threshold in sorted(set([0.50, selected_threshold])):
            sample_rows = build_sample_rows(
                candidate='mask_boundary_test',
                seed=seed,
                split='test',
                threshold=threshold,
                prob_maps=prob_maps,
                true_masks=true_masks,
                dataset=dataset,
                area_edges=test_area_edges,
                low_signal_indices=low_signal_indices,
            )
            test_metric_rows.extend(summarize_candidate(sample_rows, 'mask_boundary_test', seed, 'test', threshold))
    aggregate_test_rows = []
    for threshold in sorted(set([0.50, selected_threshold])):
        aggregate_test_rows.extend(aggregate_seed_rows(test_metric_rows, 'mask_boundary_test', 'test', threshold))
    all_rows.extend(test_metric_rows + aggregate_test_rows)
    all_rows.extend(read_baseline_rows())

    write_metrics(all_rows)
    judgment = write_summary(all_rows, selected_threshold, baseline_metrics)

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f"Accepted: {judgment['accepted']}")


if __name__ == '__main__':
    main()
