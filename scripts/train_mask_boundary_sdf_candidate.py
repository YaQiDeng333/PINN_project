import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import (  # noqa: E402
    BzEncoder,
    MFLDataset,
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
    42: 'checkpoints/mask_boundary_candidate/best_mask_boundary_seed42.pt',
    123: 'checkpoints/mask_boundary_candidate/best_mask_boundary_seed123.pt',
    2026: 'checkpoints/mask_boundary_candidate/best_mask_boundary_seed2026.pt',
}
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'

SEEDS = [42, 123, 2026]
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
SDF_CLIP_PIXELS = 8.0
LAMBDA_SDF = 0.1

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_boundary_sdf_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_sdf_candidate_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_boundary_sdf_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'mask_boundary_sdf_candidate'

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


class MaskBoundarySDFModel(nn.Module):
    def __init__(self, signal_length, signal_channels=1, latent_dim=64, coord_feature_dim=84):
        super().__init__()
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        input_dim = coord_feature_dim + latent_dim
        self.shared_decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.mask_head = nn.Linear(64, 1)
        self.sdf_head = nn.Linear(64, 1)

    def forward(self, bz_signal, coords):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(bz_signal.shape[0], -1, -1)
        bz_latent = self.bz_encoder(bz_signal)
        coord_features = feature_mapping(coords)
        bz_features = bz_latent.unsqueeze(1).expand(-1, coord_features.shape[1], -1)
        features = torch.cat([bz_features, coord_features], dim=-1)
        hidden = self.shared_decoder(features)
        mask_logit = self.mask_head(hidden).squeeze(-1)
        sdf_pred = torch.tanh(self.sdf_head(hidden).squeeze(-1))
        return mask_logit, sdf_pred


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def compute_sdf_targets(dataset):
    masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    targets = np.empty(masks.shape, dtype=np.float32)
    for idx, mask in enumerate(masks):
        if np.any(mask):
            outside = distance_transform_edt(~mask)
            inside = distance_transform_edt(mask)
            signed = outside - inside
        else:
            signed = np.full(mask.shape, SDF_CLIP_PIXELS, dtype=np.float32)
        targets[idx] = np.clip(signed / SDF_CLIP_PIXELS, -1.0, 1.0).astype(np.float32)
    return targets.reshape(targets.shape[0], -1)


def soft_dice_loss(logits, target_mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target_mask.reshape(target_mask.shape[0], -1)
    intersection = torch.sum(probs_flat * target_flat, dim=1)
    pred_sum = torch.sum(probs_flat, dim=1)
    target_sum = torch.sum(target_flat, dim=1)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return torch.mean(1.0 - dice)


def combined_loss(mask_logits, sdf_pred, target_mask, target_sdf, pos_weight):
    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask, pos_weight=pos_weight)
    dice = soft_dice_loss(mask_logits, target_mask)
    sdf = F.smooth_l1_loss(sdf_pred, target_sdf)
    return bce + dice + LAMBDA_SDF * sdf, bce, dice, sdf


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
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        output = model(signals, coords)
        logits = output[0] if isinstance(output, tuple) else output
        probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = probs[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
    return prob_maps, true_masks


def load_sdf_checkpoint(path, signal_length, signal_channels, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskBoundarySDFModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
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


def train_one_seed(seed, device, pos_weight_value):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    train_sdf_targets = compute_sdf_targets(train_dataset)
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    model = MaskBoundarySDFModel(
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
    best_path = CHECKPOINT_DIR / f'best_mask_boundary_sdf_seed{seed}.pt'

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_sdf = 0.0
        total_samples = 0
        for signals, mu_targets, indices in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            target_sdf = torch.from_numpy(train_sdf_targets[indices.numpy()]).to(device)
            optimizer.zero_grad(set_to_none=True)
            mask_logits, sdf_pred = model(signals, coords)
            loss, bce, dice, sdf = combined_loss(mask_logits, sdf_pred, target_mask, target_sdf, pos_weight)
            loss.backward()
            optimizer.step()

            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
            total_sdf += float(sdf.item()) * batch_size
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
                    'model': 'mask_boundary_sdf_model',
                    'dataset': 'v3_complex',
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'loss': 'BCEWithLogits + soft Dice + 0.1 * SmoothL1(SDF)',
                    'lambda_sdf': LAMBDA_SDF,
                    'sdf_clip_pixels': SDF_CLIP_PIXELS,
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'sdf_target': 'clipped signed distance transform / SDF_CLIP_PIXELS',
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
            f"seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"loss={total_loss / total_samples:.6e} | "
            f"bce={total_bce / total_samples:.6e} | "
            f"dice_loss={total_dice / total_samples:.6e} | "
            f"sdf_loss={total_sdf / total_samples:.6e} | "
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
        if model_type == 'sdf':
            model, _ = load_sdf_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, device)
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
        if row['candidate'] == 'mask_boundary_sdf_val_scan_mean'
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


def format_val_scan(rows, baseline_overall):
    lines = [
        '| threshold | val IoU | val Dice | val area_error | val center_error | val pred_area=0 | selected eligible |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    selected = [
        row for row in rows
        if row['candidate'] == 'mask_boundary_sdf_val_scan_mean'
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
        sdf_mean = find_row(rows, 'mask_boundary_sdf_test_mean', group_type, group, threshold=selected_threshold)
        sdf_std = find_row(rows, 'mask_boundary_sdf_test_std', group_type, group, threshold=selected_threshold)
        lines.append(
            f"| {group} | current mask-only baseline | {CURRENT_BASELINE_THRESHOLD:.2f} | "
            f"{metric_with_std(base_mean, base_std, 'iou')} | "
            f"{metric_with_std(base_mean, base_std, 'dice')} | "
            f"{metric_with_std(base_mean, base_std, 'area_error')} | "
            f"{metric_with_std(base_mean, base_std, 'center_error')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | mask-only + SDF | {selected_threshold:.2f} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'iou')} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'dice')} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'area_error')} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'center_error')} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'pred_area_zero')} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(sdf_mean, sdf_std, 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group, selected_threshold):
    baseline = find_row(rows, 'current_mask_boundary_baseline_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
    candidate = find_row(rows, 'mask_boundary_sdf_test_mean', group_type, group, threshold=selected_threshold)
    return {
        'iou_not_down': float(candidate['iou']) >= float(baseline['iou']) - 1e-6,
        'dice_not_down': float(candidate['dice']) >= float(baseline['dice']) - 1e-6,
        'area_error_close': float(candidate['area_error']) <= float(baseline['area_error']) + 0.02,
        'pred_area_zero_not_up': float(candidate['pred_area_zero']) <= float(baseline['pred_area_zero']) + 1e-6,
    }


def write_summary(rows, best_infos, checkpoint_paths, selected_threshold, pos_weight, mask_fraction):
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
    summary = f"""# v3_complex mask-only boundary-SDF supervision candidate

This RESULT_DRIVEN_EXPERIMENT trains an independent mask-only boundary/SDF candidate for seeds 42, 123, and 2026. It does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Model and loss

The model keeps the direct Bz -> mask setup. It reuses BzEncoder and Fourier coordinate features, then uses a shared decoder with two heads:

* mask_logit head for defect probability
* sdf_pred head for clipped signed distance supervision

The SDF target is constructed from `target_mu_norm < 0.5` masks with scipy distance_transform_edt, clipped to +/- {SDF_CLIP_PIXELS:.1f} pixels and normalized to [-1, 1]. Training loss is BCEWithLogits + soft Dice + {LAMBDA_SDF:.2f} * SmoothL1(SDF). No loss-weight sweep, adaptive threshold, post-processing, or ensemble is used.

Train mask positive fraction: {mask_fraction:.6f}. BCE pos_weight uses sqrt(neg/pos) capped at {POS_WEIGHT_CAP:.1f}; value used: {pos_weight:.6f}.

## Selected checkpoints

{chr(10).join(best_lines)}

## Validation threshold calibration

The three seeds share one validation-selected probability threshold. Threshold candidates: {', '.join(f'{value:.2f}' for value in THRESHOLDS)}.

Current baseline reference for threshold eligibility: mask-only boundary CURRENT_BASELINE test mean at threshold={CURRENT_BASELINE_THRESHOLD:.2f}, IoU={float(baseline_overall['iou']):.4f}, Dice={float(baseline_overall['dice']):.4f}.

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


def select_preview_samples(sdf_means, baseline_means):
    baseline_by_index = {row['index']: row for row in baseline_means}
    rows = []
    for row in sdf_means:
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
    take('mask_boundary_sdf_failure', failures, 3)

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

    train_dataset_for_weight = MFLDataset(TRAIN_DATA)
    pos_weight, mask_fraction = compute_pos_weight(train_dataset_for_weight)
    print(f'mask positive fraction: {mask_fraction:.6f}')
    print(f'pos_weight: {pos_weight:.6f}')

    checkpoint_paths = []
    best_infos = []
    for seed in SEEDS:
        print(f'Training mask boundary SDF candidate seed={seed}')
        checkpoint_path, best_info = train_one_seed(seed, device, pos_weight)
        checkpoint_paths.append(checkpoint_path)
        best_infos.append(best_info)

    val_area_edges = get_area_edges(MFLDataset(VAL_DATA))
    test_dataset_for_edges = MFLDataset(TEST_DATA)
    test_area_edges = get_area_edges(test_dataset_for_edges)
    low_signal_indices = load_low_signal_indices()

    baseline_rows, baseline_sample_rows, _ = evaluate_checkpoint_family(
        checkpoints=CURRENT_BASELINE_CHECKPOINTS,
        model_type='mask_only',
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

    sdf_checkpoints = {seed: str(path.relative_to(ROOT)) for seed, path in zip(SEEDS, checkpoint_paths)}
    val_rows, _, _ = evaluate_checkpoint_family(
        checkpoints=sdf_checkpoints,
        model_type='sdf',
        candidate='mask_boundary_sdf_val_scan',
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

    test_rows, sdf_sample_rows, sdf_prob_cache = evaluate_checkpoint_family(
        checkpoints=sdf_checkpoints,
        model_type='sdf',
        candidate='mask_boundary_sdf_test',
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
    )

    sdf_means = sample_mean_metrics(sdf_sample_rows, SEEDS, selected_threshold)
    baseline_means = sample_mean_metrics(baseline_sample_rows, SEEDS, CURRENT_BASELINE_THRESHOLD)
    selected_samples = select_preview_samples(sdf_means, baseline_means)
    preview_paths = generate_previews(selected_samples, sdf_prob_cache, sdf_sample_rows, selected_threshold)

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote previews: {PREVIEW_DIR} ({len(preview_paths)} png)')
    print(f"Accepted by metric gate: {judgment['accepted']}")


if __name__ == '__main__':
    main()
