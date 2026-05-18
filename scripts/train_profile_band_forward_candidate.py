import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import project_path, set_seed, signal_shape_info  # noqa: E402
from scripts.train_geometry_boundary_candidate import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    LATENT_DIM,
    MASK_THRESHOLD_NORM,
    RASTER_TEMPERATURE,
    SEEDS,
    TEST_DATA,
    THRESHOLDS,
    TRAIN_DATA,
    TRAIN_SELECTION_THRESHOLD,
    VAL_DATA,
    SingleDefectDataset,
    split_type_counts,
)
from scripts.train_geometry_forward_consistency_candidate import (  # noqa: E402
    CURRENT_BASELINE_THRESHOLD,
    build_sample_rows,
    dataset_low_signal_indices,
    evaluate_current_baseline,
    get_metric_row,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    BzEncoder,
    compute_pos_weight,
    get_area_edges,
    make_loader,
    mask_loss,
    safe_nanmean,
    safe_nanstd,
    threshold_matches,
)
from scripts.train_mask_boundary_grid_forward_consistency_candidate import load_forward_surrogate  # noqa: E402


LAMBDA_FORWARD = 0.10
LR = 1e-3
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}
FORCE_THREE_SEED = os.environ.get('PINN_FORCE_THREE_SEED', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'profile_band_forward_candidate'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_profile_band_forward_candidate_summary.txt'
ORACLE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_profile_band_oracle_metrics.csv'
SCREENING_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_profile_band_forward_screening.csv'
CANDIDATE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_profile_band_forward_candidate_metrics.csv'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'profile_band_forward_candidate'
QUAD_SUPERVISED_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_oracle_quad_supervised_candidate_metrics.csv'

METRIC_KEYS = [
    'iou',
    'dice',
    'area_error',
    'center_error',
    'pred_area_zero',
    'pred_area_lt_true',
    'pred_area_gt_true',
    'bz_mse',
    'composite',
    'macro_area_composite',
    'avg_support_length',
    'empty_support_count',
    'full_width_support_count',
]


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORACLE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREENING_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = []
    surrogate_path = ROOT / 'checkpoints' / 'mask_to_bz_forward_surrogate' / 'best_mask_to_bz_forward_surrogate.pt'
    if not surrogate_path.exists():
        missing.append(str(surrogate_path.relative_to(ROOT)))
    if not QUAD_SUPERVISED_METRICS.exists():
        missing.append(str(QUAD_SUPERVISED_METRICS.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError('Missing required input(s): ' + ', '.join(missing))


def oracle_profile_from_mask(mask, dataset):
    height, width = mask.shape
    y_coords = dataset.y.astype(np.float32)
    support = np.zeros((width,), dtype=np.float32)
    center_y = np.full((width,), float(np.mean(y_coords)), dtype=np.float32)
    half_height = np.zeros((width,), dtype=np.float32)
    for x_idx in range(width):
        ys = np.where(mask[:, x_idx])[0]
        if ys.size == 0:
            continue
        support[x_idx] = 1.0
        y_min = float(y_coords[int(ys.min())])
        y_max = float(y_coords[int(ys.max())])
        center_y[x_idx] = 0.5 * (y_min + y_max)
        half_height[x_idx] = max(0.5 * (y_max - y_min), 0.0)
    return support, center_y, half_height


def oracle_profiles(dataset):
    masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    support = []
    center_y = []
    half_height = []
    for mask in masks:
        s, c, h = oracle_profile_from_mask(mask, dataset)
        support.append(s)
        center_y.append(c)
        half_height.append(h)
    return (
        np.stack(support).astype(np.float32),
        np.stack(center_y).astype(np.float32),
        np.stack(half_height).astype(np.float32),
    )


def render_profile_logits(support_logits, center_y, half_height, y_coords):
    y = y_coords.to(support_logits.device, dtype=support_logits.dtype)[None, :, None]
    center = center_y[:, None, :]
    half = half_height[:, None, :]
    vertical = (half - torch.abs(y - center)) / RASTER_TEMPERATURE
    logits = support_logits[:, None, :] + vertical
    return logits.reshape(support_logits.shape[0], -1)


def render_oracle_profiles(support, center_y, half_height, dataset, device):
    support_logits = torch.from_numpy(np.where(support > 0.5, 12.0, -12.0).astype(np.float32)).to(device)
    center = torch.from_numpy(center_y.astype(np.float32)).to(device)
    half = torch.from_numpy(half_height.astype(np.float32)).to(device)
    y_coords = torch.from_numpy(dataset.y.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = render_profile_logits(support_logits, center, half, y_coords)
        probs = torch.sigmoid(logits).reshape(support.shape[0], *dataset.mu_maps.shape[1:])
    return probs.detach().cpu().numpy().astype(np.float32)


class ProfileBandForwardModel(nn.Module):
    def __init__(self, signal_length, signal_channels, out_width, y_range, latent_dim=LATENT_DIM):
        super().__init__()
        self.out_width = int(out_width)
        self.y_min = float(y_range[0])
        self.y_max = float(y_range[1])
        self.max_half_height = max(0.25, self.y_max - self.y_min)
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3 * self.out_width),
        )
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.bias[: self.out_width] = -1.0
                final.bias[self.out_width : 2 * self.out_width] = 0.0
                final.bias[2 * self.out_width :] = -0.5

    def decode(self, raw):
        support_logits = raw[:, : self.out_width]
        center_raw = raw[:, self.out_width : 2 * self.out_width]
        half_raw = raw[:, 2 * self.out_width :]
        center_y = self.y_min + torch.sigmoid(center_raw) * (self.y_max - self.y_min)
        half_height = torch.clamp(F.softplus(half_raw), min=0.0, max=self.max_half_height)
        return support_logits, center_y, half_height

    def forward(self, bz_signal, y_coords, return_profile=False):
        raw = self.head(self.bz_encoder(bz_signal))
        support_logits, center_y, half_height = self.decode(raw)
        logits = render_profile_logits(support_logits, center_y, half_height, y_coords)
        if return_profile:
            return logits, support_logits, center_y, half_height
        return logits


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse', 'avg_support_length']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'empty_support_count': 0,
            'full_width_support_count': 0,
            'composite': float('nan'),
        })
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse', 'avg_support_length']:
        summary[key] = safe_nanmean([float(row[key]) for row in rows])
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['empty_support_count'] = int(sum(int(row['empty_support']) for row in rows))
    summary['full_width_support_count'] = int(sum(int(row['full_width_support']) for row in rows))
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
    for key in [key for key in METRIC_KEYS if key != 'macro_area_composite']:
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
    macro_area_composite = safe_nanmean([area_summaries[group]['composite'] for group in ['small', 'medium', 'large']])
    rows.append(metric_row(candidate, seed, split, threshold, 'overall', 'all', overall, macro_area_composite))
    for group in ['small', 'medium', 'large']:
        rows.append(metric_row(candidate, seed, split, threshold, 'area_bin', group, area_summaries[group], macro_area_composite))
    for group in ['low_signal', 'non_low_signal']:
        rows.append(metric_row(
            candidate,
            seed,
            split,
            threshold,
            'signal_bin',
            group,
            summarize_samples([row for row in sample_rows if row['signal_bin'] == group]),
            macro_area_composite,
        ))
    for defect_type in sorted({row['defect_type'] for row in sample_rows}):
        rows.append(metric_row(
            candidate,
            seed,
            split,
            threshold,
            'defect_type',
            defect_type,
            summarize_samples([row for row in sample_rows if row['defect_type'] == defect_type]),
            macro_area_composite,
        ))
    return rows


def aggregate_seed_rows(metric_rows, source_candidate, split, threshold):
    rows = []
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
            values = [float(row[key]) for row in selected if row.get(key, '') != '']
            mean_row[key] = safe_nanmean(values)
            std_row[key] = safe_nanstd(values)
        rows.extend([mean_row, std_row])
    return rows


def support_diagnostics(support_probs, mask_width):
    lengths = np.sum(support_probs >= 0.5, axis=1).astype(np.float32)
    return lengths, lengths == 0, lengths >= int(0.95 * mask_width)


def build_profile_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, support_probs, dataset):
    area_edges = get_area_edges(dataset)
    low_signal_indices = dataset_low_signal_indices(dataset)
    lengths, empty, full = support_diagnostics(support_probs, prob_maps.shape[-1])
    base_rows = build_sample_rows(
        candidate,
        seed,
        split,
        threshold,
        prob_maps,
        true_masks,
        bz_mses,
        dataset,
        area_edges,
        low_signal_indices,
    )
    for idx, row in enumerate(base_rows):
        row['avg_support_length'] = float(lengths[idx])
        row['empty_support'] = int(empty[idx])
        row['full_width_support'] = int(full[idx])
    return base_rows


def evaluate_oracle_profile(split, data_path, device):
    dataset = SingleDefectDataset(data_path)
    true_masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    support, center_y, half_height = oracle_profiles(dataset)
    prob_maps = render_oracle_profiles(support, center_y, half_height, dataset, device)
    bz_mses = np.full((len(dataset),), np.nan, dtype=np.float32)
    rows = build_profile_rows(
        'profile_band_oracle',
        'oracle',
        split,
        0.50,
        prob_maps,
        true_masks,
        bz_mses,
        support,
        dataset,
    )
    return summarize_candidate(rows, 'profile_band_oracle', 'oracle', split, 0.50), rows, (support, center_y, half_height, prob_maps, true_masks, dataset)


@torch.no_grad()
def predict_model(model, forward_model, dataset, device, bz_target_dataset=None):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    bz_mses = np.empty((len(dataset),), dtype=np.float32)
    support_probs = np.empty((len(dataset), grid_shape[1]), dtype=np.float32)
    center_y = np.empty((len(dataset), grid_shape[1]), dtype=np.float32)
    half_height = np.empty((len(dataset), grid_shape[1]), dtype=np.float32)
    y_coords = torch.from_numpy(dataset.y.astype(np.float32)).to(device)
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        logits, support_logits, batch_center, batch_half = model(signals, y_coords, return_profile=True)
        probs = torch.sigmoid(logits).reshape(signals.shape[0], *grid_shape)
        if bz_target_dataset is None:
            target_bz = signals
        else:
            batch_indices = indices.detach().cpu().numpy().astype(np.int64)
            target_bz = torch.from_numpy(bz_target_dataset.signals[batch_indices]).to(device)
        bz_hat = forward_model(probs.unsqueeze(1))
        batch_bz_mse = torch.mean((bz_hat - target_bz) ** 2, dim=1).detach().cpu().numpy()
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, idx_tensor in enumerate(indices):
            idx = int(idx_tensor.item())
            prob_maps[idx] = probs[batch_pos].detach().cpu().numpy()
            true_masks[idx] = batch_true[batch_pos]
            bz_mses[idx] = batch_bz_mse[batch_pos]
            support_probs[idx] = torch.sigmoid(support_logits[batch_pos]).detach().cpu().numpy()
            center_y[idx] = batch_center[batch_pos].detach().cpu().numpy()
            half_height[idx] = batch_half[batch_pos].detach().cpu().numpy()
    return prob_maps, true_masks, bz_mses, support_probs, center_y, half_height


def evaluate_model_for_selection(model, dataset, device):
    dummy_forward = lambda mask_prob: torch.zeros((mask_prob.shape[0], dataset.signals.shape[-1]), device=mask_prob.device)
    prob_maps, true_masks, bz_mses, support_probs, _, _ = predict_model(model, dummy_forward, dataset, device)
    rows = build_profile_rows(
        'selection',
        'selection',
        'val',
        TRAIN_SELECTION_THRESHOLD,
        prob_maps,
        true_masks,
        bz_mses,
        support_probs,
        dataset,
    )
    return summarize_samples(rows)


def train_one_seed(seed, device, pos_weight_value, forward_model, surrogate_checkpoint):
    set_seed(seed)
    train_dataset = SingleDefectDataset(
        TRAIN_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    val_dataset = SingleDefectDataset(
        VAL_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    out_width = int(train_dataset.mu_maps.shape[-1])
    y_range = (float(np.min(train_dataset.y)), float(np.max(train_dataset.y)))
    model = ProfileBandForwardModel(signal_length, signal_channels, out_width, y_range, LATENT_DIM).to(device)
    y_coords = torch.from_numpy(train_dataset.y.astype(np.float32)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_profile_band_forward_seed{seed}.pt'
    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        print(f'Reusing existing checkpoint seed={seed}: {best_path}')
        return best_path, checkpoint.get('val_metrics', {})
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_mask = 0.0
        total_forward = 0.0
        total_samples = 0
        for signals, mu_targets, _ in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals, y_coords)
            loss_mask, _, _ = mask_loss(logits, target_mask, pos_weight)
            mask_prob = torch.sigmoid(logits).reshape(signals.shape[0], *train_dataset.mu_maps.shape[1:])
            bz_hat = forward_model(mask_prob.unsqueeze(1))
            loss_forward = F.mse_loss(bz_hat, signals)
            loss = loss_mask + LAMBDA_FORWARD * loss_forward
            loss.backward()
            optimizer.step()
            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_mask += float(loss_mask.item()) * batch_size
            total_forward += float(loss_forward.item()) * batch_size
            total_samples += batch_size
        val_summary = evaluate_model_for_selection(model, val_dataset, device)
        score = val_summary['composite']
        if score > best_score:
            best_score = score
            best_info = {
                'seed': seed,
                'epoch': epoch,
                'selection_score': float(score),
                'val_iou': float(val_summary['iou']),
                'val_dice': float(val_summary['dice']),
                'val_area_error': float(val_summary['area_error']),
                'val_center_error': float(val_summary['center_error']),
                'val_empty_support': int(val_summary['empty_support_count']),
                'val_full_width_support': int(val_summary['full_width_support_count']),
            }
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'args': {
                        'model': 'profile_band_forward_candidate',
                        'seed': seed,
                        'latent_dim': LATENT_DIM,
                        'signal_length': signal_length,
                        'signal_channels': signal_channels,
                        'out_width': out_width,
                        'y_range': y_range,
                        'lambda_forward': LAMBDA_FORWARD,
                        'temperature': RASTER_TEMPERATURE,
                    },
                    'signal_mean': train_dataset.signal_mean,
                    'signal_std': train_dataset.signal_std,
                    'val_metrics': best_info,
                },
                best_path,
            )
        print(
            f"seed={seed} epoch={epoch:03d} | loss={total_loss / total_samples:.6e} | "
            f"mask_loss={total_mask / total_samples:.6e} | forward={total_forward / total_samples:.6e} | "
            f"val_iou={val_summary['iou']:.6e} | val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | empty={val_summary['empty_support_count']} | "
            f"full={val_summary['full_width_support_count']} | score={score:.6e}"
        )
    return best_path, best_info


def load_model(path, dataset, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = checkpoint.get('args', {})
    signal_length, signal_channels = signal_shape_info(dataset.signals)
    model = ProfileBandForwardModel(
        signal_length,
        signal_channels,
        int(args.get('out_width', dataset.mu_maps.shape[-1])),
        tuple(args.get('y_range', (float(np.min(dataset.y)), float(np.max(dataset.y))))),
        int(args.get('latent_dim', LATENT_DIM)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_checkpoints(checkpoints, candidate, split, data_path, thresholds, device, forward_model, surrogate_checkpoint):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    preview_cache = {}
    bz_target_dataset = SingleDefectDataset(
        data_path,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        dataset = SingleDefectDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        model, _ = load_model(checkpoint_path, dataset, device)
        bz_target = None
        if abs(float(checkpoint['signal_mean']) - float(surrogate_checkpoint['signal_mean'])) > 1e-10 or abs(float(checkpoint['signal_std']) - float(surrogate_checkpoint['signal_std'])) > 1e-10:
            bz_target = bz_target_dataset
        prob_maps, true_masks, bz_mses, support_probs, center_y, half_height = predict_model(
            model,
            forward_model,
            dataset,
            device,
            bz_target_dataset=bz_target,
        )
        preview_cache[seed] = (prob_maps, true_masks, support_probs, center_y, half_height, dataset)
        for threshold in thresholds:
            sample_rows = build_profile_rows(
                candidate,
                seed,
                split,
                threshold,
                prob_maps,
                true_masks,
                bz_mses,
                support_probs,
                dataset,
            )
            sample_rows_by_seed_threshold[(seed, threshold)] = sample_rows
            metric_rows.extend(summarize_candidate(sample_rows, candidate, seed, split, threshold))
    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows(metric_rows, candidate, split, threshold))
    return metric_rows, sample_rows_by_seed_threshold, preview_cache


def select_threshold(rows, candidate_name):
    candidates = [
        row for row in rows
        if row['candidate'] == f'{candidate_name}_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    if not candidates:
        candidates = [
            row for row in rows
            if row['candidate'] == candidate_name
            and row['split'] == 'val'
            and row['group_type'] == 'overall'
            and row['group'] == 'all'
        ]
    return max(candidates, key=lambda row: float(row['composite']))


def get_metric(rows, candidate, threshold=None, split='test', group_type='overall', group='all', seed=None):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['split'] == split
        and row['group_type'] == group_type
        and row['group'] == group
    ]
    if threshold is not None:
        selected = [row for row in selected if threshold_matches(row['threshold'], threshold)]
    if seed is not None:
        selected = [row for row in selected if str(row['seed']) == str(seed)]
    return selected[0] if selected else None


def load_quad_supervised_reference():
    rows = []
    with open(QUAD_SUPERVISED_METRICS, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] == 'test' and row['candidate'] in {
                'oracle_quad_supervised_screening',
                'current_forward_baseline_single_defect_mean',
            }:
                for key in METRIC_KEYS:
                    row.setdefault(key, '')
                rows.append(row)
    return rows


def oracle_capacity_pass(metric_rows):
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    oracle = get_metric(metric_rows, 'profile_band_oracle', 0.50)
    if base is None or oracle is None:
        return False, {'reason': 'missing oracle/base rows'}
    checks = {
        'oracle_iou_above_current': float(oracle['iou']) >= float(base['iou']),
        'oracle_dice_above_current': float(oracle['dice']) >= float(base['dice']),
        'oracle_area_not_worse': float(oracle['area_error']) <= float(base['area_error']) + 0.05,
    }
    return bool(all(checks.values())), checks


def seed_gate(metric_rows, selected_threshold):
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    cand = get_metric(metric_rows, 'profile_band_forward_screening', selected_threshold, seed=42)
    base_poly = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD, 'test', 'defect_type', 'polygon')
    cand_poly = get_metric(metric_rows, 'profile_band_forward_screening', selected_threshold, 'test', 'defect_type', 'polygon', seed=42)
    base_rot = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD, 'test', 'defect_type', 'rotated_rect')
    cand_rot = get_metric(metric_rows, 'profile_band_forward_screening', selected_threshold, 'test', 'defect_type', 'rotated_rect', seed=42)
    if base is None or cand is None:
        return False, {'reason': 'missing seed gate rows'}
    checks = {
        'iou_not_obviously_below_current': float(cand['iou']) >= float(base['iou']) - 0.02,
        'dice_not_obviously_below_current': float(cand['dice']) >= float(base['dice']) - 0.02,
        'area_error_not_worse': float(cand['area_error']) <= float(base['area_error']) + 0.03,
        'pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'bz_mse_not_worse': float(cand['bz_mse']) <= float(base['bz_mse']) * 1.10,
        'support_not_collapsed': float(cand['empty_support_count']) == 0.0 and float(cand['full_width_support_count']) == 0.0,
        'polygon_or_rotated_signal': (
            cand_poly is not None and base_poly is not None and float(cand_poly['iou']) >= float(base_poly['iou']) - 0.02
        ) or (
            cand_rot is not None and base_rot is not None and float(cand_rot['iou']) >= float(base_rot['iou']) - 0.02
        ),
    }
    return bool(all(checks.values())), checks


def final_gate(metric_rows, selected_threshold):
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    cand = get_metric(metric_rows, 'profile_band_forward_candidate_mean', selected_threshold)
    if base is None or cand is None:
        return False, {'reason': 'missing final gate rows'}
    checks = {
        'final_iou_not_below_current': float(cand['iou']) >= float(base['iou']) - 0.01,
        'final_dice_not_below_current': float(cand['dice']) >= float(base['dice']) - 0.01,
        'final_area_not_worse': float(cand['area_error']) <= float(base['area_error']) + 0.03,
        'final_pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'final_bz_mse_not_worse': float(cand['bz_mse']) <= float(base['bz_mse']) * 1.10,
        'final_support_not_collapsed': float(cand['empty_support_count']) == 0.0 and float(cand['full_width_support_count']) == 0.0,
    }
    return bool(all(checks.values())), checks


def write_csv(path, rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'threshold', 'n'] + METRIC_KEYS
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_screening(rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'threshold', 'n'] + METRIC_KEYS
    screening = [
        row for row in rows
        if row['candidate'] in {'profile_band_forward_screening', 'profile_band_forward_screening_mean'}
        and row['split'] == 'val'
    ]
    with open(SCREENING_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in screening:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def fmt(row, key):
    if row is None or row.get(key, '') == '':
        return 'NA'
    return f"{float(row[key]):.4f}"


def table_row(label, row):
    return (
        f"| {label} | {fmt(row, 'threshold')} | {fmt(row, 'n')} | {fmt(row, 'iou')} | "
        f"{fmt(row, 'dice')} | {fmt(row, 'area_error')} | {fmt(row, 'center_error')} | "
        f"{fmt(row, 'pred_area_zero')} | {fmt(row, 'bz_mse')} | "
        f"{fmt(row, 'avg_support_length')} | {fmt(row, 'empty_support_count')} | {fmt(row, 'full_width_support_count')} |"
    )


def write_previews(seed, selected_threshold, sample_rows_by_seed_threshold, preview_cache, oracle_cache):
    if (seed, selected_threshold) not in sample_rows_by_seed_threshold:
        return []
    rows = sample_rows_by_seed_threshold[(seed, selected_threshold)]
    prob_maps, true_masks, support_probs, center_y, half_height, dataset = preview_cache[seed]
    oracle_support, oracle_center, oracle_half, oracle_prob, _, _ = oracle_cache
    selected = sorted(rows, key=lambda row: float(row['iou']), reverse=True)[:3]
    selected += sorted(rows, key=lambda row: float(row['iou']))[:3]
    selected += sorted([row for row in rows if row['defect_type'] == 'polygon'], key=lambda row: float(row['area_error']), reverse=True)[:3]
    selected += sorted([row for row in rows if row['defect_type'] == 'rotated_rect'], key=lambda row: float(row['area_error']), reverse=True)[:3]
    selected += sorted(rows, key=lambda row: abs(float(row['iou']) - safe_nanmean([float(r['iou']) for r in rows])))[:12]
    seen = []
    for row in selected:
        idx = int(row['sample_index'])
        if idx not in seen:
            seen.append(idx)
        if len(seen) >= 12:
            break
    x = dataset.x
    written = []
    for rank, sample_idx in enumerate(seen, start=1):
        row = rows[sample_idx]
        prob = prob_maps[sample_idx]
        pred_mask = prob >= selected_threshold
        true_mask = true_masks[sample_idx]
        oracle_mask = oracle_prob[sample_idx] >= 0.50
        fig, axes = plt.subplots(1, 5, figsize=(21, 4), constrained_layout=True)
        axes[0].imshow(true_mask, origin='lower', cmap='gray')
        axes[0].set_title('true mask')
        axes[1].imshow(oracle_mask, origin='lower', cmap='gray')
        axes[1].set_title('oracle profile')
        im = axes[2].imshow(prob, origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('pred probability')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow(pred_mask, origin='lower', cmap='gray')
        axes[3].set_title(f'pred mask t={selected_threshold:.2f}')
        axes[4].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[4].imshow(pred_mask, origin='lower', cmap='Reds', alpha=0.35)
        if np.any(true_mask) and np.any(~true_mask):
            axes[4].contour(true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        if np.any(pred_mask) and np.any(~pred_mask):
            axes[4].contour(pred_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        axes[4].set_title('overlay')
        # Draw profile center and upper/lower bands on the probability panel.
        upper = center_y[sample_idx] + half_height[sample_idx]
        lower = center_y[sample_idx] - half_height[sample_idx]
        axes[2].plot(np.arange(len(x)), np.interp(upper, dataset.y, np.arange(len(dataset.y))), color='white', linewidth=0.8)
        axes[2].plot(np.arange(len(x)), np.interp(lower, dataset.y, np.arange(len(dataset.y))), color='white', linewidth=0.8)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={row['original_index']} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} BzMSE={float(row['bz_mse']):.3e} "
            f"support={float(row['avg_support_length']):.1f}",
            fontsize=9,
        )
        path = PREVIEW_DIR / f'profile_band_forward_seed{seed}_rank{rank:02d}_sample{row["original_index"]}_{row["defect_type"]}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def write_summary(oracle_rows, metric_rows, selected_threshold, oracle_pass, oracle_checks, stage_b_entered, accepted, gate_checks, checkpoint_paths, best_infos, preview_paths, counts):
    oracle = get_metric(oracle_rows, 'profile_band_oracle', 0.50)
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    quad = get_metric(metric_rows, 'oracle_quad_supervised_screening', 0.95, seed=42)
    seed42 = get_metric(metric_rows, 'profile_band_forward_screening', selected_threshold, seed=42) if selected_threshold is not None else None
    final = get_metric(metric_rows, 'profile_band_forward_candidate_mean', selected_threshold) if stage_b_entered else None
    lines = [
        '# v3_complex profile-band forward candidate',
        '',
        '## Single-defect subset',
        '',
        'The subset keeps polygon / rotated_rect samples and excludes multi_defect. Original data files are not modified.',
        '',
        '| split | polygon | rotated_rect | total |',
        '|---|---:|---:|---:|',
    ]
    for split in ['train', 'val', 'test']:
        c = counts[split]
        lines.append(f"| {split} | {c.get('polygon', 0)} | {c.get('rotated_rect', 0)} | {sum(c.values())} |")
    lines.extend([
        '',
        '## Profile representation',
        '',
        '* Oracle profile is constructed per x-column from true-mask vertical support: support(x), center_y(x), and half_height(x).',
        '* The trained model predicts support logits, center_y, and nonnegative half_height for every mask column.',
        '* Renderer: `support_logit + (half_height - abs(y - center_y)) / temperature` with fixed temperature.',
        f'* Loss: BCEWithLogits + soft Dice + `{LAMBDA_FORWARD}` * frozen mask-to-Bz surrogate MSE.',
        '* No profile v2, parameter MSE, support loss, SDF, boundary head, temperature search, or multi-component matching is used.',
        '',
        '## Oracle profile capacity',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | avg_support_length | empty_support | full_width_support |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        table_row('profile_band_oracle test', oracle),
        '',
        f'Oracle capacity pass: {oracle_pass}',
    ])
    for key, value in oracle_checks.items():
        lines.append(f'* {key}: {value}')
    lines.extend([
        '',
        '## Seed=42 gate',
        '',
        f'* validation-selected threshold: `{selected_threshold if selected_threshold is not None else "N/A"}`',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | avg_support_length | empty_support | full_width_support |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        table_row('CURRENT_BASELINE on same subset', base),
        table_row('Step 18.8 oracle-quad supervised seed=42', quad),
        table_row('profile-band + forward seed=42', seed42),
        '',
        '## Gate checks',
        '',
    ])
    for key, value in gate_checks.items():
        lines.append(f'* {key}: {value}')
    lines.extend([
        '',
        f'Entered 3 seed: {stage_b_entered}',
        f'Accepted by gate: {accepted}',
        '',
    ])
    if best_infos:
        lines.extend([
            '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | empty_support | full_width_support | checkpoint |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---|',
        ])
        for info, path in zip(best_infos, checkpoint_paths):
            lines.append(
                f"| {info['seed']} | {info['epoch']} | {info['selection_score']:.6e} | "
                f"{info['val_iou']:.4f} | {info['val_dice']:.4f} | {info['val_area_error']:.4f} | "
                f"{info['val_empty_support']} | {info['val_full_width_support']} | `{path.relative_to(ROOT)}` |"
            )
    if stage_b_entered and final is not None:
        lines.extend([
            '',
            '## Three-seed result',
            '',
            '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | avg_support_length | empty_support | full_width_support |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
            table_row('profile-band + forward mean', final),
        ])
    lines.extend([
        '',
        '## Conclusion',
        '',
    ])
    if accepted:
        lines.append('Profile-band representation satisfies the gate and has candidate value for follow-up review.')
    else:
        lines.append('Profile-band representation does not satisfy the gate. Stop profile v2, temperature/support-loss tuning, parameter MSE, and multi-component extensions from this result.')
    lines.append(f'Preview files: {len(preview_paths)} written to `{PREVIEW_DIR.relative_to(ROOT)}`.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = load_forward_surrogate(device)

    train_dataset = SingleDefectDataset(
        TRAIN_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    val_dataset = SingleDefectDataset(
        VAL_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    test_dataset = SingleDefectDataset(
        TEST_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    counts = {
        'train': split_type_counts(train_dataset),
        'val': split_type_counts(val_dataset),
        'test': split_type_counts(test_dataset),
    }
    pos_weight_value, positive_fraction = compute_pos_weight(train_dataset)
    print(f'Single-defect counts: {counts}')
    print(f'pos_weight={pos_weight_value:.4f}, positive_fraction={positive_fraction:.6f}')

    oracle_val_rows, _, _ = evaluate_oracle_profile('val', VAL_DATA, device)
    oracle_test_rows, _, oracle_cache = evaluate_oracle_profile('test', TEST_DATA, device)
    oracle_rows = oracle_val_rows + oracle_test_rows
    current_rows = evaluate_current_baseline('test', TEST_DATA, device, forward_model)
    reference_rows = load_quad_supervised_reference()
    metric_rows = current_rows + reference_rows
    oracle_pass, oracle_checks = oracle_capacity_pass(metric_rows + oracle_rows)
    write_csv(ORACLE_METRICS_PATH, oracle_rows)

    checkpoint_paths = []
    best_infos = []
    preview_paths = []
    selected_threshold = None
    stage_b_entered = False
    accepted = False
    gate_checks = oracle_checks if not oracle_pass else {}

    if oracle_pass:
        seed42_path, seed42_info = train_one_seed(42, device, pos_weight_value, forward_model, surrogate_checkpoint)
        checkpoint_paths.append(seed42_path)
        best_infos.append(seed42_info)
        screening_checkpoints = {42: seed42_path}
        val_rows, _, _ = evaluate_checkpoints(
            screening_checkpoints,
            'profile_band_forward_screening',
            'val',
            VAL_DATA,
            THRESHOLDS,
            device,
            forward_model,
            surrogate_checkpoint,
        )
        threshold_row = select_threshold(val_rows, 'profile_band_forward_screening')
        selected_threshold = float(threshold_row['threshold'])
        test_rows, sample_rows, preview_cache = evaluate_checkpoints(
            screening_checkpoints,
            'profile_band_forward_screening',
            'test',
            TEST_DATA,
            [selected_threshold],
            device,
            forward_model,
            surrogate_checkpoint,
        )
        metric_rows.extend(val_rows + test_rows)
        passed_seed_gate, gate_checks = seed_gate(metric_rows, selected_threshold)
        preview_paths = write_previews(42, selected_threshold, sample_rows, preview_cache, oracle_cache)
        stage_b_entered = bool(passed_seed_gate or FORCE_THREE_SEED)
        if stage_b_entered:
            checkpoints = {42: seed42_path}
            for seed in [123, 2026]:
                path, info = train_one_seed(seed, device, pos_weight_value, forward_model, surrogate_checkpoint)
                checkpoints[seed] = path
                checkpoint_paths.append(path)
                best_infos.append(info)
            val_rows, _, _ = evaluate_checkpoints(
                checkpoints,
                'profile_band_forward_candidate',
                'val',
                VAL_DATA,
                THRESHOLDS,
                device,
                forward_model,
                surrogate_checkpoint,
            )
            threshold_row = select_threshold(val_rows, 'profile_band_forward_candidate')
            selected_threshold = float(threshold_row['threshold'])
            test_rows, sample_rows, preview_cache = evaluate_checkpoints(
                checkpoints,
                'profile_band_forward_candidate',
                'test',
                TEST_DATA,
                [selected_threshold],
                device,
                forward_model,
                surrogate_checkpoint,
            )
            metric_rows.extend(val_rows + test_rows)
            accepted, gate_checks = final_gate(metric_rows, selected_threshold)
            preview_paths = write_previews(42, selected_threshold, sample_rows, preview_cache, oracle_cache)

    write_screening(metric_rows)
    write_csv(CANDIDATE_METRICS_PATH, metric_rows)
    write_summary(
        oracle_rows,
        metric_rows,
        selected_threshold,
        oracle_pass,
        oracle_checks,
        stage_b_entered,
        accepted,
        gate_checks,
        checkpoint_paths,
        best_infos,
        preview_paths,
        counts,
    )
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote oracle metrics: {ORACLE_METRICS_PATH}')
    print(f'Wrote screening metrics: {SCREENING_PATH}')
    print(f'Wrote candidate metrics: {CANDIDATE_METRICS_PATH}')
    print(f'Preview count: {len(preview_paths)}')


if __name__ == '__main__':
    main()
