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
from scripts.train_deformable_quad_forward_candidate import (  # noqa: E402
    quad_oracle_vertices,
    rasterize_quad,
    coord_tuple,
    draw_vertices,
    invalid_quad,
    vertices_out_of_image,
)
from scripts.train_geometry_forward_consistency_candidate import (  # noqa: E402
    CURRENT_BASELINE_THRESHOLD,
    build_sample_rows,
    dataset_low_signal_indices,
    find_row,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    BzEncoder,
    compute_pos_weight,
    compute_mask_metrics,
    area_bin,
    get_area_edges,
    make_loader,
    mask_loss,
    safe_nanmean,
    safe_nanstd,
    threshold_matches,
)


LR = 1e-3
LAMBDA_VERTEX = 1.0
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'oracle_quad_supervised_candidate'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_oracle_quad_supervised_candidate_summary.txt'
PSEUDO_LABEL_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_oracle_quad_pseudo_label_metrics.csv'
SCREENING_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_oracle_quad_supervised_screening.csv'
CANDIDATE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_oracle_quad_supervised_candidate_metrics.csv'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'oracle_quad_supervised_candidate'
REFERENCE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_deformable_quad_forward_candidate_metrics.csv'
ORACLE_REFERENCE_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_deformable_quad_oracle_metrics.csv'

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
    'vertex_mae',
    'vertex_rmse',
    'invalid_count',
    'out_of_image_count',
]


class OracleQuadDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, signal_mean=None, signal_std=None):
        self.base = SingleDefectDataset(npz_path, signal_mean=signal_mean, signal_std=signal_std)
        self.signals = self.base.signals
        self.mu_maps = self.base.mu_maps
        self.defect_types = self.base.defect_types
        self.original_indices = self.base.original_indices
        self.x = self.base.x
        self.y = self.base.y
        self.signal_mean = self.base.signal_mean
        self.signal_std = self.base.signal_std
        masks = self.mu_maps < MASK_THRESHOLD_NORM
        self.oracle_vertices = np.stack([quad_oracle_vertices(mask, self.base) for mask in masks], axis=0).astype(np.float32)
        self.oracle_vertices_norm = normalize_vertices(self.oracle_vertices, self.base).astype(np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.mu_maps[idx].reshape(-1)),
            torch.from_numpy(self.oracle_vertices_norm[idx]),
            idx,
        )


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PSEUDO_LABEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREENING_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = []
    for path in [REFERENCE_METRICS_PATH, ORACLE_REFERENCE_PATH]:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError('Missing required input(s): ' + ', '.join(missing))


def normalize_vertices(vertices, dataset):
    vertices = np.asarray(vertices, dtype=np.float32)
    x_min, x_max = float(np.min(dataset.x)), float(np.max(dataset.x))
    y_min, y_max = float(np.min(dataset.y)), float(np.max(dataset.y))
    out = np.empty_like(vertices, dtype=np.float32)
    out[..., 0] = 2.0 * (vertices[..., 0] - x_min) / (x_max - x_min) - 1.0
    out[..., 1] = 2.0 * (vertices[..., 1] - y_min) / (y_max - y_min) - 1.0
    return np.clip(out, -1.0, 1.0)


def denormalize_vertices(vertices_norm, dataset):
    x_min, x_max = float(np.min(dataset.x)), float(np.max(dataset.x))
    y_min, y_max = float(np.min(dataset.y)), float(np.max(dataset.y))
    out = torch.empty_like(vertices_norm)
    out[..., 0] = x_min + 0.5 * (vertices_norm[..., 0] + 1.0) * (x_max - x_min)
    out[..., 1] = y_min + 0.5 * (vertices_norm[..., 1] + 1.0) * (y_max - y_min)
    return out


def vertex_permutations(target):
    variants = []
    for shift in range(4):
        variants.append(torch.roll(target, shifts=shift, dims=1))
    reversed_target = torch.flip(target, dims=[1])
    for shift in range(4):
        variants.append(torch.roll(reversed_target, shifts=shift, dims=1))
    return variants


def cyclic_smooth_l1(pred, target):
    losses = []
    for variant in vertex_permutations(target):
        losses.append(F.smooth_l1_loss(pred, variant, reduction='none').mean(dim=(1, 2)))
    stacked = torch.stack(losses, dim=1)
    return torch.min(stacked, dim=1).values.mean()


def minimal_vertex_errors(pred_norm, target_norm):
    pred = np.asarray(pred_norm, dtype=np.float32)
    target = np.asarray(target_norm, dtype=np.float32)
    maes = np.empty((pred.shape[0],), dtype=np.float32)
    rmses = np.empty((pred.shape[0],), dtype=np.float32)
    for i in range(pred.shape[0]):
        variants = []
        for shift in range(4):
            variants.append(np.roll(target[i], shift=shift, axis=0))
        rev = target[i][::-1]
        for shift in range(4):
            variants.append(np.roll(rev, shift=shift, axis=0))
        errors = [pred[i] - variant for variant in variants]
        mse = np.asarray([np.mean(err ** 2) for err in errors], dtype=np.float32)
        best = int(np.argmin(mse))
        best_err = errors[best]
        maes[i] = float(np.mean(np.abs(best_err)))
        rmses[i] = float(np.sqrt(np.mean(best_err ** 2)))
    return maes, rmses


class OracleQuadSupervisedModel(nn.Module):
    def __init__(self, signal_length, signal_channels=1, latent_dim=64, init_bias=None):
        super().__init__()
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Linear(96, 8),
        )
        if init_bias is not None and isinstance(self.head[-1], nn.Linear):
            with torch.no_grad():
                clipped = np.clip(init_bias.reshape(-1), -0.95, 0.95)
                self.head[-1].bias[:] = torch.from_numpy(np.arctanh(clipped).astype(np.float32))

    def forward(self, bz_signal, dataset, coords, return_vertices=False):
        pred_norm = torch.tanh(self.head(self.bz_encoder(bz_signal))).reshape(-1, 4, 2)
        vertices = denormalize_vertices(pred_norm, dataset)
        logits = rasterize_quad(vertices, coords)
        if return_vertices:
            return logits, vertices, pred_norm
        return logits


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error', 'vertex_mae', 'vertex_rmse']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
            'invalid_count': 0,
            'out_of_image_count': 0,
        })
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error', 'vertex_mae', 'vertex_rmse']:
        summary[key] = safe_nanmean([float(row[key]) for row in rows])
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['invalid_count'] = int(sum(int(row['invalid']) for row in rows))
    summary['out_of_image_count'] = int(sum(int(row['out_of_image']) for row in rows))
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


def build_rows(candidate, seed, split, threshold, prob_maps, true_masks, pred_vertices, pred_vertices_norm, oracle_vertices_norm, dataset):
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    area_edges = get_area_edges(dataset.base)
    low_signal_indices = dataset_low_signal_indices(dataset.base)
    vertex_mae, vertex_rmse = minimal_vertex_errors(pred_vertices_norm, oracle_vertices_norm)
    rows = []
    for idx in range(len(dataset)):
        pred_mask = prob_maps[idx] >= threshold
        metrics = compute_mask_metrics(pred_mask, true_masks[idx], x_grid, y_grid)
        metrics.update({
            'candidate': candidate,
            'seed': seed,
            'split': split,
            'threshold': threshold,
            'sample_index': idx,
            'original_index': int(dataset.original_indices[idx]),
            'defect_type': str(dataset.defect_types[idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if idx in low_signal_indices else 'non_low_signal',
            'vertex_mae': float(vertex_mae[idx]),
            'vertex_rmse': float(vertex_rmse[idx]),
            'invalid': int(invalid_quad(pred_vertices[idx])),
            'out_of_image': int(vertices_out_of_image(pred_vertices[idx], dataset.base)),
        })
        rows.append(metrics)
    return rows


def evaluate_pseudo_labels(split, data_path, device):
    dataset = OracleQuadDataset(data_path)
    true_masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    prob_maps = render_oracle(dataset.oracle_vertices, dataset, device)
    rows = build_rows(
        'oracle_quad_pseudo_label',
        'oracle',
        split,
        0.50,
        prob_maps,
        true_masks,
        dataset.oracle_vertices,
        dataset.oracle_vertices_norm,
        dataset.oracle_vertices_norm,
        dataset,
    )
    return summarize_candidate(rows, 'oracle_quad_pseudo_label', 'oracle', split, 0.50), rows


def render_oracle(vertices, dataset, device):
    with torch.no_grad():
        verts = torch.from_numpy(vertices.astype(np.float32)).to(device)
        logits = rasterize_quad(verts, coord_tuple(dataset.base, device))
        probs = torch.sigmoid(logits).reshape(vertices.shape[0], *dataset.mu_maps.shape[1:])
    return probs.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_model(model, dataset, device):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    vertices = np.empty((len(dataset), 4, 2), dtype=np.float32)
    vertices_norm = np.empty((len(dataset), 4, 2), dtype=np.float32)
    coords = coord_tuple(dataset.base, device)
    model.eval()
    for signals, mu_targets, _, indices in loader:
        signals = signals.to(device)
        logits, batch_vertices, batch_norm = model(signals, dataset.base, coords, return_vertices=True)
        probs = torch.sigmoid(logits).reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, idx_tensor in enumerate(indices):
            idx = int(idx_tensor.item())
            prob_maps[idx] = probs[batch_pos].detach().cpu().numpy()
            true_masks[idx] = batch_true[batch_pos]
            vertices[idx] = batch_vertices[batch_pos].detach().cpu().numpy()
            vertices_norm[idx] = batch_norm[batch_pos].detach().cpu().numpy()
    return prob_maps, true_masks, vertices, vertices_norm


def evaluate_model_for_selection(model, dataset, device):
    prob_maps, true_masks, vertices, vertices_norm = predict_model(model, dataset, device)
    sample_rows = build_rows(
        'selection',
        'selection',
        'val',
        TRAIN_SELECTION_THRESHOLD,
        prob_maps,
        true_masks,
        vertices,
        vertices_norm,
        dataset.oracle_vertices_norm,
        dataset,
    )
    return summarize_samples(sample_rows)


def train_one_seed(seed, device, pos_weight_value, init_bias):
    set_seed(seed)
    train_dataset = OracleQuadDataset(TRAIN_DATA)
    val_dataset = OracleQuadDataset(
        VAL_DATA,
        signal_mean=train_dataset.signal_mean,
        signal_std=train_dataset.signal_std,
    )
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    model = OracleQuadSupervisedModel(signal_length, signal_channels, LATENT_DIM, init_bias=init_bias).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    coords = coord_tuple(train_dataset.base, device)
    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_oracle_quad_supervised_seed{seed}.pt'
    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        print(f'Reusing existing checkpoint seed={seed}: {best_path}')
        return best_path, checkpoint.get('val_metrics', {})
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_vertex = 0.0
        total_samples = 0
        for signals, mu_targets, oracle_vertices, _ in train_loader:
            signals = signals.to(device)
            oracle_vertices = oracle_vertices.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).float()
            optimizer.zero_grad()
            logits, _, pred_norm = model(signals, train_dataset.base, coords, return_vertices=True)
            loss_mask, _, _ = mask_loss(logits, target_mask, pos_weight)
            loss_vertex = cyclic_smooth_l1(pred_norm, oracle_vertices)
            loss = loss_mask + LAMBDA_VERTEX * loss_vertex
            loss.backward()
            optimizer.step()
            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_vertex += float(loss_vertex.item()) * batch_size
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
                'val_vertex_mae': float(val_summary['vertex_mae']),
                'lambda_vertex': LAMBDA_VERTEX,
            }
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'args': {
                        'model': 'oracle_quad_supervised_candidate',
                        'seed': seed,
                        'latent_dim': LATENT_DIM,
                        'signal_length': signal_length,
                        'signal_channels': signal_channels,
                        'lambda_vertex': LAMBDA_VERTEX,
                        'temperature': RASTER_TEMPERATURE,
                    },
                    'signal_mean': train_dataset.signal_mean,
                    'signal_std': train_dataset.signal_std,
                    'val_metrics': best_info,
                    'init_bias': init_bias,
                },
                best_path,
            )
        print(
            f"seed={seed} epoch={epoch:03d} | loss={total_loss / total_samples:.6e} | "
            f"vertex={total_vertex / total_samples:.6e} | val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | val_area_error={val_summary['area_error']:.6e} | "
            f"val_vertex_mae={val_summary['vertex_mae']:.6e} | score={score:.6e}"
        )
    return best_path, best_info


def load_model(path, dataset, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    signal_length, signal_channels = signal_shape_info(dataset.signals)
    model = OracleQuadSupervisedModel(
        signal_length,
        signal_channels,
        int(checkpoint.get('args', {}).get('latent_dim', LATENT_DIM)),
        init_bias=checkpoint.get('init_bias'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_checkpoints(checkpoints, candidate, split, data_path, thresholds, device):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    preview_cache = {}
    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        dataset = OracleQuadDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        model, _ = load_model(checkpoint_path, dataset, device)
        prob_maps, true_masks, vertices, vertices_norm = predict_model(model, dataset, device)
        preview_cache[seed] = (prob_maps, true_masks, vertices, vertices_norm, dataset)
        for threshold in thresholds:
            sample_rows = build_rows(
                candidate,
                seed,
                split,
                threshold,
                prob_maps,
                true_masks,
                vertices,
                vertices_norm,
                dataset.oracle_vertices_norm,
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


def load_reference_rows():
    rows = []
    with open(REFERENCE_METRICS_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] == 'test' and row['candidate'] in {
                'current_forward_baseline_single_defect_mean',
                'deformable_quad_forward_screening',
            }:
                for key in METRIC_KEYS:
                    row.setdefault(key, '')
                rows.append(row)
    return rows


def load_oracle_reference():
    rows = []
    with open(ORACLE_REFERENCE_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] == 'test' and row['candidate'] == 'deformable_quad_oracle':
                for key in METRIC_KEYS:
                    row.setdefault(key, '')
                rows.append(row)
    return rows


def seed_gate(metric_rows, selected_threshold):
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    prev = get_metric(metric_rows, 'deformable_quad_forward_screening', 0.95, seed=42)
    cand = get_metric(metric_rows, 'oracle_quad_supervised_screening', selected_threshold, seed=42)
    cand_poly = get_metric(metric_rows, 'oracle_quad_supervised_screening', selected_threshold, 'test', 'defect_type', 'polygon', seed=42)
    base_poly = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD, 'test', 'defect_type', 'polygon')
    cand_rot = get_metric(metric_rows, 'oracle_quad_supervised_screening', selected_threshold, 'test', 'defect_type', 'rotated_rect', seed=42)
    base_rot = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD, 'test', 'defect_type', 'rotated_rect')
    if base is None or prev is None or cand is None:
        return False, {'reason': 'missing comparison rows'}
    checks = {
        'area_error_below_18_7_quad_forward': float(cand['area_error']) <= float(prev['area_error']) - 0.03,
        'iou_not_obviously_below_current': float(cand['iou']) >= float(base['iou']) - 0.02,
        'dice_not_obviously_below_current': float(cand['dice']) >= float(base['dice']) - 0.02,
        'pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'vertex_invalid_zero': float(cand['invalid_count']) == 0.0,
        'vertices_out_of_image_zero': float(cand['out_of_image_count']) == 0.0,
        'polygon_or_rotated_signal': (
            cand_poly is not None and base_poly is not None and float(cand_poly['iou']) >= float(base_poly['iou']) - 0.02
        ) or (
            cand_rot is not None and base_rot is not None and float(cand_rot['iou']) >= float(base_rot['iou']) - 0.02
        ),
    }
    return bool(all(checks.values())), checks


def final_gate(metric_rows, selected_threshold):
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    prev = get_metric(metric_rows, 'deformable_quad_forward_screening', 0.95, seed=42)
    cand = get_metric(metric_rows, 'oracle_quad_supervised_candidate_mean', selected_threshold)
    if base is None or prev is None or cand is None:
        return False, {'reason': 'missing final rows'}
    checks = {
        'final_area_error_below_18_7_quad_forward': float(cand['area_error']) <= float(prev['area_error']) - 0.03,
        'final_iou_not_obviously_below_current': float(cand['iou']) >= float(base['iou']) - 0.02,
        'final_dice_not_obviously_below_current': float(cand['dice']) >= float(base['dice']) - 0.02,
        'final_pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'final_vertex_invalid_zero': float(cand['invalid_count']) == 0.0,
        'final_vertices_out_of_image_zero': float(cand['out_of_image_count']) == 0.0,
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
        if row['candidate'] in {'oracle_quad_supervised_screening', 'oracle_quad_supervised_screening_mean'}
        and row['split'] == 'val'
    ]
    with open(SCREENING_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in screening:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_previews(seed, selected_threshold, sample_rows_by_seed_threshold, preview_cache):
    if (seed, selected_threshold) not in sample_rows_by_seed_threshold:
        return []
    prob_maps, true_masks, vertices, vertices_norm, dataset = preview_cache[seed]
    rows = sample_rows_by_seed_threshold[(seed, selected_threshold)]
    oracle_prob = render_oracle(dataset.oracle_vertices, dataset, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
    written = []
    for rank, sample_idx in enumerate(seen, start=1):
        row = rows[sample_idx]
        pred_mask = prob_maps[sample_idx] >= selected_threshold
        true_mask = true_masks[sample_idx]
        oracle_mask = oracle_prob[sample_idx] >= 0.50
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
        axes[0].imshow(true_mask, origin='lower', cmap='gray')
        axes[0].set_title('true mask')
        axes[1].imshow(oracle_mask, origin='lower', cmap='gray')
        draw_vertices(axes[1], dataset.oracle_vertices[sample_idx], dataset.base)
        axes[1].set_title('oracle quad')
        im = axes[2].imshow(prob_maps[sample_idx], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('pred probability')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow(pred_mask, origin='lower', cmap='gray')
        draw_vertices(axes[3], vertices[sample_idx], dataset.base)
        axes[3].set_title(f'pred mask t={selected_threshold:.2f}')
        axes[4].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[4].imshow(pred_mask, origin='lower', cmap='Reds', alpha=0.35)
        if np.any(true_mask) and np.any(~true_mask):
            axes[4].contour(true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        if np.any(pred_mask) and np.any(~pred_mask):
            axes[4].contour(pred_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        draw_vertices(axes[4], vertices[sample_idx], dataset.base)
        axes[4].set_title('overlay')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={row['original_index']} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} vertex_mae={float(row['vertex_mae']):.3f}",
            fontsize=9,
        )
        path = PREVIEW_DIR / f'oracle_quad_supervised_seed{seed}_rank{rank:02d}_sample{row["original_index"]}_{row["defect_type"]}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def fmt(row, key):
    if row is None or row.get(key, '') == '':
        return 'NA'
    return f"{float(row[key]):.4f}"


def table_row(label, row):
    return (
        f"| {label} | {fmt(row, 'threshold')} | {fmt(row, 'n')} | {fmt(row, 'iou')} | "
        f"{fmt(row, 'dice')} | {fmt(row, 'area_error')} | {fmt(row, 'center_error')} | "
        f"{fmt(row, 'pred_area_zero')} | {fmt(row, 'vertex_mae')} | {fmt(row, 'vertex_rmse')} | "
        f"{fmt(row, 'invalid_count')} | {fmt(row, 'out_of_image_count')} |"
    )


def write_summary(pseudo_rows, metric_rows, selected_threshold, stage_b_entered, accepted, gate_checks, checkpoint_paths, best_infos, preview_paths, counts):
    pseudo = get_metric(pseudo_rows, 'oracle_quad_pseudo_label', 0.50)
    base = get_metric(metric_rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    prev = get_metric(metric_rows, 'deformable_quad_forward_screening', 0.95, seed=42)
    seed42 = get_metric(metric_rows, 'oracle_quad_supervised_screening', selected_threshold, seed=42) if selected_threshold is not None else None
    final = get_metric(metric_rows, 'oracle_quad_supervised_candidate_mean', selected_threshold) if stage_b_entered else None
    lines = [
        '# v3_complex oracle-quad supervised candidate diagnostic',
        '',
        '## Single-defect subset',
        '',
        'The subset keeps polygon / rotated_rect samples and excludes multi_defect. Original data files are not modified.',
        '',
        '| split | polygon | rotated_rect | total |',
        '|---|---:|---:|---:|',
    ]
    for split, count in counts.items():
        lines.append(f"| {split} | {count.get('polygon', 0)} | {count.get('rotated_rect', 0)} | {sum(count.values())} |")
    lines.extend([
        '',
        '## Pseudo-labels and loss',
        '',
        '* Oracle quad pseudo-labels use the same contour/PCA quadrant-extreme method as Step 18.7.',
        '* Vertices are sorted by angle around the polygon center and normalized to [-1, 1].',
        '* Vertex loss uses SmoothL1 over the minimum cyclic shift / reversed ordering, reducing vertex-order ambiguity.',
        f'* Training loss: BCEWithLogits + soft Dice + `{LAMBDA_VERTEX}` * vertex SmoothL1.',
        '* No forward consistency, SDF loss, boundary head, lambda search, temperature search, more vertices, or multi-component matching is used.',
        '',
        '## Oracle pseudo-label quality',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | vertex_MAE | vertex_RMSE | invalid | out_of_image |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        table_row('oracle_quad_pseudo_label test', pseudo),
        '',
        '## Seed=42 gate',
        '',
        f"* validation-selected threshold: `{selected_threshold if selected_threshold is not None else 'NA'}`",
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | vertex_MAE | vertex_RMSE | invalid | out_of_image |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        table_row('CURRENT_BASELINE on same subset', base),
        table_row('Step 18.7 deformable quad + forward seed=42', prev),
        table_row('oracle-quad supervised seed=42', seed42),
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
    ])
    if checkpoint_paths:
        lines.extend(['', '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | val_vertex_mae | checkpoint |', '|---:|---:|---:|---:|---:|---:|---:|---|'])
        for path, info in zip(checkpoint_paths, best_infos):
            lines.append(
                f"| {info.get('seed', 'NA')} | {info.get('epoch', 'NA')} | {float(info.get('selection_score', float('nan'))):.6e} | "
                f"{float(info.get('val_iou', float('nan'))):.4f} | {float(info.get('val_dice', float('nan'))):.4f} | "
                f"{float(info.get('val_area_error', float('nan'))):.4f} | {float(info.get('val_vertex_mae', float('nan'))):.4f} | `{path.relative_to(ROOT)}` |"
            )
    if final is not None:
        lines.extend(['', '## 3-seed result', '', table_row('oracle-quad supervised 3-seed mean', final)])
    lines.extend(['', '## Conclusion', ''])
    if not stage_b_entered:
        lines.append('Seed=42 did not satisfy the diagnostic gate. Explicit oracle-quad vertex supervision did not show enough evidence that Bz can reliably learn the oracle deformable-quadrilateral parameters. Stop geometry parameterization here: do not continue oracle-quad v2, more vertices, multi-component, forward consistency add-on, lambda tuning, or temperature tuning from this result.')
    elif not accepted:
        lines.append('3-seed validation did not satisfy the diagnostic gate. Stop geometry parameterization here.')
    else:
        lines.append('Oracle-quad supervised model satisfies the diagnostic gate; geometry parameterization remains viable, pending review.')
    if preview_paths:
        lines.append(f'Preview files: {len(preview_paths)} written to `{PREVIEW_DIR.relative_to(ROOT)}`.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    counts = {
        'train': split_type_counts(SingleDefectDataset(TRAIN_DATA)),
        'val': split_type_counts(SingleDefectDataset(VAL_DATA)),
        'test': split_type_counts(SingleDefectDataset(TEST_DATA)),
    }
    pseudo_rows = []
    for split, path in [('val', VAL_DATA), ('test', TEST_DATA)]:
        metric_rows, _ = evaluate_pseudo_labels(split, path, device)
        pseudo_rows.extend(metric_rows)
    write_csv(PSEUDO_LABEL_METRICS_PATH, pseudo_rows)

    pseudo_test = get_metric(pseudo_rows, 'oracle_quad_pseudo_label', 0.50)
    if pseudo_test is None or float(pseudo_test['iou']) < 0.80:
        write_csv(SCREENING_PATH, [])
        metric_rows = load_reference_rows() + load_oracle_reference()
        write_csv(CANDIDATE_METRICS_PATH, metric_rows)
        write_summary(pseudo_rows, metric_rows, None, False, False, {'pseudo_label_quality_passed': False}, [], [], [], counts)
        print('pseudo_label_quality_passed=False')
        return

    train_dataset = OracleQuadDataset(TRAIN_DATA)
    init_bias = train_dataset.oracle_vertices_norm.mean(axis=0)
    pos_weight, mask_fraction = compute_pos_weight(train_dataset.base)
    print(f'train mask positive fraction={mask_fraction:.6f}, pos_weight={pos_weight:.6f}')

    checkpoint_42, info_42 = train_one_seed(42, device, pos_weight, init_bias)
    val_rows, _, _ = evaluate_checkpoints(
        {42: checkpoint_42},
        'oracle_quad_supervised_screening',
        'val',
        VAL_DATA,
        THRESHOLDS,
        device,
    )
    selected = select_threshold(val_rows, 'oracle_quad_supervised_screening')
    selected_threshold = float(selected['threshold'])
    test_rows, sample_rows_by_seed_threshold, preview_cache = evaluate_checkpoints(
        {42: checkpoint_42},
        'oracle_quad_supervised_screening',
        'test',
        TEST_DATA,
        [selected_threshold],
        device,
    )
    metric_rows = load_reference_rows() + load_oracle_reference() + val_rows + test_rows
    stage_b_entered, gate_checks = seed_gate(metric_rows, selected_threshold)
    checkpoint_paths = [checkpoint_42]
    best_infos = [info_42]
    preview_paths = write_previews(42, selected_threshold, sample_rows_by_seed_threshold, preview_cache)
    accepted = False
    if stage_b_entered:
        for seed in [123, 2026]:
            path, info = train_one_seed(seed, device, pos_weight, init_bias)
            checkpoint_paths.append(path)
            best_infos.append(info)
        final_checkpoints = {seed: path for seed, path in zip(SEEDS, checkpoint_paths)}
        final_val_rows, _, _ = evaluate_checkpoints(
            final_checkpoints,
            'oracle_quad_supervised_candidate',
            'val',
            VAL_DATA,
            THRESHOLDS,
            device,
        )
        final_selected = select_threshold(final_val_rows, 'oracle_quad_supervised_candidate')
        selected_threshold = float(final_selected['threshold'])
        final_test_rows, sample_rows_by_seed_threshold, preview_cache = evaluate_checkpoints(
            final_checkpoints,
            'oracle_quad_supervised_candidate',
            'test',
            TEST_DATA,
            [selected_threshold],
            device,
        )
        metric_rows.extend(final_val_rows)
        metric_rows.extend(final_test_rows)
        preview_paths = write_previews(42, selected_threshold, sample_rows_by_seed_threshold, preview_cache)
        accepted, final_checks = final_gate(metric_rows, selected_threshold)
        gate_checks.update(final_checks)

    write_screening(val_rows)
    write_csv(CANDIDATE_METRICS_PATH, metric_rows)
    write_summary(pseudo_rows, metric_rows, selected_threshold, stage_b_entered, accepted, gate_checks, checkpoint_paths, best_infos, preview_paths, counts)
    seed42 = get_metric(metric_rows, 'oracle_quad_supervised_screening', selected_threshold, seed=42)
    print(f'pseudo_label_quality_passed=True')
    print(f'seed42_completed=True')
    print(f'selected_threshold={selected_threshold}')
    print(
        f"seed42_test: IoU={float(seed42['iou']):.6f}, Dice={float(seed42['dice']):.6f}, "
        f"area_error={float(seed42['area_error']):.6f}, vertex_mae={float(seed42['vertex_mae']):.6f}, "
        f"invalid={float(seed42['invalid_count']):.0f}, out_of_image={float(seed42['out_of_image_count']):.0f}"
    )
    print(f'entered_3_seed={stage_b_entered}')
    print(f'accepted={accepted}')
    print(f'wrote pseudo-label metrics: {PSEUDO_LABEL_METRICS_PATH}')
    print(f'wrote screening: {SCREENING_PATH}')
    print(f'wrote candidate metrics: {CANDIDATE_METRICS_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    print(f'wrote previews: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
