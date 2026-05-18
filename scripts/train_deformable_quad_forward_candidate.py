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
    CURRENT_BASELINE_CHECKPOINTS,
    CURRENT_BASELINE_THRESHOLD,
    GEOMETRY_ONLY_THRESHOLD,
    build_sample_rows,
    dataset_low_signal_indices,
    evaluate_current_baseline,
    find_row,
    get_metric_row,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    BzEncoder,
    compute_mask_metrics,
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
RESIDUAL_SCALE = 0.35
MIN_W = 0.20
MIN_H = 0.20
MAX_W = 10.0
MAX_H = 8.0
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'deformable_quad_forward_candidate'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_deformable_quad_forward_candidate_summary.txt'
ORACLE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_deformable_quad_oracle_metrics.csv'
SCREENING_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_deformable_quad_forward_screening.csv'
CANDIDATE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_deformable_quad_forward_candidate_metrics.csv'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'deformable_quad_forward_candidate'
FORWARD_SURROGATE = 'checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt'
ROTATED_BOX_REFERENCE_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_geometry_forward_consistency_candidate_metrics.csv'

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
    'avg_polygon_area',
    'invalid_count',
    'out_of_image_count',
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
    if not Path(project_path(FORWARD_SURROGATE)).exists():
        missing.append(FORWARD_SURROGATE)
    if not ROTATED_BOX_REFERENCE_METRICS.exists():
        missing.append(str(ROTATED_BOX_REFERENCE_METRICS.relative_to(ROOT)))
    for path in CURRENT_BASELINE_CHECKPOINTS.values():
        if not Path(project_path(path)).exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError('Missing required input(s): ' + ', '.join(missing))


class DeformableQuadForwardModel(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channels=1,
        latent_dim=64,
        x_range=(-15.0, 15.0),
        y_range=(0.0, 10.0),
    ):
        super().__init__()
        self.x_min, self.x_max = map(float, x_range)
        self.y_min, self.y_max = map(float, y_range)
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
            nn.Linear(96, 13),
        )
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.bias[:5] = torch.tensor([0.0, 0.0, -0.7, -0.7, 0.0])
                final.bias[5:] = 0.0

    def decode(self, raw):
        cx = self.x_min + torch.sigmoid(raw[:, 0]) * (self.x_max - self.x_min)
        cy = self.y_min + torch.sigmoid(raw[:, 1]) * (self.y_max - self.y_min)
        width = MIN_W + torch.sigmoid(raw[:, 2]) * (MAX_W - MIN_W)
        height = MIN_H + torch.sigmoid(raw[:, 3]) * (MAX_H - MIN_H)
        angle = np.pi * torch.tanh(raw[:, 4])
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            device=raw.device,
            dtype=raw.dtype,
        )[None, :, :]
        scale = torch.stack([width, height], dim=1)[:, None, :]
        local = base * scale
        residual = torch.tanh(raw[:, 5:].reshape(-1, 4, 2)) * (RESIDUAL_SCALE * scale)
        local = local + residual
        cos_a = torch.cos(angle)[:, None]
        sin_a = torch.sin(angle)[:, None]
        x = cx[:, None] + cos_a * local[:, :, 0] - sin_a * local[:, :, 1]
        y = cy[:, None] + sin_a * local[:, :, 0] + cos_a * local[:, :, 1]
        return torch.stack([x, y], dim=2)

    def forward(self, bz_signal, coords=None, return_vertices=False):
        vertices = self.decode(self.head(self.bz_encoder(bz_signal)))
        if coords is None:
            raise ValueError('DeformableQuadForwardModel requires coordinate tuple (x_coords, y_coords).')
        logits = rasterize_quad(vertices, coords)
        if return_vertices:
            return logits, vertices
        return logits


def rasterize_quad(vertices, coords):
    x_coords, y_coords = coords
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    x_grid = xx.to(vertices.device, dtype=vertices.dtype)[None, :, :]
    y_grid = yy.to(vertices.device, dtype=vertices.dtype)[None, :, :]
    verts = vertices
    signed_area = 0.5 * torch.sum(
        verts[:, :, 0] * torch.roll(verts[:, :, 1], shifts=-1, dims=1)
        - torch.roll(verts[:, :, 0], shifts=-1, dims=1) * verts[:, :, 1],
        dim=1,
    )
    orient = torch.where(signed_area >= 0, 1.0, -1.0).to(vertices.dtype)
    edge_scores = []
    for i in range(4):
        p0 = verts[:, i, :]
        p1 = verts[:, (i + 1) % 4, :]
        ex = p1[:, 0] - p0[:, 0]
        ey = p1[:, 1] - p0[:, 1]
        px = x_grid - p0[:, 0, None, None]
        py = y_grid - p0[:, 1, None, None]
        cross = ex[:, None, None] * py - ey[:, None, None] * px
        edge_len = torch.sqrt(ex ** 2 + ey ** 2).clamp_min(1e-6)
        edge_scores.append((orient[:, None, None] * cross) / edge_len[:, None, None])
    inside_score = torch.stack(edge_scores, dim=0).amin(dim=0)
    return inside_score.reshape(vertices.shape[0], -1) / RASTER_TEMPERATURE


def coord_tuple(dataset, device):
    return (
        torch.from_numpy(dataset.x.astype(np.float32)).to(device),
        torch.from_numpy(dataset.y.astype(np.float32)).to(device),
    )


def polygon_area(vertices):
    v = np.asarray(vertices, dtype=np.float64)
    return 0.5 * float(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))


def order_vertices(vertices):
    vertices = np.asarray(vertices, dtype=np.float32)
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    ordered = vertices[np.argsort(angles)]
    if polygon_area(ordered) < 0:
        ordered = ordered[::-1].copy()
    return ordered.astype(np.float32)


def pca_box_vertices(mask, dataset):
    x_vals = np.asarray(dataset.x, dtype=np.float32)
    y_vals = np.asarray(dataset.y, dtype=np.float32)
    yy, xx = np.where(mask)
    if len(xx) == 0:
        cx = float(x_vals[len(x_vals) // 2])
        cy = float(y_vals[len(y_vals) // 2])
        width = height = 1.0
        angle = 0.0
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32)
        return order_vertices(corners + np.array([cx, cy], dtype=np.float32))
    coords = np.stack([x_vals[xx], y_vals[yy]], axis=1).astype(np.float64)
    center = coords.mean(axis=0)
    if len(coords) < 3:
        axes = np.eye(2, dtype=np.float64)
    else:
        centered = coords - center[None, :]
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        axes = evecs[:, order]
    proj = (coords - center[None, :]) @ axes
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    dx = float(abs(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 1.0
    dy = float(abs(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 1.0
    mins -= np.array([dx, dy]) * 0.5
    maxs += np.array([dx, dy]) * 0.5
    local = np.array([
        [mins[0], mins[1]],
        [maxs[0], mins[1]],
        [maxs[0], maxs[1]],
        [mins[0], maxs[1]],
    ])
    vertices = center[None, :] + local @ axes.T
    return order_vertices(vertices)


def quad_oracle_vertices(mask, dataset):
    x_vals = np.asarray(dataset.x, dtype=np.float32)
    y_vals = np.asarray(dataset.y, dtype=np.float32)
    yy, xx = np.where(mask)
    if len(xx) < 4:
        return pca_box_vertices(mask, dataset)
    coords = np.stack([x_vals[xx], y_vals[yy]], axis=1).astype(np.float64)
    center = coords.mean(axis=0)
    cov = np.cov((coords - center[None, :]).T)
    evals, evecs = np.linalg.eigh(cov)
    axes = evecs[:, np.argsort(evals)[::-1]]
    local = (coords - center[None, :]) @ axes
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
    points = []
    fallback_box = pca_box_vertices(mask, dataset)
    fallback_local = (fallback_box - center[None, :]) @ axes
    for i, sign in enumerate(signs):
        quadrant = (np.sign(local[:, 0]) == sign[0]) & (np.sign(local[:, 1]) == sign[1])
        scores = local @ sign
        if np.any(quadrant):
            idx_candidates = np.where(quadrant)[0]
            idx = idx_candidates[int(np.argmax(scores[idx_candidates]))]
            point = coords[idx]
        else:
            point = center + fallback_local[i] @ axes.T
        points.append(point)
    points = np.asarray(points, dtype=np.float64)
    points = center[None, :] + 1.04 * (points - center[None, :])
    return order_vertices(points)


def vertices_out_of_image(vertices, dataset):
    x_min, x_max = float(np.min(dataset.x)), float(np.max(dataset.x))
    y_min, y_max = float(np.min(dataset.y)), float(np.max(dataset.y))
    v = np.asarray(vertices)
    return bool(np.any(v[:, 0] < x_min) or np.any(v[:, 0] > x_max) or np.any(v[:, 1] < y_min) or np.any(v[:, 1] > y_max))


def edges_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def invalid_quad(vertices):
    v = np.asarray(vertices, dtype=np.float64)
    if abs(polygon_area(v)) < 1e-6:
        return True
    return bool(edges_intersect(v[0], v[1], v[2], v[3]) or edges_intersect(v[1], v[2], v[3], v[0]))


def render_vertices(vertices_np, dataset, device):
    coords = coord_tuple(dataset, device)
    vertices = torch.from_numpy(vertices_np.astype(np.float32)).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(rasterize_quad(vertices, coords)).reshape(vertices_np.shape[0], *dataset.mu_maps.shape[1:])
    return probs.detach().cpu().numpy().astype(np.float32)


def compute_bz_mses(forward_model, prob_maps, target_signals, device, batch_size=32):
    out = np.empty((len(prob_maps),), dtype=np.float32)
    forward_model.eval()
    with torch.no_grad():
        for start in range(0, len(prob_maps), batch_size):
            end = min(start + batch_size, len(prob_maps))
            prob = torch.from_numpy(prob_maps[start:end]).to(device=device, dtype=torch.float32)
            target = torch.from_numpy(target_signals[start:end]).to(device=device, dtype=torch.float32)
            pred = forward_model(prob.unsqueeze(1))
            out[start:end] = torch.mean((pred - target) ** 2, dim=1).detach().cpu().numpy()
    return out


def summarize_samples_with_bz(rows, vertices=None, dataset=None):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
            'avg_polygon_area': float('nan'),
            'invalid_count': 0,
            'out_of_image_count': 0,
        })
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse']:
        summary[key] = safe_nanmean([float(row[key]) for row in rows])
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    if vertices is None or dataset is None:
        summary['avg_polygon_area'] = float('nan')
        summary['invalid_count'] = 0
        summary['out_of_image_count'] = 0
    else:
        sample_indices = [int(row['sample_index']) for row in rows]
        areas = [abs(polygon_area(vertices[idx])) for idx in sample_indices]
        summary['avg_polygon_area'] = safe_nanmean(areas)
        summary['invalid_count'] = int(sum(invalid_quad(vertices[idx]) for idx in sample_indices))
        summary['out_of_image_count'] = int(sum(vertices_out_of_image(vertices[idx], dataset) for idx in sample_indices))
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


def summarize_candidate(sample_rows, candidate, seed, split, threshold, vertices=None, dataset=None):
    rows = []
    overall = summarize_samples_with_bz(sample_rows, vertices, dataset)
    area_summaries = {
        group: summarize_samples_with_bz([row for row in sample_rows if row['area_bin'] == group], vertices, dataset)
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
            summarize_samples_with_bz([row for row in sample_rows if row['signal_bin'] == group], vertices, dataset),
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
            summarize_samples_with_bz([row for row in sample_rows if row['defect_type'] == defect_type], vertices, dataset),
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
            values = [float(row[key]) for row in selected]
            mean_row[key] = safe_nanmean(values)
            std_row[key] = safe_nanstd(values)
        rows.extend([mean_row, std_row])
    return rows


def sample_rows_from_prob(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset):
    area_edges = get_area_edges(dataset)
    low_signal_indices = dataset_low_signal_indices(dataset)
    return build_sample_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset, area_edges, low_signal_indices)


def evaluate_oracle_split(split, data_path, device, forward_model, surrogate_checkpoint):
    dataset = SingleDefectDataset(
        data_path,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    true_masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    box_vertices = np.stack([pca_box_vertices(mask, dataset) for mask in true_masks], axis=0)
    quad_vertices = np.stack([quad_oracle_vertices(mask, dataset) for mask in true_masks], axis=0)
    rows = []
    for candidate, vertices in [('box_oracle', box_vertices), ('deformable_quad_oracle', quad_vertices)]:
        prob_maps = render_vertices(vertices, dataset, device)
        bz_mses = compute_bz_mses(forward_model, prob_maps, dataset.signals, device)
        sample_rows = sample_rows_from_prob(candidate, 'oracle', split, 0.50, prob_maps, true_masks, bz_mses, dataset)
        rows.extend(summarize_candidate(sample_rows, candidate, 'oracle', split, 0.50, vertices, dataset))
    return rows


def write_csv(path, rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'threshold', 'n'] + METRIC_KEYS
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def load_reference_rows():
    keep = {
        'current_forward_baseline_single_defect_mean',
        'current_forward_baseline_single_defect_std',
        'geometry_forward_screening',
        'geometry_forward_candidate_mean',
        'geometry_forward_candidate_std',
    }
    rows = []
    with open(ROTATED_BOX_REFERENCE_METRICS, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['candidate'] in keep and row['split'] == 'test':
                for key in METRIC_KEYS:
                    row.setdefault(key, '')
                rows.append(row)
    return rows


def oracle_capacity_pass(oracle_rows):
    box = get_metric_row(oracle_rows, 'box_oracle', 0.50)
    quad = get_metric_row(oracle_rows, 'deformable_quad_oracle', 0.50)
    box_polygon = get_metric_row(oracle_rows, 'box_oracle', 0.50, 'defect_type', 'polygon')
    quad_polygon = get_metric_row(oracle_rows, 'deformable_quad_oracle', 0.50, 'defect_type', 'polygon')
    if box is None or quad is None or box_polygon is None or quad_polygon is None:
        return False, {'reason': 'missing oracle rows'}
    checks = {
        'quad_overall_composite_not_worse': float(quad['composite']) >= float(box['composite']) - 0.02,
        'quad_polygon_iou_not_worse': float(quad_polygon['iou']) >= float(box_polygon['iou']) - 0.02,
        'quad_invalid_count_zero': float(quad['invalid_count']) == 0.0,
    }
    return bool(all(checks.values())), checks


@torch.no_grad()
def predict_quad_maps(model, dataset, coords, device, forward_model, bz_target_dataset=None):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    vertices = np.empty((len(dataset), 4, 2), dtype=np.float32)
    bz_mses = np.empty((len(dataset),), dtype=np.float32)
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        logits, batch_vertices = model(signals, coords, return_vertices=True)
        probs = torch.sigmoid(logits).reshape(signals.shape[0], *grid_shape)
        if bz_target_dataset is None:
            target_bz = signals
        else:
            batch_indices = indices.detach().cpu().numpy().astype(np.int64)
            target_bz = torch.from_numpy(bz_target_dataset.signals[batch_indices]).to(device)
        bz_hat = forward_model(probs.unsqueeze(1))
        batch_bz_mse = torch.mean((bz_hat - target_bz) ** 2, dim=1).detach().cpu().numpy()
        prob_np = probs.detach().cpu().numpy()
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = prob_np[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
            vertices[sample_idx] = batch_vertices[batch_pos].detach().cpu().numpy()
            bz_mses[sample_idx] = batch_bz_mse[batch_pos]
    return prob_maps, true_masks, vertices, bz_mses


def evaluate_model_for_selection(model, dataset, coords, device):
    prob_maps, true_masks, _, bz_mses = predict_quad_maps(
        model,
        dataset,
        coords,
        device,
        forward_model=lambda mask_prob: torch.zeros((mask_prob.shape[0], dataset.signals.shape[-1]), device=mask_prob.device),
    )
    rows = sample_rows_from_prob('selection', 'selection', 'val', TRAIN_SELECTION_THRESHOLD, prob_maps, true_masks, bz_mses, dataset)
    summary = summarize_samples_with_bz(rows)
    return summary


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
    model = DeformableQuadForwardModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
        x_range=(float(train_dataset.x.min()), float(train_dataset.x.max())),
        y_range=(float(train_dataset.y.min()), float(train_dataset.y.max())),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    coords_train = coord_tuple(train_dataset, device)
    coords_val = coord_tuple(val_dataset, device)

    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_deformable_quad_forward_seed{seed}.pt'
    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu')
        print(f'Reusing existing checkpoint for seed={seed}: {best_path}')
        return best_path, checkpoint.get('val_metrics', {})

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_forward = 0.0
        total_samples = 0
        for signals, mu_targets, _ in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).float()
            optimizer.zero_grad()
            logits = model(signals, coords_train)
            loss_mask, _, _ = mask_loss(logits, target_mask, pos_weight)
            mask_prob = torch.sigmoid(logits).reshape(signals.shape[0], *train_dataset.mu_maps.shape[1:])
            bz_hat = forward_model(mask_prob.unsqueeze(1))
            loss_forward = F.mse_loss(bz_hat, signals)
            loss = loss_mask + LAMBDA_FORWARD * loss_forward
            loss.backward()
            optimizer.step()
            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_forward += float(loss_forward.item()) * batch_size
            total_samples += batch_size
        val_summary = evaluate_model_for_selection(model, val_dataset, coords_val, device)
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
                'lambda_forward': LAMBDA_FORWARD,
            }
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'args': {
                        'model': 'deformable_quad_forward_candidate',
                        'seed': seed,
                        'latent_dim': LATENT_DIM,
                        'signal_channels': signal_channels,
                        'signal_length': signal_length,
                        'lambda_forward': LAMBDA_FORWARD,
                        'residual_scale': RESIDUAL_SCALE,
                        'temperature': RASTER_TEMPERATURE,
                        'mask_target': 'target_mu_norm < 0.5',
                    },
                    'signal_mean': train_dataset.signal_mean,
                    'signal_std': train_dataset.signal_std,
                    'val_metrics': best_info,
                },
                best_path,
            )
        print(
            f"seed={seed} epoch={epoch:03d} | loss={total_loss / total_samples:.6e} | "
            f"forward_mse={total_forward / total_samples:.6e} | val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | val_area_error={val_summary['area_error']:.6e} | "
            f"score={score:.6e}"
        )
    return best_path, best_info


def load_quad_checkpoint(path, dataset, device):
    checkpoint = torch.load(path, map_location='cpu')
    signal_length, signal_channels = signal_shape_info(dataset.signals)
    model = DeformableQuadForwardModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=int(checkpoint.get('args', {}).get('latent_dim', LATENT_DIM)),
        x_range=(float(dataset.x.min()), float(dataset.x.max())),
        y_range=(float(dataset.y.min()), float(dataset.y.max())),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_quad_checkpoints(checkpoints, candidate, split, data_path, thresholds, device, forward_model, surrogate_checkpoint):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    vertex_cache = {}
    bz_target_dataset = SingleDefectDataset(
        data_path,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dataset = SingleDefectDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        model, _ = load_quad_checkpoint(checkpoint_path, dataset, device)
        coords = coord_tuple(dataset, device)
        prob_maps, true_masks, vertices, bz_mses = predict_quad_maps(
            model,
            dataset,
            coords,
            device,
            forward_model,
            bz_target_dataset=bz_target_dataset,
        )
        vertex_cache[seed] = (vertices, prob_maps, true_masks, dataset)
        for threshold in thresholds:
            sample_rows = sample_rows_from_prob(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset)
            sample_rows_by_seed_threshold[(seed, threshold)] = sample_rows
            metric_rows.extend(summarize_candidate(sample_rows, candidate, seed, split, threshold, vertices, dataset))
    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows(metric_rows, candidate, split, threshold))
    return metric_rows, sample_rows_by_seed_threshold, vertex_cache


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


def screening_pass(rows, selected_threshold):
    base = find_row(rows, 'current_forward_baseline_single_defect_mean', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    rotated = get_metric_row(rows, 'geometry_forward_screening', GEOMETRY_ONLY_THRESHOLD, seed=42)
    cand = get_metric_row(rows, 'deformable_quad_forward_screening', selected_threshold, seed=42)
    cand_poly = get_metric_row(rows, 'deformable_quad_forward_screening', selected_threshold, 'defect_type', 'polygon', seed=42)
    cand_rot = get_metric_row(rows, 'deformable_quad_forward_screening', selected_threshold, 'defect_type', 'rotated_rect', seed=42)
    base_poly = find_row(rows, 'current_forward_baseline_single_defect_mean', 'defect_type', 'polygon', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    base_rot = find_row(rows, 'current_forward_baseline_single_defect_mean', 'defect_type', 'rotated_rect', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    if base is None or rotated is None or cand is None:
        return False, {'reason': 'missing comparison rows'}
    checks = {
        'area_error_below_rotated_box_forward': float(cand['area_error']) <= float(rotated['area_error']) - 0.03,
        'iou_not_obviously_below_current': float(cand['iou']) >= float(base['iou']) - 0.02,
        'dice_not_obviously_below_current': float(cand['dice']) >= float(base['dice']) - 0.02,
        'pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'bz_mse_not_obviously_worse_than_current': float(cand['bz_mse']) <= float(base['bz_mse']) * 1.15,
        'vertex_invalid_zero': float(cand['invalid_count']) == 0.0,
        'polygon_or_rotated_signal': (
            cand_poly is not None and base_poly is not None and float(cand_poly['iou']) >= float(base_poly['iou']) - 0.02
        ) or (
            cand_rot is not None and base_rot is not None and float(cand_rot['iou']) >= float(base_rot['iou']) - 0.02
        ),
    }
    return bool(all(checks.values())), checks


def final_accept(rows, selected_threshold):
    base = find_row(rows, 'current_forward_baseline_single_defect_mean', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    cand = get_metric_row(rows, 'deformable_quad_forward_candidate_mean', selected_threshold)
    rotated = find_row(rows, 'geometry_forward_candidate_mean', split='test', threshold=GEOMETRY_ONLY_THRESHOLD)
    if base is None or cand is None or rotated is None:
        return False, {'reason': 'missing final rows'}
    checks = {
        'final_area_error_below_rotated_box_forward': float(cand['area_error']) <= float(rotated['area_error']) - 0.03,
        'final_iou_not_obviously_below_current': float(cand['iou']) >= float(base['iou']) - 0.02,
        'final_dice_not_obviously_below_current': float(cand['dice']) >= float(base['dice']) - 0.02,
        'final_pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'final_bz_mse_not_obviously_worse_than_current': float(cand['bz_mse']) <= float(base['bz_mse']) * 1.15,
        'final_vertex_invalid_zero': float(cand['invalid_count']) == 0.0,
    }
    return bool(all(checks.values())), checks


def write_screening(rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'threshold', 'n'] + METRIC_KEYS
    screening = [
        row for row in rows
        if row['candidate'] in {'deformable_quad_forward_screening', 'deformable_quad_forward_screening_mean'}
        and row['split'] == 'val'
    ]
    with open(SCREENING_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in screening:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_previews(seed, selected_threshold, sample_rows_by_seed_threshold, vertex_cache):
    if (seed, selected_threshold) not in sample_rows_by_seed_threshold:
        return []
    vertices, prob_maps, true_masks, dataset = vertex_cache[seed]
    rows = sample_rows_by_seed_threshold[(seed, selected_threshold)]
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
        prob = prob_maps[sample_idx]
        pred_mask = prob >= selected_threshold
        true_mask = true_masks[sample_idx]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        axes[0].imshow(true_mask, origin='lower', cmap='gray')
        axes[0].set_title('true mask')
        im = axes[1].imshow(prob, origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('pred probability')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(pred_mask, origin='lower', cmap='gray')
        axes[2].set_title(f'pred mask t={selected_threshold:.2f}')
        draw_vertices(axes[2], vertices[sample_idx], dataset)
        axes[3].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[3].imshow(pred_mask, origin='lower', cmap='Reds', alpha=0.35)
        if np.any(true_mask) and np.any(~true_mask):
            axes[3].contour(true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        if np.any(pred_mask) and np.any(~pred_mask):
            axes[3].contour(pred_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        draw_vertices(axes[3], vertices[sample_idx], dataset)
        axes[3].set_title('overlay + quad')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={row['original_index']} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} BzMSE={float(row['bz_mse']):.3e}",
            fontsize=9,
        )
        path = PREVIEW_DIR / f'deformable_quad_seed{seed}_rank{rank:02d}_sample{row["original_index"]}_{row["defect_type"]}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def draw_vertices(ax, vertices, dataset):
    v = np.asarray(vertices, dtype=np.float32)
    v = np.vstack([v, v[0]])
    cols = np.interp(v[:, 0], dataset.x, np.arange(len(dataset.x)))
    rows = np.interp(v[:, 1], dataset.y, np.arange(len(dataset.y)))
    ax.plot(cols, rows, color='yellow', linewidth=1.4)


def fmt(row, key):
    if row is None or row.get(key, '') == '':
        return 'NA'
    return f"{float(row[key]):.4f}"


def table_row(label, row):
    return (
        f"| {label} | {fmt(row, 'threshold')} | {fmt(row, 'n')} | {fmt(row, 'iou')} | "
        f"{fmt(row, 'dice')} | {fmt(row, 'area_error')} | {fmt(row, 'center_error')} | "
        f"{fmt(row, 'pred_area_zero')} | {fmt(row, 'bz_mse')} | {fmt(row, 'invalid_count')} |"
    )


def write_summary(oracle_rows, metric_rows, screening_rows, selected_threshold, oracle_ok, oracle_checks, stage_b_entered, accepted, gate_checks, checkpoint_paths, best_infos, preview_paths, counts):
    current = find_row(metric_rows, 'current_forward_baseline_single_defect_mean', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    rotated_seed = get_metric_row(metric_rows, 'geometry_forward_screening', GEOMETRY_ONLY_THRESHOLD, seed=42)
    box_oracle = get_metric_row(oracle_rows, 'box_oracle', 0.50)
    quad_oracle = get_metric_row(oracle_rows, 'deformable_quad_oracle', 0.50)
    seed42 = get_metric_row(metric_rows, 'deformable_quad_forward_screening', selected_threshold, seed=42) if selected_threshold is not None else None
    final = get_metric_row(metric_rows, 'deformable_quad_forward_candidate_mean', selected_threshold) if stage_b_entered else None
    lines = [
        '# v3_complex deformable quadrilateral + forward consistency candidate',
        '',
        '## Single-defect subset',
        '',
        'Subset is built in-script from v3_complex by keeping polygon / rotated_rect and excluding multi_defect. Original data files are not modified.',
        '',
        '| split | polygon | rotated_rect | total |',
        '|---|---:|---:|---:|',
    ]
    for split, count in counts.items():
        lines.append(f"| {split} | {count.get('polygon', 0)} | {count.get('rotated_rect', 0)} | {sum(count.values())} |")
    lines.extend([
        '',
        '## Method',
        '',
        '* Geometry head: BzEncoder -> coarse rotated box + 4 bounded corner residual offsets.',
        f'* Residual scale is fixed at `{RESIDUAL_SCALE}` of width/height. No residual-scale search is used.',
        f'* Rasterizer: pure PyTorch soft quadrilateral half-plane rasterizer with fixed temperature `{RASTER_TEMPERATURE}`.',
        f'* Loss: BCEWithLogits + soft Dice + `{LAMBDA_FORWARD}` * frozen mask-to-Bz surrogate MSE.',
        '* No SDF loss, boundary head, vertex supervision, polygon-vertex count prediction, retrieval, post-processing, lambda search, or temperature search is used.',
        '',
        '## Oracle capacity check',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | invalid_count |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        table_row('box_oracle test', box_oracle),
        table_row('deformable_quad_oracle test', quad_oracle),
        '',
    ])
    for key, value in oracle_checks.items():
        lines.append(f'* {key}: {value}')
    lines.extend([
        f'* oracle_capacity_passed: {oracle_ok}',
        '',
        '## Seed=42 gate',
        '',
        f'* completed: {seed42 is not None}',
        f"* validation-selected threshold: `{selected_threshold if selected_threshold is not None else 'NA'}`",
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | invalid_count |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        table_row('CURRENT_BASELINE on same subset', current),
        table_row('rotated-box geometry+forward seed=42', rotated_seed),
        table_row('deformable quad + forward seed=42', seed42),
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
        lines.extend(['', '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | checkpoint |', '|---:|---:|---:|---:|---:|---:|---|'])
        for path, info in zip(checkpoint_paths, best_infos):
            lines.append(
                f"| {info.get('seed', 'NA')} | {info.get('epoch', 'NA')} | {float(info.get('selection_score', float('nan'))):.6e} | "
                f"{float(info.get('val_iou', float('nan'))):.4f} | {float(info.get('val_dice', float('nan'))):.4f} | "
                f"{float(info.get('val_area_error', float('nan'))):.4f} | `{path.relative_to(ROOT)}` |"
            )
    if final is not None:
        lines.extend(['', '## 3-seed result', '', table_row('deformable quad + forward 3-seed mean', final)])
    lines.extend([
        '',
        '## Conclusion',
        '',
    ])
    if not oracle_ok:
        lines.append('Oracle capacity did not pass; deformable quad branch is stopped before training.')
    elif not stage_b_entered:
        lines.append('Seed=42 did not satisfy the gate; deformable quad branch is stopped. Do not continue deformable quad v2, residual-scale tuning, lambda tuning, temperature tuning, polygon vertices, or multi-component variants from this result.')
    elif not accepted:
        lines.append('3-seed validation did not satisfy the gate; deformable quad branch is stopped. Do not continue deformable quad v2, residual-scale tuning, lambda tuning, temperature tuning, polygon vertices, or multi-component variants from this result.')
    else:
        lines.append('Deformable quad + forward consistency satisfies the bounded candidate gate and has candidate value, pending review.')
    if preview_paths:
        lines.append(f'Preview files: {len(preview_paths)} written to `{PREVIEW_DIR.relative_to(ROOT)}`.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = load_forward_surrogate(device)

    counts = {
        'train': split_type_counts(SingleDefectDataset(TRAIN_DATA)),
        'val': split_type_counts(SingleDefectDataset(VAL_DATA)),
        'test': split_type_counts(SingleDefectDataset(TEST_DATA)),
    }
    oracle_rows = []
    oracle_rows.extend(evaluate_oracle_split('val', VAL_DATA, device, forward_model, surrogate_checkpoint))
    oracle_rows.extend(evaluate_oracle_split('test', TEST_DATA, device, forward_model, surrogate_checkpoint))
    write_csv(ORACLE_METRICS_PATH, oracle_rows)
    oracle_ok, oracle_checks = oracle_capacity_pass([row for row in oracle_rows if row['split'] == 'test'])

    metric_rows = load_reference_rows()
    baseline_test_rows = evaluate_current_baseline('test', TEST_DATA, device, forward_model)
    # Keep the official reference rows from Step 18.5 for CURRENT_BASELINE and rotated-box comparisons.
    if not any(row['candidate'] == 'current_forward_baseline_single_defect_mean' for row in metric_rows):
        metric_rows.extend(baseline_test_rows)

    checkpoint_paths = []
    best_infos = []
    screening_rows = []
    selected_threshold = None
    stage_b_entered = False
    accepted = False
    gate_checks = {}
    preview_paths = []

    if oracle_ok:
        pos_weight, mask_fraction = compute_pos_weight(SingleDefectDataset(TRAIN_DATA))
        print(f'train mask positive fraction={mask_fraction:.6f}, pos_weight={pos_weight:.6f}')
        checkpoint_42, info_42 = train_one_seed(42, device, pos_weight, forward_model, surrogate_checkpoint)
        checkpoint_paths.append(checkpoint_42)
        best_infos.append(info_42)
        val_rows, _, _ = evaluate_quad_checkpoints(
            {42: checkpoint_42},
            'deformable_quad_forward_screening',
            'val',
            VAL_DATA,
            THRESHOLDS,
            device,
            forward_model,
            surrogate_checkpoint,
        )
        selected = select_threshold(val_rows, 'deformable_quad_forward_screening')
        selected_threshold = float(selected['threshold'])
        screening_rows = val_rows
        test_rows, sample_rows_by_seed_threshold, vertex_cache = evaluate_quad_checkpoints(
            {42: checkpoint_42},
            'deformable_quad_forward_screening',
            'test',
            TEST_DATA,
            [selected_threshold],
            device,
            forward_model,
            surrogate_checkpoint,
        )
        metric_rows.extend(val_rows)
        metric_rows.extend(test_rows)
        preview_paths = write_previews(42, selected_threshold, sample_rows_by_seed_threshold, vertex_cache)
        stage_b_entered, gate_checks = screening_pass(metric_rows, selected_threshold)
        if stage_b_entered:
            for seed in [123, 2026]:
                path, info = train_one_seed(seed, device, pos_weight, forward_model, surrogate_checkpoint)
                checkpoint_paths.append(path)
                best_infos.append(info)
            final_checkpoints = {seed: path for seed, path in zip(SEEDS, checkpoint_paths)}
            final_val_rows, _, _ = evaluate_quad_checkpoints(
                final_checkpoints,
                'deformable_quad_forward_candidate',
                'val',
                VAL_DATA,
                THRESHOLDS,
                device,
                forward_model,
                surrogate_checkpoint,
            )
            final_selected = select_threshold(final_val_rows, 'deformable_quad_forward_candidate')
            selected_threshold = float(final_selected['threshold'])
            final_test_rows, sample_rows_by_seed_threshold, vertex_cache = evaluate_quad_checkpoints(
                final_checkpoints,
                'deformable_quad_forward_candidate',
                'test',
                TEST_DATA,
                [selected_threshold],
                device,
                forward_model,
                surrogate_checkpoint,
            )
            metric_rows.extend(final_val_rows)
            metric_rows.extend(final_test_rows)
            preview_paths = write_previews(42, selected_threshold, sample_rows_by_seed_threshold, vertex_cache)
            accepted, final_checks = final_accept(metric_rows, selected_threshold)
            gate_checks.update(final_checks)

    write_screening(screening_rows)
    write_csv(CANDIDATE_METRICS_PATH, metric_rows)
    write_summary(
        oracle_rows,
        metric_rows,
        screening_rows,
        selected_threshold,
        oracle_ok,
        oracle_checks,
        stage_b_entered,
        accepted,
        gate_checks,
        checkpoint_paths,
        best_infos,
        preview_paths,
        counts,
    )
    print(f'oracle_capacity_passed={oracle_ok}')
    print(f'seed42_completed={bool(checkpoint_paths)}')
    print(f'selected_threshold={selected_threshold}')
    print(f'entered_3_seed={stage_b_entered}')
    print(f'accepted={accepted}')
    print(f'wrote oracle metrics: {ORACLE_METRICS_PATH}')
    print(f'wrote screening: {SCREENING_PATH}')
    print(f'wrote candidate metrics: {CANDIDATE_METRICS_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    if preview_paths:
        print(f'wrote previews: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
