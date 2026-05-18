import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import project_path, signal_shape_info  # noqa: E402
from scripts.train_geometry_boundary_candidate import (  # noqa: E402
    MASK_THRESHOLD_NORM,
    RASTER_TEMPERATURE,
    SINGLE_DEFECT_TYPES,
    TEST_DATA,
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
    find_row,
    row_line,
    summarize_candidate,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    get_area_edges,
    load_grid_checkpoint,
    make_loader,
    predict_prob_maps,
    safe_nanmean,
    safe_nanstd,
    threshold_matches,
)
from scripts.train_mask_boundary_grid_forward_consistency_candidate import load_forward_surrogate  # noqa: E402


LAMBDA_FORWARD = 0.10
OPT_STEPS = 50
OPT_LR = 0.03
MIN_W = 0.20
MIN_H = 0.20
MAX_W = 10.0
MAX_H = 8.0
OBJECTIVES = ['proposal_only', 'proposal_forward']

METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_proposal_shape_refinement_gate_metrics.csv'
VALIDATION_GRID_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_proposal_shape_refinement_validation_grid.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_proposal_shape_refinement_gate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'proposal_shape_refinement_gate'
REFERENCE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_geometry_forward_consistency_candidate_metrics.csv'

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
]


def ensure_outputs():
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = []
    for path in CURRENT_BASELINE_CHECKPOINTS.values():
        if not Path(project_path(path)).exists():
            missing.append(path)
    if not Path(project_path('checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt')).exists():
        missing.append('checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt')
    if not REFERENCE_METRICS_PATH.exists():
        missing.append(str(REFERENCE_METRICS_PATH.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError('Missing required input(s): ' + ', '.join(missing))


def load_reference_rows():
    keep = {
        'current_forward_baseline_single_defect_mean',
        'current_forward_baseline_single_defect_std',
        'geometry_only_reference_mean',
        'geometry_only_reference_std',
    }
    rows = []
    with open(REFERENCE_METRICS_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['candidate'] in keep and row['split'] == 'test':
                rows.append(row)
    return rows


def summarize_samples_with_bz(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
        })
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse']:
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
        'bz_mse',
        'composite',
    ]:
        row[key] = summary[key]
    row['macro_area_composite'] = macro_area_composite
    return row


def summarize_rows(sample_rows, candidate, seed, split, threshold):
    rows = []
    overall = summarize_samples_with_bz(sample_rows)
    area_summaries = {
        group: summarize_samples_with_bz([row for row in sample_rows if row['area_bin'] == group])
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
        rows.append(metric_row(
            candidate,
            seed,
            split,
            threshold,
            'signal_bin',
            group,
            summarize_samples_with_bz([row for row in sample_rows if row['signal_bin'] == group]),
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
            summarize_samples_with_bz([row for row in sample_rows if row['defect_type'] == defect_type]),
            macro_area_composite,
        ))
    return rows


def compute_bz_mses(forward_model, prob_maps, target_signals, device, batch_size=32):
    values = np.empty((len(prob_maps),), dtype=np.float32)
    forward_model.eval()
    with torch.no_grad():
        for start in range(0, len(prob_maps), batch_size):
            end = min(start + batch_size, len(prob_maps))
            prob = torch.from_numpy(prob_maps[start:end]).to(device=device, dtype=torch.float32)
            target = torch.from_numpy(target_signals[start:end]).to(device=device, dtype=torch.float32)
            pred = forward_model(prob.unsqueeze(1))
            values[start:end] = torch.mean((pred - target) ** 2, dim=1).detach().cpu().numpy()
    return values


def predict_current_baseline_mean_prob(data_path, device, forward_model, surrogate_checkpoint):
    target_dataset = SingleDefectDataset(
        data_path,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    seed_probs = []
    for checkpoint_path in CURRENT_BASELINE_CHECKPOINTS.values():
        checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu')
        dataset = SingleDefectDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        model, _ = load_grid_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, out_shape, device)
        prob_maps, _ = predict_prob_maps(model, dataset, None, device)
        seed_probs.append(prob_maps)
    mean_prob = np.mean(np.stack(seed_probs, axis=0), axis=0).astype(np.float32)
    true_masks = (target_dataset.mu_maps < MASK_THRESHOLD_NORM)
    bz_mses = compute_bz_mses(forward_model, mean_prob, target_dataset.signals, device)
    return mean_prob, true_masks, bz_mses, target_dataset


def largest_component(mask):
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) > len(best):
                best = component
    out = np.zeros_like(mask, dtype=bool)
    if best:
        yy, xx = zip(*best)
        out[np.asarray(yy), np.asarray(xx)] = True
    return out


def init_box_from_component(component, prob_map, dataset):
    x_vals = np.asarray(dataset.x, dtype=np.float32)
    y_vals = np.asarray(dataset.y, dtype=np.float32)
    dx = float(abs(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 1.0
    dy = float(abs(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 1.0
    if not np.any(component):
        peak_y, peak_x = np.unravel_index(int(np.argmax(prob_map)), prob_map.shape)
        return np.asarray([x_vals[peak_x], y_vals[peak_y], 1.0, 1.0, 0.0], dtype=np.float32), True

    yy, xx = np.where(component)
    coords = np.stack([x_vals[xx], y_vals[yy]], axis=1).astype(np.float64)
    center = coords.mean(axis=0)
    if len(coords) < 3:
        angle = 0.0
        width = max(MIN_W, dx * 2.0)
        height = max(MIN_H, dy * 2.0)
    else:
        centered = coords - center[None, :]
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        primary = evecs[:, 0]
        secondary = evecs[:, 1]
        proj_x = centered @ primary
        proj_y = centered @ secondary
        width = float(proj_x.max() - proj_x.min() + dx)
        height = float(proj_y.max() - proj_y.min() + dy)
        angle = float(np.arctan2(primary[1], primary[0]))
    width = float(np.clip(width, MIN_W, MAX_W))
    height = float(np.clip(height, MIN_H, MAX_H))
    cx = float(np.clip(center[0], float(x_vals.min()), float(x_vals.max())))
    cy = float(np.clip(center[1], float(y_vals.min()), float(y_vals.max())))
    return np.asarray([cx, cy, width, height, angle], dtype=np.float32), False


def init_boxes_from_coarse(mean_prob, threshold, dataset):
    params = np.empty((len(mean_prob), 5), dtype=np.float32)
    empty_count = 0
    coarse_masks = mean_prob >= threshold
    components = np.zeros_like(coarse_masks, dtype=bool)
    for idx in range(len(mean_prob)):
        component = largest_component(coarse_masks[idx])
        params[idx], was_empty = init_box_from_component(component, mean_prob[idx], dataset)
        components[idx] = component
        empty_count += int(was_empty)
    return params, components, empty_count


def inverse_sigmoid(value):
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def params_to_raw(params, dataset):
    x_min = float(np.min(dataset.x))
    x_max = float(np.max(dataset.x))
    y_min = float(np.min(dataset.y))
    y_max = float(np.max(dataset.y))
    raw = np.empty_like(params, dtype=np.float32)
    raw[:, 0] = inverse_sigmoid((params[:, 0] - x_min) / (x_max - x_min))
    raw[:, 1] = inverse_sigmoid((params[:, 1] - y_min) / (y_max - y_min))
    raw[:, 2] = inverse_sigmoid((params[:, 2] - MIN_W) / (MAX_W - MIN_W))
    raw[:, 3] = inverse_sigmoid((params[:, 3] - MIN_H) / (MAX_H - MIN_H))
    raw[:, 4] = np.arctanh(np.clip(params[:, 4] / np.pi, -0.999, 0.999))
    return raw


def decode_raw(raw, dataset):
    x_min = float(np.min(dataset.x))
    x_max = float(np.max(dataset.x))
    y_min = float(np.min(dataset.y))
    y_max = float(np.max(dataset.y))
    cx = x_min + torch.sigmoid(raw[:, 0]) * (x_max - x_min)
    cy = y_min + torch.sigmoid(raw[:, 1]) * (y_max - y_min)
    width = MIN_W + torch.sigmoid(raw[:, 2]) * (MAX_W - MIN_W)
    height = MIN_H + torch.sigmoid(raw[:, 3]) * (MAX_H - MIN_H)
    angle = np.pi * torch.tanh(raw[:, 4])
    return torch.stack([cx, cy, width, height, angle], dim=1)


def rasterize_params(params, dataset):
    x_coords = torch.as_tensor(dataset.x, device=params.device, dtype=params.dtype)
    y_coords = torch.as_tensor(dataset.y, device=params.device, dtype=params.dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    x_grid = xx[None, :, :]
    y_grid = yy[None, :, :]
    cx = params[:, 0, None, None]
    cy = params[:, 1, None, None]
    width = params[:, 2, None, None]
    height = params[:, 3, None, None]
    angle = params[:, 4, None, None]
    dx = x_grid - cx
    dy = y_grid - cy
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    margin_x = 0.5 * width - torch.abs(local_x)
    margin_y = 0.5 * height - torch.abs(local_y)
    return torch.minimum(margin_x, margin_y) / RASTER_TEMPERATURE


def refine_params(objective, init_params, coarse_prob, dataset, forward_model, device):
    raw = torch.tensor(params_to_raw(init_params, dataset), device=device, dtype=torch.float32, requires_grad=True)
    coarse = torch.tensor(coarse_prob, device=device, dtype=torch.float32)
    target_bz = torch.tensor(dataset.signals, device=device, dtype=torch.float32)
    optimizer = optim.Adam([raw], lr=OPT_LR)
    forward_model.eval()
    for _ in range(OPT_STEPS):
        optimizer.zero_grad()
        params = decode_raw(raw, dataset)
        prob = torch.sigmoid(rasterize_params(params, dataset))
        proposal_loss = F.mse_loss(prob, coarse)
        if objective == 'proposal_forward':
            bz_hat = forward_model(prob.unsqueeze(1))
            forward_loss = F.mse_loss(bz_hat, target_bz)
            loss = proposal_loss + LAMBDA_FORWARD * forward_loss
        else:
            loss = proposal_loss
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        params = decode_raw(raw, dataset)
        prob = torch.sigmoid(rasterize_params(params, dataset))
        bz_hat = forward_model(prob.unsqueeze(1))
        bz_mses = torch.mean((bz_hat - target_bz) ** 2, dim=1)
    return prob.detach().cpu().numpy().astype(np.float32), params.detach().cpu().numpy().astype(np.float32), bz_mses.detach().cpu().numpy().astype(np.float32)


def evaluate_prob_maps(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset):
    area_edges = get_area_edges(dataset)
    low_signal_indices = dataset_low_signal_indices(dataset)
    sample_rows = build_sample_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset, area_edges, low_signal_indices)
    return sample_rows, summarize_rows(sample_rows, candidate, seed, split, threshold)


def run_split(split, data_path, device, forward_model, surrogate_checkpoint):
    mean_prob, true_masks, coarse_bz_mses, dataset = predict_current_baseline_mean_prob(data_path, device, forward_model, surrogate_checkpoint)
    init_params, coarse_components, empty_count = init_boxes_from_coarse(mean_prob, CURRENT_BASELINE_THRESHOLD, dataset)
    coarse_rows, coarse_metrics = evaluate_prob_maps(
        'current_baseline_mean_probability_proposal',
        'mean_prob',
        split,
        CURRENT_BASELINE_THRESHOLD,
        mean_prob,
        true_masks,
        coarse_bz_mses,
        dataset,
    )
    outputs = {
        'dataset': dataset,
        'mean_prob': mean_prob,
        'true_masks': true_masks,
        'init_params': init_params,
        'coarse_components': coarse_components,
        'empty_count': empty_count,
        'coarse_rows': coarse_rows,
        'coarse_metrics': coarse_metrics,
        'objectives': {},
    }
    for objective in OBJECTIVES:
        refined_prob, refined_params, refined_bz_mses = refine_params(objective, init_params, mean_prob, dataset, forward_model, device)
        sample_rows, metric_rows = evaluate_prob_maps(
            f'proposal_refinement_{objective}',
            objective,
            split,
            CURRENT_BASELINE_THRESHOLD,
            refined_prob,
            true_masks,
            refined_bz_mses,
            dataset,
        )
        outputs['objectives'][objective] = {
            'prob': refined_prob,
            'params': refined_params,
            'bz_mses': refined_bz_mses,
            'sample_rows': sample_rows,
            'metric_rows': metric_rows,
        }
    return outputs


def select_objective(val_outputs):
    candidates = []
    for objective in OBJECTIVES:
        row = find_metric_row(val_outputs['objectives'][objective]['metric_rows'], f'proposal_refinement_{objective}', 'overall', 'all')
        candidates.append(row)
    best = max(candidates, key=lambda row: float(row['composite']))
    return str(best['seed']), best


def find_metric_row(rows, candidate, group_type='overall', group='all', split=None):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['group_type'] == group_type
        and row['group'] == group
    ]
    if split is not None:
        selected = [row for row in selected if row['split'] == split]
    return selected[0] if selected else None


def write_metrics(rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'threshold', 'n'] + METRIC_KEYS
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_validation_grid(val_outputs):
    fieldnames = ['objective', 'threshold', 'n', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'bz_mse', 'composite']
    with open(VALIDATION_GRID_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for objective in OBJECTIVES:
            row = find_metric_row(val_outputs['objectives'][objective]['metric_rows'], f'proposal_refinement_{objective}')
            writer.writerow({
                'objective': objective,
                'threshold': row['threshold'],
                'n': row['n'],
                'iou': row['iou'],
                'dice': row['dice'],
                'area_error': row['area_error'],
                'center_error': row['center_error'],
                'pred_area_zero': row['pred_area_zero'],
                'bz_mse': row['bz_mse'],
                'composite': row['composite'],
            })


def draw_box(ax, params, dataset):
    cx, cy, width, height, angle = [float(v) for v in params]
    corners = np.asarray([
        [-0.5 * width, -0.5 * height],
        [0.5 * width, -0.5 * height],
        [0.5 * width, 0.5 * height],
        [-0.5 * width, 0.5 * height],
        [-0.5 * width, -0.5 * height],
    ], dtype=np.float32)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    box_x = cx + cos_a * corners[:, 0] - sin_a * corners[:, 1]
    box_y = cy + sin_a * corners[:, 0] + cos_a * corners[:, 1]
    box_cols = np.interp(box_x, dataset.x, np.arange(len(dataset.x)))
    box_rows = np.interp(box_y, dataset.y, np.arange(len(dataset.y)))
    ax.plot(box_cols, box_rows, color='yellow', linewidth=1.4)


def write_previews(test_outputs, selected_objective):
    data = test_outputs['objectives'][selected_objective]
    rows = data['sample_rows']
    mean_prob = test_outputs['mean_prob']
    refined_prob = data['prob']
    true_masks = test_outputs['true_masks']
    params = data['params']
    dataset = test_outputs['dataset']
    selected = sorted(rows, key=lambda row: float(row['iou']), reverse=True)[:3]
    selected += sorted(rows, key=lambda row: float(row['iou']))[:3]
    selected += sorted([row for row in rows if row['defect_type'] == 'polygon'], key=lambda row: float(row['area_error']), reverse=True)[:3]
    selected += sorted([row for row in rows if row['defect_type'] == 'rotated_rect'], key=lambda row: float(row['area_error']), reverse=True)[:3]
    median_iou = safe_nanmean([float(row['iou']) for row in rows])
    selected += sorted(rows, key=lambda row: abs(float(row['iou']) - median_iou))[:12]
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
        baseline_mask = mean_prob[sample_idx] >= CURRENT_BASELINE_THRESHOLD
        refined_mask = refined_prob[sample_idx] >= CURRENT_BASELINE_THRESHOLD
        true_mask = true_masks[sample_idx]
        fig, axes = plt.subplots(1, 6, figsize=(22, 4), constrained_layout=True)
        axes[0].imshow(true_mask, origin='lower', cmap='gray')
        axes[0].set_title('true mask')
        im1 = axes[1].imshow(mean_prob[sample_idx], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('baseline prob')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(baseline_mask, origin='lower', cmap='gray')
        axes[2].set_title('baseline mask')
        im2 = axes[3].imshow(refined_prob[sample_idx], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title('refined prob')
        fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
        axes[4].imshow(refined_mask, origin='lower', cmap='gray')
        axes[4].set_title('refined mask + box')
        draw_box(axes[4], params[sample_idx], dataset)
        axes[5].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[5].imshow(refined_mask, origin='lower', cmap='Reds', alpha=0.35)
        if np.any(true_mask) and np.any(~true_mask):
            axes[5].contour(true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        if np.any(refined_mask) and np.any(~refined_mask):
            axes[5].contour(refined_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        draw_box(axes[5], params[sample_idx], dataset)
        axes[5].set_title('overlay + box')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={row['original_index']} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} BzMSE={float(row['bz_mse']):.3e}",
            fontsize=9,
        )
        path = PREVIEW_DIR / f'proposal_refinement_rank{rank:02d}_sample{row["original_index"]}_{row["defect_type"]}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def fmt(row, key):
    return f"{float(row[key]):.4f}"


def write_summary(rows, val_outputs, test_outputs, selected_objective, preview_paths, accepted):
    base = find_row(rows, 'current_forward_baseline_single_defect_mean', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    geom = find_row(rows, 'geometry_only_reference_mean', split='test', threshold=GEOMETRY_ONLY_THRESHOLD)
    coarse_test = find_metric_row(test_outputs['coarse_metrics'], 'current_baseline_mean_probability_proposal')
    selected_val = find_metric_row(val_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_refinement_{selected_objective}')
    selected_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_refinement_{selected_objective}')
    proposal_only_val = find_metric_row(val_outputs['objectives']['proposal_only']['metric_rows'], 'proposal_refinement_proposal_only')
    proposal_forward_val = find_metric_row(val_outputs['objectives']['proposal_forward']['metric_rows'], 'proposal_refinement_proposal_forward')
    polygon_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_refinement_{selected_objective}', 'defect_type', 'polygon')
    rotated_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_refinement_{selected_objective}', 'defect_type', 'rotated_rect')

    lines = [
        '# v3_complex proposal shape refinement gate',
        '',
        '## Scope',
        '',
        'This Step 18.6 script is independent. It does not train a neural network, does not update the baseline checkpoint family, and does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, PINN optimization route, or NEXT_STEP.md.',
        '',
        'The single-defect subset keeps `polygon` and `rotated_rect` samples and excludes `multi_defect`.',
        '',
        '| split | polygon | rotated_rect | total | empty coarse masks |',
        '|---|---:|---:|---:|---:|',
    ]
    for split, output in [('val', val_outputs), ('test', test_outputs)]:
        count = split_type_counts(output['dataset'])
        lines.append(f"| {split} | {count.get('polygon', 0)} | {count.get('rotated_rect', 0)} | {sum(count.values())} | {output['empty_count']} |")
    lines.extend([
        '',
        '## Proposal generation',
        '',
        f'* CURRENT_BASELINE checkpoints are the 3-seed forward-consistency family recorded in CURRENT_BASELINE.md.',
        f'* The coarse proposal probability is the pixelwise mean probability from the 3 checkpoints.',
        f'* The coarse mask uses the CURRENT_BASELINE threshold `{CURRENT_BASELINE_THRESHOLD:.2f}`.',
        '* The largest connected component of the coarse mask initializes a rotated box by centroid + PCA width / height / angle.',
        '* Empty coarse masks are initialized from the probability peak and a default small box.',
        f'* Test-time optimization updates only `(cx, cy, w, h, angle)` for `{OPT_STEPS}` Adam steps at lr `{OPT_LR}`.',
        f'* `proposal_forward` uses lambda_forward `{LAMBDA_FORWARD}` with the frozen mask-to-Bz surrogate.',
        '',
        '## Validation objective selection',
        '',
        '| objective | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | score |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        f"| proposal_only | {fmt(proposal_only_val, 'iou')} | {fmt(proposal_only_val, 'dice')} | {fmt(proposal_only_val, 'area_error')} | {fmt(proposal_only_val, 'center_error')} | {float(proposal_only_val['pred_area_zero']):.2f} | {float(proposal_only_val['bz_mse']):.6e} | {fmt(proposal_only_val, 'composite')} |",
        f"| proposal_forward | {fmt(proposal_forward_val, 'iou')} | {fmt(proposal_forward_val, 'dice')} | {fmt(proposal_forward_val, 'area_error')} | {fmt(proposal_forward_val, 'center_error')} | {float(proposal_forward_val['pred_area_zero']):.2f} | {float(proposal_forward_val['bz_mse']):.6e} | {fmt(proposal_forward_val, 'composite')} |",
        '',
        f'Selected objective: `{selected_objective}` using validation IoU + Dice - area_error.',
        '',
        '## Test comparison',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        row_line('CURRENT_BASELINE official 3-seed mean on same subset', base),
        row_line('CURRENT_BASELINE 3-seed mean probability proposal', coarse_test),
        row_line('geometry-only direct Bz -> rotated box reference', geom),
        row_line(f'proposal refinement ({selected_objective})', selected_test),
        '',
        '## Type breakdown for selected refinement',
        '',
        '| defect_type | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        row_line('polygon', polygon_test),
        row_line('rotated_rect', rotated_test),
        '',
        '## Conclusion',
        '',
    ])
    area_ok = selected_test is not None and base is not None and float(selected_test['area_error']) <= float(base['area_error']) + 0.03
    bz_ok = selected_test is not None and base is not None and float(selected_test['bz_mse']) <= float(base['bz_mse']) * 1.10
    iou_ok = selected_test is not None and base is not None and float(selected_test['iou']) >= float(base['iou']) - 0.01
    dice_ok = selected_test is not None and base is not None and float(selected_test['dice']) >= float(base['dice']) - 0.01
    lines.extend([
        f'* area_error controlled vs CURRENT_BASELINE: {area_ok}',
        f'* Bz residual controlled vs CURRENT_BASELINE: {bz_ok}',
        f'* IoU controlled vs CURRENT_BASELINE: {iou_ok}',
        f'* Dice controlled vs CURRENT_BASELINE: {dice_ok}',
        f'* Accepted by gate: {accepted}',
        '',
    ])
    if accepted:
        lines.append('Proposal shape refinement has candidate value: it improves or preserves the main CURRENT_BASELINE metrics while making polygon / rotated_rect outputs more shape-constrained.')
    else:
        lines.append('Proposal shape refinement is not accepted as a formal candidate. In this run, validation selected proposal_forward because it lowers area_error and Bz MSE, but the selected test result drops IoU / Dice against the CURRENT_BASELINE. The preview set shows less round-blob appearance by construction, but polygon samples are still reduced to rectangles and offset failures remain. This is consistent with a low-dimensional rectangle refinement explaining the surrogate Bz better without preserving enough mask overlap, so this branch should stop and not continue into refinement v2, lambda search, polygon vertices, or multi-component variants.')
    lines.append(f'Preview files: {len(preview_paths)} written to `{PREVIEW_DIR.relative_to(ROOT)}`.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = load_forward_surrogate(device)

    val_outputs = run_split('val', VAL_DATA, device, forward_model, surrogate_checkpoint)
    selected_objective, selected_val = select_objective(val_outputs)
    test_outputs = run_split('test', TEST_DATA, device, forward_model, surrogate_checkpoint)

    rows = load_reference_rows()
    rows.extend(val_outputs['coarse_metrics'])
    rows.extend(test_outputs['coarse_metrics'])
    for objective in OBJECTIVES:
        rows.extend(val_outputs['objectives'][objective]['metric_rows'])
    rows.extend(test_outputs['objectives'][selected_objective]['metric_rows'])

    selected_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_refinement_{selected_objective}')
    base = find_row(rows, 'current_forward_baseline_single_defect_mean', split='test', threshold=CURRENT_BASELINE_THRESHOLD)
    accepted = bool(
        selected_test is not None
        and base is not None
        and float(selected_test['iou']) >= float(base['iou']) - 0.01
        and float(selected_test['dice']) >= float(base['dice']) - 0.01
        and float(selected_test['area_error']) <= float(base['area_error']) + 0.03
        and float(selected_test['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6
        and float(selected_test['bz_mse']) <= float(base['bz_mse']) * 1.10
    )

    write_metrics(rows)
    write_validation_grid(val_outputs)
    preview_paths = write_previews(test_outputs, selected_objective)
    write_summary(rows, val_outputs, test_outputs, selected_objective, preview_paths, accepted)

    print(f'completed=True')
    print(f'val_empty_masks={val_outputs["empty_count"]}, test_empty_masks={test_outputs["empty_count"]}')
    print(f'selected_objective={selected_objective}')
    print(
        f"selected_val: IoU={float(selected_val['iou']):.6f}, Dice={float(selected_val['dice']):.6f}, "
        f"area_error={float(selected_val['area_error']):.6f}, BzMSE={float(selected_val['bz_mse']):.6e}"
    )
    print(
        f"selected_test: IoU={float(selected_test['iou']):.6f}, Dice={float(selected_test['dice']):.6f}, "
        f"area_error={float(selected_test['area_error']):.6f}, BzMSE={float(selected_test['bz_mse']):.6e}"
    )
    print(f'accepted={accepted}')
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote validation grid: {VALIDATION_GRID_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    print(f'wrote previews: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
