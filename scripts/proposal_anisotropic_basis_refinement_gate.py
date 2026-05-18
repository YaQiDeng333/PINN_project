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

from scripts.proposal_shape_refinement_gate import (  # noqa: E402
    compute_bz_mses,
    largest_component,
    predict_current_baseline_mean_prob,
)
from scripts.train_anisotropic_basis_forward_candidate import (  # noqa: E402
    ACTIVE_THRESHOLD,
    MAX_SCALE_X,
    MAX_SCALE_Y,
    MIN_SCALE,
    RASTER_TEMPERATURE,
    active_component_stats,
    initial_basis_params,
    render_basis_logits,
)
from scripts.train_geometry_boundary_candidate import (  # noqa: E402
    MASK_THRESHOLD_NORM,
    TEST_DATA,
    VAL_DATA,
    SingleDefectDataset,
    split_type_counts,
)
from scripts.train_geometry_forward_consistency_candidate import (  # noqa: E402
    CURRENT_BASELINE_THRESHOLD,
    build_sample_rows,
    dataset_low_signal_indices,
    evaluate_current_baseline,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    get_area_edges,
    safe_nanmean,
    safe_nanstd,
)
from scripts.train_mask_boundary_grid_forward_consistency_candidate import load_forward_surrogate  # noqa: E402


K_BASIS = 4
LAMBDA_FORWARD = 0.10
OPT_STEPS = 50
OPT_LR = 0.03
OBJECTIVES = ['proposal_only', 'proposal_forward']

SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_proposal_anisotropic_basis_refinement_gate_summary.txt'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_proposal_anisotropic_basis_refinement_gate_metrics.csv'
VALIDATION_GRID_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_proposal_anisotropic_basis_refinement_validation_grid.csv'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'proposal_anisotropic_basis_refinement_gate'
DIRECT_BASIS_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_anisotropic_basis_forward_candidate_metrics.csv'
ORACLE_BASIS_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_anisotropic_basis_oracle_metrics.csv'
ROTATED_BOX_REFINEMENT_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_proposal_shape_refinement_gate_metrics.csv'

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
    'active_component_count',
    'collapsed_component_count',
    'out_of_image_center_count',
    'avg_scale_x',
    'avg_scale_y',
]


def ensure_outputs():
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = []
    surrogate_path = ROOT / 'checkpoints' / 'mask_to_bz_forward_surrogate' / 'best_mask_to_bz_forward_surrogate.pt'
    if not surrogate_path.exists():
        missing.append(str(surrogate_path.relative_to(ROOT)))
    for path in [DIRECT_BASIS_METRICS, ORACLE_BASIS_METRICS, ROTATED_BOX_REFINEMENT_METRICS]:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError('Missing required input(s): ' + ', '.join(missing))


def coord_grids(dataset, device):
    x_coords = torch.from_numpy(dataset.x.astype(np.float32)).to(device)
    y_coords = torch.from_numpy(dataset.y.astype(np.float32)).to(device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    return xx, yy


def init_basis_from_coarse(mean_prob, threshold, dataset):
    coarse_masks = mean_prob >= threshold
    params = np.empty((len(mean_prob), K_BASIS, 6), dtype=np.float32)
    empty_count = 0
    components = np.zeros_like(coarse_masks, dtype=bool)
    for idx in range(len(mean_prob)):
        component = largest_component(coarse_masks[idx])
        if not np.any(component):
            empty_count += 1
            peak_y, peak_x = np.unravel_index(int(np.argmax(mean_prob[idx])), mean_prob[idx].shape)
            cx = float(dataset.x[peak_x])
            cy = float(dataset.y[peak_y])
            sample_params = np.zeros((K_BASIS, 6), dtype=np.float32)
            offsets = np.linspace(-0.45, 0.45, K_BASIS, dtype=np.float32)
            for k, offset in enumerate(offsets):
                sample_params[k] = np.asarray([cx + float(offset), cy, 0.45, 0.45, 0.0, 2.0], dtype=np.float32)
        else:
            sample_params = initial_basis_params(component, dataset, K_BASIS)
            sample_params[:, 5] = 2.0
        params[idx] = sample_params
        components[idx] = component
    return params, components, empty_count


def inverse_sigmoid(value):
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def params_to_raw(params, dataset):
    x_min, x_max = float(np.min(dataset.x)), float(np.max(dataset.x))
    y_min, y_max = float(np.min(dataset.y)), float(np.max(dataset.y))
    raw = np.empty_like(params, dtype=np.float32)
    raw[..., 0] = inverse_sigmoid((params[..., 0] - x_min) / (x_max - x_min))
    raw[..., 1] = inverse_sigmoid((params[..., 1] - y_min) / (y_max - y_min))
    raw[..., 2] = np.log(np.clip(params[..., 2] - MIN_SCALE, 1e-3, None))
    raw[..., 3] = np.log(np.clip(params[..., 3] - MIN_SCALE, 1e-3, None))
    raw[..., 4] = np.arctanh(np.clip(params[..., 4] / np.pi, -0.999, 0.999))
    raw[..., 5] = params[..., 5]
    return raw


def decode_raw(raw, dataset):
    x_min, x_max = float(np.min(dataset.x)), float(np.max(dataset.x))
    y_min, y_max = float(np.min(dataset.y)), float(np.max(dataset.y))
    cx = x_min + torch.sigmoid(raw[..., 0]) * (x_max - x_min)
    cy = y_min + torch.sigmoid(raw[..., 1]) * (y_max - y_min)
    sx = MIN_SCALE + F.softplus(raw[..., 2]).clamp(max=MAX_SCALE_X - MIN_SCALE)
    sy = MIN_SCALE + F.softplus(raw[..., 3]).clamp(max=MAX_SCALE_Y - MIN_SCALE)
    angle = np.pi * torch.tanh(raw[..., 4])
    amp = raw[..., 5]
    return torch.stack([cx, cy, sx, sy, angle, amp], dim=-1)


def refine_params(objective, init_params, coarse_prob, dataset, forward_model, device):
    raw = torch.tensor(params_to_raw(init_params, dataset), device=device, dtype=torch.float32, requires_grad=True)
    coarse = torch.tensor(coarse_prob, device=device, dtype=torch.float32)
    target_bz = torch.tensor(dataset.signals, device=device, dtype=torch.float32)
    coords = coord_grids(dataset, device)
    optimizer = optim.Adam([raw], lr=OPT_LR)
    forward_model.eval()
    for _ in range(OPT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        params = decode_raw(raw, dataset)
        logits = render_basis_logits(params, coords[0], coords[1])
        prob = torch.sigmoid(logits).reshape_as(coarse)
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
        logits = render_basis_logits(params, coords[0], coords[1])
        prob = torch.sigmoid(logits).reshape_as(coarse)
        bz_hat = forward_model(prob.unsqueeze(1))
        bz_mses = torch.mean((bz_hat - target_bz) ** 2, dim=1)
    return (
        prob.detach().cpu().numpy().astype(np.float32),
        params.detach().cpu().numpy().astype(np.float32),
        bz_mses.detach().cpu().numpy().astype(np.float32),
    )


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse', 'active_component_count', 'collapsed_component_count', 'out_of_image_center_count', 'avg_scale_x', 'avg_scale_y']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
        })
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse', 'active_component_count', 'collapsed_component_count', 'out_of_image_center_count', 'avg_scale_x', 'avg_scale_y']:
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


def build_basis_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, params, dataset):
    area_edges = get_area_edges(dataset)
    low_signal_indices = dataset_low_signal_indices(dataset)
    diag = active_component_stats(params, dataset)
    rows = build_sample_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset, area_edges, low_signal_indices)
    for idx, row in enumerate(rows):
        row['active_component_count'] = float(diag['active'][idx])
        row['collapsed_component_count'] = float(diag['collapsed'][idx])
        row['out_of_image_center_count'] = float(diag['out'][idx])
        row['avg_scale_x'] = float(diag['scale_x'][idx])
        row['avg_scale_y'] = float(diag['scale_y'][idx])
    return rows


def evaluate_prob_maps(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, params, dataset):
    sample_rows = build_basis_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, params, dataset)
    return sample_rows, summarize_candidate(sample_rows, candidate, seed, split, threshold)


def zero_component_params(n_samples):
    params = np.zeros((n_samples, K_BASIS, 6), dtype=np.float32)
    params[..., 2] = np.nan
    params[..., 3] = np.nan
    params[..., 5] = -20.0
    return params


def run_split(split, data_path, device, forward_model, surrogate_checkpoint):
    mean_prob, true_masks, coarse_bz_mses, dataset = predict_current_baseline_mean_prob(
        data_path,
        device,
        forward_model,
        surrogate_checkpoint,
    )
    init_params, coarse_components, empty_count = init_basis_from_coarse(mean_prob, CURRENT_BASELINE_THRESHOLD, dataset)
    coarse_rows, coarse_metrics = evaluate_prob_maps(
        'current_baseline_mean_probability_proposal',
        'mean_prob',
        split,
        CURRENT_BASELINE_THRESHOLD,
        mean_prob,
        true_masks,
        coarse_bz_mses,
        zero_component_params(len(dataset)),
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
        refined_prob, refined_params, refined_bz_mses = refine_params(
            objective,
            init_params,
            mean_prob,
            dataset,
            forward_model,
            device,
        )
        sample_rows, metric_rows = evaluate_prob_maps(
            f'proposal_basis_refinement_{objective}',
            objective,
            split,
            CURRENT_BASELINE_THRESHOLD,
            refined_prob,
            true_masks,
            refined_bz_mses,
            refined_params,
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


def select_objective(val_outputs):
    candidates = []
    for objective in OBJECTIVES:
        row = find_metric_row(val_outputs['objectives'][objective]['metric_rows'], f'proposal_basis_refinement_{objective}')
        candidates.append((objective, row))
    return max(candidates, key=lambda item: float(item[1]['composite']))


def load_reference_rows():
    rows = []
    for path, candidates in [
        (ORACLE_BASIS_METRICS, {'anisotropic_basis_oracle_k4'}),
        (DIRECT_BASIS_METRICS, {'anisotropic_basis_forward_screening'}),
        (ROTATED_BOX_REFINEMENT_METRICS, {'proposal_refinement_proposal_forward'}),
    ]:
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['split'] == 'test' and row['candidate'] in candidates:
                    for key in METRIC_KEYS:
                        row.setdefault(key, '')
                    rows.append(row)
    return rows


def write_csv(path, rows):
    fieldnames = ['candidate', 'seed', 'split', 'group_type', 'group', 'threshold', 'n'] + METRIC_KEYS
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_validation_grid(val_outputs):
    fieldnames = ['objective', 'threshold', 'n', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'bz_mse', 'composite', 'active_component_count', 'collapsed_component_count']
    with open(VALIDATION_GRID_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for objective in OBJECTIVES:
            row = find_metric_row(val_outputs['objectives'][objective]['metric_rows'], f'proposal_basis_refinement_{objective}')
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
                'active_component_count': row['active_component_count'],
                'collapsed_component_count': row['collapsed_component_count'],
            })


def draw_components(ax, params, dataset):
    theta = np.linspace(0, 2 * np.pi, 80)
    for comp in params:
        if 1.0 / (1.0 + np.exp(-comp[5])) < ACTIVE_THRESHOLD:
            continue
        cx, cy, sx, sy, angle = comp[:5]
        x = sx * np.cos(theta)
        y = sy * np.sin(theta)
        ca = np.cos(angle)
        sa = np.sin(angle)
        world_x = cx + ca * x - sa * y
        world_y = cy + sa * x + ca * y
        px = np.interp(world_x, dataset.x, np.arange(len(dataset.x)))
        py = np.interp(world_y, dataset.y, np.arange(len(dataset.y)))
        ax.plot(px, py, color='cyan', linewidth=0.8)
        ax.scatter(np.interp(cx, dataset.x, np.arange(len(dataset.x))), np.interp(cy, dataset.y, np.arange(len(dataset.y))), s=8, color='cyan')


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
        draw_components(axes[3], params[sample_idx], dataset)
        axes[3].set_title('refined basis prob')
        fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
        axes[4].imshow(refined_mask, origin='lower', cmap='gray')
        draw_components(axes[4], params[sample_idx], dataset)
        axes[4].set_title('refined mask + basis')
        axes[5].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[5].imshow(refined_mask, origin='lower', cmap='Reds', alpha=0.35)
        if np.any(true_mask) and np.any(~true_mask):
            axes[5].contour(true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        if np.any(refined_mask) and np.any(~refined_mask):
            axes[5].contour(refined_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        draw_components(axes[5], params[sample_idx], dataset)
        axes[5].set_title('overlay + basis')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={row['original_index']} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} BzMSE={float(row['bz_mse']):.3e} "
            f"active={float(row['active_component_count']):.1f}",
            fontsize=9,
        )
        path = PREVIEW_DIR / f'proposal_basis_refinement_rank{rank:02d}_sample{row["original_index"]}_{row["defect_type"]}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def fmt(row, key):
    if row is None or row.get(key, '') == '':
        return 'NA'
    return f"{float(row[key]):.4f}"


def row_line(label, row):
    return (
        f"| {label} | {fmt(row, 'threshold')} | {fmt(row, 'n')} | {fmt(row, 'iou')} | "
        f"{fmt(row, 'dice')} | {fmt(row, 'area_error')} | {fmt(row, 'center_error')} | "
        f"{fmt(row, 'pred_area_zero')} | {fmt(row, 'bz_mse')} | {fmt(row, 'active_component_count')} | "
        f"{fmt(row, 'collapsed_component_count')} | {fmt(row, 'avg_scale_x')} | {fmt(row, 'avg_scale_y')} |"
    )


def evaluate_gate(base, selected, rotated):
    if base is None or selected is None:
        return False, {'reason': 'missing comparison rows'}
    checks = {
        'iou_not_obviously_below_current': float(selected['iou']) >= float(base['iou']) - 0.01,
        'dice_not_obviously_below_current': float(selected['dice']) >= float(base['dice']) - 0.01,
        'area_error_not_worse': float(selected['area_error']) <= float(base['area_error']) + 0.03,
        'pred_area_zero_not_up': float(selected['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'bz_mse_not_worse': float(selected['bz_mse']) <= float(base['bz_mse']) * 1.10,
        'components_not_collapsed': float(selected['collapsed_component_count']) <= 0.5 and float(selected['active_component_count']) >= 1.0,
    }
    if rotated is not None:
        checks['better_than_rotated_box_proposal_iou'] = float(selected['iou']) >= float(rotated['iou']) - 1e-6
        checks['better_than_rotated_box_proposal_dice'] = float(selected['dice']) >= float(rotated['dice']) - 1e-6
    return bool(all(checks.values())), checks


def write_summary(rows, val_outputs, test_outputs, selected_objective, preview_paths, accepted, gate_checks):
    base = find_metric_row(rows, 'current_forward_baseline_single_defect_mean', split='test')
    coarse = find_metric_row(test_outputs['coarse_metrics'], 'current_baseline_mean_probability_proposal')
    rotated = find_metric_row(rows, 'proposal_refinement_proposal_forward', split='test')
    basis_oracle = find_metric_row(rows, 'anisotropic_basis_oracle_k4', split='test')
    direct_basis = find_metric_row(rows, 'anisotropic_basis_forward_screening', split='test')
    selected_val = find_metric_row(val_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_basis_refinement_{selected_objective}')
    selected_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_basis_refinement_{selected_objective}')
    proposal_only_val = find_metric_row(val_outputs['objectives']['proposal_only']['metric_rows'], 'proposal_basis_refinement_proposal_only')
    proposal_forward_val = find_metric_row(val_outputs['objectives']['proposal_forward']['metric_rows'], 'proposal_basis_refinement_proposal_forward')
    polygon_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_basis_refinement_{selected_objective}', 'defect_type', 'polygon')
    rotated_test = find_metric_row(test_outputs['objectives'][selected_objective]['metric_rows'], f'proposal_basis_refinement_{selected_objective}', 'defect_type', 'rotated_rect')

    lines = [
        '# v3_complex proposal anisotropic basis refinement gate',
        '',
        '## Scope',
        '',
        'This Step 19.3 script is independent. It does not train a neural network, update baseline checkpoints, or modify the main train/evaluate/data files.',
        '',
        'The single-defect subset keeps polygon / rotated_rect samples and excludes multi_defect.',
        '',
        '| split | polygon | rotated_rect | total | empty coarse masks |',
        '|---|---:|---:|---:|---:|',
    ]
    for split, output in [('val', val_outputs), ('test', test_outputs)]:
        count = split_type_counts(output['dataset'])
        lines.append(f"| {split} | {count.get('polygon', 0)} | {count.get('rotated_rect', 0)} | {sum(count.values())} | {output['empty_count']} |")
    lines.extend([
        '',
        '## Proposal and initialization',
        '',
        '* CURRENT_BASELINE coarse proposal is the 3-seed mean probability from the forward-consistency baseline at threshold 0.80.',
        '* K=4 basis components are initialized from the largest connected coarse mask component using PCA-axis partitioning.',
        '* Empty coarse masks use the probability peak plus a small default component layout.',
        f'* Refinement uses Adam for {OPT_STEPS} steps at lr={OPT_LR}; temperature and K are fixed.',
        f'* proposal_forward objective uses lambda_forward={LAMBDA_FORWARD}.',
        '',
        '## Validation objective selection',
        '',
        '| objective | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | active | collapsed | avg_scale_x | avg_scale_y |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        row_line('proposal_only val', proposal_only_val),
        row_line('proposal_forward val', proposal_forward_val),
        '',
        f'Validation selected objective: `{selected_objective}`',
        '',
        '## Test comparison',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | active | collapsed | avg_scale_x | avg_scale_y |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        row_line('CURRENT_BASELINE same subset', base),
        row_line('baseline mean-prob coarse proposal', coarse),
        row_line('K=4 anisotropic basis oracle', basis_oracle),
        row_line('direct Bz -> basis seed=42', direct_basis),
        row_line('rotated-box proposal refinement', rotated),
        row_line('proposal anisotropic basis refinement', selected_test),
        '',
        '## Defect-type selected test metrics',
        '',
        '| group | IoU | Dice | area_error | Bz MSE | active | collapsed |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| polygon | {fmt(polygon_test, 'iou')} | {fmt(polygon_test, 'dice')} | {fmt(polygon_test, 'area_error')} | {fmt(polygon_test, 'bz_mse')} | {fmt(polygon_test, 'active_component_count')} | {fmt(polygon_test, 'collapsed_component_count')} |",
        f"| rotated_rect | {fmt(rotated_test, 'iou')} | {fmt(rotated_test, 'dice')} | {fmt(rotated_test, 'area_error')} | {fmt(rotated_test, 'bz_mse')} | {fmt(rotated_test, 'active_component_count')} | {fmt(rotated_test, 'collapsed_component_count')} |",
        '',
        '## Gate checks',
        '',
    ])
    for key, value in gate_checks.items():
        lines.append(f'* {key}: {value}')
    lines.extend([
        '',
        f'Accepted by gate: {accepted}',
        '',
        '## Conclusion',
        '',
    ])
    if accepted:
        lines.append('Proposal anisotropic basis refinement satisfies the numeric metric gate and has value for follow-up review.')
        lines.append('Manual preview inspection is still required before treating it as a formal candidate: the main gain may be Bz residual and area control rather than true polygon / rotated_rect straight-edge fitting.')
    else:
        lines.append('Proposal anisotropic basis refinement does not satisfy the gate. Stop K / temperature / steps / lambda / basis refinement v2 / multi-component matching from this result.')
    lines.append(f'Preview files: {len(preview_paths)} written to `{PREVIEW_DIR.relative_to(ROOT)}`.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = load_forward_surrogate(device)

    val_outputs = run_split('val', VAL_DATA, device, forward_model, surrogate_checkpoint)
    selected_objective, selected_val_row = select_objective(val_outputs)
    test_outputs = run_split('test', TEST_DATA, device, forward_model, surrogate_checkpoint)
    selected_test_rows = test_outputs['objectives'][selected_objective]['metric_rows']
    reference_rows = load_reference_rows()
    current_rows = evaluate_current_baseline('test', TEST_DATA, device, forward_model)
    all_rows = []
    all_rows.extend(current_rows)
    all_rows.extend(reference_rows)
    all_rows.extend(val_outputs['coarse_metrics'])
    all_rows.extend(test_outputs['coarse_metrics'])
    for output in [val_outputs, test_outputs]:
        for objective in OBJECTIVES:
            all_rows.extend(output['objectives'][objective]['metric_rows'])
    selected_test = find_metric_row(selected_test_rows, f'proposal_basis_refinement_{selected_objective}')
    base = find_metric_row(current_rows, 'current_forward_baseline_single_defect_mean', split='test')
    rotated = find_metric_row(reference_rows, 'proposal_refinement_proposal_forward', split='test')
    accepted, gate_checks = evaluate_gate(base, selected_test, rotated)
    preview_paths = write_previews(test_outputs, selected_objective)

    write_csv(METRICS_PATH, all_rows)
    write_validation_grid(val_outputs)
    write_summary(all_rows, val_outputs, test_outputs, selected_objective, preview_paths, accepted, gate_checks)
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote validation grid: {VALIDATION_GRID_PATH}')
    print(f'Preview count: {len(preview_paths)}')


if __name__ == '__main__':
    main()
