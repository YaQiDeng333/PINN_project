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

from train_pinn import MFLDataset, build_coord_grid, project_path, signal_shape_info  # noqa: E402
from scripts.train_geometry_forward_consistency_candidate import CURRENT_BASELINE_CHECKPOINTS  # noqa: E402
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    MASK_THRESHOLD_NORM,
    area_bin,
    compute_mask_metrics,
    get_area_edges,
    load_grid_checkpoint,
    make_loader,
    predict_prob_maps,
    safe_nanmean,
)
from scripts.train_mask_boundary_grid_forward_consistency_candidate import (  # noqa: E402
    dataset_low_signal_indices,
    load_forward_surrogate,
)


VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

CURRENT_BASELINE_THRESHOLD = 0.80
OPT_STEPS = 50
OPT_LR = 0.03
ALPHA_PROPOSAL = 1.0
ALPHA_FORWARD = 0.10
ALPHA_TV = 0.001
REFINE_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPS = 1e-4
DRIFT_THRESHOLD = 0.05

OBJECTIVES = ['proposal_only', 'proposal_forward']

SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_logit_refinement_gate_summary.txt'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_logit_refinement_gate_metrics.csv'
VALIDATION_GRID_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_logit_refinement_validation_grid.csv'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'mask_logit_refinement_gate'


def ensure_outputs():
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = []
    for path in CURRENT_BASELINE_CHECKPOINTS.values():
        if not Path(project_path(path)).exists():
            missing.append(path)
    surrogate_path = ROOT / 'checkpoints' / 'mask_to_bz_forward_surrogate' / 'best_mask_to_bz_forward_surrogate.pt'
    if not surrogate_path.exists():
        missing.append(str(surrogate_path.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError('Missing required checkpoint(s): ' + ', '.join(missing))


def compute_bz_mses(forward_model, prob_maps, target_signals, device, batch_size=EVAL_BATCH_SIZE):
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
    target_dataset = MFLDataset(
        data_path,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    seed_probs = []
    for checkpoint_path in CURRENT_BASELINE_CHECKPOINTS.values():
        checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu')
        dataset = MFLDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        model, _ = load_grid_checkpoint(
            Path(project_path(checkpoint_path)),
            signal_length,
            signal_channels,
            out_shape,
            device,
        )
        coords = build_coord_grid(dataset.x, dataset.y).to(device)
        prob_maps, _ = predict_prob_maps(model, dataset, coords, device)
        seed_probs.append(prob_maps)
    mean_prob = np.mean(np.stack(seed_probs, axis=0), axis=0).astype(np.float32)
    true_masks = target_dataset.mu_maps < MASK_THRESHOLD_NORM
    bz_mses = compute_bz_mses(forward_model, mean_prob, target_dataset.signals, device)
    return mean_prob, true_masks, bz_mses, target_dataset


def total_variation(prob):
    dy = torch.mean(torch.abs(prob[:, 1:, :] - prob[:, :-1, :]))
    dx = torch.mean(torch.abs(prob[:, :, 1:] - prob[:, :, :-1]))
    return dx + dy


def refine_probs(objective, coarse_prob, target_signals, forward_model, device):
    refined = np.empty_like(coarse_prob, dtype=np.float32)
    losses = []
    for start in range(0, len(coarse_prob), REFINE_BATCH_SIZE):
        end = min(start + REFINE_BATCH_SIZE, len(coarse_prob))
        coarse = torch.from_numpy(coarse_prob[start:end]).to(device=device, dtype=torch.float32)
        target = torch.from_numpy(target_signals[start:end]).to(device=device, dtype=torch.float32)
        logits = torch.logit(torch.clamp(coarse, EPS, 1.0 - EPS)).detach().clone()
        logits.requires_grad_(True)
        optimizer = optim.Adam([logits], lr=OPT_LR)
        final_loss = None
        for _ in range(OPT_STEPS):
            optimizer.zero_grad(set_to_none=True)
            prob = torch.sigmoid(logits)
            proposal_loss = F.mse_loss(prob, coarse)
            tv_loss = total_variation(prob)
            loss = ALPHA_PROPOSAL * proposal_loss + ALPHA_TV * tv_loss
            if objective == 'proposal_forward':
                bz_hat = forward_model(prob.unsqueeze(1))
                forward_loss = F.mse_loss(bz_hat, target)
                loss = loss + ALPHA_FORWARD * forward_loss
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        refined[start:end] = torch.sigmoid(logits.detach()).cpu().numpy().astype(np.float32)
        losses.append(final_loss)
    return refined, float(np.mean(losses))


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in [
            'iou',
            'dice',
            'area_error',
            'center_error',
            'bz_mse',
            'mean_abs_change',
            'large_probability_drift',
            'bz_improves_mask_worsens',
        ]:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
        })
        return summary
    for key in ['iou', 'dice', 'area_error', 'center_error', 'bz_mse', 'mean_abs_change']:
        summary[key] = safe_nanmean([float(row[key]) for row in rows])
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['large_probability_drift'] = int(sum(int(row['large_probability_drift']) for row in rows))
    summary['bz_improves_mask_worsens'] = int(sum(int(row['bz_improves_mask_worsens']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    return summary


def metric_row(candidate, split, threshold, group_type, group, summary):
    row = {
        'candidate': candidate,
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
        'mean_abs_change',
        'large_probability_drift',
        'bz_improves_mask_worsens',
    ]:
        row[key] = summary[key]
    return row


def summarize_candidate(sample_rows, candidate, split, threshold):
    rows = []
    rows.append(metric_row(candidate, split, threshold, 'overall', 'all', summarize_samples(sample_rows)))
    for group in ['small', 'medium', 'large']:
        rows.append(metric_row(
            candidate,
            split,
            threshold,
            'area_bin',
            group,
            summarize_samples([row for row in sample_rows if row['area_bin'] == group]),
        ))
    for group in ['low_signal', 'non_low_signal']:
        rows.append(metric_row(
            candidate,
            split,
            threshold,
            'signal_bin',
            group,
            summarize_samples([row for row in sample_rows if row['signal_bin'] == group]),
        ))
    for defect_type in sorted({row['defect_type'] for row in sample_rows}):
        rows.append(metric_row(
            candidate,
            split,
            threshold,
            'defect_type',
            defect_type,
            summarize_samples([row for row in sample_rows if row['defect_type'] == defect_type]),
        ))
    return rows


def build_sample_rows(
    candidate,
    split,
    threshold,
    prob_maps,
    true_masks,
    bz_mses,
    dataset,
    area_edges,
    low_signal_indices,
    coarse_prob=None,
    baseline_sample_rows=None,
    baseline_bz_mses=None,
):
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    baseline_by_index = {}
    if baseline_sample_rows is not None:
        baseline_by_index = {int(row['sample_index']): row for row in baseline_sample_rows}
    rows = []
    for sample_idx in range(len(dataset)):
        pred_mask = prob_maps[sample_idx] >= threshold
        metrics = compute_mask_metrics(pred_mask, true_masks[sample_idx], x_grid, y_grid)
        if coarse_prob is None:
            mean_abs_change = 0.0
        else:
            mean_abs_change = float(np.mean(np.abs(prob_maps[sample_idx] - coarse_prob[sample_idx])))
        large_drift = int(mean_abs_change > DRIFT_THRESHOLD)
        bz_improves_mask_worsens = 0
        if baseline_by_index and baseline_bz_mses is not None:
            base = baseline_by_index[sample_idx]
            bz_better = float(bz_mses[sample_idx]) < float(baseline_bz_mses[sample_idx])
            mask_worse = (
                float(metrics['iou']) < float(base['iou']) - 1e-9
                or float(metrics['dice']) < float(base['dice']) - 1e-9
            )
            bz_improves_mask_worsens = int(bz_better and mask_worse)
        metrics.update({
            'candidate': candidate,
            'split': split,
            'threshold': threshold,
            'sample_index': sample_idx,
            'defect_type': str(dataset.defect_types[sample_idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if sample_idx in low_signal_indices else 'non_low_signal',
            'bz_mse': float(bz_mses[sample_idx]),
            'mean_abs_change': mean_abs_change,
            'large_probability_drift': large_drift,
            'bz_improves_mask_worsens': bz_improves_mask_worsens,
        })
        rows.append(metrics)
    return rows


def find_row(rows, candidate, split='test', group_type='overall', group='all'):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['split'] == split
        and row['group_type'] == group_type
        and row['group'] == group
    ]
    return selected[0] if selected else None


def run_split(split, data_path, device, forward_model, surrogate_checkpoint):
    coarse_prob, true_masks, baseline_bz_mses, dataset = predict_current_baseline_mean_prob(
        data_path,
        device,
        forward_model,
        surrogate_checkpoint,
    )
    area_edges = get_area_edges(dataset)
    low_signal_indices = dataset_low_signal_indices(dataset)
    baseline_sample_rows = build_sample_rows(
        'current_baseline',
        split,
        CURRENT_BASELINE_THRESHOLD,
        coarse_prob,
        true_masks,
        baseline_bz_mses,
        dataset,
        area_edges,
        low_signal_indices,
    )
    baseline_metric_rows = summarize_candidate(
        baseline_sample_rows,
        'current_baseline',
        split,
        CURRENT_BASELINE_THRESHOLD,
    )

    objective_outputs = {}
    for objective in OBJECTIVES:
        refined_prob, opt_loss = refine_probs(objective, coarse_prob, dataset.signals, forward_model, device)
        refined_bz_mses = compute_bz_mses(forward_model, refined_prob, dataset.signals, device)
        sample_rows = build_sample_rows(
            objective,
            split,
            CURRENT_BASELINE_THRESHOLD,
            refined_prob,
            true_masks,
            refined_bz_mses,
            dataset,
            area_edges,
            low_signal_indices,
            coarse_prob=coarse_prob,
            baseline_sample_rows=baseline_sample_rows,
            baseline_bz_mses=baseline_bz_mses,
        )
        objective_outputs[objective] = {
            'prob': refined_prob,
            'bz_mses': refined_bz_mses,
            'sample_rows': sample_rows,
            'metric_rows': summarize_candidate(sample_rows, objective, split, CURRENT_BASELINE_THRESHOLD),
            'opt_loss': opt_loss,
        }

    return {
        'split': split,
        'dataset': dataset,
        'true_masks': true_masks,
        'coarse_prob': coarse_prob,
        'baseline_bz_mses': baseline_bz_mses,
        'baseline_sample_rows': baseline_sample_rows,
        'baseline_metric_rows': baseline_metric_rows,
        'objectives': objective_outputs,
    }


def select_objective(val_outputs):
    rows = list(val_outputs['baseline_metric_rows'])
    for objective in OBJECTIVES:
        rows.extend(val_outputs['objectives'][objective]['metric_rows'])
    candidates = [
        find_row(rows, 'current_baseline', split='val'),
        find_row(rows, 'proposal_only', split='val'),
        find_row(rows, 'proposal_forward', split='val'),
    ]
    candidates = [row for row in candidates if row is not None]
    selected = max(candidates, key=lambda row: float(row['composite']))
    return str(selected['candidate']), selected


def write_metrics(rows):
    fieldnames = [
        'candidate',
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
        'bz_mse',
        'composite',
        'mean_abs_change',
        'large_probability_drift',
        'bz_improves_mask_worsens',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_validation_grid(val_outputs):
    rows = []
    for candidate, metric_rows in [
        ('current_baseline', val_outputs['baseline_metric_rows']),
        ('proposal_only', val_outputs['objectives']['proposal_only']['metric_rows']),
        ('proposal_forward', val_outputs['objectives']['proposal_forward']['metric_rows']),
    ]:
        overall = find_row(metric_rows, candidate, split='val')
        if overall is not None:
            rows.append(overall)
    fieldnames = [
        'candidate',
        'threshold',
        'n',
        'iou',
        'dice',
        'area_error',
        'center_error',
        'pred_area_zero',
        'bz_mse',
        'composite',
        'mean_abs_change',
        'large_probability_drift',
        'bz_improves_mask_worsens',
    ]
    with open(VALIDATION_GRID_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def preview_indices(selected_rows):
    rows = list(selected_rows)
    picks = []

    def add(label, candidates):
        for row in candidates:
            idx = int(row['sample_index'])
            if idx not in [item[0] for item in picks]:
                picks.append((idx, label))
                return

    sorted_by_iou = sorted(rows, key=lambda row: float(row['iou']))
    add('failure_low_iou', sorted_by_iou)
    add('failure_high_area_error', sorted(rows, key=lambda row: float(row['area_error']), reverse=True))
    add('success_high_iou', sorted(rows, key=lambda row: float(row['iou']), reverse=True))
    median_iou = np.median([float(row['iou']) for row in rows])
    add('typical_median_iou', sorted(rows, key=lambda row: abs(float(row['iou']) - median_iou)))
    for group in ['small', 'medium', 'large']:
        add(group, [row for row in sorted_by_iou if row['area_bin'] == group])
    add('low_signal', [row for row in sorted_by_iou if row['signal_bin'] == 'low_signal'])
    for defect_type in ['polygon', 'rotated_rect', 'multi_defect']:
        add(defect_type, [row for row in sorted_by_iou if row['defect_type'] == defect_type])
    add('bz_improves_mask_worsens', [row for row in rows if int(row['bz_improves_mask_worsens']) == 1])
    for row in sorted_by_iou:
        if len(picks) >= 12:
            break
        add('fill', [row])
    return picks[:12]


def draw_overlay(ax, true_mask, pred_mask):
    image = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    image[..., 1] = true_mask.astype(np.float32)
    image[..., 0] = pred_mask.astype(np.float32)
    image[..., 2] = np.logical_and(true_mask, pred_mask).astype(np.float32)
    ax.imshow(image, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])


def write_previews(test_outputs, selected_objective):
    if selected_objective == 'current_baseline':
        prob = test_outputs['coarse_prob']
        rows = test_outputs['baseline_sample_rows']
        bz_mses = test_outputs['baseline_bz_mses']
    else:
        output = test_outputs['objectives'][selected_objective]
        prob = output['prob']
        rows = output['sample_rows']
        bz_mses = output['bz_mses']
    coarse_prob = test_outputs['coarse_prob']
    true_masks = test_outputs['true_masks']
    paths = []
    for sample_idx, label in preview_indices(rows):
        row = rows[sample_idx]
        coarse_mask = coarse_prob[sample_idx] >= CURRENT_BASELINE_THRESHOLD
        refined_mask = prob[sample_idx] >= CURRENT_BASELINE_THRESHOLD
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.ravel()
        axes[0].imshow(true_masks[sample_idx], cmap='gray', origin='lower')
        axes[0].set_title('true mask')
        axes[1].imshow(coarse_prob[sample_idx], cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        axes[1].set_title('coarse prob')
        axes[2].imshow(coarse_mask, cmap='gray', origin='lower')
        axes[2].set_title('coarse mask')
        axes[3].imshow(prob[sample_idx], cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        axes[3].set_title('refined prob')
        axes[4].imshow(prob[sample_idx] - coarse_prob[sample_idx], cmap='coolwarm', origin='lower', vmin=-0.5, vmax=0.5)
        axes[4].set_title('refined - coarse')
        draw_overlay(axes[5], true_masks[sample_idx], refined_mask)
        axes[5].set_title('overlay red=pred green=true')
        for ax in axes[:5]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={sample_idx} label={label} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} BzMSE={float(bz_mses[sample_idx]):.3e}"
        )
        fig.tight_layout()
        out_path = PREVIEW_DIR / f"{sample_idx:03d}_{label}_{row['defect_type']}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        paths.append(out_path)
    return paths


def fmt(value):
    try:
        return f'{float(value):.6f}'
    except (TypeError, ValueError):
        return str(value)


def write_summary(rows, val_outputs, test_outputs, selected_objective, preview_paths, accepted):
    val_base = find_row(rows, 'current_baseline', split='val')
    val_only = find_row(rows, 'proposal_only', split='val')
    val_forward = find_row(rows, 'proposal_forward', split='val')
    test_base = find_row(rows, 'current_baseline', split='test')
    test_selected = find_row(rows, selected_objective, split='test') if selected_objective != 'current_baseline' else test_base
    test_only = find_row(rows, 'proposal_only', split='test')
    test_forward = find_row(rows, 'proposal_forward', split='test')
    selected_rows = (
        test_outputs['baseline_sample_rows']
        if selected_objective == 'current_baseline'
        else test_outputs['objectives'][selected_objective]['sample_rows']
    )
    bz_mask_tradeoff = int(sum(int(row['bz_improves_mask_worsens']) for row in selected_rows))
    large_drift = int(sum(int(row['large_probability_drift']) for row in selected_rows))
    summary = f"""# v3_complex CURRENT_BASELINE mask-logit refinement gate

This Step 19.4 script is independent. It does not train a neural network and does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, PINN优化路线.md, or NEXT_STEP.md.

## Current baseline coarse prediction

CURRENT_BASELINE is loaded from the three mask-only grid decoder + forward consistency checkpoints recorded in scripts.train_geometry_forward_consistency_candidate.CURRENT_BASELINE_CHECKPOINTS:

* `{CURRENT_BASELINE_CHECKPOINTS[42]}`
* `{CURRENT_BASELINE_CHECKPOINTS[123]}`
* `{CURRENT_BASELINE_CHECKPOINTS[2026]}`

The script computes each seed probability map and uses the 3-seed mean probability as `coarse_prob`. The fixed CURRENT_BASELINE threshold is `{CURRENT_BASELINE_THRESHOLD}`. Bz residual is computed with the frozen mask-to-Bz surrogate loaded via `load_forward_surrogate`.

## Refinement setup

For each validation/test sample, `coarse_prob` is clamped and converted to logits. Only those logits are optimized for {OPT_STEPS} Adam steps at lr={OPT_LR}. No true mask is used during optimization.

* proposal_only: MSE(refined_prob, coarse_prob) + {ALPHA_TV} * TV(refined_prob)
* proposal_forward: MSE(refined_prob, coarse_prob) + {ALPHA_FORWARD} * MSE(frozen_surrogate(refined_prob), Bz_obs) + {ALPHA_TV} * TV(refined_prob)

No threshold sweep, post-processing, weight tuning, or refinement v2 is used.

## Validation selection

| candidate | IoU | Dice | area_error | pred_area=0 | Bz MSE | score |
|---|---:|---:|---:|---:|---:|---:|
| current_baseline | {fmt(val_base['iou'])} | {fmt(val_base['dice'])} | {fmt(val_base['area_error'])} | {fmt(val_base['pred_area_zero'])} | {fmt(val_base['bz_mse'])} | {fmt(val_base['composite'])} |
| proposal_only | {fmt(val_only['iou'])} | {fmt(val_only['dice'])} | {fmt(val_only['area_error'])} | {fmt(val_only['pred_area_zero'])} | {fmt(val_only['bz_mse'])} | {fmt(val_only['composite'])} |
| proposal_forward | {fmt(val_forward['iou'])} | {fmt(val_forward['dice'])} | {fmt(val_forward['area_error'])} | {fmt(val_forward['pred_area_zero'])} | {fmt(val_forward['bz_mse'])} | {fmt(val_forward['composite'])} |

Validation-selected objective: `{selected_objective}`.

## Test comparison

| candidate | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_baseline | {fmt(test_base['iou'])} | {fmt(test_base['dice'])} | {fmt(test_base['area_error'])} | {fmt(test_base['center_error'])} | {fmt(test_base['pred_area_zero'])} | {fmt(test_base['bz_mse'])} | {fmt(test_base['composite'])} |
| proposal_only | {fmt(test_only['iou'])} | {fmt(test_only['dice'])} | {fmt(test_only['area_error'])} | {fmt(test_only['center_error'])} | {fmt(test_only['pred_area_zero'])} | {fmt(test_only['bz_mse'])} | {fmt(test_only['composite'])} |
| proposal_forward | {fmt(test_forward['iou'])} | {fmt(test_forward['dice'])} | {fmt(test_forward['area_error'])} | {fmt(test_forward['center_error'])} | {fmt(test_forward['pred_area_zero'])} | {fmt(test_forward['bz_mse'])} | {fmt(test_forward['composite'])} |
| selected | {fmt(test_selected['iou'])} | {fmt(test_selected['dice'])} | {fmt(test_selected['area_error'])} | {fmt(test_selected['center_error'])} | {fmt(test_selected['pred_area_zero'])} | {fmt(test_selected['bz_mse'])} | {fmt(test_selected['composite'])} |

## Diagnostics

* mean_abs_change from coarse probability: {fmt(test_selected['mean_abs_change'])}
* large probability drift count, using mean_abs_change > {DRIFT_THRESHOLD}: {large_drift}
* samples where Bz MSE improves while IoU or Dice worsens: {bz_mask_tradeoff}
* preview PNG count: {len(preview_paths)}

## Conclusion

Accepted by gate: {accepted}

"""
    if selected_objective == 'current_baseline':
        summary += (
            "Neither bounded refinement objective beat the CURRENT_BASELINE validation score. "
            "The selected result is no_refinement / CURRENT_BASELINE, so mask-logit refinement has no positive validation signal.\n"
        )
    else:
        bz_down = float(test_selected['bz_mse']) < float(test_base['bz_mse'])
        bz_meaningful = float(test_selected['bz_mse']) <= float(test_base['bz_mse']) * 0.95
        iou_kept = float(test_selected['iou']) >= float(test_base['iou']) - 0.005
        dice_kept = float(test_selected['dice']) >= float(test_base['dice']) - 0.005
        area_ok = float(test_selected['area_error']) <= float(test_base['area_error']) + 0.005
        meaningful_change = float(test_selected['mean_abs_change']) >= 0.001
        summary += (
            f"Selected refinement Bz residual decreased: {bz_down}. "
            f"Meaningful Bz residual decrease (>=5%): {bz_meaningful}. "
            f"IoU kept: {iou_kept}. Dice kept: {dice_kept}. Area error controlled: {area_ok}. "
            f"Meaningful probability change: {meaningful_change}. "
            "If Bz residual goes down but mask metrics or visual boundary quality degrade, this is treated as surrogate over-optimization rather than a new candidate direction.\n"
        )
    SUMMARY_PATH.write_text(summary, encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = load_forward_surrogate(device)

    val_outputs = run_split('val', VAL_DATA, device, forward_model, surrogate_checkpoint)
    selected_objective, selected_val = select_objective(val_outputs)
    test_outputs = run_split('test', TEST_DATA, device, forward_model, surrogate_checkpoint)

    rows = []
    rows.extend(val_outputs['baseline_metric_rows'])
    rows.extend(test_outputs['baseline_metric_rows'])
    for objective in OBJECTIVES:
        rows.extend(val_outputs['objectives'][objective]['metric_rows'])
        rows.extend(test_outputs['objectives'][objective]['metric_rows'])

    test_base = find_row(rows, 'current_baseline', split='test')
    test_selected = find_row(rows, selected_objective, split='test') if selected_objective != 'current_baseline' else test_base
    accepted = bool(
        selected_objective != 'current_baseline'
        and test_selected is not None
        and test_base is not None
        and float(test_selected['iou']) >= float(test_base['iou']) - 0.005
        and float(test_selected['dice']) >= float(test_base['dice']) - 0.005
        and float(test_selected['area_error']) <= float(test_base['area_error']) + 0.005
        and float(test_selected['pred_area_zero']) <= float(test_base['pred_area_zero']) + 1e-6
        and float(test_selected['bz_mse']) <= float(test_base['bz_mse']) * 0.95
        and float(test_selected['mean_abs_change']) >= 0.001
    )

    write_metrics(rows)
    write_validation_grid(val_outputs)
    preview_paths = write_previews(test_outputs, selected_objective)
    write_summary(rows, val_outputs, test_outputs, selected_objective, preview_paths, accepted)

    print('completed=True')
    print(f'selected_objective={selected_objective}')
    print(
        f"selected_val: IoU={float(selected_val['iou']):.6f}, Dice={float(selected_val['dice']):.6f}, "
        f"area_error={float(selected_val['area_error']):.6f}, BzMSE={float(selected_val['bz_mse']):.6e}"
    )
    if test_selected is not None:
        print(
            f"selected_test: IoU={float(test_selected['iou']):.6f}, Dice={float(test_selected['dice']):.6f}, "
            f"area_error={float(test_selected['area_error']):.6f}, BzMSE={float(test_selected['bz_mse']):.6e}"
        )
    print(f'accepted={accepted}')
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote validation grid: {VALIDATION_GRID_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    print(f'wrote previews: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
