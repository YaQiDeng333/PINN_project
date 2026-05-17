import csv
import os
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

from train_pinn import MFLDataset, build_coord_grid, project_path, set_seed, signal_shape_info  # noqa: E402
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    BATCH_SIZE,
    CURRENT_BASELINE_THRESHOLD,
    EVAL_BATCH_SIZE,
    GRID_BASE_CHANNELS,
    GRID_LOW_SHAPE,
    LATENT_DIM,
    MASK_THRESHOLD_NORM,
    MaskBoundaryGridModel,
    area_bin,
    compute_mask_metrics,
    compute_pos_weight,
    get_area_edges,
    load_grid_checkpoint,
    make_loader,
    mask_loss,
    safe_nanmean,
    safe_nanstd,
    threshold_matches,
)
from scripts.train_mask_to_bz_forward_surrogate import (  # noqa: E402
    CHECKPOINT_PATH as SURROGATE_CHECKPOINT_PATH,
    MaskToBzForwardSurrogate,
)


TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

CURRENT_BASELINE_CHECKPOINTS = {
    42: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed42.pt',
    123: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed123.pt',
    2026: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed2026.pt',
}

SEEDS = [42, 123, 2026]
SCREENING_SEED = 42
EPOCHS = 50
LR = 1e-3
TRAIN_SELECTION_THRESHOLD = 0.5
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
LAMBDA_FORWARD = 0.05
POSITIVE_SIGNAL_AREA_TOLERANCE = 0.02
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}
FORCE_THREE_SEED = os.environ.get('PINN_FORCE_THREE_SEED', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_boundary_forward_consistency_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_forward_consistency_candidate_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_boundary_forward_consistency_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'mask_boundary_forward_consistency_candidate'

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
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = [path for path in CURRENT_BASELINE_CHECKPOINTS.values() if not Path(project_path(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing current grid baseline checkpoints: ' + ', '.join(missing))
    if not SURROGATE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Missing forward surrogate checkpoint: {SURROGATE_CHECKPOINT_PATH}')


def load_forward_surrogate(device):
    checkpoint = torch.load(SURROGATE_CHECKPOINT_PATH, map_location=device)
    args = checkpoint.get('args', {})
    model = MaskToBzForwardSurrogate(
        out_length=int(args.get('signal_length', 200)),
        out_shape=tuple(args.get('out_shape', (100, 200))),
        use_coords=bool(args.get('use_coords', True)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint


def dataset_low_signal_indices(dataset):
    signals = np.asarray(dataset.signals, dtype=np.float32)
    flat = signals if signals.ndim == 2 else signals.reshape(signals.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    threshold = np.quantile(max_abs, 1 / 3)
    return {idx for idx, value in enumerate(max_abs) if float(value) <= float(threshold)}


def summarize_samples(rows):
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


def build_sample_rows(candidate, seed, split, threshold, prob_maps, true_masks, bz_mses, dataset, area_edges, low_signal_indices):
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
            'bz_mse': float(bz_mses[sample_idx]),
        })
        rows.append(metrics)
    return rows


@torch.no_grad()
def predict_prob_maps_and_bz(model, forward_model, dataset, coords, device):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    bz_mses = np.empty((len(dataset),), dtype=np.float32)
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        output = model(signals, coords)
        logits = output[0] if isinstance(output, tuple) else output
        probs = torch.sigmoid(logits).reshape(signals.shape[0], *grid_shape)
        bz_hat = forward_model(probs.unsqueeze(1))
        batch_bz_mse = torch.mean((bz_hat - signals) ** 2, dim=1).detach().cpu().numpy()
        prob_np = probs.detach().cpu().numpy()
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = prob_np[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
            bz_mses[sample_idx] = batch_bz_mse[batch_pos]
    return prob_maps, true_masks, bz_mses


def evaluate_model_for_selection(model, dataset, coords, device, area_edges):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    model.eval()
    with torch.no_grad():
        for signals, mu_targets, indices in loader:
            signals = signals.to(device)
            logits = model(signals, coords)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
            batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
            for batch_pos, sample_idx_tensor in enumerate(indices):
                sample_idx = int(sample_idx_tensor.item())
                prob_maps[sample_idx] = probs[batch_pos]
                true_masks[sample_idx] = batch_true[batch_pos]
    sample_rows = build_sample_rows(
        'selection',
        'selection',
        'val',
        TRAIN_SELECTION_THRESHOLD,
        prob_maps,
        true_masks,
        np.zeros((len(dataset),), dtype=np.float32),
        dataset,
        area_edges,
        set(),
    )
    return summarize_samples(sample_rows)


def train_one_seed(seed, device, pos_weight_value, forward_model, surrogate_checkpoint):
    set_seed(seed)
    train_dataset = MFLDataset(
        TRAIN_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    val_dataset = MFLDataset(
        VAL_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    out_shape = tuple(train_dataset.mu_maps.shape[1:])
    model = MaskBoundaryGridModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
        out_shape=out_shape,
        low_shape=GRID_LOW_SHAPE,
        base_channels=GRID_BASE_CHANNELS,
    ).to(device)
    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_mask_boundary_forward_consistency_seed{seed}.pt'

    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu')
        print(f'Reusing existing checkpoint seed={seed}: {best_path}')
        return best_path, checkpoint.get('val_metrics', {})

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_mask = 0.0
        total_forward = 0.0
        total_samples = 0
        for signals, mu_targets, indices in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            mask_logits = model(signals, coords)
            loss_mask, _, _ = mask_loss(mask_logits, target_mask, pos_weight)
            mask_prob = torch.sigmoid(mask_logits).reshape(signals.shape[0], *out_shape)
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
                    'model': 'mask_boundary_grid_forward_consistency_candidate',
                    'dataset': 'v3_complex',
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'loss': 'BCEWithLogits + soft Dice + 0.05 * frozen mask-to-Bz surrogate MSE',
                    'lambda_forward': LAMBDA_FORWARD,
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'surrogate_checkpoint': str(SURROGATE_CHECKPOINT_PATH.relative_to(ROOT)),
                    'out_shape': out_shape,
                    'low_shape': GRID_LOW_SHAPE,
                    'base_channels': GRID_BASE_CHANNELS,
                    'signal_channels': signal_channels,
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
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
            f"mask_loss={total_mask / total_samples:.6e} | "
            f"forward_mse={total_forward / total_samples:.6e} | "
            f"val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | "
            f"score={selection_score:.6e}"
        )

    return best_path, best_info


def evaluate_checkpoint_family(checkpoints, candidate, split, data_path, thresholds, device, forward_model):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    prob_cache = {}
    area_dataset = MFLDataset(data_path)
    area_edges = get_area_edges(area_dataset)
    low_signal_indices = dataset_low_signal_indices(area_dataset)
    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(project_path(str(checkpoint_path)), map_location='cpu') if not isinstance(checkpoint_path, Path) else torch.load(checkpoint_path, map_location='cpu')
        dataset = MFLDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        if candidate.startswith('current'):
            model, _ = load_grid_checkpoint(Path(project_path(str(checkpoint_path))), signal_length, signal_channels, out_shape, device)
        else:
            model, _ = load_grid_checkpoint(checkpoint_path, signal_length, signal_channels, out_shape, device)
        coords = build_coord_grid(dataset.x, dataset.y).to(device)
        prob_maps, true_masks, bz_mses = predict_prob_maps_and_bz(model, forward_model, dataset, coords, device)
        prob_cache[seed] = (prob_maps, true_masks, dataset)
        for threshold in thresholds:
            sample_rows = build_sample_rows(
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
            sample_rows_by_seed_threshold[(seed, threshold)] = sample_rows
            metric_rows.extend(summarize_candidate(sample_rows, candidate, seed, split, threshold))
    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows(metric_rows, candidate, split, threshold))
    return metric_rows, sample_rows_by_seed_threshold, prob_cache


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
    return selected[0] if selected else None


def select_threshold(rows, candidate_name):
    candidates = [
        row for row in rows
        if row['candidate'] == f'{candidate_name}_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    return max(candidates, key=lambda row: float(row['composite']))


def positive_signal(rows, candidate_name, threshold):
    base = find_row(rows, 'current_grid_baseline_val_mean', 'overall', 'all', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    cand = find_row(rows, f'{candidate_name}_mean', 'overall', 'all', split='val', threshold=threshold)
    if base is None or cand is None:
        return False, 'missing validation comparison rows'
    checks = {
        'iou_not_down': float(cand['iou']) >= float(base['iou']) - 1e-6,
        'dice_not_down': float(cand['dice']) >= float(base['dice']) - 1e-6,
        'area_error_close': float(cand['area_error']) <= float(base['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE,
        'pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'bz_mse_down': float(cand['bz_mse']) < float(base['bz_mse']),
    }
    reason = ', '.join(f'{key}={value}' for key, value in checks.items())
    return bool(all(checks.values())), reason


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
        'bz_mse',
        'composite',
        'macro_area_composite',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def fmt(value, metric=None):
    if value is None:
        return 'N/A'
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    if mean_row is None or std_row is None:
        return 'N/A'
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_val_table(rows, candidate_name):
    lines = [
        '| candidate | threshold | IoU | Dice | area_error | pred_area=0 | Bz MSE | score |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sorted(
        [r for r in rows if r['candidate'] == f'{candidate_name}_mean' and r['split'] == 'val' and r['group_type'] == 'overall' and r['group'] == 'all'],
        key=lambda item: float(item['threshold']),
    ):
        lines.append(
            f"| {candidate_name} | {float(row['threshold']):.2f} | {float(row['iou']):.4f} | "
            f"{float(row['dice']):.4f} | {float(row['area_error']):.4f} | "
            f"{float(row['pred_area_zero']):.2f} | {float(row['bz_mse']):.6e} | {float(row['composite']):.4f} |"
        )
    return '\n'.join(lines)


def format_test_comparison(rows, selected_threshold):
    lines = [
        '| group | candidate | threshold | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group_type, group in [('overall', 'all'), ('area_bin', 'small'), ('signal_bin', 'low_signal'), ('defect_type', 'polygon'), ('defect_type', 'rotated_rect'), ('defect_type', 'multi_defect')]:
        base_mean = find_row(rows, 'current_grid_baseline_test_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        base_std = find_row(rows, 'current_grid_baseline_test_std', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        cand_mean = find_row(rows, 'forward_consistency_test_mean', group_type, group, threshold=selected_threshold)
        cand_std = find_row(rows, 'forward_consistency_test_std', group_type, group, threshold=selected_threshold)
        if base_mean is not None:
            lines.append(
                f"| {group} | current grid baseline | {CURRENT_BASELINE_THRESHOLD:.2f} | "
                f"{metric_with_std(base_mean, base_std, 'iou')} | "
                f"{metric_with_std(base_mean, base_std, 'dice')} | "
                f"{metric_with_std(base_mean, base_std, 'area_error')} | "
                f"{metric_with_std(base_mean, base_std, 'center_error')} | "
                f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
                f"{metric_with_std(base_mean, base_std, 'bz_mse')} |"
            )
        if cand_mean is not None:
            lines.append(
                f"| {group} | forward consistency | {selected_threshold:.2f} | "
                f"{metric_with_std(cand_mean, cand_std, 'iou')} | "
                f"{metric_with_std(cand_mean, cand_std, 'dice')} | "
                f"{metric_with_std(cand_mean, cand_std, 'area_error')} | "
                f"{metric_with_std(cand_mean, cand_std, 'center_error')} | "
                f"{metric_with_std(cand_mean, cand_std, 'pred_area_zero')} | "
                f"{metric_with_std(cand_mean, cand_std, 'bz_mse')} |"
            )
    return '\n'.join(lines)


def write_previews(seed, selected_threshold, sample_rows_by_seed_threshold, prob_cache):
    if (seed, selected_threshold) not in sample_rows_by_seed_threshold:
        return
    rows = sample_rows_by_seed_threshold[(seed, selected_threshold)]
    prob_maps, true_masks, dataset = prob_cache[seed]
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    selected = sorted(rows, key=lambda row: float(row['iou']), reverse=True)[:3]
    selected += sorted([row for row in rows if row['signal_bin'] == 'low_signal'], key=lambda row: float(row['iou']), reverse=True)[:3]
    selected += sorted(rows, key=lambda row: float(row['iou']))[:3]
    selected += sorted(rows, key=lambda row: abs(float(row['iou']) - safe_nanmean([float(r['iou']) for r in rows])))[:3]
    seen = []
    for row in selected:
        idx = int(row['sample_index'])
        if idx not in seen:
            seen.append(idx)
        if len(seen) >= 12:
            break
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
        axes[3].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[3].imshow(pred_mask, origin='lower', cmap='Reds', alpha=0.35)
        axes[3].contour(x_grid, y_grid, true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        axes[3].contour(x_grid, y_grid, pred_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        axes[3].set_title('overlay')
        fig.suptitle(
            f"sample={sample_idx} type={row['defect_type']} IoU={float(row['iou']):.3f} "
            f"Dice={float(row['dice']):.3f} area_error={float(row['area_error']):.3f} "
            f"BzMSE={float(row['bz_mse']):.3e}",
            fontsize=10,
        )
        fig.savefig(PREVIEW_DIR / f'forward_consistency_seed{seed}_rank{rank:02d}_sample{sample_idx}_{row["defect_type"]}.png', dpi=150)
        plt.close(fig)


def write_summary(rows, screening_info, stage_b_entered, selected_threshold, accepted, checkpoint_paths, best_infos):
    summary = f"""# v3_complex mask boundary grid forward consistency candidate

This Step 18.2B script is independent and does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, or NEXT_STEP.md.

Frozen forward surrogate: `{SURROGATE_CHECKPOINT_PATH.relative_to(ROOT)}`

Loss: BCEWithLogits + soft Dice + {LAMBDA_FORWARD} * MSE(frozen_surrogate(mask_prob), observed_normalized_Bz). The forward surrogate is frozen. No lambda search, post-processing, adaptive threshold, COMSOL, or forward consistency v2 is used.

## Seed=42 screening

Validation-selected seed=42 threshold: {screening_info.get('threshold', 'N/A')}

Positive signal: {screening_info.get('positive_signal', False)}

Reason: {screening_info.get('reason', 'N/A')}

{format_val_table(rows, 'forward_consistency_screening')}

## Stage B

Entered 3-seed validation: {stage_b_entered}

Validation selected final threshold: {selected_threshold if selected_threshold is not None else 'N/A'}

"""
    if best_infos:
        summary += "| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | checkpoint |\n"
        summary += "|---:|---:|---:|---:|---:|---:|---|\n"
        for info, path in zip(best_infos, checkpoint_paths):
            summary += (
                f"| {info['seed']} | {info['epoch']} | {info['selection_score']:.6e} | "
                f"{info['val_iou']:.4f} | {info['val_dice']:.4f} | {info['val_area_error']:.4f} | "
                f"{path.relative_to(ROOT)} |\n"
            )
    if stage_b_entered and selected_threshold is not None:
        summary += "\n## Test comparison\n\n"
        summary += format_test_comparison(rows, selected_threshold)
        base = find_row(rows, 'current_grid_baseline_test_mean', 'overall', 'all', threshold=CURRENT_BASELINE_THRESHOLD)
        cand = find_row(rows, 'forward_consistency_test_mean', 'overall', 'all', threshold=selected_threshold)
        bz_down = cand is not None and base is not None and float(cand['bz_mse']) < float(base['bz_mse'])
        summary += f"\n\nBz residual decreased: {bz_down}\n"
    else:
        summary += "\nStage B was not entered, so no 3-seed test comparison or preview PNGs are generated.\n"
    summary += f"\nAccepted by metric gate: {accepted}\n"
    if not accepted:
        summary += "\nConclusion: forward consistency does not satisfy the acceptance gate in this run; do not continue lambda_forward tuning or forward consistency v2 from this result.\n"
    SUMMARY_PATH.write_text(summary, encoding='utf-8')


def main():
    ensure_outputs()
    check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = load_forward_surrogate(device)
    train_dataset = MFLDataset(
        TRAIN_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    pos_weight, mask_fraction = compute_pos_weight(train_dataset)
    print(f'train mask positive fraction={mask_fraction:.6f}, pos_weight={pos_weight:.6f}')

    baseline_val_rows, _, _ = evaluate_checkpoint_family(
        CURRENT_BASELINE_CHECKPOINTS,
        'current_grid_baseline_val',
        'val',
        VAL_DATA,
        [CURRENT_BASELINE_THRESHOLD],
        device,
        forward_model,
    )
    baseline_test_rows, _, _ = evaluate_checkpoint_family(
        CURRENT_BASELINE_CHECKPOINTS,
        'current_grid_baseline_test',
        'test',
        TEST_DATA,
        [CURRENT_BASELINE_THRESHOLD],
        device,
        forward_model,
    )

    checkpoint_42, info_42 = train_one_seed(SCREENING_SEED, device, pos_weight, forward_model, surrogate_checkpoint)
    screening_rows, _, _ = evaluate_checkpoint_family(
        {SCREENING_SEED: checkpoint_42},
        'forward_consistency_screening',
        'val',
        VAL_DATA,
        THRESHOLDS,
        device,
        forward_model,
    )
    screening_all_rows = baseline_val_rows + screening_rows
    selected_screening = select_threshold(screening_all_rows, 'forward_consistency_screening')
    screening_threshold = float(selected_screening['threshold'])
    ok, reason = positive_signal(screening_all_rows, 'forward_consistency_screening', screening_threshold)
    screening_info = {
        'threshold': f'{screening_threshold:.2f}',
        'positive_signal': ok,
        'reason': reason,
    }
    print(f'screening threshold={screening_threshold:.2f}, positive_signal={ok}, reason={reason}')

    all_rows = baseline_val_rows + baseline_test_rows + screening_rows
    checkpoint_paths = [checkpoint_42]
    best_infos = [info_42]
    stage_b_entered = False
    selected_threshold = None
    accepted = False

    if ok or FORCE_THREE_SEED:
        stage_b_entered = True
        if FORCE_THREE_SEED and not ok:
            screening_info['reason'] = screening_info['reason'] + '; controlled extension forced 3-seed validation'
        checkpoint_paths = []
        best_infos = []
        for seed in SEEDS:
            if seed == SCREENING_SEED:
                checkpoint_paths.append(checkpoint_42)
                best_infos.append(info_42)
            else:
                path, info = train_one_seed(seed, device, pos_weight, forward_model, surrogate_checkpoint)
                checkpoint_paths.append(path)
                best_infos.append(info)
        checkpoints = {seed: path for seed, path in zip(SEEDS, checkpoint_paths)}
        val_rows, _, _ = evaluate_checkpoint_family(
            checkpoints,
            'forward_consistency_val',
            'val',
            VAL_DATA,
            THRESHOLDS,
            device,
            forward_model,
        )
        selected = select_threshold(val_rows, 'forward_consistency_val')
        selected_threshold = float(selected['threshold'])
        test_rows, sample_rows, prob_cache = evaluate_checkpoint_family(
            checkpoints,
            'forward_consistency_test',
            'test',
            TEST_DATA,
            THRESHOLDS,
            device,
            forward_model,
        )
        all_rows = baseline_val_rows + baseline_test_rows + screening_rows + val_rows + test_rows
        base = find_row(all_rows, 'current_grid_baseline_test_mean', 'overall', 'all', threshold=CURRENT_BASELINE_THRESHOLD)
        cand = find_row(all_rows, 'forward_consistency_test_mean', 'overall', 'all', threshold=selected_threshold)
        polygon = find_row(all_rows, 'forward_consistency_test_mean', 'defect_type', 'polygon', threshold=selected_threshold)
        base_polygon = find_row(all_rows, 'current_grid_baseline_test_mean', 'defect_type', 'polygon', threshold=CURRENT_BASELINE_THRESHOLD)
        rotated = find_row(all_rows, 'forward_consistency_test_mean', 'defect_type', 'rotated_rect', threshold=selected_threshold)
        base_rotated = find_row(all_rows, 'current_grid_baseline_test_mean', 'defect_type', 'rotated_rect', threshold=CURRENT_BASELINE_THRESHOLD)
        accepted = bool(
            cand is not None
            and float(cand['iou']) >= float(base['iou']) - 1e-6
            and float(cand['dice']) >= float(base['dice']) - 1e-6
            and float(cand['area_error']) <= float(base['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE
            and float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6
            and float(cand['bz_mse']) < float(base['bz_mse'])
            and (
                (polygon is not None and float(polygon['iou']) >= float(base_polygon['iou']) - 1e-6)
                or (rotated is not None and float(rotated['iou']) >= float(base_rotated['iou']) - 1e-6)
            )
        )
        write_previews(SEEDS[0], selected_threshold, sample_rows, prob_cache)
        print(f'final selected threshold={selected_threshold:.2f}, accepted={accepted}')

    write_metrics(all_rows)
    write_summary(all_rows, screening_info, stage_b_entered, selected_threshold, accepted, checkpoint_paths, best_infos)
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    preview_status = str(PREVIEW_DIR) if stage_b_entered else 'not generated'
    print(f'wrote previews: {preview_status}')


if __name__ == '__main__':
    main()
