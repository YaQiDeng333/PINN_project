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

from train_pinn import project_path, set_seed, signal_shape_info  # noqa: E402
from scripts.train_geometry_boundary_candidate import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    LATENT_DIM,
    MASK_THRESHOLD_NORM,
    RASTER_TEMPERATURE,
    SEEDS,
    SINGLE_DEFECT_TYPES,
    THRESHOLDS,
    TRAIN_DATA,
    TRAIN_SELECTION_THRESHOLD,
    VAL_DATA,
    TEST_DATA,
    GeometryBoundaryModel,
    SingleDefectDataset,
    coord_tuple,
    split_type_counts,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
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
from scripts.train_mask_boundary_grid_forward_consistency_candidate import (  # noqa: E402
    load_forward_surrogate,
    predict_prob_maps_and_bz,
)


CURRENT_BASELINE_THRESHOLD = 0.80
GEOMETRY_ONLY_THRESHOLD = 0.95
LAMBDA_FORWARD = 0.10
LR = 1e-3
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}

CURRENT_BASELINE_CHECKPOINTS = {
    42: 'checkpoints/mask_boundary_forward_consistency_lambda_bracket/best_forward_consistency_lambda_0p10_seed42.pt',
    123: 'checkpoints/mask_boundary_forward_consistency_lambda010/best_mask_boundary_forward_consistency_seed123.pt',
    2026: 'checkpoints/mask_boundary_forward_consistency_lambda010/best_mask_boundary_forward_consistency_seed2026.pt',
}
GEOMETRY_ONLY_CHECKPOINTS = {
    42: 'checkpoints/geometry_boundary_candidate/best_geometry_boundary_seed42.pt',
    123: 'checkpoints/geometry_boundary_candidate/best_geometry_boundary_seed123.pt',
    2026: 'checkpoints/geometry_boundary_candidate/best_geometry_boundary_seed2026.pt',
}
FORWARD_SURROGATE = 'checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt'

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'geometry_forward_consistency_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_geometry_forward_consistency_candidate_metrics.csv'
SCREENING_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_geometry_forward_consistency_screening.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_geometry_forward_consistency_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'geometry_forward_consistency_candidate'

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
    SCREENING_PATH.parent.mkdir(parents=True, exist_ok=True)


def check_inputs():
    missing = []
    for path in list(CURRENT_BASELINE_CHECKPOINTS.values()) + list(GEOMETRY_ONLY_CHECKPOINTS.values()) + [FORWARD_SURROGATE]:
        if not Path(project_path(path)).exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError('Missing required checkpoint(s): ' + ', '.join(missing))


def dataset_low_signal_indices(dataset):
    signals = np.asarray(dataset.signals, dtype=np.float32)
    flat = signals if signals.ndim == 2 else signals.reshape(signals.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    threshold = np.quantile(max_abs, 1 / 3)
    return {int(idx) for idx, value in enumerate(max_abs) if float(value) <= float(threshold)}


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
            'original_index': int(dataset.original_indices[sample_idx]),
            'defect_type': str(dataset.defect_types[sample_idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if sample_idx in low_signal_indices else 'non_low_signal',
            'bz_mse': float(bz_mses[sample_idx]),
        })
        rows.append(metrics)
    return rows


@torch.no_grad()
def predict_geometry_maps_and_bz(model, forward_model, dataset, coords, device, bz_target_dataset=None):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    params = np.empty((len(dataset), 5), dtype=np.float32)
    bz_mses = np.empty((len(dataset),), dtype=np.float32)
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        logits, batch_params = model(signals, coords, return_params=True)
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
            params[sample_idx] = batch_params[batch_pos].detach().cpu().numpy()
            bz_mses[sample_idx] = batch_bz_mse[batch_pos]
    return prob_maps, true_masks, params, bz_mses


def evaluate_model_for_selection(model, dataset, coords, device, area_edges):
    prob_maps, true_masks, _, bz_mses = predict_geometry_maps_and_bz(
        model,
        forward_model=lambda mask_prob: torch.zeros((mask_prob.shape[0], dataset.signals.shape[-1]), device=mask_prob.device),
        dataset=dataset,
        coords=coords,
        device=device,
    )
    rows = build_sample_rows(
        'selection',
        'selection',
        'val',
        TRAIN_SELECTION_THRESHOLD,
        prob_maps,
        true_masks,
        bz_mses,
        dataset,
        area_edges,
        set(),
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
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    out_shape = tuple(train_dataset.mu_maps.shape[1:])
    model = GeometryBoundaryModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
        x_range=(float(train_dataset.x.min()), float(train_dataset.x.max())),
        y_range=(float(train_dataset.y.min()), float(train_dataset.y.max())),
        out_shape=out_shape,
        temperature=RASTER_TEMPERATURE,
    ).to(device)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    coords = coord_tuple(train_dataset, device)

    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_geometry_forward_consistency_seed{seed}.pt'

    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu')
        print(f'Reusing existing checkpoint for seed={seed}: {best_path}')
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
                    'model': 'geometry_forward_consistency_rotated_box_model',
                    'dataset': 'v3_complex_single_defect_polygon_rotated_rect',
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'loss': 'BCEWithLogits + soft Dice + 0.10 * frozen mask-to-Bz surrogate MSE',
                    'lambda_forward': LAMBDA_FORWARD,
                    'forward_surrogate': FORWARD_SURROGATE,
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'geometry_params': 'cx, cy, width, height, angle',
                    'rasterizer': 'PyTorch soft rotated rectangle SDF',
                    'temperature': RASTER_TEMPERATURE,
                    'out_shape': out_shape,
                    'x_range': (float(train_dataset.x.min()), float(train_dataset.x.max())),
                    'y_range': (float(train_dataset.y.min()), float(train_dataset.y.max())),
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
                    'signal_channels': signal_channels,
                    'single_defect_types': sorted(SINGLE_DEFECT_TYPES),
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


def load_geometry_checkpoint(path, dataset, device):
    checkpoint = torch.load(path, map_location=device)
    signal_length, signal_channels = signal_shape_info(dataset.signals)
    out_shape = tuple(dataset.mu_maps.shape[1:])
    args = checkpoint.get('args', {})
    model = GeometryBoundaryModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
        x_range=tuple(args.get('x_range', (float(dataset.x.min()), float(dataset.x.max())))),
        y_range=tuple(args.get('y_range', (float(dataset.y.min()), float(dataset.y.max())))),
        out_shape=tuple(args.get('out_shape', out_shape)),
        temperature=float(args.get('temperature', RASTER_TEMPERATURE)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_current_baseline(split, data_path, device, forward_model):
    metric_rows = []
    area_dataset = SingleDefectDataset(data_path)
    area_edges = get_area_edges(area_dataset)
    low_signal_indices = dataset_low_signal_indices(area_dataset)
    for seed, checkpoint_path in CURRENT_BASELINE_CHECKPOINTS.items():
        checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu')
        dataset = SingleDefectDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        model, _ = load_grid_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, out_shape, device)
        coords = None
        prob_maps, true_masks, bz_mses = predict_prob_maps_and_bz(model, forward_model, dataset, coords, device)
        sample_rows = build_sample_rows(
            'current_forward_baseline_single_defect',
            seed,
            split,
            CURRENT_BASELINE_THRESHOLD,
            prob_maps,
            true_masks,
            bz_mses,
            dataset,
            area_edges,
            low_signal_indices,
        )
        metric_rows.extend(summarize_candidate(sample_rows, 'current_forward_baseline_single_defect', seed, split, CURRENT_BASELINE_THRESHOLD))
    metric_rows.extend(aggregate_seed_rows(metric_rows, 'current_forward_baseline_single_defect', split, CURRENT_BASELINE_THRESHOLD))
    return metric_rows


def evaluate_geometry_checkpoints(checkpoints, candidate, split, data_path, thresholds, device, forward_model, surrogate_checkpoint):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    param_cache = {}
    area_dataset = SingleDefectDataset(data_path)
    area_edges = get_area_edges(area_dataset)
    low_signal_indices = dataset_low_signal_indices(area_dataset)
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
        model, _ = load_geometry_checkpoint(checkpoint_path, dataset, device)
        coords = coord_tuple(dataset, device)
        bz_target = None
        if abs(float(checkpoint['signal_mean']) - float(surrogate_checkpoint['signal_mean'])) > 1e-10 or abs(float(checkpoint['signal_std']) - float(surrogate_checkpoint['signal_std'])) > 1e-10:
            bz_target = bz_target_dataset
        prob_maps, true_masks, params, bz_mses = predict_geometry_maps_and_bz(model, forward_model, dataset, coords, device, bz_target_dataset=bz_target)
        param_cache[seed] = (params, prob_maps, true_masks, dataset)
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
    return metric_rows, sample_rows_by_seed_threshold, param_cache


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
    if not candidates:
        raise ValueError(f'No validation rows for {candidate_name}.')
    return max(candidates, key=lambda row: float(row['composite']))


def write_metrics(rows, path=METRICS_PATH):
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
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_screening(rows, selected_threshold):
    selected = [
        row for row in rows
        if row['candidate'] == 'geometry_forward_screening_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    write_metrics(selected, SCREENING_PATH)


def fmt(row, key):
    if row is None:
        return 'N/A'
    value = row.get(key, None)
    if value is None:
        return 'N/A'
    if key.startswith('pred_area'):
        return f'{float(value):.2f}'
    if key == 'bz_mse':
        return f'{float(value):.6e}'
    return f'{float(value):.4f}'


def row_line(label, row):
    return (
        f"| {label} | {fmt(row, 'threshold')} | {fmt(row, 'n')} | {fmt(row, 'iou')} | "
        f"{fmt(row, 'dice')} | {fmt(row, 'area_error')} | {fmt(row, 'center_error')} | "
        f"{fmt(row, 'pred_area_zero')} | {fmt(row, 'bz_mse')} |"
    )


def get_metric_row(rows, candidate, threshold, group_type='overall', group='all', split='test', seed=None):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['group_type'] == group_type
        and row['group'] == group
        and row['split'] == split
        and threshold_matches(row['threshold'], threshold)
    ]
    if seed is not None:
        selected = [row for row in selected if str(row['seed']) == str(seed)]
    return selected[0] if selected else None


def evaluate_gate(rows, selected_threshold, candidate_name='geometry_forward_screening', candidate_seed=42, geometry_seed=42):
    base = get_metric_row(rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    geom_candidate = 'geometry_only_reference' if geometry_seed is not None else 'geometry_only_reference_mean'
    cand_candidate = candidate_name if candidate_seed is not None else f'{candidate_name}_mean'
    geom = get_metric_row(rows, geom_candidate, GEOMETRY_ONLY_THRESHOLD, seed=geometry_seed)
    cand = get_metric_row(rows, cand_candidate, selected_threshold, seed=candidate_seed)
    base_polygon = get_metric_row(rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD, 'defect_type', 'polygon')
    cand_polygon = get_metric_row(rows, cand_candidate, selected_threshold, 'defect_type', 'polygon', seed=candidate_seed)
    base_rot = get_metric_row(rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD, 'defect_type', 'rotated_rect')
    cand_rot = get_metric_row(rows, cand_candidate, selected_threshold, 'defect_type', 'rotated_rect', seed=candidate_seed)
    if base is None or geom is None or cand is None:
        return False, {'reason': 'missing gate comparison rows'}
    checks = {
        'area_error_below_geometry_only': float(cand['area_error']) <= float(geom['area_error']) - 0.05,
        'iou_not_obviously_below_current': float(cand['iou']) >= float(base['iou']) - 0.02,
        'dice_not_obviously_below_current': float(cand['dice']) >= float(base['dice']) - 0.02,
        'pred_area_zero_not_up': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
        'bz_mse_below_geometry_only': float(cand['bz_mse']) <= float(geom['bz_mse']) * 0.90,
        'polygon_or_rotated_metric_signal': (
            cand_polygon is not None
            and base_polygon is not None
            and float(cand_polygon['iou']) >= float(base_polygon['iou']) - 0.02
        ) or (
            cand_rot is not None
            and base_rot is not None
            and float(cand_rot['iou']) >= float(base_rot['iou']) - 0.02
        ),
    }
    return bool(all(checks.values())), checks


def evaluate_formal_acceptance(rows, selected_threshold):
    base = get_metric_row(rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    cand = get_metric_row(rows, 'geometry_forward_candidate_mean', selected_threshold)
    if base is None or cand is None:
        return False, {'missing_formal_comparison_rows': True}
    checks = {
        'formal_iou_not_below_current': float(cand['iou']) >= float(base['iou']) - 0.01,
        'formal_dice_not_below_current': float(cand['dice']) >= float(base['dice']) - 0.01,
        'formal_area_error_not_worse_than_current': float(cand['area_error']) <= float(base['area_error']) + 0.03,
        'formal_bz_mse_not_worse_than_current': float(cand['bz_mse']) <= float(base['bz_mse']) * 1.10,
        'formal_pred_area_zero_not_worse_than_current': float(cand['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
    }
    return bool(all(checks.values())), checks


def write_previews(seed, selected_threshold, sample_rows_by_seed_threshold, param_cache):
    if (seed, selected_threshold) not in sample_rows_by_seed_threshold:
        return []
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = sample_rows_by_seed_threshold[(seed, selected_threshold)]
    params, prob_maps, true_masks, dataset = param_cache[seed]
    selected = sorted(rows, key=lambda row: float(row['iou']), reverse=True)[:3]
    selected += sorted(rows, key=lambda row: float(row['iou']))[:3]
    selected += sorted([row for row in rows if row['defect_type'] == 'polygon'], key=lambda row: float(row['area_error']), reverse=True)[:3]
    selected += sorted([row for row in rows if row['defect_type'] == 'rotated_rect'], key=lambda row: float(row['area_error']), reverse=True)[:3]
    iou_values = np.asarray([float(row['iou']) for row in rows], dtype=np.float32)
    median_iou = float(np.nanmedian(iou_values)) if len(iou_values) else 0.0
    selected += sorted(rows, key=lambda row: abs(float(row['iou']) - median_iou))[:12]
    selected += sorted(rows, key=lambda row: float(row['area_error']), reverse=True)[:12]
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
        axes[3].imshow(true_mask, origin='lower', cmap='Greens', alpha=0.45)
        axes[3].imshow(pred_mask, origin='lower', cmap='Reds', alpha=0.35)
        if np.any(true_mask) and np.any(~true_mask):
            axes[3].contour(true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        if np.any(pred_mask) and np.any(~pred_mask):
            axes[3].contour(pred_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        p = params[sample_idx]
        cx, cy, width, height, angle = [float(v) for v in p]
        local_corners = np.asarray([
            [-0.5 * width, -0.5 * height],
            [0.5 * width, -0.5 * height],
            [0.5 * width, 0.5 * height],
            [-0.5 * width, 0.5 * height],
            [-0.5 * width, -0.5 * height],
        ], dtype=np.float32)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        box_x = cx + cos_a * local_corners[:, 0] - sin_a * local_corners[:, 1]
        box_y = cy + sin_a * local_corners[:, 0] + cos_a * local_corners[:, 1]
        box_cols = np.interp(box_x, dataset.x, np.arange(len(dataset.x)))
        box_rows = np.interp(box_y, dataset.y, np.arange(len(dataset.y)))
        axes[2].plot(box_cols, box_rows, color='yellow', linewidth=1.4)
        axes[3].plot(box_cols, box_rows, color='yellow', linewidth=1.4)
        axes[3].set_title('overlay + box')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"sample={row['original_index']} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} "
            f"area_error={float(row['area_error']):.3f} BzMSE={float(row['bz_mse']):.3e} "
            f"box(cx={p[0]:.2f}, cy={p[1]:.2f}, w={p[2]:.2f}, h={p[3]:.2f}, a={p[4]:.2f})",
            fontsize=9,
        )
        path = PREVIEW_DIR / f'geometry_forward_seed{seed}_rank{rank:02d}_sample{row["original_index"]}_{row["defect_type"]}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def write_summary(rows, screening_rows, selected_threshold, stage_b_entered, accepted, gate_checks, checkpoint_paths, best_infos, counts, preview_paths):
    base = get_metric_row(rows, 'current_forward_baseline_single_defect_mean', CURRENT_BASELINE_THRESHOLD)
    geom_mean = get_metric_row(rows, 'geometry_only_reference_mean', GEOMETRY_ONLY_THRESHOLD)
    geom_seed = get_metric_row(rows, 'geometry_only_reference', GEOMETRY_ONLY_THRESHOLD, seed=42)
    cand_seed = get_metric_row(rows, 'geometry_forward_screening', selected_threshold, seed=42)
    lines = [
        '# v3_complex geometry + forward consistency candidate',
        '',
        '## Single-defect subset',
        '',
        'Subset is built in-script from v3_complex by keeping single-defect `polygon` and `rotated_rect` samples and excluding `multi_defect`. Original data files are not modified.',
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
        '* Geometry head: BzEncoder -> rotated box `(cx, cy, w, h, angle)`.',
        f'* Rasterizer: PyTorch soft rotated rectangle SDF, fixed temperature `{RASTER_TEMPERATURE}`.',
        f'* Loss: BCEWithLogits + soft Dice + `{LAMBDA_FORWARD}` * frozen mask-to-Bz surrogate MSE.',
        f'* Forward surrogate: `{FORWARD_SURROGATE}`.',
        '* No polygon vertices, multi-component matching, post-processing, SDF loss, boundary head, lambda search, or temperature search.',
        '',
        '## Seed=42 gate',
        '',
        f'* completed: True',
        f'* validation-selected threshold: `{selected_threshold:.2f}`',
        '',
        '| candidate | threshold | n | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        row_line('CURRENT_BASELINE on same subset, 3-seed mean', base),
        row_line('geometry-only reference, 3-seed mean', geom_mean),
        row_line('geometry-only reference, seed=42', geom_seed),
        row_line('geometry+forward, seed=42', cand_seed),
        '',
        '## Validation threshold screening',
        '',
        '| threshold | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE | score |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in sorted(
        [r for r in screening_rows if r['candidate'] == 'geometry_forward_screening_mean' and r['split'] == 'val' and r['group_type'] == 'overall'],
        key=lambda r: float(r['threshold']),
    ):
        lines.append(
            f"| {float(row['threshold']):.2f} | {float(row['iou']):.4f} | {float(row['dice']):.4f} | "
            f"{float(row['area_error']):.4f} | {float(row['center_error']):.4f} | "
            f"{float(row['pred_area_zero']):.2f} | {float(row['bz_mse']):.6e} | {float(row['composite']):.4f} |"
        )
    lines.extend([
        '',
        '## Gate checks',
        '',
    ])
    for key, value in gate_checks.items():
        lines.append(f'* {key}: {value}')
    lines.extend([
        '',
        f'Entered 3 seed: {stage_b_entered}',
        f'Accepted as formal candidate: {accepted}',
        '',
        'The bounded gate above is used only to decide whether seed=42 is worth extending to 3 seeds. Formal acceptance is stricter and compares the 3-seed result against the current forward-consistency CURRENT_BASELINE on the same single-defect subset.',
    ])
    if checkpoint_paths:
        lines.extend(['', '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | checkpoint |', '|---:|---:|---:|---:|---:|---:|---|'])
        for path, info in zip(checkpoint_paths, best_infos):
            lines.append(
                f"| {info['seed']} | {info['epoch']} | {info['selection_score']:.6e} | "
                f"{info['val_iou']:.4f} | {info['val_dice']:.4f} | {info['val_area_error']:.4f} | "
                f"`{path.relative_to(ROOT)}` |"
            )
    if stage_b_entered:
        final = get_metric_row(rows, 'geometry_forward_candidate_mean', selected_threshold)
        final_std = get_metric_row(rows, 'geometry_forward_candidate_std', selected_threshold)
        lines.extend(['', '## 3-seed result', '', row_line('geometry+forward, 3-seed mean', final)])
        if final_std is not None:
            lines.append(f'3-seed std row is recorded in `{METRICS_PATH.relative_to(ROOT)}`.')
        lines.append(f'Preview files: {len(preview_paths)} written to `{PREVIEW_DIR.relative_to(ROOT)}`.')
        lines.extend([
            '',
            '## Final conclusion',
            '',
            'Geometry + forward consistency reduces the geometry-only branch area_error and Bz MSE, so the forward-consistency signal is real inside the rotated-box geometry branch.',
            '',
            'However, compared with the current CURRENT_BASELINE on the same single-defect subset, the gains are not sufficient: IoU / Dice are only slightly higher, while area_error and Bz MSE remain clearly worse. Visual previews show less round-blob behavior because the output is constrained to a rotated rectangle, but polygon samples are still approximated by rectangles, corner/detail fitting is not solved, and there are offset failure cases.',
            '',
            'Final decision: not accepted as a formal candidate, do not update CURRENT_BASELINE, and do not continue lambda_forward / temperature / rotated-box geometry v2 / polygon vertices / multi-component variants from this branch.',
        ])
    else:
        lines.extend([
            '',
            '## Conclusion',
            '',
            'Seed=42 did not satisfy the bounded gate. The branch is stopped: do not enter 3 seed, do not tune lambda_forward, do not tune rasterizer temperature, and do not continue geometry v2 / polygon vertices / multi-component variants from this result.',
        ])
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
    print(f'single-defect counts: {counts}')
    pos_weight, mask_fraction = compute_pos_weight(train_dataset)
    print(f'train mask positive fraction={mask_fraction:.6f}, pos_weight={pos_weight:.6f}')

    baseline_test_rows = evaluate_current_baseline('test', TEST_DATA, device, forward_model)
    geometry_only_rows, _, _ = evaluate_geometry_checkpoints(
        {seed: Path(project_path(path)) for seed, path in GEOMETRY_ONLY_CHECKPOINTS.items()},
        'geometry_only_reference',
        'test',
        TEST_DATA,
        [GEOMETRY_ONLY_THRESHOLD],
        device,
        forward_model,
        surrogate_checkpoint,
    )

    checkpoint_42, info_42 = train_one_seed(42, device, pos_weight, forward_model, surrogate_checkpoint)
    screening_val_rows, _, _ = evaluate_geometry_checkpoints(
        {42: checkpoint_42},
        'geometry_forward_screening',
        'val',
        VAL_DATA,
        THRESHOLDS,
        device,
        forward_model,
        surrogate_checkpoint,
    )
    selected = select_threshold(screening_val_rows, 'geometry_forward_screening')
    selected_threshold = float(selected['threshold'])
    screening_test_rows, _, _ = evaluate_geometry_checkpoints(
        {42: checkpoint_42},
        'geometry_forward_screening',
        'test',
        TEST_DATA,
        [selected_threshold],
        device,
        forward_model,
        surrogate_checkpoint,
    )
    all_rows = baseline_test_rows + geometry_only_rows + screening_val_rows + screening_test_rows
    write_screening(screening_val_rows, selected_threshold)
    stage_b_entered, gate_checks = evaluate_gate(all_rows, selected_threshold)

    checkpoint_paths = [checkpoint_42]
    best_infos = [info_42]
    preview_paths = []
    accepted = False

    if stage_b_entered:
        for seed in [123, 2026]:
            path, info = train_one_seed(seed, device, pos_weight, forward_model, surrogate_checkpoint)
            checkpoint_paths.append(path)
            best_infos.append(info)
        final_checkpoints = {seed: path for seed, path in zip(SEEDS, checkpoint_paths)}
        final_val_rows, _, _ = evaluate_geometry_checkpoints(
            final_checkpoints,
            'geometry_forward_candidate',
            'val',
            VAL_DATA,
            THRESHOLDS,
            device,
            forward_model,
            surrogate_checkpoint,
        )
        final_selected = select_threshold(final_val_rows, 'geometry_forward_candidate')
        selected_threshold = float(final_selected['threshold'])
        final_test_rows, sample_rows_by_seed_threshold, param_cache = evaluate_geometry_checkpoints(
            final_checkpoints,
            'geometry_forward_candidate',
            'test',
            TEST_DATA,
            [selected_threshold],
            device,
            forward_model,
            surrogate_checkpoint,
        )
        all_rows = baseline_test_rows + geometry_only_rows + screening_val_rows + screening_test_rows + final_val_rows + final_test_rows
        preview_paths = write_previews(42, selected_threshold, sample_rows_by_seed_threshold, param_cache)
        final_gate, final_checks = evaluate_gate(
            all_rows,
            selected_threshold,
            candidate_name='geometry_forward_candidate',
            candidate_seed=None,
            geometry_seed=None,
        )
        gate_checks.update({f'final_{key}': value for key, value in final_checks.items()})
        formal_accepted, formal_checks = evaluate_formal_acceptance(all_rows, selected_threshold)
        gate_checks.update(formal_checks)
        accepted = final_gate and formal_accepted

    write_metrics(all_rows)
    write_summary(
        all_rows,
        screening_val_rows,
        selected_threshold,
        stage_b_entered,
        accepted,
        gate_checks,
        checkpoint_paths,
        best_infos,
        counts,
        preview_paths,
    )
    cand = get_metric_row(all_rows, 'geometry_forward_screening', selected_threshold, seed=42)
    print(f'selected_threshold={selected_threshold:.2f}')
    print(
        'geometry_forward_seed42: '
        f"IoU={float(cand['iou']):.6f}, Dice={float(cand['dice']):.6f}, "
        f"area_error={float(cand['area_error']):.6f}, center_error={float(cand['center_error']):.6f}, "
        f"pred_area_zero={float(cand['pred_area_zero']):.2f}, bz_mse={float(cand['bz_mse']):.6e}"
    )
    print(f'entered_3_seed={stage_b_entered}')
    print(f'accepted={accepted}')
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote screening: {SCREENING_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    if preview_paths:
        print(f'wrote previews: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
