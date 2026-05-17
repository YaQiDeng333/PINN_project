import csv
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'
CURRENT_BASELINE_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_mask_boundary_grid_candidate_metrics.csv'

METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_exemplar_shape_retrieval_baseline_metrics.csv'
VALIDATION_GRID_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_exemplar_shape_retrieval_validation_grid.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_exemplar_shape_retrieval_baseline_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'exemplar_shape_retrieval_baseline'

FEATURE_REPRESENTATIONS = ['raw_zscore', 'norm_shape', 'deriv_shape', 'stats_plus_shape']
DISTANCES = ['l2', 'cosine']
RETRIEVAL_MODES = ['top1', 'top3_mean', 'top5_mean']
MEAN_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
CURRENT_BASELINE_THRESHOLD = 0.90
MASK_THRESHOLD_NORM = 0.5
MU_SCALE = 1000.0

METRIC_KEYS = [
    'iou',
    'dice',
    'area_error',
    'center_error',
    'pred_area_zero',
    'pred_area_lt_true',
    'pred_area_gt_true',
    'composite',
]


def project_path(*parts):
    path = (ROOT / Path(*parts)).resolve()
    if ROOT.resolve() not in path.parents and path != ROOT.resolve():
        raise ValueError(f'Path must stay inside project directory: {path}')
    return path


def ensure_outputs():
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def load_npz(path):
    data = np.load(project_path(path), allow_pickle=False)
    raw_mu = data['mu_maps'].astype(np.float32)
    if float(np.nanmax(raw_mu)) > 2.0:
        mu_norm = raw_mu / MU_SCALE
        mask_rule = 'target_mu_raw < 500'
    else:
        mu_norm = raw_mu
        mask_rule = 'target_mu_norm < 0.5'
    return {
        'signals': data['signals'].astype(np.float32),
        'mu_norm': mu_norm.astype(np.float32),
        'masks': (mu_norm < MASK_THRESHOLD_NORM),
        'defect_types': data['defect_types'].astype(str),
        'x': data['x'].astype(np.float32),
        'y': data['y'].astype(np.float32),
        'mask_rule': mask_rule,
    }


def as_2d_signals(signals):
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 2:
        return signals
    if signals.ndim == 3:
        return signals.reshape(signals.shape[0], -1)
    raise ValueError(f'Unsupported signal shape: {signals.shape}')


def normalize_per_sample(signals):
    denom = np.max(np.abs(signals), axis=1, keepdims=True) + 1e-8
    return signals / denom


def derivative_stack(norm_signals):
    first = np.gradient(norm_signals, axis=1, edge_order=2).astype(np.float32)
    second = np.gradient(first, axis=1, edge_order=2).astype(np.float32)
    return first, second


def scalar_stats(raw_signals):
    max_abs = np.max(np.abs(raw_signals), axis=1)
    peak_to_peak = np.ptp(raw_signals, axis=1)
    l2_energy = np.sqrt(np.mean(np.square(raw_signals), axis=1))
    return np.stack([max_abs, peak_to_peak, l2_energy], axis=1).astype(np.float32)


def fit_feature_transform(train_raw, representation):
    raw = as_2d_signals(train_raw)
    if representation == 'raw_zscore':
        mean = float(raw.mean())
        std = float(raw.std() + 1e-8)
        features = (raw - mean) / std
        return {'representation': representation, 'raw_mean': mean, 'raw_std': std}, features.astype(np.float32)

    norm = normalize_per_sample(raw)
    if representation == 'norm_shape':
        return {'representation': representation}, norm.astype(np.float32)

    first, second = derivative_stack(norm)
    if representation == 'deriv_shape':
        return {'representation': representation}, np.concatenate([norm, first, second], axis=1).astype(np.float32)

    if representation == 'stats_plus_shape':
        stats = scalar_stats(raw)
        stat_mean = stats.mean(axis=0)
        stat_std = stats.std(axis=0) + 1e-8
        stats_norm = (stats - stat_mean[None, :]) / stat_std[None, :]
        features = np.concatenate([norm, first, second, stats_norm], axis=1)
        return {
            'representation': representation,
            'stat_mean': stat_mean.astype(float).tolist(),
            'stat_std': stat_std.astype(float).tolist(),
        }, features.astype(np.float32)

    raise ValueError(f'Unsupported representation: {representation}')


def transform_features(raw_signals, transform):
    raw = as_2d_signals(raw_signals)
    representation = transform['representation']
    if representation == 'raw_zscore':
        return ((raw - float(transform['raw_mean'])) / float(transform['raw_std'])).astype(np.float32)

    norm = normalize_per_sample(raw)
    if representation == 'norm_shape':
        return norm.astype(np.float32)

    first, second = derivative_stack(norm)
    if representation == 'deriv_shape':
        return np.concatenate([norm, first, second], axis=1).astype(np.float32)

    if representation == 'stats_plus_shape':
        stats = scalar_stats(raw)
        stat_mean = np.asarray(transform['stat_mean'], dtype=np.float32)
        stat_std = np.asarray(transform['stat_std'], dtype=np.float32)
        stats_norm = (stats - stat_mean[None, :]) / stat_std[None, :]
        return np.concatenate([norm, first, second, stats_norm], axis=1).astype(np.float32)

    raise ValueError(f'Unsupported representation: {representation}')


def topk_indices(train_features, query_features, distance, k=5, batch_size=64):
    train = train_features.astype(np.float32)
    query = query_features.astype(np.float32)
    if distance == 'cosine':
        train = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-8)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

    all_indices = []
    all_scores = []
    for start in range(0, len(query), batch_size):
        q = query[start:start + batch_size]
        if distance == 'l2':
            q_norm = np.sum(np.square(q), axis=1, keepdims=True)
            t_norm = np.sum(np.square(train), axis=1)[None, :]
            distances = q_norm + t_norm - 2.0 * q @ train.T
            part = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(distances, part, axis=1)
            order = np.argsort(vals, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
            score = np.take_along_axis(vals, order, axis=1)
        elif distance == 'cosine':
            sims = q @ train.T
            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-vals, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
            score = np.take_along_axis(vals, order, axis=1)
        else:
            raise ValueError(f'Unsupported distance: {distance}')
        all_indices.append(idx)
        all_scores.append(score)
    return np.vstack(all_indices), np.vstack(all_scores)


def predict_from_neighbors(train_masks, neighbor_indices, retrieval_mode, threshold):
    if retrieval_mode == 'top1':
        prob_maps = train_masks[neighbor_indices[:, 0]].astype(np.float32)
        pred_masks = prob_maps.astype(bool)
        return prob_maps, pred_masks

    if retrieval_mode == 'top3_mean':
        k = 3
    elif retrieval_mode == 'top5_mean':
        k = 5
    else:
        raise ValueError(f'Unsupported retrieval mode: {retrieval_mode}')
    prob_maps = train_masks[neighbor_indices[:, :k]].mean(axis=1).astype(np.float32)
    pred_masks = prob_maps >= float(threshold)
    return prob_maps, pred_masks


def mask_center(mask, x_grid, y_grid):
    if not np.any(mask):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([float(x_grid[mask].mean()), float(y_grid[mask].mean())], dtype=np.float32)


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


def get_area_edges(masks, x, y):
    dx = float(abs(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = float(abs(y[1] - y[0])) if len(y) > 1 else 1.0
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64) * dx * dy
    return np.quantile(areas, [1 / 3, 2 / 3])


def area_bin(true_area, edges):
    if true_area <= edges[0]:
        return 'small'
    if true_area <= edges[1]:
        return 'medium'
    return 'large'


def load_low_signal_indices_for_test():
    if not SIGNAL_AUDIT_PATH.exists():
        return set()
    with open(SIGNAL_AUDIT_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    values = sorted(float(row['max_abs_bz']) for row in rows)
    threshold = values[min(66, len(values) - 1)]
    return {int(row['sample_index']) for row in rows if float(row['max_abs_bz']) <= threshold}


def low_signal_indices_from_signals(signals):
    raw = as_2d_signals(signals)
    values = np.max(np.abs(raw), axis=1)
    threshold = np.quantile(values, 1 / 3)
    return {int(idx) for idx, value in enumerate(values) if float(value) <= float(threshold)}


def safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


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


def build_sample_rows(candidate, split, pred_masks, true_masks, data, area_edges, low_signal_indices, config):
    x_grid, y_grid = np.meshgrid(data['x'], data['y'])
    config_fields = {
        key: config[key]
        for key in ['feature_representation', 'distance', 'retrieval_mode', 'threshold', 'neighbor_k']
    }
    rows = []
    for idx in range(len(true_masks)):
        metrics = compute_mask_metrics(pred_masks[idx], true_masks[idx], x_grid, y_grid)
        metrics.update({
            'candidate': candidate,
            'split': split,
            'sample_index': idx,
            'defect_type': str(data['defect_types'][idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if idx in low_signal_indices else 'non_low_signal',
            **config_fields,
        })
        rows.append(metrics)
    return rows


def summarize_candidate(sample_rows, candidate, split, config):
    config_fields = {
        key: config[key]
        for key in ['feature_representation', 'distance', 'retrieval_mode', 'threshold', 'neighbor_k']
    }
    rows = []
    groups = [('overall', 'all', sample_rows)]
    for group in ['small', 'medium', 'large']:
        groups.append(('area_bin', group, [row for row in sample_rows if row['area_bin'] == group]))
    for group in ['low_signal', 'non_low_signal']:
        groups.append(('signal_bin', group, [row for row in sample_rows if row['signal_bin'] == group]))
    for defect_type in sorted({row['defect_type'] for row in sample_rows}):
        groups.append(('defect_type', defect_type, [row for row in sample_rows if row['defect_type'] == defect_type]))

    for group_type, group, selected in groups:
        summary = summarize_samples(selected)
        row = {
            'candidate': candidate,
            'split': split,
            'group_type': group_type,
            'group': group,
            'n': summary['n'],
            **config_fields,
        }
        for key in METRIC_KEYS:
            row[key] = summary[key]
        rows.append(row)
    return rows


def read_current_baseline_rows():
    with open(CURRENT_BASELINE_METRICS, newline='', encoding='utf-8') as f:
        source_rows = list(csv.DictReader(f))
    out = []
    for row in source_rows:
        if row['candidate'] not in ('mask_boundary_grid_test_mean', 'mask_boundary_grid_test_std'):
            continue
        if row['split'] != 'test':
            continue
        out.append({
            'candidate': 'current_grid_decoder_baseline_mean' if row['candidate'].endswith('_mean') else 'current_grid_decoder_baseline_std',
            'split': 'test',
            'group_type': row['group_type'],
            'group': row['group'],
            'n': row['n'],
            'feature_representation': 'current_baseline',
            'distance': 'current_baseline',
            'retrieval_mode': 'current_baseline',
            'threshold': CURRENT_BASELINE_THRESHOLD,
            'neighbor_k': '',
            'iou': row['iou'],
            'dice': row['dice'],
            'area_error': row['area_error'],
            'center_error': row['center_error'],
            'pred_area_zero': row['pred_area_zero'],
            'pred_area_lt_true': row['pred_area_lt_true'],
            'pred_area_gt_true': row['pred_area_gt_true'],
            'composite': row['composite'],
        })
    return out


def find_row(rows, candidate, group_type='overall', group='all'):
    return next(
        row for row in rows
        if row['candidate'] == candidate
        and row['group_type'] == group_type
        and row['group'] == group
    )


def evaluate_config(train_masks, data, neighbor_indices, split, area_edges, low_signal_indices, config):
    prob_maps, pred_masks = predict_from_neighbors(
        train_masks,
        neighbor_indices,
        config['retrieval_mode'],
        config['threshold'],
    )
    sample_rows = build_sample_rows(
        candidate='exemplar_shape_retrieval',
        split=split,
        pred_masks=pred_masks,
        true_masks=data['masks'],
        data=data,
        area_edges=area_edges,
        low_signal_indices=low_signal_indices,
        config=config,
    )
    metric_rows = summarize_candidate(sample_rows, 'exemplar_shape_retrieval', split, config)
    return metric_rows, sample_rows, prob_maps, pred_masks


def validation_grid(train, val):
    val_area_edges = get_area_edges(val['masks'], val['x'], val['y'])
    val_low_signal = low_signal_indices_from_signals(val['signals'])
    grid_rows = []
    best = None
    best_neighbors = None
    feature_cache = {}

    for representation in FEATURE_REPRESENTATIONS:
        transform, train_features = fit_feature_transform(train['signals'], representation)
        val_features = transform_features(val['signals'], transform)
        feature_cache[representation] = (transform, train_features)
        for distance in DISTANCES:
            neighbors, scores = topk_indices(train_features, val_features, distance, k=5)
            for retrieval_mode in RETRIEVAL_MODES:
                thresholds = ['binary'] if retrieval_mode == 'top1' else MEAN_THRESHOLDS
                for threshold in thresholds:
                    config = {
                        'feature_representation': representation,
                        'distance': distance,
                        'retrieval_mode': retrieval_mode,
                        'threshold': threshold,
                        'neighbor_k': 1 if retrieval_mode == 'top1' else int(retrieval_mode[3]),
                    }
                    metric_rows, sample_rows, _, _ = evaluate_config(
                        train['masks'],
                        val,
                        neighbors,
                        'val',
                        val_area_edges,
                        val_low_signal,
                        config,
                    )
                    overall = find_row(metric_rows, 'exemplar_shape_retrieval', 'overall', 'all')
                    grid_row = dict(config)
                    for key in ['n', *METRIC_KEYS]:
                        grid_row[key] = overall[key]
                    grid_rows.append(grid_row)
                    score = float(overall['composite'])
                    if best is None or score > float(best['composite']):
                        best = grid_row
                        best_neighbors = neighbors
    return grid_rows, best, best_neighbors, feature_cache


def write_validation_grid(rows):
    fieldnames = [
        'feature_representation',
        'distance',
        'retrieval_mode',
        'threshold',
        'neighbor_k',
        'n',
        *METRIC_KEYS,
    ]
    with open(VALIDATION_GRID_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_metrics(rows):
    fieldnames = [
        'candidate',
        'split',
        'group_type',
        'group',
        'feature_representation',
        'distance',
        'retrieval_mode',
        'threshold',
        'neighbor_k',
        'n',
        *METRIC_KEYS,
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def fmt(row, key):
    return f"{float(row[key]):.4f}"


def comparison_line(rows, group_type, group):
    base = find_row(rows, 'current_grid_decoder_baseline_mean', group_type, group)
    retrieval = find_row(rows, 'exemplar_shape_retrieval', group_type, group)
    return (
        f"| {group} | {fmt(base, 'iou')} | {fmt(base, 'dice')} | {fmt(base, 'area_error')} | "
        f"{fmt(base, 'center_error')} | {float(base['pred_area_zero']):.2f} | "
        f"{fmt(retrieval, 'iou')} | {fmt(retrieval, 'dice')} | {fmt(retrieval, 'area_error')} | "
        f"{fmt(retrieval, 'center_error')} | {float(retrieval['pred_area_zero']):.2f} |"
    )


def improvement_status(rows, group_type, group):
    base = find_row(rows, 'current_grid_decoder_baseline_mean', group_type, group)
    retrieval = find_row(rows, 'exemplar_shape_retrieval', group_type, group)
    return {
        'iou_not_down': float(retrieval['iou']) >= float(base['iou']) - 1e-6,
        'dice_not_down': float(retrieval['dice']) >= float(base['dice']) - 1e-6,
        'area_error_close': float(retrieval['area_error']) <= float(base['area_error']) + 0.02,
        'pred_area_zero_not_up': float(retrieval['pred_area_zero']) <= float(base['pred_area_zero']) + 1e-6,
    }


def write_summary(rows, validation_rows, best_config, mask_rule, preview_count):
    overall = improvement_status(rows, 'overall', 'all')
    small = improvement_status(rows, 'area_bin', 'small')
    low = improvement_status(rows, 'signal_bin', 'low_signal')
    polygon = improvement_status(rows, 'defect_type', 'polygon')
    rotated = improvement_status(rows, 'defect_type', 'rotated_rect')
    accepted = bool(
        overall['iou_not_down']
        and overall['dice_not_down']
        and overall['area_error_close']
        and overall['pred_area_zero_not_up']
        and small['iou_not_down']
        and small['dice_not_down']
        and low['iou_not_down']
        and low['dice_not_down']
        and polygon['iou_not_down']
        and rotated['iou_not_down']
    )

    top_rows = sorted(validation_rows, key=lambda row: float(row['composite']), reverse=True)[:10]
    val_lines = [
        '| rank | feature | distance | mode | threshold | IoU | Dice | area_error | center_error | pred_area=0 | score |',
        '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for rank, row in enumerate(top_rows, start=1):
        val_lines.append(
            f"| {rank} | {row['feature_representation']} | {row['distance']} | {row['retrieval_mode']} | "
            f"{row['threshold']} | {float(row['iou']):.4f} | {float(row['dice']):.4f} | "
            f"{float(row['area_error']):.4f} | {float(row['center_error']):.4f} | "
            f"{float(row['pred_area_zero']):.2f} | {float(row['composite']):.4f} |"
        )

    compare_lines = [
        '| group | baseline IoU | baseline Dice | baseline area_error | baseline center_error | baseline pred_area=0 | retrieval IoU | retrieval Dice | retrieval area_error | retrieval center_error | retrieval pred_area=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group_type, groups in [
        ('overall', ['all']),
        ('area_bin', ['small', 'medium', 'large']),
        ('signal_bin', ['low_signal', 'non_low_signal']),
        ('defect_type', ['polygon', 'rotated_rect', 'multi_defect']),
    ]:
        for group in groups:
            compare_lines.append(comparison_line(rows, group_type, group))

    status_lines = [
        f"* overall: {overall}",
        f"* small: {small}",
        f"* low_signal: {low}",
        f"* polygon: {polygon}",
        f"* rotated_rect: {rotated}",
    ]

    summary = f"""# v3_complex exemplar retrieval shape-prior baseline

This RESULT_DRIVEN_EXPERIMENT evaluates a non-neural exemplar retrieval shape prior. It does not train a network and does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

Target mask rule used: {mask_rule}.

## Validation grid

The train split is the retrieval dictionary. Validation chooses feature representation + distance + retrieval mode + threshold by IoU + Dice - area_error. Test labels are not used for configuration selection.

Feature representations: {', '.join(FEATURE_REPRESENTATIONS)}.
Distances: {', '.join(DISTANCES)}.
Retrieval modes: {', '.join(RETRIEVAL_MODES)}.
Mean-mask thresholds: {', '.join(f'{value:.2f}' for value in MEAN_THRESHOLDS)}.

Selected best config:

* feature_representation: {best_config['feature_representation']}
* distance: {best_config['distance']}
* retrieval_mode: {best_config['retrieval_mode']}
* threshold: {best_config['threshold']}
* validation score: {float(best_config['composite']):.6f}

Top validation configurations:

{chr(10).join(val_lines)}

## Test comparison

{chr(10).join(compare_lines)}

## Gate checks

{chr(10).join(status_lines)}

Accepted by metric gate: {accepted}

Preview PNG count: {preview_count}. Preview PNGs are written to `{PREVIEW_DIR.relative_to(ROOT)}` for visual inspection.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return accepted


def sample_scores(sample_rows):
    return [dict(row, mask_score=float(row['iou']) + float(row['dice']) - float(row['area_error'])) for row in sample_rows]


def select_preview_samples(sample_rows):
    rows = sample_scores(sample_rows)
    selected = []
    selected_indices = set()

    def take(category, candidates, n=3):
        count = 0
        for item in candidates:
            if item['sample_index'] in selected_indices:
                continue
            selected_indices.add(item['sample_index'])
            copy = dict(item)
            copy['category'] = category
            selected.append(copy)
            count += 1
            if count == n:
                break

    small_polygon = [row for row in rows if row['area_bin'] == 'small' and row['defect_type'] == 'polygon']
    small_polygon.sort(key=lambda row: row['mask_score'], reverse=True)
    take('small_polygon_best', small_polygon, 3)

    low_signal = [row for row in rows if row['signal_bin'] == 'low_signal']
    low_signal.sort(key=lambda row: row['mask_score'], reverse=True)
    take('low_signal_best', low_signal, 3)

    failures = [row for row in rows if row['sample_index'] not in selected_indices]
    failures.sort(key=lambda row: (row['mask_score'], row['iou'], -row['area_error']))
    take('exemplar_retrieval_failure', failures, 3)

    remaining = [row for row in rows if row['sample_index'] not in selected_indices]
    if remaining:
        median_score = float(np.median([row['mask_score'] for row in remaining]))
        ordinary = [row for row in remaining if row['area_bin'] == 'medium'] or remaining
        ordinary.sort(key=lambda row: abs(row['mask_score'] - median_score))
        take('ordinary_medium', ordinary, 3)
    return selected


def safe_name(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('_')


def generate_previews(selected, test_data, train_data, prob_maps, pred_masks, neighbor_indices, sample_rows, best_config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    row_by_index = {row['sample_index']: row for row in sample_rows}
    x_grid, y_grid = np.meshgrid(test_data['x'], test_data['y'])
    extent = [
        float(test_data['x'].min()),
        float(test_data['x'].max()),
        float(test_data['y'].min()),
        float(test_data['y'].max()),
    ]
    written = []
    for item in selected:
        idx = int(item['sample_index'])
        nn_idx = int(neighbor_indices[idx, 0])
        row = row_by_index[idx]
        true = test_data['masks'][idx]
        pred = pred_masks[idx]
        prob = prob_maps[idx]
        retrieved = train_data['masks'][nn_idx]
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), constrained_layout=True)
        fig.suptitle(
            f"{item['category']} | sample {idx} | nn={nn_idx} | "
            f"query={test_data['defect_types'][idx]} retrieved={train_data['defect_types'][nn_idx]} | "
            f"IoU={float(row['iou']):.3f}, Dice={float(row['dice']):.3f}, area_err={float(row['area_error']):.3f}",
            fontsize=10,
        )
        axes[0].imshow(true, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
        axes[0].set_title('query true mask')
        if best_config['retrieval_mode'] == 'top1':
            axes[1].imshow(retrieved, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
            axes[1].set_title('retrieved top1 mask')
        else:
            im = axes[1].imshow(prob, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
            axes[1].set_title('topK mean probability')
            fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
        axes[2].imshow(pred, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1)
        axes[2].set_title(f"pred mask ({best_config['retrieval_mode']}, thr={best_config['threshold']})")
        axes[3].imshow(true, origin='lower', cmap='gray', extent=extent, vmin=0, vmax=1, alpha=0.25)
        axes[3].contour(x_grid, y_grid, true.astype(float), levels=[0.5], colors=['lime'], linewidths=1.2)
        if pred.any():
            axes[3].contour(x_grid, y_grid, pred.astype(float), levels=[0.5], colors=['red'], linewidths=1.2)
        axes[3].set_title('overlay: true green, pred red')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        filename = f"{safe_name(item['category'])}_sample{idx:03d}_nn{nn_idx:03d}_{safe_name(test_data['defect_types'][idx])}.png"
        path = PREVIEW_DIR / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def main():
    ensure_outputs()
    train = load_npz(TRAIN_DATA)
    val = load_npz(VAL_DATA)
    test = load_npz(TEST_DATA)
    mask_rule = train['mask_rule']
    if val['mask_rule'] != mask_rule or test['mask_rule'] != mask_rule:
        raise ValueError('Inconsistent mask normalization rule across splits')

    validation_rows, best_config, _, feature_cache = validation_grid(train, val)
    write_validation_grid(validation_rows)

    transform, train_features = feature_cache[best_config['feature_representation']]
    test_features = transform_features(test['signals'], transform)
    test_neighbors, _ = topk_indices(train_features, test_features, best_config['distance'], k=5)
    test_area_edges = get_area_edges(test['masks'], test['x'], test['y'])
    test_low_signal = load_low_signal_indices_for_test()
    test_metric_rows, test_sample_rows, prob_maps, pred_masks = evaluate_config(
        train['masks'],
        test,
        test_neighbors,
        'test',
        test_area_edges,
        test_low_signal,
        best_config,
    )
    baseline_rows = read_current_baseline_rows()
    all_rows = baseline_rows + test_metric_rows
    write_metrics(all_rows)

    selected = select_preview_samples(test_sample_rows)
    preview_paths = generate_previews(
        selected,
        test,
        train,
        prob_maps,
        pred_masks,
        test_neighbors,
        test_sample_rows,
        best_config,
    )
    accepted = write_summary(all_rows, validation_rows, best_config, mask_rule, len(preview_paths))

    print(f"Selected config: {best_config}")
    print(f'Wrote validation grid: {VALIDATION_GRID_PATH}')
    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote previews: {PREVIEW_DIR} ({len(preview_paths)} png)')
    print(f'Accepted by metric gate: {accepted}')


if __name__ == '__main__':
    main()
