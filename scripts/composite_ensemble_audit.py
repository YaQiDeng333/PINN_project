import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_pinn import compute_sample_metrics, load_model, predict_batch_maps
from train_pinn import MFLDataset, MU_SCALE, build_coord_grid, project_path, signal_shape_info


TEST_DATA = 'data/training_data_v3_complex_test.npz'
THRESHOLD = 500.0
BATCH_SIZE = 8
POINT_CHUNK = 4096

CHECKPOINTS = {
    42: 'checkpoints/best_model_v3_complex_composite_seed42.pt',
    123: 'checkpoints/best_model_v3_complex_composite_seed123.pt',
    2026: 'checkpoints/best_model_v3_complex_composite_seed2026.pt',
}

METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_composite_ensemble_gate_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_composite_ensemble_gate_summary.txt'
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'

METRIC_KEYS = [
    'mse',
    'mae',
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


def ensure_outputs():
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def check_required_checkpoints():
    missing = [path for path in CHECKPOINTS.values() if not Path(project_path(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing composite checkpoints: ' + ', '.join(missing))


def make_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def get_area_edges(dataset):
    masks = dataset.mu_maps < (THRESHOLD / MU_SCALE)
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


@torch.no_grad()
def predict_checkpoint(checkpoint_path, signal_length, device):
    model, signal_mean, signal_std, _ = load_model(checkpoint_path, signal_length, device)
    dataset = MFLDataset(TEST_DATA, signal_mean=signal_mean, signal_std=signal_std)
    coords = build_coord_grid(dataset.x, dataset.y).to(device)
    grid_shape = dataset.mu_maps.shape[1:]
    pred_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)

    for signals, mu_targets, indices in make_loader(dataset):
        batch_pred = predict_batch_maps(
            model=model,
            signals=signals,
            coords=coords,
            grid_shape=grid_shape,
            device=device,
            point_chunk=POINT_CHUNK,
        )
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) * MU_SCALE
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            pred_maps[sample_idx] = batch_pred[batch_pos]
            true_maps[sample_idx] = batch_true[batch_pos]

    return pred_maps, true_maps, dataset


def build_sample_rows(candidate, seed, pred_maps, true_maps, dataset, area_edges, low_signal_indices):
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    rows = []
    for sample_idx in range(len(dataset)):
        metrics, _, _ = compute_sample_metrics(
            pred_mu=pred_maps[sample_idx],
            true_mu=true_maps[sample_idx],
            x_grid=x_grid,
            y_grid=y_grid,
            threshold=THRESHOLD,
        )
        row = {
            'candidate': candidate,
            'seed': seed,
            'sample_index': sample_idx,
            'defect_type': str(dataset.defect_types[sample_idx]),
            'area_bin': area_bin(float(metrics['true_area']), area_edges),
            'signal_bin': 'low_signal' if sample_idx in low_signal_indices else 'non_low_signal',
        }
        row.update(metrics)
        rows.append(row)
    return rows


def summarize_samples(rows):
    summary = {'n': len(rows)}
    if not rows:
        for key in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']:
            summary[key] = float('nan')
        summary.update({
            'pred_area_zero': 0,
            'pred_area_lt_true': 0,
            'pred_area_gt_true': 0,
            'composite': float('nan'),
        })
        return summary

    for key in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']:
        summary[key] = float(np.nanmean([float(row[key]) for row in rows]))
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    return summary


def metric_row(candidate, seed, group_type, group, summary, macro_area_composite):
    row = {
        'candidate': candidate,
        'seed': seed,
        'group_type': group_type,
        'group': group,
        'threshold': int(THRESHOLD),
        'n': summary['n'],
    }
    for key in [
        'mse',
        'mae',
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


def summarize_candidate(sample_rows, candidate, seed):
    rows = []
    overall = summarize_samples(sample_rows)
    area_summaries = {
        group: summarize_samples([row for row in sample_rows if row['area_bin'] == group])
        for group in ['small', 'medium', 'large']
    }
    macro_area_composite = float(np.nanmean([
        area_summaries[group]['composite']
        for group in ['small', 'medium', 'large']
    ]))

    rows.append(metric_row(candidate, seed, 'overall', 'all', overall, macro_area_composite))
    for group in ['small', 'medium', 'large']:
        rows.append(metric_row(candidate, seed, 'area_bin', group, area_summaries[group], macro_area_composite))
    for group in ['low_signal', 'non_low_signal']:
        selected = [row for row in sample_rows if row['signal_bin'] == group]
        rows.append(metric_row(candidate, seed, 'signal_bin', group, summarize_samples(selected), macro_area_composite))
    return rows


def aggregate_seed_rows(metric_rows, source_candidate):
    aggregate_rows = []
    groups = sorted({
        (row['group_type'], row['group'])
        for row in metric_rows
        if row['candidate'] == source_candidate
    })
    for group_type, group in groups:
        selected = [
            row for row in metric_rows
            if row['candidate'] == source_candidate and row['group_type'] == group_type and row['group'] == group
        ]
        mean_row = {
            'candidate': f'{source_candidate}_mean',
            'seed': 'mean',
            'group_type': group_type,
            'group': group,
            'threshold': int(THRESHOLD),
            'n': selected[0]['n'] if selected else 0,
        }
        std_row = dict(mean_row)
        std_row['candidate'] = f'{source_candidate}_std'
        std_row['seed'] = 'sample_std'
        for key in METRIC_KEYS:
            values = np.array([float(row[key]) for row in selected], dtype=np.float64)
            mean_row[key] = float(np.nanmean(values))
            std_row[key] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.extend([mean_row, std_row])
    return aggregate_rows


def find_row(rows, candidate, group_type='overall', group='all'):
    return next(
        row for row in rows
        if row['candidate'] == candidate and row['group_type'] == group_type and row['group'] == group
    )


def format_value(value, metric):
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    if metric in ('mse', 'mae'):
        return f'{float(value):.4e}'
    return f'{float(value):.4f}'


def format_baseline_metric(mean_row, std_row, metric):
    return f"{format_value(mean_row[metric], metric)} +/- {format_value(std_row[metric], metric)}"


def format_comparison_table(rows, group_type, groups):
    lines = [
        '| group | candidate | MSE | MAE | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    base_mean_candidate = 'current_baseline_composite_mean'
    base_std_candidate = 'current_baseline_composite_std'
    for group in groups:
        base_mean = find_row(rows, base_mean_candidate, group_type, group)
        base_std = find_row(rows, base_std_candidate, group_type, group)
        ensemble = find_row(rows, 'mean_mu_ensemble', group_type, group)
        lines.append(
            f"| {group} | current baseline 3-seed mean | "
            f"{format_baseline_metric(base_mean, base_std, 'mse')} | "
            f"{format_baseline_metric(base_mean, base_std, 'mae')} | "
            f"{format_baseline_metric(base_mean, base_std, 'iou')} | "
            f"{format_baseline_metric(base_mean, base_std, 'dice')} | "
            f"{format_baseline_metric(base_mean, base_std, 'area_error')} | "
            f"{format_baseline_metric(base_mean, base_std, 'center_error')} | "
            f"{format_baseline_metric(base_mean, base_std, 'pred_area_zero')} | "
            f"{format_baseline_metric(base_mean, base_std, 'pred_area_lt_true')} | "
            f"{format_baseline_metric(base_mean, base_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | mean_mu ensemble | "
            f"{format_value(ensemble['mse'], 'mse')} | "
            f"{format_value(ensemble['mae'], 'mae')} | "
            f"{format_value(ensemble['iou'], 'iou')} | "
            f"{format_value(ensemble['dice'], 'dice')} | "
            f"{format_value(ensemble['area_error'], 'area_error')} | "
            f"{format_value(ensemble['center_error'], 'center_error')} | "
            f"{format_value(ensemble['pred_area_zero'], 'pred_area_zero')} | "
            f"{format_value(ensemble['pred_area_lt_true'], 'pred_area_lt_true')} | "
            f"{format_value(ensemble['pred_area_gt_true'], 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group):
    baseline = find_row(rows, 'current_baseline_composite_mean', group_type, group)
    ensemble = find_row(rows, 'mean_mu_ensemble', group_type, group)
    return {
        'iou_up': float(ensemble['iou']) > float(baseline['iou']),
        'dice_up': float(ensemble['dice']) > float(baseline['dice']),
        'area_error_not_worse': float(ensemble['area_error']) <= float(baseline['area_error']),
        'pred_area_zero_not_worse': float(ensemble['pred_area_zero']) <= float(baseline['pred_area_zero']),
        'mse_cost_ok': float(ensemble['mse']) <= float(baseline['mse']) * 1.05,
        'mae_cost_ok': float(ensemble['mae']) <= float(baseline['mae']) * 1.05,
    }


def status_line(name, status):
    return (
        f"* {name}: IoU up={status['iou_up']}, Dice up={status['dice_up']}, "
        f"area_error not worse={status['area_error_not_worse']}, "
        f"pred_area=0 not worse={status['pred_area_zero_not_worse']}"
    )


def write_metrics(rows):
    fieldnames = [
        'candidate',
        'seed',
        'group_type',
        'group',
        'threshold',
        'n',
        'mse',
        'mae',
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


def write_summary(rows, checkpoints_found):
    overall = improvement_status(rows, 'overall', 'all')
    small = improvement_status(rows, 'area_bin', 'small')
    medium = improvement_status(rows, 'area_bin', 'medium')
    large = improvement_status(rows, 'area_bin', 'large')
    low_signal = improvement_status(rows, 'signal_bin', 'low_signal')

    overall_accept = (
        overall['iou_up']
        and overall['dice_up']
        and overall['area_error_not_worse']
        and overall['pred_area_zero_not_worse']
        and overall['mse_cost_ok']
        and overall['mae_cost_ok']
    )
    small_low_ok = (
        small['iou_up']
        and small['dice_up']
        and small['area_error_not_worse']
        and small['pred_area_zero_not_worse']
        and low_signal['iou_up']
        and low_signal['dice_up']
        and low_signal['area_error_not_worse']
        and low_signal['pred_area_zero_not_worse']
    )
    accepted = bool(overall_accept and small_low_ok)
    keep_candidate_record = accepted

    summary = f"""# v3_complex composite-selection checkpoint ensemble gate

This FAST_ANALYSIS_GATE evaluates only one checkpoint ensemble: pixelwise mean of physical-scale pred_mu maps from the three current composite-selection checkpoints. It does not train models, tune thresholds, modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, or update CURRENT_BASELINE.md.

Checkpoint set found: {checkpoints_found}

Checkpoints:

* {CHECKPOINTS[42]}
* {CHECKPOINTS[123]}
* {CHECKPOINTS[2026]}

Dataset: {TEST_DATA}

Mask rule: mean_mu < {THRESHOLD:.0f}

## Overall comparison

{format_comparison_table(rows, 'overall', ['all'])}

## Area-bin comparison

{format_comparison_table(rows, 'area_bin', ['small', 'medium', 'large'])}

## Low-signal comparison

{format_comparison_table(rows, 'signal_bin', ['low_signal', 'non_low_signal'])}

## Gate checks

{status_line('overall', overall)}

{status_line('small', small)}

{status_line('medium', medium)}

{status_line('large', large)}

{status_line('low_signal', low_signal)}

* MSE cost within 5%: {overall['mse_cost_ok']}
* MAE cost within 5%: {overall['mae_cost_ok']}
* Accepted by this gate: {accepted}
* Keep mean_mu ensemble as candidate record: {keep_candidate_record}
* Keep this gate output as diagnostic record: True

No majority vote, union mask, min_mu, threshold sweep, adaptive threshold, or other ensemble variant was tested.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'accepted': accepted,
        'keep_candidate_record': keep_candidate_record,
        'overall': overall,
        'small': small,
        'medium': medium,
        'large': large,
        'low_signal': low_signal,
    }


def main():
    ensure_outputs()
    check_required_checkpoints()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_test = np.load(project_path(TEST_DATA), allow_pickle=False)
    signal_length, _ = signal_shape_info(raw_test['signals'])
    low_signal_indices = load_low_signal_indices()

    seed_sample_rows = []
    seed_metric_rows = []
    pred_maps_by_seed = []
    reference_true_maps = None
    reference_dataset = None
    area_edges = None

    print(f'Using device: {device}')
    for seed, checkpoint_path in CHECKPOINTS.items():
        print(f'Loading checkpoint seed={seed}: {checkpoint_path}')
        pred_maps, true_maps, dataset = predict_checkpoint(checkpoint_path, signal_length, device)
        if reference_true_maps is None:
            reference_true_maps = true_maps
            reference_dataset = dataset
            area_edges = get_area_edges(dataset)
        pred_maps_by_seed.append(pred_maps)
        sample_rows = build_sample_rows(
            candidate='current_baseline_composite',
            seed=seed,
            pred_maps=pred_maps,
            true_maps=true_maps,
            dataset=dataset,
            area_edges=area_edges,
            low_signal_indices=low_signal_indices,
        )
        seed_sample_rows.extend(sample_rows)
        seed_metric_rows.extend(summarize_candidate(sample_rows, 'current_baseline_composite', seed))

    mean_mu = np.mean(np.stack(pred_maps_by_seed, axis=0), axis=0)
    ensemble_sample_rows = build_sample_rows(
        candidate='mean_mu_ensemble',
        seed='ensemble',
        pred_maps=mean_mu,
        true_maps=reference_true_maps,
        dataset=reference_dataset,
        area_edges=area_edges,
        low_signal_indices=low_signal_indices,
    )
    ensemble_metric_rows = summarize_candidate(ensemble_sample_rows, 'mean_mu_ensemble', 'ensemble')
    aggregate_rows = aggregate_seed_rows(seed_metric_rows, 'current_baseline_composite')
    all_metric_rows = seed_metric_rows + aggregate_rows + ensemble_metric_rows

    write_metrics(all_metric_rows)
    judgment = write_summary(all_metric_rows, checkpoints_found=True)

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f"Accepted: {judgment['accepted']}")
    print(f"Keep candidate record: {judgment['keep_candidate_record']}")


if __name__ == '__main__':
    main()
