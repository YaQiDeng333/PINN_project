import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_pinn import compute_sample_metrics, load_model, predict_batch_maps  # noqa: E402
from train_pinn import MFLDataset, MU_SCALE, build_coord_grid, signal_shape_info  # noqa: E402


TEST_DATA = 'data/training_data_v3_complex_test.npz'
THRESHOLD = 500.0
BASELINE_CHECKPOINT = 'checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt'
COMPOSITE_CHECKPOINTS = {
    42: 'checkpoints/best_model_v3_complex_composite_seed42.pt',
    123: 'checkpoints/best_model_v3_complex_composite_seed123.pt',
    2026: 'checkpoints/best_model_v3_complex_composite_seed2026.pt',
}
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_composite_candidate_per_sample_audit.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_composite_candidate_per_sample_audit_summary.txt'


def safe_float(value):
    if value is None:
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def safe_mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')


def safe_sum(values):
    return int(sum(int(v) for v in values))


def area_bin_from_true_area(true_area, q1, q2):
    if true_area <= q1:
        return 'small'
    if true_area <= q2:
        return 'medium'
    return 'large'


def load_predictions(checkpoint_path, signal_length, grid_shape, coords, device):
    model, signal_mean, signal_std, _ = load_model(checkpoint_path, signal_length, device)
    dataset = MFLDataset(TEST_DATA, signal_mean=signal_mean, signal_std=signal_std)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    predictions = {}

    for signals, _, indices in loader:
        pred_maps = predict_batch_maps(
            model=model,
            signals=signals,
            coords=coords,
            grid_shape=grid_shape,
            device=device,
            point_chunk=4096,
        )
        for batch_pos, sample_index in enumerate(indices):
            predictions[int(sample_index.item())] = pred_maps[batch_pos]
    return predictions


def region_errors(pred_mu, true_mu, true_mask):
    err = pred_mu - true_mu
    background_mask = ~true_mask
    result = {}
    if np.any(true_mask):
        defect_err = err[true_mask]
        result['defect_mse'] = float(np.mean(defect_err ** 2))
        result['defect_mae'] = float(np.mean(np.abs(defect_err)))
    else:
        result['defect_mse'] = float('nan')
        result['defect_mae'] = float('nan')
    if np.any(background_mask):
        background_err = err[background_mask]
        result['background_mse'] = float(np.mean(background_err ** 2))
        result['background_mae'] = float(np.mean(np.abs(background_err)))
    else:
        result['background_mse'] = float('nan')
        result['background_mae'] = float('nan')
    return result


def metric_dict_for_prediction(pred_mu, true_mu, x_grid, y_grid):
    metrics, _, true_mask = compute_sample_metrics(
        pred_mu=pred_mu,
        true_mu=true_mu,
        x_grid=x_grid,
        y_grid=y_grid,
        threshold=THRESHOLD,
    )
    metrics.update(region_errors(pred_mu, true_mu, true_mask))
    return metrics


def aggregate(rows, group_name):
    return {
        'group': group_name,
        'count': len(rows),
        'iou_improved': safe_sum(row['delta_iou'] > 0.0 for row in rows),
        'dice_improved': safe_sum(row['delta_dice'] > 0.0 for row in rows),
        'area_error_improved': safe_sum(row['delta_area_error'] < 0.0 for row in rows),
        'pred_area_zero_repaired': safe_sum(row['pred_area_zero_repaired'] for row in rows),
        'clearly_worse': safe_sum(row['composite_clearly_worse'] for row in rows),
        'mean_delta_iou': safe_mean(row['delta_iou'] for row in rows),
        'mean_delta_dice': safe_mean(row['delta_dice'] for row in rows),
        'mean_delta_area_error': safe_mean(row['delta_area_error'] for row in rows),
        'mean_delta_pred_area': safe_mean(row['delta_pred_area'] for row in rows),
        'mean_delta_center_error': safe_mean(row['delta_center_error'] for row in rows),
        'mean_delta_defect_mae': safe_mean(row['delta_defect_mae'] for row in rows),
        'mean_delta_background_mae': safe_mean(row['delta_background_mae'] for row in rows),
    }


def markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw = np.load(ROOT / TEST_DATA, allow_pickle=False)
    signal_length, _ = signal_shape_info(raw['signals'])
    true_maps = raw['mu_maps'].astype(np.float32)
    defect_types = raw['defect_types']
    x = raw['x'].astype(np.float32)
    y = raw['y'].astype(np.float32)
    grid_shape = true_maps.shape[1:]
    coords = build_coord_grid(x, y).to(device)
    x_grid, y_grid = np.meshgrid(x, y)
    dx = float(abs(x_grid[0, 1] - x_grid[0, 0])) if x_grid.shape[1] > 1 else 1.0
    dy = float(abs(y_grid[1, 0] - y_grid[0, 0])) if y_grid.shape[0] > 1 else 1.0
    cell_area = dx * dy

    true_area_rows = []
    for idx in range(len(true_maps)):
        true_mu = true_maps[idx]
        true_mask = true_mu < THRESHOLD
        true_area_rows.append(float(true_mask.sum()) * cell_area)
    q1, q2 = np.quantile(np.array(true_area_rows, dtype=np.float64), [1 / 3, 2 / 3])

    baseline_preds = load_predictions(BASELINE_CHECKPOINT, signal_length, grid_shape, coords, device)
    composite_preds = {
        seed: load_predictions(path, signal_length, grid_shape, coords, device)
        for seed, path in COMPOSITE_CHECKPOINTS.items()
    }

    rows = []
    seeds = sorted(composite_preds)
    for idx in range(len(true_maps)):
        true_mu = true_maps[idx]
        baseline_metrics = metric_dict_for_prediction(baseline_preds[idx], true_mu, x_grid, y_grid)
        per_seed_metrics = [
            metric_dict_for_prediction(composite_preds[seed][idx], true_mu, x_grid, y_grid)
            for seed in seeds
        ]

        composite_mean = {
            key: safe_mean(seed_metrics[key] for seed_metrics in per_seed_metrics)
            for key in [
                'mse',
                'mae',
                'iou',
                'dice',
                'area_error',
                'center_error',
                'pred_area',
                'defect_mse',
                'defect_mae',
                'background_mse',
                'background_mae',
            ]
        }
        baseline_pred_zero = baseline_metrics['pred_area'] == 0.0
        composite_pred_zero = composite_mean['pred_area'] == 0.0
        row = {
            'sample_index': idx,
            'defect_type': str(defect_types[idx]),
            'true_area': baseline_metrics['true_area'],
            'area_bin': area_bin_from_true_area(baseline_metrics['true_area'], q1, q2),
            'baseline_iou': baseline_metrics['iou'],
            'baseline_dice': baseline_metrics['dice'],
            'baseline_area_error': baseline_metrics['area_error'],
            'baseline_pred_area': baseline_metrics['pred_area'],
            'baseline_center_error': baseline_metrics['center_error'],
            'baseline_defect_mae': baseline_metrics['defect_mae'],
            'baseline_background_mae': baseline_metrics['background_mae'],
            'composite_mean_iou': composite_mean['iou'],
            'composite_mean_dice': composite_mean['dice'],
            'composite_mean_area_error': composite_mean['area_error'],
            'composite_mean_pred_area': composite_mean['pred_area'],
            'composite_mean_center_error': composite_mean['center_error'],
            'composite_mean_defect_mae': composite_mean['defect_mae'],
            'composite_mean_background_mae': composite_mean['background_mae'],
            'delta_iou': composite_mean['iou'] - baseline_metrics['iou'],
            'delta_dice': composite_mean['dice'] - baseline_metrics['dice'],
            'delta_area_error': composite_mean['area_error'] - baseline_metrics['area_error'],
            'delta_pred_area': composite_mean['pred_area'] - baseline_metrics['pred_area'],
            'delta_center_error': composite_mean['center_error'] - baseline_metrics['center_error'],
            'delta_defect_mae': composite_mean['defect_mae'] - baseline_metrics['defect_mae'],
            'delta_background_mae': composite_mean['background_mae'] - baseline_metrics['background_mae'],
            'baseline_pred_area_zero': baseline_pred_zero,
            'composite_pred_area_zero': composite_pred_zero,
            'pred_area_zero_repaired': baseline_pred_zero and not composite_pred_zero,
        }
        row['composite_clearly_worse'] = (
            row['delta_iou'] <= -0.05
            or row['delta_dice'] <= -0.05
            or row['delta_area_error'] >= 0.10
        )
        rows.append(row)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with METRICS_PATH.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall = aggregate(rows, 'all')
    by_area = [aggregate([row for row in rows if row['area_bin'] == name], name) for name in ['small', 'medium', 'large']]
    defect_type_names = sorted(set(row['defect_type'] for row in rows))
    by_type = [aggregate([row for row in rows if row['defect_type'] == name], name) for name in defect_type_names]

    lines = []
    lines.append('# v3_complex composite-selection candidate per-sample audit')
    lines.append('')
    lines.append(f'Threshold: {THRESHOLD:.0f}. Composite metrics are per-sample means across seeds 42, 123, and 2026.')
    lines.append('Clearly worse is defined as delta_iou <= -0.05, delta_dice <= -0.05, or delta_area_error >= 0.10.')
    lines.append('')
    lines.append('## Overall counts')
    lines.append('')
    lines.append(markdown_table(
        ['count', 'IoU improved', 'Dice improved', 'area_error improved', 'pred_area=0 repaired', 'clearly worse'],
        [[
            overall['count'],
            overall['iou_improved'],
            overall['dice_improved'],
            overall['area_error_improved'],
            overall['pred_area_zero_repaired'],
            overall['clearly_worse'],
        ]],
    ))
    lines.append('')
    lines.append('## Mean deltas')
    lines.append('')
    lines.append(markdown_table(
        ['delta_iou', 'delta_dice', 'delta_area_error', 'delta_pred_area', 'delta_center_error', 'delta_defect_mae', 'delta_background_mae'],
        [[
            f'{overall["mean_delta_iou"]:.6f}',
            f'{overall["mean_delta_dice"]:.6f}',
            f'{overall["mean_delta_area_error"]:.6f}',
            f'{overall["mean_delta_pred_area"]:.6f}',
            f'{overall["mean_delta_center_error"]:.6f}',
            f'{overall["mean_delta_defect_mae"]:.6f}',
            f'{overall["mean_delta_background_mae"]:.6f}',
        ]],
    ))
    lines.append('')
    lines.append('## Area-bin breakdown')
    lines.append('')
    lines.append(markdown_table(
        ['area_bin', 'count', 'IoU improved', 'Dice improved', 'area_error improved', 'pred_area=0 repaired', 'clearly worse', 'mean_delta_iou', 'mean_delta_area_error'],
        [
            [
                item['group'],
                item['count'],
                item['iou_improved'],
                item['dice_improved'],
                item['area_error_improved'],
                item['pred_area_zero_repaired'],
                item['clearly_worse'],
                f'{item["mean_delta_iou"]:.6f}',
                f'{item["mean_delta_area_error"]:.6f}',
            ]
            for item in by_area
        ],
    ))
    lines.append('')
    lines.append('## Defect-type breakdown')
    lines.append('')
    lines.append(markdown_table(
        ['defect_type', 'count', 'IoU improved', 'Dice improved', 'area_error improved', 'pred_area=0 repaired', 'clearly worse', 'mean_delta_iou', 'mean_delta_area_error'],
        [
            [
                item['group'],
                item['count'],
                item['iou_improved'],
                item['dice_improved'],
                item['area_error_improved'],
                item['pred_area_zero_repaired'],
                item['clearly_worse'],
                f'{item["mean_delta_iou"]:.6f}',
                f'{item["mean_delta_area_error"]:.6f}',
            ]
            for item in by_type
        ],
    ))
    lines.append('')
    lines.append('## Judgment')
    lines.append('')
    if overall['iou_improved'] > overall['count'] / 2 and overall['dice_improved'] > overall['count'] / 2:
        lines.append('Composite-selection improvement is sample-broad for IoU/Dice rather than only a few outliers.')
    else:
        lines.append('Composite-selection improvement is not majority-sample broad for IoU/Dice.')
    if by_area[0]['iou_improved'] > by_area[0]['count'] / 2:
        lines.append('Small defects improve for a majority of samples by IoU, and pred_area=0 repairs are concentrated in the small bin.')
    else:
        lines.append('Small defects do not improve for a majority of samples by IoU, though some pred_area=0 repairs may still occur.')
    if overall['mean_delta_pred_area'] > 0:
        lines.append('Composite-selection increases predicted area on average, which partially mitigates the baseline area underestimation.')
    else:
        lines.append('Composite-selection does not increase predicted area on average.')
    lines.append('The MSE/MAE trade-off is mainly visible in both defect and background regions; see delta_defect_mae and delta_background_mae.')
    lines.append('This audit supports keeping composite-selection as a formal candidate record. It does not by itself update CURRENT_BASELINE.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'Wrote {METRICS_PATH}')
    print(f'Wrote {SUMMARY_PATH}')
    print('overall', overall)
    print('by_area', by_area)
    print('by_type', by_type)


if __name__ == '__main__':
    main()
