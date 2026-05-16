import csv
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_pinn import compute_sample_metrics, predict_batch_maps
from train_pinn import (
    MFLDataset,
    MU_SCALE,
    PINN,
    build_coord_grid,
    project_path,
    run_epoch,
    set_seed,
    signal_shape_info,
)


DATASET = 'v3_complex'
TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

SEEDS = [42, 123, 2026]
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
LATENT_DIM = 64
LAMBDA_TV = 2e-6
THRESHOLD = 500.0

CURRENT_BASELINE_CHECKPOINTS = {
    42: 'checkpoints/best_model_v3_complex_composite_seed42.pt',
    123: 'checkpoints/best_model_v3_complex_composite_seed123.pt',
    2026: 'checkpoints/best_model_v3_complex_composite_seed2026.pt',
}

CRITERIA = {
    'best_composite': 'composite',
    'best_macro_area_composite': 'macro_area_composite',
}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'macro_area_selection_audit'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_macro_area_selection_audit_metrics.csv'
EPOCH_LOG_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_macro_area_selection_epoch_log.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_macro_area_selection_audit_summary.txt'
SIGNAL_AUDIT_PATH = ROOT / 'results' / 'metrics' / 'v3_current_baseline_signal_difficulty_audit.csv'


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_loader(dataset, batch_size, shuffle=False, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def checkpoint_path(seed, criterion):
    return CHECKPOINT_DIR / f'seed_{seed}' / f'v3_complex_macro_area_selection_seed{seed}_{criterion}.pt'


def get_area_edges(dataset):
    masks = dataset.mu_maps < (500.0 / MU_SCALE)
    dx = float(abs(dataset.x[1] - dataset.x[0])) if len(dataset.x) > 1 else 1.0
    dy = float(abs(dataset.y[1] - dataset.y[0])) if len(dataset.y) > 1 else 1.0
    cell_area = dx * dy
    true_areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64) * cell_area
    return np.quantile(true_areas, [1 / 3, 2 / 3])


def area_bin(true_area, edges):
    if true_area <= edges[0]:
        return 'small'
    if true_area <= edges[1]:
        return 'medium'
    return 'large'


@torch.no_grad()
def evaluate_model(model, dataset, coords, device, threshold, area_edges=None, batch_size=8):
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    rows = []
    model.eval()
    for signals, mu_targets, indices in loader:
        pred_maps = predict_batch_maps(
            model=model,
            signals=signals,
            coords=coords,
            grid_shape=grid_shape,
            device=device,
            point_chunk=4096,
        )
        true_maps = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) * MU_SCALE
        for batch_idx, sample_idx_tensor in enumerate(indices):
            metrics, _, _ = compute_sample_metrics(
                pred_mu=pred_maps[batch_idx],
                true_mu=true_maps[batch_idx],
                x_grid=x_grid,
                y_grid=y_grid,
                threshold=threshold,
            )
            metrics['sample_index'] = int(sample_idx_tensor.item())
            if area_edges is not None:
                metrics['area_bin'] = area_bin(float(metrics['true_area']), area_edges)
            rows.append(metrics)
    return rows


def summarize_rows(rows):
    metric_names = ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']
    summary = {
        name: float(np.nanmean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    summary['n'] = int(len(rows))
    return summary


def summarize_bins(rows):
    bin_summaries = {}
    for name in ['small', 'medium', 'large']:
        selected = [row for row in rows if row.get('area_bin') == name]
        bin_summaries[name] = summarize_rows(selected)
    macro = float(np.mean([bin_summaries[name]['composite'] for name in ['small', 'medium', 'large']]))
    return bin_summaries, macro


def summarize_low_signal(rows, low_signal_indices):
    low = [row for row in rows if int(row['sample_index']) in low_signal_indices]
    non_low = [row for row in rows if int(row['sample_index']) not in low_signal_indices]
    return {
        'low_signal': summarize_rows(low),
        'non_low_signal': summarize_rows(non_low),
    }


def flatten_epoch_metrics(seed, epoch, overall, bins, macro):
    row = {
        'seed': seed,
        'epoch': epoch,
        'mse': overall['mse'],
        'mae': overall['mae'],
        'iou': overall['iou'],
        'dice': overall['dice'],
        'area_error': overall['area_error'],
        'center_error': overall['center_error'],
        'pred_area_zero': overall['pred_area_zero'],
        'pred_area_lt_true': overall['pred_area_lt_true'],
        'pred_area_gt_true': overall['pred_area_gt_true'],
        'composite': overall['composite'],
        'macro_area_composite': macro,
    }
    for name in ['small', 'medium', 'large']:
        prefix = f'{name}_'
        for key in ['iou', 'dice', 'area_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite']:
            row[prefix + key] = bins[name][key]
    return row


def save_checkpoint(model, seed, epoch, selected_metrics, train_dataset, criterion):
    path = checkpoint_path(seed, criterion)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': deepcopy(model.state_dict()),
        'args': {
            'dataset': DATASET,
            'latent_dim': LATENT_DIM,
            'model_variant': 'baseline',
            'decoder_variant': 'standard',
            'aux_mask_head': False,
            'signal_channels': int(getattr(model, 'signal_channels', 1)),
            'lambda_tv': LAMBDA_TV,
            'loss_type': 'mse',
            'seed': int(seed),
            'epochs': EPOCHS,
            'selection_criterion': criterion,
        },
        'epoch': int(epoch),
        'best_val_loss': float(selected_metrics['mse']),
        'selection_val_metrics': {key: float(value) for key, value in selected_metrics.items()},
        'signal_mean': float(train_dataset.signal_mean),
        'signal_std': float(train_dataset.signal_std),
    }
    torch.save(checkpoint, path)
    return path


def train_seed(seed, device):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    model = PINN(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
        model_variant='baseline',
        decoder_variant='standard',
        aux_mask_head=False,
    ).to(device)
    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    grid_shape = train_dataset.mu_maps.shape[1:]
    train_loader = make_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best = {
        name: {
            'value': -float('inf'),
            'epoch': None,
            'metrics': None,
            'checkpoint': None,
        }
        for name in CRITERIA
    }
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            coords=coords,
            optimizer=optimizer,
            criterion=criterion,
            lambda_tv=LAMBDA_TV,
            grid_shape=grid_shape,
            device=device,
            loss_type='mse',
        )
        val_rows = evaluate_model(model, val_dataset, coords, device, threshold=THRESHOLD, area_edges=val_area_edges)
        overall = summarize_rows(val_rows)
        bins, macro = summarize_bins(val_rows)
        epoch_metrics = flatten_epoch_metrics(seed, epoch, overall, bins, macro)
        epoch_rows.append(epoch_metrics)
        for criterion_name, metric_name in CRITERIA.items():
            value = epoch_metrics[metric_name]
            if value > best[criterion_name]['value']:
                selected_metrics = dict(epoch_metrics)
                path = save_checkpoint(model, seed, epoch, selected_metrics, train_dataset, criterion_name)
                best[criterion_name] = {
                    'value': value,
                    'epoch': epoch,
                    'metrics': selected_metrics,
                    'checkpoint': str(path.relative_to(ROOT)),
                }
        print(
            f"seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_mse={train_metrics['mse_loss']:.6e} | "
            f"val_composite={overall['composite']:.6e} | "
            f"val_macro_area_composite={macro:.6e} | "
            f"small_iou={bins['small']['iou']:.6e} | small_zero={bins['small']['pred_area_zero']}"
        )
    selected_rows = []
    for criterion_name, info in best.items():
        row = dict(info['metrics'])
        row.update({
            'seed': seed,
            'candidate': criterion_name,
            'selection_epoch': info['epoch'],
            'selection_score': info['value'],
            'checkpoint': info['checkpoint'],
        })
        selected_rows.append(row)
    return epoch_rows, selected_rows


def load_checkpoint_model(path, signal_length, signal_channels, device):
    checkpoint = torch.load(project_path(str(path)), map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    checkpoint_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
    model = PINN(
        signal_length=signal_length,
        signal_channels=int(checkpoint_args.get('signal_channels', signal_channels)),
        latent_dim=int(checkpoint_args.get('latent_dim', LATENT_DIM)),
        model_variant=checkpoint_args.get('model_variant', 'baseline'),
        decoder_variant=checkpoint_args.get('decoder_variant', 'standard'),
        aux_mask_head=bool(checkpoint_args.get('aux_mask_head', False)),
    ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if any(key.startswith('module.') for key in state_dict.keys()):
            stripped = {
                key[len('module.'):] if key.startswith('module.') else key: value
                for key, value in state_dict.items()
            }
            model.load_state_dict(stripped)
        else:
            raise
    model.eval()
    return model, checkpoint


def eval_candidate(seed, candidate, checkpoint_path_value, device, signal_length, signal_channels, test_area_edges, low_signal_indices):
    model, checkpoint = load_checkpoint_model(checkpoint_path_value, signal_length, signal_channels, device)
    dataset = MFLDataset(
        TEST_DATA,
        signal_mean=float(checkpoint.get('signal_mean', 0.0)),
        signal_std=float(checkpoint.get('signal_std', 1.0)),
    )
    coords = build_coord_grid(dataset.x, dataset.y).to(device)
    rows = evaluate_model(model, dataset, coords, device, threshold=THRESHOLD, area_edges=test_area_edges)
    overall = summarize_rows(rows)
    bins, macro = summarize_bins(rows)
    low = summarize_low_signal(rows, low_signal_indices)
    out = []
    for group_type, group, metrics in [('overall', 'all', overall)]:
        out.append(make_metrics_row(seed, candidate, group_type, group, metrics, macro))
    for group, metrics in bins.items():
        out.append(make_metrics_row(seed, candidate, 'area_bin', group, metrics, macro))
    for group, metrics in low.items():
        out.append(make_metrics_row(seed, candidate, 'signal_bin', group, metrics, macro))
    return out


def make_metrics_row(seed, candidate, group_type, group, metrics, macro):
    row = {
        'seed': seed,
        'candidate': candidate,
        'group_type': group_type,
        'group': group,
        'threshold': int(THRESHOLD),
        'macro_area_composite': macro,
    }
    for key in ['n', 'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error',
                'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite']:
        row[key] = metrics[key]
    return row


def load_low_signal_indices():
    if not SIGNAL_AUDIT_PATH.exists():
        return set()
    with open(SIGNAL_AUDIT_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    values = sorted(float(row['max_abs_bz']) for row in rows)
    threshold = values[66]
    return {int(row['sample_index']) for row in rows if float(row['max_abs_bz']) <= threshold}


def write_epoch_log(rows):
    fieldnames = [
        'seed', 'epoch', 'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error',
        'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite',
        'macro_area_composite',
    ]
    for name in ['small', 'medium', 'large']:
        for key in ['iou', 'dice', 'area_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite']:
            fieldnames.append(f'{name}_{key}')
    with open(EPOCH_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_metrics(rows):
    fieldnames = [
        'seed', 'candidate', 'group_type', 'group', 'threshold', 'n',
        'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error',
        'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true',
        'composite', 'macro_area_composite',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def mean_std(rows, candidate, group_type='overall', group='all'):
    selected = [
        row for row in rows
        if row['candidate'] == candidate and row['group_type'] == group_type and row['group'] == group
    ]
    result = {'candidate': candidate, 'group_type': group_type, 'group': group, 'n_seeds': len(selected)}
    for metric in [
        'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error',
        'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true',
        'composite', 'macro_area_composite',
    ]:
        values = np.array([float(row[metric]) for row in selected], dtype=np.float64)
        result[f'{metric}_mean'] = float(values.mean())
        result[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def paired_diff(rows, left, right, group_type='overall', group='all'):
    diffs = []
    for seed in SEEDS:
        left_row = next(row for row in rows if row['seed'] == seed and row['candidate'] == left and row['group_type'] == group_type and row['group'] == group)
        right_row = next(row for row in rows if row['seed'] == seed and row['candidate'] == right and row['group_type'] == group_type and row['group'] == group)
        diff = {'seed': seed}
        for metric in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'composite', 'macro_area_composite']:
            diff[metric] = float(left_row[metric]) - float(right_row[metric])
        diffs.append(diff)
    return {metric: float(np.mean([row[metric] for row in diffs])) for metric in diffs[0] if metric != 'seed'}


def format_mean_std_table(stats):
    lines = [
        '| candidate | group | MSE | MAE | IoU | Dice | area_error | center_error | pred_area=0 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in stats:
        lines.append(
            f"| {row['candidate']} | {row['group']} | "
            f"{row['mse_mean']:.4e} +/- {row['mse_std']:.2e} | "
            f"{row['mae_mean']:.4e} +/- {row['mae_std']:.2e} | "
            f"{row['iou_mean']:.4f} +/- {row['iou_std']:.4f} | "
            f"{row['dice_mean']:.4f} +/- {row['dice_std']:.4f} | "
            f"{row['area_error_mean']:.4f} +/- {row['area_error_std']:.4f} | "
            f"{row['center_error_mean']:.4f} +/- {row['center_error_std']:.4f} | "
            f"{row['pred_area_zero_mean']:.2f} +/- {row['pred_area_zero_std']:.2f} |"
        )
    return '\n'.join(lines)


def write_summary(selected_rows, metrics_rows):
    selection_lines = [
        '| seed | best_composite epoch | best_composite score | best_macro_area_composite epoch | best_macro_area_composite score |',
        '|---:|---:|---:|---:|---:|',
    ]
    for seed in SEEDS:
        best_comp = next(row for row in selected_rows if row['seed'] == seed and row['candidate'] == 'best_composite')
        best_macro = next(row for row in selected_rows if row['seed'] == seed and row['candidate'] == 'best_macro_area_composite')
        selection_lines.append(
            f"| {seed} | {best_comp['selection_epoch']} | {best_comp['selection_score']:.6e} | "
            f"{best_macro['selection_epoch']} | {best_macro['selection_score']:.6e} |"
        )

    candidates = ['current_baseline_composite', 'best_composite', 'best_macro_area_composite']
    overall_stats = [mean_std(metrics_rows, candidate) for candidate in candidates]
    area_stats = []
    for group in ['small', 'medium', 'large']:
        for candidate in candidates:
            area_stats.append(mean_std(metrics_rows, candidate, 'area_bin', group))
    signal_stats = []
    for group in ['low_signal', 'non_low_signal']:
        for candidate in candidates:
            signal_stats.append(mean_std(metrics_rows, candidate, 'signal_bin', group))

    macro_vs_current = paired_diff(metrics_rows, 'best_macro_area_composite', 'current_baseline_composite')
    macro_vs_best_comp = paired_diff(metrics_rows, 'best_macro_area_composite', 'best_composite')
    small_macro = mean_std(metrics_rows, 'best_macro_area_composite', 'area_bin', 'small')
    small_current = mean_std(metrics_rows, 'current_baseline_composite', 'area_bin', 'small')
    overall_macro = mean_std(metrics_rows, 'best_macro_area_composite')
    overall_current = mean_std(metrics_rows, 'current_baseline_composite')

    small_improved = (
        small_macro['iou_mean'] > small_current['iou_mean']
        and small_macro['dice_mean'] > small_current['dice_mean']
        and small_macro['pred_area_zero_mean'] <= small_current['pred_area_zero_mean']
    )
    overall_not_worse = (
        overall_macro['iou_mean'] >= overall_current['iou_mean'] - 0.01
        and overall_macro['dice_mean'] >= overall_current['dice_mean'] - 0.01
        and overall_macro['area_error_mean'] <= overall_current['area_error_mean'] + 0.02
    )
    mse_mae_cost_ok = (
        overall_macro['mse_mean'] <= overall_current['mse_mean'] * 1.05
        and overall_macro['mae_mean'] <= overall_current['mae_mean'] * 1.05
    )
    accepted = small_improved and overall_not_worse and mse_mae_cost_ok

    summary = f"""# v3_complex area-bin-balanced composite selection audit

This controlled diagnostic trains v3_complex baseline runs for seeds 42, 123, and 2026 with MSE loss, lambda_tv=2e-6, epochs=50, and threshold=500 evaluation. It does not modify train_pinn.py, evaluate_pinn.py, or data_generator_v2.py.

Validation area bins are defined only from validation-set true_area tertiles. Test-set bins are used only for reporting.

Definitions:

* overall composite = IoU + Dice - area_error
* macro_area_composite = mean over validation small / medium / large bins of (IoU_bin + Dice_bin - area_error_bin)

## Selection epochs

{chr(10).join(selection_lines)}

## Overall test mean +/- sample std

{format_mean_std_table(overall_stats)}

## Area-bin test mean +/- sample std

{format_mean_std_table(area_stats)}

## Low-signal test mean +/- sample std

{format_mean_std_table(signal_stats)}

## Paired differences

best_macro_area_composite - current_baseline_composite:

* ΔMSE = {macro_vs_current['mse']:.6e}
* ΔMAE = {macro_vs_current['mae']:.6e}
* ΔIoU = {macro_vs_current['iou']:.6e}
* ΔDice = {macro_vs_current['dice']:.6e}
* Δarea_error = {macro_vs_current['area_error']:.6e}
* Δpred_area=0 = {macro_vs_current['pred_area_zero']:.6e}

best_macro_area_composite - best_composite:

* ΔMSE = {macro_vs_best_comp['mse']:.6e}
* ΔMAE = {macro_vs_best_comp['mae']:.6e}
* ΔIoU = {macro_vs_best_comp['iou']:.6e}
* ΔDice = {macro_vs_best_comp['dice']:.6e}
* Δarea_error = {macro_vs_best_comp['area_error']:.6e}
* Δpred_area=0 = {macro_vs_best_comp['pred_area_zero']:.6e}

## Judgment

Small-bin improved versus current baseline without more empty predictions: {small_improved}.

Overall shape metrics not meaningfully worse than current baseline: {overall_not_worse}.

MSE / MAE cost within 5% of current composite-selection baseline: {mse_mae_cost_ok}.

Accepted as macro-area selection signal: {accepted}.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'accepted': accepted,
        'overall_stats': overall_stats,
        'area_stats': area_stats,
        'signal_stats': signal_stats,
        'macro_vs_current': macro_vs_current,
        'macro_vs_best_comp': macro_vs_best_comp,
    }


def main():
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    all_epoch_rows = []
    selected_rows = []
    for seed in SEEDS:
        print(f'Training seed={seed}')
        epoch_rows, seed_selected_rows = train_seed(seed, device)
        all_epoch_rows.extend(epoch_rows)
        selected_rows.extend(seed_selected_rows)

    raw_test = np.load(project_path(TEST_DATA), allow_pickle=False)
    signal_length, signal_channels = signal_shape_info(raw_test['signals'])
    test_dataset_for_bins = MFLDataset(TEST_DATA)
    test_area_edges = get_area_edges(test_dataset_for_bins)
    low_signal_indices = load_low_signal_indices()

    metrics_rows = []
    for seed, checkpoint in CURRENT_BASELINE_CHECKPOINTS.items():
        metrics_rows.extend(eval_candidate(
            seed=seed,
            candidate='current_baseline_composite',
            checkpoint_path_value=checkpoint,
            device=device,
            signal_length=signal_length,
            signal_channels=signal_channels,
            test_area_edges=test_area_edges,
            low_signal_indices=low_signal_indices,
        ))
    for selected in selected_rows:
        metrics_rows.extend(eval_candidate(
            seed=selected['seed'],
            candidate=selected['candidate'],
            checkpoint_path_value=ROOT / selected['checkpoint'],
            device=device,
            signal_length=signal_length,
            signal_channels=signal_channels,
            test_area_edges=test_area_edges,
            low_signal_indices=low_signal_indices,
        ))

    write_epoch_log(all_epoch_rows)
    write_metrics(metrics_rows)
    judgment = write_summary(selected_rows, metrics_rows)

    print(f'Wrote epoch log: {EPOCH_LOG_PATH}')
    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f"Accepted: {judgment['accepted']}")


if __name__ == '__main__':
    main()
