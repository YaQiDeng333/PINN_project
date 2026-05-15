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
CURRENT_BASELINE = 'checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt'

SEEDS = [42, 123, 2026]
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
LATENT_DIM = 64
LAMBDA_TV = 2e-6
THRESHOLDS = [500.0, 600.0]

CRITERIA = {
    'best_val_mse': ('min', 'mse'),
    'best_val_iou': ('max', 'iou'),
    'best_val_dice': ('max', 'dice'),
    'best_val_area_error': ('min', 'area_error'),
    'best_composite': ('max', 'composite'),
}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'model_selection_3seed'
SEED42_REUSE_DIR = ROOT / 'checkpoints' / 'model_selection_audit'
SEED42_REUSE_METRICS = ROOT / 'results' / 'metrics' / 'v3_complex_model_selection_audit_metrics.csv'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_model_selection_3seed_metrics.csv'
EPOCH_LOG_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_model_selection_3seed_epoch_log.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_model_selection_3seed_summary.txt'


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
    if seed == 42:
        reused = SEED42_REUSE_DIR / f'v3_complex_model_selection_{criterion}.pt'
        if reused.exists():
            return reused
    return CHECKPOINT_DIR / f'seed_{seed}' / f'v3_complex_model_selection_seed{seed}_{criterion}.pt'


def save_audit_checkpoint(model, seed, epoch, val_metrics, train_dataset, criterion):
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
        'best_val_loss': float(val_metrics['mse']),
        'selection_val_metrics': {key: float(value) for key, value in val_metrics.items()},
        'signal_mean': float(train_dataset.signal_mean),
        'signal_std': float(train_dataset.signal_std),
    }
    torch.save(checkpoint, path)
    return path


@torch.no_grad()
def evaluate_model(model, dataset, coords, device, threshold, batch_size=8):
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
            rows.append(metrics)
    return rows


def summarize_metric_rows(rows):
    metric_names = ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']
    summary = {
        name: float(np.nanmean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    summary['pred_area_zero'] = int(sum(float(row['pred_area']) == 0.0 for row in rows))
    summary['pred_area_lt_true'] = int(sum(float(row['pred_area']) < float(row['true_area']) for row in rows))
    summary['pred_area_gt_true'] = int(sum(float(row['pred_area']) > float(row['true_area']) for row in rows))
    summary['composite'] = float(summary['iou'] + summary['dice'] - summary['area_error'])
    return summary


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
        stripped = {
            key[len('module.'):] if key.startswith('module.') else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(stripped)
    model.eval()
    return model, checkpoint


def load_seed42_reuse():
    if not SEED42_REUSE_METRICS.exists():
        return None
    existing_checkpoints = [checkpoint_path(42, criterion) for criterion in CRITERIA]
    if not all(path.exists() for path in existing_checkpoints):
        return None

    with open(SEED42_REUSE_METRICS, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    val_epoch_rows = []
    selected = {}
    for row in rows:
        row_type = row.get('row_type', '')
        if row_type not in ('val_epoch', 'selected_val'):
            continue
        converted = {
            'seed': 42,
            'row_type': row_type,
            'criterion': row.get('criterion', ''),
            'epoch': int(row['epoch']) if row.get('epoch') else '',
            'split': row.get('split', 'val'),
            'threshold': 500,
            'checkpoint': row.get('checkpoint', ''),
        }
        for key in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'composite']:
            converted[key] = float(row[key])
        for key in ['pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true']:
            converted[key] = int(float(row[key]))
        if row_type == 'val_epoch':
            val_epoch_rows.append(converted)
        else:
            criterion = row['criterion']
            converted['checkpoint'] = str(checkpoint_path(42, criterion).relative_to(ROOT))
            selected[criterion] = converted
    return val_epoch_rows, selected


def has_complete_epoch_log(seed):
    if not EPOCH_LOG_PATH.exists():
        return False
    with open(EPOCH_LOG_PATH, newline='', encoding='utf-8') as f:
        rows = csv.DictReader(f)
        count = sum(
            1
            for row in rows
            if row.get('row_type') == 'val_epoch'
            and row.get('seed') == str(seed)
        )
    return count >= EPOCHS


def load_selected_from_checkpoints(seed):
    selected = {}
    for criterion in CRITERIA:
        path = checkpoint_path(seed, criterion)
        checkpoint = torch.load(project_path(str(path)), map_location='cpu')
        metrics = checkpoint['selection_val_metrics']
        selected[criterion] = {
            'seed': seed,
            'row_type': 'selected_val',
            'criterion': criterion,
            'epoch': int(checkpoint['epoch']),
            'split': 'val',
            'threshold': 500,
            'checkpoint': str(path.relative_to(ROOT)),
            **metrics,
            'composite': float(metrics['iou'] + metrics['dice'] - metrics['area_error']),
        }
    return selected


def load_epoch_rows(seed):
    rows_out = []
    with open(EPOCH_LOG_PATH, newline='', encoding='utf-8') as f:
        rows = csv.DictReader(f)
        for row in rows:
            if row.get('row_type') != 'val_epoch' or row.get('seed') != str(seed):
                continue
            converted = {
                'seed': int(row['seed']),
                'row_type': row['row_type'],
                'criterion': row['criterion'],
                'epoch': int(row['epoch']),
                'split': row['split'],
                'threshold': int(float(row['threshold'])),
                'checkpoint': row.get('checkpoint', ''),
            }
            for key in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'composite']:
                converted[key] = float(row[key])
            for key in ['pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true']:
                converted[key] = int(float(row[key]))
            rows_out.append(converted)
    return rows_out


def train_seed(seed, device):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
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
            'value': float('inf') if direction == 'min' else -float('inf'),
            'epoch': None,
            'metrics': None,
            'checkpoint': None,
        }
        for name, (direction, _) in CRITERIA.items()
    }
    val_epoch_rows = []

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
        val_rows = evaluate_model(model, val_dataset, coords, device, threshold=500.0)
        val_metrics = summarize_metric_rows(val_rows)
        val_epoch_rows.append({
            'seed': seed,
            'row_type': 'val_epoch',
            'criterion': 'epoch',
            'epoch': epoch,
            'split': 'val',
            'threshold': 500,
            'checkpoint': '',
            **val_metrics,
        })

        for criterion_name, (direction, metric_name) in CRITERIA.items():
            value = val_metrics[metric_name]
            improved = value < best[criterion_name]['value'] if direction == 'min' else value > best[criterion_name]['value']
            if improved:
                path = save_audit_checkpoint(model, seed, epoch, val_metrics, train_dataset, criterion_name)
                best[criterion_name] = {
                    'value': value,
                    'epoch': epoch,
                    'metrics': dict(val_metrics),
                    'checkpoint': str(path.relative_to(ROOT)),
                }
        print(
            f"seed={seed} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_mse={train_metrics['mse_loss']:.6e} | "
            f"val_mse={val_metrics['mse']:.6e} | val_iou={val_metrics['iou']:.6e} | "
            f"val_dice={val_metrics['dice']:.6e} | val_area_error={val_metrics['area_error']:.6e}"
        )

    selected = {}
    for criterion, info in best.items():
        selected[criterion] = {
            'seed': seed,
            'row_type': 'selected_val',
            'criterion': criterion,
            'epoch': info['epoch'],
            'split': 'val',
            'threshold': 500,
            'checkpoint': info['checkpoint'],
            **info['metrics'],
        }
    return val_epoch_rows, selected


def eval_checkpoint_row(seed, criterion, threshold, path, device, signal_length, signal_channels):
    model, checkpoint = load_checkpoint_model(path, signal_length, signal_channels, device)
    signal_mean = float(checkpoint.get('signal_mean', 0.0))
    signal_std = float(checkpoint.get('signal_std', 1.0))
    dataset = MFLDataset(TEST_DATA, signal_mean=signal_mean, signal_std=signal_std)
    coords = build_coord_grid(dataset.x, dataset.y).to(device)
    metrics = summarize_metric_rows(evaluate_model(model, dataset, coords, device, threshold=threshold))
    return {
        'seed': seed,
        'row_type': 'test_candidate',
        'criterion': criterion,
        'epoch': '',
        'split': 'test',
        'threshold': int(threshold),
        'checkpoint': str(path),
        **metrics,
    }


def write_csv(path, rows):
    fieldnames = [
        'seed', 'row_type', 'criterion', 'epoch', 'split', 'threshold', 'checkpoint',
        'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error',
        'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def mean_std(rows, criterion, threshold):
    selected = [row for row in rows if row['criterion'] == criterion and int(row['threshold']) == int(threshold)]
    result = {'criterion': criterion, 'threshold': int(threshold), 'n': len(selected)}
    for metric in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite']:
        values = np.array([float(row[metric]) for row in selected], dtype=np.float64)
        result[f'{metric}_mean'] = float(values.mean())
        result[f'{metric}_std'] = float(values.std(ddof=0))
    return result


def paired_diff(rows, left, right, threshold):
    diffs = []
    for seed in SEEDS:
        left_row = next(row for row in rows if row['seed'] == seed and row['criterion'] == left and int(row['threshold']) == int(threshold))
        right_row = next(row for row in rows if row['seed'] == seed and row['criterion'] == right and int(row['threshold']) == int(threshold))
        diff = {'pair': f'{left} - {right}', 'threshold': int(threshold), 'seed': seed}
        for metric in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite']:
            diff[metric] = float(left_row[metric]) - float(right_row[metric])
        diffs.append(diff)
    mean = {'pair': f'{left} - {right}', 'threshold': int(threshold), 'seed': 'mean'}
    for metric in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite']:
        mean[metric] = float(np.mean([row[metric] for row in diffs]))
    return diffs, mean


def format_mean_std(stats):
    lines = [
        '| criterion | threshold | MSE | MAE | IoU | Dice | area_error | center_error | pred_area=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in stats:
        lines.append(
            f"| {row['criterion']} | {row['threshold']} | "
            f"{row['mse_mean']:.4e} ± {row['mse_std']:.2e} | "
            f"{row['mae_mean']:.4e} ± {row['mae_std']:.2e} | "
            f"{row['iou_mean']:.4f} ± {row['iou_std']:.4f} | "
            f"{row['dice_mean']:.4f} ± {row['dice_std']:.4f} | "
            f"{row['area_error_mean']:.4f} ± {row['area_error_std']:.4f} | "
            f"{row['center_error_mean']:.4f} ± {row['center_error_std']:.4f} | "
            f"{row['pred_area_zero_mean']:.2f} ± {row['pred_area_zero_std']:.2f} |"
        )
    return '\n'.join(lines)


def format_selection_epochs(selected_rows):
    lines = [
        '| seed | best_val_mse | best_val_iou | best_val_dice | best_val_area_error | best_composite |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for seed in SEEDS:
        epochs = {}
        for criterion in CRITERIA:
            row = next(item for item in selected_rows if item['seed'] == seed and item['criterion'] == criterion)
            epochs[criterion] = row['epoch']
        lines.append(
            f"| {seed} | {epochs['best_val_mse']} | {epochs['best_val_iou']} | {epochs['best_val_dice']} | "
            f"{epochs['best_val_area_error']} | {epochs['best_composite']} |"
        )
    return '\n'.join(lines)


def format_pair_means(pair_means):
    lines = [
        '| pair | threshold | ΔMSE | ΔMAE | ΔIoU | ΔDice | Δarea_error | Δcenter_error | Δpred_area=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in pair_means:
        lines.append(
            f"| {row['pair']} | {row['threshold']} | {row['mse']:.4e} | {row['mae']:.4e} | "
            f"{row['iou']:.4f} | {row['dice']:.4f} | {row['area_error']:.4f} | "
            f"{row['center_error']:.4f} | {row['pred_area_zero']:.2f} |"
        )
    return '\n'.join(lines)


def write_summary(selected_rows, test_rows):
    criteria_order = ['current_baseline', *CRITERIA.keys()]
    stats_500 = [mean_std(test_rows, criterion, 500) for criterion in criteria_order]
    stats_600 = [mean_std(test_rows, criterion, 600) for criterion in criteria_order]
    pair_means = []
    for threshold in THRESHOLDS:
        for left in ['best_composite', 'best_val_iou', 'best_val_dice']:
            _, mean = paired_diff(test_rows, left, 'best_val_mse', threshold)
            pair_means.append(mean)

    composite_500 = next(row for row in stats_500 if row['criterion'] == 'best_composite')
    mse_500 = next(row for row in stats_500 if row['criterion'] == 'best_val_mse')
    baseline_500 = next(row for row in stats_500 if row['criterion'] == 'current_baseline')
    stable_vs_mse = (
        composite_500['iou_mean'] > mse_500['iou_mean']
        and composite_500['dice_mean'] > mse_500['dice_mean']
        and composite_500['area_error_mean'] <= mse_500['area_error_mean']
        and composite_500['pred_area_zero_mean'] <= mse_500['pred_area_zero_mean']
    )
    stable_vs_baseline = (
        composite_500['iou_mean'] > baseline_500['iou_mean']
        and composite_500['dice_mean'] > baseline_500['dice_mean']
        and composite_500['area_error_mean'] <= baseline_500['area_error_mean']
    )
    mse_mae_cost_severe = (
        composite_500['mse_mean'] > baseline_500['mse_mean'] * 1.15
        or composite_500['mae_mean'] > baseline_500['mae_mean'] * 1.15
    )
    accepted = stable_vs_mse and not mse_mae_cost_severe

    summary = f"""# v3_complex model selection criterion 3-seed validation

This controlled diagnostic trains baseline v3_complex runs for seeds 42, 123, and 2026 with MSE loss, lambda_tv=2e-6, epochs=50, and threshold=500/600 evaluation. Seed=42 reuses the Step 13.1 checkpoint and validation epoch log. Seeds 123 and 2026 are newly trained. No changes were made to train_pinn.py, evaluate_pinn.py, or data_generator_v2.py.

## Selection epochs

{format_selection_epochs(selected_rows)}

## Test mean ± std at threshold=500

{format_mean_std(stats_500)}

## Test mean ± std at threshold=600

{format_mean_std(stats_600)}

## Paired differences versus best_val_mse

{format_pair_means(pair_means)}

## Diagnostic judgment

best_composite stable improvement over best_val_mse at threshold=500: {stable_vs_mse}.

best_composite mean better than CURRENT_BASELINE at threshold=500 by IoU/Dice without worse area_error: {stable_vs_baseline}.

MSE/MAE cost severe by >15% relative to CURRENT_BASELINE: {mse_mae_cost_severe}.

Accepted as model-selection criterion candidate: {accepted}.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'accepted': accepted,
        'stable_vs_mse': stable_vs_mse,
        'stable_vs_baseline': stable_vs_baseline,
        'mse_mae_cost_severe': mse_mae_cost_severe,
        'stats_500': stats_500,
        'stats_600': stats_600,
        'pair_means': pair_means,
    }


def main():
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    all_epoch_rows = []
    selected_rows = []

    reused = load_seed42_reuse()
    if reused is None:
        print('Seed 42 reuse unavailable; training seed=42.')
        epoch_rows, selected = train_seed(42, device)
    else:
        print('Reusing seed=42 Step 13.1 checkpoint and epoch log.')
        epoch_rows, selected = reused
    all_epoch_rows.extend(epoch_rows)
    selected_rows.extend(selected.values())

    for seed in [123, 2026]:
        complete = (
            all(checkpoint_path(seed, criterion).exists() for criterion in CRITERIA)
            and has_complete_epoch_log(seed)
        )
        if complete:
            print(f'Reusing existing checkpoints for seed={seed}.')
            all_epoch_rows.extend(load_epoch_rows(seed))
            selected_rows.extend(load_selected_from_checkpoints(seed).values())
        else:
            print(f'Training seed={seed}.')
            epoch_rows, selected = train_seed(seed, device)
            all_epoch_rows.extend(epoch_rows)
            selected_rows.extend(selected.values())

    # Evaluate current baseline and all candidates on thresholds 500 and 600.
    raw_test = np.load(project_path(TEST_DATA), allow_pickle=False)
    signal_length, signal_channels = signal_shape_info(raw_test['signals'])
    test_rows = []

    for threshold in THRESHOLDS:
        model, checkpoint = load_checkpoint_model(CURRENT_BASELINE, signal_length, signal_channels, device)
        dataset = MFLDataset(
            TEST_DATA,
            signal_mean=float(checkpoint.get('signal_mean', 0.0)),
            signal_std=float(checkpoint.get('signal_std', 1.0)),
        )
        coords = build_coord_grid(dataset.x, dataset.y).to(device)
        metrics = summarize_metric_rows(evaluate_model(model, dataset, coords, device, threshold=threshold))
        for seed in SEEDS:
            test_rows.append({
                'seed': seed,
                'row_type': 'test_candidate',
                'criterion': 'current_baseline',
                'epoch': '',
                'split': 'test',
                'threshold': int(threshold),
                'checkpoint': CURRENT_BASELINE,
                **metrics,
            })

    for selected in selected_rows:
        for threshold in THRESHOLDS:
            path = ROOT / selected['checkpoint']
            model, checkpoint = load_checkpoint_model(path, signal_length, signal_channels, device)
            dataset = MFLDataset(
                TEST_DATA,
                signal_mean=float(checkpoint.get('signal_mean', 0.0)),
                signal_std=float(checkpoint.get('signal_std', 1.0)),
            )
            coords = build_coord_grid(dataset.x, dataset.y).to(device)
            metrics = summarize_metric_rows(evaluate_model(model, dataset, coords, device, threshold=threshold))
            test_rows.append({
                'seed': selected['seed'],
                'row_type': 'test_candidate',
                'criterion': selected['criterion'],
                'epoch': selected['epoch'],
                'split': 'test',
                'threshold': int(threshold),
                'checkpoint': selected['checkpoint'],
                **metrics,
            })

    write_csv(EPOCH_LOG_PATH, all_epoch_rows)
    write_csv(METRICS_PATH, selected_rows + test_rows)
    judgment = write_summary(selected_rows, test_rows)

    print(f'Wrote epoch log: {EPOCH_LOG_PATH}')
    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Accepted: {judgment["accepted"]}')


if __name__ == '__main__':
    main()
