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

SEED = 42
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
LATENT_DIM = 64
LAMBDA_TV = 2e-6
THRESHOLD = 500.0

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'model_selection_audit'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_model_selection_audit_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_model_selection_audit_summary.txt'


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_loader(dataset, batch_size, shuffle=False, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def save_audit_checkpoint(model, epoch, val_metrics, train_dataset, criterion_name):
    checkpoint_path = CHECKPOINT_DIR / f'v3_complex_model_selection_{criterion_name}.pt'
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
            'seed': SEED,
            'epochs': EPOCHS,
            'selection_criterion': criterion_name,
        },
        'epoch': int(epoch),
        'best_val_loss': float(val_metrics['mse']),
        'selection_val_metrics': {key: float(value) for key, value in val_metrics.items()},
        'signal_mean': float(train_dataset.signal_mean),
        'signal_std': float(train_dataset.signal_std),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@torch.no_grad()
def evaluate_model(model, dataset, coords, device, batch_size=8):
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
            sample_index = int(sample_idx_tensor.item())
            metrics, _, _ = compute_sample_metrics(
                pred_mu=pred_maps[batch_idx],
                true_mu=true_maps[batch_idx],
                x_grid=x_grid,
                y_grid=y_grid,
                threshold=THRESHOLD,
            )
            metrics['sample_index'] = sample_index
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
    summary['composite'] = float(summary['dice'] + summary['iou'] - summary['area_error'])
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


def write_metrics_csv(val_epoch_rows, candidate_rows):
    fieldnames = [
        'row_type',
        'criterion',
        'epoch',
        'split',
        'checkpoint',
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
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in val_epoch_rows + candidate_rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def table(rows, columns):
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] + ['---:'] * (len(columns) - 1)) + ' |',
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f'{value:.8e}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def write_summary(selection_rows, test_rows, current_baseline_row):
    criterion_rows = [row for row in selection_rows if row['row_type'] == 'selected_val']
    test_candidate_rows = [row for row in test_rows if row['criterion'] != 'current_baseline']
    best_non_mse = max(
        test_candidate_rows,
        key=lambda row: (row['iou'] + row['dice'] - row['area_error']),
    )
    mse_test = next(row for row in test_candidate_rows if row['criterion'] == 'best_val_mse')
    clearly_better_than_baseline = (
        best_non_mse['iou'] > current_baseline_row['iou']
        and best_non_mse['dice'] > current_baseline_row['dice']
        and best_non_mse['area_error'] <= current_baseline_row['area_error']
    )
    clearly_better_than_mse = (
        best_non_mse['iou'] > mse_test['iou']
        and best_non_mse['dice'] > mse_test['dice']
        and best_non_mse['area_error'] <= mse_test['area_error']
    )
    bottleneck = clearly_better_than_baseline or clearly_better_than_mse

    summary = f"""# v3_complex model selection metric audit

This diagnostic trains one seed=42 v3_complex baseline run for 50 epochs with MSE loss and lambda_tv=2e-6. Each epoch is evaluated on validation with MSE, MAE, IoU, Dice, area_error, center_error, pred_area=0, pred_area<true_area, and pred_area>true_area. Candidate checkpoints are selected by best val MSE, IoU, Dice, area_error, and composite = Dice + IoU - area_error.

No changes were made to train_pinn.py or evaluate_pinn.py.

## Selected validation checkpoints

{table(criterion_rows, ['criterion', 'epoch', 'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite'])}

## Test metrics

{table(test_rows, ['criterion', 'epoch', 'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error', 'pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'composite'])}

## Diagnostic judgment

Best non-MSE test composite candidate: {best_non_mse['criterion']} at epoch {best_non_mse['epoch']}.

Clearly better than CURRENT_BASELINE by IoU/Dice without worse area_error: {clearly_better_than_baseline}.

Clearly better than the best_val_mse checkpoint by IoU/Dice without worse area_error: {clearly_better_than_mse}.

Model selection metric is likely a major bottleneck: {bottleneck}.
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return {
        'best_non_mse': best_non_mse,
        'clearly_better_than_baseline': clearly_better_than_baseline,
        'clearly_better_than_mse': clearly_better_than_mse,
        'bottleneck': bottleneck,
    }


def main():
    ensure_dirs()
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using random seed: {SEED}')

    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    test_dataset = MFLDataset(TEST_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)

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
    train_loader = make_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_specs = {
        'best_val_mse': ('min', 'mse'),
        'best_val_iou': ('max', 'iou'),
        'best_val_dice': ('max', 'dice'),
        'best_val_area_error': ('min', 'area_error'),
        'best_composite': ('max', 'composite'),
    }
    best = {
        name: {
            'value': float('inf') if direction == 'min' else -float('inf'),
            'epoch': None,
            'metrics': None,
            'checkpoint': None,
        }
        for name, (direction, _) in best_specs.items()
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
        val_rows = evaluate_model(model, val_dataset, coords, device)
        val_metrics = summarize_metric_rows(val_rows)
        epoch_row = {
            'row_type': 'val_epoch',
            'criterion': 'epoch',
            'epoch': epoch,
            'split': 'val',
            'checkpoint': '',
            **val_metrics,
        }
        val_epoch_rows.append(epoch_row)

        for criterion_name, (direction, metric_name) in best_specs.items():
            metric_value = val_metrics[metric_name]
            improved = metric_value < best[criterion_name]['value'] if direction == 'min' else metric_value > best[criterion_name]['value']
            if improved:
                checkpoint_path = save_audit_checkpoint(model, epoch, val_metrics, train_dataset, criterion_name)
                best[criterion_name] = {
                    'value': metric_value,
                    'epoch': epoch,
                    'metrics': dict(val_metrics),
                    'checkpoint': str(checkpoint_path.relative_to(ROOT)),
                }

        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_mse={train_metrics['mse_loss']:.6e} | "
            f"val_mse={val_metrics['mse']:.6e} | "
            f"val_iou={val_metrics['iou']:.6e} | "
            f"val_dice={val_metrics['dice']:.6e} | "
            f"val_area_error={val_metrics['area_error']:.6e}"
        )

    selection_rows = []
    for criterion_name, info in best.items():
        selection_rows.append({
            'row_type': 'selected_val',
            'criterion': criterion_name,
            'epoch': info['epoch'],
            'split': 'val',
            'checkpoint': info['checkpoint'],
            **info['metrics'],
        })

    # Evaluate current baseline with its own signal normalization.
    raw_test = np.load(project_path(TEST_DATA), allow_pickle=False)
    signal_length_raw, signal_channels_raw = signal_shape_info(raw_test['signals'])
    current_model, current_checkpoint = load_checkpoint_model(CURRENT_BASELINE, signal_length_raw, signal_channels_raw, device)
    current_mean = float(current_checkpoint.get('signal_mean', 0.0))
    current_std = float(current_checkpoint.get('signal_std', 1.0))
    current_test_dataset = MFLDataset(TEST_DATA, signal_mean=current_mean, signal_std=current_std)
    test_coords = build_coord_grid(current_test_dataset.x, current_test_dataset.y).to(device)
    current_test = summarize_metric_rows(evaluate_model(current_model, current_test_dataset, test_coords, device))
    current_baseline_row = {
        'row_type': 'test_candidate',
        'criterion': 'current_baseline',
        'epoch': '',
        'split': 'test',
        'checkpoint': CURRENT_BASELINE,
        **current_test,
    }

    # Evaluate candidates using training normalization saved in checkpoints.
    test_rows = [current_baseline_row]
    for criterion_name, info in best.items():
        candidate_model, checkpoint = load_checkpoint_model(info['checkpoint'], signal_length_raw, signal_channels_raw, device)
        candidate_mean = float(checkpoint.get('signal_mean', train_dataset.signal_mean))
        candidate_std = float(checkpoint.get('signal_std', train_dataset.signal_std))
        candidate_dataset = MFLDataset(TEST_DATA, signal_mean=candidate_mean, signal_std=candidate_std)
        candidate_coords = build_coord_grid(candidate_dataset.x, candidate_dataset.y).to(device)
        candidate_metrics = summarize_metric_rows(evaluate_model(candidate_model, candidate_dataset, candidate_coords, device))
        test_rows.append({
            'row_type': 'test_candidate',
            'criterion': criterion_name,
            'epoch': info['epoch'],
            'split': 'test',
            'checkpoint': info['checkpoint'],
            **candidate_metrics,
        })

    write_metrics_csv(val_epoch_rows + selection_rows, test_rows)
    judgment = write_summary(selection_rows, test_rows, current_baseline_row)

    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Model selection bottleneck: {judgment["bottleneck"]}')


if __name__ == '__main__':
    main()
