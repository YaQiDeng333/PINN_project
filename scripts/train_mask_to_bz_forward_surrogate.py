import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import project_path, set_seed  # noqa: E402


TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

SEED = 42
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3
MASK_THRESHOLD_NORM = 0.5
MU_SCALE = 1000.0
R2_ACCEPT_FLOOR = 0.8
LOW_SIGNAL_R2_WARN_FLOOR = 0.5
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_to_bz_forward_surrogate'
CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_mask_to_bz_forward_surrogate.pt'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_mask_to_bz_forward_surrogate_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_mask_to_bz_forward_surrogate_summary.txt'


class MaskToBzDataset(Dataset):
    def __init__(self, npz_path, signal_mean=None, signal_std=None):
        data = np.load(project_path(npz_path), allow_pickle=False)
        self.raw_signals = data['signals'].astype(np.float32)
        self.mu_maps = data['mu_maps'].astype(np.float32) / MU_SCALE
        self.defect_types = data['defect_types'].astype(str)
        self.metadata = data['metadata']
        self.x = data['x'].astype(np.float32)
        self.y = data['y'].astype(np.float32)
        if signal_mean is None:
            signal_mean = float(self.raw_signals.mean())
        if signal_std is None:
            signal_std = float(self.raw_signals.std() + 1e-8)
        self.signal_mean = float(signal_mean)
        self.signal_std = float(signal_std)
        self.signals = (self.raw_signals - self.signal_mean) / self.signal_std
        self.masks = (self.mu_maps < MASK_THRESHOLD_NORM).astype(np.float32)
        self.true_areas = self.masks.reshape(self.masks.shape[0], -1).sum(axis=1).astype(np.float64)
        self.max_abs_bz = np.max(np.abs(self.raw_signals.reshape(self.raw_signals.shape[0], -1)), axis=1)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.masks[idx][None, :, :]),
            torch.from_numpy(self.signals[idx]),
            idx,
        )


class MaskToBzForwardSurrogate(nn.Module):
    def __init__(self, out_length=200, out_shape=(100, 200), use_coords=True):
        super().__init__()
        self.out_length = int(out_length)
        self.out_shape = tuple(out_shape)
        self.use_coords = bool(use_coords)
        in_channels = 3 if self.use_coords else 1
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 16)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, self.out_length),
        )

    def coord_channels(self, batch_size, device, dtype):
        height, width = self.out_shape
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return coords.expand(batch_size, -1, -1, -1)

    def forward(self, mask_prob):
        if mask_prob.dim() == 3:
            mask_prob = mask_prob.unsqueeze(1)
        x = mask_prob
        if self.use_coords:
            coords = self.coord_channels(x.shape[0], x.device, x.dtype)
            x = torch.cat([x, coords], dim=1)
        return self.head(self.encoder(x))


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_loader(dataset, shuffle=False):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def pearson_corr(pred, true):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred_centered = pred - pred.mean()
    true_centered = true - true.mean()
    denom = np.sqrt(np.sum(pred_centered ** 2) * np.sum(true_centered ** 2))
    if denom <= 1e-12:
        return float('nan')
    return float(np.sum(pred_centered * true_centered) / denom)


def metric_dict(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    sse = float(np.sum(diff ** 2))
    sst = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else float('nan')
    corr = pearson_corr(pred, true)
    peak_error = float(np.mean(np.abs(np.argmax(np.abs(pred), axis=1) - np.argmax(np.abs(true), axis=1))))
    return {
        'bz_mse': mse,
        'bz_mae': mae,
        'r2': r2,
        'pearson_corr': corr,
        'peak_pos_error': peak_error,
    }


@torch.no_grad()
def predict(model, dataset, device):
    model.eval()
    loader = make_loader(dataset, shuffle=False)
    pred = np.empty_like(dataset.signals, dtype=np.float32)
    true = np.empty_like(dataset.signals, dtype=np.float32)
    for masks, signals, indices in loader:
        masks = masks.to(device)
        output = model(masks).cpu().numpy()
        for batch_pos, idx_tensor in enumerate(indices):
            idx = int(idx_tensor.item())
            pred[idx] = output[batch_pos]
            true[idx] = signals[batch_pos].numpy()
    return pred, true


def split_area_edges(dataset):
    return np.quantile(dataset.true_areas, [1 / 3, 2 / 3])


def area_bin(value, edges):
    if value <= edges[0]:
        return 'small'
    if value <= edges[1]:
        return 'medium'
    return 'large'


def low_signal_indices(dataset):
    threshold = np.quantile(dataset.max_abs_bz, 1 / 3)
    return {idx for idx, value in enumerate(dataset.max_abs_bz) if float(value) <= float(threshold)}


def summarize_split(candidate, split, pred, true, dataset):
    rows = []
    edges = split_area_edges(dataset)
    low_indices = low_signal_indices(dataset)
    groups = [('overall', 'all', list(range(len(dataset))))]
    for group in ['small', 'medium', 'large']:
        groups.append((
            'area_bin',
            group,
            [idx for idx, area in enumerate(dataset.true_areas) if area_bin(float(area), edges) == group],
        ))
    for group in ['low_signal', 'non_low_signal']:
        if group == 'low_signal':
            indices = sorted(low_indices)
        else:
            indices = [idx for idx in range(len(dataset)) if idx not in low_indices]
        groups.append(('signal_bin', group, indices))
    for defect_type in sorted(set(dataset.defect_types.astype(str))):
        groups.append((
            'defect_type',
            defect_type,
            [idx for idx, value in enumerate(dataset.defect_types.astype(str)) if value == defect_type],
        ))

    for group_type, group, indices in groups:
        if indices:
            metrics = metric_dict(pred[indices], true[indices])
        else:
            metrics = {
                'bz_mse': float('nan'),
                'bz_mae': float('nan'),
                'r2': float('nan'),
                'pearson_corr': float('nan'),
                'peak_pos_error': float('nan'),
            }
        row = {
            'candidate': candidate,
            'split': split,
            'group_type': group_type,
            'group': group,
            'n': len(indices),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def train_surrogate(device):
    set_seed(SEED)
    train_dataset = MaskToBzDataset(TRAIN_DATA)
    val_dataset = MaskToBzDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    signal_length = train_dataset.signals.shape[1]
    out_shape = train_dataset.masks.shape[1:]
    model = MaskToBzForwardSurrogate(out_length=signal_length, out_shape=out_shape, use_coords=True).to(device)

    if REUSE_EXISTING and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Reusing existing surrogate checkpoint: {CHECKPOINT_PATH}')
        return model, checkpoint.get('best_info', {})

    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loader = make_loader(train_dataset, shuffle=True)
    best_info = None
    best_r2 = -float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for masks, signals, indices in train_loader:
            masks = masks.to(device)
            signals = signals.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(masks)
            loss = F.mse_loss(pred, signals)
            loss.backward()
            optimizer.step()
            batch_size = masks.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        val_pred, val_true = predict(model, val_dataset, device)
        val_metrics = metric_dict(val_pred, val_true)
        if val_metrics['r2'] > best_r2:
            best_r2 = val_metrics['r2']
            best_info = {
                'epoch': epoch,
                'train_mse': total_loss / total_samples,
                **val_metrics,
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': {
                    'model': 'mask_to_bz_forward_surrogate',
                    'input': 'target_mask = target_mu_norm < 0.5 plus fixed x/y coordinate channels',
                    'output': 'normalized Bz signal',
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'lr': LR,
                    'seed': SEED,
                    'loss': 'MSE(pred_bz_norm, true_bz_norm)',
                    'signal_length': signal_length,
                    'out_shape': out_shape,
                    'use_coords': True,
                },
                'signal_mean': float(train_dataset.signal_mean),
                'signal_std': float(train_dataset.signal_std),
                'best_info': best_info,
            }, CHECKPOINT_PATH)

        print(
            f"epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_mse={total_loss / total_samples:.6e} | "
            f"val_mse={val_metrics['bz_mse']:.6e} | "
            f"val_mae={val_metrics['bz_mae']:.6e} | "
            f"val_r2={val_metrics['r2']:.6f} | "
            f"val_corr={val_metrics['pearson_corr']:.6f}"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, best_info


def write_metrics(rows):
    fieldnames = [
        'candidate',
        'split',
        'group_type',
        'group',
        'n',
        'bz_mse',
        'bz_mae',
        'r2',
        'pearson_corr',
        'peak_pos_error',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def find_row(rows, split, group_type='overall', group='all'):
    return next(
        row for row in rows
        if row['split'] == split
        and row['group_type'] == group_type
        and row['group'] == group
    )


def fmt(row):
    return (
        f"MSE={float(row['bz_mse']):.6e}, MAE={float(row['bz_mae']):.6e}, "
        f"R2={float(row['r2']):.4f}, corr={float(row['pearson_corr']):.4f}, "
        f"peak_pos_error={float(row['peak_pos_error']):.2f}"
    )


def write_summary(rows, best_info, train_dataset, val_dataset, test_dataset):
    val_overall = find_row(rows, 'val')
    test_overall = find_row(rows, 'test')
    test_small = find_row(rows, 'test', 'area_bin', 'small')
    test_low = find_row(rows, 'test', 'signal_bin', 'low_signal')
    accepted = (
        float(test_overall['r2']) >= R2_ACCEPT_FLOOR
        and float(test_overall['pearson_corr']) >= 0.85
        and float(test_low['r2']) >= LOW_SIGNAL_R2_WARN_FLOOR
    )

    summary = f"""# v3_complex mask-to-Bz forward surrogate

This script trains a lightweight mask-to-Bz forward surrogate for Step 18.2A. It does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, or NEXT_STEP.md.

## Data and target

Input mask: `target_mu_norm < 0.5`, equivalent to raw `target_mu < 500`.

The model receives a binary mask plus fixed x/y coordinate channels and predicts the normalized Bz signal. The signal normalization uses train split mean/std and is stored in the checkpoint for reuse by any frozen forward-consistency candidate.

| split | samples | signal_length | mask_shape |
|---|---:|---:|---|
| train | {len(train_dataset)} | {train_dataset.signals.shape[1]} | {train_dataset.masks.shape[1:]} |
| val | {len(val_dataset)} | {val_dataset.signals.shape[1]} | {val_dataset.masks.shape[1:]} |
| test | {len(test_dataset)} | {test_dataset.signals.shape[1]} | {test_dataset.masks.shape[1:]} |

## Model

Simple Conv2d encoder over mask + coordinate channels, adaptive pooling, and an MLP head to output the full Bz signal. Loss is MSE(pred_bz_norm, true_bz_norm). No COMSOL, neural operator, large framework, post-processing, or mask-model training is used in this stage.

Best validation epoch: {best_info.get('epoch', 'N/A')}

## Overall metrics

All Bz metrics are computed on normalized Bz signals.

* val: {fmt(val_overall)}
* test: {fmt(test_overall)}

## Small / low-signal reliability

* test small: {fmt(test_small)}
* test low_signal: {fmt(test_low)}

## Defect-type test metrics

| type | n | MSE | MAE | R2 | corr | peak_pos_error |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in [r for r in rows if r['split'] == 'test' and r['group_type'] == 'defect_type']:
        summary += (
            f"| {row['group']} | {row['n']} | {float(row['bz_mse']):.6e} | "
            f"{float(row['bz_mae']):.6e} | {float(row['r2']):.4f} | "
            f"{float(row['pearson_corr']):.4f} | {float(row['peak_pos_error']):.2f} |\n"
        )

    summary += f"""
## Gate judgment

Surrogate accepted for Step 18.2B: {accepted}

Acceptance rule: test R2 must be clearly above {R2_ACCEPT_FLOOR}, shape correlation must be high, and low-signal samples must not collapse. If this is false, forward consistency is stopped at 18.2A.

Conclusion: {'The mask-to-Bz surrogate is reliable enough to freeze as a forward consistency module.' if accepted else 'The mask-to-Bz surrogate is not reliable enough; do not enter the forward-consistency candidate stage.'}
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return accepted


def main():
    ensure_outputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    train_dataset = MaskToBzDataset(TRAIN_DATA)
    val_dataset = MaskToBzDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    test_dataset = MaskToBzDataset(TEST_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    model, best_info = train_surrogate(device)

    rows = []
    for split, dataset in [('val', val_dataset), ('test', test_dataset)]:
        pred, true = predict(model, dataset, device)
        rows.extend(summarize_split('mask_to_bz_forward_surrogate', split, pred, true, dataset))
    write_metrics(rows)
    accepted = write_summary(rows, best_info or {}, train_dataset, val_dataset, test_dataset)
    test_overall = find_row(rows, 'test')
    print(
        'test_overall: '
        f"MSE={float(test_overall['bz_mse']):.6e}, "
        f"MAE={float(test_overall['bz_mae']):.6e}, "
        f"R2={float(test_overall['r2']):.6f}, "
        f"corr={float(test_overall['pearson_corr']):.6f}, "
        f"peak_pos_error={float(test_overall['peak_pos_error']):.2f}"
    )
    print(f'accepted={accepted}')
    print(f'wrote checkpoint: {CHECKPOINT_PATH}')
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
