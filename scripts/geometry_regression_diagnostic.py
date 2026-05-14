import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MU_SCALE = 1000.0
MASK_THRESHOLD_NORM = 0.5

TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'
FAILURE_AUDIT = 'results/metrics/v3_current_baseline_failure_audit.csv'

METRICS_PATH = 'results/metrics/v3_complex_geometry_regression_diagnostic_metrics.csv'
SUMMARY_PATH = 'results/summaries/v3_complex_geometry_regression_diagnostic_summary.txt'


def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path):
    return np.load(project_path(path), allow_pickle=False)


def compute_labels(data):
    mu_norm = data['mu_maps'].astype(np.float32) / MU_SCALE
    masks = mu_norm < MASK_THRESHOLD_NORM
    flat_masks = masks.reshape(masks.shape[0], -1)
    area = flat_masks.mean(axis=1).astype(np.float32)
    counts = flat_masks.sum(axis=1).astype(np.float32)

    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    x01 = (x - x.min()) / (x.max() - x.min())
    y01 = (y - y.min()) / (y.max() - y.min())
    xx, yy = np.meshgrid(x01, y01)
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)

    centroid_x = np.full(masks.shape[0], 0.5, dtype=np.float32)
    centroid_y = np.full(masks.shape[0], 0.5, dtype=np.float32)
    valid = counts > 0
    if np.any(valid):
        centroid_x[valid] = (flat_masks[valid] * xx_flat[None, :]).sum(axis=1) / counts[valid]
        centroid_y[valid] = (flat_masks[valid] * yy_flat[None, :]).sum(axis=1) / counts[valid]

    return np.stack([area, centroid_x, centroid_y], axis=1).astype(np.float32), counts


def signal_features(signals):
    if signals.ndim == 3:
        signals = signals[:, 0, :]
    return {
        'max_abs_bz': np.max(np.abs(signals), axis=1),
        'peak_to_peak_bz': np.ptp(signals, axis=1),
        'mean_abs_bz': np.mean(np.abs(signals), axis=1),
        'std_bz': np.std(signals, axis=1),
        'l2_energy_bz': np.sum(signals ** 2, axis=1),
    }


class GeometryDataset(Dataset):
    def __init__(self, data, labels, signal_mean, signal_std, target_mean, target_std):
        signals = data['signals'].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        self.signals = (signals - signal_mean) / signal_std
        self.labels = labels.astype(np.float32)
        self.labels_std = (self.labels - target_mean) / target_std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.labels_std[idx]),
        )


class GeometryRegressor(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, signals):
        if signals.dim() == 2:
            signals = signals.unsqueeze(1)
        return self.net(signals)


def train_model(train_loader, val_loader, signal_length, device, epochs, lr):
    model = GeometryRegressor(signal_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for signals, targets in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            pred = model(signals)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.shape[0]
            train_count += signals.shape[0]

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for signals, targets in val_loader:
                signals = signals.to(device)
                targets = targets.to(device)
                pred = model(signals)
                loss = F.mse_loss(pred, targets)
                val_loss += loss.item() * signals.shape[0]
                val_count += signals.shape[0]

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_val


@torch.no_grad()
def predict(model, loader, target_mean, target_std, device):
    model.eval()
    preds = []
    for signals, _ in loader:
        signals = signals.to(device)
        pred_std = model(signals).cpu().numpy()
        preds.append(pred_std * target_std + target_mean)
    return np.vstack(preds).astype(np.float32)


def regression_metrics(y_true, y_pred):
    rows = {}
    for idx, name in enumerate(['area', 'centroid_x', 'centroid_y']):
        err = y_pred[:, idx] - y_true[:, idx]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = float(np.sum((y_true[:, idx] - y_true[:, idx].mean()) ** 2))
        r2 = float('nan') if denom == 0.0 else float(1.0 - np.sum(err ** 2) / denom)
        rows[f'{name}_mae'] = mae
        rows[f'{name}_rmse'] = rmse
        rows[f'{name}_r2'] = r2
    return rows


def read_baseline_iou():
    path = project_path(FAILURE_AUDIT)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'sample_index' in row and 'iou' in row:
                out[int(float(row['sample_index']))] = float(row['iou'])
    return out


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float('nan')
    if np.std(a[valid]) == 0.0 or np.std(b[valid]) == 0.0:
        return float('nan')
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def main():
    set_seed(42)
    train_data = load_npz(TRAIN_DATA)
    val_data = load_npz(VAL_DATA)
    test_data = load_npz(TEST_DATA)

    train_labels, train_counts = compute_labels(train_data)
    val_labels, _ = compute_labels(val_data)
    test_labels, test_counts = compute_labels(test_data)

    raw_train_signals = train_data['signals'].astype(np.float32)
    if raw_train_signals.ndim == 3:
        raw_train_signals = raw_train_signals[:, 0, :]
    signal_mean = float(raw_train_signals.mean())
    signal_std = float(raw_train_signals.std() + 1e-8)
    target_mean = train_labels.mean(axis=0).astype(np.float32)
    target_std = (train_labels.std(axis=0) + 1e-8).astype(np.float32)

    train_ds = GeometryDataset(train_data, train_labels, signal_mean, signal_std, target_mean, target_std)
    val_ds = GeometryDataset(val_data, val_labels, signal_mean, signal_std, target_mean, target_std)
    test_ds = GeometryDataset(test_data, test_labels, signal_mean, signal_std, target_mean, target_std)

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    signal_length = raw_train_signals.shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history, best_val = train_model(train_loader, val_loader, signal_length, device, epochs=120, lr=1e-3)
    pred = predict(model, test_loader, target_mean, target_std, device)

    train_area = train_labels[:, 0]
    area_q1, area_q2 = np.percentile(train_area, [33.3333333333, 66.6666666667])
    test_area = test_labels[:, 0]
    area_bins = np.where(test_area <= area_q1, 'small', np.where(test_area <= area_q2, 'medium', 'large'))

    train_signal_stats = signal_features(train_data['signals'].astype(np.float32))
    low_cutoff = float(np.percentile(train_signal_stats['max_abs_bz'], 33.3333333333))
    test_signal_stats = signal_features(test_data['signals'].astype(np.float32))
    low_signal = test_signal_stats['max_abs_bz'] <= low_cutoff

    baseline_iou_by_index = read_baseline_iou()
    geom_abs_error = np.sqrt(np.sum((pred - test_labels) ** 2, axis=1))
    baseline_iou = np.array([baseline_iou_by_index.get(idx, np.nan) for idx in range(len(test_labels))])

    os.makedirs(project_path('results/metrics'), exist_ok=True)
    os.makedirs(project_path('results/summaries'), exist_ok=True)

    metrics_path = project_path(METRICS_PATH)
    fieldnames = [
        'sample_index',
        'area_bin',
        'low_signal',
        'true_area',
        'pred_area',
        'area_abs_error',
        'true_centroid_x',
        'pred_centroid_x',
        'centroid_x_abs_error',
        'true_centroid_y',
        'pred_centroid_y',
        'centroid_y_abs_error',
        'geometry_l2_error',
        'max_abs_bz',
        'peak_to_peak_bz',
        'mean_abs_bz',
        'std_bz',
        'l2_energy_bz',
        'current_baseline_iou',
    ]
    with open(metrics_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(test_labels)):
            writer.writerow({
                'sample_index': idx,
                'area_bin': area_bins[idx],
                'low_signal': bool(low_signal[idx]),
                'true_area': float(test_labels[idx, 0]),
                'pred_area': float(pred[idx, 0]),
                'area_abs_error': float(abs(pred[idx, 0] - test_labels[idx, 0])),
                'true_centroid_x': float(test_labels[idx, 1]),
                'pred_centroid_x': float(pred[idx, 1]),
                'centroid_x_abs_error': float(abs(pred[idx, 1] - test_labels[idx, 1])),
                'true_centroid_y': float(test_labels[idx, 2]),
                'pred_centroid_y': float(pred[idx, 2]),
                'centroid_y_abs_error': float(abs(pred[idx, 2] - test_labels[idx, 2])),
                'geometry_l2_error': float(geom_abs_error[idx]),
                'max_abs_bz': float(test_signal_stats['max_abs_bz'][idx]),
                'peak_to_peak_bz': float(test_signal_stats['peak_to_peak_bz'][idx]),
                'mean_abs_bz': float(test_signal_stats['mean_abs_bz'][idx]),
                'std_bz': float(test_signal_stats['std_bz'][idx]),
                'l2_energy_bz': float(test_signal_stats['l2_energy_bz'][idx]),
                'current_baseline_iou': float(baseline_iou[idx]) if np.isfinite(baseline_iou[idx]) else '',
            })

    overall = regression_metrics(test_labels, pred)
    grouped = {}
    for group in ['small', 'medium', 'large']:
        mask = area_bins == group
        grouped[group] = regression_metrics(test_labels[mask], pred[mask])
        grouped[group]['count'] = int(mask.sum())
    grouped['low_signal'] = regression_metrics(test_labels[low_signal], pred[low_signal])
    grouped['low_signal']['count'] = int(low_signal.sum())
    grouped['non_low_signal'] = regression_metrics(test_labels[~low_signal], pred[~low_signal])
    grouped['non_low_signal']['count'] = int((~low_signal).sum())

    corr_geom_iou = corrcoef_safe(geom_abs_error, baseline_iou)
    corr_area_iou = corrcoef_safe(np.abs(pred[:, 0] - test_labels[:, 0]), baseline_iou)
    corr_center_iou = corrcoef_safe(
        np.sqrt(np.sum((pred[:, 1:] - test_labels[:, 1:]) ** 2, axis=1)),
        baseline_iou,
    )

    summary_path = project_path(SUMMARY_PATH)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('# v3_complex direct geometry regression diagnostic\\n\\n')
        f.write('This diagnostic trains an independent lightweight regressor from single-channel Bz to normalized defect area and centroid. It does not modify the reconstruction model or CURRENT_BASELINE.\\n\\n')
        f.write('Configuration: v3_complex train/val/test, seed=42, epochs=120, target labels standardized during training and reported in normalized [0, 1] geometry units.\\n\\n')
        f.write(f'Best validation standardized MSE: {best_val:.8e}\\n\\n')
        f.write('## Overall test metrics\\n\\n')
        f.write('| target | MAE | RMSE | R2 |\\n')
        f.write('|---|---:|---:|---:|\\n')
        for name in ['area', 'centroid_x', 'centroid_y']:
            mae = overall[f'{name}_mae']
            rmse = overall[f'{name}_rmse']
            r2 = overall[f'{name}_r2']
            f.write(f'| {name} | {mae:.8e} | {rmse:.8e} | {r2:.8e} |\\n')
        f.write('\\n## Area-bin and low-signal metrics\\n\\n')
        f.write('| group | count | area MAE | area R2 | cx MAE | cx R2 | cy MAE | cy R2 |\\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\\n')
        for group, stats in grouped.items():
            f.write(
                f'| {group} | {stats["count"]} | {stats["area_mae"]:.8e} | '
                f'{stats["area_r2"]:.8e} | {stats["centroid_x_mae"]:.8e} | '
                f'{stats["centroid_x_r2"]:.8e} | {stats["centroid_y_mae"]:.8e} | '
                f'{stats["centroid_y_r2"]:.8e} |\\n'
            )
        f.write('\\n## Relationship with CURRENT_BASELINE IoU\\n\\n')
        f.write(f'corr(geometry_l2_error, baseline_iou): {corr_geom_iou:.8e}\\n')
        f.write(f'corr(area_abs_error, baseline_iou): {corr_area_iou:.8e}\\n')
        f.write(f'corr(center_abs_error, baseline_iou): {corr_center_iou:.8e}\\n\\n')
        f.write('## Diagnostic judgment\\n\\n')
        f.write('Area prediction is considered useful only if R2 is clearly positive and errors are not concentrated in small/low-signal samples. Centroid prediction is considered useful only if centroid_x/y R2 are clearly positive.\\n')
        if overall['area_r2'] > 0.3 and overall['centroid_x_r2'] > 0.3 and overall['centroid_y_r2'] > 0.3:
            f.write('Result: geometry is recoverable enough from single Bz to support a future geometry-conditioned reconstruction hypothesis.\\n')
        else:
            f.write('Result: single Bz does not stably support all global geometry targets in this lightweight diagnostic; geometry-conditioned reconstruction is not strongly supported by this gate.\\n')

    print('device', device)
    print('best_val', best_val)
    print('overall', overall)
    print('grouped', grouped)
    print('corr_geom_iou', corr_geom_iou)
    print('metrics_path', metrics_path)
    print('summary_path', summary_path)


if __name__ == '__main__':
    main()
