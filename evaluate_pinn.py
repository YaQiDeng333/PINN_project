import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_pinn import MFLDataset, MU_SCALE, PINN, build_coord_grid, project_path


MASK_THRESHOLD = 500.0
EVAL_DATASETS = {
    'simple': 'data/training_data_test.npz',
    'v3_complex': 'data/training_data_v3_complex_test.npz',
    'v4_balanced_complex': 'data/training_data_v4_balanced_complex_test.npz',
}


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_model(checkpoint_path, signal_length, device):
    checkpoint = torch.load(project_path(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        checkpoint_args = checkpoint.get('args', {})
        latent_dim = int(checkpoint_args.get('latent_dim', 64))
        model_variant = checkpoint_args.get('model_variant', 'baseline')
        decoder_variant = checkpoint_args.get('decoder_variant', 'standard')
        signal_mean = float(checkpoint.get('signal_mean', 0.0))
        signal_std = float(checkpoint.get('signal_std', 1.0))
        checkpoint_info = checkpoint
    else:
        state_dict = checkpoint
        latent_dim = 64
        model_variant = 'baseline'
        decoder_variant = 'standard'
        signal_mean = None
        signal_std = None
        checkpoint_info = {'model_state_dict': checkpoint}

    model = PINN(
        signal_length=signal_length,
        latent_dim=latent_dim,
        model_variant=model_variant,
        decoder_variant=decoder_variant,
    ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = {
            key.replace('module.', '', 1): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)

    model.eval()
    return model, signal_mean, signal_std, checkpoint_info


@torch.no_grad()
def predict_batch_maps(model, signals, coords, grid_shape, device, point_chunk=4096):
    model.eval()
    signals = signals.to(device)
    pred_chunks = []

    for start in range(0, coords.shape[0], point_chunk):
        coord_chunk = coords[start:start + point_chunk]
        pred_chunk = model(signals, coord_chunk)
        pred_chunks.append(pred_chunk.cpu())

    pred = torch.cat(pred_chunks, dim=1).numpy()
    return pred.reshape(signals.shape[0], *grid_shape) * MU_SCALE


def mask_center(mask, x_grid, y_grid):
    if not np.any(mask):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([
        float(x_grid[mask].mean()),
        float(y_grid[mask].mean()),
    ], dtype=np.float32)


def compute_sample_metrics(pred_mu, true_mu, x_grid, y_grid, threshold=MASK_THRESHOLD):
    pred_mask = pred_mu < threshold
    true_mask = true_mu < threshold

    mse = float(np.mean((pred_mu - true_mu) ** 2))
    mae = float(np.mean(np.abs(pred_mu - true_mu)))

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
        'mse': mse,
        'mae': mae,
        'iou': float(iou),
        'dice': float(dice),
        'area_error': float(area_error),
        'center_error': center_error,
        'pred_area': float(pred_area),
        'true_area': float(true_area),
        'pred_center_x': float(pred_center[0]),
        'pred_center_y': float(pred_center[1]),
        'true_center_x': float(true_center[0]),
        'true_center_y': float(true_center[1]),
    }, pred_mask, true_mask


def compute_mu_calibration(pred_mu, true_mask):
    defect_values = pred_mu[true_mask]
    background_values = pred_mu[~true_mask]
    return defect_values.astype(np.float64), background_values.astype(np.float64)


def average_metrics(rows):
    metric_names = ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']
    return {
        name: float(np.nanmean([row[name] for row in rows]))
        for name in metric_names
    }


def save_metrics_txt(avg, rows, output_path, checkpoint_path, threshold):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('PINN evaluation metrics\n')
        f.write(f'checkpoint: {checkpoint_path}\n')
        f.write(f'test_samples: {len(rows)}\n')
        f.write(f'mask_threshold: mu < {threshold}\n')
        f.write('area_error: relative absolute area error\n')
        f.write('center_error: Euclidean center distance in coordinate units (mm)\n\n')
        for key in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']:
            f.write(f'{key}: {avg[key]:.8e}\n')


def save_metrics_csv(rows, output_path):
    fieldnames = [
        'sample_index',
        'defect_type',
        'area_bin',
        'num_defects',
        'complexity_level',
        'mse',
        'mae',
        'iou',
        'dice',
        'area_error',
        'center_error',
        'pred_area',
        'true_area',
        'pred_center_x',
        'pred_center_y',
        'true_center_x',
        'true_center_y',
        'pred_area_gt_true',
    ]
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_nanmean(rows, key):
    values = [row[key] for row in rows if key in row]
    if not values:
        return float('nan')
    return float(np.nanmean(values))


def summarize_mu_values(values):
    if values.size == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'p10': float('nan'),
            'p90': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
        }
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def build_summary_row(args, checkpoint_info, avg, rows, defect_mu_values, background_mu_values):
    checkpoint_args = checkpoint_info.get('args', {}) if isinstance(checkpoint_info, dict) else {}
    model_variant = checkpoint_args.get('model_variant', 'baseline')
    decoder_variant = checkpoint_args.get('decoder_variant', 'standard')
    polygon_rows = [row for row in rows if row['defect_type'] == 'polygon']
    small_polygon_rows = [
        row for row in polygon_rows
        if str(row.get('area_bin', '')) == 'small'
    ]
    medium_polygon_rows = [
        row for row in polygon_rows
        if str(row.get('area_bin', '')) == 'medium'
    ]
    multi_defect_rows = [row for row in rows if row['defect_type'] == 'multi_defect']

    defect_stats = summarize_mu_values(defect_mu_values)
    background_stats = summarize_mu_values(background_mu_values)
    summary_name = args.summary_name or model_variant

    return {
        'model_name': summary_name,
        'model_variant': model_variant,
        'decoder_variant': decoder_variant,
        'checkpoint': args.checkpoint,
        'test_samples': len(rows),
        'mse': avg['mse'],
        'mae': avg['mae'],
        'iou': avg['iou'],
        'dice': avg['dice'],
        'area_error': avg['area_error'],
        'center_error': avg['center_error'],
        'polygon_area_error': safe_nanmean(polygon_rows, 'area_error'),
        'small_polygon_count': len(small_polygon_rows),
        'small_polygon_pred_area_zero_count': sum(row['pred_area'] == 0.0 for row in small_polygon_rows),
        'small_polygon_iou_zero_count': sum(row['iou'] == 0.0 for row in small_polygon_rows),
        'small_polygon_iou': safe_nanmean(small_polygon_rows, 'iou'),
        'small_polygon_dice': safe_nanmean(small_polygon_rows, 'dice'),
        'medium_polygon_area_error': safe_nanmean(medium_polygon_rows, 'area_error'),
        'multi_defect_center_error': safe_nanmean(multi_defect_rows, 'center_error'),
        'pred_area_gt_true_count': sum(row['pred_area'] > row['true_area'] for row in rows),
        'defect_mu_mean': defect_stats['mean'],
        'defect_mu_median': defect_stats['median'],
        'defect_mu_p10': defect_stats['p10'],
        'defect_mu_p90': defect_stats['p90'],
        'defect_mu_min': defect_stats['min'],
        'defect_mu_max': defect_stats['max'],
        'background_mu_mean': background_stats['mean'],
        'background_mu_median': background_stats['median'],
    }


def save_summary_csv(summary_row, output_path, append=False):
    fieldnames = list(summary_row.keys())
    ensure_parent_dir(output_path)
    file_exists = os.path.exists(output_path)
    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not file_exists:
            writer.writeheader()
        writer.writerow(summary_row)


def save_comparison_figure(pred_mu, true_mu, pred_mask, true_mask, x, y, sample_idx, defect_type, output_path):
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    im0 = ax[0, 0].imshow(pred_mu, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=MU_SCALE)
    ax[0, 0].set_title(f'Predicted $\\mu_r$ [{defect_type.upper()}]')
    plt.colorbar(im0, ax=ax[0, 0], label='$\\mu_r$')

    im1 = ax[0, 1].imshow(true_mu, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=MU_SCALE)
    ax[0, 1].set_title(f'True $\\mu_r$ [{defect_type.upper()}]')
    plt.colorbar(im1, ax=ax[0, 1], label='$\\mu_r$')

    ax[1, 0].imshow(pred_mask, extent=extent, origin='lower', cmap='gray')
    ax[1, 0].set_title('Predicted defect mask')

    ax[1, 1].imshow(true_mask, extent=extent, origin='lower', cmap='gray')
    ax[1, 1].set_title('True defect mask')

    fig.suptitle(f'Test sample {sample_idx}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_test_set(args):
    if args.test_data is None:
        args.test_data = EVAL_DATASETS[args.dataset]
    os.makedirs(project_path('results'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_test = np.load(project_path(args.test_data), allow_pickle=False)
    signal_length = raw_test['signals'].shape[1]

    model, signal_mean, signal_std, checkpoint_info = load_model(args.checkpoint, signal_length, device)
    checkpoint_loaded = isinstance(checkpoint_info, dict)

    test_dataset = MFLDataset(
        args.test_data,
        signal_mean=signal_mean,
        signal_std=signal_std,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    coords = build_coord_grid(test_dataset.x, test_dataset.y).to(device)
    grid_shape = test_dataset.mu_maps.shape[1:]
    X, Y = np.meshgrid(test_dataset.x, test_dataset.y)

    rows = []
    figures = {}
    defect_mu_chunks = []
    background_mu_chunks = []
    figure_indices = set(np.linspace(0, len(test_dataset) - 1, min(args.num_figures, len(test_dataset)), dtype=int))

    print(f'Using device: {device}')
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Test samples: {len(test_dataset)}')

    for signals, mu_targets, batch_indices in test_loader:
        pred_maps = predict_batch_maps(
            model=model,
            signals=signals,
            coords=coords,
            grid_shape=grid_shape,
            device=device,
            point_chunk=args.point_chunk,
        )
        true_maps = mu_targets.numpy().reshape(-1, *grid_shape) * MU_SCALE

        for batch_pos, sample_idx_tensor in enumerate(batch_indices):
            sample_idx = int(sample_idx_tensor.item())
            defect_type = str(test_dataset.defect_types[sample_idx])
            metrics, pred_mask, true_mask = compute_sample_metrics(
                pred_maps[batch_pos],
                true_maps[batch_pos],
                X,
                Y,
                threshold=args.mask_threshold,
            )
            defect_values, background_values = compute_mu_calibration(pred_maps[batch_pos], true_mask)
            if defect_values.size:
                defect_mu_chunks.append(defect_values)
            if background_values.size:
                background_mu_chunks.append(background_values)
            metadata = test_dataset.metadata[sample_idx]
            row = {
                'sample_index': sample_idx,
                'defect_type': defect_type,
                'area_bin': str(metadata['area_bin']) if 'area_bin' in test_dataset.metadata.dtype.names else '',
                'num_defects': float(metadata['num_defects']) if 'num_defects' in test_dataset.metadata.dtype.names else float('nan'),
                'complexity_level': float(metadata['complexity_level']) if 'complexity_level' in test_dataset.metadata.dtype.names else float('nan'),
                **metrics,
                'pred_area_gt_true': bool(metrics['pred_area'] > metrics['true_area']),
            }
            rows.append(row)

            if sample_idx in figure_indices:
                figures[sample_idx] = (
                    pred_maps[batch_pos].copy(),
                    true_maps[batch_pos].copy(),
                    pred_mask.copy(),
                    true_mask.copy(),
                    defect_type,
                )

    rows.sort(key=lambda row: row['sample_index'])
    avg = average_metrics(rows)

    output_prefix = f'{args.output_prefix}_' if args.output_prefix else ''
    metrics_txt = project_path(args.metrics_txt) if args.metrics_txt else project_path('results', f'{output_prefix}evaluation_metrics.txt')
    metrics_csv = project_path(args.metrics_csv) if args.metrics_csv else project_path('results', f'{output_prefix}evaluation_metrics.csv')
    ensure_parent_dir(metrics_txt)
    ensure_parent_dir(metrics_csv)
    save_metrics_txt(avg, rows, metrics_txt, args.checkpoint, args.mask_threshold)
    save_metrics_csv(rows, metrics_csv)
    if args.summary_csv:
        defect_mu_values = np.concatenate(defect_mu_chunks) if defect_mu_chunks else np.array([], dtype=np.float64)
        background_mu_values = (
            np.concatenate(background_mu_chunks)
            if background_mu_chunks
            else np.array([], dtype=np.float64)
        )
        summary_csv = project_path(args.summary_csv)
        summary_row = build_summary_row(
            args=args,
            checkpoint_info=checkpoint_info,
            avg=avg,
            rows=rows,
            defect_mu_values=defect_mu_values,
            background_mu_values=background_mu_values,
        )
        save_summary_csv(summary_row, summary_csv, append=args.summary_append)

    figure_paths = []
    figures_dir = project_path(args.figures_dir) if args.figures_dir else project_path('results')
    os.makedirs(figures_dir, exist_ok=True)
    for sample_idx in sorted(figures):
        pred_mu, true_mu, pred_mask, true_mask, defect_type = figures[sample_idx]
        output_path = os.path.join(figures_dir, f'{output_prefix}evaluation_sample_{sample_idx:03d}.png')
        save_comparison_figure(
            pred_mu=pred_mu,
            true_mu=true_mu,
            pred_mask=pred_mask,
            true_mask=true_mask,
            x=test_dataset.x,
            y=test_dataset.y,
            sample_idx=sample_idx,
            defect_type=defect_type,
            output_path=output_path,
        )
        figure_paths.append(output_path)

    print('Average metrics:')
    for key in ['mse', 'mae', 'iou', 'dice', 'area_error', 'center_error']:
        print(f'{key}: {avg[key]:.8e}')
    print(f'Saved metrics txt to {metrics_txt}')
    print(f'Saved metrics csv to {metrics_csv}')
    if args.summary_csv:
        print(f'Saved summary csv to {summary_csv}')
    for path in figure_paths:
        print(f'Saved comparison figure to {path}')
    print(f'checkpoint_loaded: {checkpoint_loaded}')

    return avg, rows, figure_paths, checkpoint_loaded


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Bz + coordinate PINN on the test set.')
    parser.add_argument('--dataset', choices=sorted(EVAL_DATASETS), default='simple')
    parser.add_argument('--test-data', default=None)
    parser.add_argument('--checkpoint', '--model', dest='checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--output_prefix', '--output-prefix', dest='output_prefix', default='')
    parser.add_argument('--metrics-txt', default=None)
    parser.add_argument('--metrics-csv', default=None)
    parser.add_argument('--figures-dir', default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--point-chunk', type=int, default=4096)
    parser.add_argument('--mask-threshold', type=float, default=MASK_THRESHOLD)
    parser.add_argument('--num-figures', type=int, default=3)
    parser.add_argument('--summary-csv', default=None)
    parser.add_argument('--summary-name', default='')
    parser.add_argument('--summary-append', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_test_set(parse_args())
