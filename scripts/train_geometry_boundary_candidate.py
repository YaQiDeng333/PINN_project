import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import (  # noqa: E402
    BzEncoder,
    MFLDataset,
    project_path,
    set_seed,
    signal_shape_info,
)
from scripts.train_mask_boundary_grid_candidate import (  # noqa: E402
    CURRENT_BASELINE_THRESHOLD,
    MaskBoundaryGridModel,
    aggregate_seed_rows,
    area_bin,
    compute_mask_metrics,
    get_area_edges,
    load_grid_checkpoint,
    make_loader,
    mask_loss,
    predict_prob_maps,
    safe_nanmean,
    safe_nanstd,
    summarize_candidate,
    summarize_samples,
    threshold_matches,
)


TRAIN_DATA = 'data/training_data_v3_complex_train.npz'
VAL_DATA = 'data/training_data_v3_complex_val.npz'
TEST_DATA = 'data/training_data_v3_complex_test.npz'

CURRENT_BASELINE_CHECKPOINTS = {
    42: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed42.pt',
    123: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed123.pt',
    2026: 'checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed2026.pt',
}

SEEDS = [42, 123, 2026]
EPOCHS = 50
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 1e-3
LATENT_DIM = 64
MASK_THRESHOLD_NORM = 0.5
TRAIN_SELECTION_THRESHOLD = 0.5
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
SINGLE_DEFECT_TYPES = {'polygon', 'rotated_rect', 'rect', 'circle', 'ellipse'}
RASTER_TEMPERATURE = 0.08
POS_WEIGHT_CAP = 8.0
REUSE_EXISTING = os.environ.get('PINN_REUSE_EXISTING', '').lower() in {'1', 'true', 'yes'}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'geometry_boundary_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_geometry_boundary_candidate_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_geometry_boundary_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'geometry_boundary_candidate'

METRIC_KEYS = [
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


class SingleDefectDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, signal_mean=None, signal_std=None):
        self.base = MFLDataset(npz_path, signal_mean=signal_mean, signal_std=signal_std)
        defect_types = np.asarray([str(item) for item in self.base.defect_types])
        keep = np.array([item in SINGLE_DEFECT_TYPES and item != 'multi_defect' for item in defect_types], dtype=bool)
        if not np.any(keep):
            raise ValueError(f'No single-defect samples found in {npz_path}')
        self.original_indices = np.where(keep)[0].astype(np.int64)
        self.signals = self.base.signals[keep]
        self.mu_maps = self.base.mu_maps[keep]
        self.defect_types = self.base.defect_types[keep]
        self.metadata = self.base.metadata[keep]
        self.metadata_keys = self.base.metadata_keys
        self.x = self.base.x
        self.y = self.base.y
        self.depths = self.base.depths[keep]
        self.lift_offs = self.base.lift_offs[keep]
        self.signal_mean = self.base.signal_mean
        self.signal_std = self.base.signal_std
        self.npz_path = npz_path

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.mu_maps[idx].reshape(-1)),
            idx,
        )


class GeometryBoundaryModel(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channels=1,
        latent_dim=64,
        x_range=(-15.0, 15.0),
        y_range=(0.0, 10.0),
        out_shape=(100, 200),
        temperature=RASTER_TEMPERATURE,
    ):
        super().__init__()
        self.x_min, self.x_max = map(float, x_range)
        self.y_min, self.y_max = map(float, y_range)
        self.out_shape = tuple(out_shape)
        self.temperature = float(temperature)
        self.min_w = 0.20
        self.min_h = 0.20
        self.max_w = 10.0
        self.max_h = 8.0
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.bias[:] = torch.tensor([0.0, 0.0, -0.7, -0.7, 0.0])

    def decode_params(self, raw_params):
        cx = self.x_min + torch.sigmoid(raw_params[:, 0]) * (self.x_max - self.x_min)
        cy = self.y_min + torch.sigmoid(raw_params[:, 1]) * (self.y_max - self.y_min)
        width = self.min_w + torch.sigmoid(raw_params[:, 2]) * (self.max_w - self.min_w)
        height = self.min_h + torch.sigmoid(raw_params[:, 3]) * (self.max_h - self.min_h)
        angle = np.pi * torch.tanh(raw_params[:, 4])
        return torch.stack([cx, cy, width, height, angle], dim=1)

    def rasterize(self, params, x_coords, y_coords):
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        x_grid = xx.to(params.device, dtype=params.dtype)[None, :, :]
        y_grid = yy.to(params.device, dtype=params.dtype)[None, :, :]

        cx = params[:, 0, None, None]
        cy = params[:, 1, None, None]
        width = params[:, 2, None, None]
        height = params[:, 3, None, None]
        angle = params[:, 4, None, None]

        dx = x_grid - cx
        dy = y_grid - cy
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        local_x = cos_a * dx + sin_a * dy
        local_y = -sin_a * dx + cos_a * dy

        margin_x = 0.5 * width - torch.abs(local_x)
        margin_y = 0.5 * height - torch.abs(local_y)
        inside_margin = torch.minimum(margin_x, margin_y)
        logits = inside_margin / self.temperature
        return logits.reshape(params.shape[0], -1)

    def forward(self, bz_signal, coords=None, return_params=False):
        latent = self.bz_encoder(bz_signal)
        params = self.decode_params(self.head(latent))
        if coords is None:
            raise ValueError('GeometryBoundaryModel requires coordinate tuple (x_coords, y_coords).')
        x_coords, y_coords = coords
        logits = self.rasterize(params, x_coords, y_coords)
        if return_params:
            return logits, params
        return logits


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_current_baseline_checkpoints():
    missing = [path for path in CURRENT_BASELINE_CHECKPOINTS.values() if not Path(project_path(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing current grid decoder checkpoints: ' + ', '.join(missing))


def split_type_counts(dataset):
    values, counts = np.unique(dataset.defect_types.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(values, counts)}


def dataset_low_signal_indices(dataset):
    signals = np.asarray(dataset.signals, dtype=np.float32)
    flat = signals if signals.ndim == 2 else signals.reshape(signals.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    threshold = np.quantile(max_abs, 1 / 3)
    return {int(idx) for idx, value in enumerate(max_abs) if float(value) <= float(threshold)}


def compute_pos_weight(dataset):
    masks = dataset.mu_maps < MASK_THRESHOLD_NORM
    pos = float(masks.sum())
    neg = float(masks.size - masks.sum())
    raw = np.sqrt(neg / max(pos, 1.0))
    return float(min(raw, POS_WEIGHT_CAP)), float(pos / masks.size)


def coord_tuple(dataset, device):
    return (
        torch.from_numpy(dataset.x.astype(np.float32)).to(device),
        torch.from_numpy(dataset.y.astype(np.float32)).to(device),
    )


def subset_sample_rows(candidate, seed, split, threshold, prob_maps, true_masks, dataset, area_edges, low_signal_indices):
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
        })
        rows.append(metrics)
    return rows


@torch.no_grad()
def predict_geometry_maps(model, dataset, coords, device):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    params = np.empty((len(dataset), 5), dtype=np.float32)
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        logits, batch_params = model(signals, coords, return_params=True)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = probs[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
            params[sample_idx] = batch_params[batch_pos].detach().cpu().numpy()
    return prob_maps, true_masks, params


def evaluate_model_for_selection(model, dataset, coords, device, area_edges):
    prob_maps, true_masks, _ = predict_geometry_maps(model, dataset, coords, device)
    rows = subset_sample_rows(
        candidate='selection',
        seed='selection',
        split='val',
        threshold=TRAIN_SELECTION_THRESHOLD,
        prob_maps=prob_maps,
        true_masks=true_masks,
        dataset=dataset,
        area_edges=area_edges,
        low_signal_indices=set(),
    )
    return summarize_samples(rows)


def train_one_seed(seed, device, pos_weight_value):
    set_seed(seed)
    train_dataset = SingleDefectDataset(TRAIN_DATA)
    val_dataset = SingleDefectDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
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
    best_path = CHECKPOINT_DIR / f'best_geometry_boundary_seed{seed}.pt'

    if REUSE_EXISTING and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu')
        info = checkpoint.get('val_metrics')
        print(f'Reusing existing checkpoint for seed={seed}: {best_path}')
        return best_path, info

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_samples = 0
        for signals, mu_targets, indices in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            mask_logits = model(signals, coords)
            loss, bce, dice = mask_loss(mask_logits, target_mask, pos_weight)
            loss.backward()
            optimizer.step()

            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
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
                    'model': 'geometry_boundary_rotated_box_model',
                    'dataset': 'v3_complex_single_defect_polygon_rotated_rect',
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'loss': 'BCEWithLogits + soft Dice on differentiably rasterized rotated box mask',
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'geometry_params': 'cx, cy, width, height, angle',
                    'rasterizer': 'PyTorch soft rectangle SDF logits = min(w/2-|x_local|, h/2-|y_local|) / temperature',
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
            f"bce={total_bce / total_samples:.6e} | "
            f"dice_loss={total_dice / total_samples:.6e} | "
            f"val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | "
            f"score={selection_score:.6e}"
        )

    return best_path, best_info


def load_geometry_checkpoint(path, signal_length, signal_channels, out_shape, dataset, device):
    checkpoint = torch.load(path, map_location=device)
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


def evaluate_baseline_family(split, data_path, device):
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
        prob_maps, true_masks = predict_prob_maps(model, dataset, coords, device)
        sample_rows = subset_sample_rows(
            candidate='current_grid_baseline_single_defect',
            seed=seed,
            split=split,
            threshold=CURRENT_BASELINE_THRESHOLD,
            prob_maps=prob_maps,
            true_masks=true_masks,
            dataset=dataset,
            area_edges=area_edges,
            low_signal_indices=low_signal_indices,
        )
        metric_rows.extend(
            summarize_candidate(
                sample_rows,
                'current_grid_baseline_single_defect',
                seed,
                split,
                CURRENT_BASELINE_THRESHOLD,
            )
        )
    metric_rows.extend(
        aggregate_seed_rows(
            metric_rows,
            'current_grid_baseline_single_defect',
            split,
            CURRENT_BASELINE_THRESHOLD,
        )
    )
    return metric_rows


def evaluate_geometry_family(checkpoints, split, data_path, thresholds, device):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    param_cache = {}
    area_dataset = SingleDefectDataset(data_path)
    area_edges = get_area_edges(area_dataset)
    low_signal_indices = dataset_low_signal_indices(area_dataset)
    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dataset = SingleDefectDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        model, _ = load_geometry_checkpoint(checkpoint_path, signal_length, signal_channels, out_shape, dataset, device)
        coords = coord_tuple(dataset, device)
        prob_maps, true_masks, params = predict_geometry_maps(model, dataset, coords, device)
        param_cache[seed] = (params, prob_maps, true_masks, dataset)
        for threshold in thresholds:
            sample_rows = subset_sample_rows(
                candidate='geometry_boundary_candidate',
                seed=seed,
                split=split,
                threshold=threshold,
                prob_maps=prob_maps,
                true_masks=true_masks,
                dataset=dataset,
                area_edges=area_edges,
                low_signal_indices=low_signal_indices,
            )
            sample_rows_by_seed_threshold[(seed, threshold)] = sample_rows
            metric_rows.extend(
                summarize_candidate(
                    sample_rows,
                    'geometry_boundary_candidate',
                    seed,
                    split,
                    threshold,
                )
            )
    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows(metric_rows, 'geometry_boundary_candidate', split, threshold))
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
    if not selected:
        return None
    return selected[0]


def select_threshold(rows):
    candidates = [
        row for row in rows
        if row['candidate'] == 'geometry_boundary_candidate_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    if not candidates:
        raise ValueError('No validation threshold rows found for geometry candidate.')
    return max(candidates, key=lambda row: float(row['composite']))


def write_metrics(rows):
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
        'composite',
        'macro_area_composite',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def fmt(value, metric):
    if value is None:
        return 'N/A'
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_comparison_table(rows, group_type, groups, selected_threshold):
    lines = [
        '| group | candidate | threshold | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group in groups:
        base_mean = find_row(rows, 'current_grid_baseline_single_defect_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        base_std = find_row(rows, 'current_grid_baseline_single_defect_std', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        geom_mean = find_row(rows, 'geometry_boundary_candidate_mean', group_type, group, threshold=selected_threshold)
        geom_std = find_row(rows, 'geometry_boundary_candidate_std', group_type, group, threshold=selected_threshold)
        if base_mean is not None:
            lines.append(
                f"| {group} | current grid baseline on same subset | {CURRENT_BASELINE_THRESHOLD:.2f} | "
                f"{metric_with_std(base_mean, base_std, 'iou')} | "
                f"{metric_with_std(base_mean, base_std, 'dice')} | "
                f"{metric_with_std(base_mean, base_std, 'area_error')} | "
                f"{metric_with_std(base_mean, base_std, 'center_error')} | "
                f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
                f"{metric_with_std(base_mean, base_std, 'pred_area_lt_true')} | "
                f"{metric_with_std(base_mean, base_std, 'pred_area_gt_true')} |"
            )
        if geom_mean is not None:
            lines.append(
                f"| {group} | geometry rotated-box candidate | {selected_threshold:.2f} | "
                f"{metric_with_std(geom_mean, geom_std, 'iou')} | "
                f"{metric_with_std(geom_mean, geom_std, 'dice')} | "
                f"{metric_with_std(geom_mean, geom_std, 'area_error')} | "
                f"{metric_with_std(geom_mean, geom_std, 'center_error')} | "
                f"{metric_with_std(geom_mean, geom_std, 'pred_area_zero')} | "
                f"{metric_with_std(geom_mean, geom_std, 'pred_area_lt_true')} | "
                f"{metric_with_std(geom_mean, geom_std, 'pred_area_gt_true')} |"
            )
    return '\n'.join(lines)


def format_val_scan(rows):
    lines = [
        '| threshold | val IoU | val Dice | val area_error | val center_error | val pred_area=0 | score |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    selected = [
        row for row in rows
        if row['candidate'] == 'geometry_boundary_candidate_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    for row in sorted(selected, key=lambda item: float(item['threshold'])):
        lines.append(
            f"| {float(row['threshold']):.2f} | {float(row['iou']):.4f} | "
            f"{float(row['dice']):.4f} | {float(row['area_error']):.4f} | "
            f"{float(row['center_error']):.4f} | {float(row['pred_area_zero']):.2f} | "
            f"{float(row['composite']):.4f} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, group_type, group, selected_threshold):
    baseline = find_row(rows, 'current_grid_baseline_single_defect_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
    candidate = find_row(rows, 'geometry_boundary_candidate_mean', group_type, group, threshold=selected_threshold)
    if baseline is None or candidate is None:
        return None
    return {
        'iou_not_obviously_down': float(candidate['iou']) >= float(baseline['iou']) - 0.02,
        'dice_not_obviously_down': float(candidate['dice']) >= float(baseline['dice']) - 0.02,
        'area_error_not_obviously_worse': float(candidate['area_error']) <= float(baseline['area_error']) + 0.05,
        'pred_area_zero_not_up': float(candidate['pred_area_zero']) <= float(baseline['pred_area_zero']) + 1e-6,
        'iou_improved': float(candidate['iou']) > float(baseline['iou']),
        'dice_improved': float(candidate['dice']) > float(baseline['dice']),
        'area_error_improved': float(candidate['area_error']) < float(baseline['area_error']),
    }


def rotated_box_corners(params):
    cx, cy, width, height, angle = [float(v) for v in params]
    half_w = 0.5 * width
    half_h = 0.5 * height
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h],
        [-half_w, -half_h],
    ], dtype=np.float32)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)


def select_preview_indices(rows):
    test_rows = [row for row in rows if row['split'] == 'test']
    polygon = sorted([row for row in test_rows if row['defect_type'] == 'polygon'], key=lambda row: float(row['iou']), reverse=True)
    rotated = sorted([row for row in test_rows if row['defect_type'] == 'rotated_rect'], key=lambda row: float(row['iou']), reverse=True)
    worst = sorted(test_rows, key=lambda row: float(row['iou']))
    median = sorted(test_rows, key=lambda row: abs(float(row['iou']) - safe_nanmean([float(r['iou']) for r in test_rows])))
    selected = []
    for group in [polygon[:3], rotated[:3], worst[:3], median[:3]]:
        for row in group:
            idx = int(row['sample_index'])
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= 12:
                return selected
    for row in sorted(test_rows, key=lambda row: float(row['iou']), reverse=True):
        idx = int(row['sample_index'])
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= 12:
            break
    return selected


def write_previews(seed, selected_threshold, sample_rows_by_seed_threshold, param_cache):
    rows = sample_rows_by_seed_threshold[(seed, selected_threshold)]
    params, prob_maps, true_masks, dataset = param_cache[seed]
    x_grid, y_grid = np.meshgrid(dataset.x, dataset.y)
    indices = select_preview_indices(rows)
    for rank, sample_idx in enumerate(indices, start=1):
        row = rows[sample_idx]
        prob = prob_maps[sample_idx]
        pred_mask = prob >= selected_threshold
        true_mask = true_masks[sample_idx]
        corners = rotated_box_corners(params[sample_idx])

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        axes[0].imshow(true_mask, origin='lower', extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()], cmap='gray')
        axes[0].set_title('true mask')
        im = axes[1].imshow(prob, origin='lower', extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()], cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('pred probability')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(pred_mask, origin='lower', extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()], cmap='gray')
        axes[2].plot(corners[:, 0], corners[:, 1], color='cyan', linewidth=1.5)
        axes[2].set_title(f'pred mask t={selected_threshold:.2f}')
        axes[3].imshow(true_mask, origin='lower', extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()], cmap='Greens', alpha=0.45)
        axes[3].imshow(pred_mask, origin='lower', extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()], cmap='Reds', alpha=0.35)
        axes[3].contour(x_grid, y_grid, true_mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        axes[3].contour(x_grid, y_grid, pred_mask.astype(float), levels=[0.5], colors='red', linewidths=1.0)
        axes[3].plot(corners[:, 0], corners[:, 1], color='cyan', linewidth=1.5)
        axes[3].set_title('overlay')
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        fig.suptitle(
            f"sample={int(row['original_index'])} subset_idx={sample_idx} type={row['defect_type']} "
            f"IoU={float(row['iou']):.3f} Dice={float(row['dice']):.3f} area_error={float(row['area_error']):.3f}",
            fontsize=10,
        )
        output_path = PREVIEW_DIR / f'geometry_boundary_seed{seed}_rank{rank:02d}_sample{int(row["original_index"])}_{row["defect_type"]}.png'
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def write_summary(rows, best_infos, checkpoint_paths, selected_threshold, train_counts, val_counts, test_counts, pos_weight, mask_fraction):
    overall = improvement_status(rows, 'overall', 'all', selected_threshold)
    polygon = improvement_status(rows, 'defect_type', 'polygon', selected_threshold)
    rotated = improvement_status(rows, 'defect_type', 'rotated_rect', selected_threshold)
    small = improvement_status(rows, 'area_bin', 'small', selected_threshold)
    low = improvement_status(rows, 'signal_bin', 'low_signal', selected_threshold)
    accepted = bool(
        overall
        and polygon
        and rotated
        and overall['iou_not_obviously_down']
        and overall['dice_not_obviously_down']
        and overall['area_error_not_obviously_worse']
        and overall['pred_area_zero_not_up']
        and polygon['iou_not_obviously_down']
        and rotated['iou_not_obviously_down']
        and polygon['area_error_not_obviously_worse']
        and rotated['area_error_not_obviously_worse']
    )

    best_lines = [
        '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | checkpoint |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    for info, checkpoint_path in zip(best_infos, checkpoint_paths):
        best_lines.append(
            f"| {info['seed']} | {info['epoch']} | {info['selection_score']:.6e} | "
            f"{info['val_iou']:.4f} | {info['val_dice']:.4f} | {info['val_area_error']:.4f} | "
            f"{checkpoint_path.relative_to(ROOT)} |"
        )

    status_lines = []
    for name, status in [('overall', overall), ('polygon', polygon), ('rotated_rect', rotated), ('small', small), ('low_signal', low)]:
        if status is None:
            status_lines.append(f'* {name}: not available')
        else:
            status_lines.append(
                f"* {name}: IoU improved={status['iou_improved']}, Dice improved={status['dice_improved']}, "
                f"area_error improved={status['area_error_improved']}, "
                f"IoU not obviously down={status['iou_not_obviously_down']}, "
                f"area_error not obviously worse={status['area_error_not_obviously_worse']}, "
                f"pred_area=0 not up={status['pred_area_zero_not_up']}"
            )

    summary = f"""# v3_complex single-defect geometry boundary candidate

This RESULT_DRIVEN_EXPERIMENT trains an independent single-defect geometry decoder candidate. It does not modify train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, PINN optimization route notes, or NEXT_STEP.md.

## Single-defect subset

The script loads v3_complex train / val / test and filters defect_types in the script. `multi_defect` is excluded. In this dataset the executable single-defect subset contains polygon and rotated_rect samples.

| split | polygon | rotated_rect | total |
|---|---:|---:|---:|
| train | {train_counts.get('polygon', 0)} | {train_counts.get('rotated_rect', 0)} | {sum(train_counts.values())} |
| val | {val_counts.get('polygon', 0)} | {val_counts.get('rotated_rect', 0)} | {sum(val_counts.values())} |
| test | {test_counts.get('polygon', 0)} | {test_counts.get('rotated_rect', 0)} | {sum(test_counts.values())} |

## Model and rasterizer

The model uses BzEncoder -> latent -> geometry head. The geometry head predicts a single rotated box:

* cx
* cy
* width
* height
* angle

The predicted parameters are converted into full-resolution mask logits with a pure PyTorch differentiable soft rectangle rasterizer. For each pixel, coordinates are rotated into the predicted box local coordinate frame. The logit is:

`min(width / 2 - abs(x_local), height / 2 - abs(y_local)) / temperature`

The fixed rasterization temperature is {RASTER_TEMPERATURE:.2f}. The model uses BCEWithLogits + soft Dice on the rendered mask. No geometry parameter MSE, SDF loss, boundary head, forward consistency, post-processing, polygon vertices, or multi-component matching is used.

Train mask positive fraction on the single-defect subset: {mask_fraction:.6f}. BCE pos_weight uses sqrt(neg/pos) capped at {POS_WEIGHT_CAP}; value used: {pos_weight:.6f}.

## Selected checkpoints

{chr(10).join(best_lines)}

## Validation threshold calibration

The three seeds share one validation-selected probability threshold. Threshold candidates: 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95. The validation selection score is IoU + Dice - area_error.

Selected threshold: {selected_threshold:.2f}

{format_val_scan(rows)}

## Overall single-defect test comparison

{format_comparison_table(rows, 'overall', ['all'], selected_threshold)}

## Defect-type test comparison

{format_comparison_table(rows, 'defect_type', ['polygon', 'rotated_rect'], selected_threshold)}

## Area-bin test comparison

{format_comparison_table(rows, 'area_bin', ['small', 'medium', 'large'], selected_threshold)}

## Low-signal test comparison

{format_comparison_table(rows, 'signal_bin', ['low_signal', 'non_low_signal'], selected_threshold)}

## Gate checks

{chr(10).join(status_lines)}

Accepted by metric gate: {accepted}

Visual note: preview PNGs are written to `results/previews/geometry_boundary_candidate`. Because the rasterizer is a single rotated rectangle by construction, the prediction is less round/blob-like than a free-form grid mask, but it only counts as useful if it aligns with the true polygon / rotated_rect and does not degrade IoU / Dice / area_error.

Conclusion: {'The single-defect rotated-box geometry parameterization passes the minimal metric gate on this subset.' if accepted else 'The single-defect rotated-box geometry parameterization does not pass the minimal acceptance gate; do not continue geometry v2, polygon vertices, multi-component geometry, or forward consistency from this result alone.'}
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return accepted


def main():
    ensure_outputs()
    check_current_baseline_checkpoints()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = SingleDefectDataset(TRAIN_DATA)
    val_dataset = SingleDefectDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    test_dataset = SingleDefectDataset(TEST_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    train_counts = split_type_counts(train_dataset)
    val_counts = split_type_counts(val_dataset)
    test_counts = split_type_counts(test_dataset)
    print(f'train single-defect counts: {train_counts}')
    print(f'val single-defect counts: {val_counts}')
    print(f'test single-defect counts: {test_counts}')

    pos_weight, mask_fraction = compute_pos_weight(train_dataset)
    print(f'train mask positive fraction={mask_fraction:.6f}, pos_weight={pos_weight:.6f}')

    checkpoint_paths = []
    best_infos = []
    for seed in SEEDS:
        checkpoint_path, best_info = train_one_seed(seed, device, pos_weight)
        checkpoint_paths.append(checkpoint_path)
        best_infos.append(best_info)

    geometry_checkpoints = {seed: path for seed, path in zip(SEEDS, checkpoint_paths)}
    baseline_test_rows = evaluate_baseline_family('test', TEST_DATA, device)
    geometry_val_rows, _, _ = evaluate_geometry_family(geometry_checkpoints, 'val', VAL_DATA, THRESHOLDS, device)
    selected = select_threshold(geometry_val_rows)
    selected_threshold = float(selected['threshold'])
    geometry_test_rows, sample_rows_by_seed_threshold, param_cache = evaluate_geometry_family(
        geometry_checkpoints,
        'test',
        TEST_DATA,
        THRESHOLDS,
        device,
    )
    all_rows = baseline_test_rows + geometry_val_rows + geometry_test_rows
    write_metrics(all_rows)
    accepted = write_summary(
        all_rows,
        best_infos,
        checkpoint_paths,
        selected_threshold,
        train_counts,
        val_counts,
        test_counts,
        pos_weight,
        mask_fraction,
    )
    write_previews(SEEDS[0], selected_threshold, sample_rows_by_seed_threshold, param_cache)

    overall_mean = find_row(all_rows, 'geometry_boundary_candidate_mean', 'overall', 'all', threshold=selected_threshold)
    print(f'selected_threshold={selected_threshold:.2f}')
    print(
        'geometry_test_mean: '
        f"IoU={float(overall_mean['iou']):.6f}, "
        f"Dice={float(overall_mean['dice']):.6f}, "
        f"area_error={float(overall_mean['area_error']):.6f}, "
        f"center_error={float(overall_mean['center_error']):.6f}, "
        f"pred_area_zero={float(overall_mean['pred_area_zero']):.2f}"
    )
    print(f'accepted={accepted}')
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')
    print(f'wrote previews: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
