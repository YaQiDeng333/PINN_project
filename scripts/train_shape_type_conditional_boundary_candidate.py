import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    MaskBoundaryGridModel,
    build_sample_rows,
    compute_pos_weight,
    evaluate_checkpoint_family as evaluate_grid_checkpoint_family,
    get_area_edges,
    load_low_signal_indices,
    make_loader,
    mask_loss,
    safe_nanmean,
    safe_nanstd,
    sample_mean_metrics,
    select_preview_samples,
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
SCREENING_SEED = 42
EPOCHS = 50
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 1e-3
LATENT_DIM = 64
TYPE_EMBED_DIM = 16
LAMBDA_TYPE = 0.1
MASK_THRESHOLD_NORM = 0.5
TRAIN_SELECTION_THRESHOLD = 0.5
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
CURRENT_BASELINE_THRESHOLD = 0.90
POS_WEIGHT_CAP = 8.0
GRID_BASE_CHANNELS = 64
GRID_LOW_SHAPE = (10, 20)
POSITIVE_SIGNAL_AREA_TOLERANCE = 0.02
RANDOM_ACCURACY_MARGIN = 0.12

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'shape_type_conditional_boundary_candidate'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_shape_type_conditional_boundary_candidate_metrics.csv'
SCREENING_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_shape_type_conditional_boundary_screening.csv'
THRESHOLD_RESCUE_METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_shape_type_conditional_boundary_threshold_rescue.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_shape_type_conditional_boundary_candidate_summary.txt'
PREVIEW_DIR = ROOT / 'results' / 'previews' / 'shape_type_conditional_boundary_candidate'

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


class ShapeTypeConditionalGridModel(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channels=1,
        latent_dim=64,
        type_count=3,
        type_embed_dim=16,
        out_shape=(100, 200),
        low_shape=GRID_LOW_SHAPE,
        base_channels=GRID_BASE_CHANNELS,
    ):
        super().__init__()
        self.out_shape = tuple(out_shape)
        self.low_shape = tuple(low_shape)
        self.base_channels = int(base_channels)
        self.type_count = int(type_count)
        self.type_embed_dim = int(type_embed_dim)
        self.bz_encoder = BzEncoder(
            signal_length=signal_length,
            signal_channels=signal_channels,
            latent_dim=latent_dim,
        )
        self.type_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.type_count),
        )
        self.type_embedding = nn.Embedding(self.type_count, self.type_embed_dim)
        low_h, low_w = self.low_shape
        self.project = nn.Sequential(
            nn.Linear(latent_dim + self.type_embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, self.base_channels * low_h * low_w),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, bz_signal, oracle_type_idx=None):
        batch_size = bz_signal.shape[0]
        latent = self.bz_encoder(bz_signal)
        type_logits = self.type_classifier(latent)
        if oracle_type_idx is None:
            type_probs = torch.softmax(type_logits, dim=1)
            type_embed = type_probs @ self.type_embedding.weight
        else:
            type_embed = self.type_embedding(oracle_type_idx)
        low_h, low_w = self.low_shape
        features = self.project(torch.cat([latent, type_embed], dim=1)).view(
            batch_size,
            self.base_channels,
            low_h,
            low_w,
        )
        logits = self.decoder(features)
        if tuple(logits.shape[-2:]) != self.out_shape:
            logits = F.interpolate(logits, size=self.out_shape, mode='bilinear', align_corners=False)
        return logits[:, 0].reshape(batch_size, -1), type_logits


def ensure_outputs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREENING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_RESCUE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def check_current_baseline_checkpoints():
    missing = [path for path in CURRENT_BASELINE_CHECKPOINTS.values() if not Path(project_path(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing current mask boundary grid checkpoints: ' + ', '.join(missing))


def load_type_labels(data_path):
    data = np.load(project_path(data_path), allow_pickle=False)
    if 'defect_types' not in data.files:
        return None
    labels = np.array([str(value) for value in data['defect_types']])
    if labels.size == 0:
        return None
    return labels


def build_type_mapping():
    labels_by_split = {}
    for split, path in [('train', TRAIN_DATA), ('val', VAL_DATA), ('test', TEST_DATA)]:
        labels = load_type_labels(path)
        if labels is None:
            return None, None
        labels_by_split[split] = labels
    classes = sorted(set(labels_by_split['train'].tolist()))
    if not classes:
        return None, None
    type_to_idx = {name: idx for idx, name in enumerate(classes)}
    for split, labels in labels_by_split.items():
        unknown = sorted(set(labels.tolist()) - set(classes))
        if unknown:
            raise ValueError(f'{split} contains shape types not present in train: {unknown}')
    counts = {
        split: {name: int(np.sum(labels == name)) for name in classes}
        for split, labels in labels_by_split.items()
    }
    return type_to_idx, counts


def type_indices_for_dataset(dataset, type_to_idx):
    return torch.tensor([type_to_idx[str(value)] for value in dataset.defect_types], dtype=torch.long)


def dataset_low_signal_indices(data_path):
    dataset = MFLDataset(data_path)
    signals = np.asarray(dataset.signals, dtype=np.float32)
    flat = signals if signals.ndim == 2 else signals.reshape(signals.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    threshold = np.quantile(max_abs, 1 / 3)
    return {int(idx) for idx, value in enumerate(max_abs) if float(value) <= float(threshold)}


def load_conditional_checkpoint(path, signal_length, signal_channels, out_shape, device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get('args', {})
    model = ShapeTypeConditionalGridModel(
        signal_length=signal_length,
        signal_channels=int(args.get('signal_channels', signal_channels)),
        latent_dim=int(args.get('latent_dim', LATENT_DIM)),
        type_count=int(args.get('type_count', 3)),
        type_embed_dim=int(args.get('type_embed_dim', TYPE_EMBED_DIM)),
        out_shape=tuple(args.get('out_shape', out_shape)),
        low_shape=tuple(args.get('low_shape', GRID_LOW_SHAPE)),
        base_channels=int(args.get('base_channels', GRID_BASE_CHANNELS)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def type_accuracy(type_logits, type_targets):
    pred = torch.argmax(type_logits, dim=1)
    return float((pred == type_targets).float().mean().item())


@torch.no_grad()
def predict_conditioned_prob_maps(model, dataset, type_to_idx, device, oracle_type=False):
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    grid_shape = dataset.mu_maps.shape[1:]
    prob_maps = np.empty((len(dataset), *grid_shape), dtype=np.float32)
    true_masks = np.empty((len(dataset), *grid_shape), dtype=bool)
    pred_type_idx = np.empty((len(dataset),), dtype=np.int64)
    true_type_idx_all = type_indices_for_dataset(dataset, type_to_idx)
    correct = 0
    total = 0
    model.eval()
    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        idx_np = indices.numpy().astype(np.int64)
        true_type_idx = true_type_idx_all[idx_np].to(device)
        oracle_arg = true_type_idx if oracle_type else None
        logits, type_logits = model(signals, oracle_arg)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(signals.shape[0], *grid_shape)
        batch_true = mu_targets.numpy().reshape(signals.shape[0], *grid_shape) < MASK_THRESHOLD_NORM
        pred_idx = torch.argmax(type_logits, dim=1)
        correct += int((pred_idx == true_type_idx).sum().item())
        total += int(signals.shape[0])
        for batch_pos, sample_idx_tensor in enumerate(indices):
            sample_idx = int(sample_idx_tensor.item())
            prob_maps[sample_idx] = probs[batch_pos]
            true_masks[sample_idx] = batch_true[batch_pos]
            pred_type_idx[sample_idx] = int(pred_idx[batch_pos].item())
    return prob_maps, true_masks, float(correct / max(total, 1)), pred_type_idx


def evaluate_model_for_selection(model, dataset, type_to_idx, device, area_edges):
    prob_maps, true_masks, acc, _ = predict_conditioned_prob_maps(model, dataset, type_to_idx, device, oracle_type=False)
    rows = build_sample_rows(
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
    summary = summarize_samples(rows)
    summary['type_accuracy'] = acc
    return summary


def train_one_seed(seed, device, pos_weight_value, type_to_idx, reuse_existing=False):
    set_seed(seed)
    train_dataset = MFLDataset(TRAIN_DATA)
    val_dataset = MFLDataset(VAL_DATA, signal_mean=train_dataset.signal_mean, signal_std=train_dataset.signal_std)
    val_area_edges = get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    out_shape = tuple(train_dataset.mu_maps.shape[1:])
    model = ShapeTypeConditionalGridModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=LATENT_DIM,
        type_count=len(type_to_idx),
        type_embed_dim=TYPE_EMBED_DIM,
        out_shape=out_shape,
    ).to(device)
    train_type_idx = type_indices_for_dataset(train_dataset, type_to_idx)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    best_score = -float('inf')
    best_info = None
    best_path = CHECKPOINT_DIR / f'best_shape_type_conditional_seed{seed}.pt'

    if reuse_existing and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu')
        info = checkpoint.get('val_metrics')
        if info is None:
            info = {
                'seed': seed,
                'epoch': int(checkpoint.get('epoch', 0)),
                'selection_score': float(checkpoint.get('selection_score', float('nan'))),
                'val_iou': float('nan'),
                'val_dice': float('nan'),
                'val_area_error': float('nan'),
                'val_center_error': float('nan'),
                'val_pred_area_zero': float('nan'),
                'val_type_accuracy': float('nan'),
            }
        print(f'Reusing existing shape-type checkpoint for seed={seed}: {best_path}')
        return best_path, info

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_mask = 0.0
        total_type = 0.0
        total_type_acc = 0.0
        total_samples = 0
        for signals, mu_targets, indices in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            type_targets = train_type_idx[indices.numpy().astype(np.int64)].to(device)
            optimizer.zero_grad(set_to_none=True)
            mask_logits, type_logits = model(signals)
            mask_loss_value, bce, dice = mask_loss(mask_logits, target_mask, pos_weight)
            ce = F.cross_entropy(type_logits, type_targets)
            loss = mask_loss_value + LAMBDA_TYPE * ce
            loss.backward()
            optimizer.step()

            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_mask += float(mask_loss_value.item()) * batch_size
            total_type += float(ce.item()) * batch_size
            total_type_acc += type_accuracy(type_logits.detach(), type_targets) * batch_size
            total_samples += batch_size

        val_summary = evaluate_model_for_selection(model, val_dataset, type_to_idx, device, val_area_edges)
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
                'val_type_accuracy': val_summary['type_accuracy'],
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': {
                    'model': 'shape_type_conditional_grid_model',
                    'dataset': 'v3_complex',
                    'seed': seed,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'latent_dim': LATENT_DIM,
                    'type_count': len(type_to_idx),
                    'type_embed_dim': TYPE_EMBED_DIM,
                    'lambda_type': LAMBDA_TYPE,
                    'loss': 'BCEWithLogits + soft Dice + 0.1 * shape_type CE',
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'decoder': 'latent plus soft predicted shape-type embedding into grid decoder',
                    'out_shape': out_shape,
                    'low_shape': GRID_LOW_SHAPE,
                    'base_channels': GRID_BASE_CHANNELS,
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
                    'signal_channels': signal_channels,
                    'type_to_idx': type_to_idx,
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
            f"mask={total_mask / total_samples:.6e} | "
            f"type_ce={total_type / total_samples:.6e} | "
            f"train_type_acc={total_type_acc / total_samples:.4f} | "
            f"val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | "
            f"val_type_acc={val_summary['type_accuracy']:.4f} | "
            f"score={selection_score:.6e}"
        )

    return best_path, best_info


def add_type_accuracy(metric_rows, candidate, split, threshold, type_accuracy_value):
    for row in metric_rows:
        if (
            row['candidate'] == candidate
            and row['split'] == split
            and threshold_matches(row['threshold'], threshold)
            and row['group_type'] == 'overall'
            and row['group'] == 'all'
        ):
            row['type_accuracy'] = type_accuracy_value


def aggregate_seed_rows_with_type(metric_rows, source_candidate, split, threshold):
    aggregate_rows = []
    groups = sorted({
        (row['group_type'], row['group'])
        for row in metric_rows
        if row['candidate'] == source_candidate
        and row['split'] == split
        and threshold_matches(row['threshold'], threshold)
    })
    for group_type, group in groups:
        selected = [
            row for row in metric_rows
            if row['candidate'] == source_candidate
            and row['split'] == split
            and threshold_matches(row['threshold'], threshold)
            and row['group_type'] == group_type
            and row['group'] == group
        ]
        mean_row = {
            'candidate': f'{source_candidate}_mean',
            'seed': 'mean',
            'split': split,
            'group_type': group_type,
            'group': group,
            'threshold': threshold,
            'n': selected[0]['n'] if selected else 0,
        }
        std_row = dict(mean_row)
        std_row['candidate'] = f'{source_candidate}_std'
        std_row['seed'] = 'sample_std'
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in selected]
            mean_row[key] = safe_nanmean(values)
            std_row[key] = safe_nanstd(values)
        if group_type == 'overall' and group == 'all':
            type_values = [float(row['type_accuracy']) for row in selected if row.get('type_accuracy') not in (None, '')]
            if type_values:
                mean_row['type_accuracy'] = safe_nanmean(type_values)
                std_row['type_accuracy'] = safe_nanstd(type_values)
        aggregate_rows.extend([mean_row, std_row])
    return aggregate_rows


def evaluate_conditional_checkpoint_family(
    checkpoints,
    candidate,
    split,
    data_path,
    thresholds,
    device,
    area_edges,
    low_signal_indices,
    type_to_idx,
    oracle_type=False,
):
    metric_rows = []
    sample_rows_by_seed_threshold = {}
    prob_cache = {}
    type_acc_by_seed = {}

    for seed, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu')
        dataset = MFLDataset(
            data_path,
            signal_mean=float(checkpoint['signal_mean']),
            signal_std=float(checkpoint['signal_std']),
        )
        signal_length, signal_channels = signal_shape_info(dataset.signals)
        out_shape = tuple(dataset.mu_maps.shape[1:])
        model, _ = load_conditional_checkpoint(Path(project_path(checkpoint_path)), signal_length, signal_channels, out_shape, device)
        prob_maps, true_masks, acc, pred_type_idx = predict_conditioned_prob_maps(
            model,
            dataset,
            type_to_idx,
            device,
            oracle_type=oracle_type,
        )
        type_acc_by_seed[seed] = acc
        prob_cache[seed] = (prob_maps, true_masks, dataset, pred_type_idx)
        for threshold in thresholds:
            sample_rows = build_sample_rows(
                candidate=candidate,
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
            rows = summarize_candidate(sample_rows, candidate, seed, split, threshold)
            add_type_accuracy(rows, candidate, split, threshold, acc)
            metric_rows.extend(rows)

    for threshold in thresholds:
        metric_rows.extend(aggregate_seed_rows_with_type(metric_rows, candidate, split, threshold))
    return metric_rows, sample_rows_by_seed_threshold, prob_cache, type_acc_by_seed


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
        raise KeyError(f'Missing row candidate={candidate}, split={split}, group_type={group_type}, group={group}, threshold={threshold}')
    return selected[0]


def get_overall_mean(rows, candidate, threshold, split='test'):
    return find_row(rows, f'{candidate}_mean', 'overall', 'all', split=split, threshold=threshold)


def select_threshold(rows, baseline_overall):
    validation_means = [
        row for row in rows
        if row['candidate'] == 'shape_type_conditional_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    eligible = [
        row for row in validation_means
        if float(row['iou']) >= float(baseline_overall['iou']) - 1e-6
        and float(row['dice']) >= float(baseline_overall['dice']) - 1e-6
    ]
    if eligible:
        return min(
            eligible,
            key=lambda row: (
                float(row['area_error']),
                -float(row['composite']),
                float(row['pred_area_zero']),
            ),
        )
    return max(validation_means, key=lambda row: float(row['composite']))


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
        'type_accuracy',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_threshold_rescue_metrics(rows):
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
        'type_accuracy',
    ]
    with open(THRESHOLD_RESCUE_METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_screening_metrics(screening_records):
    fieldnames = [
        'stage',
        'candidate',
        'seed',
        'threshold',
        'type_accuracy',
        'iou',
        'dice',
        'area_error',
        'center_error',
        'pred_area_zero',
        'small_iou',
        'small_dice',
        'small_area_error',
        'low_signal_iou',
        'low_signal_dice',
        'low_signal_area_error',
        'polygon_iou',
        'polygon_dice',
        'polygon_area_error',
        'rotated_rect_iou',
        'rotated_rect_dice',
        'rotated_rect_area_error',
        'multi_defect_iou',
        'multi_defect_dice',
        'multi_defect_area_error',
        'positive_signal',
        'checkpoint',
    ]
    with open(SCREENING_METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in screening_records:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def screening_record_from_rows(stage, candidate_label, seed, threshold, rows, checkpoint, positive_signal=''):
    overall = find_row(rows, candidate_label, 'overall', 'all', split='val', threshold=threshold)
    small = find_row(rows, candidate_label, 'area_bin', 'small', split='val', threshold=threshold)
    low = find_row(rows, candidate_label, 'signal_bin', 'low_signal', split='val', threshold=threshold)
    polygon = find_row(rows, candidate_label, 'defect_type', 'polygon', split='val', threshold=threshold)
    rotated = find_row(rows, candidate_label, 'defect_type', 'rotated_rect', split='val', threshold=threshold)
    multi = find_row(rows, candidate_label, 'defect_type', 'multi_defect', split='val', threshold=threshold)
    return {
        'stage': stage,
        'candidate': candidate_label,
        'seed': seed,
        'threshold': threshold,
        'type_accuracy': overall.get('type_accuracy', ''),
        'iou': overall['iou'],
        'dice': overall['dice'],
        'area_error': overall['area_error'],
        'center_error': overall['center_error'],
        'pred_area_zero': overall['pred_area_zero'],
        'small_iou': small['iou'],
        'small_dice': small['dice'],
        'small_area_error': small['area_error'],
        'low_signal_iou': low['iou'],
        'low_signal_dice': low['dice'],
        'low_signal_area_error': low['area_error'],
        'polygon_iou': polygon['iou'],
        'polygon_dice': polygon['dice'],
        'polygon_area_error': polygon['area_error'],
        'rotated_rect_iou': rotated['iou'],
        'rotated_rect_dice': rotated['dice'],
        'rotated_rect_area_error': rotated['area_error'],
        'multi_defect_iou': multi['iou'],
        'multi_defect_dice': multi['dice'],
        'multi_defect_area_error': multi['area_error'],
        'positive_signal': positive_signal,
        'checkpoint': checkpoint,
    }


def positive_signal(
    pred_overall,
    pred_small,
    pred_low,
    pred_polygon,
    pred_rotated,
    baseline_overall,
    baseline_small,
    baseline_low,
    baseline_polygon,
    baseline_rotated,
    type_acc,
    class_count,
):
    random_floor = 1.0 / max(class_count, 1) + RANDOM_ACCURACY_MARGIN
    checks = {
        'overall_iou_not_down': float(pred_overall['iou']) >= float(baseline_overall['iou']) - 1e-6,
        'overall_dice_not_down': float(pred_overall['dice']) >= float(baseline_overall['dice']) - 1e-6,
        'overall_area_close': float(pred_overall['area_error']) <= float(baseline_overall['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE,
        'overall_pred_zero_not_up': float(pred_overall['pred_area_zero']) <= float(baseline_overall['pred_area_zero']) + 1e-6,
        'type_acc_not_random': float(type_acc) >= random_floor,
        'small_iou_not_much_down': float(pred_small['iou']) >= float(baseline_small['iou']) - 0.02,
        'small_dice_not_much_down': float(pred_small['dice']) >= float(baseline_small['dice']) - 0.02,
        'small_area_not_much_worse': float(pred_small['area_error']) <= float(baseline_small['area_error']) + 0.05,
        'low_signal_iou_not_much_down': float(pred_low['iou']) >= float(baseline_low['iou']) - 0.02,
        'low_signal_dice_not_much_down': float(pred_low['dice']) >= float(baseline_low['dice']) - 0.02,
        'low_signal_area_not_much_worse': float(pred_low['area_error']) <= float(baseline_low['area_error']) + 0.05,
    }
    polygon_improved = (
        float(pred_polygon['iou']) >= float(baseline_polygon['iou']) - 1e-6
        and float(pred_polygon['dice']) >= float(baseline_polygon['dice']) - 1e-6
        and float(pred_polygon['area_error']) <= float(baseline_polygon['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE
    )
    rotated_improved = (
        float(pred_rotated['iou']) >= float(baseline_rotated['iou']) - 1e-6
        and float(pred_rotated['dice']) >= float(baseline_rotated['dice']) - 1e-6
        and float(pred_rotated['area_error']) <= float(baseline_rotated['area_error']) + POSITIVE_SIGNAL_AREA_TOLERANCE
    )
    checks['polygon_or_rotated_improved'] = polygon_improved or rotated_improved
    return bool(all(checks.values())), checks


def fmt(value, metric):
    if value in ('', None):
        return 'N/A'
    if metric in ('pred_area_zero', 'pred_area_lt_true', 'pred_area_gt_true', 'n'):
        return f'{float(value):.2f}'
    return f'{float(value):.4f}'


def metric_with_std(mean_row, std_row, metric):
    return f"{fmt(mean_row[metric], metric)} +/- {fmt(std_row[metric], metric)}"


def format_type_counts(counts):
    lines = ['| split | multi_defect | polygon | rotated_rect |', '|---|---:|---:|---:|']
    for split in ['train', 'val', 'test']:
        row = counts[split]
        lines.append(f"| {split} | {row.get('multi_defect', 0)} | {row.get('polygon', 0)} | {row.get('rotated_rect', 0)} |")
    return '\n'.join(lines)


def format_screening_records(records):
    lines = [
        '| candidate | threshold | type_acc | IoU | Dice | area_error | pred_area=0 | small IoU | low-signal IoU | polygon IoU | rotated_rect IoU | multi_defect IoU | positive_signal |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in records:
        lines.append(
            f"| {row['candidate']} | {float(row['threshold']):.2f} | {fmt(row['type_accuracy'], 'iou')} | "
            f"{float(row['iou']):.4f} | {float(row['dice']):.4f} | {float(row['area_error']):.4f} | "
            f"{float(row['pred_area_zero']):.2f} | {float(row['small_iou']):.4f} | "
            f"{float(row['low_signal_iou']):.4f} | {float(row['polygon_iou']):.4f} | "
            f"{float(row['rotated_rect_iou']):.4f} | {float(row['multi_defect_iou']):.4f} | "
            f"{row['positive_signal']} |"
        )
    return '\n'.join(lines)


def format_val_scan(rows, baseline_overall):
    lines = [
        '| threshold | val IoU | val Dice | val area_error | val pred_area=0 | eligible |',
        '|---:|---:|---:|---:|---:|---|',
    ]
    selected = [
        row for row in rows
        if row['candidate'] == 'shape_type_conditional_val_scan_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    for row in sorted(selected, key=lambda item: float(item['threshold'])):
        eligible = float(row['iou']) >= float(baseline_overall['iou']) - 1e-6 and float(row['dice']) >= float(baseline_overall['dice']) - 1e-6
        lines.append(
            f"| {float(row['threshold']):.2f} | {float(row['iou']):.4f} | {float(row['dice']):.4f} | "
            f"{float(row['area_error']):.4f} | {float(row['pred_area_zero']):.2f} | {eligible} |"
        )
    return '\n'.join(lines)


def format_comparison_table(rows, candidate_name, selected_threshold, group_type, groups):
    lines = [
        '| group | candidate | threshold | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group in groups:
        base_mean = find_row(rows, 'current_mask_boundary_baseline_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        base_std = find_row(rows, 'current_mask_boundary_baseline_std', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
        cand_mean = find_row(rows, f'{candidate_name}_mean', group_type, group, threshold=selected_threshold)
        cand_std = find_row(rows, f'{candidate_name}_std', group_type, group, threshold=selected_threshold)
        lines.append(
            f"| {group} | current grid decoder baseline | {CURRENT_BASELINE_THRESHOLD:.2f} | "
            f"{metric_with_std(base_mean, base_std, 'iou')} | "
            f"{metric_with_std(base_mean, base_std, 'dice')} | "
            f"{metric_with_std(base_mean, base_std, 'area_error')} | "
            f"{metric_with_std(base_mean, base_std, 'center_error')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_zero')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(base_mean, base_std, 'pred_area_gt_true')} |"
        )
        lines.append(
            f"| {group} | shape-type conditional predicted-type | {selected_threshold:.2f} | "
            f"{metric_with_std(cand_mean, cand_std, 'iou')} | "
            f"{metric_with_std(cand_mean, cand_std, 'dice')} | "
            f"{metric_with_std(cand_mean, cand_std, 'area_error')} | "
            f"{metric_with_std(cand_mean, cand_std, 'center_error')} | "
            f"{metric_with_std(cand_mean, cand_std, 'pred_area_zero')} | "
            f"{metric_with_std(cand_mean, cand_std, 'pred_area_lt_true')} | "
            f"{metric_with_std(cand_mean, cand_std, 'pred_area_gt_true')} |"
        )
    return '\n'.join(lines)


def improvement_status(rows, candidate_name, group_type, group, selected_threshold):
    baseline = find_row(rows, 'current_mask_boundary_baseline_mean', group_type, group, threshold=CURRENT_BASELINE_THRESHOLD)
    candidate = find_row(rows, f'{candidate_name}_mean', group_type, group, threshold=selected_threshold)
    return {
        'iou_not_down': float(candidate['iou']) >= float(baseline['iou']) - 1e-6,
        'dice_not_down': float(candidate['dice']) >= float(baseline['dice']) - 1e-6,
        'area_error_close': float(candidate['area_error']) <= float(baseline['area_error']) + 0.02,
        'pred_area_zero_not_up': float(candidate['pred_area_zero']) <= float(baseline['pred_area_zero']) + 1e-6,
    }


def write_summary(
    rows,
    screening_records,
    type_to_idx,
    type_counts,
    best_infos,
    checkpoint_paths,
    selected_threshold,
    entered_stage_b,
    pos_weight,
    mask_fraction,
    accepted=False,
    preview_count=0,
):
    class_list = ', '.join(f'{name}={idx}' for name, idx in type_to_idx.items())
    screening_text = format_screening_records(screening_records)
    if not entered_stage_b:
        baseline_overall = find_row(rows, 'current_mask_boundary_baseline_val_reference_mean', 'overall', 'all', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
        summary = f"""# v3_complex shape-type conditional boundary candidate

Shape/type label exists: True

Shape/type classes: {class_list}

{format_type_counts(type_counts)}

This RESULT_DRIVEN_EXPERIMENT_PACK trains an independent shape-type conditional mask boundary model without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Model

The deployable path uses BzEncoder -> latent, predicts shape_type logits, converts the predicted softmax probabilities into a soft type embedding, concatenates the embedding with latent, and decodes a mask using the grid decoder. Inference does not use true shape_type.

Loss: BCEWithLogits + soft Dice + {LAMBDA_TYPE:.1f} * shape_type cross entropy. No SDF, boundary head/loss, mixture-of-experts v2, adaptive threshold, or post-processing is used.

Train mask positive fraction: {mask_fraction:.6f}; pos_weight={pos_weight:.6f}.

## Stage A: seed=42 screening

Threshold calibration rescue check is included in Stage A. The existing seed=42 checkpoint is evaluated on validation thresholds {', '.join(f'{value:.2f}' for value in THRESHOLDS)} for both predicted-type deployable mode and oracle true-type diagnostic mode. The threshold shown below is selected by IoU + Dice - area_error.

Current validation baseline at threshold={CURRENT_BASELINE_THRESHOLD:.2f}: IoU={float(baseline_overall['iou']):.4f}, Dice={float(baseline_overall['dice']):.4f}, area_error={float(baseline_overall['area_error']):.4f}, pred_area=0={float(baseline_overall['pred_area_zero']):.2f}.

{screening_text}

Oracle true-type upper-bound uses the same checkpoint and true shape_type embedding. It is diagnostic only and is not deployable.

Stage B entered: False
Validation selected threshold: N/A
Accepted by metric gate: False

The predicted-type model did not pass the validation positive-signal gate after threshold calibration, so no 3-seed expansion is run and no type-conditioned v2 or threshold variant is continued.
"""
        SUMMARY_PATH.write_text(summary, encoding='utf-8')
        return

    candidate_name = 'shape_type_conditional_test'
    overall = improvement_status(rows, candidate_name, 'overall', 'all', selected_threshold)
    small = improvement_status(rows, candidate_name, 'area_bin', 'small', selected_threshold)
    low = improvement_status(rows, candidate_name, 'signal_bin', 'low_signal', selected_threshold)
    polygon = improvement_status(rows, candidate_name, 'defect_type', 'polygon', selected_threshold)
    rotated = improvement_status(rows, candidate_name, 'defect_type', 'rotated_rect', selected_threshold)
    multi = improvement_status(rows, candidate_name, 'defect_type', 'multi_defect', selected_threshold)
    accepted = bool(
        overall['iou_not_down']
        and overall['dice_not_down']
        and overall['area_error_close']
        and overall['pred_area_zero_not_up']
        and small['iou_not_down']
        and low['iou_not_down']
        and (polygon['iou_not_down'] or rotated['iou_not_down'] or multi['iou_not_down'])
    )

    best_lines = [
        '| seed | best_epoch | best_val_score | val_IoU | val_Dice | val_area_error | val_type_acc | checkpoint |',
        '|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for info, checkpoint_path in zip(best_infos, checkpoint_paths):
        best_lines.append(
            f"| {info['seed']} | {info['epoch']} | {info['selection_score']:.6e} | "
            f"{info['val_iou']:.4f} | {info['val_dice']:.4f} | {info['val_area_error']:.4f} | "
            f"{info['val_type_accuracy']:.4f} | {checkpoint_path.relative_to(ROOT)} |"
        )

    baseline_overall = find_row(rows, 'current_mask_boundary_baseline_mean', 'overall', 'all', threshold=CURRENT_BASELINE_THRESHOLD)
    summary = f"""# v3_complex shape-type conditional boundary candidate

Shape/type label exists: True

Shape/type classes: {class_list}

{format_type_counts(type_counts)}

This RESULT_DRIVEN_EXPERIMENT_PACK trains an independent shape-type conditional mask boundary model without modifying train_pinn.py, evaluate_pinn.py, data_generator_v2.py, CURRENT_BASELINE.md, README.md, EXPERIMENT_LOG.md, the route document, or NEXT_STEP.md.

## Stage A: seed=42 screening

{screening_text}

Oracle true-type upper-bound uses true shape_type embedding and is diagnostic only; it is not deployable.

Stage B entered: True

## Selected checkpoints

{chr(10).join(best_lines)}

## Validation threshold calibration

Threshold candidates: {', '.join(f'{value:.2f}' for value in THRESHOLDS)}

Current baseline reference: threshold={CURRENT_BASELINE_THRESHOLD:.2f}, IoU={float(baseline_overall['iou']):.4f}, Dice={float(baseline_overall['dice']):.4f}, area_error={float(baseline_overall['area_error']):.4f}.

Selected threshold: {selected_threshold:.2f}

{format_val_scan(rows, baseline_overall)}

## Overall test comparison

{format_comparison_table(rows, candidate_name, selected_threshold, 'overall', ['all'])}

## Area-bin test comparison

{format_comparison_table(rows, candidate_name, selected_threshold, 'area_bin', ['small', 'medium', 'large'])}

## Low-signal test comparison

{format_comparison_table(rows, candidate_name, selected_threshold, 'signal_bin', ['low_signal', 'non_low_signal'])}

## Defect-type test comparison

{format_comparison_table(rows, candidate_name, selected_threshold, 'defect_type', ['polygon', 'rotated_rect', 'multi_defect'])}

## Gate checks

* overall: {overall}
* small: {small}
* low_signal: {low}
* polygon: {polygon}
* rotated_rect: {rotated}
* multi_defect: {multi}

Preview PNG count: {preview_count}

Accepted by metric gate: {accepted}
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def generate_previews(selected, prob_cache, sample_rows_by_seed_threshold, selected_threshold, type_to_idx):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    idx_to_type = {idx: name for name, idx in type_to_idx.items()}
    written = []
    for item in selected:
        idx = int(item['index'])
        seed_probs = []
        seed_rows = []
        pred_type_votes = []
        dataset = None
        true_masks = None
        for seed in SEEDS:
            prob_maps, masks, ds, pred_type_idx = prob_cache[seed]
            dataset = ds
            true_masks = masks
            seed_probs.append(prob_maps[idx])
            pred_type_votes.append(int(pred_type_idx[idx]))
            seed_rows.append(next(row for row in sample_rows_by_seed_threshold[(seed, selected_threshold)] if row['sample_index'] == idx))
        prob_map = np.mean(seed_probs, axis=0)
        true_mask = true_masks[idx]
        pred_mask = prob_map >= selected_threshold
        true_edge = true_mask ^ np.pad(true_mask[1:-1, 1:-1], ((1, 1), (1, 1)), mode='constant')
        pred_edge = pred_mask ^ np.pad(pred_mask[1:-1, 1:-1], ((1, 1), (1, 1)), mode='constant')
        overlay = np.zeros((*true_mask.shape, 3), dtype=np.float32)
        overlay[..., 0] = pred_edge.astype(np.float32)
        overlay[..., 1] = true_edge.astype(np.float32)
        overlay[..., 2] = pred_mask.astype(np.float32) * 0.25
        mean_iou = safe_nanmean([float(row['iou']) for row in seed_rows])
        mean_dice = safe_nanmean([float(row['dice']) for row in seed_rows])
        mean_area_error = safe_nanmean([float(row['area_error']) for row in seed_rows])
        pred_type = idx_to_type[max(set(pred_type_votes), key=pred_type_votes.count)]
        true_type = str(dataset.defect_types[idx])

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
        axes[0].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('true mask')
        im = axes[1].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
        axes[1].set_title('probability')
        axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'pred >= {selected_threshold:.2f}')
        axes[3].imshow(overlay)
        axes[3].set_title('overlay G=true R=pred')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{item['category']} | sample={idx} | true={true_type} pred={pred_type} | "
            f"IoU={mean_iou:.3f} Dice={mean_dice:.3f} area_error={mean_area_error:.3f}",
            fontsize=9,
        )
        filename = f"{safe_name(item['category'])}_sample{idx:03d}_{safe_name(true_type)}.png"
        path = PREVIEW_DIR / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def main():
    ensure_outputs()
    check_current_baseline_checkpoints()
    type_to_idx, type_counts = build_type_mapping()
    if type_to_idx is None:
        SUMMARY_PATH.write_text(
            '# v3_complex shape-type conditional boundary candidate\n\nShape/type label exists: False\n\nshape-type conditional model is not executable without labels.\n',
            encoding='utf-8',
        )
        write_screening_metrics([])
        write_metrics([])
        print('Shape/type label missing; stopping.')
        return

    print(f'Shape/type labels found: {type_to_idx}')
    print(f'Type counts: {type_counts}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset_for_weight = MFLDataset(TRAIN_DATA)
    pos_weight, mask_fraction = compute_pos_weight(train_dataset_for_weight)
    print(f'mask positive fraction: {mask_fraction:.6f}')
    print(f'pos_weight: {pos_weight:.6f}')

    val_area_edges = get_area_edges(MFLDataset(VAL_DATA))
    test_area_edges = get_area_edges(MFLDataset(TEST_DATA))
    val_low_signal = dataset_low_signal_indices(VAL_DATA)
    test_low_signal = load_low_signal_indices()

    baseline_val_rows, _, _ = evaluate_grid_checkpoint_family(
        checkpoints=CURRENT_BASELINE_CHECKPOINTS,
        model_type='grid',
        candidate='current_mask_boundary_baseline_val_reference',
        split='val',
        data_path=VAL_DATA,
        thresholds=[CURRENT_BASELINE_THRESHOLD],
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=val_low_signal,
    )
    baseline_val_overall = find_row(baseline_val_rows, 'current_mask_boundary_baseline_val_reference_mean', 'overall', 'all', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    baseline_val_small = find_row(baseline_val_rows, 'current_mask_boundary_baseline_val_reference_mean', 'area_bin', 'small', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    baseline_val_low = find_row(baseline_val_rows, 'current_mask_boundary_baseline_val_reference_mean', 'signal_bin', 'low_signal', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    baseline_val_polygon = find_row(baseline_val_rows, 'current_mask_boundary_baseline_val_reference_mean', 'defect_type', 'polygon', split='val', threshold=CURRENT_BASELINE_THRESHOLD)
    baseline_val_rotated = find_row(baseline_val_rows, 'current_mask_boundary_baseline_val_reference_mean', 'defect_type', 'rotated_rect', split='val', threshold=CURRENT_BASELINE_THRESHOLD)

    expected_screening_checkpoint = CHECKPOINT_DIR / f'best_shape_type_conditional_seed{SCREENING_SEED}.pt'
    if not expected_screening_checkpoint.exists():
        raise FileNotFoundError(f'Missing seed=42 checkpoint for threshold rescue: {expected_screening_checkpoint}')

    print(f'Stage A threshold calibration rescue seed={SCREENING_SEED}')
    checkpoint_path, best_info = train_one_seed(SCREENING_SEED, device, pos_weight, type_to_idx, reuse_existing=True)
    checkpoint_rel = str(checkpoint_path.relative_to(ROOT))

    screening_checkpoints = {SCREENING_SEED: checkpoint_rel}
    pred_rows, _, _, pred_accs = evaluate_conditional_checkpoint_family(
        checkpoints=screening_checkpoints,
        candidate='shape_type_conditional_screening_predicted',
        split='val',
        data_path=VAL_DATA,
        thresholds=THRESHOLDS,
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=val_low_signal,
        type_to_idx=type_to_idx,
        oracle_type=False,
    )
    oracle_rows, _, _, oracle_accs = evaluate_conditional_checkpoint_family(
        checkpoints=screening_checkpoints,
        candidate='shape_type_conditional_screening_oracle',
        split='val',
        data_path=VAL_DATA,
        thresholds=THRESHOLDS,
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=val_low_signal,
        type_to_idx=type_to_idx,
        oracle_type=True,
    )
    pred_overall_candidates = [
        row for row in pred_rows
        if row['candidate'] == 'shape_type_conditional_screening_predicted_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    oracle_overall_candidates = [
        row for row in oracle_rows
        if row['candidate'] == 'shape_type_conditional_screening_oracle_mean'
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    pred_overall = max(pred_overall_candidates, key=lambda row: float(row['composite']))
    oracle_overall = max(oracle_overall_candidates, key=lambda row: float(row['composite']))
    pred_threshold = float(pred_overall['threshold'])
    oracle_threshold = float(oracle_overall['threshold'])
    pred_small = find_row(pred_rows, 'shape_type_conditional_screening_predicted_mean', 'area_bin', 'small', split='val', threshold=pred_threshold)
    pred_low = find_row(pred_rows, 'shape_type_conditional_screening_predicted_mean', 'signal_bin', 'low_signal', split='val', threshold=pred_threshold)
    pred_polygon = find_row(pred_rows, 'shape_type_conditional_screening_predicted_mean', 'defect_type', 'polygon', split='val', threshold=pred_threshold)
    pred_rotated = find_row(pred_rows, 'shape_type_conditional_screening_predicted_mean', 'defect_type', 'rotated_rect', split='val', threshold=pred_threshold)
    pred_type_acc = float(pred_overall.get('type_accuracy', pred_accs[SCREENING_SEED]))
    passed, checks = positive_signal(
        pred_overall,
        pred_small,
        pred_low,
        pred_polygon,
        pred_rotated,
        baseline_val_overall,
        baseline_val_small,
        baseline_val_low,
        baseline_val_polygon,
        baseline_val_rotated,
        pred_type_acc,
        len(type_to_idx),
    )

    pred_record = screening_record_from_rows(
        'stage_a',
        'shape_type_conditional_screening_predicted_mean',
        SCREENING_SEED,
        pred_threshold,
        pred_rows,
        checkpoint_rel,
        positive_signal=passed,
    )
    oracle_record = screening_record_from_rows(
        'stage_a_oracle_upper_bound',
        'shape_type_conditional_screening_oracle_mean',
        SCREENING_SEED,
        oracle_threshold,
        oracle_rows,
        checkpoint_rel,
        positive_signal='diagnostic_only',
    )
    screening_records = [pred_record, oracle_record]
    write_screening_metrics(screening_records)
    write_threshold_rescue_metrics(baseline_val_rows + pred_rows + oracle_rows)
    print(f'Stage A positive signal: {passed}; checks={checks}')

    baseline_test_rows, baseline_sample_rows, _ = evaluate_grid_checkpoint_family(
        checkpoints=CURRENT_BASELINE_CHECKPOINTS,
        model_type='grid',
        candidate='current_mask_boundary_baseline',
        split='test',
        data_path=TEST_DATA,
        thresholds=[CURRENT_BASELINE_THRESHOLD],
        device=device,
        area_edges=test_area_edges,
        low_signal_indices=test_low_signal,
    )
    baseline_test_overall = get_overall_mean(baseline_test_rows, 'current_mask_boundary_baseline', CURRENT_BASELINE_THRESHOLD, split='test')

    if not passed:
        all_rows = baseline_val_rows + pred_rows + oracle_rows + baseline_test_rows
        write_metrics(all_rows)
        write_threshold_rescue_metrics(baseline_val_rows + pred_rows + oracle_rows)
        write_summary(
            rows=all_rows,
            screening_records=screening_records,
            type_to_idx=type_to_idx,
            type_counts=type_counts,
            best_infos=[best_info],
            checkpoint_paths=[checkpoint_path],
            selected_threshold=None,
            entered_stage_b=False,
            pos_weight=pos_weight,
            mask_fraction=mask_fraction,
        )
        print(f'Wrote screening metrics: {SCREENING_METRICS_PATH}')
        print(f'Wrote metrics: {METRICS_PATH}')
        print(f'Wrote summary: {SUMMARY_PATH}')
        print('Accepted by metric gate: False')
        return

    checkpoint_paths = [checkpoint_path]
    best_infos = [best_info]
    for seed in [seed for seed in SEEDS if seed != SCREENING_SEED]:
        print(f'Stage B training seed={seed}')
        path, info = train_one_seed(seed, device, pos_weight, type_to_idx)
        checkpoint_paths.append(path)
        best_infos.append(info)

    candidate_checkpoints = {seed: str(path.relative_to(ROOT)) for seed, path in zip(SEEDS, checkpoint_paths)}
    val_rows, _, _, _ = evaluate_conditional_checkpoint_family(
        checkpoints=candidate_checkpoints,
        candidate='shape_type_conditional_val_scan',
        split='val',
        data_path=VAL_DATA,
        thresholds=THRESHOLDS,
        device=device,
        area_edges=val_area_edges,
        low_signal_indices=val_low_signal,
        type_to_idx=type_to_idx,
        oracle_type=False,
    )
    selected_row = select_threshold(val_rows, baseline_test_overall)
    selected_threshold = float(selected_row['threshold'])
    print(f'Selected threshold: {selected_threshold:.2f}')

    test_rows, candidate_sample_rows, candidate_prob_cache, _ = evaluate_conditional_checkpoint_family(
        checkpoints=candidate_checkpoints,
        candidate='shape_type_conditional_test',
        split='test',
        data_path=TEST_DATA,
        thresholds=[selected_threshold],
        device=device,
        area_edges=test_area_edges,
        low_signal_indices=test_low_signal,
        type_to_idx=type_to_idx,
        oracle_type=False,
    )
    oracle_test_rows, _, _, _ = evaluate_conditional_checkpoint_family(
        checkpoints=candidate_checkpoints,
        candidate='shape_type_conditional_oracle_test',
        split='test',
        data_path=TEST_DATA,
        thresholds=[selected_threshold],
        device=device,
        area_edges=test_area_edges,
        low_signal_indices=test_low_signal,
        type_to_idx=type_to_idx,
        oracle_type=True,
    )

    all_rows = baseline_test_rows + val_rows + test_rows + oracle_test_rows
    write_metrics(all_rows)
    candidate_means = sample_mean_metrics(candidate_sample_rows, SEEDS, selected_threshold)
    baseline_means = sample_mean_metrics(baseline_sample_rows, SEEDS, CURRENT_BASELINE_THRESHOLD)
    selected_samples = select_preview_samples(candidate_means, baseline_means)
    preview_paths = generate_previews(selected_samples, candidate_prob_cache, candidate_sample_rows, selected_threshold, type_to_idx)
    write_summary(
        rows=all_rows,
        screening_records=screening_records,
        type_to_idx=type_to_idx,
        type_counts=type_counts,
        best_infos=best_infos,
        checkpoint_paths=checkpoint_paths,
        selected_threshold=selected_threshold,
        entered_stage_b=True,
        pos_weight=pos_weight,
        mask_fraction=mask_fraction,
        preview_count=len(preview_paths),
    )
    print(f'Wrote metrics: {METRICS_PATH}')
    print(f'Wrote summary: {SUMMARY_PATH}')
    print(f'Wrote previews: {PREVIEW_DIR} ({len(preview_paths)} png)')


if __name__ == '__main__':
    main()
