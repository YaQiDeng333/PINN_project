import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pinn import MFLDataset, build_coord_grid, set_seed, signal_shape_info  # noqa: E402
import scripts.train_mask_boundary_grid_forward_consistency_candidate as fc  # noqa: E402


TRAIN_DATA = fc.TRAIN_DATA
VAL_DATA = fc.VAL_DATA
TEST_DATA = fc.TEST_DATA
SEED = 42
EPOCHS = 50
LAMBDA_VALUES = [0.02, 0.05, 0.10]
LAMBDA_REUSE_CHECKPOINT = {
    0.05: ROOT / 'checkpoints' / 'mask_boundary_forward_consistency_candidate' / 'best_mask_boundary_forward_consistency_seed42.pt'
}

CHECKPOINT_DIR = ROOT / 'checkpoints' / 'mask_boundary_forward_consistency_lambda_bracket'
METRICS_PATH = ROOT / 'results' / 'metrics' / 'v3_complex_forward_consistency_lambda_bracket_metrics.csv'
SUMMARY_PATH = ROOT / 'results' / 'summaries' / 'v3_complex_forward_consistency_lambda_bracket_summary.txt'


def lambda_tag(value):
    return f'lambda_{value:.2f}'.replace('.', 'p')


def checkpoint_path(value):
    return CHECKPOINT_DIR / f'best_forward_consistency_{lambda_tag(value)}_seed{SEED}.pt'


def train_one_lambda(value, device, pos_weight_value, forward_model, surrogate_checkpoint):
    if value in LAMBDA_REUSE_CHECKPOINT and LAMBDA_REUSE_CHECKPOINT[value].exists():
        print(f'Reusing lambda={value:.2f} checkpoint: {LAMBDA_REUSE_CHECKPOINT[value]}')
        checkpoint = torch.load(LAMBDA_REUSE_CHECKPOINT[value], map_location='cpu')
        return LAMBDA_REUSE_CHECKPOINT[value], checkpoint.get('val_metrics', {})

    set_seed(SEED)
    train_dataset = MFLDataset(
        TRAIN_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    val_dataset = MFLDataset(
        VAL_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    val_area_edges = fc.get_area_edges(val_dataset)
    signal_length, signal_channels = signal_shape_info(train_dataset.signals)
    out_shape = tuple(train_dataset.mu_maps.shape[1:])
    model = fc.MaskBoundaryGridModel(
        signal_length=signal_length,
        signal_channels=signal_channels,
        latent_dim=fc.LATENT_DIM,
        out_shape=out_shape,
        low_shape=fc.GRID_LOW_SHAPE,
        base_channels=fc.GRID_BASE_CHANNELS,
    ).to(device)
    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    train_loader = fc.make_loader(train_dataset, fc.BATCH_SIZE, shuffle=True, seed=SEED)
    optimizer = optim.Adam(model.parameters(), lr=fc.LR)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    best_score = -float('inf')
    best_info = None
    best_path = checkpoint_path(value)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_mask = 0.0
        total_forward = 0.0
        total_samples = 0
        for signals, mu_targets, indices in train_loader:
            signals = signals.to(device)
            target_mask = (mu_targets.to(device) < fc.MASK_THRESHOLD_NORM).to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            mask_logits = model(signals, coords)
            loss_mask, _, _ = fc.mask_loss(mask_logits, target_mask, pos_weight)
            mask_prob = torch.sigmoid(mask_logits).reshape(signals.shape[0], *out_shape)
            bz_hat = forward_model(mask_prob.unsqueeze(1))
            loss_forward = F.mse_loss(bz_hat, signals)
            loss = loss_mask + value * loss_forward
            loss.backward()
            optimizer.step()

            batch_size = signals.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_mask += float(loss_mask.item()) * batch_size
            total_forward += float(loss_forward.item()) * batch_size
            total_samples += batch_size

        val_summary = fc.evaluate_model_for_selection(model, val_dataset, coords, device, val_area_edges)
        selection_score = val_summary['composite']
        if selection_score > best_score:
            best_score = selection_score
            best_info = {
                'seed': SEED,
                'lambda_forward': value,
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
                    'model': 'mask_boundary_grid_forward_consistency_lambda_bracket',
                    'dataset': 'v3_complex',
                    'seed': SEED,
                    'epochs': EPOCHS,
                    'batch_size': fc.BATCH_SIZE,
                    'latent_dim': fc.LATENT_DIM,
                    'loss': f'BCEWithLogits + soft Dice + {value} * frozen mask-to-Bz surrogate MSE',
                    'lambda_forward': value,
                    'pos_weight': pos_weight_value,
                    'mask_target': 'target_mu_norm < 0.5',
                    'surrogate_checkpoint': str(fc.SURROGATE_CHECKPOINT_PATH.relative_to(ROOT)),
                    'out_shape': out_shape,
                    'low_shape': fc.GRID_LOW_SHAPE,
                    'base_channels': fc.GRID_BASE_CHANNELS,
                    'signal_channels': signal_channels,
                    'selection_metric': 'val_iou + val_dice - val_area_error at mask_prob>=0.5',
                },
                'signal_mean': float(train_dataset.signal_mean),
                'signal_std': float(train_dataset.signal_std),
                'epoch': epoch,
                'selection_score': float(selection_score),
                'val_metrics': best_info,
            }, best_path)

        print(
            f"lambda={value:.2f} epoch {epoch:03d}/{EPOCHS:03d} | "
            f"loss={total_loss / total_samples:.6e} | "
            f"mask_loss={total_mask / total_samples:.6e} | "
            f"forward_mse={total_forward / total_samples:.6e} | "
            f"val_iou={val_summary['iou']:.6e} | "
            f"val_dice={val_summary['dice']:.6e} | "
            f"val_area_error={val_summary['area_error']:.6e} | "
            f"score={selection_score:.6e}"
        )

    return best_path, best_info


def write_metrics(rows):
    fieldnames = [
        'candidate',
        'lambda_forward',
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
        'bz_mse',
        'composite',
        'macro_area_composite',
    ]
    with open(METRICS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, '') for key in fieldnames}
            writer.writerow(out)


def add_lambda_field(rows, value):
    for row in rows:
        row['lambda_forward'] = value
    return rows


def find_row(rows, candidate, split='test', group_type='overall', group='all', threshold=None):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['split'] == split
        and row['group_type'] == group_type
        and row['group'] == group
    ]
    if threshold is not None:
        selected = [row for row in selected if fc.threshold_matches(row['threshold'], threshold)]
    return selected[0] if selected else None


def select_threshold(rows, candidate):
    selected = [
        row for row in rows
        if row['candidate'] == candidate
        and row['split'] == 'val'
        and row['group_type'] == 'overall'
        and row['group'] == 'all'
    ]
    return max(selected, key=lambda row: float(row['composite']))


def fmt(row, key):
    return f"{float(row[key]):.4f}"


def format_overall_table(rows, selected_thresholds):
    lines = [
        '| candidate | threshold | IoU | Dice | area_error | center_error | pred_area=0 | Bz MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    baseline = find_row(rows, 'current_grid_baseline_test_mean', threshold=fc.CURRENT_BASELINE_THRESHOLD)
    lines.append(
        f"| CURRENT_BASELINE | {fc.CURRENT_BASELINE_THRESHOLD:.2f} | {fmt(baseline, 'iou')} | "
        f"{fmt(baseline, 'dice')} | {fmt(baseline, 'area_error')} | "
        f"{fmt(baseline, 'center_error')} | {float(baseline['pred_area_zero']):.2f} | "
        f"{float(baseline['bz_mse']):.6e} |"
    )
    for value in LAMBDA_VALUES:
        candidate = f'{lambda_tag(value)}_test_mean'
        threshold = selected_thresholds[value]
        row = find_row(rows, candidate, threshold=threshold)
        lines.append(
            f"| lambda={value:.2f} | {threshold:.2f} | {fmt(row, 'iou')} | "
            f"{fmt(row, 'dice')} | {fmt(row, 'area_error')} | "
            f"{fmt(row, 'center_error')} | {float(row['pred_area_zero']):.2f} | "
            f"{float(row['bz_mse']):.6e} |"
        )
    return '\n'.join(lines)


def group_status(rows, value, threshold, group_type, group):
    base = find_row(rows, 'current_grid_baseline_test_mean', group_type=group_type, group=group, threshold=fc.CURRENT_BASELINE_THRESHOLD)
    cand = find_row(rows, f'{lambda_tag(value)}_test_mean', group_type=group_type, group=group, threshold=threshold)
    if base is None or cand is None:
        return None
    return {
        'iou_delta': float(cand['iou']) - float(base['iou']),
        'dice_delta': float(cand['dice']) - float(base['dice']),
        'area_error_delta': float(cand['area_error']) - float(base['area_error']),
        'pred_area_zero_delta': float(cand['pred_area_zero']) - float(base['pred_area_zero']),
        'bz_mse_delta': float(cand['bz_mse']) - float(base['bz_mse']),
    }


def acceptance_status(rows, value, threshold):
    overall = group_status(rows, value, threshold, 'overall', 'all')
    small = group_status(rows, value, threshold, 'area_bin', 'small')
    low = group_status(rows, value, threshold, 'signal_bin', 'low_signal')
    if not overall or not small or not low:
        return False
    return bool(
        overall['iou_delta'] > 0
        and overall['dice_delta'] > 0
        and overall['bz_mse_delta'] < -0.05
        and overall['area_error_delta'] <= 0.02
        and overall['pred_area_zero_delta'] <= 1.0
        and small['area_error_delta'] <= 0.02
        and low['area_error_delta'] <= 0.02
    )


def write_summary(rows, selected_thresholds, checkpoint_paths, best_infos):
    accepted = {value: acceptance_status(rows, value, selected_thresholds[value]) for value in LAMBDA_VALUES}
    best_lambda = max(
        LAMBDA_VALUES,
        key=lambda value: float(find_row(rows, f'{lambda_tag(value)}_test_mean', threshold=selected_thresholds[value])['composite']),
    )
    summary = f"""# v3_complex forward consistency lambda bracket check

This Step 18.3 check evaluates only three fixed lambda_forward values: 0.02, 0.05, and 0.10. It reuses the frozen mask-to-Bz surrogate from Step 18.2 and does not retrain the surrogate. Seed=42 only. No lambda sweep beyond these values, no forward consistency v2, no threshold trick beyond the fixed validation candidate list, and no Git commit are performed for this step.

Frozen surrogate: `{fc.SURROGATE_CHECKPOINT_PATH.relative_to(ROOT)}`

## Checkpoints

| lambda_forward | checkpoint | source |
|---:|---|---|
"""
    for value in LAMBDA_VALUES:
        source = 'reused Step 18.2 seed=42' if value in LAMBDA_REUSE_CHECKPOINT else 'new seed=42 training'
        summary += f"| {value:.2f} | `{checkpoint_paths[value].relative_to(ROOT)}` | {source} |\n"

    summary += "\n## Validation-selected thresholds\n\n"
    summary += "| lambda_forward | selected threshold | best_epoch | val_IoU | val_Dice | val_area_error |\n"
    summary += "|---:|---:|---:|---:|---:|---:|\n"
    for value in LAMBDA_VALUES:
        info = best_infos[value] or {}
        summary += (
            f"| {value:.2f} | {selected_thresholds[value]:.2f} | {info.get('epoch', 'reused')} | "
            f"{float(info.get('val_iou', float('nan'))):.4f} | "
            f"{float(info.get('val_dice', float('nan'))):.4f} | "
            f"{float(info.get('val_area_error', float('nan'))):.4f} |\n"
        )

    summary += "\n## Overall test metrics\n\n"
    summary += format_overall_table(rows, selected_thresholds)
    summary += "\n\n## Gate deltas vs CURRENT_BASELINE\n\n"
    summary += "| lambda_forward | overall dIoU | overall dDice | overall dAreaErr | dPredArea0 | dBzMSE | small dAreaErr | low_signal dAreaErr | accepted |\n"
    summary += "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    for value in LAMBDA_VALUES:
        threshold = selected_thresholds[value]
        overall = group_status(rows, value, threshold, 'overall', 'all')
        small = group_status(rows, value, threshold, 'area_bin', 'small')
        low = group_status(rows, value, threshold, 'signal_bin', 'low_signal')
        summary += (
            f"| {value:.2f} | {overall['iou_delta']:.4f} | {overall['dice_delta']:.4f} | "
            f"{overall['area_error_delta']:.4f} | {overall['pred_area_zero_delta']:.2f} | "
            f"{overall['bz_mse_delta']:.6e} | {small['area_error_delta']:.4f} | "
            f"{low['area_error_delta']:.4f} | {accepted[value]} |\n"
        )

    summary += f"""
## Judgment

Best lambda by test composite score: {best_lambda:.2f}

Any lambda meeting all bracket criteria: {any(accepted.values())}

Conclusion: {'At least one lambda keeps the Bz residual gain while controlling area_error / pred_area=0 enough to justify a future 3-seed validation.' if any(accepted.values()) else 'None of the fixed lambda values simultaneously preserves IoU / Dice / Bz residual gains and controls area_error, pred_area=0, small, and low-signal behavior. Stop forward consistency tuning under this bracket.'}
"""
    SUMMARY_PATH.write_text(summary, encoding='utf-8')
    return best_lambda, accepted


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fc.check_inputs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    forward_model, surrogate_checkpoint = fc.load_forward_surrogate(device)
    train_dataset = MFLDataset(
        TRAIN_DATA,
        signal_mean=float(surrogate_checkpoint['signal_mean']),
        signal_std=float(surrogate_checkpoint['signal_std']),
    )
    pos_weight, mask_fraction = fc.compute_pos_weight(train_dataset)
    print(f'train mask positive fraction={mask_fraction:.6f}, pos_weight={pos_weight:.6f}')

    baseline_rows, _, _ = fc.evaluate_checkpoint_family(
        fc.CURRENT_BASELINE_CHECKPOINTS,
        'current_grid_baseline_test',
        'test',
        TEST_DATA,
        [fc.CURRENT_BASELINE_THRESHOLD],
        device,
        forward_model,
    )
    all_rows = add_lambda_field(baseline_rows, '')
    checkpoint_paths = {}
    best_infos = {}
    selected_thresholds = {}

    for value in LAMBDA_VALUES:
        ckpt_path, best_info = train_one_lambda(value, device, pos_weight, forward_model, surrogate_checkpoint)
        checkpoint_paths[value] = ckpt_path
        best_infos[value] = best_info
        candidate_val = f'{lambda_tag(value)}_val'
        candidate_test = f'{lambda_tag(value)}_test'
        val_rows, _, _ = fc.evaluate_checkpoint_family(
            {SEED: ckpt_path},
            candidate_val,
            'val',
            VAL_DATA,
            fc.THRESHOLDS,
            device,
            forward_model,
        )
        val_rows = add_lambda_field(val_rows, value)
        selected = select_threshold(val_rows, f'{candidate_val}_mean')
        selected_thresholds[value] = float(selected['threshold'])
        test_rows, _, _ = fc.evaluate_checkpoint_family(
            {SEED: ckpt_path},
            candidate_test,
            'test',
            TEST_DATA,
            fc.THRESHOLDS,
            device,
            forward_model,
        )
        test_rows = add_lambda_field(test_rows, value)
        all_rows.extend(val_rows)
        all_rows.extend(test_rows)
        row = find_row(all_rows, f'{candidate_test}_mean', threshold=selected_thresholds[value])
        print(
            f"lambda={value:.2f} selected_threshold={selected_thresholds[value]:.2f} | "
            f"IoU={float(row['iou']):.6f} Dice={float(row['dice']):.6f} "
            f"area_error={float(row['area_error']):.6f} pred_area_zero={float(row['pred_area_zero']):.2f} "
            f"BzMSE={float(row['bz_mse']):.6e}"
        )

    write_metrics(all_rows)
    best_lambda, accepted = write_summary(all_rows, selected_thresholds, checkpoint_paths, best_infos)
    print(f'best_lambda={best_lambda:.2f}')
    print(f'accepted={accepted}')
    print(f'wrote metrics: {METRICS_PATH}')
    print(f'wrote summary: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
