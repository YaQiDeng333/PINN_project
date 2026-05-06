import argparse
import csv
import os
import re
import subprocess
import sys


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON = sys.executable


TV_LAMBDAS = [
    ('0', 0.0),
    ('1e-6', 1e-6),
    ('5e-6', 5e-6),
    ('1e-5', 1e-5),
    ('5e-5', 5e-5),
    ('1e-4', 1e-4),
]

LBFGS_REFINE_TRAIN_SAMPLES = [8, 16, 32]
LBFGS_LRS = [0.1, 0.5]
LBFGS_OUTER_STEPS = [5, 10]


def project_path(*parts):
    path = os.path.abspath(os.path.join(PROJECT_DIR, *parts))
    if os.path.commonpath([PROJECT_DIR, path]) != PROJECT_DIR:
        raise ValueError(f'Path must stay inside project directory: {path}')
    return path


def run_command(args, log_path):
    print('RUN:', ' '.join(args), flush=True)
    proc = subprocess.run(
        args,
        cwd=PROJECT_DIR,
        text=True,
        encoding='utf-8',
        errors='replace',
        capture_output=True,
    )

    with open(project_path(log_path), 'w', encoding='utf-8') as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write('\n[stderr]\n')
            f.write(proc.stderr)

    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)

    proc.check_returncode()
    return proc.stdout


def parse_train_mse(stdout):
    matches = re.findall(r'mse_loss:\s*([0-9.eE+-]+)', stdout)
    return float(matches[-1]) if matches else float('nan')


def parse_metrics_txt(path):
    metrics = {}
    with open(project_path(path), 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in {'mse', 'mae', 'iou', 'dice', 'area_error', 'center_error'}:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
    return metrics


def write_csv(path, rows, fieldnames):
    with open(project_path(path), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_rows(rows):
    for metric, reverse in [('val_iou', True), ('val_dice', True), ('val_mae', False)]:
        sorted_rows = sorted(rows, key=lambda row: row[metric], reverse=reverse)
        for rank, row in enumerate(sorted_rows, start=1):
            row[f'{metric}_rank'] = rank

    for row in rows:
        row['selection_score'] = row['val_iou_rank'] + row['val_dice_rank'] + row['val_mae_rank']

    return min(rows, key=lambda row: (row['selection_score'], row['val_mse']))


def evaluate_model(model_path, data_path, output_prefix, log_path):
    metrics_path = f'results/{output_prefix}_evaluation_metrics.txt'
    if os.path.exists(project_path(metrics_path)):
        return parse_metrics_txt(metrics_path)

    run_command(
        [
            PYTHON,
            'evaluate_pinn.py',
            '--model',
            model_path,
            '--test-data',
            data_path,
            '--output_prefix',
            output_prefix,
            '--num-figures',
            '0',
        ],
        log_path,
    )
    return parse_metrics_txt(metrics_path)


def run_tv_sweep(epochs):
    rows = []

    for label, lambda_tv in TV_LAMBDAS:
        checkpoint_path = f'checkpoints/best_model_tv_{label}.pt'
        loss_curve_path = f'results/loss_curve_tv_{label}.png'
        preview_path = f'results/reconstruction_preview_tv_{label}.png'
        train_log = f'results/sweep_tv_{label}_train.log'

        if os.path.exists(project_path(checkpoint_path)) and os.path.exists(project_path(train_log)):
            with open(project_path(train_log), 'r', encoding='utf-8') as f:
                stdout = f.read()
        else:
            stdout = run_command(
                [
                    PYTHON,
                    'train_pinn.py',
                    '--mode',
                    'adam_tv',
                    '--epochs',
                    str(epochs),
                    '--lambda-tv',
                    str(lambda_tv),
                    '--checkpoint-path',
                    checkpoint_path,
                    '--loss-curve-path',
                    loss_curve_path,
                    '--preview-path',
                    preview_path,
                ],
                train_log,
            )

        train_eval = evaluate_model(
            checkpoint_path,
            'data/training_data_train.npz',
            f'sweep_train_tv_{label}',
            f'results/sweep_train_tv_{label}_eval.log',
        )
        val_eval = evaluate_model(
            checkpoint_path,
            'data/training_data_val.npz',
            f'sweep_val_tv_{label}',
            f'results/sweep_val_tv_{label}_eval.log',
        )

        rows.append({
            'lambda_label': label,
            'lambda_tv': lambda_tv,
            'model_path': checkpoint_path,
            'train_mse': train_eval['mse'],
            'train_mse_normalized_final_epoch': parse_train_mse(stdout),
            'val_mse': val_eval['mse'],
            'val_mae': val_eval['mae'],
            'val_iou': val_eval['iou'],
            'val_dice': val_eval['dice'],
            'val_area_error': val_eval['area_error'],
            'val_center_error': val_eval['center_error'],
        })

        write_csv(
            'results/tv_lambda_sweep.csv',
            rows,
            [
                'lambda_label',
                'lambda_tv',
                'model_path',
                'train_mse',
                'train_mse_normalized_final_epoch',
                'val_mse',
                'val_mae',
                'val_iou',
                'val_dice',
                'val_area_error',
                'val_center_error',
            ],
        )

    best = rank_rows(rows)
    for row in rows:
        row['recommended_by_val'] = row is best

    write_csv(
        'results/tv_lambda_sweep.csv',
        rows,
        [
            'lambda_label',
            'lambda_tv',
            'model_path',
            'train_mse',
            'train_mse_normalized_final_epoch',
            'val_mse',
            'val_mae',
            'val_iou',
            'val_dice',
            'val_area_error',
            'val_center_error',
            'val_iou_rank',
            'val_dice_rank',
            'val_mae_rank',
            'selection_score',
            'recommended_by_val',
        ],
    )
    return rows, best


def lbfgs_label(refine_train_samples, lr, outer_steps):
    lr_label = str(lr).replace('.', 'p')
    return f'rs{refine_train_samples}_lr{lr_label}_os{outer_steps}'


def run_lbfgs_sweep(best_tv):
    rows = []
    init_checkpoint = best_tv['model_path']
    lambda_tv = best_tv['lambda_tv']
    lambda_label = best_tv['lambda_label']

    for refine_train_samples in LBFGS_REFINE_TRAIN_SAMPLES:
        for lr in LBFGS_LRS:
            for outer_steps in LBFGS_OUTER_STEPS:
                label = lbfgs_label(refine_train_samples, lr, outer_steps)
                checkpoint_path = f'checkpoints/best_model_tv_{lambda_label}_lbfgs_{label}.pt'
                loss_curve_path = f'results/loss_curve_tv_{lambda_label}_lbfgs_{label}.png'
                preview_path = f'results/reconstruction_preview_tv_{lambda_label}_lbfgs_{label}.png'
                train_log = f'results/sweep_lbfgs_{label}_train.log'

                if not os.path.exists(project_path(checkpoint_path)):
                    run_command(
                        [
                            PYTHON,
                            'train_pinn.py',
                            '--mode',
                            'lbfgs_refine',
                            '--lambda-tv',
                            str(lambda_tv),
                            '--lbfgs-init-checkpoint',
                            init_checkpoint,
                            '--lbfgs-checkpoint-path',
                            checkpoint_path,
                            '--lbfgs-loss-curve-path',
                            loss_curve_path,
                            '--lbfgs-preview-path',
                            preview_path,
                            '--refine-train-samples',
                            str(refine_train_samples),
                            '--refine-val-samples',
                            '16',
                            '--lbfgs-lr',
                            str(lr),
                            '--lbfgs-max-iter',
                            '20',
                            '--lbfgs-history-size',
                            '20',
                            '--lbfgs-outer-steps',
                            str(outer_steps),
                        ],
                        train_log,
                    )

                val_eval = evaluate_model(
                    checkpoint_path,
                    'data/training_data_val.npz',
                    f'sweep_val_lbfgs_{label}',
                    f'results/sweep_val_lbfgs_{label}_eval.log',
                )

                rows.append({
                    'label': label,
                    'init_checkpoint': init_checkpoint,
                    'model_path': checkpoint_path,
                    'lambda_tv': lambda_tv,
                    'refine_train_samples': refine_train_samples,
                    'refine_val_samples': 16,
                    'lr': lr,
                    'max_iter': 20,
                    'history_size': 20,
                    'outer_steps': outer_steps,
                    'val_mse': val_eval['mse'],
                    'val_mae': val_eval['mae'],
                    'val_iou': val_eval['iou'],
                    'val_dice': val_eval['dice'],
                    'val_area_error': val_eval['area_error'],
                    'val_center_error': val_eval['center_error'],
                    'improves_val_mae': val_eval['mae'] < best_tv['val_mae'],
                    'improves_val_iou': val_eval['iou'] > best_tv['val_iou'],
                    'improves_val_dice': val_eval['dice'] > best_tv['val_dice'],
                })

                write_csv(
                    'results/lbfgs_sweep.csv',
                    rows,
                    [
                        'label',
                        'init_checkpoint',
                        'model_path',
                        'lambda_tv',
                        'refine_train_samples',
                        'refine_val_samples',
                        'lr',
                        'max_iter',
                        'history_size',
                        'outer_steps',
                        'val_mse',
                        'val_mae',
                        'val_iou',
                        'val_dice',
                        'val_area_error',
                        'val_center_error',
                        'improves_val_mae',
                        'improves_val_iou',
                        'improves_val_dice',
                    ],
                )

    best_lbfgs = rank_rows(rows)
    for row in rows:
        row['selection_score'] = row.get('selection_score')
        row['recommended_by_val'] = row is best_lbfgs

    write_csv(
        'results/lbfgs_sweep.csv',
        rows,
        [
            'label',
            'init_checkpoint',
            'model_path',
            'lambda_tv',
            'refine_train_samples',
            'refine_val_samples',
            'lr',
            'max_iter',
            'history_size',
            'outer_steps',
            'val_mse',
            'val_mae',
            'val_iou',
            'val_dice',
            'val_area_error',
            'val_center_error',
            'improves_val_mae',
            'improves_val_iou',
            'improves_val_dice',
            'val_iou_rank',
            'val_dice_rank',
            'val_mae_rank',
            'selection_score',
            'recommended_by_val',
        ],
    )
    return rows, best_lbfgs


def save_recommendation(best_tv, best_lbfgs, tv_test, lbfgs_test):
    lbfgs_ok = (
        best_lbfgs['val_mae'] <= best_tv['val_mae']
        and best_lbfgs['val_iou'] >= best_tv['val_iou']
        and best_lbfgs['val_dice'] >= best_tv['val_dice']
    )

    with open(project_path('results/parameter_sweep_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('Parameter sweep summary\n\n')
        f.write(f'Recommended model: {best_tv["model_path"]}\n')
        f.write(f'Recommended lambda_tv: {best_tv["lambda_tv"]} ({best_tv["lambda_label"]})\n')
        f.write('Selection rule: lowest rank sum of val_iou(desc), val_dice(desc), val_mae(asc).\n\n')
        f.write('Best TV val metrics:\n')
        for key in ['val_mse', 'val_mae', 'val_iou', 'val_dice', 'val_area_error', 'val_center_error']:
            f.write(f'{key}: {best_tv[key]:.8e}\n')
        f.write('\nBest L-BFGS val metrics:\n')
        for key in ['val_mse', 'val_mae', 'val_iou', 'val_dice', 'val_area_error', 'val_center_error']:
            f.write(f'{key}: {best_lbfgs[key]:.8e}\n')
        f.write(f'\nL-BFGS recommended as default: {lbfgs_ok}\n')
        if not lbfgs_ok:
            f.write('Reason: L-BFGS did not jointly improve val_mae, val_iou, and val_dice over the best TV model.\n')

        f.write('\nBest TV test metrics:\n')
        for key, value in tv_test.items():
            f.write(f'{key}: {value:.8e}\n')
        f.write('\nBest L-BFGS test metrics:\n')
        for key, value in lbfgs_test.items():
            f.write(f'{key}: {value:.8e}\n')


def run_all(epochs):
    os.makedirs(project_path('results'), exist_ok=True)
    os.makedirs(project_path('checkpoints'), exist_ok=True)

    tv_rows, best_tv = run_tv_sweep(epochs)
    lbfgs_rows, best_lbfgs = run_lbfgs_sweep(best_tv)

    tv_test = evaluate_model(
        best_tv['model_path'],
        'data/training_data_test.npz',
        'best_tv_test',
        'results/best_tv_test_eval.log',
    )
    lbfgs_test = evaluate_model(
        best_lbfgs['model_path'],
        'data/training_data_test.npz',
        'best_lbfgs_test',
        'results/best_lbfgs_test_eval.log',
    )
    save_recommendation(best_tv, best_lbfgs, tv_test, lbfgs_test)

    print('Recommended TV model:', best_tv['model_path'], flush=True)
    print('Recommended lambda_tv:', best_tv['lambda_tv'], flush=True)
    print('Best L-BFGS candidate:', best_lbfgs['model_path'], flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run TV/L-BFGS parameter sweeps.')
    parser.add_argument('--epochs', type=int, default=20)
    return parser.parse_args()


if __name__ == '__main__':
    run_all(parse_args().epochs)
