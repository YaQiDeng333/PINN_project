import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
MU_SCALE = 1000.0
MASK_THRESHOLD = 500.0
B0 = 1.5
DATASET_CONFIGS = {
    'simple': {
        'train_data': 'data/training_data_train.npz',
        'val_data': 'data/training_data_val.npz',
        'checkpoint_path': 'checkpoints/best_model_tv.pt',
        'physics_checkpoint_path': 'checkpoints/best_model_tv_phy.pt',
        'loss_curve_path': 'results/loss_curve_tv.png',
        'physics_loss_curve_path': 'results/loss_curve_tv_phy.png',
        'preview_path': 'results/reconstruction_preview_tv.png',
        'physics_preview_path': 'results/reconstruction_preview_tv_phy.png',
        'physics_loss_log_path': 'results/physics_loss_log.csv',
    },
    'v3_complex': {
        'train_data': 'data/training_data_v3_complex_train.npz',
        'val_data': 'data/training_data_v3_complex_val.npz',
        'checkpoint_path': 'checkpoints/best_model_v3_complex_tv.pt',
        'physics_checkpoint_path': 'checkpoints/best_model_v3_complex_tv_phy.pt',
        'loss_curve_path': 'results/loss_curves/loss_curve_v3_complex_tv.png',
        'physics_loss_curve_path': 'results/loss_curves/loss_curve_v3_complex_tv_phy.png',
        'preview_path': 'results/previews/reconstruction_preview_v3_complex_tv.png',
        'physics_preview_path': 'results/previews/reconstruction_preview_v3_complex_tv_phy.png',
        'physics_loss_log_path': 'results/archive/physics_loss_log_v3_complex.csv',
    },
    'v4_balanced_complex': {
        'train_data': 'data/training_data_v4_balanced_complex_train.npz',
        'val_data': 'data/training_data_v4_balanced_complex_val.npz',
        'checkpoint_path': 'checkpoints/best_model_v4_balanced_complex_tv.pt',
        'physics_checkpoint_path': 'checkpoints/best_model_v4_balanced_complex_tv_phy.pt',
        'loss_curve_path': 'results/loss_curves/loss_curve_v4_balanced_complex_tv.png',
        'physics_loss_curve_path': 'results/loss_curves/loss_curve_v4_balanced_complex_tv_phy.png',
        'preview_path': 'results/previews/reconstruction_preview_v4_balanced_complex_tv.png',
        'physics_preview_path': 'results/previews/reconstruction_preview_v4_balanced_complex_tv_phy.png',
        'physics_loss_log_path': 'results/archive/physics_loss_log_v4_balanced_complex.csv',
    },
}


def project_path(*parts):
    path = os.path.abspath(os.path.join(PROJECT_DIR, *parts))
    if os.path.commonpath([PROJECT_DIR, path]) != PROJECT_DIR:
        raise ValueError(f'Path must stay inside project directory: {path}')
    return path


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MFLDataset(Dataset):
    def __init__(self, npz_path, signal_mean=None, signal_std=None):
        data = np.load(project_path(npz_path), allow_pickle=False)
        self.signals = data['signals'].astype(np.float32)
        self.mu_maps = data['mu_maps'].astype(np.float32) / MU_SCALE
        self.defect_types = data['defect_types']
        self.metadata = data['metadata']
        self.metadata_keys = data['metadata_keys'] if 'metadata_keys' in data.files else None
        self.x = data['x'].astype(np.float32)
        self.y = data['y'].astype(np.float32)
        self.depths = self.metadata['depth'].astype(np.float32)
        self.lift_offs = self.metadata['lift_off'].astype(np.float32)

        if signal_mean is None:
            signal_mean = float(self.signals.mean())
        if signal_std is None:
            signal_std = float(self.signals.std() + 1e-8)

        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.signals = (self.signals - self.signal_mean) / self.signal_std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.mu_maps[idx].reshape(-1)),
            idx,
        )


def build_coord_grid(x, y):
    x_norm = x / max(abs(float(x.min())), abs(float(x.max())))
    y_norm = 2.0 * (y - float(y.min())) / (float(y.max()) - float(y.min())) - 1.0
    X, Y = np.meshgrid(x_norm, y_norm)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    return torch.from_numpy(coords)


def feature_mapping(coords, num_frequencies=21):
    freqs = torch.linspace(1, 10, num_frequencies, device=coords.device, dtype=coords.dtype)
    x = coords[..., 0:1]
    y = coords[..., 1:2]

    feats = []
    for freq in freqs:
        feats.extend([
            torch.sin(freq * x),
            torch.cos(freq * x),
            torch.sin(freq * y),
            torch.cos(freq * y),
        ])
    return torch.cat(feats, dim=-1)


class BzEncoder(nn.Module):
    def __init__(self, signal_length, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.GELU(),
        )

    def forward(self, bz_signal):
        return self.encoder(bz_signal.unsqueeze(1))


class PINN(nn.Module):
    def __init__(self, signal_length, coord_feature_dim=84, latent_dim=64):
        super().__init__()
        self.bz_encoder = BzEncoder(signal_length=signal_length, latent_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(coord_feature_dim + latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, bz_signal, coords):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(bz_signal.shape[0], -1, -1)

        bz_latent = self.bz_encoder(bz_signal)
        coord_features = feature_mapping(coords)
        bz_features = bz_latent.unsqueeze(1).expand(-1, coord_features.shape[1], -1)
        features = torch.cat([bz_features, coord_features], dim=-1)
        return self.decoder(features).squeeze(-1)


def tv_loss(mu_pred_map):
    if mu_pred_map.dim() == 3:
        mu_pred_map = mu_pred_map.unsqueeze(1)

    tv_h = torch.mean(torch.abs(mu_pred_map[:, :, 1:, :] - mu_pred_map[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(mu_pred_map[:, :, :, 1:] - mu_pred_map[:, :, :, :-1]))
    return tv_h + tv_w


def simplified_forward_bz(mu_pred_map, sensor_x, depth, lift_off, signal_mean, signal_std):
    # Simplified physics model matching data_generator_v2.py at the signal level:
    # estimate soft defect area and center_x from predicted mu, then use the
    # generator's area * depth dipole-like Bz formula. Shape and center_y are
    # not explicitly modeled in the generated signal, so they are ignored here.
    soft_defect = torch.sigmoid((0.5 - mu_pred_map) / 0.05)
    area_pixels = soft_defect.sum(dim=(1, 2)).clamp_min(1e-6)

    x_grid = sensor_x.view(1, 1, -1)
    center_x = (soft_defect * x_grid).sum(dim=(1, 2)) / area_pixels

    dx = sensor_x.unsqueeze(0) - center_x.unsqueeze(1)
    dist = torch.sqrt(dx ** 2 + lift_off.unsqueeze(1) ** 2)
    signal_amp = area_pixels * depth * 0.12
    bz_pred = B0 + dx / (dist ** 3 + 1e-6) * signal_amp.unsqueeze(1)
    return (bz_pred - signal_mean) / signal_std


def physics_loss(mu_pred_map, target_signals, sensor_x, depth, lift_off, signal_mean, signal_std):
    bz_pred = simplified_forward_bz(
        mu_pred_map=mu_pred_map,
        sensor_x=sensor_x,
        depth=depth,
        lift_off=lift_off,
        signal_mean=signal_mean,
        signal_std=signal_std,
    )
    return torch.mean((bz_pred - target_signals) ** 2)


def normalized_defect_threshold():
    return MASK_THRESHOLD / MU_SCALE


def compute_weighted_mse_loss(pred, target, defect_weight):
    defect_threshold = normalized_defect_threshold()
    defect_mask = target < defect_threshold
    weights = torch.ones_like(target)
    weights = torch.where(defect_mask, weights * defect_weight, weights)
    return torch.mean(weights * (pred - target) ** 2)


def compute_soft_dice_loss(pred, target, eps=1e-6):
    defect_threshold = normalized_defect_threshold()
    target_mask = (target < defect_threshold).to(dtype=pred.dtype)
    pred_soft_mask = torch.clamp(1.0 - pred / defect_threshold, min=0.0, max=1.0)

    pred_flat = pred_soft_mask.reshape(pred.shape[0], -1)
    target_flat = target_mask.reshape(target.shape[0], -1)
    intersection = torch.sum(pred_flat * target_flat, dim=1)
    pred_sum = torch.sum(pred_flat, dim=1)
    target_sum = torch.sum(target_flat, dim=1)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return torch.mean(1.0 - dice)


def compute_area_loss(pred, target, area_loss_type='symmetric', eps=1e-6):
    defect_threshold = normalized_defect_threshold()
    target_mask = (target < defect_threshold).to(dtype=pred.dtype)
    pred_soft_mask = torch.clamp(1.0 - pred / defect_threshold, min=0.0, max=1.0)

    pred_area = torch.sum(pred_soft_mask.reshape(pred.shape[0], -1), dim=1)
    true_area = torch.sum(target_mask.reshape(target.shape[0], -1), dim=1)
    area_diff = pred_area - true_area
    if area_loss_type == 'symmetric':
        area_penalty = torch.abs(area_diff)
    elif area_loss_type == 'over_only':
        area_penalty = torch.relu(area_diff)
    else:
        raise ValueError(f'Unsupported area_loss_type: {area_loss_type}')
    return torch.mean(area_penalty / (true_area + eps))


def get_batch_physics_params(dataset, indices, device):
    indices_np = indices.detach().cpu().numpy()
    depth = torch.from_numpy(dataset.depths[indices_np]).to(device)
    lift_off = torch.from_numpy(dataset.lift_offs[indices_np]).to(device)
    return depth, lift_off


def run_epoch(
    model,
    loader,
    coords,
    optimizer,
    criterion,
    lambda_tv,
    grid_shape,
    device,
    lambda_phy=0.0,
    physics_dataset=None,
    sensor_x=None,
    loss_type='mse',
    defect_weight=10.0,
    lambda_dice=0.05,
    lambda_area=0.0,
    area_loss_type='symmetric',
):
    model.train()
    total_unweighted_mse_loss = 0.0
    total_weighted_mse_loss = 0.0
    total_soft_dice_loss = 0.0
    total_area_loss = 0.0
    total_tv_loss = 0.0
    total_physics_loss = 0.0
    total_total_loss = 0.0
    total_samples = 0

    for signals, mu_targets, indices in loader:
        signals = signals.to(device)
        mu_targets = mu_targets.to(device)

        optimizer.zero_grad()
        pred = model(signals, coords)
        pred_map = pred.reshape(signals.shape[0], *grid_shape)
        unweighted_mse = criterion(pred, mu_targets)
        soft_dice = pred.new_tensor(0.0)
        area = pred.new_tensor(0.0)
        if loss_type == 'weighted_mse':
            weighted_mse = compute_weighted_mse_loss(pred, mu_targets, defect_weight)
            data_loss = weighted_mse
        elif loss_type == 'weighted_mse_dice':
            weighted_mse = compute_weighted_mse_loss(pred, mu_targets, defect_weight)
            soft_dice = compute_soft_dice_loss(pred, mu_targets)
            data_loss = weighted_mse + lambda_dice * soft_dice
        elif loss_type == 'weighted_mse_dice_area':
            weighted_mse = compute_weighted_mse_loss(pred, mu_targets, defect_weight)
            soft_dice = compute_soft_dice_loss(pred, mu_targets)
            area = compute_area_loss(pred, mu_targets, area_loss_type=area_loss_type)
            data_loss = weighted_mse + lambda_dice * soft_dice + lambda_area * area
        elif loss_type == 'mse':
            weighted_mse = unweighted_mse
            data_loss = unweighted_mse
        else:
            raise ValueError(f'Unsupported loss_type: {loss_type}')
        tv = tv_loss(pred_map * MU_SCALE)
        phy = pred.new_tensor(0.0)
        if lambda_phy > 0.0:
            if physics_dataset is None or sensor_x is None:
                raise ValueError('physics_dataset and sensor_x are required when lambda_phy > 0.')
            depth, lift_off = get_batch_physics_params(physics_dataset, indices, device)
            phy = physics_loss(
                mu_pred_map=pred_map,
                target_signals=signals,
                sensor_x=sensor_x,
                depth=depth,
                lift_off=lift_off,
                signal_mean=physics_dataset.signal_mean,
                signal_std=physics_dataset.signal_std,
            )
        total = data_loss + lambda_tv * tv + lambda_phy * phy
        total.backward()
        optimizer.step()

        batch_size = signals.shape[0]
        total_unweighted_mse_loss += unweighted_mse.item() * batch_size
        total_weighted_mse_loss += weighted_mse.item() * batch_size
        total_soft_dice_loss += soft_dice.item() * batch_size
        total_area_loss += area.item() * batch_size
        total_tv_loss += tv.item() * batch_size
        total_physics_loss += phy.item() * batch_size
        total_total_loss += total.item() * batch_size
        total_samples += batch_size

    total_samples = max(total_samples, 1)
    return {
        'mse_loss': total_unweighted_mse_loss / total_samples,
        'unweighted_mse_loss': total_unweighted_mse_loss / total_samples,
        'weighted_mse_loss': total_weighted_mse_loss / total_samples,
        'soft_dice_loss': total_soft_dice_loss / total_samples,
        'area_loss': total_area_loss / total_samples,
        'tv_loss': total_tv_loss / total_samples,
        'physics_loss': total_physics_loss / total_samples,
        'total_loss': total_total_loss / total_samples,
    }


@torch.no_grad()
def evaluate(model, loader, coords, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for signals, mu_targets, _ in loader:
        signals = signals.to(device)
        mu_targets = mu_targets.to(device)
        pred = model(signals, coords)
        loss = criterion(pred, mu_targets)

        batch_size = signals.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict_full_map(model, signal, coords, grid_shape, device, point_chunk=4096):
    model.eval()
    signal = signal.unsqueeze(0).to(device)
    preds = []

    for start in range(0, coords.shape[0], point_chunk):
        coord_chunk = coords[start:start + point_chunk]
        pred = model(signal, coord_chunk).squeeze(0)
        preds.append(pred.cpu())

    return torch.cat(preds, dim=0).numpy().reshape(grid_shape) * MU_SCALE


def save_loss_curve(
    train_mse_losses,
    train_tv_losses,
    train_total_losses,
    val_mse_losses,
    output_path,
    train_physics_losses=None,
    train_weighted_mse_losses=None,
    train_soft_dice_losses=None,
    train_area_losses=None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(train_mse_losses, label='Train Unweighted MSE Loss')
    if train_weighted_mse_losses is not None:
        plt.plot(train_weighted_mse_losses, label='Train Weighted MSE Loss')
    if train_soft_dice_losses is not None:
        plt.plot(train_soft_dice_losses, label='Train Soft Dice Loss')
    if train_area_losses is not None:
        plt.plot(train_area_losses, label='Train Area Loss')
    plt.plot(train_tv_losses, label='Train TV Loss')
    if train_physics_losses is not None:
        plt.plot(train_physics_losses, label='Train Physics Loss')
    plt.plot(train_total_losses, label='Train Total Loss')
    plt.plot(val_mse_losses, label='Val Unweighted MSE Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_physics_loss_log(rows, output_path):
    fieldnames = [
        'epoch',
        'mse_loss',
        'unweighted_mse_loss',
        'weighted_mse_loss',
        'soft_dice_loss',
        'area_loss',
        'tv_loss',
        'physics_loss',
        'total_loss',
        'val_mse_loss',
        'val_unweighted_mse_loss',
    ]
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_validation_visualization(model, val_dataset, coords, device, output_path, sample_idx=0, show=False):
    signal = torch.from_numpy(val_dataset.signals[sample_idx])
    true_mu = val_dataset.mu_maps[sample_idx] * MU_SCALE
    pred_mu = predict_full_map(model, signal, coords, true_mu.shape, device)
    defect_type = str(val_dataset.defect_types[sample_idx])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    extent = [float(val_dataset.x.min()), float(val_dataset.x.max()),
              float(val_dataset.y.min()), float(val_dataset.y.max())]

    im0 = ax[0].imshow(pred_mu, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=MU_SCALE)
    ax[0].set_title(f'Predicted $\\mu_r$ [{defect_type.upper()}]')
    plt.colorbar(im0, ax=ax[0], label='$\\mu_r$')

    im1 = ax[1].imshow(true_mu, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=MU_SCALE)
    ax[1].set_title(f'True $\\mu_r$ [{defect_type.upper()}]')
    plt.colorbar(im1, ax=ax[1], label='$\\mu_r$')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, best_val_loss, args, train_dataset, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'signal_mean': train_dataset.signal_mean,
        'signal_std': train_dataset.signal_std,
        'args': vars(args),
    }, checkpoint_path)


def load_checkpoint_model(checkpoint_path, signal_length, fallback_latent_dim, device):
    checkpoint = torch.load(project_path(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        checkpoint_args = checkpoint.get('args', {})
        latent_dim = int(checkpoint_args.get('latent_dim', fallback_latent_dim))
        signal_mean = checkpoint.get('signal_mean')
        signal_std = checkpoint.get('signal_std')
    else:
        state_dict = checkpoint
        checkpoint_args = {}
        latent_dim = fallback_latent_dim
        signal_mean = None
        signal_std = None

    model = PINN(signal_length=signal_length, latent_dim=latent_dim).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = {
            key.replace('module.', '', 1): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)

    return model, checkpoint, checkpoint_args, signal_mean, signal_std


@torch.no_grad()
def compute_lbfgs_metrics(model, signals, mu_targets, coords, criterion, lambda_tv, grid_shape):
    pred = model(signals, coords)
    pred_map = pred.reshape(signals.shape[0], *grid_shape)
    mse = criterion(pred, mu_targets)
    tv = tv_loss(pred_map * MU_SCALE)
    total = mse + lambda_tv * tv
    return {
        'mse_loss': float(mse.item()),
        'tv_loss': float(tv.item()),
        'total_loss': float(total.item()),
    }


def configure_dataset_paths(args):
    config = DATASET_CONFIGS[args.dataset]
    if args.train_data is None:
        args.train_data = config['train_data']
    if args.val_data is None:
        args.val_data = config['val_data']


def configure_adam_paths(args):
    configure_dataset_paths(args)
    config = DATASET_CONFIGS[args.dataset]
    use_physics = args.mode == 'adam_tv_phy'
    if args.checkpoint_path is None:
        args.checkpoint_path = config['physics_checkpoint_path'] if use_physics else config['checkpoint_path']
    if args.loss_curve_path is None:
        args.loss_curve_path = config['physics_loss_curve_path'] if use_physics else config['loss_curve_path']
    if args.preview_path is None:
        args.preview_path = config['physics_preview_path'] if use_physics else config['preview_path']
    if args.physics_loss_log_path is None:
        args.physics_loss_log_path = config['physics_loss_log_path']
    if use_physics and not args.init_checkpoint:
        args.init_checkpoint = 'checkpoints/best_model_tv_5e-6.pt'


def train_adam_tv(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    configure_adam_paths(args)
    os.makedirs(project_path('checkpoints'), exist_ok=True)
    os.makedirs(project_path('results'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using random seed: {args.seed}')

    use_physics = args.mode == 'adam_tv_phy'
    raw_train = np.load(project_path(args.train_data), allow_pickle=False)
    signal_length = raw_train['signals'].shape[1]

    if args.init_checkpoint:
        print(f'Loading Adam initial checkpoint: {args.init_checkpoint}')
        model, checkpoint, checkpoint_args, signal_mean, signal_std = load_checkpoint_model(
            checkpoint_path=args.init_checkpoint,
            signal_length=signal_length,
            fallback_latent_dim=args.latent_dim,
            device=device,
        )
        if signal_mean is None or signal_std is None:
            train_dataset_for_stats = MFLDataset(args.train_data)
            signal_mean = train_dataset_for_stats.signal_mean
            signal_std = train_dataset_for_stats.signal_std
        train_dataset = MFLDataset(args.train_data, signal_mean=signal_mean, signal_std=signal_std)
    else:
        train_dataset = MFLDataset(args.train_data)
        model = PINN(
            signal_length=train_dataset.signals.shape[1],
            latent_dim=args.latent_dim,
        ).to(device)

    val_dataset = MFLDataset(
        args.val_data,
        signal_mean=train_dataset.signal_mean,
        signal_std=train_dataset.signal_std,
    )

    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    total_points = coords.shape[0]
    grid_shape = train_dataset.mu_maps.shape[1:]

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sensor_x = torch.from_numpy(train_dataset.x).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_mse_losses = []
    train_weighted_mse_losses = []
    train_soft_dice_losses = []
    train_area_losses = []
    train_tv_losses = []
    train_physics_losses = []
    train_total_losses = []
    val_mse_losses = []
    loss_log_rows = []
    best_val_loss = float('inf')
    checkpoint_path = project_path(args.checkpoint_path)
    ensure_parent_dir(checkpoint_path)
    effective_lambda_phy = args.lambda_phy if use_physics else 0.0

    print('Model: BzEncoder(signal -> latent) + Fourier(x,y) + MLP -> mu(x,y)')
    print(f'Dataset: {args.dataset}')
    print(f'Train data: {args.train_data}')
    print(f'Val data: {args.val_data}')
    print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Points/map: {total_points}')
    print(f'loss_type: {args.loss_type}')
    if args.loss_type in ('weighted_mse', 'weighted_mse_dice', 'weighted_mse_dice_area'):
        print(f'defect_weight: {args.defect_weight:.2f}')
    if args.loss_type in ('weighted_mse_dice', 'weighted_mse_dice_area'):
        print(f'lambda_dice: {args.lambda_dice:.2e}')
    if args.loss_type == 'weighted_mse_dice_area':
        print(f'lambda_area: {args.lambda_area:.2e}')
        print(f'area_loss_type: {args.area_loss_type}')
    print(f'lambda_tv: {args.lambda_tv:.2e}')
    print(f'lambda_phy: {effective_lambda_phy:.2e}')

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            coords=coords,
            optimizer=optimizer,
            criterion=criterion,
            lambda_tv=args.lambda_tv,
            grid_shape=grid_shape,
            device=device,
            lambda_phy=effective_lambda_phy,
            physics_dataset=train_dataset if use_physics else None,
            sensor_x=sensor_x if use_physics else None,
            loss_type=args.loss_type,
            defect_weight=args.defect_weight,
            lambda_dice=args.lambda_dice,
            lambda_area=args.lambda_area,
            area_loss_type=args.area_loss_type,
        )
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            coords=coords,
            criterion=criterion,
            device=device,
        )

        train_mse_losses.append(train_metrics['mse_loss'])
        train_weighted_mse_losses.append(train_metrics['weighted_mse_loss'])
        train_soft_dice_losses.append(train_metrics['soft_dice_loss'])
        train_area_losses.append(train_metrics['area_loss'])
        train_tv_losses.append(train_metrics['tv_loss'])
        train_physics_losses.append(train_metrics['physics_loss'])
        train_total_losses.append(train_metrics['total_loss'])
        val_mse_losses.append(val_loss)
        loss_log_rows.append({
            'epoch': epoch,
            'mse_loss': f'{train_metrics["mse_loss"]:.8e}',
            'unweighted_mse_loss': f'{train_metrics["unweighted_mse_loss"]:.8e}',
            'weighted_mse_loss': f'{train_metrics["weighted_mse_loss"]:.8e}',
            'soft_dice_loss': f'{train_metrics["soft_dice_loss"]:.8e}',
            'area_loss': f'{train_metrics["area_loss"]:.8e}',
            'tv_loss': f'{train_metrics["tv_loss"]:.8e}',
            'physics_loss': f'{train_metrics["physics_loss"]:.8e}',
            'total_loss': f'{train_metrics["total_loss"]:.8e}',
            'val_mse_loss': f'{val_loss:.8e}',
            'val_unweighted_mse_loss': f'{val_loss:.8e}',
        })
        if args.loss_type == 'weighted_mse':
            print(
                f'Epoch {epoch:03d}/{args.epochs:03d} | '
                f'weighted_mse_loss: {train_metrics["weighted_mse_loss"]:.6e} | '
                f'unweighted_mse_loss: {train_metrics["unweighted_mse_loss"]:.6e} | '
                f'tv_loss: {train_metrics["tv_loss"]:.6e} | '
                f'physics_loss: {train_metrics["physics_loss"]:.6e} | '
                f'total_loss: {train_metrics["total_loss"]:.6e} | '
                f'val_unweighted_mse_loss: {val_loss:.6e}'
            )
        elif args.loss_type in ('weighted_mse_dice', 'weighted_mse_dice_area'):
            print(
                f'Epoch {epoch:03d}/{args.epochs:03d} | '
                f'weighted_mse_loss: {train_metrics["weighted_mse_loss"]:.6e} | '
                f'unweighted_mse_loss: {train_metrics["unweighted_mse_loss"]:.6e} | '
                f'soft_dice_loss: {train_metrics["soft_dice_loss"]:.6e} | '
                f'area_loss: {train_metrics["area_loss"]:.6e} | '
                f'total_loss: {train_metrics["total_loss"]:.6e} | '
                f'val_unweighted_mse_loss: {val_loss:.6e}'
            )
        else:
            print(
                f'Epoch {epoch:03d}/{args.epochs:03d} | '
                f'mse_loss: {train_metrics["mse_loss"]:.6e} | '
                f'tv_loss: {train_metrics["tv_loss"]:.6e} | '
                f'physics_loss: {train_metrics["physics_loss"]:.6e} | '
                f'total_loss: {train_metrics["total_loss"]:.6e} | '
                f'val_mse_loss: {val_loss:.6e}'
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, args, train_dataset, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_curve_path = project_path(args.loss_curve_path)
    val_vis_path = project_path(args.preview_path)
    ensure_parent_dir(loss_curve_path)
    ensure_parent_dir(val_vis_path)
    physics_losses_for_plot = train_physics_losses if use_physics else None
    save_loss_curve(
        train_mse_losses,
        train_tv_losses,
        train_total_losses,
        val_mse_losses,
        loss_curve_path,
        train_physics_losses=physics_losses_for_plot,
        train_weighted_mse_losses=(
            train_weighted_mse_losses
            if args.loss_type in ('weighted_mse', 'weighted_mse_dice', 'weighted_mse_dice_area')
            else None
        ),
        train_soft_dice_losses=(
            train_soft_dice_losses
            if args.loss_type in ('weighted_mse_dice', 'weighted_mse_dice_area')
            else None
        ),
        train_area_losses=train_area_losses if args.loss_type == 'weighted_mse_dice_area' else None,
    )
    save_validation_visualization(
        model=model,
        val_dataset=val_dataset,
        coords=coords,
        device=device,
        output_path=val_vis_path,
        sample_idx=0,
        show=args.show,
    )
    if use_physics:
        physics_loss_log_path = project_path(args.physics_loss_log_path)
        ensure_parent_dir(physics_loss_log_path)
        save_physics_loss_log(loss_log_rows, physics_loss_log_path)

    print(f'Best val mse loss: {best_val_loss:.6e}')
    print(f'Saved best model to {checkpoint_path}')
    print(f'Saved loss curve to {loss_curve_path}')
    print(f'Saved validation visualization to {val_vis_path}')
    if use_physics:
        print(f'Saved physics loss log to {physics_loss_log_path}')


def refine_with_lbfgs(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    configure_dataset_paths(args)
    os.makedirs(project_path('checkpoints'), exist_ok=True)
    os.makedirs(project_path('results'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using random seed: {args.seed}')
    print(f'Loading L-BFGS initial checkpoint: {args.lbfgs_init_checkpoint}')

    raw_train = np.load(project_path(args.train_data), allow_pickle=False)
    signal_length = raw_train['signals'].shape[1]
    model, checkpoint, checkpoint_args, signal_mean, signal_std = load_checkpoint_model(
        checkpoint_path=args.lbfgs_init_checkpoint,
        signal_length=signal_length,
        fallback_latent_dim=args.latent_dim,
        device=device,
    )

    if signal_mean is None or signal_std is None:
        train_dataset_for_stats = MFLDataset(args.train_data)
        signal_mean = train_dataset_for_stats.signal_mean
        signal_std = train_dataset_for_stats.signal_std

    train_dataset = MFLDataset(args.train_data, signal_mean=signal_mean, signal_std=signal_std)
    val_dataset = MFLDataset(args.val_data, signal_mean=signal_mean, signal_std=signal_std)

    refine_train_count = min(args.refine_train_samples, len(train_dataset))
    refine_val_count = min(args.refine_val_samples, len(val_dataset))
    train_subset = Subset(train_dataset, list(range(refine_train_count)))
    val_subset = Subset(val_dataset, list(range(refine_val_count)))

    train_signals = torch.stack([sample[0] for sample in train_subset]).to(device)
    train_targets = torch.stack([sample[1] for sample in train_subset]).to(device)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    grid_shape = train_dataset.mu_maps.shape[1:]
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn='strong_wolfe',
    )

    train_mse_losses = []
    train_tv_losses = []
    train_total_losses = []
    val_mse_losses = []
    best_val_loss = float('inf')
    checkpoint_path = project_path(args.lbfgs_checkpoint_path)
    ensure_parent_dir(checkpoint_path)

    print('Mode: L-BFGS refine from TV checkpoint')
    print(f'Dataset: {args.dataset}')
    print(f'Train data: {args.train_data}')
    print(f'Val data: {args.val_data}')
    print(f'refine_train_samples: {refine_train_count} | refine_val_samples: {refine_val_count}')
    print(
        f'lbfgs_lr: {args.lbfgs_lr} | max_iter: {args.lbfgs_max_iter} | '
        f'history_size: {args.lbfgs_history_size} | outer_steps: {args.lbfgs_outer_steps}'
    )
    print(f'lambda_tv: {args.lambda_tv:.2e}')

    last_closure_metrics = {'mse_loss': float('nan'), 'tv_loss': float('nan'), 'total_loss': float('nan')}

    for outer_step in range(1, args.lbfgs_outer_steps + 1):
        def closure():
            nonlocal last_closure_metrics
            optimizer.zero_grad()
            pred = model(train_signals, coords)
            pred_map = pred.reshape(train_signals.shape[0], *grid_shape)
            mse = criterion(pred, train_targets)
            tv = tv_loss(pred_map * MU_SCALE)
            total = mse + args.lambda_tv * tv
            total.backward()
            last_closure_metrics = {
                'mse_loss': float(mse.detach().item()),
                'tv_loss': float(tv.detach().item()),
                'total_loss': float(total.detach().item()),
            }
            return total

        optimizer.step(closure)
        train_metrics = compute_lbfgs_metrics(
            model=model,
            signals=train_signals,
            mu_targets=train_targets,
            coords=coords,
            criterion=criterion,
            lambda_tv=args.lambda_tv,
            grid_shape=grid_shape,
        )
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            coords=coords,
            criterion=criterion,
            device=device,
        )

        train_mse_losses.append(train_metrics['mse_loss'])
        train_tv_losses.append(train_metrics['tv_loss'])
        train_total_losses.append(train_metrics['total_loss'])
        val_mse_losses.append(val_loss)

        print(
            f'L-BFGS Step {outer_step:03d}/{args.lbfgs_outer_steps:03d} | '
            f'lbfgs_total_loss: {train_metrics["total_loss"]:.6e} | '
            f'lbfgs_mse_loss: {train_metrics["mse_loss"]:.6e} | '
            f'lbfgs_tv_loss: {train_metrics["tv_loss"]:.6e} | '
            f'val_mse_loss: {val_loss:.6e}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, outer_step, best_val_loss, args, train_dataset, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_curve_path = project_path(args.lbfgs_loss_curve_path)
    preview_path = project_path(args.lbfgs_preview_path)
    ensure_parent_dir(loss_curve_path)
    ensure_parent_dir(preview_path)
    save_loss_curve(train_mse_losses, train_tv_losses, train_total_losses, val_mse_losses, loss_curve_path)
    save_validation_visualization(
        model=model,
        val_dataset=val_dataset,
        coords=coords,
        device=device,
        output_path=preview_path,
        sample_idx=0,
        show=args.show,
    )

    print(f'Best L-BFGS val mse loss: {best_val_loss:.6e}')
    print(f'Saved L-BFGS refined model to {checkpoint_path}')
    print(f'Saved L-BFGS loss curve to {loss_curve_path}')
    print(f'Saved L-BFGS validation visualization to {preview_path}')


def run_full_process(args=None):
    if args is None:
        args = parse_args()

    if args.mode == 'lbfgs_refine':
        refine_with_lbfgs(args)
    else:
        train_adam_tv(args)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bz + coordinate PINN for MFL inversion.')
    parser.add_argument('--mode', choices=['adam_tv', 'adam_tv_phy', 'lbfgs_refine'], default='adam_tv')
    parser.add_argument('--dataset', choices=sorted(DATASET_CONFIGS), default='simple')
    parser.add_argument('--train-data', default=None)
    parser.add_argument('--val-data', default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--loss-type',
        choices=['mse', 'weighted_mse', 'weighted_mse_dice', 'weighted_mse_dice_area'],
        default='mse',
    )
    parser.add_argument('--defect-weight', type=float, default=10.0)
    parser.add_argument('--lambda-dice', type=float, default=0.05)
    parser.add_argument('--lambda-area', type=float, default=0.0)
    parser.add_argument('--area-loss-type', choices=['symmetric', 'over_only'], default='symmetric')
    parser.add_argument('--lambda-tv', type=float, default=5e-6)
    parser.add_argument('--lambda-phy', type=float, default=1e-4)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--checkpoint-path', default=None)
    parser.add_argument('--loss-curve-path', default=None)
    parser.add_argument('--preview-path', default=None)
    parser.add_argument('--physics-loss-log-path', default=None)
    parser.add_argument('--lbfgs-init-checkpoint', default='checkpoints/best_model_tv.pt')
    parser.add_argument('--lbfgs-checkpoint-path', default='checkpoints/best_model_tv_lbfgs.pt')
    parser.add_argument('--lbfgs-loss-curve-path', default='results/loss_curve_tv_lbfgs.png')
    parser.add_argument('--lbfgs-preview-path', default='results/reconstruction_preview_tv_lbfgs.png')
    parser.add_argument('--refine-train-samples', type=int, default=8)
    parser.add_argument('--refine-val-samples', type=int, default=16)
    parser.add_argument('--lbfgs-lr', type=float, default=0.5)
    parser.add_argument('--lbfgs-max-iter', type=int, default=20)
    parser.add_argument('--lbfgs-history-size', type=int, default=20)
    parser.add_argument('--lbfgs-outer-steps', type=int, default=10)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run_full_process()
