import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
MU_SCALE = 1000.0


def project_path(*parts):
    path = os.path.abspath(os.path.join(PROJECT_DIR, *parts))
    if os.path.commonpath([PROJECT_DIR, path]) != PROJECT_DIR:
        raise ValueError(f'Path must stay inside project directory: {path}')
    return path


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


def run_epoch(model, loader, coords, optimizer, criterion, lambda_tv, grid_shape, device):
    model.train()
    total_mse_loss = 0.0
    total_tv_loss = 0.0
    total_total_loss = 0.0
    total_samples = 0

    for signals, mu_targets, _ in loader:
        signals = signals.to(device)
        mu_targets = mu_targets.to(device)

        optimizer.zero_grad()
        pred = model(signals, coords)
        pred_map = pred.reshape(signals.shape[0], *grid_shape)
        mse = criterion(pred, mu_targets)
        tv = tv_loss(pred_map * MU_SCALE)
        total = mse + lambda_tv * tv
        total.backward()
        optimizer.step()

        batch_size = signals.shape[0]
        total_mse_loss += mse.item() * batch_size
        total_tv_loss += tv.item() * batch_size
        total_total_loss += total.item() * batch_size
        total_samples += batch_size

    total_samples = max(total_samples, 1)
    return {
        'mse_loss': total_mse_loss / total_samples,
        'tv_loss': total_tv_loss / total_samples,
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


def save_loss_curve(train_mse_losses, train_tv_losses, train_total_losses, val_mse_losses, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_mse_losses, label='Train MSE Loss')
    plt.plot(train_tv_losses, label='Train TV Loss')
    plt.plot(train_total_losses, label='Train Total Loss')
    plt.plot(val_mse_losses, label='Val MSE Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training with TV Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


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


def run_full_process(args=None):
    if args is None:
        args = parse_args()

    os.makedirs(project_path('checkpoints'), exist_ok=True)
    os.makedirs(project_path('results'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = MFLDataset(args.train_data)
    val_dataset = MFLDataset(
        args.val_data,
        signal_mean=train_dataset.signal_mean,
        signal_std=train_dataset.signal_std,
    )

    coords = build_coord_grid(train_dataset.x, train_dataset.y).to(device)
    total_points = coords.shape[0]
    grid_shape = train_dataset.mu_maps.shape[1:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PINN(
        signal_length=train_dataset.signals.shape[1],
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_mse_losses = []
    train_tv_losses = []
    train_total_losses = []
    val_mse_losses = []
    best_val_loss = float('inf')
    checkpoint_path = project_path(args.checkpoint_path)

    print('Model: BzEncoder(signal -> latent) + Fourier(x,y) + MLP -> mu(x,y)')
    print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Points/map: {total_points}')
    print(f'lambda_tv: {args.lambda_tv:.2e}')

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
            f'Epoch {epoch:03d}/{args.epochs:03d} | '
            f'mse_loss: {train_metrics["mse_loss"]:.6e} | '
            f'tv_loss: {train_metrics["tv_loss"]:.6e} | '
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
    save_loss_curve(train_mse_losses, train_tv_losses, train_total_losses, val_mse_losses, loss_curve_path)
    save_validation_visualization(
        model=model,
        val_dataset=val_dataset,
        coords=coords,
        device=device,
        output_path=val_vis_path,
        sample_idx=0,
        show=args.show,
    )

    print(f'Best val mse loss: {best_val_loss:.6e}')
    print(f'Saved best model to {checkpoint_path}')
    print(f'Saved loss curve to {loss_curve_path}')
    print(f'Saved validation visualization to {val_vis_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bz + coordinate PINN for MFL inversion.')
    parser.add_argument('--train-data', default='data/training_data_train.npz')
    parser.add_argument('--val-data', default='data/training_data_val.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda-tv', type=float, default=1e-4)
    parser.add_argument('--checkpoint-path', default='checkpoints/best_model_tv.pt')
    parser.add_argument('--loss-curve-path', default='results/loss_curve_tv.png')
    parser.add_argument('--preview-path', default='results/reconstruction_preview_tv.png')
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run_full_process()
