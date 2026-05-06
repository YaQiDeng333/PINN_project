import argparse
import os

import numpy as np


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
SHAPE_POOL = ['rect', 'circle', 'ellipse', 'triangle']

METADATA_DTYPE = np.dtype([
    ('defect_type', 'U16'),
    ('center_x', 'f4'),
    ('center_y', 'f4'),
    ('width', 'f4'),
    ('height', 'f4'),
    ('radius', 'f4'),
    ('ellipse_a', 'f4'),
    ('ellipse_b', 'f4'),
    ('angle', 'f4'),
    ('triangle_vertices', 'f4', (3, 2)),
    ('area', 'f4'),
    ('depth', 'f4'),
    ('lift_off', 'f4'),
    ('noise_level', 'f4'),
])


def _init_metadata(num_samples):
    metadata = np.empty(num_samples, dtype=METADATA_DTYPE)
    metadata['defect_type'] = ''
    for name in METADATA_DTYPE.names:
        if name != 'defect_type':
            metadata[name] = np.nan
    return metadata


def _as_range(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return float(value[0]), float(value[1])
    value = float(value)
    return value, value


def _sample_uniform(rng, value):
    low, high = _as_range(value)
    return float(rng.uniform(low, high)) if high > low else low


def _generate_dataset(num_samples=1, grid_size=(100, 200), seed=None,
                      lift_off=2.0, noise_level=0.2, depth=(0.5, 2.0)):
    # 1. Space coordinates (unit: mm)
    x = np.linspace(-15, 15, grid_size[1], dtype=np.float32)
    y = np.linspace(0, 10, grid_size[0], dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # 2. Physical parameters
    B0 = 1.5
    mu_bg = 1000.0
    rng = np.random.default_rng(seed)

    signals = np.empty((num_samples, grid_size[1]), dtype=np.float32)
    mu_maps = np.empty((num_samples, grid_size[0], grid_size[1]), dtype=np.float32)
    defect_types = np.empty(num_samples, dtype='<U16')
    metadata = _init_metadata(num_samples)

    for idx in range(num_samples):
        mu_map = np.ones(grid_size, dtype=np.float32) * mu_bg

        # 3. Random defect type and center
        defect_type = str(rng.choice(SHAPE_POOL))
        center_x = float(rng.uniform(-8, 8))
        center_y = float(rng.uniform(3, 7))

        width = height = radius = ellipse_a = ellipse_b = angle = np.nan
        triangle_vertices = np.full((3, 2), np.nan, dtype=np.float32)

        # 4. Generate shape mask
        if defect_type == 'rect':
            width = float(rng.uniform(2, 5))
            height = float(rng.uniform(2, 4))
            mask = (np.abs(X - center_x) < width / 2) & (np.abs(Y - center_y) < height / 2)
            area = width * height

        elif defect_type == 'circle':
            radius = float(rng.uniform(1.5, 3))
            mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2
            area = np.pi * radius ** 2

        elif defect_type == 'ellipse':
            ellipse_a = float(rng.uniform(2, 5))
            ellipse_b = float(rng.uniform(0.5, 2))
            angle = float(rng.uniform(0, np.pi))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_shift = X - center_x
            y_shift = Y - center_y
            x_rot = x_shift * cos_a + y_shift * sin_a
            y_rot = -x_shift * sin_a + y_shift * cos_a
            mask = (x_rot ** 2 / ellipse_a ** 2 + y_rot ** 2 / ellipse_b ** 2) <= 1
            area = np.pi * ellipse_a * ellipse_b

        elif defect_type == 'triangle':
            width = float(rng.uniform(3, 6))
            height = float(rng.uniform(3, 6))
            y_bottom = center_y - height / 2
            y_top = center_y + height / 2
            y_norm = (Y - y_bottom) / height
            mask = (np.abs(X - center_x) < (width / 2) * y_norm) & (y_norm >= 0) & (y_norm <= 1)
            triangle_vertices = np.array([
                [center_x - width / 2, y_top],
                [center_x + width / 2, y_top],
                [center_x, y_bottom],
            ], dtype=np.float32)
            area = 0.5 * width * height

        else:
            raise ValueError(f'Unsupported defect_type: {defect_type}')

        sample_depth = _sample_uniform(rng, depth)
        sample_lift_off = _sample_uniform(rng, lift_off)
        sample_noise_level = _sample_uniform(rng, noise_level)

        mu_map[mask] = 1.0

        # 5. Dynamic physical signal generation
        defect_area_pixels = float(np.sum(mask))
        signal_amp = defect_area_pixels * sample_depth * 0.12
        dist = np.sqrt((X - center_x) ** 2 + (sample_lift_off + (10 - Y)) ** 2)
        bz_signal = B0 + (X - center_x) / (dist ** 3 + 1e-6) * signal_amp
        bz_at_liftoff = bz_signal[-1, :]

        # 6. Add Gaussian noise
        noise = rng.normal(0, sample_noise_level, bz_at_liftoff.shape)
        bz_noisy = bz_at_liftoff + noise

        signals[idx] = bz_noisy.astype(np.float32)
        mu_maps[idx] = mu_map
        defect_types[idx] = defect_type

        metadata[idx]['defect_type'] = defect_type
        metadata[idx]['center_x'] = center_x
        metadata[idx]['center_y'] = center_y
        metadata[idx]['width'] = width
        metadata[idx]['height'] = height
        metadata[idx]['radius'] = radius
        metadata[idx]['ellipse_a'] = ellipse_a
        metadata[idx]['ellipse_b'] = ellipse_b
        metadata[idx]['angle'] = angle
        metadata[idx]['triangle_vertices'] = triangle_vertices
        metadata[idx]['area'] = area
        metadata[idx]['depth'] = sample_depth
        metadata[idx]['lift_off'] = sample_lift_off
        metadata[idx]['noise_level'] = sample_noise_level

    return {
        'signals': signals,
        'mu_maps': mu_maps,
        'defect_types': defect_types,
        'metadata': metadata,
        'x': x,
        'y': y,
    }


def _save_dataset(dataset, output_path):
    output_path = os.path.abspath(os.path.join(PROJECT_DIR, output_path))
    if os.path.commonpath([PROJECT_DIR, output_path]) != PROJECT_DIR:
        raise ValueError(f'Output path must stay inside project directory: {output_path}')

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        output_path,
        signals=dataset['signals'],
        mu_maps=dataset['mu_maps'],
        defect_types=dataset['defect_types'],
        metadata=dataset['metadata'],
        x=dataset['x'],
        y=dataset['y'],
    )
    print(f'Saved {len(dataset["signals"])} samples to {output_path}')


def generate_training_data(num_samples=1, grid_size=(100, 200),
                           output_path='data/training_data_train.npz', seed=None):
    dataset = _generate_dataset(num_samples=num_samples, grid_size=grid_size, seed=seed)
    _save_dataset(dataset, output_path)
    return (
        dataset['x'],
        dataset['y'],
        dataset['signals'][0],
        dataset['mu_maps'][0],
        dataset['defect_types'][0],
    )


def generate_dataset_splits(train_samples=1000, val_samples=200, test_samples=200,
                            grid_size=(100, 200), output_dir='data', seed=None):
    split_sizes = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
    }

    saved_paths = {}
    rng = np.random.default_rng(seed)

    for split_name, sample_count in split_sizes.items():
        split_seed = int(rng.integers(0, np.iinfo(np.int32).max)) if seed is not None else None
        dataset = _generate_dataset(num_samples=sample_count, grid_size=grid_size, seed=split_seed)
        output_path = os.path.join(output_dir, f'training_data_{split_name}.npz')
        _save_dataset(dataset, output_path)
        saved_paths[split_name] = output_path

    return saved_paths


def visualize_random_sample(npz_path='data/training_data_train.npz', seed=None):
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=False)
    signals = data['signals']
    mu_maps = data['mu_maps']
    defect_types = data['defect_types']
    x = data['x']
    y = data['y']

    rng = np.random.default_rng(seed)
    sample_idx = int(rng.integers(0, len(signals)))
    defect_type = str(defect_types[sample_idx])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(x, signals[sample_idx], color='tab:red', label='Bz at Lift-off (Noisy)')
    ax[0].set_title(f'Simulated MFL Signal - Defect Shape: {defect_type.upper()}')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    im = ax[1].imshow(
        mu_maps[sample_idx],
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        origin='lower',
        cmap='viridis',
    )
    ax[1].set_title(f'Ground Truth $\\mu_r$ - Shape: {defect_type.upper()}')
    plt.colorbar(im, ax=ax[1], label='$\\mu_r$')
    plt.tight_layout()
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(description='Generate PINN/MFL defect datasets.')
    parser.add_argument('--train-samples', type=int, default=1000)
    parser.add_argument('--val-samples', type=int, default=200)
    parser.add_argument('--test-samples', type=int, default=200)
    parser.add_argument('--grid-y', type=int, default=100)
    parser.add_argument('--grid-x', type=int, default=200)
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--show', action='store_true', help='Show a random training sample after generation.')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    paths = generate_dataset_splits(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        grid_size=(args.grid_y, args.grid_x),
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.show:
        visualize_random_sample(paths['train'], seed=args.seed)
