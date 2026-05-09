import argparse
import os

import numpy as np


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
SIMPLE_SHAPE_POOL = ['rect', 'circle', 'ellipse', 'triangle']
COMPLEX_SHAPE_POOL = ['rotated_rect', 'polygon', 'multi_defect']
SHAPE_POOL = SIMPLE_SHAPE_POOL
MAX_COMPONENTS = 5
MAX_POLYGON_VERTICES = 12
DEFECT_MU = 1.0
V4_DATASET_NAME = 'v4_balanced_complex'
V4_POLYGON_MIN_MASK_PIXELS = 30
V4_POLYGON_MIN_SIGNAL_SNR = 5.0
V4_POLYGON_MAX_ATTEMPTS = 80
V4_AREA_BIN_SMALL_MAX_PIXELS = 120
V4_AREA_BIN_MEDIUM_MAX_PIXELS = 500
V4_MULTI_MAX_OVERLAP_FRACTION = 0.25
V4_MULTI_MIN_CENTER_DISTANCE = 2.0
V4_LEVEL_WEIGHTS = [('level1', 0.30), ('level2', 0.40), ('level3', 0.30)]
V4_LEVEL2_WEIGHTS = [('polygon', 0.75), ('rotated_rect', 0.25)]
V4_POLYGON_VERTEX_WEIGHTS = [(5, 0.30), (6, 0.20), (7, 0.20), (8, 0.15), (9, 0.15)]
V4_POLYGON_AREA_BIN_WEIGHTS = [('small', 0.40), ('medium', 0.35), ('large', 0.25)]
V4_MULTI_DEFECT_WEIGHTS = [(2, 0.40), (3, 0.60)]

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
    ('num_defects', 'f4'),
    ('component_types', 'U16', (MAX_COMPONENTS,)),
    ('component_centers', 'f4', (MAX_COMPONENTS, 2)),
    ('component_sizes', 'f4', (MAX_COMPONENTS, 4)),
    ('component_angles', 'f4', (MAX_COMPONENTS,)),
    ('polygon_vertices', 'f4', (MAX_POLYGON_VERTICES, 2)),
    ('num_vertices', 'f4'),
    ('min_mu', 'f4'),
    ('complexity_level', 'f4'),
    ('mask_pixels', 'f4'),
    ('signal_peak_to_peak', 'f4'),
    ('signal_snr', 'f4'),
    ('area_bin', 'U16'),
    ('balance_group', 'U32'),
])
METADATA_KEYS = np.array(METADATA_DTYPE.names)


def _init_metadata(num_samples):
    metadata = np.empty(num_samples, dtype=METADATA_DTYPE)
    for name in METADATA_DTYPE.names:
        field_dtype = metadata.dtype.fields[name][0]
        base_dtype = field_dtype.subdtype[0] if field_dtype.subdtype else field_dtype
        if base_dtype.kind in ('U', 'S'):
            metadata[name] = ''
        else:
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


def _allocate_counts(total, weighted_items):
    if total <= 0:
        return {key: 0 for key, _ in weighted_items}

    raw_counts = np.array([weight for _, weight in weighted_items], dtype=np.float64)
    raw_counts = raw_counts / raw_counts.sum() * total
    counts = np.floor(raw_counts).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        fractions = raw_counts - counts
        for idx in np.argsort(fractions)[::-1][:remainder]:
            counts[idx] += 1
    return {key: int(count) for (key, _), count in zip(weighted_items, counts)}


def _cycled_choices(values, count):
    if count <= 0:
        return []
    return [values[idx % len(values)] for idx in range(count)]


def _mask_pixel_count(mask):
    return int(np.sum(mask))


def _area_bin(mask_pixels):
    if mask_pixels < V4_AREA_BIN_SMALL_MAX_PIXELS:
        return 'small'
    if mask_pixels < V4_AREA_BIN_MEDIUM_MAX_PIXELS:
        return 'medium'
    return 'large'


def _signal_quality(bz_at_liftoff, B0, sample_noise_level):
    signal_delta = bz_at_liftoff - B0
    peak_to_peak = float(np.ptp(signal_delta))
    signal_snr = peak_to_peak / max(float(sample_noise_level), 1e-6)
    return peak_to_peak, signal_snr


def _safe_project_path(path):
    output_path = os.path.abspath(os.path.join(PROJECT_DIR, path))
    if os.path.commonpath([PROJECT_DIR, output_path]) != PROJECT_DIR:
        raise ValueError(f'Output path must stay inside project directory: {output_path}')
    return output_path


def _unique_path(path):
    path = _safe_project_path(path)
    if not os.path.exists(path):
        return path

    stem, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f'{stem}_{counter}{ext}'
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _pixel_area(x, y):
    dx = float(abs(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = float(abs(y[1] - y[0])) if len(y) > 1 else 1.0
    return dx * dy


def _mask_area(mask, x, y):
    return float(mask.sum()) * _pixel_area(x, y)


def _mask_center(mask, X, Y, fallback_x, fallback_y):
    if not np.any(mask):
        return float(fallback_x), float(fallback_y)
    return float(X[mask].mean()), float(Y[mask].mean())


def _polygon_mask(X, Y, vertices):
    inside = np.zeros(X.shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        crosses = ((y0 > Y) != (y1 > Y)) & (
            X < (x1 - x0) * (Y - y0) / (y1 - y0 + 1e-12) + x0
        )
        inside ^= crosses
        x0, y0 = x1, y1
    return inside


def _sample_polygon_vertices(rng, center_x, center_y, num_vertices,
                             radius_x=None, radius_y=None,
                             jitter_range=(0.65, 1.15)):
    if radius_x is None:
        radius_x = float(rng.uniform(1.8, 3.5))
    if radius_y is None:
        radius_y = float(rng.uniform(1.0, 2.5))

    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    radial_jitter = rng.uniform(jitter_range[0], jitter_range[1], num_vertices)
    vertices = np.stack([
        center_x + np.cos(angles) * radius_x * radial_jitter,
        center_y + np.sin(angles) * radius_y * radial_jitter,
    ], axis=1).astype(np.float32)
    return vertices


def _empty_component_metadata():
    return {
        'width': np.nan,
        'height': np.nan,
        'radius': np.nan,
        'ellipse_a': np.nan,
        'ellipse_b': np.nan,
        'angle': np.nan,
        'triangle_vertices': np.full((3, 2), np.nan, dtype=np.float32),
        'polygon_vertices': np.full((MAX_POLYGON_VERTICES, 2), np.nan, dtype=np.float32),
        'num_vertices': np.nan,
    }


def _generate_component(defect_type, rng, X, Y, x, y,
                        center_x=None, center_y=None, size_scale=1.0):
    if center_x is None:
        center_x = float(rng.uniform(-8, 8))
    if center_y is None:
        center_y = float(rng.uniform(3, 7))

    info = _empty_component_metadata()

    if defect_type == 'rect':
        width = float(rng.uniform(2, 5) * size_scale)
        height = float(rng.uniform(2, 4) * size_scale)
        mask = (np.abs(X - center_x) < width / 2) & (np.abs(Y - center_y) < height / 2)
        area = width * height
        info.update(width=width, height=height)

    elif defect_type == 'circle':
        radius = float(rng.uniform(1.5, 3) * size_scale)
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2
        area = np.pi * radius ** 2
        info.update(radius=radius)

    elif defect_type == 'ellipse':
        ellipse_a = float(rng.uniform(2, 5) * size_scale)
        ellipse_b = float(rng.uniform(0.5, 2) * size_scale)
        angle = float(rng.uniform(0, np.pi))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_shift = X - center_x
        y_shift = Y - center_y
        x_rot = x_shift * cos_a + y_shift * sin_a
        y_rot = -x_shift * sin_a + y_shift * cos_a
        mask = (x_rot ** 2 / ellipse_a ** 2 + y_rot ** 2 / ellipse_b ** 2) <= 1
        area = np.pi * ellipse_a * ellipse_b
        info.update(ellipse_a=ellipse_a, ellipse_b=ellipse_b, angle=angle)

    elif defect_type == 'triangle':
        width = float(rng.uniform(3, 6) * size_scale)
        height = float(rng.uniform(3, 6) * size_scale)
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
        info.update(width=width, height=height, triangle_vertices=triangle_vertices)

    elif defect_type == 'rotated_rect':
        width = float(rng.uniform(2, 5) * size_scale)
        height = float(rng.uniform(1.5, 4) * size_scale)
        angle = float(rng.uniform(0, np.pi))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_shift = X - center_x
        y_shift = Y - center_y
        x_rot = x_shift * cos_a + y_shift * sin_a
        y_rot = -x_shift * sin_a + y_shift * cos_a
        mask = (np.abs(x_rot) < width / 2) & (np.abs(y_rot) < height / 2)
        area = width * height
        info.update(width=width, height=height, angle=angle)

    elif defect_type == 'polygon':
        num_vertices = int(rng.integers(5, 10))
        vertices = _sample_polygon_vertices(rng, center_x, center_y, num_vertices)
        mask = _polygon_mask(X, Y, vertices)
        area = _mask_area(mask, x, y)
        polygon_vertices = np.full((MAX_POLYGON_VERTICES, 2), np.nan, dtype=np.float32)
        polygon_vertices[:num_vertices] = vertices
        info.update(
            width=float(vertices[:, 0].max() - vertices[:, 0].min()),
            height=float(vertices[:, 1].max() - vertices[:, 1].min()),
            polygon_vertices=polygon_vertices,
            num_vertices=float(num_vertices),
        )

    else:
        raise ValueError(f'Unsupported defect_type: {defect_type}')

    return {
        'defect_type': defect_type,
        'center_x': center_x,
        'center_y': center_y,
        'mask': mask,
        'area': float(area),
        **info,
    }


def _generate_balanced_polygon_component(rng, X, Y, x, y,
                                         num_vertices=None,
                                         target_area_bin='medium'):
    if num_vertices is None:
        num_vertices = int(rng.choice([item[0] for item in V4_POLYGON_VERTEX_WEIGHTS]))

    radius_ranges = {
        'small': ((0.7, 1.4), (0.35, 0.85)),
        'medium': ((1.4, 2.6), (0.75, 1.6)),
        'large': ((2.8, 4.4), (1.5, 2.9)),
    }
    radius_x_range, radius_y_range = radius_ranges.get(target_area_bin, radius_ranges['medium'])

    radius_x = float(rng.uniform(*radius_x_range))
    radius_y = float(rng.uniform(*radius_y_range))
    margin_x = radius_x * 1.25
    margin_y = radius_y * 1.25
    center_x = float(rng.uniform(
        max(-8.0, float(x.min()) + margin_x),
        min(8.0, float(x.max()) - margin_x),
    ))
    center_y = float(rng.uniform(
        max(3.0, float(y.min()) + margin_y),
        min(7.0, float(y.max()) - margin_y),
    ))

    vertices = _sample_polygon_vertices(
        rng,
        center_x,
        center_y,
        int(num_vertices),
        radius_x=radius_x,
        radius_y=radius_y,
        jitter_range=(0.75, 1.20),
    )
    mask = _polygon_mask(X, Y, vertices)
    polygon_vertices = np.full((MAX_POLYGON_VERTICES, 2), np.nan, dtype=np.float32)
    polygon_vertices[:num_vertices] = vertices

    return {
        'defect_type': 'polygon',
        'center_x': center_x,
        'center_y': center_y,
        'mask': mask,
        'area': _mask_area(mask, x, y),
        'width': float(vertices[:, 0].max() - vertices[:, 0].min()),
        'height': float(vertices[:, 1].max() - vertices[:, 1].min()),
        'radius': np.nan,
        'ellipse_a': np.nan,
        'ellipse_b': np.nan,
        'angle': np.nan,
        'triangle_vertices': np.full((3, 2), np.nan, dtype=np.float32),
        'polygon_vertices': polygon_vertices,
        'num_vertices': float(num_vertices),
    }


def _generate_nonempty_component(defect_type, rng, X, Y, x, y,
                                 center_x=None, center_y=None, size_scale=1.0):
    for _ in range(20):
        component = _generate_component(
            defect_type=defect_type,
            rng=rng,
            X=X,
            Y=Y,
            x=x,
            y=y,
            center_x=center_x,
            center_y=center_y,
            size_scale=size_scale,
        )
        if np.any(component['mask']):
            return component
        center_x = None
        center_y = None
    return component


def _generate_multi_defect(rng, X, Y, x, y, num_defects=None, balanced=False):
    if num_defects is None:
        num_defects = int(rng.integers(2, 4))
    component_pool = ['rect', 'circle', 'ellipse', 'rotated_rect']
    component_types = np.full((MAX_COMPONENTS,), '', dtype='<U16')
    component_centers = np.full((MAX_COMPONENTS, 2), np.nan, dtype=np.float32)
    component_sizes = np.full((MAX_COMPONENTS, 4), np.nan, dtype=np.float32)
    component_angles = np.full((MAX_COMPONENTS,), np.nan, dtype=np.float32)

    combined_mask = np.zeros(X.shape, dtype=bool)
    components = []

    for comp_idx in range(num_defects):
        best_component = None
        best_score = (np.inf, -np.inf)
        attempts = 80 if balanced else 1

        for _ in range(attempts):
            comp_type = str(rng.choice(component_pool))
            component = _generate_nonempty_component(
                defect_type=comp_type,
                rng=rng,
                X=X,
                Y=Y,
                x=x,
                y=y,
                size_scale=0.75,
            )
            component_mask = component['mask']
            overlap_fraction = 0.0
            center_distance = np.inf

            if np.any(combined_mask):
                overlap_pixels = float(np.sum(combined_mask & component_mask))
                overlap_fraction = overlap_pixels / max(float(np.sum(component_mask)), 1.0)
                existing_centers = np.array(
                    [[item['center_x'], item['center_y']] for item in components],
                    dtype=np.float32,
                )
                current_center = np.array([component['center_x'], component['center_y']], dtype=np.float32)
                center_distance = float(np.min(np.linalg.norm(existing_centers - current_center, axis=1)))

            score = (overlap_fraction, -center_distance)
            if best_component is None or score < best_score:
                best_component = component
                best_score = score

            if (
                not balanced
                or (
                    overlap_fraction <= V4_MULTI_MAX_OVERLAP_FRACTION
                    and center_distance >= V4_MULTI_MIN_CENTER_DISTANCE
                )
            ):
                break
        else:
            component = best_component

        combined_mask |= component['mask']
        components.append(component)

        component_types[comp_idx] = comp_type
        component_centers[comp_idx] = [component['center_x'], component['center_y']]
        component_sizes[comp_idx] = [
            component['width'],
            component['height'],
            component['radius'] if np.isfinite(component['radius']) else component['ellipse_a'],
            component['ellipse_b'],
        ]
        component_angles[comp_idx] = component['angle']

    center_x, center_y = _mask_center(combined_mask, X, Y, 0.0, 5.0)
    return {
        'defect_type': 'multi_defect',
        'center_x': center_x,
        'center_y': center_y,
        'mask': combined_mask,
        'area': _mask_area(combined_mask, x, y),
        'width': np.nan,
        'height': np.nan,
        'radius': np.nan,
        'ellipse_a': np.nan,
        'ellipse_b': np.nan,
        'angle': np.nan,
        'triangle_vertices': np.full((3, 2), np.nan, dtype=np.float32),
        'polygon_vertices': np.full((MAX_POLYGON_VERTICES, 2), np.nan, dtype=np.float32),
        'num_vertices': np.nan,
        'num_defects': float(num_defects),
        'component_types': component_types,
        'component_centers': component_centers,
        'component_sizes': component_sizes,
        'component_angles': component_angles,
        'components': components,
    }


def _build_v4_balanced_schedule(num_samples, rng):
    level_counts = _allocate_counts(num_samples, V4_LEVEL_WEIGHTS)
    schedule = []

    simple_types = _cycled_choices(SIMPLE_SHAPE_POOL, level_counts['level1'])
    for defect_type in simple_types:
        schedule.append({
            'defect_type': defect_type,
            'complexity_level': 1.0,
            'balance_group': f'level1_{defect_type}',
        })

    level2_counts = _allocate_counts(level_counts['level2'], V4_LEVEL2_WEIGHTS)
    vertex_counts = _allocate_counts(level2_counts['polygon'], V4_POLYGON_VERTEX_WEIGHTS)
    area_bin_counts = _allocate_counts(level2_counts['polygon'], V4_POLYGON_AREA_BIN_WEIGHTS)
    area_bins = []
    for area_bin, count in area_bin_counts.items():
        area_bins.extend([area_bin] * count)
    rng.shuffle(area_bins)
    polygon_idx = 0

    for num_vertices, count in vertex_counts.items():
        for _ in range(count):
            target_area_bin = area_bins[polygon_idx] if polygon_idx < len(area_bins) else 'medium'
            polygon_idx += 1
            schedule.append({
                'defect_type': 'polygon',
                'num_vertices': int(num_vertices),
                'target_area_bin': target_area_bin,
                'complexity_level': 2.0,
                'balance_group': f'level2_polygon_v{num_vertices}_{target_area_bin}',
            })

    for _ in range(level2_counts['rotated_rect']):
        schedule.append({
            'defect_type': 'rotated_rect',
            'complexity_level': 2.0,
            'balance_group': 'level2_rotated_rect',
        })

    multi_counts = _allocate_counts(level_counts['level3'], V4_MULTI_DEFECT_WEIGHTS)
    for num_defects, count in multi_counts.items():
        for _ in range(count):
            schedule.append({
                'defect_type': 'multi_defect',
                'num_defects': int(num_defects),
                'complexity_level': 3.0,
                'balance_group': f'level3_multi_defect_{num_defects}',
            })

    if len(schedule) != num_samples:
        raise RuntimeError(f'v4 schedule size mismatch: {len(schedule)} != {num_samples}')

    rng.shuffle(schedule)
    return schedule


def _generate_v4_shape_from_spec(spec, rng, X, Y, x, y):
    defect_type = spec['defect_type']

    if defect_type == 'multi_defect':
        shape_info = _generate_multi_defect(
            rng,
            X,
            Y,
            x,
            y,
            num_defects=spec.get('num_defects'),
            balanced=True,
        )
        signal_components = shape_info['components']
    elif defect_type == 'polygon':
        shape_info = _generate_balanced_polygon_component(
            rng,
            X,
            Y,
            x,
            y,
            num_vertices=spec.get('num_vertices'),
            target_area_bin=spec.get('target_area_bin', 'medium'),
        )
        signal_components = [shape_info]
    else:
        shape_info = _generate_nonempty_component(defect_type, rng, X, Y, x, y)
        signal_components = [shape_info]

    shape_info['balance_group'] = spec['balance_group']
    return shape_info, signal_components, float(spec['complexity_level'])


def _generate_bz_signal(x, signal_components, sample_depth, sample_lift_off, B0):
    bz_at_liftoff = np.ones_like(x, dtype=np.float32) * B0
    for component in signal_components:
        area_pixels = float(np.sum(component['mask']))
        if area_pixels <= 0:
            continue
        center_x = float(component['center_x'])
        signal_amp = area_pixels * sample_depth * 0.12
        dist = np.sqrt((x - center_x) ** 2 + sample_lift_off ** 2)
        bz_at_liftoff += (x - center_x) / (dist ** 3 + 1e-6) * signal_amp
    return bz_at_liftoff


def _fill_metadata(metadata_row, shape_info, sample_depth, sample_lift_off,
                   sample_noise_level, complexity_level,
                   signal_peak_to_peak, signal_snr):
    metadata_row['defect_type'] = shape_info['defect_type']
    metadata_row['center_x'] = shape_info['center_x']
    metadata_row['center_y'] = shape_info['center_y']
    metadata_row['width'] = shape_info['width']
    metadata_row['height'] = shape_info['height']
    metadata_row['radius'] = shape_info['radius']
    metadata_row['ellipse_a'] = shape_info['ellipse_a']
    metadata_row['ellipse_b'] = shape_info['ellipse_b']
    metadata_row['angle'] = shape_info['angle']
    metadata_row['triangle_vertices'] = shape_info['triangle_vertices']
    metadata_row['area'] = shape_info['area']
    metadata_row['depth'] = sample_depth
    metadata_row['lift_off'] = sample_lift_off
    metadata_row['noise_level'] = sample_noise_level
    metadata_row['num_defects'] = shape_info.get('num_defects', 1.0)
    metadata_row['component_types'] = shape_info.get(
        'component_types',
        np.full((MAX_COMPONENTS,), '', dtype='<U16'),
    )
    metadata_row['component_centers'] = shape_info.get(
        'component_centers',
        np.full((MAX_COMPONENTS, 2), np.nan, dtype=np.float32),
    )
    metadata_row['component_sizes'] = shape_info.get(
        'component_sizes',
        np.full((MAX_COMPONENTS, 4), np.nan, dtype=np.float32),
    )
    metadata_row['component_angles'] = shape_info.get(
        'component_angles',
        np.full((MAX_COMPONENTS,), np.nan, dtype=np.float32),
    )
    metadata_row['polygon_vertices'] = shape_info['polygon_vertices']
    metadata_row['num_vertices'] = shape_info['num_vertices']
    metadata_row['min_mu'] = DEFECT_MU
    metadata_row['complexity_level'] = complexity_level
    mask_pixels = _mask_pixel_count(shape_info['mask'])
    metadata_row['mask_pixels'] = float(mask_pixels)
    metadata_row['signal_peak_to_peak'] = signal_peak_to_peak
    metadata_row['signal_snr'] = signal_snr
    metadata_row['area_bin'] = _area_bin(mask_pixels)
    metadata_row['balance_group'] = shape_info.get('balance_group', '')


def _generate_dataset(num_samples=1, grid_size=(100, 200), seed=None,
                      lift_off=2.0, noise_level=0.2, depth=(0.5, 2.0),
                      complex_shapes=False, dataset='simple'):
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
    shape_pool = COMPLEX_SHAPE_POOL if complex_shapes else SIMPLE_SHAPE_POOL
    use_v4_balanced = dataset == V4_DATASET_NAME
    v4_schedule = _build_v4_balanced_schedule(num_samples, rng) if use_v4_balanced else None

    for idx in range(num_samples):
        if use_v4_balanced and v4_schedule[idx]['defect_type'] == 'polygon':
            best_state = None
            best_score = -np.inf

            for _ in range(V4_POLYGON_MAX_ATTEMPTS):
                mu_map = np.ones(grid_size, dtype=np.float32) * mu_bg
                shape_info, signal_components, complexity_level = _generate_v4_shape_from_spec(
                    v4_schedule[idx],
                    rng,
                    X,
                    Y,
                    x,
                    y,
                )
                mask = shape_info['mask']
                sample_depth = _sample_uniform(rng, depth)
                sample_lift_off = _sample_uniform(rng, lift_off)
                sample_noise_level = _sample_uniform(rng, noise_level)
                mu_map[mask] = DEFECT_MU
                bz_at_liftoff = _generate_bz_signal(
                    x=x,
                    signal_components=signal_components,
                    sample_depth=sample_depth,
                    sample_lift_off=sample_lift_off,
                    B0=B0,
                )
                signal_peak_to_peak, signal_snr = _signal_quality(
                    bz_at_liftoff,
                    B0,
                    sample_noise_level,
                )

                mask_pixels = _mask_pixel_count(mask)
                mask_score = mask_pixels / max(float(V4_POLYGON_MIN_MASK_PIXELS), 1.0)
                snr_score = signal_snr / max(float(V4_POLYGON_MIN_SIGNAL_SNR), 1e-6)
                score = min(mask_score, snr_score)
                state = (
                    mu_map,
                    shape_info,
                    sample_depth,
                    sample_lift_off,
                    sample_noise_level,
                    bz_at_liftoff,
                    signal_peak_to_peak,
                    signal_snr,
                    mask,
                    complexity_level,
                )

                if score > best_score:
                    best_state = state
                    best_score = score

                if (
                    mask_pixels >= V4_POLYGON_MIN_MASK_PIXELS
                    and signal_snr >= V4_POLYGON_MIN_SIGNAL_SNR
                ):
                    break

            (
                mu_map,
                shape_info,
                sample_depth,
                sample_lift_off,
                sample_noise_level,
                bz_at_liftoff,
                signal_peak_to_peak,
                signal_snr,
                mask,
                complexity_level,
            ) = best_state

        else:
            mu_map = np.ones(grid_size, dtype=np.float32) * mu_bg

            # 3. Random defect type and shape. In complex mode the first samples
            # cycle through the new classes so small validation runs cover all of them.
            if use_v4_balanced:
                shape_info, signal_components, complexity_level = _generate_v4_shape_from_spec(
                    v4_schedule[idx],
                    rng,
                    X,
                    Y,
                    x,
                    y,
                )
            else:
                if complex_shapes and idx < len(shape_pool):
                    defect_type = shape_pool[idx]
                else:
                    defect_type = str(rng.choice(shape_pool))

                if defect_type == 'multi_defect':
                    shape_info = _generate_multi_defect(rng, X, Y, x, y)
                    signal_components = shape_info['components']
                    complexity_level = 3.0
                else:
                    shape_info = _generate_nonempty_component(defect_type, rng, X, Y, x, y)
                    signal_components = [shape_info]
                    complexity_level = 2.0 if complex_shapes else 1.0

            mask = shape_info['mask']

            sample_depth = _sample_uniform(rng, depth)
            sample_lift_off = _sample_uniform(rng, lift_off)
            sample_noise_level = _sample_uniform(rng, noise_level)

            mu_map[mask] = DEFECT_MU

            # 5. Dynamic physical signal generation
            bz_at_liftoff = _generate_bz_signal(
                x=x,
                signal_components=signal_components,
                sample_depth=sample_depth,
                sample_lift_off=sample_lift_off,
                B0=B0,
            )
            signal_peak_to_peak, signal_snr = _signal_quality(
                bz_at_liftoff,
                B0,
                sample_noise_level,
            )

        # 6. Add Gaussian noise
        noise = rng.normal(0, sample_noise_level, bz_at_liftoff.shape)
        bz_noisy = bz_at_liftoff + noise

        signals[idx] = bz_noisy.astype(np.float32)
        mu_maps[idx] = mu_map
        defect_types[idx] = shape_info['defect_type']

        _fill_metadata(
            metadata_row=metadata[idx],
            shape_info=shape_info,
            sample_depth=sample_depth,
            sample_lift_off=sample_lift_off,
            sample_noise_level=sample_noise_level,
            complexity_level=complexity_level,
            signal_peak_to_peak=signal_peak_to_peak,
            signal_snr=signal_snr,
        )

    return {
        'signals': signals,
        'mu_maps': mu_maps,
        'defect_types': defect_types,
        'metadata': metadata,
        'x': x,
        'y': y,
    }


def _save_dataset(dataset, output_path):
    output_path = _safe_project_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        signals=dataset['signals'],
        mu_maps=dataset['mu_maps'],
        defect_types=dataset['defect_types'],
        metadata=dataset['metadata'],
        metadata_keys=METADATA_KEYS,
        x=dataset['x'],
        y=dataset['y'],
    )
    print(f'Saved {len(dataset["signals"])} samples to {output_path}')


def generate_training_data(num_samples=1, grid_size=(100, 200),
                           output_path='data/training_data_train.npz', seed=None,
                           complex_shapes=False, dataset='simple'):
    generated_dataset = _generate_dataset(
        num_samples=num_samples,
        grid_size=grid_size,
        seed=seed,
        complex_shapes=complex_shapes,
        dataset=dataset,
    )
    _save_dataset(generated_dataset, output_path)
    return (
        generated_dataset['x'],
        generated_dataset['y'],
        generated_dataset['signals'][0],
        generated_dataset['mu_maps'][0],
        generated_dataset['defect_types'][0],
    )


def generate_dataset_splits(train_samples=1000, val_samples=200, test_samples=200,
                            grid_size=(100, 200), output_dir='data', seed=None,
                            complex_shapes=False, dataset_prefix='training_data',
                            dataset='simple'):
    split_sizes = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
    }

    saved_paths = {}
    rng = np.random.default_rng(seed)

    for split_name, sample_count in split_sizes.items():
        split_seed = int(rng.integers(0, np.iinfo(np.int32).max)) if seed is not None else None
        generated_dataset = _generate_dataset(
            num_samples=sample_count,
            grid_size=grid_size,
            seed=split_seed,
            complex_shapes=complex_shapes,
            dataset=dataset,
        )
        output_path = os.path.join(output_dir, f'{dataset_prefix}_{split_name}.npz')
        _save_dataset(generated_dataset, output_path)
        saved_paths[split_name] = output_path

    return saved_paths


def visualize_random_sample(npz_path='data/training_data_train.npz', seed=None):
    import matplotlib.pyplot as plt

    data = np.load(_safe_project_path(npz_path), allow_pickle=False)
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


def _select_visual_check_indices(defect_types, metadata, sample_count, rng, focus_v4=False):
    if not focus_v4:
        if sample_count == len(defect_types):
            return np.arange(sample_count)
        return rng.choice(len(defect_types), size=sample_count, replace=False)

    selected = []

    def add_first(mask):
        candidates = np.where(mask)[0]
        for candidate in candidates:
            candidate = int(candidate)
            if candidate not in selected:
                selected.append(candidate)
                return

    polygon_mask = defect_types == 'polygon'
    add_first(polygon_mask & (metadata['num_vertices'] == 5))

    polygon_indices = np.where(polygon_mask)[0]
    if len(polygon_indices) > 0:
        smallest_polygon = int(polygon_indices[np.argmin(metadata['mask_pixels'][polygon_indices])])
        if smallest_polygon not in selected:
            selected.append(smallest_polygon)

    add_first((defect_types == 'multi_defect') & (metadata['num_defects'] == 3))
    add_first(defect_types == 'rotated_rect')
    add_first(metadata['complexity_level'] == 1)

    remaining = [idx for idx in range(len(defect_types)) if idx not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, sample_count - len(selected))])
    return np.array(selected[:sample_count], dtype=int)


def save_visual_check_samples(npz_path='data/training_data_train.npz',
                              output_dir='results/previews',
                              output_prefix='data_check',
                              num_samples=5,
                              seed=None,
                              focus_v4=False):
    import matplotlib.pyplot as plt

    data = np.load(_safe_project_path(npz_path), allow_pickle=False)
    signals = data['signals']
    mu_maps = data['mu_maps']
    defect_types = data['defect_types']
    metadata = data['metadata']
    x = data['x']
    y = data['y']

    output_dir = _safe_project_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, len(signals))
    if sample_count <= 0:
        return []

    sample_indices = _select_visual_check_indices(
        defect_types,
        metadata,
        sample_count,
        rng,
        focus_v4=focus_v4,
    )

    saved_paths = []
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    for out_idx, sample_idx in enumerate(sample_indices):
        sample_idx = int(sample_idx)
        defect_type = str(defect_types[sample_idx])
        num_defects = float(metadata[sample_idx]['num_defects'])

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(x, signals[sample_idx], color='tab:red', label='Bz at Lift-off (Noisy)')
        ax[0].set_title(f'Bz Signal - {defect_type} - sample {sample_idx}')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        im = ax[1].imshow(
            mu_maps[sample_idx],
            extent=extent,
            origin='lower',
            cmap='viridis',
        )
        ax[1].set_title(f'Mu Map - {defect_type} - num_defects={num_defects:g}')
        plt.colorbar(im, ax=ax[1], label='$\\mu_r$')

        plt.tight_layout()
        output_path = _unique_path(os.path.join(output_dir, f'{output_prefix}_{out_idx:03d}.png'))
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def _distribution_text(values):
    values, counts = np.unique(values, return_counts=True)
    return ', '.join(f'{value}={count}' for value, count in zip(values, counts))


def _numeric_distribution_text(values):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 'none'
    unique_values, counts = np.unique(values.astype(int), return_counts=True)
    return ', '.join(f'{value}={count}' for value, count in zip(unique_values, counts))


def _target_area_bin_from_balance_group(balance_group):
    parts = str(balance_group).split('_')
    if parts and parts[-1] in ('small', 'medium', 'large'):
        return parts[-1]
    return 'unknown'


def _polygon_area_bin_cross_table(metadata, polygon_idx):
    bins = ['small', 'medium', 'large']
    table = {target: {actual: 0 for actual in bins} for target in bins}
    table['unknown'] = {actual: 0 for actual in bins}

    for sample_idx in polygon_idx:
        target = _target_area_bin_from_balance_group(metadata['balance_group'][sample_idx])
        actual = str(metadata['area_bin'][sample_idx])
        if target not in table:
            table[target] = {bin_name: 0 for bin_name in bins}
        if actual in bins:
            table[target][actual] += 1

    lines = ['polygon_target_vs_actual_area_bin:']
    for target in ['small', 'medium', 'large', 'unknown']:
        counts = ', '.join(f'{actual}={table[target][actual]}' for actual in bins)
        lines.append(f'  target_{target}: {counts}')
    return lines


def write_dataset_summary(saved_paths, output_path, dataset_name='dataset'):
    output_path = _safe_project_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        f'{dataset_name} 数据集检查摘要',
        '',
        f'polygon_min_mask_pixels = {V4_POLYGON_MIN_MASK_PIXELS}',
        f'polygon_min_signal_snr = {V4_POLYGON_MIN_SIGNAL_SNR:g}',
        f'area_bin_small = mask_pixels < {V4_AREA_BIN_SMALL_MAX_PIXELS}',
        f'area_bin_medium = {V4_AREA_BIN_SMALL_MAX_PIXELS} <= mask_pixels < {V4_AREA_BIN_MEDIUM_MAX_PIXELS}',
        f'area_bin_large = mask_pixels >= {V4_AREA_BIN_MEDIUM_MAX_PIXELS}',
        '',
    ]

    required_keys = {'signals', 'mu_maps', 'defect_types', 'metadata', 'metadata_keys', 'x', 'y'}

    for split_name, npz_path in saved_paths.items():
        data = np.load(_safe_project_path(npz_path), allow_pickle=False)
        keys = set(data.files)
        signals = data['signals']
        mu_maps = data['mu_maps']
        defect_types = data['defect_types'].astype(str)
        metadata = data['metadata']
        metadata_keys = data['metadata_keys'].astype(str)
        mask = mu_maps < 500

        missing_keys = sorted(required_keys - keys)
        metadata_keys_match = list(metadata_keys) == list(metadata.dtype.names)
        signal_has_bad = bool(np.isnan(signals).any() or np.isinf(signals).any())
        mu_has_bad = bool(np.isnan(mu_maps).any() or np.isinf(mu_maps).any())
        empty_masks = int(np.sum(mask.reshape(mask.shape[0], -1).sum(axis=1) == 0))
        all_mask_pixels = metadata['mask_pixels']
        all_signal_snr = metadata['signal_snr']

        polygon_idx = np.where(defect_types == 'polygon')[0]
        multi_idx = np.where(defect_types == 'multi_defect')[0]

        lines.extend([
            f'## {split_name}',
            '',
            f'file = {npz_path}',
            f'samples = {len(signals)}',
            f'keys_ok = {not missing_keys}',
            f'missing_keys = {missing_keys}',
            f'metadata_keys_match_metadata = {metadata_keys_match}',
            f'signals_shape = {signals.shape}',
            f'mu_maps_shape = {mu_maps.shape}',
            f'signal_has_nan_or_inf = {signal_has_bad}',
            f'mu_map_has_nan_or_inf = {mu_has_bad}',
            f'empty_defect_masks = {empty_masks}',
            f'defect_type_distribution = {_distribution_text(defect_types)}',
            f'complexity_level_distribution = {_numeric_distribution_text(metadata["complexity_level"])}',
            f'mask_pixels_min = {float(np.nanmin(all_mask_pixels)):.2f}',
            f'mask_pixels_mean = {float(np.nanmean(all_mask_pixels)):.2f}',
            f'mask_pixels_max = {float(np.nanmax(all_mask_pixels)):.2f}',
            f'signal_snr_min = {float(np.nanmin(all_signal_snr)):.4f}',
            f'signal_snr_mean = {float(np.nanmean(all_signal_snr)):.4f}',
            f'signal_snr_max = {float(np.nanmax(all_signal_snr)):.4f}',
        ])

        if len(polygon_idx) > 0:
            polygon_mask_pixels = metadata['mask_pixels'][polygon_idx]
            polygon_snr = metadata['signal_snr'][polygon_idx]
            below_min_pixels = int(np.sum(polygon_mask_pixels < V4_POLYGON_MIN_MASK_PIXELS))
            below_min_snr = int(np.sum(polygon_snr < V4_POLYGON_MIN_SIGNAL_SNR))
            lines.extend([
                f'polygon_num_vertices_distribution = {_numeric_distribution_text(metadata["num_vertices"][polygon_idx])}',
                f'polygon_area_bin_distribution = {_distribution_text(metadata["area_bin"][polygon_idx].astype(str))}',
                f'polygon_mask_pixels_min = {float(np.nanmin(polygon_mask_pixels)):.2f}',
                f'polygon_mask_pixels_mean = {float(np.nanmean(polygon_mask_pixels)):.2f}',
                f'polygon_mask_pixels_max = {float(np.nanmax(polygon_mask_pixels)):.2f}',
                f'polygon_below_min_mask_pixels = {below_min_pixels}',
                f'polygon_signal_snr_min = {float(np.nanmin(polygon_snr)):.4f}',
                f'polygon_signal_snr_mean = {float(np.nanmean(polygon_snr)):.4f}',
                f'polygon_signal_snr_max = {float(np.nanmax(polygon_snr)):.4f}',
                f'polygon_below_min_signal_snr = {below_min_snr}',
            ])
            lines.extend(_polygon_area_bin_cross_table(metadata, polygon_idx))
        else:
            lines.append('polygon_num_vertices_distribution = none')

        if len(multi_idx) > 0:
            lines.append(
                f'multi_defect_num_defects_distribution = '
                f'{_numeric_distribution_text(metadata["num_defects"][multi_idx])}'
            )
        else:
            lines.append('multi_defect_num_defects_distribution = none')

        lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Saved dataset summary to {output_path}')
    return output_path


def _parse_args():
    parser = argparse.ArgumentParser(description='Generate PINN/MFL defect datasets.')
    parser.add_argument('--complex', action='store_true', dest='complex_shapes',
                        help='Generate v3 complex defect splits with rotated_rect, polygon, and multi_defect.')
    parser.add_argument('--dataset', choices=['simple', 'v3_complex', V4_DATASET_NAME], default=None,
                        help='Dataset version to generate. Use v4_balanced_complex for balanced complex defects.')
    parser.add_argument('--train-samples', type=int, default=1000)
    parser.add_argument('--val-samples', type=int, default=200)
    parser.add_argument('--test-samples', type=int, default=200)
    parser.add_argument('--grid-y', type=int, default=100)
    parser.add_argument('--grid-x', type=int, default=200)
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--dataset-prefix', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--show', action='store_true', help='Show a random training sample after generation.')
    parser.add_argument('--visual-check-samples', type=int, default=5)
    parser.add_argument('--visual-output-dir', default='results/previews')
    parser.add_argument('--summary-output', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    dataset_name = args.dataset
    if dataset_name is None:
        dataset_name = 'v3_complex' if args.complex_shapes else 'simple'
    complex_shapes = args.complex_shapes or dataset_name in ('v3_complex', V4_DATASET_NAME)

    dataset_prefix = args.dataset_prefix
    if dataset_prefix is None:
        if dataset_name == V4_DATASET_NAME:
            dataset_prefix = 'training_data_v4_balanced_complex'
        elif dataset_name == 'v3_complex':
            dataset_prefix = 'training_data_v3_complex'
        else:
            dataset_prefix = 'training_data'

    paths = generate_dataset_splits(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        grid_size=(args.grid_y, args.grid_x),
        output_dir=args.output_dir,
        seed=args.seed,
        complex_shapes=complex_shapes,
        dataset_prefix=dataset_prefix,
        dataset=dataset_name,
    )

    if complex_shapes and args.visual_check_samples > 0:
        visual_prefix = (
            'data_v4_balanced_complex_check'
            if dataset_name == V4_DATASET_NAME
            else 'data_v3_complex_check'
        )
        saved_visuals = save_visual_check_samples(
            paths['train'],
            output_dir=args.visual_output_dir,
            output_prefix=visual_prefix,
            num_samples=args.visual_check_samples,
            seed=args.seed,
            focus_v4=dataset_name == V4_DATASET_NAME,
        )
        for path in saved_visuals:
            print(f'Saved visual check to {path}')

    if dataset_name == V4_DATASET_NAME:
        summary_output = args.summary_output or 'results/summaries/v4_balanced_complex_dataset_summary.txt'
        write_dataset_summary(
            paths,
            output_path=summary_output,
            dataset_name=dataset_name,
        )

    if args.show:
        visualize_random_sample(paths['train'], seed=args.seed)
