# coding=utf-8
"""Run the original SOT implementation on the repository's MATLAB datasets.

Run from the repository root, for example:
    python sot/main.py --dataset OPP --source-device 3 --target-device 5
"""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path.cwd()
SOT_DIR = REPO_ROOT / 'sot'
if str(SOT_DIR) not in sys.path:
    sys.path.insert(0, str(SOT_DIR))

from SOT import SOT


DATASET_FILES = {
    'DSADS': ('dsads.mat', 'data_dsads', -2),
    'OPP': ('opp_loco.mat', 'data_opp_loco', -3),
    'PAMAP': ('pamap.mat', 'data_pamap', -2),
}
N_FEATURES_PER_DEVICE = 81


def load_mat_domains(dataset, source_device, target_device):
    """Load the same device slices, labels, and rows used by demo.m."""
    filename, variable, label_column = DATASET_FILES[dataset]
    mat_path = REPO_ROOT / 'data' / filename
    if not mat_path.is_file():
        raise FileNotFoundError(f'MAT dataset not found: {mat_path}')

    data = loadmat(mat_path).get(variable)
    if data is None or data.ndim != 2:
        raise ValueError(f'{mat_path} does not contain a 2-D {variable} array.')

    def feature_slice(device):
        if device < 1:
            raise ValueError('Device indices must be positive integers.')
        start = (device - 1) * N_FEATURES_PER_DEVICE
        stop = start + N_FEATURES_PER_DEVICE
        if stop > data.shape[1] + label_column:
            raise ValueError(
                f'{dataset} device {device} needs columns {start + 1}:{stop}, '
                f'but {filename} only has {data.shape[1]} columns.')
        return slice(start, stop)

    source_columns = feature_slice(source_device)
    target_columns = feature_slice(target_device)
    label_index = data.shape[1] + label_column
    x_source = np.asarray(data[:, source_columns], dtype=np.float64)
    y_source = np.asarray(data[:, label_index], dtype=np.float64).reshape(-1)
    x_target = np.asarray(data[:, target_columns], dtype=np.float64)
    y_target = np.asarray(data[:, label_index], dtype=np.float64).reshape(-1)
    return x_source, y_source, x_target, y_target


def zscore_rows_domain(x):
    """Match MATLAB zscore's per-feature sample-standard-deviation scaling."""
    if x.shape[0] < 2:
        raise ValueError('Each domain needs at least two rows for z-score scaling.')
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    normalized = (x - mean) / scale
    if not np.isfinite(normalized).all():
        raise ValueError('Z-scored features contain NaN or infinite values.')
    return normalized


def run_sot(dataset='OPP', source_device=3, target_device=5, seed=0,
            clusters=19, reg_e=0.5, reg_cl=1.0, reg_ce=3.0):
    if source_device == target_device:
        raise ValueError('Source and target devices must be different.')

    x_source, y_source_raw, x_target, y_target_raw = load_mat_domains(
        dataset, source_device, target_device)

    # Use a fixed permutation to mirror demo.m's row shuffle reproducibly.
    permutation_rng = np.random.default_rng(seed)
    x_source_order = permutation_rng.permutation(x_source.shape[0])
    x_target_order = permutation_rng.permutation(x_target.shape[0])
    x_source, y_source_raw = x_source[x_source_order], y_source_raw[x_source_order]
    x_target, y_target_raw = x_target[x_target_order], y_target_raw[x_target_order]
    x_source = zscore_rows_domain(x_source)
    x_target = zscore_rows_domain(x_target)

    # SOT.py assumes class labels are 0..K-1; restore original labels for scoring.
    class_labels, y_source = np.unique(y_source_raw, return_inverse=True)
    target_is_known = np.isin(y_target_raw, class_labels)
    y_target = np.full(y_target_raw.shape, -1, dtype=np.int64)
    y_target[target_is_known] = np.searchsorted(
        class_labels, y_target_raw[target_is_known])
    if class_labels.size < 2:
        raise ValueError('SOT needs at least two source classes.')

    random.seed(seed)
    np.random.seed(seed)
    with tempfile.TemporaryDirectory(prefix='sot_run_') as run_dir:
        root_dir = run_dir + os.sep
        data_dir = Path(run_dir) / 'data'
        model_dir = Path(run_dir) / 'model'
        data_dir.mkdir()
        model_dir.mkdir()

        estimator = SOT('ACT', root_dir, clusters, reg_e, reg_cl, reg_ce)
        encoded_predictions, _ = estimator.fit_predict(
            x_source, y_source, x_target, y_target,
            str(data_dir / 'source.json'), 'source',
            str(data_dir / 'target.json'), str(model_dir / 'target_gmm'),
            'target')

    encoded_predictions = np.asarray(encoded_predictions).reshape(-1)
    if encoded_predictions.size != y_target_raw.size:
        raise ValueError(
            f'SOT returned {encoded_predictions.size} predictions for '
            f'{y_target_raw.size} target rows.')
    if (not np.isfinite(encoded_predictions).all()
            or np.any(encoded_predictions < 0)
            or np.any(encoded_predictions >= class_labels.size)):
        raise ValueError('SOT returned an invalid class prediction.')

    predictions = class_labels[encoded_predictions.astype(np.int64)]
    accuracy = float(np.mean(predictions == y_target_raw))
    return accuracy, predictions, y_target_raw


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run SOT on a MATLAB .mat dataset.')
    parser.add_argument('--dataset', choices=DATASET_FILES, default='OPP')
    parser.add_argument('--source-device', type=int, default=3)
    parser.add_argument('--target-device', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clusters', type=int, default=19)
    parser.add_argument('--reg-e', type=float, default=0.5)
    parser.add_argument('--reg-cl', type=float, default=1.0)
    parser.add_argument('--reg-ce', type=float, default=3.0)
    args = parser.parse_args(argv)

    accuracy, _, y_target = run_sot(
        dataset=args.dataset,
        source_device=args.source_device,
        target_device=args.target_device,
        seed=args.seed,
        clusters=args.clusters,
        reg_e=args.reg_e,
        reg_cl=args.reg_cl,
        reg_ce=args.reg_ce)
    print(f'Dataset: {args.dataset}, device {args.source_device} -> {args.target_device}')
    print(f'Source/target rows: {y_target.size}/{y_target.size}')
    print(f'Seed: {args.seed}')
    print(f'SOT accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')


if __name__ == '__main__':
    main()
