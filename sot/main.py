# coding: utf-8
"""Run SOT on the same within- and cross-dataset tasks as the MATLAB demos.

Examples
--------
python sot/main.py --dataset DSADS --source-device 2 --target-device 3
python sot/main.py --all
python sot/main.py --cross --source-dataset DSADS --target-dataset PAMAP \
    --source-device 1 --target-device 2
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.neighbors import KNeighborsClassifier

from SOT import SOT


REPO_ROOT = Path(__file__).resolve().parent.parent
N_FEATURES_PER_DEVICE = 81
WITHIN_DATASETS = {
    'DSADS': ('dsads.mat', 'data_dsads', -2),
    'OPP': ('opp_loco.mat', 'data_opp_loco', -3),
    'PAMAP': ('pamap.mat', 'data_pamap', -2),
}
CROSS_DATASETS = {
    'DSADS': ('cross_dsads.mat', 'data_dsads'),
    'OPP': ('cross_opp.mat', 'data_opp'),
    'PAMAP': ('cross_pamap.mat', 'data_pamap'),
}
WITHIN_TASKS = (
    ('DSADS', 2, 3),
    ('DSADS', 4, 5),
    ('OPP', 2, 4),
    ('OPP', 3, 5),
    ('DSADS', 2, 1),
    ('PAMAP', 1, 2),
    ('OPP', 3, 1),
    ('OPP', 2, 1),
)


def _device_columns(device):
    if device < 1 or device > 5:
        raise ValueError('Device indices must be in the range 1..5.')
    start = (device - 1) * N_FEATURES_PER_DEVICE
    return slice(start, start + N_FEATURES_PER_DEVICE)


def _load_mat_array(filename, variable):
    path = REPO_ROOT / 'data' / filename
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    data = loadmat(path).get(variable)
    if data is None or data.ndim != 2:
        raise ValueError(f'{path} does not contain the expected 2-D {variable} array.')
    return np.asarray(data, dtype=np.float64)


def load_domain(dataset, device, *, cross_dataset=False):
    """Load one 81-feature device domain, as in demo.m/demo_cross.m."""
    columns = _device_columns(device)
    if cross_dataset:
        filename, variable = CROSS_DATASETS[dataset]
        data = _load_mat_array(filename, variable)
        labels = data[:, -1]
        # demo_cross.m maps PAMAP's four IDs to the shared activity order.
        if dataset == 'PAMAP':
            mapping = np.array([4, 3, 1, 2])
            label_indices = labels.astype(np.int64) - 1
            if not np.isin(label_indices, np.arange(len(mapping))).all():
                raise ValueError('cross_pamap.mat has an unexpected activity label.')
            labels = mapping[label_indices]
    else:
        filename, variable, label_offset = WITHIN_DATASETS[dataset]
        data = _load_mat_array(filename, variable)
        labels = data[:, data.shape[1] + label_offset]

    if columns.stop > data.shape[1]:
        raise ValueError(f'{dataset} has no device {device} feature block.')
    return data[:, columns], labels


def zscore_matlab(x):
    """MATLAB zscore semantics: sample standard deviation, constant -> zero."""
    x = np.asarray(x, dtype=np.float64)
    scale = np.std(x, axis=0, ddof=1)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    normalised = (x - np.mean(x, axis=0)) / scale
    if not np.isfinite(normalised).all():
        raise ValueError('Z-score preprocessing created NaN or infinity.')
    return normalised


def prepare_domains(source_dataset, target_dataset, source_device, target_device,
                    *, cross_dataset=False, seed=0):
    """Load, independently shuffle, and standardise source and target domains."""
    source_x, source_y = load_domain(
        source_dataset, source_device, cross_dataset=cross_dataset)
    target_x, target_y = load_domain(
        target_dataset, target_device, cross_dataset=cross_dataset)
    rng = np.random.default_rng(seed)
    source_order = rng.permutation(len(source_x))
    target_order = rng.permutation(len(target_x))
    source_x, source_y = source_x[source_order], source_y[source_order]
    target_x, target_y = target_x[target_order], target_y[target_order]
    return zscore_matlab(source_x), source_y, zscore_matlab(target_x), target_y


def encode_labels(source_y, target_y):
    """Encode labels for SOT while ensuring every target class is known."""
    classes, encoded_source = np.unique(source_y, return_inverse=True)
    if not np.isin(target_y, classes).all():
        unknown = np.setdiff1d(np.unique(target_y), classes)
        raise ValueError(f'Target contains classes absent from source: {unknown.tolist()}')
    return encoded_source, np.searchsorted(classes, target_y), classes


def source_only_1nn_accuracy(source_x, source_y, target_x, target_y):
    """Reference score for detecting adaptation that hurts this task."""
    return float(
        KNeighborsClassifier(n_neighbors=1).fit(source_x, source_y).score(
            target_x, target_y))


def run_jcsl(source_x, source_y, target_x, target_y, dim=30,
             matlab_executable='matlab'):
    """Run the repository's MATLAB JCSL implementation for one prepared task.

    The task runner leaves this call commented out below so SOT can be assessed
    independently.  It is ready to enable for a like-for-like comparison.
    """
    def matlab_quote(path):
        return path.as_posix().replace("'", "''")

    with tempfile.TemporaryDirectory(prefix='jcsl_task_') as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / 'input.mat'
        output_path = temp_path / 'output.mat'
        savemat(input_path, {
            'Sx': source_x,
            'Sy': np.asarray(source_y).reshape(-1, 1),
            'Tx': target_x,
            'Ty': np.asarray(target_y).reshape(-1, 1),
        })
        command = (
            f"addpath(genpath('{matlab_quote(REPO_ROOT)}')); "
            f"load('{matlab_quote(input_path)}'); "
            f"acc=JCSL(Sx,Sy,Tx,Ty,{int(dim)}); "
            f"save('{matlab_quote(output_path)}','acc');"
        )
        completed = subprocess.run(
            [matlab_executable, '-batch', command],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
        if completed.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                'MATLAB JCSL failed.\n'
                f'stdout:\n{completed.stdout}\n'
                f'stderr:\n{completed.stderr}')
        return float(np.asarray(loadmat(output_path)['acc']).squeeze())


def run_task(source_dataset, source_device, target_device, *,
             target_dataset=None, cross_dataset=False, seed=0, clusters=None,
             reg_e=0.5, reg_cl=1.0, reg_ce=3.0,
             sinkhorn_inner_iter=20000, sinkhorn_inner_tol=1e-7,
             show_baseline=False):
    """Run one demo-style domain-adaptation task and return SOT accuracy."""
    target_dataset = target_dataset or source_dataset
    if not cross_dataset and source_dataset == target_dataset and source_device == target_device:
        raise ValueError('Source and target devices must differ within one dataset.')

    source_x, source_y_raw, target_x, target_y_raw = prepare_domains(
        source_dataset, target_dataset, source_device, target_device,
        cross_dataset=cross_dataset, seed=seed)
    source_y, target_y, classes = encode_labels(source_y_raw, target_y_raw)
    clusters = int(clusters) if clusters is not None else len(classes)

    with tempfile.TemporaryDirectory(prefix='sot_task_') as temp_dir:
        root_dir = temp_dir + os.sep
        data_dir = Path(temp_dir) / 'data'
        model_dir = Path(temp_dir) / 'model'
        data_dir.mkdir()
        model_dir.mkdir()
        estimator = SOT(
            'ACT', root_dir, clusters, reg_e, reg_cl, reg_ce,
            random_state=seed,
            sinkhorn_inner_iter=sinkhorn_inner_iter,
            sinkhorn_inner_tol=sinkhorn_inner_tol,
        )
        _, accuracy = estimator.fit_predict(
            source_x, source_y, target_x, target_y,
            str(data_dir / 'source.json'), 'source',
            str(data_dir / 'target.json'), str(model_dir / 'target_gmm'),
            'target')

    task_name = f'{source_dataset} {source_device} -> {target_dataset} {target_device}'
    print(f'{task_name}: SOT accuracy {accuracy * 100:.2f}%')
    if show_baseline:
        baseline_accuracy = source_only_1nn_accuracy(
            source_x, source_y, target_x, target_y)
        print(f'{task_name}: source-only 1NN accuracy {baseline_accuracy * 100:.2f}%')

    # Enable this only to benchmark JCSL on this identical preprocessed pair.
    # jcsl_accuracy = run_jcsl(source_x, source_y_raw, target_x, target_y_raw, dim=30)
    # print(f'{task_name}: JCSL accuracy {jcsl_accuracy:.2f}%')

    return accuracy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true',
                        help='Run the eight within-dataset tasks from demo.m.')
    parser.add_argument('--dataset', choices=WITHIN_DATASETS,
                        help='Dataset for a single within-dataset task.')
    parser.add_argument('--source-dataset', choices=WITHIN_DATASETS,
                        help='Source dataset for a cross-dataset task.')
    parser.add_argument('--target-dataset', choices=WITHIN_DATASETS,
                        help='Target dataset for a cross-dataset task.')
    parser.add_argument('--source-device', type=int, help='1-based source device index.')
    parser.add_argument('--target-device', type=int, help='1-based target device index.')
    parser.add_argument('--cross', action='store_true',
                        help='Use cross_*.mat, as in demo_cross.m.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clusters', type=int,
                        help='Target GMM clusters (default: number of classes).')
    parser.add_argument('--reg-e', type=float, default=0.5,
                        help='Entropic OT regularisation lambda.')
    parser.add_argument('--reg-cl', type=float, default=1.0,
                        help='Class regularisation eta.')
    parser.add_argument('--reg-ce', type=float, default=3.0,
                        help='Partial-transport regularisation lambda1.')
    parser.add_argument('--sinkhorn-inner-iter', type=int, default=20000,
                        help='Maximum iterations for each inner Sinkhorn solve.')
    parser.add_argument('--sinkhorn-inner-tol', type=float, default=1e-7,
                        help='Inner Sinkhorn convergence tolerance.')
    parser.add_argument('--show-baseline', action='store_true',
                        help='Also print the source-only 1NN reference score.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    options = dict(
        seed=args.seed,
        clusters=args.clusters,
        reg_e=args.reg_e,
        reg_cl=args.reg_cl,
        reg_ce=args.reg_ce,
        sinkhorn_inner_iter=args.sinkhorn_inner_iter,
        sinkhorn_inner_tol=args.sinkhorn_inner_tol,
        show_baseline=args.show_baseline,
    )
    use_default_suite = (
        not args.cross
        and args.dataset is None
        and args.source_dataset is None
        and args.target_dataset is None
        and args.source_device is None
        and args.target_device is None
    )
    if args.all or use_default_suite:
        if args.cross:
            raise ValueError('--all is only defined for within-dataset demo.m tasks.')
        for dataset, source_device, target_device in WITHIN_TASKS:
            run_task(dataset, source_device, target_device, **options)
        return

    if args.cross:
        if not all((args.source_dataset, args.target_dataset,
                    args.source_device, args.target_device)):
            raise ValueError(
                '--cross requires --source-dataset, --target-dataset, '
                '--source-device, and --target-device.')
        run_task(
            args.source_dataset, args.source_device, args.target_device,
            target_dataset=args.target_dataset, cross_dataset=True, **options)
        return

    dataset = args.dataset or 'DSADS'
    source_device = args.source_device if args.source_device is not None else 2
    target_device = args.target_device if args.target_device is not None else 3
    run_task(dataset, source_device, target_device, **options)


if __name__ == '__main__':
    main()
