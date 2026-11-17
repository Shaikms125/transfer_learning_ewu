# coding=utf-8
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# SOT.py imports this private NumPy module, which was removed in NumPy 2.
try:
    from numpy.lib.function_base import rot90  # noqa: F401
except ImportError:
    function_base = types.ModuleType('numpy.lib.function_base')
    function_base.rot90 = np.rot90
    sys.modules['numpy.lib.function_base'] = function_base

from SOT import SOT


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    'DSADS': ('dsads.mat', 'data_dsads', -2),
    'OPP': ('opp_loco.mat', 'data_opp_loco', -3),
    'PAMAP': ('pamap.mat', 'data_pamap', -2),
}
TASKS = (
    ('DSADS', 2, 3),
    ('DSADS', 4, 5),
    ('OPP', 2, 4),
    ('OPP', 3, 5),
    ('DSADS', 2, 1),
    ('PAMAP', 1, 2),
    ('OPP', 3, 1),
    ('OPP', 2, 1),
)
SEED = 0


def load_domains(dataset, source_device, target_device):
    filename, variable, label_offset = DATASETS[dataset]
    data = loadmat(REPO_ROOT / 'data' / filename)[variable]
    source_start = (source_device - 1) * 81
    target_start = (target_device - 1) * 81
    label_column = data.shape[1] + label_offset
    source = data[:, list(range(source_start, source_start + 81)) + [label_column]]
    target = data[:, list(range(target_start, target_start + 81)) + [label_column]]
    return source, target


def zscore(x):
    scale = np.std(x, axis=0, ddof=1)
    scale[scale == 0] = 1
    return (x - np.mean(x, axis=0)) / scale


def run_task(dataset, source_device, target_device):
    np.random.seed(SEED)
    source, target = load_domains(dataset, source_device, target_device)
    source = source[np.random.permutation(len(source))]
    target = target[np.random.permutation(len(target))]
    Sx, Sy_raw = zscore(source[:, :-1]), source[:, -1]
    Tx, Ty_raw = zscore(target[:, :-1]), target[:, -1]

    # SOT expects source labels encoded as consecutive integers starting at zero.
    classes, Sy = np.unique(Sy_raw, return_inverse=True)
    if not np.isin(Ty_raw, classes).all():
        raise ValueError(f'{dataset} source and target have different activity labels.')
    Ty = np.searchsorted(classes, Ty_raw)

    with tempfile.TemporaryDirectory(prefix='sot_task_') as temp_dir:
        root_dir = temp_dir + os.sep
        data_dir = Path(temp_dir) / 'data'
        model_dir = Path(temp_dir) / 'model'
        data_dir.mkdir()
        model_dir.mkdir()
        tsot = SOT('ACT', root_dir, 19, 0.5, 1, 3)
        _, accuracy = tsot.fit_predict(
            Sx, Sy, Tx, Ty,
            str(data_dir / 'source.json'), 'D',
            str(data_dir / 'target.json'), str(model_dir / 'target_model'), 'H')

    print(f'{dataset} {source_device} -> {target_device}: SOT accuracy {accuracy * 100:.2f}%')


if __name__ == '__main__':
    for task in TASKS:
        run_task(*task)
