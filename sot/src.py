#coding=utf-8
import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
from SOT import SOT

file_name = 'PAMAP'      # 'OPP' | 'DSADS' | 'PAMAP'
i, j = 1, 2            # 1-based body positions, same as demo.m (OPP: RLA -> LLA)

MAT = {'OPP': ('data/opp_loco.mat', 'data_opp_loco'),
       'DSADS': ('data/dsads.mat', 'data_dsads'),
       'PAMAP': ('data/pamap.mat', 'data_pamap')}
# MATLAB label column: OPP = end-2, DSADS/PAMAP = end-1  ->  Python negative index
LABEL_COL = {'OPP': -3, 'DSADS': -2, 'PAMAP': -2}


def col_set(idx):
    """MATLAB col_sets{idx} = [(idx-1)*81+1 : idx*81]  ->  0-based slice."""
    return slice((idx - 1) * 81, idx * 81)


def load_data(file_name, src_index, tar_index):
    path, var = MAT[file_name]
    data = loadmat(path)[var]                 # if this errors with a v7.3 file, see note below
    lab = data[:, LABEL_COL[file_name]]
    src = np.column_stack([data[:, col_set(src_index)], lab])
    tar = np.column_stack([data[:, col_set(tar_index)], lab])
    return src, tar


def zscore_matlab(x):
    # MATLAB zscore: ddof=1, and constant columns become 0 instead of NaN
    return np.nan_to_num(zscore(x, axis=0, ddof=1))


if __name__ == '__main__':
    src_domain, tar_domain = load_data(file_name, i, j)

    # random shuffle (randperm)
    src_domain = src_domain[np.random.permutation(src_domain.shape[0])]
    tar_domain = tar_domain[np.random.permutation(tar_domain.shape[0])]

    Sx, Sy = src_domain[:, :-1], src_domain[:, -1].astype(int)
    Tx, Ty = tar_domain[:, :-1], tar_domain[:, -1].astype(int)

    Sx = zscore_matlab(Sx)
    Tx = zscore_matlab(Tx)

    # --- NEW: make labels 0,1,2,... and count the classes ---
    print('src labels:', np.unique(Sy, return_counts=True))
    print('tar labels:', np.unique(Ty, return_counts=True))

    classes = np.unique(np.concatenate([Sy, Ty]))
    remap = {c: k for k, c in enumerate(classes)}
    Sy = np.array([remap[c] for c in Sy])
    Ty = np.array([remap[c] for c in Ty])
    n_classes = len(classes)
    print('number of classes:', n_classes)
    # --------------------------------------------------------

    sname, tname = f'{file_name}_{i}', f'{file_name}_{j}'
    os.makedirs('./clustertemp/data', exist_ok=True)
    os.makedirs('./clustertemp/model', exist_ok=True)
    spath = f'./data/test_{file_name}_diag_SG.json'
    tpath = f'./clustertemp/data/test_{file_name}_{sname}_{tname}_diag_TG.json'
    tmodelpath = f'./clustertemp/model/test_{file_name}_{tname}'

    tsot = SOT('ACT', './clustertemp/', n_classes, 0.5, 1, 3)   # <-- 19 became n_classes
    pred, acc = tsot.fit_predict(Sx, Sy, Tx, Ty, spath, sname, tpath, tmodelpath, tname)
    print('Acc: %.2f' % acc)