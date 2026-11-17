# coding: utf-8
"""Substructural Optimal Transport (SOT).

This preserves the published algorithmic structure while making the numerical
choices explicit and reproducible.  The inner Sinkhorn solver is given enough
iterations for the GMM-to-GMM transport problem solved by SOT.
"""

from collections import Counter
import json
import os

import joblib
import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn import mixture
from sklearn.neighbors import KNeighborsClassifier


_EPS = np.finfo(np.float64).eps


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _as_probability_weights(weights, name):
    """Return finite, non-negative weights with unit total mass."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError(f'{name} must not be empty.')
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f'{name} must be finite and non-negative.')
    total = weights.sum()
    if total <= 0:
        raise ValueError(f'{name} must have positive total mass.')
    return weights / total


def _normalised_sqeuclidean_cost(source, target):
    """Build the ground cost used by SOT without division by zero."""
    cost = cdist(source, target, metric='sqeuclidean')
    if not np.isfinite(cost).all():
        raise ValueError('The SOT ground-cost matrix contains NaN or infinity.')
    positive_costs = cost[cost > 0]
    scale = np.median(positive_costs) if positive_costs.size else 1.0
    return cost / max(float(scale), _EPS)


def _validate_transport(plan, source_weights, target_weights, name):
    """Fail rather than replace an unsuccessful OT plan with a baseline."""
    plan = np.asarray(plan, dtype=np.float64)
    if not np.isfinite(plan).all() or np.any(plan < -1e-12):
        raise RuntimeError(f'{name} returned an invalid transport plan.')
    if not np.isclose(plan.sum(), 1.0, rtol=1e-5, atol=1e-7):
        raise RuntimeError(f'{name} returned transport mass {plan.sum():.6g}, expected 1.')

    # POT's iterative solver is approximate.  Checking the marginals catches a
    # non-converged answer instead of allowing it to produce a misleading score.
    row_error = np.max(np.abs(plan.sum(axis=1) - source_weights))
    col_error = np.max(np.abs(plan.sum(axis=0) - target_weights))
    if max(row_error, col_error) > 2e-4:
        raise RuntimeError(
            f'{name} did not satisfy its marginals '
            f'(row error={row_error:.3g}, column error={col_error:.3g}).')
    return plan


def gmm_target(X, components, rootdir, filepath, modelpath, targetname,
               covtype='diag', random_state=None, n_init=5):
    """Fit and persist the target GMM used to create target substructures."""
    X = np.asarray(X, dtype=np.float64)
    if components < 1 or components > len(X):
        raise ValueError(
            f'Target GMM needs 1..{len(X)} components; got {components}.')
    os.makedirs(os.path.join(rootdir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(rootdir, 'model'), exist_ok=True)

    gmm = mixture.GaussianMixture(
        n_components=components,
        covariance_type=covtype,
        random_state=random_state,
        n_init=n_init,
        reg_covar=1e-6,
    ).fit(X)
    data = {
        'xntmu': gmm.weights_,
        'xnt1': gmm.means_,
        'xnt2': np.sqrt(gmm.covariances_),
    }
    with open(filepath, 'w', encoding='utf-8') as handle:
        json.dump({targetname: data}, handle, cls=MyEncoder)
    joblib.dump(gmm, modelpath)


def gmm_source_class(X, weight, component_candidates, covtype='diag',
                     random_state=None, n_init=5):
    """Select a class-conditional source GMM by BIC."""
    X = np.asarray(X, dtype=np.float64)
    candidates = np.asarray(component_candidates, dtype=int).reshape(-1)
    candidates = candidates[(candidates >= 1) & (candidates <= len(X))]
    if candidates.size == 0:
        raise ValueError('No valid source-GMM component candidates were supplied.')

    best_gmm = None
    best_bic = np.inf
    for components in candidates:
        candidate = mixture.GaussianMixture(
            n_components=int(components),
            covariance_type=covtype,
            random_state=random_state,
            n_init=n_init,
            reg_covar=1e-6,
        ).fit(X)
        candidate_bic = candidate.bic(X)
        if candidate_bic >= best_bic:
            # Intended authors' stopping rule, without indexing beyond the
            # final candidate or refitting a different model after selection.
            break
        best_gmm = candidate
        best_bic = candidate_bic

    means = best_gmm.means_
    scales = np.sqrt(best_gmm.covariances_)
    masses = best_gmm.weights_ * weight
    return masses, means, scales, best_gmm


def gmm_source(xs, ys, filepath, sourcename, component_candidates=None,
               covtype='diag', random_state=None, n_init=5):
    """Fit one BIC-selected GMM for every encoded source class."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.int64).reshape(-1)
    classes = np.unique(ys)
    expected = np.arange(classes.size)
    if not np.array_equal(classes, expected):
        raise ValueError('SOT source labels must be encoded as 0, 1, ..., K-1.')

    class_counts = Counter(ys)
    class_mass = len(ys)
    class_means, class_scales, class_weights, class_labels = [], [], [], []
    for class_index in expected:
        samples = xs[ys == class_index]
        candidates = component_candidates
        if candidates is None:
            candidates = np.arange(1, len(samples) + 1)
        masses, means, scales, _ = gmm_source_class(
            samples,
            class_counts[class_index] / class_mass,
            candidates,
            covtype=covtype,
            random_state=random_state,
            n_init=n_init,
        )
        class_weights.append(masses)
        class_means.append(means)
        class_scales.append(scales)
        class_labels.append(np.full(len(means), class_index, dtype=np.int64))

    data = {
        'xmu': np.concatenate(class_weights),
        'xn1': np.vstack(class_means),
        'xn2': np.vstack(class_scales),
        'yns': np.concatenate(class_labels),
    }
    with open(filepath, 'w', encoding='utf-8') as handle:
        json.dump({sourcename: data}, handle, cls=MyEncoder)


def entropic_partial_wasserstein(a, b, M, reg, m=1):
    """The partial-transport initialization used by the published SOT code."""
    if reg <= 0:
        raise ValueError('reg_ce must be strictly positive.')
    M = np.asarray(M, dtype=np.float64)
    source_weights = _as_probability_weights(a, 'partial source weights')
    target_weights = _as_probability_weights(b, 'partial target weights')
    if M.shape != (len(source_weights), len(target_weights)):
        raise ValueError('The partial-transport cost matrix has the wrong shape.')

    transported_mass = min(float(m), source_weights.sum(), target_weights.sum())
    kernel = np.exp(-M / reg)
    kernel_sum = kernel.sum()
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise RuntimeError('Partial transport underflowed; increase reg_ce.')
    kernel *= transported_mass / kernel_sum
    column_sum = kernel.sum(axis=0)
    if np.any(column_sum <= _EPS):
        raise RuntimeError('Partial transport has an empty target substructure.')
    return kernel * (target_weights / column_sum)[None, :]


class SOT:
    def __init__(self, taskname='ACT', root_dir='./clustertemp/', d=200,
                 reg_e=0.5, reg_cl=0.1, reg_ce=0.1, rule='median',
                 random_state=0, gmm_n_init=5, sinkhorn_outer_iter=10,
                 sinkhorn_inner_iter=20000, sinkhorn_inner_tol=1e-7):
        if d < 1:
            raise ValueError('d must be positive.')
        if reg_e <= 0 or reg_cl <= 0 or reg_ce <= 0:
            raise ValueError('SOT regularization parameters must be positive.')
        self.taskname = taskname
        self.root_dir = root_dir
        self.d = int(d)
        self.reg_e = float(reg_e)
        self.reg_cl = float(reg_cl)
        self.reg_ce = float(reg_ce)
        self.rule = rule
        self.random_state = random_state
        self.gmm_n_init = int(gmm_n_init)
        self.sinkhorn_outer_iter = int(sinkhorn_outer_iter)
        self.sinkhorn_inner_iter = int(sinkhorn_inner_iter)
        self.sinkhorn_inner_tol = float(sinkhorn_inner_tol)

    def get_target(self, filepath, modelpath, targetname):
        with open(filepath, encoding='utf-8') as handle:
            self.diag_t = json.load(handle)[targetname]
        self.diag_g = joblib.load(modelpath)

    def get_source(self, filepath, sourcename):
        with open(filepath, encoding='utf-8') as handle:
            self.diag_s = json.load(handle)[sourcename]

    def partot_DA(self, Sx, Sy, target_substructures, target_samples):
        """Adapt GMM substructures and classify the raw target samples."""
        target_substructures = np.asarray(target_substructures, dtype=np.float64)
        M = _normalised_sqeuclidean_cost(Sx, target_substructures)
        # Algorithm 1 in the SOT paper sets the target substructure weights to
        # 1 / kt.  The target GMM mixture masses are stored for completeness,
        # but are not the OT target marginal in the published method.
        target_weights = (
            np.ones(len(target_substructures), dtype=np.float64)
            / len(target_substructures)
        )
        initial = entropic_partial_wasserstein(
            np.ones(len(Sx), dtype=np.float64) / len(Sx),
            target_weights,
            M,
            self.reg_ce,
            m=1,
        )

        source_weights = _as_probability_weights(initial.sum(axis=1), 'source OT weights')
        target_weights = _as_probability_weights(initial.sum(axis=0), 'target OT weights')
        transport = ot.da.sinkhorn_lpl1_mm(
            source_weights,
            Sy,
            target_weights,
            M,
            self.reg_e,
            self.reg_cl,
            numItermax=self.sinkhorn_outer_iter,
            numInnerItermax=self.sinkhorn_inner_iter,
            stopInnerThr=self.sinkhorn_inner_tol,
        )
        transport = _validate_transport(
            transport, source_weights, target_weights, 'sinkhorn_lpl1_mm')

        row_mass = transport.sum(axis=1)
        transported_source = (transport @ target_substructures) / row_mass[:, None]
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(transported_source, Sy)
        target_clusters = self.diag_g.predict(target_samples)
        return classifier.predict(target_substructures)[target_clusters]

    def fit_predict(self, Sx, Sy, Tx, Ty, sfilepath, sourcename, tfilepath,
                    tmodelpath, targetname):
        Sx = np.asarray(Sx, dtype=np.float64)
        Sy = np.asarray(Sy, dtype=np.int64).reshape(-1)
        Tx = np.asarray(Tx, dtype=np.float64)
        Ty = np.asarray(Ty, dtype=np.int64).reshape(-1)
        if Sx.ndim != 2 or Tx.ndim != 2 or Sx.shape[1] != Tx.shape[1]:
            raise ValueError('Source and target features must be 2-D with the same width.')
        if len(Sx) != len(Sy) or len(Tx) != len(Ty):
            raise ValueError('Feature and label counts must match.')

        gmm_source(
            Sx, Sy, sfilepath, sourcename,
            random_state=self.random_state,
            n_init=self.gmm_n_init,
        )
        self.get_source(sfilepath, sourcename)
        gmm_target(
            Tx, self.d, self.root_dir, tfilepath, tmodelpath, targetname,
            random_state=self.random_state,
            n_init=self.gmm_n_init,
        )
        self.get_target(tfilepath, tmodelpath, targetname)

        source = self.diag_s
        target = self.diag_t
        predictions = self.partot_DA(
            np.asarray(source['xn1']),
            np.asarray(source['yns'], dtype=np.int64),
            np.asarray(target['xnt1']),
            Tx,
        )
        return predictions, float(np.mean(Ty == predictions))
