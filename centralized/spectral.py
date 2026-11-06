"""Centralized spectral clustering ceiling, with an honest out-of-sample rule.

Implemented here rather than wrapped around sklearn's ``SpectralClustering``
for one reason: sklearn only fits, it cannot embed new points. A ceiling that
cannot score the TEST block would force the comparison back onto the training
block, where every federated method looks worse than it is.

The pipeline is Ng-Jordan-Weiss on a Gaussian affinity, then k-means on the
embedding, and the TEST block reaches the same embedding through the Nyström
extension -- the same mechanism `ours` uses, so the two are comparable.

    affinity      W_ij = exp(-gamma ||x_i - x_j||^2)         (FIT x FIT)
    normalised    Wn   = D^-1/2 W D^-1/2
    embedding     eigenvectors of Wn for the K largest eigenvalues
    out-of-sample embed(x) = (k_x^T normalized) U / lambda   (Nyström)

A k-nearest-neighbour affinity is deliberately not offered: that graph cannot be
built without the TEST points, so any test score computed on it would be
transductive and would overstate the ceiling.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

DEFAULT_CFG = {'gamma_mode': 'one_over_d', 'normalize_rows': True,
               'assign': 'kmeans', 'n_init': 10, 'max_iter': 300}


def gamma_value(X, mode, rng_seed=0):
    """Gaussian bandwidth: 1/d (sklearn's rbf default) or the median heuristic."""
    if mode == 'one_over_d':
        return 1.0 / max(X.shape[1], 1)
    if mode == 'median':
        # median pairwise SQUARED distance, from a subsample so this stays cheap
        rng = np.random.default_rng(rng_seed)
        n = len(X)
        sample = X[rng.choice(n, size=min(n, 500), replace=False)]
        squared = cdist(sample, sample, metric='sqeuclidean')
        median_sq = float(np.median(squared[np.triu_indices(len(sample), 1)]))
        return 1.0 / max(2.0 * median_sq, 1e-12)
    raise ValueError('gamma_mode must be one_over_d or median, got %r' % mode)


def _affinity(X, Y, gamma):
    from sklearn.metrics.pairwise import rbf_kernel
    return rbf_kernel(X, Y, gamma=gamma)


def run(X_train, X_test, K, seed, cfg=None):
    """``cfg`` keys: gamma_mode, normalize_rows, assign, n_init."""
    from sklearn.cluster import KMeans

    from experiments.partition_checks import require_spectral_memory
    # Two dense float64 matrices cover affinity layout conversion / LAPACK
    # workspace. Normalization reuses the buffer; no full eigenvector matrix
    # or TEST-by-FIT matrix is retained alongside it.
    require_spectral_memory(len(X_train), dense_arrays=2)
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    K, seed = int(K), int(seed)

    gamma = gamma_value(X_train, cfg['gamma_mode'])
    weight = np.asfortranarray(_affinity(X_train, X_train, gamma))
    degree = weight.sum(axis=1)
    degree_isqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    weight *= degree_isqrt[:, None]
    weight *= degree_isqrt[None, :]

    # Same dense symmetric eigenproblem, computing only the required top K.
    # Fortran order lets LAPACK overwrite its input rather than copy n*n.
    eigenvalues, eigenvectors = eigh(
        weight, subset_by_index=(len(weight) - K, len(weight) - 1),
        driver='evr', overwrite_a=True, check_finite=True)
    del weight
    lambdas = np.maximum(eigenvalues[::-1], 1e-12)
    basis = eigenvectors[:, ::-1]

    def _row_normalize(matrix):
        if not cfg['normalize_rows']:
            return matrix
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norm, 1e-12)

    embed_train = _row_normalize(basis)

    if cfg['assign'] != 'kmeans':
        raise ValueError("only assign='kmeans' is supported: discretize has no "
                         "out-of-sample form and would make the test score "
                         "transductive")
    model = KMeans(n_clusters=K, n_init=int(cfg['n_init']),
                   max_iter=int(cfg['max_iter']), random_state=seed)
    train_labels = model.fit_predict(embed_train)
    centers_embed = model.cluster_centers_
    kmeans_iters = int(model.n_iter_)

    # Nyström: the TEST point is projected through its affinity to the FIT points
    # and rescaled by the eigenvalues that produced the FIT embedding.
    embed_test = np.empty((len(X_test), K), dtype=float)
    for start in range(0, len(X_test), 512):
        stop = min(start + 512, len(X_test))
        cross = _affinity(X_test[start:stop], X_train, gamma)
        test_degree = cross.sum(axis=1)
        cross *= (1.0 / np.sqrt(np.maximum(test_degree, 1e-12)))[:, None]
        cross *= degree_isqrt[None, :]
        embed_test[start:stop] = _row_normalize((cross @ basis) / lambdas[None, :])
        del cross
    test_labels = np.argmin(cdist(embed_test, centers_embed), axis=1)

    return {
        'train_labels': np.asarray(train_labels, dtype=int),
        'test_labels': np.asarray(test_labels, dtype=int),
        'test_basis': 'nystrom',
        'extra': {'gamma': float(gamma), 'gamma_mode': cfg['gamma_mode'],
                  'normalize_rows': bool(cfg['normalize_rows']),
                  'top_eigenvalues': [round(float(v), 4) for v in lambdas],
                  # only the k-means on the embedding iterates; the eigendecomposition
                  # is exact and has no budget of its own
                  'n_iter': kmeans_iters,
                  'hit_cap': bool(kmeans_iters >= int(cfg['max_iter']))},
    }
