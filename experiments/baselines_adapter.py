"""One uniform entry point for the existing ffcm / fkm / nnfc implementations.

The three baselines live in ``federated/{ffcm,fkm,nnfc}`` and return bare
centers (or a 4-tuple for nnfc). This module only adapts their *interface*: it
never re-implements or re-tunes their algorithm, so the paired comparison
against ``ours`` rests on the same FIT rows, the same K and the same round cap.

Out-of-sample behaviour is reported as ``not_applicable`` (i.e. N/A) on purpose. FKM and FFCM hard-assign
a point by nearest center only as a post-hoc convenience, and NNFC is one-shot
with no query extension at all -- none of them defines a Nyström query embedding,
so quoting a ``test_proxy_*`` number for them would silently compare different
things.
"""
from __future__ import annotations

import numpy as np

BASELINE_METHODS = ('fkm', 'ffcm', 'nnfc')

# The literal string 'N/A' would be re-read as NaN by pandas (it is in the
# default na_values list), so the CSV cell says 'not_applicable' and stays
# distinguishable from a genuinely missing value.
OOS_NOT_APPLICABLE = 'not_applicable'

NOTES = {
    'fkm': 'FKM: federated k-means, per-round local update then server averaging; '
           'no embedding space, OOS assignment not defined',
    'ffcm': 'FFCM: federated fuzzy c-means, per-round local memberships then server '
            'aggregation; no embedding space, OOS assignment not defined',
    'nnfc': 'NNFC: one-shot federated nearest-neighbour clustering (local anchors + '
            'SNN merge); single round, no embedding space, OOS assignment not defined',
}


def nearest_center_labels(X, centers):
    """Hard assignment used only to score a baseline's returned centers."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != X.shape[1]:
        raise ValueError('Baseline centers have shape %s, FIT rows have %s'
                         % (centers.shape, X.shape))
    if not np.isfinite(centers).all():
        raise ValueError('Baseline returned non-finite centers; refusing to score them')
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def run_baseline(method, X_list, X_train, y_train, X_test, y_test, num_clusters,
                 num_rounds=10, random_state=42, fuzziness=2.0, k_neighbors=2,
                 dp_sigma=0.0):
    """Run one baseline and return centers plus FIT/TEST hard labels.

    Returns
    -------
    dict with keys ``centers``, ``train_labels``, ``test_labels``, ``rounds``,
    ``note`` and ``oos_status``.
    """
    if method not in BASELINE_METHODS:
        raise ValueError('Unknown baseline %r; expected one of %s'
                         % (method, list(BASELINE_METHODS)))
    # Every baseline reseeds the global numpy stream internally; isolate it so a
    # baseline run cannot leak state into the next run of another method.
    np.random.seed(random_state)
    rounds = num_rounds
    if method == 'fkm':
        from federated.fkm.fkm_trainer import fkm_clustering
        # center_init_method is data-adaptive on purpose: the historical
        # 'random' path draws centers from N(0,1) (see initialize_centers, which
        # hardcodes data_normalized=True) and collapses onto 2-3 clusters on any
        # non-standardized block (e.g. the /16 digits). federated_kmeans++ samples
        # centers from the clients' own data, so it has no scale dependence.
        centers = fkm_clustering(
            X_train, y_train, X_test, y_test, X_list, num_clusters,
            center_init_method='federated_kmeans++',
            n_rounds=num_rounds, dp_sigma=dp_sigma, random_state=random_state)
    elif method == 'ffcm':
        from federated.ffcm.ffcm_trainer import ffcm_clustering
        # Same scale-dependence fix; kmeans_plusplus uses the pooled FIT rows and
        # matches ours' benchmark initialization.
        centers = ffcm_clustering(
            X_train, y_train, X_test, y_test, X_list, num_clusters,
            center_init_method='kmeans_plusplus',
            n_rounds=num_rounds, m=fuzziness, dp_sigma=dp_sigma,
            random_state=random_state)
    else:
        from federated.nnfc.nnfc_trainer import federated_nnfc_clustering
        outcome = federated_nnfc_clustering(
            X_train, y_train, X_test, y_test, X_list, num_clusters,
            dp_sigma=dp_sigma, k_neighbors=k_neighbors, random_state=random_state)
        # Its docstring promises a 4-tuple but the implementation returns
        # (global_centers, history) on success and a 4-tuple on the empty edge
        # cases; take the first element either way instead of asserting a shape.
        centers = outcome[0] if isinstance(outcome, tuple) else outcome
        rounds = 1
    centers = np.asarray(centers, dtype=float)
    return {
        'centers': centers,
        'train_labels': nearest_center_labels(X_train, centers),
        'test_labels': nearest_center_labels(X_test, centers),
        'rounds': rounds,
        'note': NOTES[method],
        'oos_status': OOS_NOT_APPLICABLE,
    }
