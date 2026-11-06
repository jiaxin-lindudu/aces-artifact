"""Centralized fuzzy c-means ceiling on the pooled FIT block."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

DEFAULT_CFG = {'fuzziness': 2.0, 'max_iter': 300, 'tol': 1e-5}


def _membership(squared, fuzziness):
    """The one membership definition; imported from the federated model on purpose.

    A centralized FCM ceiling is only a bound for the federated FCM if the two
    share the definition of U -- including the exact-zero-distance case and the
    m -> 1 limit. Duplicating the formula here would let the two drift apart and
    silently invalidate every comparison drawn from them.
    """
    from federated.ours.aces_clustering import (
        fuzzy_membership_from_squared_distances)
    return fuzzy_membership_from_squared_distances(squared, fuzziness)


def run(X_train, X_test, K, seed, cfg=None):
    """Textbook fuzzy c-means; ``cfg`` keys: fuzziness, max_iter, tol.

    The centre update weights each point by u**m, NOT by u -- dropping the
    exponent is the textbook FCM error and collapses every centre onto the plain
    data mean. ``crispness`` (mean max membership, uniform would be 1/K) is
    reported so a collapsed run can be told apart from a good one: at m >= 1.5
    in a 64-d space the memberships sit on the uniform fixed point and the ACC
    that comes out is not a ceiling at all.
    """
    from sklearn.cluster import kmeans_plusplus

    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    fuzziness = float(cfg['fuzziness'])
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    K = int(K)

    centers, _ = kmeans_plusplus(X_train, n_clusters=K, random_state=int(seed))
    previous = None
    n_iter = 0
    for n_iter in range(1, int(cfg['max_iter']) + 1):
        squared = cdist(X_train, centers, metric='sqeuclidean')
        membership = _membership(squared, fuzziness)
        weights = membership ** fuzziness
        denominator = weights.sum(axis=0)[:, None]
        new_centers = (weights.T @ X_train) / np.maximum(denominator, 1e-12)
        # `previous` is an ndarray: `previous or 0.0` would trigger an elementwise
        # truth test, so the reference norm has to be taken behind an explicit None check.
        reference = 0.0 if previous is None else float(np.linalg.norm(previous))
        shift = (np.inf if previous is None
                 else float(np.linalg.norm(new_centers - previous)))
        if shift <= float(cfg['tol']) * max(1.0, reference):
            centers = new_centers
            break
        previous = centers = new_centers

    squared = cdist(X_train, centers, metric='sqeuclidean')
    membership = _membership(squared, fuzziness)
    from experiments.baselines_adapter import nearest_center_labels
    return {
        'train_labels': np.asarray(membership.argmax(axis=1), dtype=int),
        'test_labels': np.asarray(nearest_center_labels(X_test, centers), dtype=int),
        'test_basis': 'centers',
        'extra': {'crispness': float(membership.max(axis=1).mean()),
                  'uniform_crispness': 1.0 / K, 'n_iter': int(n_iter),
                  'hit_cap': bool(n_iter >= int(cfg['max_iter']))},
    }
