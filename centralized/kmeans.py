"""Centralized K-Means ceiling on the pooled FIT block."""
from __future__ import annotations

import numpy as np

DEFAULT_CFG = {'init': 'k-means++', 'n_init': 10, 'max_iter': 300}


def run(X_train, X_test, K, seed, cfg=None):
    """Fit on the pooled FIT block, then label both blocks.

    ``cfg`` keys: ``init`` ('k-means++' | 'random'), ``n_init`` (restarts).
    Both are real knobs here: ``n_init=1`` with ``init='random'`` measures how
    much of the score is owed to the seeding rather than to the data.
    """
    from sklearn.cluster import KMeans

    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    model = KMeans(n_clusters=int(K), init=cfg['init'], n_init=int(cfg['n_init']),
                   max_iter=int(cfg['max_iter']), random_state=int(seed))
    train_labels = model.fit_predict(X_train)
    test_labels = model.predict(X_test)
    return {
        'train_labels': np.asarray(train_labels, dtype=int),
        'test_labels': np.asarray(test_labels, dtype=int),
        'test_basis': 'centers',
        'extra': {'inertia': float(model.inertia_),
                  # n_iter_ counts the iterations the BEST restart actually used;
                  # hitting max_iter on it means the cap, not the data, stopped it.
                  'n_iter': int(model.n_iter_),
                  'hit_cap': bool(model.n_iter_ >= int(cfg['max_iter']))},
    }
