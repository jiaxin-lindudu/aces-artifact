"""One-shot k-FED with explicit paper/released-code cleanup conventions.

References: Dennis et al., ICML 2021, Algorithms 1--2; the public
``metastableB/kfed`` implementation was inspected, not copied (no license).
``author`` means its projected-mean initialization/converged cleanup recipe,
not bitwise reproduction of its randomized SVD or global NumPy RNG stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, kmeans_plusplus


@dataclass
class LocalKFedResult:
    centers: np.ndarray
    labels: np.ndarray
    iterations: int = 0
    converged: bool = True
    empty_stable_sets: int = 0


class KFedClient:
    """Algorithm 1: spectral initialization followed by Lloyd on raw data."""

    def __init__(self, client_id: int, X: np.ndarray, local_clusters: int,
                 random_state: int = 42, max_lloyd_iter: int = 300,
                 implementation: str = "author"):
        self.client_id = int(client_id)
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2 or len(self.X) == 0 or not np.isfinite(self.X).all():
            raise ValueError("X must be a non-empty 2D array")
        if not isinstance(local_clusters, (int, np.integer)) or local_clusters < 1:
            raise ValueError("local_clusters must be positive")
        self.requested_clusters = int(local_clusters)
        self.local_clusters = min(int(local_clusters), len(np.unique(self.X, axis=0)))
        if (implementation not in ("author", "paper")
                or not isinstance(max_lloyd_iter, (int, np.integer))
                or max_lloyd_iter < 1):
            raise ValueError("invalid implementation or Lloyd iteration cap")
        self.implementation = implementation
        self.random_state = int(random_state)
        self.max_lloyd_iter = int(max_lloyd_iter)
        self.result: LocalKFedResult | None = None

    def fit(self) -> LocalKFedResult:
        k = self.local_clusters
        # The released implementation handles n <= k as a trivial partition.
        # This also covers duplicated records after the documented distinct-row
        # cap, without asking k-means++ for more distinct points than exist.
        unique, inverse = np.unique(self.X, axis=0, return_inverse=True)
        if k == len(unique):
            self.result = LocalKFedResult(unique, inverse, 0, True, 0)
            return self.result

        # ====================================================================
        # Local spectral initialization (Algorithm 1, steps 2--3)
        # ====================================================================
        # Algorithm 1 projects A itself onto its top right-singular subspace;
        # it does not center the matrix first.
        _, _, vt = np.linalg.svd(self.X, full_matrices=False)
        rank = min(k, vt.shape[0])
        projected = self.X @ vt[:rank].T

        # Author's initializer is standard D-squared k-means++, NOT an
        # already-converged projected KMeans model or ten restarts.
        projected_centers, _ = kmeans_plusplus(projected, n_clusters=k,
            random_state=self.random_state, n_local_trials=1)
        projected_dist = cdist(projected, projected_centers)

        # Stable sets S_r from Algorithm 1. If a stable set is empty, use the
        # full Voronoi cell, which is the natural finite-sample safeguard.
        theta = np.empty((k, self.X.shape[1]), dtype=float)
        nearest = np.argmin(projected_dist, axis=1)
        empty_stable_sets = 0
        for r in range(k):
            others = np.delete(projected_dist, r, axis=1)
            stable = (np.ones(len(self.X), dtype=bool) if k == 1 else
                      projected_dist[:, r] <= np.min(others, axis=1) / 3.0)
            indices = np.flatnonzero(stable)
            if not len(indices):
                empty_stable_sets += 1
                indices = np.flatnonzero(nearest == r)
            if not len(indices):
                indices = np.array([int(np.argmin(projected_dist[:, r]))])
            theta[r] = (projected[indices].mean(axis=0) @ vt[:rank]
                        if self.implementation == "author" else self.X[indices].mean(axis=0))

        # ====================================================================
        # Local Lloyd updates on original features (Algorithm 1, step 4)
        # ====================================================================
        if self.implementation == "author":
            model = KMeans(n_clusters=k, init=theta, n_init=1,
                random_state=self.random_state, max_iter=self.max_lloyd_iter, tol=1e-4).fit(self.X)
            labels = model.predict(self.X)
            # Audit stationarity instead of assuming a capped fit converged.
            updated = model.cluster_centers_.copy()
            for r in range(k):
                if np.any(labels == r):
                    updated[r] = self.X[labels == r].mean(axis=0)
            converged = bool(np.allclose(updated, model.cluster_centers_, rtol=0., atol=1e-8))
            self.result = LocalKFedResult(model.cluster_centers_, labels,
                int(model.n_iter_), converged, empty_stable_sets)
            return self.result

        labels = np.zeros(len(self.X), dtype=int)
        converged = False
        for iteration in range(1, self.max_lloyd_iter + 1):
            labels = np.argmin(cdist(self.X, theta), axis=1)
            updated = theta.copy()
            for r in range(k):
                members = self.X[labels == r]
                if len(members):
                    updated[r] = members.mean(axis=0)
            if np.allclose(updated, theta, rtol=0.0, atol=1e-10):
                theta = updated
                converged = True
                break
            theta = updated

        labels = np.argmin(cdist(self.X, theta), axis=1)
        self.result = LocalKFedResult(theta, labels, iteration, converged, empty_stable_sets)
        return self.result


class KFedServer:
    """Farthest-first; paper one-step or author's converged server Lloyd."""

    def __init__(self, num_clusters: int, random_state: int = 42,
                 implementation: str = "author", max_lloyd_iter: int = 300):
        if not isinstance(num_clusters, (int, np.integer)) or num_clusters < 1:
            raise ValueError("num_clusters must be a positive integer")
        if (implementation not in ("author", "paper")
                or not isinstance(max_lloyd_iter, (int, np.integer))
                or max_lloyd_iter < 1):
            raise ValueError("invalid implementation or Lloyd iteration cap")
        self.num_clusters = int(num_clusters)
        self.implementation, self.max_lloyd_iter = implementation, max_lloyd_iter
        self.iterations, self.converged = 0, False
        self.random_state = int(random_state)
        self.global_centers: np.ndarray | None = None
        self.local_center_labels: List[np.ndarray] = []

    def aggregate(self, local_results: Sequence[LocalKFedResult]) -> np.ndarray:
        if not local_results:
            raise ValueError("At least one local result is required")
        local_centers = [np.asarray(r.centers, dtype=float) for r in local_results]
        if any(c.ndim != 2 or not len(c) or c.shape[1] != local_centers[0].shape[1]
               or not np.isfinite(c).all() for c in local_centers):
            raise ValueError("local centers must be nonempty finite matrices of equal width")
        all_centers = np.vstack(local_centers)
        k = self.num_clusters
        if (all_centers.ndim != 2 or not np.isfinite(all_centers).all()
                or len(np.unique(all_centers, axis=0)) < k
                or len(local_centers[0]) > k):
            raise ValueError("insufficient distinct prototypes or local K exceeds global K")

        # ====================================================================
        # Farthest-first server initialization (Algorithm 2, steps 2--6)
        # ====================================================================
        # Paper: pick any device. We deterministically pick the first one.
        seeds = [c.copy() for c in local_centers[0][:k]]
        while len(seeds) < k:
            dist = cdist(all_centers, np.asarray(seeds)).min(axis=1)
            seeds.append(all_centers[int(np.argmax(dist))].copy())
        seeds = np.asarray(seeds[:k])

        # Paper pseudocode and released implementation differ here. Do not
        # label a converged sklearn cleanup as exactly one Lloyd iteration.
        center_labels = np.argmin(cdist(all_centers, seeds), axis=1)
        global_centers = seeds.copy()
        for r in range(k):
            members = all_centers[center_labels == r]
            if len(members):
                global_centers[r] = members.mean(axis=0)

        self.iterations = 1
        self.converged = False
        if self.implementation == "author":
            model = KMeans(n_clusters=k, init=seeds, n_init=1,
                random_state=self.random_state, max_iter=self.max_lloyd_iter,
                tol=1e-4).fit(all_centers)
            global_centers, center_labels = model.cluster_centers_, model.predict(all_centers)
            self.iterations = int(model.n_iter_)
            updated = global_centers.copy()
            for r in range(k):
                if np.any(center_labels == r):
                    updated[r] = all_centers[center_labels == r].mean(axis=0)
            self.converged = bool(np.allclose(updated, global_centers, rtol=0., atol=1e-8))

        self.global_centers = global_centers
        self.local_center_labels = []
        offset = 0
        for centers in local_centers:
            n = len(centers)
            # A fixed wire dtype prevents platform/sklearn int32-vs-int64
            # differences from silently changing the reported downlink bytes.
            self.local_center_labels.append(np.asarray(center_labels[offset:offset + n], dtype=np.int64))
            offset += n
        return global_centers

    def induced_labels(self, client_index: int,
                       local_result: LocalKFedResult) -> np.ndarray:
        if self.global_centers is None:
            raise RuntimeError("aggregate must be called before induced_labels")
        mapping = self.local_center_labels[client_index]
        labels = np.asarray(local_result.labels)
        if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= len(mapping))):
            raise ValueError("local labels must index the uploaded local centers")
        return mapping[labels]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.global_centers is None:
            raise RuntimeError("aggregate must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.global_centers.shape[1] or not np.isfinite(X).all():
            raise ValueError("query data must be finite and match the fitted feature width")
        return np.argmin(cdist(X, self.global_centers), axis=1)
