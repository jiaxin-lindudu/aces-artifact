"""Bounded relative statistics and postprocessing, FastLloyd Algorithms 1/2."""
import numpy as np
from scipy.spatial.distance import cdist


def sphere_packing_centers(clusters, dimensions, bound, seed):
    rng = np.random.default_rng(seed)
    separation = float(bound)
    for _ in range(64):
        centers = []
        for _ in range(100 * clusters):
            c = rng.uniform(-bound + separation, bound - separation, dimensions)
            if not centers or np.min(np.linalg.norm(np.asarray(centers) - c, axis=1)) >= 2 * separation:
                centers.append(c)
                if len(centers) == clusters:
                    return np.asarray(centers)
        separation /= 2
    raise RuntimeError("public sphere-packing initialization failed")


class FastLloydClient:
    def __init__(self, data, bound):
        # Public domain projection is part of the DP query, not a private scaler.
        self.data = np.clip(np.asarray(data, float), -bound, bound)

    def statistics(self, centers, radius):
        distances = cdist(self.data, centers)
        labels = distances.argmin(axis=1)
        valid = distances[np.arange(len(labels)), labels] < radius
        relative = np.zeros_like(centers)
        counts = np.bincount(labels[valid], minlength=len(centers)).astype(float)
        np.add.at(relative, labels[valid], self.data[valid] - centers[labels[valid]])
        return np.concatenate((relative.ravel(), counts)), int(valid.sum())


def update_centers(previous, packed, radius, bound, count_floor=1.0):
    k, d = previous.shape
    relative = packed[:k*d].reshape(k, d)
    counts = packed[k*d:]
    displacement = np.zeros_like(previous)
    usable = counts > count_floor
    displacement[usable] = relative[usable] / counts[usable, None]
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    displacement *= np.minimum(1, radius / np.maximum(norms, 1e-30))
    # Reflection with period 2*domain-width; matches the author's implementation.
    folded = (previous + displacement + bound) % (4 * bound)
    folded = np.where(folded > 2 * bound, 4 * bound - folded, folded)
    return folded - bound
