import numpy as np
from scipy.spatial.distance import cdist

def check_and_fix_center_collapse(centers, min_distance_threshold=0.1, data_std=0.3):
    K = len(centers)
    distances = cdist(centers, centers)
    np.fill_diagonal(distances, np.inf)
    min_dist = distances.min()
    collapsed = min_dist < min_distance_threshold
    if collapsed:
        print(f"    WARNING: Center collapse detected! Min distance = {min_dist:.4f}")
        collapsed_pairs = np.argwhere(distances < min_distance_threshold)
        fixed_centers = centers.copy()
        reinit_indices = set()
        for i, j in collapsed_pairs:
            if i < j:
                reinit_indices.add(j)
        for idx in reinit_indices:
            fixed_centers[idx] = np.random.randn(centers.shape[1]) * data_std + centers.mean(axis=0)
            print(f"      Re-initialized center {idx}")
        return fixed_centers, True
    return centers, False

def enforce_minimum_separation(centers, min_separation=0.2, max_iterations=10):
    centers = centers.copy()
    for iteration in range(max_iterations):
        distances = cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)
        min_dist = distances.min()
        if min_dist >= min_separation:
            break
        repulsion_force = np.zeros_like(centers)
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                if distances[i, j] < min_separation:
                    direction = centers[j] - centers[i]
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        force_magnitude = (min_separation - distances[i, j]) * 0.1
                        repulsion_force[i] -= direction * force_magnitude
                        repulsion_force[j] += direction * force_magnitude
        centers += repulsion_force
    return centers
