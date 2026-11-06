import numpy as np
from federated.ffcm.dp_mechanism import add_gaussian_noise

class Client:
    def __init__(self, id, X, model=None, dp_sigma=1.0):
        self.id = id
        self.X = X
        self.model = model
        self.dp_sigma = dp_sigma

    def local_computation(self, centers, edge_index, use_distance=True, fuzziness=2.0):
        X_np = self.X if isinstance(self.X, np.ndarray) else self.X.cpu().detach().numpy()
        centers_np = centers if isinstance(centers, np.ndarray) else centers.cpu().detach().numpy()
        if use_distance:
            membership = self._compute_fuzzy_membership_distance(X_np, centers_np, fuzziness)
        else:
            # 保持distance版本为主, 可以扩展
            membership = self._compute_fuzzy_membership_distance(X_np, centers_np, fuzziness)
        u_ij_m = membership ** fuzziness
        S_k = np.einsum('nk,nd->kd', u_ij_m, X_np)
        n_k = u_ij_m.sum(axis=0)

        S_k_noisy = add_gaussian_noise(S_k, self.dp_sigma)
        n_k_noisy = add_gaussian_noise(n_k, self.dp_sigma)

        return S_k_noisy, n_k_noisy

    def _compute_fuzzy_membership_distance(self, X, centers, m):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        distances = np.maximum(distances, 1e-10)
        power = 2.0 / (m - 1)
        membership = np.zeros_like(distances)
        for i in range(len(X)):
            for j in range(len(centers)):
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** power)
                membership[i, j] = 1.0 / sum_term
        membership = membership / membership.sum(axis=1, keepdims=True)
        return membership


class Server:
    def __init__(self, aggregation_fn, gnn_model=None):
        self.model = gnn_model
        self.agg_fn = aggregation_fn

    def aggregate(self, stat_list):
        S = sum(S for S, n in stat_list)
        n = sum(n for S, n in stat_list)
        return S / (n[:, None] + 1e-10)
