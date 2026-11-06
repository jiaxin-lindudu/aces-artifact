"""Federated K-means implementation with Model Averaging (MA) and Gradient Sharing (GS)."""
import numpy as np


class KMeansClient:
    """Client for hard K-means clustering (Model Averaging and Gradient Sharing)"""

    def __init__(self, client_id, X, dp_sigma=0.0):
        self.id = client_id
        self.X = X
        self.N = len(X)
        self.dp_sigma = dp_sigma
        self.v = None

    def compute_distances_for_init(self, centers):
        """Compute squared distances for federated k-means++ initialization"""
        distances = np.array([np.linalg.norm(self.X - c, axis=1) ** 2 for c in centers])
        min_distances = np.min(distances, axis=0)
        return min_distances.sum()

    def select_profile_for_init(self, centers, v_prime):
        """Select profile for federated k-means++ initialization"""
        distances = np.array([np.linalg.norm(self.X - c, axis=1) ** 2 for c in centers])
        min_distances_sq = np.min(distances, axis=0)

        cumulative = np.cumsum(min_distances_sq)
        total = cumulative[-1]
        threshold = v_prime * total

        for idx, cum_val in enumerate(cumulative):
            if cum_val >= threshold:
                return self.X[idx]

        return self.X[-1]

    def local_model_averaging(self, centers, Q=20, B=40):
        """
        Local computation for Model Averaging (Algorithm 3 in paper).
        Performs mini-batch k-means for Q local iterations.

        Args:
            centers: Current global centers (K x M)
            Q: Number of local iterations
            B: Mini-batch size

        Returns:
            updated_centers: Local updated centers (K x M)
            v: Cluster sizes (K,)
        """
        K = len(centers)
        centers_local = centers.copy()
        v = np.zeros(K, dtype=int)

        for q in range(Q):
            # Randomly select mini-batch
            batch_size = min(B, self.N)
            batch_indices = np.random.choice(self.N, size=batch_size, replace=False)
            batch = self.X[batch_indices]

            # For each sample in batch
            for sample in batch:
                # Find nearest center (hard assignment)
                distances = np.linalg.norm(centers_local - sample, axis=1)
                c_idx = np.argmin(distances)

                # Update cluster size
                v[c_idx] += 1

                # Incremental center update (Eq. 7-8 in paper)
                eta = 1.0 / v[c_idx]
                centers_local[c_idx] = (1 - eta) * centers_local[c_idx] + eta * sample

        # Add differential privacy noise if needed
        if self.dp_sigma > 0:
            centers_local += np.random.normal(0, self.dp_sigma, centers_local.shape)

        self.v = v
        return centers_local, v

    def local_gradient_sharing(self, centers, Q=20, B=40):
        """
        Local computation for Gradient Sharing (Algorithm 4 in paper).
        Computes gradient statistics for Q local iterations.

        Args:
            centers: Current global centers (K x M)
            Q: Number of local iterations
            B: Mini-batch size

        Returns:
            gradients: List of Q gradient statistics, each containing:
                       - H_TH: H^T @ H (K x K)
                       - X_TH: X^T @ H (M x K)
        """
        gradients = []
        K = len(centers)

        for q in range(Q):
            # Randomly select mini-batch
            batch_size = min(B, self.N)
            batch_indices = np.random.choice(self.N, size=batch_size, replace=False)
            batch = self.X[batch_indices]  # B x M

            # Compute cluster assignments (hard assignment)
            distances = np.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # Create assignment matrix H (B x K)
            # H[i, k] = 1 if sample i is assigned to cluster k
            H = np.zeros((len(batch), K))
            H[np.arange(len(batch)), labels] = 1

            # Compute gradient statistics
            # H_TH = H^T @ H (K x K): cluster overlap / size information
            # X_TH = X^T @ H (M x K): sum of assigned samples per cluster
            H_TH = H.T @ H  # K x K
            X_TH = batch.T @ H  # M x K

            # Add differential privacy noise if needed
            if self.dp_sigma > 0:
                H_TH += np.random.normal(0, self.dp_sigma, H_TH.shape)
                X_TH += np.random.normal(0, self.dp_sigma, X_TH.shape)

            gradients.append((H_TH, X_TH))

        return gradients


class KMeansServer:
    """Server for hard K-means clustering"""

    def __init__(self, K, M):
        self.K = K
        self.M = M
        self.centers = None

    def federated_kmeans_plusplus_init(self, clients, seed=0):
        """
        Federated k-means++ initialization (Algorithm 2 in paper)

        Args:
            clients: List of KMeansClient objects
            seed: Random seed

        Returns:
            centers: Initial centers (K x M)
        """
        np.random.seed(seed)

        # Step 1: Randomly select one profile from one client
        client_idx = np.random.randint(len(clients))
        profile_idx = np.random.randint(clients[client_idx].N)
        centers = [clients[client_idx].X[profile_idx]]

        # Step 2-K: Iteratively select K-1 more centers
        for k in range(1, self.K):
            # Each client computes sum of squared distances
            D_p = [client.compute_distances_for_init(centers) for client in clients]
            D_total = sum(D_p)

            if D_total == 0:
                # Fallback: randomly select
                client_idx = np.random.randint(len(clients))
                profile_idx = np.random.randint(clients[client_idx].N)
                centers.append(clients[client_idx].X[profile_idx])
                continue

            # Server generates random value and selects client
            v = np.random.rand()
            cumulative = 0
            selected_client_idx = 0

            for p, D in enumerate(D_p):
                cumulative += D / D_total
                if v <= cumulative:
                    selected_client_idx = p
                    break

            # Update v' for selected client
            v_prime = v - sum(D_p[:selected_client_idx]) / D_total

            # Selected client finds the profile
            new_center = clients[selected_client_idx].select_profile_for_init(
                centers, v_prime
            )
            centers.append(new_center)

        self.centers = np.array(centers)
        return self.centers

    def aggregate_model_averaging(self, client_results, weighted=True):
        """
        Aggregate centers from clients (Eq. 9-10 in paper)

        Args:
            client_results: List of (centers, v) tuples
            weighted: Whether to use weighted averaging

        Returns:
            aggregated_centers: Aggregated centers (K x M)
        """
        if weighted:
            # Weighted average based on cluster sizes (Eq. 10)
            total_v = sum(v for _, v in client_results)
            aggregated_centers = np.zeros((self.K, self.M))

            for centers_p, v_p in client_results:
                weights = v_p / (total_v + 1e-10)
                aggregated_centers += weights[:, None] * centers_p
        else:
            # Simple average (Eq. 9)
            centers_list = [centers for centers, _ in client_results]
            aggregated_centers = np.mean(centers_list, axis=0)

        return aggregated_centers

    def update_centers_with_gradient(self, C_prev, all_gradients, beta, B, alpha):
        """
        Update centers using aggregated gradients (Eq. 13-15 in paper)

        Args:
            C_prev: Previous centers (K x M)
            all_gradients: List of gradient lists from clients
            beta: Learning rate
            B: Batch size
            alpha: Number of active clients

        Returns:
            updated_centers: Updated centers (K x M)
        """
        C = C_prev.copy()
        Q = len(all_gradients[0])

        # For each local iteration
        for q in range(Q):
            # Aggregate gradients across clients for iteration q
            H_TH_sum = sum(grads[q][0] for grads in all_gradients)  # K x K
            X_TH_sum = sum(grads[q][1] for grads in all_gradients)  # M x K

            # Compute gradient (Eq. 14-15)
            # grad = H_TH_sum @ C - X_TH_sum   (K, K) @ (K, M) - (M, K) = (K, M)
            grad = (2.0 / (B * alpha)) * (H_TH_sum @ C - X_TH_sum.T)

            # Update centers (Eq. 13)
            C = C - beta * grad

        return C

# import numpy as np
#
#
# # ==================== New Client Class for K-Means MA and GS ====================
# class KMeansClient:
#     """Client for hard K-means clustering (Model Averaging and Gradient Sharing)"""
#
#     def __init__(self, client_id, X, dp_sigma=0.0):
#         self.id = client_id
#         self.X = X
#         self.N = len(X)
#         self.dp_sigma = dp_sigma
#         self.v = None
#
#     def compute_distances_for_init(self, centers):
#         """Compute squared distances for federated k-means++ initialization"""
#         distances = np.array([np.linalg.norm(self.X - c, axis=1) ** 2 for c in centers])
#         min_distances = np.min(distances, axis=0)
#         return min_distances.sum()
#
#     def select_profile_for_init(self, centers, v_prime):
#         """Select profile for federated k-means++ initialization"""
#         distances = np.array([np.linalg.norm(self.X - c, axis=1) ** 2 for c in centers])
#         min_distances_sq = np.min(distances, axis=0)
#
#         cumulative = np.cumsum(min_distances_sq)
#         total = cumulative[-1]
#         threshold = v_prime * total
#
#         for idx, cum_val in enumerate(cumulative):
#             if cum_val >= threshold:
#                 return self.X[idx]
#
#         return self.X[-1]
#
#     def local_model_averaging(self, centers, Q=20, B=40):
#         """
#         Local computation for Model Averaging (Algorithm 3 in paper).
#         Performs mini-batch k-means for Q local iterations.
#
#         Args:
#             centers: Current global centers (K x M)
#             Q: Number of local iterations
#             B: Mini-batch size
#
#         Returns:
#             updated_centers: Local updated centers (K x M)
#             v: Cluster sizes (K,)
#         """
#         K = len(centers)
#         centers_local = centers.copy()
#         v = np.zeros(K, dtype=int)
#
#         for q in range(Q):
#             # Randomly select mini-batch
#             batch_size = min(B, self.N)
#             batch_indices = np.random.choice(self.N, size=batch_size, replace=False)
#             batch = self.X[batch_indices]
#
#             # For each sample in batch
#             for sample in batch:
#                 # Find nearest center (hard assignment)
#                 distances = np.linalg.norm(centers_local - sample, axis=1)
#                 c_idx = np.argmin(distances)
#
#                 # Update cluster size
#                 v[c_idx] += 1
#
#                 # Incremental center update (Eq. 7-8 in paper)
#                 eta = 1.0 / v[c_idx]
#                 centers_local[c_idx] = (1 - eta) * centers_local[c_idx] + eta * sample
#
#         # Add differential privacy noise if needed
#         if self.dp_sigma > 0:
#             centers_local += np.random.normal(0, self.dp_sigma, centers_local.shape)
#
#         self.v = v
#         return centers_local, v
#
#     def local_gradient_sharing(self, centers, Q=20, B=40):
#         """
#         Local computation for Gradient Sharing (Algorithm 4 in paper).
#         Computes gradient statistics for Q local iterations.
#
#         Args:
#             centers: Current global centers (K x M)
#             Q: Number of local iterations
#             B: Mini-batch size
#
#         Returns:
#             gradients: List of Q gradient statistics, each containing:
#                        - H_HT: H @ H^T (K x K)
#                        - L_HT: L @ H^T (M x K)
#         """
#         gradients = []
#         K = len(centers)
#
#         for q in range(Q):
#             # Randomly select mini-batch
#             batch_size = min(B, self.N)
#             batch_indices = np.random.choice(self.N, size=batch_size, replace=False)
#             batch = self.X[batch_indices]  # B x M
#
#             # Compute cluster assignments (hard assignment)
#             distances = np.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
#             labels = np.argmin(distances, axis=1)
#
#             # Create assignment matrix H (K x B)
#             H = np.zeros((K, len(batch)))
#             H[labels, np.arange(len(batch))] = 1
#
#             # Compute gradient statistics (Eq. 14 in paper)
#             H_HT = H @ H.T  # K x K
#             L_HT = batch.T @ H.T  # M x K
#
#             # Add differential privacy noise if needed
#             if self.dp_sigma > 0:
#                 H_HT += np.random.normal(0, self.dp_sigma, H_HT.shape)
#                 L_HT += np.random.normal(0, self.dp_sigma, L_HT.shape)
#
#             gradients.append((H_HT, L_HT))
#
#         return gradients
#
#
# # ==================== New Server Class for K-Means MA and GS ====================
# class KMeansServer:
#     """Server for hard K-means clustering"""
#
#     def __init__(self, K, M):
#         self.K = K
#         self.M = M
#         self.centers = None
#
#     def federated_kmeans_plusplus_init(self, clients, seed=0):
#         """
#         Federated k-means++ initialization (Algorithm 2 in paper)
#
#         Args:
#             clients: List of KMeansClient objects
#             seed: Random seed
#
#         Returns:
#             centers: Initial centers (K x M)
#         """
#         np.random.seed(seed)
#
#         # Step 1: Randomly select one profile from one client
#         client_idx = np.random.randint(len(clients))
#         profile_idx = np.random.randint(clients[client_idx].N)
#         centers = [clients[client_idx].X[profile_idx]]
#
#         # Step 2-K: Iteratively select K-1 more centers
#         for k in range(1, self.K):
#             # Each client computes sum of squared distances
#             D_p = [client.compute_distances_for_init(centers) for client in clients]
#             D_total = sum(D_p)
#
#             if D_total == 0:
#                 # Fallback: randomly select
#                 client_idx = np.random.randint(len(clients))
#                 profile_idx = np.random.randint(clients[client_idx].N)
#                 centers.append(clients[client_idx].X[profile_idx])
#                 continue
#
#             # Server generates random value and selects client
#             v = np.random.rand()
#             cumulative = 0
#             selected_client_idx = 0
#
#             for p, D in enumerate(D_p):
#                 cumulative += D / D_total
#                 if v <= cumulative:
#                     selected_client_idx = p
#                     break
#
#             # Update v' for selected client
#             v_prime = v - sum(D_p[:selected_client_idx]) / D_total
#
#             # Selected client finds the profile
#             new_center = clients[selected_client_idx].select_profile_for_init(
#                 centers, v_prime
#             )
#             centers.append(new_center)
#
#         self.centers = np.array(centers)
#         return self.centers
#
#     def aggregate_model_averaging(self, client_results, weighted=True):
#         """
#         Aggregate centers from clients (Eq. 9-10 in paper)
#
#         Args:
#             client_results: List of (centers, v) tuples
#             weighted: Whether to use weighted averaging
#
#         Returns:
#             aggregated_centers: Aggregated centers (K x M)
#         """
#         if weighted:
#             # Weighted average based on cluster sizes (Eq. 10)
#             total_v = sum(v for _, v in client_results)
#             aggregated_centers = np.zeros((self.K, self.M))
#
#             for centers_p, v_p in client_results:
#                 weights = v_p / (total_v + 1e-10)
#                 aggregated_centers += weights[:, None] * centers_p
#         else:
#             # Simple average (Eq. 9)
#             centers_list = [centers for centers, _ in client_results]
#             aggregated_centers = np.mean(centers_list, axis=0)
#
#         return aggregated_centers
#
#     def update_centers_with_gradient(self, C_prev, all_gradients, beta, B, alpha):
#         """
#         Update centers using aggregated gradients (Eq. 13-15 in paper)
#
#         Args:
#             C_prev: Previous centers (K x M)
#             all_gradients: List of gradient lists from clients
#             beta: Learning rate
#             B: Batch size
#             alpha: Number of active clients
#
#         Returns:
#             updated_centers: Updated centers (K x M)
#         """
#         C = C_prev.copy()
#         Q = len(all_gradients[0])
#
#         # For each local iteration
#         for q in range(Q):
#             # Aggregate gradients across clients for iteration q
#             H_HT_sum = sum(grads[q][0] for grads in all_gradients)  # K x K
#             L_HT_sum = sum(grads[q][1] for grads in all_gradients)  # M x K
#
#             # Compute gradient (Eq. 14-15)
#             grad = (2.0 / (B * alpha)) * (C @ H_HT_sum - L_HT_sum.T)
#
#             # Update centers (Eq. 13)
#             C = C - beta * grad
#
#         return C
