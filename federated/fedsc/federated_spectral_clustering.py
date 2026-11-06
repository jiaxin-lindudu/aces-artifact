"""
Federated Spectral Clustering (FedSC) via Similarity Reconstruction.

Reference: "Federated Spectral Clustering via Secure Similarity Reconstruction"
           NeurIPS 2023

Two-stage algorithm:
  Stage I  — Federated similarity reconstruction: clients collaboratively learn
             a shared dictionary Z and coefficient matrices C via kernel approximation.
  Stage II — Spectral clustering on the globally reconstructed similarity matrix
             K̂ = C^T K(Z,Z) C followed by KNN sparsification.

The kernel factor updates below follow Eqs. (8), (14)-(16). The graph convention
is checked against jicongfan/Federated-Spectral-Clustering (global minmax,
complete KNN including self). That MATLAB repository has no stated license;
this is an independent NumPy/SciPy implementation, not copied source. Its
preconditioned one-step optimizer is NOT claimed to match the paper's Q-step GD.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


# ============================================================================
# FedSC Client
# ============================================================================

class FedSCClient:
    """Client for Federated Spectral Clustering via Similarity Reconstruction."""

    def __init__(self, client_id, X, dp_sigma=0.0, random_state=None):
        """
        Initialize FedSC client.

        Args:
            client_id: Unique identifier for the client
            X: Local data matrix (M x Np) — features x samples
            dp_sigma: Gaussian perturbation standard deviation (not formal DP)
            random_state: Random seed for reproducible perturbation
        """
        self.id = client_id
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError('client data must be a nonempty finite features-by-samples matrix')
        if not np.isfinite(dp_sigma) or dp_sigma < 0:
            raise ValueError('dp_sigma is perturbation-only and must be nonnegative')
        self.X = X                       # Shape: (M, Np)
        self.M = X.shape[0]              # Feature dimension
        self.Np = X.shape[1]             # Number of local samples
        self.dp_sigma = dp_sigma
        self.C = None                    # Coefficient matrix (d x Np)
        self.Z = None                    # Dictionary matrix (local copy)

        # Per-client RNG for reproducible perturbation (not a DP accountant).
        seed = random_state if random_state is not None else client_id
        self._rng = np.random.RandomState(seed)

    def update_C(self, Z, lambda_reg=0.01, kernel='rbf', sigma=1.0):
        """
        Update local coefficient matrix C.

        C = (K(Z,Z) + λI)^(-1) K(Z,X)

        Args:
            Z: Shared dictionary matrix from server (M x d)
            lambda_reg: Regularization parameter
            kernel: Kernel type ('rbf')
            sigma: Kernel bandwidth

        Returns:
            C: Updated coefficient matrix (d x Np)
        """
        self.Z = Z
        if not np.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError('lambda_reg must be finite and nonnegative')
        d = Z.shape[1]

        # Compute kernel matrices
        K_ZZ = self._compute_kernel(Z.T, Z.T, kernel, sigma)    # (d x d)
        K_ZX = self._compute_kernel(Z.T, self.X.T, kernel, sigma)  # (d x Np)

        # Solve: C = (K_ZZ + λI)^(-1) K_ZX
        K_ZZ_reg = K_ZZ + lambda_reg * np.eye(d)
        self.C = np.linalg.solve(K_ZZ_reg, K_ZX)

        return self.C

    def update_Z(self, Z_init, C, Q=20, eta=0.01, kernel='rbf',
                 sigma=1.0, lambda_reg=0.01):
        """
        Update local dictionary matrix Z via gradient descent.

        Args:
            Z_init: Initial dictionary matrix (M x d)
            C: Current coefficient matrix (d x Np)
            Q: Number of local iterations
            eta: Learning rate
            kernel: Kernel type
            sigma: Kernel bandwidth
            lambda_reg: Regularization parameter

        Returns:
            Z_updated: Updated dictionary matrix (M x d)
        """
        Z = Z_init.copy()
        if Q < 1 or not np.isfinite(eta) or eta <= 0:
            raise ValueError('Q and eta must be positive')

        for _ in range(Q):
            grad = self._compute_gradient_Z(Z, C, kernel, sigma, lambda_reg)
            Z = Z - eta * grad
            if not np.isfinite(Z).all():
                raise FloatingPointError('FedSC dictionary update became nonfinite')

        if self.dp_sigma > 0:
            noise = self._rng.normal(0, self.dp_sigma, Z.shape)
            Z = Z + noise

        return Z

    def _compute_gradient_Z(self, Z, C, kernel, sigma, lambda_reg):
        """
        Compute gradient of objective w.r.t. dictionary Z.

        ∂L/∂Z = (1/σ²)(X W_Z - Z W̄_Z) + (2/σ²)(Z Q_Z - Z Q̄_Z)
        """
        K_XZ = self._compute_kernel(self.X.T, Z.T, kernel, sigma)  # (Np x d)
        K_ZZ = self._compute_kernel(Z.T, Z.T, kernel, sigma)       # (d x d)

        # W_Z = -C^T ⊙ K(X, Z)
        W_Z = -C.T * K_XZ
        W_Z_bar = np.diag(np.sum(W_Z, axis=0))

        # Q_Z = (0.5 CC^T) ⊙ K(Z, Z)
        CCT = 0.5 * (C @ C.T)
        Q_Z = CCT * K_ZZ
        Q_Z_bar = np.diag(np.sum(Q_Z, axis=0))

        sigma_sq = sigma ** 2
        grad = (1.0 / sigma_sq) * (self.X @ W_Z - Z @ W_Z_bar)
        grad += (2.0 / sigma_sq) * (Z @ Q_Z - Z @ Q_Z_bar)

        return grad

    def reconstruction_objective(self, Z, C, lambda_reg=0.01,
                                 kernel='rbf', sigma=1.0):
        """Equation (8), including the RBF self-kernel trace, without an Np² matrix."""
        K_ZX = self._compute_kernel(Z.T, self.X.T, kernel, sigma)
        K_ZZ = self._compute_kernel(Z.T, Z.T, kernel, sigma)
        return float(0.5 * self.Np - np.sum(C * K_ZX)
                     + 0.5 * np.sum(C * (K_ZZ @ C))
                     + 0.5 * lambda_reg * np.sum(C ** 2))

    def _compute_kernel(self, X1, X2, kernel='rbf', sigma=1.0):
        """
        Compute RBF kernel matrix: K(x,y) = exp(-||x-y||² / (2σ²)).

        Args:
            X1: First dataset (n1 x M)
            X2: Second dataset (n2 x M)

        Returns:
            K: Kernel matrix (n1 x n2)
        """
        if kernel in ('rbf', 'gaussian'):
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('sigma must be finite and positive')
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
            sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
            K = np.exp(-np.maximum(sq_dists, 0.0) / (2 * sigma ** 2))
            return K
        else:
            raise NotImplementedError(f"Kernel '{kernel}' not implemented")

    def add_noise_to_data(self, noise_std):
        """Perturb local data; this helper does not establish differential privacy."""
        noise = self._rng.normal(0, noise_std, self.X.shape)
        self.X = self.X + noise


# ============================================================================
# FedSC Server
# ============================================================================

class FedSCServer:
    """Server for Federated Spectral Clustering."""

    def __init__(self, K, M, d=None):
        """
        Initialize FedSC server.

        Args:
            K: Number of clusters
            M: Feature dimension
            d: Dictionary size (default: K * 5)
        """
        self.K = K
        self.M = M
        self.d = d if d is not None else K * 5
        self.Z = None               # Shared dictionary (M x d)
        self.C_global = None        # Global coefficient stack
        self.spectral_diagnostics = {}

    def initialize_dictionary(self, clients, d=None, seed=0):
        """
        Initialize a random shared dictionary without collecting private rows.

        Args:
            clients: List of FedSCClient objects
            d: Dictionary size (overrides self.d)
            seed: Random seed

        Returns:
            Z: Initialized dictionary matrix (M x d)
        """
        rng = np.random.RandomState(seed)

        if d is not None:
            self.d = d

        self.Z = rng.normal(0.0, 1.0, size=(self.M, self.d))

        return self.Z

    def aggregate_Z(self, Z_list, weights=None):
        """
        Weighted aggregation of dictionary matrices from clients.

        Args:
            Z_list: List of local dictionary matrices
            weights: Client weights (default: uniform)

        Returns:
            Z_aggregated: Aggregated dictionary matrix
        """
        if weights is None:
            self.Z = np.mean(Z_list, axis=0)
        else:
            weights = np.asarray(weights, dtype=float)
            if (len(weights) != len(Z_list) or not np.isfinite(weights).all()
                    or np.any(weights < 0) or weights.sum() <= 0):
                raise ValueError('aggregation weights must be finite, nonnegative and nonempty')
            weights = weights / weights.sum()
            self.Z = np.zeros_like(Z_list[0])
            for i, Z_i in enumerate(Z_list):
                self.Z += weights[i] * Z_i
        return self.Z

    def federated_similarity_reconstruction(self, clients, S=10, Q=20,
                                             eta=0.01, lambda_reg=0.01,
                                             kernel='rbf', sigma=1.0,
                                             active_ratio=1.0, seed=0):
        """
        Stage I: Federated Similarity Reconstruction.

        Args:
            clients: List of FedSCClient objects
            S: Communication rounds
            Q: Local iterations per round
            eta: Learning rate
            lambda_reg: Regularization
            kernel: Kernel type
            sigma: Kernel bandwidth
            active_ratio: Fraction of active clients per round
            seed: Random seed

        Returns:
            Z: Final shared dictionary
            C_list: Coefficient matrices from all clients
        """
        rng = np.random.RandomState(seed)
        P = len(clients)
        P_active = max(1, int(P * active_ratio))

        if self.Z is None:
            self.initialize_dictionary(clients, seed=seed)

        for s in range(S):
            active_indices = rng.choice(P, P_active, replace=False)

            Z_list = []
            for idx in active_indices:
                client = clients[idx]
                client.update_C(self.Z, lambda_reg, kernel, sigma)
                Z_local = client.update_Z(
                    self.Z, client.C, Q, eta, kernel, sigma, lambda_reg
                )
                Z_list.append(Z_local)

            # Equation (11) averages participating dictionaries uniformly.
            self.aggregate_Z(Z_list)

        # Final C collection from all clients
        C_list = []
        for client in clients:
            client.update_C(self.Z, lambda_reg, kernel, sigma)
            C_list.append(client.C)

        return self.Z, C_list

    def spectral_clustering(self, clients, Z, C_list, kernel='rbf',
                            sigma=1.0, k_neighbors=None, n_eigenvecs=None,
                            random_state=0, eigen_solver='auto',
                            return_similarity=True, block_size=256):
        """
        Stage II: Spectral Clustering on reconstructed similarity matrix.

        K̂ = C^T K(Z,Z) C → KNN sparsification → spectral clustering

        Args:
            clients: List of clients
            Z: Final dictionary matrix
            C_list: Coefficient matrices from clients
            kernel: Kernel type
            sigma: Kernel bandwidth
            k_neighbors: KNN sparsification parameter
            n_eigenvecs: Number of eigenvectors (default: n_clusters)

        Returns:
            labels: Cluster assignments
            K_reconstructed: Global-minmax normalized reconstruction, or None
                             when return_similarity=False (bounded-memory path)
        """
        self.C_global = np.hstack(C_list)  # Labels follow this exact client order.
        if not np.isfinite(self.C_global).all():
            raise FloatingPointError('FedSC coefficients must be finite')
        n = self.C_global.shape[1]
        if k_neighbors is None:
            k_neighbors = max(int(np.ceil(np.log(n))), 1)
        k_neighbors = min(int(k_neighbors), n)
        if k_neighbors < 1 or block_size < 1:
            raise ValueError('k_neighbors and block_size must be positive')
        K_ZZ = self._compute_kernel(Z.T, Z.T, kernel, sigma)
        projected = K_ZZ @ self.C_global
        K_reconstructed = np.empty((n, n)) if return_similarity else None
        neighbor_indices = np.empty((n, k_neighbors), dtype=np.int64)
        neighbor_values = np.empty((n, k_neighbors), dtype=float)
        minimum, maximum = np.inf, -np.inf
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            block = self.C_global[:, start:stop].T @ projected
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
            if return_similarity:
                K_reconstructed[start:stop] = block
            for offset, row in enumerate(block):
                indices = self._largest_neighbors(row, k_neighbors)
                neighbor_indices[start + offset] = indices
                neighbor_values[start + offset] = row[indices]

        # The author's graph path maps the GLOBAL reconstructed kernel to [0,1]
        # before complete (max-union) KNN. It is not abs(K), mutual KNN or averaging.
        # Self similarities participate in the k largest entries, as in that path.
        scale = maximum - minimum
        if scale > 0:
            neighbor_values = (neighbor_values - minimum) / scale
            if return_similarity:
                K_reconstructed -= minimum
                K_reconstructed /= scale
        else:
            raise ValueError('FedSC reconstructed kernel is constant; spectral structure is undefined')
        rows = np.repeat(np.arange(n), k_neighbors)
        directed = sparse.csr_matrix((neighbor_values.ravel(),
                                      (rows, neighbor_indices.ravel())), shape=(n, n))
        K_sparse = directed.maximum(directed.T).tocsr()
        K_sparse.eliminate_zeros()
        self.spectral_diagnostics = {
            'graph_rule': 'global minmax [0,1]; top-k including self; max-union',
            'raw_kernel_min': minimum, 'raw_kernel_max': maximum,
            'neighbors': k_neighbors, 'graph_nnz': int(K_sparse.nnz),
            'graph_storage_bytes': int(K_sparse.data.nbytes + K_sparse.indices.nbytes
                                       + K_sparse.indptr.nbytes),
            'dense_similarity_retained': bool(return_similarity),
            'label_order': 'concatenated X_list order',
        }
        labels = self._standard_spectral_clustering(
            K_sparse, self.K, n_eigenvecs, random_state=random_state,
            eigen_solver=eigen_solver
        )
        return labels, K_reconstructed

    @staticmethod
    def _largest_neighbors(row, k):
        """Top similarities with deterministic sample-index tie breaking."""
        if k == len(row):
            return np.arange(len(row))
        selected = np.argpartition(row, len(row) - k)[-k:]
        cutoff = row[selected].min()
        greater = np.flatnonzero(row > cutoff)
        equal = np.flatnonzero(row == cutoff)
        return np.concatenate((greater, equal[:k - len(greater)]))

    def _knn_sparsification(self, K, k):
        """Dense reference for complete KNN on an already normalized kernel."""
        N = K.shape[0]
        K_sparse = np.zeros_like(K)
        for i in range(N):
            top_k_indices = self._largest_neighbors(K[i], min(k, N))
            K_sparse[i, top_k_indices] = K[i, top_k_indices]
        return np.maximum(K_sparse, K_sparse.T)

    def _standard_spectral_clustering(self, K, n_clusters, n_eigenvecs=None,
                                      random_state=0, eigen_solver='auto'):
        """
        Standard spectral clustering on similarity matrix.

        1. Normalized Laplacian L = D^(-1/2) K D^(-1/2)
        2. Top eigenvectors
        3. Row-normalize
        4. K-means
        """
        if n_eigenvecs is None:
            n_eigenvecs = n_clusters
        n = K.shape[0]
        if not 1 <= n_clusters <= n or not n_clusters <= n_eigenvecs <= n:
            raise ValueError('spectral dimensions must satisfy K <= n_eigenvecs <= N')
        if eigen_solver not in ('auto', 'dense', 'arpack'):
            raise ValueError('eigen_solver must be auto, dense or arpack')
        graph = sparse.csr_matrix(K, dtype=float)
        if not np.isfinite(graph.data).all() or np.any(graph.data < 0):
            raise ValueError('spectral graph must have finite nonnegative weights')
        d_diag = np.asarray(graph.sum(axis=1)).ravel()
        # Broadcasting / sparse scaling avoids constructing any dense diagonal matrix.
        inv = 1.0 / np.sqrt(d_diag + np.finfo(float).eps)
        L_norm = graph.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
        solver = ('dense' if n <= 256 or n_eigenvecs == n else 'arpack') if (
            eigen_solver == 'auto') else eigen_solver
        if solver == 'dense':
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm.toarray())
        else:
            if n_eigenvecs >= n:
                raise ValueError('ARPACK requires n_eigenvecs < N; use dense for this case')
            v0 = np.random.RandomState(random_state).normal(size=n)
            eigenvalues, eigenvectors = eigsh(
                L_norm, k=n_eigenvecs, which='LA', v0=v0,
                tol=1e-10, maxiter=max(1000, n * 10))
        idx = np.argsort(eigenvalues)[::-1][:n_eigenvecs]
        V = eigenvectors[:, idx]
        self.spectral_diagnostics['eigen_solver'] = solver
        self.spectral_diagnostics['eigenvalues'] = eigenvalues[idx].tolist()
        self.spectral_diagnostics['eigen_residual'] = float(np.linalg.norm(
            L_norm @ V - V * eigenvalues[idx], ord='fro'))
        self.spectral_diagnostics['zero_degree_rows'] = int(np.count_nonzero(d_diag == 0))
        self.spectral_diagnostics['eigenvectors'] = V.copy()

        # Row-normalize
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        row_norms[row_norms < 1e-10] = 1.0
        V = V / row_norms

        return self._kmeans(V, n_clusters, max_iter=1000, seed=random_state)

    def _kmeans(self, X, K, max_iter=100, seed=0):
        """Ng-Jordan-Weiss final KMeans, matching the author's 50-restart budget."""
        return KMeans(n_clusters=K, n_init=50, max_iter=max_iter,
                      random_state=seed, algorithm='lloyd').fit_predict(X)

    def _compute_kernel(self, X1, X2, kernel='rbf', sigma=1.0):
        """RBF kernel computation (same as client)."""
        return FedSCClient._compute_kernel(self, X1, X2, kernel, sigma)

    def federated_spectral_clustering(self, clients, S=10, Q=20, eta=0.01,
                                       lambda_reg=0.01, kernel='rbf',
                                       sigma=None, active_ratio=1.0,
                                       k_neighbors=None, seed=0,
                                       eigen_solver='auto', return_similarity=True):
        """
        Complete FedSC pipeline: Stage I + Stage II.

        Args:
            clients: List of FedSCClient objects
            S: Communication rounds
            Q: Local iterations per round
            eta: Learning rate
            lambda_reg: Regularization
            kernel: Kernel type
            sigma: Kernel bandwidth (auto-estimated if None)
            active_ratio: Fraction of active clients per round
            k_neighbors: KNN sparsification neighbors
            seed: Random seed

        Returns:
            labels: Final cluster assignments
            K_reconstructed: Reconstructed similarity matrix
        """
        if sigma is None:
            sigma = self._estimate_kernel_bandwidth(clients, seed=seed)

        Z, C_list = self.federated_similarity_reconstruction(
            clients, S, Q, eta, lambda_reg, kernel, sigma, active_ratio, seed
        )

        labels, K_reconstructed = self.spectral_clustering(
            clients, Z, C_list, kernel, sigma, k_neighbors,
            random_state=seed, eigen_solver=eigen_solver,
            return_similarity=return_similarity
        )

        return labels, K_reconstructed

    def _estimate_kernel_bandwidth(self, clients, seed=0):
        """
        Estimate bandwidth from local distance summaries, without raw uploads.
        """
        rng = np.random.RandomState(seed)
        local_means = []
        for client in clients:
            n = min(100, client.Np)
            if n < 2:
                continue
            idx = rng.choice(client.Np, n, replace=False)
            sample = client.X[:, idx].T
            distances = np.linalg.norm(
                sample[:, None, :] - sample[None, :, :], axis=2
            )
            local_means.append(float(distances[np.triu_indices(n, 1)].mean()))
        return max(float(np.mean(local_means)), 1e-6) if local_means else 1.0
