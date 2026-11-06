from typing import List, Any, Dict, Optional, Tuple
import time

import numpy as np
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class NNFCClient:
    """
    NN-FC (Nearest Neighbor-based Fuzzy C-Means) Client with support for both
    automatic and manual epsilon calculation for differential privacy.

    Args:
        id: Unique identifier for the client
        X: Local dataset for this client (n_samples, n_features)
        random_state: Random seed for reproducibility
    """

    def __init__(self, id: int, X: np.ndarray, dp_epsilon=None, random_state=42):
        if dp_epsilon is not None:
            self.manual_epsilon = dp_epsilon
            self._validate_epsilon(self.manual_epsilon)
        else:
            self.manual_epsilon = None  # Add this line

        self.id = id
        self.X = X
        self.n_samples = len(X)
        self.n_features = X.shape[1] if self.n_samples > 0 else 0
        self.noised_data = None
        self.n_clusters = None
        self.epsilon = None
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)

        # Performance tracking (per-round)
        self.encryption_time = 0
        self.communication_bytes = 0

    def _validate_epsilon(self, epsilon: float) -> None:
        """Validate the provided epsilon value."""
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"Epsilon must be a number, got {type(epsilon)}")
        if epsilon < 0:
            raise ValueError(f"Epsilon must be non-negative, got {epsilon}")
        if epsilon > 1000:
            print(f"Warning: Very large epsilon ({epsilon}) provides minimal privacy")

    def _generate_laplacian_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Generate Laplacian noise following the official implementation exactly.
        """
        if self.manual_epsilon is not None:
            self.epsilon = self.manual_epsilon
            if self.epsilon == 0:
                return data.copy()
        else:
            # Automatic epsilon calculation - CORRECTED VERSION
            try:
                euc = euclidean_distances(data)
                row, col = np.diag_indices_from(euc)
                euc[row, col] = np.max(euc)
                minvalue = np.min(euc)

                # Generate uniform noise
                noise = self.rng.uniform(0, 1, size=data.shape)

                # CORRECTED: Match official implementation
                noisenew = noise.copy()
                noisenew[noisenew >= 0.5] = 1 - noisenew[noisenew >= 0.5]

                # Compute ratio
                arr = noisenew.ravel()
                ratio = []
                for i in arr:
                    for j in arr:
                        if abs(j) > 1e-15:
                            num1, num2 = float(i) / float(j), float(j) / float(i)
                            ratio.append(max([num1, num2]))

                self.epsilon = 1.01 * max(ratio) / minvalue if ratio else 1.0

            except Exception as e:
                print(f"Error in automatic epsilon calculation: {e}")
                self.epsilon = 1.0

        # CORRECTED: Generate Laplacian noise using inverse transform sampling
        # to match official implementation
        scale = 1.0 / max(self.epsilon, 1e-10)

        # Generate uniform random variables
        uniform_noise = self.rng.uniform(0, 1, size=data.shape)

        # Apply inverse transform sampling
        laplacian_noise = np.zeros_like(uniform_noise)
        mask_low = uniform_noise < 0.5
        mask_high = uniform_noise >= 0.5

        laplacian_noise[mask_low] = scale * np.log(2 * uniform_noise[mask_low])
        laplacian_noise[mask_high] = -scale * np.log(2 * (1 - uniform_noise[mask_high]))

        return data + laplacian_noise

    def _determine_n_clusters(self, data: np.ndarray) -> int:
        """
        Determine number of clusters following official implementation.
        Note: Official doesn't have max(1, ...) but it's a good safety check.
        """
        # Official version: return min(len(data) // 3, 50)
        # Improved version with safety:
        return max(1, min(len(data) // 3, 50))

    def local_train(self, global_centers: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Perform local training and return cluster centers.

        Args:
            global_centers: Global cluster centers from server (unused in NNFC)
            **kwargs: Additional arguments (for compatibility)

        Returns:
            np.ndarray: Local cluster centers
        """
        if self.n_samples == 0:
            return np.empty((0, self.n_features))

        # Add Laplacian noise to data
        self.noised_data = self._generate_laplacian_noise(self.X)

        # Determine number of clusters
        self.n_clusters = self._determine_n_clusters(self.noised_data)

        # Apply K-Means clustering
        try:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init="auto",
                random_state=self.random_state
            ).fit(self.noised_data)
            centers = kmeans.cluster_centers_
        except Exception as e:
            print(f"K-Means failed for client {self.id}: {e}")
            if self.n_clusters == 1:
                centers = np.mean(self.noised_data, axis=0, keepdims=True)
            else:
                if len(self.noised_data) >= self.n_clusters:
                    indices = self.rng.choice(
                        len(self.noised_data),
                        self.n_clusters,
                        replace=False
                    )
                    centers = self.noised_data[indices]
                else:
                    centers = self.noised_data.copy()

        return centers

    def get_privacy_info(self) -> Dict[str, Any]:
        """
        Get privacy-related information for analysis.

        Returns:
            Dictionary containing privacy-related metrics and metadata
        """
        return {
            'client_id': self.id,
            'epsilon': self.epsilon,
            'n_clusters': self.n_clusters,
            'data_size': self.n_samples,
            'formula': 'n_clusters = min([len(data) // 3, 50])',
            'epsilon_formula': '1.01 * maxratio / minvalue' if self.manual_epsilon is None else 'manual',
            'manual_epsilon': self.manual_epsilon,
            'epsilon_source': 'manual' if self.manual_epsilon is not None else 'automatic'
        }

    def upload_local_summaries(self) -> Optional[Tuple[np.ndarray, ...]]:
        """NN-FC does not use this style of summary upload (for compatibility)."""
        return None

    def compute_dual_statistics(self, *args, **kwargs) -> Tuple[Any, ...]:
        """Placeholder for compatibility with AnchoredSpectralClient API."""
        raise NotImplementedError("NNFC_Client does not implement compute_dual_statistics")

    def get_noisy_statistics(self, *args, **kwargs) -> Tuple[Any, ...]:
        """Placeholder for compatibility with AnchoredSpectralClient API."""
        raise NotImplementedError("NNFC_Client does not implement get_noisy_statistics")


class NNFCServer:
    """
    NN-FC Server following the official implementation's SNN algorithm exactly.
    """

    def __init__(self, clients, num_clusters, k_neighbors=2):
        self.n_components = num_clusters
        self.k = k_neighbors # Initial k value from official implementation
        self.clients_list = clients
        self.all_anchor_points: Optional[np.ndarray] = None
        self.global_centers: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.fallback_used = False
        self.fallback_error = ''

    def SNN(self, k: int, nc: int, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact reproduction of the official SNN function with minor improvements.
        """
        unassigned = -1
        n, d = data.shape

        # Compute distance
        distance = squareform(pdist(data))

        # Compute neighbor
        indexDistanceAsc = np.argsort(distance)
        indexNeighbor = indexDistanceAsc[:, :k]

        # Compute shared neighbor
        indexSharedNeighbor = np.empty([n, n, k], int)
        numSharedNeighbor = np.empty([n, n], int)

        for i in range(n):
            numSharedNeighbor[i, i] = 0
            for j in range(i):
                shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j], assume_unique=True)
                numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = shared.size
                if shared.size > 0:
                    indexSharedNeighbor[j, i, :shared.size] = indexSharedNeighbor[i, j, :shared.size] = shared

        # Compute similarity - CORRECTED to match official exactly
        similarity = np.zeros([n, n])
        for i in range(n):
            for j in range(i):
                # Check if i and j are in their shared neighbors first
                if numSharedNeighbor[i, j] > 0:
                    indexShared = indexSharedNeighbor[i, j, :numSharedNeighbor[i, j]]
                    if i in indexShared and j in indexShared:
                        distanceSum = np.sum(distance[i, indexShared] + distance[j, indexShared])
                        # Safety check to avoid division by zero
                        if distanceSum > 0:
                            similarity[i, j] = similarity[j, i] = numSharedNeighbor[i, j] ** 2 / distanceSum

        # Compute ρ (rho)
        rho = np.sum(np.sort(similarity)[:, -k:], axis=1)

        # Compute δ (delta)
        distanceNeighborSum = np.empty(n)
        for i in range(n):
            distanceNeighborSum[i] = np.sum(distance[i, indexNeighbor[i]])

        indexRhoDesc = np.argsort(rho)[::-1]
        delta = np.full(n, np.inf)

        for i, a in enumerate(indexRhoDesc[1:], 1):
            for b in indexRhoDesc[:i]:
                delta[a] = min(delta[a], distance[a, b] * (distanceNeighborSum[a] + distanceNeighborSum[b]))

        delta[indexRhoDesc[0]] = -np.inf
        delta[indexRhoDesc[0]] = np.max(delta)

        # Compute γ (gamma)
        gamma = rho * delta

        # Compute centroid
        indexAssignment = np.full(n, unassigned)
        indexCentroid = np.sort(np.argsort(gamma)[-nc:])
        indexAssignment[indexCentroid] = np.arange(nc)

        # Assign non-centroid step 1
        queue = indexCentroid.tolist()
        while queue:
            a = queue.pop(0)
            for b in indexNeighbor[a]:
                if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k / 2:
                    indexAssignment[b] = indexAssignment[a]
                    queue.append(b)

        # Assign non-centroid step 2
        current_k = k
        indexUnassigned = np.argwhere(indexAssignment == unassigned).flatten()

        while indexUnassigned.size:
            numNeighborAssignment = np.zeros([indexUnassigned.size, nc], int)

            for i, a in enumerate(indexUnassigned):
                for b in indexDistanceAsc[a, :current_k]:
                    if indexAssignment[b] != unassigned:
                        numNeighborAssignment[i, indexAssignment[b]] += 1

            most = np.max(numNeighborAssignment)
            if most > 0:
                temp = np.argwhere(numNeighborAssignment == most)
                indexAssignment[indexUnassigned[temp[:, 0]]] = temp[:, 1]
                indexUnassigned = np.argwhere(indexAssignment == unassigned).flatten()
            else:
                current_k += 1

        return indexCentroid, indexAssignment

    def init_global_centers(self, method: str, client_anchor_points_list: Optional[List[Optional[np.ndarray]]] = None,
                            **kwargs) -> None:
        """Initialize global centers using the official SNN algorithm"""

        if client_anchor_points_list is None:
            self.all_anchor_points = np.array([])
            self.global_centers = np.array([])
            return

        # Validate and combine anchor points
        valid_anchor_points = [ap for ap in client_anchor_points_list if
                               isinstance(ap, np.ndarray) and ap.ndim == 2 and ap.shape[0] > 0]

        if not valid_anchor_points:
            self.all_anchor_points = np.array([])
            self.global_centers = np.array([])
            return

        self.all_anchor_points = np.vstack(valid_anchor_points)

        # Apply SNN algorithm
        try:
            centroid_indices, assignments = self.SNN(self.k, self.n_components, self.all_anchor_points)
            self.global_centers = self.all_anchor_points[centroid_indices]
            self.assignments = assignments
        except Exception as e:
            self.fallback_used = True
            self.fallback_error = str(e)
            print(f"SNN algorithm failed: {e}")
            # Fallback to simple selection
            if self.all_anchor_points.shape[0] <= self.n_components:
                self.global_centers = self.all_anchor_points
            else:
                indices = np.random.choice(self.all_anchor_points.shape[0], self.n_components, replace=False)
                self.global_centers = self.all_anchor_points[indices]

    @property
    def cluster_centers(self) -> Optional[np.ndarray]:
        """Provides access to the computed global cluster centers."""
        return self.global_centers

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Assigns data points to nearest global cluster center."""
        if self.global_centers is None or self.global_centers.shape[0] == 0:
            return None

        if X is None or X.shape[0] == 0:
            return np.array([], dtype=int)

        distances = cdist(X, self.global_centers, metric='euclidean')
        return np.argmin(distances, axis=1)


    def train_one_round(self, round_num: int) -> None:
        started = time.perf_counter()
        selected_clients = self.clients_list
        client_anchor_points_list: List[Optional[np.ndarray]] = []
        for client_obj in selected_clients:
            anchors = client_obj.local_train()
            client_anchor_points_list.append(anchors)
        self.local_seconds = time.perf_counter() - started
        self.uplink_bytes = sum(a.nbytes for a in client_anchor_points_list if a is not None)
        started = time.perf_counter()
        self.init_global_centers(method="nnfc_dpc", client_anchor_points_list=client_anchor_points_list)
        self.aggregation_seconds = time.perf_counter() - started
        self.downlink_bytes = len(selected_clients) * self.global_centers.nbytes

