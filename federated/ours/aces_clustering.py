"""Core of the "Ours" method: anchored local spectral embedding + dual-space
fuzzy aggregation, with an optional CKKS transport and an optional uncalibrated
perturbation.

READ THIS BEFORE QUOTING ANY SECURITY OR PRIVACY CLAIM
------------------------------------------------------
Three different kinds of thing live in this file, and they must not be mixed up:

1. ALGORITHM (runs inside the federated protocol)
   - ``AnchoredSpectralClient._build_anchored_graph``: the local anchored graph.
   - ``_compute_anchored_embedding``: spectral embedding, sign-canonicalised.
   - ``_align_with_procrustes``: cross-client coordinate alignment.
   - ``fused_membership`` / ``fuzzy_sufficient_statistics``: Eq. (11) fusion and
     the fuzzy sufficient statistics actually uploaded.
   - ``AnchoredSpectralServer.aggregate_statistics`` /
     ``aggregate_encrypted_statistics`` / ``enforce_topology_consistency``.

2. SIMULATION, NOT A SECURITY BOUNDARY
   - ``CKKSEncryptionManager`` and everything that touches ``tenseal``. The key
     authority and the public aggregator are two objects in ONE process. This is
     aggregation-only single-process simulation, NOT threshold CKKS, NOT process
     isolation, and NOT a deployment-grade security boundary.
   - ``dp_sigma``: an uncalibrated Gaussian perturbation. It is NOT formal
     differential privacy: no sensitivity, no composition, no epsilon/delta.
   - Packet/byte accounting: sizes of serialized Python objects, no network.

3. OFFLINE EVALUATION, OUTSIDE THE PROTOCOL
   - ``embed_out_of_sample`` (Nyström extension) and any TEST-set scoring: no
     upload, no communication, not covered by CKKS, not protected by the
     perturbation. Never quote them as protocol-internal behaviour.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
from scipy.sparse import csgraph, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes
import time

from federated.utils.rng import isolated_rng

# Try to import TenSEAL for CKKS
try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not installed. CKKS encryption unavailable.")
    print("Install with: pip install tenseal")



def canonicalize_eigenvector_signs(vectors):
    """Fix the sign of each eigenvector column deterministically.

    LAPACK is free to return either sign of an eigenvector, and the choice can
    change with the BLAS thread count. Sign flips do not change the span, but
    they DO change the coordinates that downstream code reads, so they break
    reproducibility and make Procrustes alignment rotate by an arbitrary sign
    diagonal.

    Rule: flip each column so that its largest-magnitude entry is positive.
    Ties are resolved by position (np.argmax returns the first maximum), which
    is deterministic.

    LIMITATION, stated honestly: this removes only the sign freedom. Within a
    degenerate eigenspace LAPACK may still return a differently ROTATED basis,
    and no sign rule can fix that. If exact cross-machine reproducibility
    matters, run with a fixed thread count and verify with the controlled
    experiment in MULTISEED_IMPROVEMENTS.md.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.size == 0:
        return vectors
    pivot = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivot, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return vectors * signs


def blas_thread_environment():
    """Thread-control variables that can change the returned eigenbasis.

    Recorded in the manifest so a run can be reproduced or diagnosed later.
    """
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
    return {name: os.environ.get(name) for name in names}


def fuzzy_membership_from_squared_distances(distances, fuzziness=2.0):
    """FCM membership from squared distances, including exact zero distances."""
    distances = np.asarray(distances, dtype=float)
    if fuzziness <= 1 or distances.ndim != 2 or not np.isfinite(distances).all():
        raise ValueError("Finite squared distances and fuzziness > 1 are required")
    zeros = distances <= 0
    exact = zeros.any(axis=1)
    log_weights = -np.log(np.maximum(distances, 1e-300)) / (fuzziness - 1.0)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    membership = np.exp(log_weights)
    membership /= membership.sum(axis=1, keepdims=True)
    membership[exact] = zeros[exact] / zeros[exact].sum(axis=1, keepdims=True)
    return membership


def space_scale_normalizer(centers, eps=1e-12):
    """Mean pairwise squared distance among the cluster centers of one space.

    The normalizer deliberately depends only on the globally shared centers and
    never on client-local data, so every client derives the same constant and
    federated aggregation stays on a single consistent scale.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] < 2:
        return 1.0
    pair_sq = cdist(centers, centers, metric='sqeuclidean')
    mean_pair_sq = float(pair_sq.sum() / (centers.shape[0] * (centers.shape[0] - 1)))
    return mean_pair_sq if mean_pair_sq > eps else 1.0


def fused_membership(X, X_emb, centers_orig, centers_emb, fuzziness, lambda_space,
                     normalize_spaces=True, eps=1e-12):
    """Distance-level fusion: Eq. (11), not a mixture of two memberships.

    By default each squared-distance matrix is divided by the mean pairwise
    squared distance among the centers *of that space* before weighting. Without
    this the two terms live on different scales -- embedding rows are
    L2-normalized so their squared distances are bounded by 4, while
    original-space squared distances grow with d_orig -- and the nominal
    lambda_space no longer reflects the actual share of each space.
    """
    if not 0 <= lambda_space <= 1:
        raise ValueError("lambda_space must be in [0, 1]")
    distances_orig = cdist(X, centers_orig, metric='sqeuclidean')
    distances_emb = cdist(X_emb, centers_emb, metric='sqeuclidean')
    if normalize_spaces:
        distances_orig = distances_orig / space_scale_normalizer(centers_orig, eps)
        distances_emb = distances_emb / space_scale_normalizer(centers_emb, eps)
    distances = (lambda_space * distances_orig
                 + (1.0 - lambda_space) * distances_emb)
    return fuzzy_membership_from_squared_distances(distances, fuzziness)


def fuzzy_sufficient_statistics(X, X_emb, centers_orig, centers_emb, fuzziness,
                                lambda_space, normalize_spaces=True):
    """THE single Eq. (11) entry point: fused distances -> membership -> statistics.

    Every consumer of the dual-space model (a client computing its upload, the
    FIT-side evaluation of a round, the offline TEST scoring) must go through
    this function. It is the only place where the two spaces are combined, so
    the definition of "the" fuzzy membership cannot drift between call sites.

    Returns (U, S_orig, S_emb, n_k) with U the (N, K) membership matrix and
    S_/n_k the classic FCM sufficient statistics U**m weighted.
    """
    membership = fused_membership(X, X_emb, centers_orig, centers_emb, fuzziness,
                                  lambda_space, normalize_spaces=normalize_spaces)
    weighted = membership ** fuzziness
    return membership, weighted.T @ X, weighted.T @ X_emb, weighted.sum(axis=0)


# ============================================================================
# CKKS Encryption Manager
# ============================================================================

def require_tenseal():
    """Raise an actionable ImportError instead of a NameError deep in a call."""
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL is required for the CKKS path; install it with "
                          "`pip install tenseal`, or run with use_ckks=False")


class CKKSEncryptionManager:
    """
    Manages CKKS homomorphic encryption for secure aggregation.

    SIMULATION ONLY: the key authority and the public aggregator are objects in
    one process. This is not threshold CKKS and not a deployment security
    boundary; see the module docstring.
    """

    def __init__(self,
                 poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=(60, 40, 40, 60),
                 global_scale=2 ** 40):
        """
        Initialize CKKS encryption context.

        Args:
            poly_modulus_degree: Polynomial degree (4096, 8192, 16384)
                                Higher = more secure but slower
            coeff_mod_bit_sizes: Coefficient modulus chain
            global_scale: Encoding precision (2^40 is standard)
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL required for CKKS. Install with: pip install tenseal")

        print(f"Initializing CKKS encryption (poly_degree={poly_modulus_degree})...")

        # Create CKKS context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.global_scale = global_scale
        self.poly_modulus_degree = poly_modulus_degree
        # Only addition is used; rotation/relinearization keys are not distributed.

        print("CKKS context initialized successfully.")

    def public_copy(self):
        """Public-only aggregation role; never copy the secret key."""
        public = CKKSEncryptionManager.__new__(CKKSEncryptionManager)
        public.poly_modulus_degree = self.poly_modulus_degree
        public.context_bytes = self.context.serialize(
            save_public_key=True, save_secret_key=False,
            save_galois_keys=False, save_relin_keys=False)
        public.context = ts.context_from(public.context_bytes)
        if public.context.has_secret_key():
            raise ValueError("Aggregation context must not contain a secret key")
        return public

    def encrypt_statistics(self, S_orig, S_emb, n_k):
        """Flatten all sufficient statistics and pack consecutive CKKS slots."""
        values = np.concatenate((S_orig.ravel(), S_emb.ravel(), n_k.ravel()))
        slots = self.poly_modulus_degree // 2
        return [ts.ckks_vector(self.context, values[i:i + slots].tolist()).serialize()
                for i in range(0, len(values), slots)]

    def decrypt_statistics(self, chunks, K, d_orig, d_emb):
        """Only the separate key authority may decrypt the aggregate."""
        if not self.context.has_secret_key():
            raise ValueError("A public aggregation role cannot decrypt")
        values = np.concatenate([
            np.asarray(ts.ckks_vector_from(self.context, chunk).decrypt()) for chunk in chunks])
        expected = K * (d_orig + d_emb + 1)
        # A ciphertext always holds `slots` values, so the final chunk may be
        # zero-padded. Take the leading `expected` values and reject a short
        # result instead of demanding an exact length.
        if len(values) < expected:
            raise ValueError("Packed statistic length mismatch: got %d, expected at least %d"
                             % (len(values), expected))
        values = values[:expected]
        first = K * d_orig
        second = first + K * d_emb
        return (values[:first].reshape(K, d_orig),
                values[first:second].reshape(K, d_emb), values[second:expected])

    def evaluate_ciphertext_challenge(self, K, d_orig, d_emb, trials=200, seed=42, folds=5):
        """Known, equal-shape zero/one messages; serialized-byte heuristic, NOT IND-CPA proof."""
        from federated.ours.tracker import evaluate_attack_features
        if self.context.has_secret_key():
            raise ValueError("Ciphertext attacker must receive a public-only context")
        if trials < 2 * folds or trials % 2:
            raise ValueError("Use an even number of challenge samples, at least twice the fold count")
        rng = np.random.RandomState(seed)
        target = rng.permutation(np.tile([0, 1], trials // 2))
        features = []
        # Every message has exactly the actual sufficient-statistic packing shape.
        # Target bits are used by the challenger only, never as attack features.
        for bit in target:
            chunks = self.encrypt_statistics(np.full((K, d_orig), float(bit)),
                np.full((K, d_emb), float(bit)), np.full(K, float(bit)))
            serialized = b''.join(chunks)
            values = np.frombuffer(serialized, dtype=np.uint8)
            histogram = np.histogram(values, bins=16, range=(0, 256))[0] / len(values)
            features.append(np.r_[len(serialized), histogram, values.mean(), values.std()])
        result = evaluate_attack_features(features, target, seed, folds)
        result.update(trials=trials, message_pair='all-zero versus all-one equal-shape packed statistics',
            message_values=K * (d_orig + d_emb + 1), public_context_has_secret_key=False,
            attacker_view='serialized length, 16-bin byte histogram, byte mean/std; one public key',
            advantage=abs(result['balanced_accuracy'] - .5) if 'balanced_accuracy' in result else None,
            limitation='empirical metadata/byte-feature probe only; chance performance is NOT a security proof')
        return result

    # -- Generic row/vector helpers ------------------------------------------
    # Not used by the aggregation path above (which packs everything into as few
    # ciphertexts as possible); kept because they are part of this class's
    # published surface. They add nothing homomorphic beyond encryption itself.

    def encrypt_matrix(self, matrix: np.ndarray) -> list:
        """Encrypt a (K x d) matrix row by row. Requires the secret-key context."""
        return [ts.ckks_vector(self.context, row.tolist()) for row in matrix]

    def encrypt_vector(self, vector: np.ndarray) -> ts.CKKSVector:
        """Encrypt a single vector. Requires the secret-key context."""
        return ts.ckks_vector(self.context, vector.tolist())

    def decrypt_matrix(self, encrypted_rows: list, shape: tuple) -> np.ndarray:
        """Decrypt a list of encrypted rows and reshape to `shape`."""
        return np.array([row.decrypt() for row in encrypted_rows]).reshape(shape)

    def decrypt_vector(self, encrypted_vector: ts.CKKSVector) -> np.ndarray:
        """Decrypt a single encrypted vector."""
        return np.array(encrypted_vector.decrypt())

    def aggregate_encrypted_matrices(self, encrypted_matrices_list: list) -> list:
        """Homomorphically sum per-client encrypted matrices (lists of rows)."""
        if not encrypted_matrices_list:
            return []
        aggregated = encrypted_matrices_list[0]
        for enc_matrix in encrypted_matrices_list[1:]:
            aggregated = [agg_row + enc_row
                          for agg_row, enc_row in zip(aggregated, enc_matrix)]
        return aggregated

    def aggregate_encrypted_vectors(self, encrypted_vectors_list: list) -> ts.CKKSVector:
        """Homomorphically sum encrypted vectors; raises on an empty list."""
        if not encrypted_vectors_list:
            raise ValueError("At least one encrypted vector is required")
        aggregated = encrypted_vectors_list[0]
        for enc_vec in encrypted_vectors_list[1:]:
            aggregated = aggregated + enc_vec
        return aggregated


# ============================================================================
# Modified Client with CKKS Support
# ============================================================================

class AnchoredSpectralClient:
    def __init__(self, id, X, anchor_weight, beta, dp_sigma, graph_method, k_neighbors, kernel_sigma, use_ckks=False,
                 lambda_space=None, use_anchors=True, normalize_spaces=True,
                 kernel_scale_mode='self_tuning', match_procrustes_scale=True, random_state=None,
                 eigen_solver='dense'):
        if eigen_solver not in ('dense', 'partial'):
            raise ValueError("eigen_solver must be 'dense' or 'partial'")
        if kernel_scale_mode not in ('global', 'self_tuning'):
            raise ValueError("kernel_scale_mode must be 'global' or 'self_tuning'")
        if graph_method not in ('knn', 'fully_connected'):
            raise ValueError("Invalid graph method: {}".format(graph_method))
        X = np.asarray(X)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("Client data must be a nonempty finite 2D array")
        if k_neighbors < 1 or dp_sigma < 0:
            raise ValueError("k_neighbors >= 1 and dp_sigma >= 0 are required")
        # None means "data-adaptive": the median pairwise distance of the
        # landmarks is resolved on the first graph build, never hardcoded.
        if kernel_sigma is not None and (not np.isfinite(kernel_sigma) or kernel_sigma <= 0):
            raise ValueError("kernel_sigma must be a positive float, or None for the "
                             "data-adaptive median heuristic")
        if not np.isfinite([dp_sigma, anchor_weight]).all():
            raise ValueError("Client hyperparameters must be finite")
        effective_lambda = beta if lambda_space is None else lambda_space
        if not 0 <= effective_lambda <= 1:
            raise ValueError("lambda_space/beta must be in [0, 1]")
        self.id = id
        self.X = X
        self.n_local = len(X)
        self.d_orig = X.shape[1]
        self.dp_sigma = dp_sigma
        self.graph_method = graph_method
        self.k_neighbors = k_neighbors
        self.kernel_sigma = kernel_sigma
        self.anchor_weight = anchor_weight
        self.beta = beta if lambda_space is None else lambda_space
        self.lambda_space = self.beta
        self.use_anchors = use_anchors
        self.eigen_solver = eigen_solver
        self.training_embedding = None
        self.embedding_time = 0.0
        self.statistics_time = 0.0
        self.use_ckks = use_ckks
        # Own seed base for isolated streams; the counter keeps successive
        # perturbation draws independent without touching the global RNG state.
        self.random_state = 42 if random_state is None else int(random_state)
        self._dp_draw = 0

        # Dual-space fusion: divide each squared-distance matrix by the mean
        # pairwise center distance of its own space before applying lambda_space.
        self.normalize_spaces = normalize_spaces
        # 'global' uses one kernel_sigma for every edge; 'self_tuning' uses the
        # per-node local scales of Zelnik-Manor & Perona (NeurIPS 2004).
        self.kernel_scale_mode = kernel_scale_mode
        # Rescale the Procrustes-aligned embedding to Frobenius-match the target.
        self.match_procrustes_scale = match_procrustes_scale
        # Landmark (Nystrom) state, refreshed by every embedding computation.
        self.landmark_orig = None
        self.landmark_embedding = None
        self.landmark_eigenvalues = None
        self.landmark_degrees = None
        # Local (self-tuning) scale of every landmark, so the out-of-sample
        # kernel can be built with the SAME per-point scales as the training
        # kernel instead of one global sigma.
        self.landmark_scales = None
        # Landmark rows of the eigenvectors BEFORE row normalization; the
        # Nystrom formula is linear in those components and must not see the
        # row-normalized copy.
        self.landmark_embedding_raw = None
        self.last_rotation = None
        self.last_alignment_scale = 1.0
        # Resolved once, then frozen: the OOS kernel and the training kernel
        # must not drift apart between rounds.
        self._auto_kernel_sigma = None
        self._last_scales = None

        # Performance tracking (per-round)
        self.encryption_time = 0
        self.communication_bytes = 0

    def _kernel_sigma_value(self, landmarks=None):
        """Resolve ``kernel_sigma``; ``None`` means the data-adaptive heuristic.

        The hardcoded 5.0 was meaningless on unscaled 0-16 pixels and equally
        arbitrary after standardization, so the default is now the median
        pairwise distance of the landmarks (the anchors, or the local block when
        there is no anchor block). The value is resolved ONCE and then frozen:
        both the training kernel and the out-of-sample kernel read the same
        number, and it cannot drift between rounds.
        """
        if self.kernel_sigma is not None:
            return float(self.kernel_sigma)
        if self._auto_kernel_sigma is not None:
            return float(self._auto_kernel_sigma)
        reference = landmarks if landmarks is not None and len(landmarks) >= 2 else self.X
        value = 1.0
        if len(reference) >= 2:
            value = float(np.median(pdist(reference)))
        if not np.isfinite(value) or value <= 0:
            value = 1.0
        self._auto_kernel_sigma = value
        return value

    def _local_kernel_scales(self, distances, n_extended, kernel_sigma=None):
        """Per-node Gaussian kernel scales, or None when a global sigma is used.

        Self-tuning mode follows Zelnik-Manor & Perona (NeurIPS 2004): the scale
        of node i is its distance to the k-th nearest neighbour, and edge (i, j)
        is scaled by sigma_i * sigma_j so that dense and sparse regions of a
        non-IID client enter the affinity on comparable terms. A single global
        kernel_sigma cannot do this, because client densities differ.
        """
        if self.kernel_scale_mode != 'self_tuning' or n_extended < 2:
            return None
        k = min(self.k_neighbors, n_extended - 1)
        if k < 1:
            return None
        scales = np.empty(n_extended, dtype=float)
        for i in range(n_extended):
            others = np.delete(distances[i], i)
            scales[i] = np.partition(others, k - 1)[k - 1]
        # Duplicated samples yield a zero local scale; fall back to the global
        # kernel_sigma there rather than dividing by zero.
        fallback = self._kernel_sigma_value() if kernel_sigma is None else float(kernel_sigma)
        return np.where(scales > 1e-12, scales, fallback)

    def _oos_local_scales(self, distances, kernel_sigma=None):
        """Local scale of a query point: its distance to the k-th nearest landmark.

        The training local scale was the distance to the k-th nearest node of the
        extended graph; for a query point only the landmarks are available, so
        the same rule is applied to the landmark set. This is the sample-out
        extension of Zelnik-Manor & Perona and it is what keeps
        ``denom = scale_x * landmark_scales[j]`` identical in form to the
        training denominator ``scales[i] * scales[j]``.
        """
        fallback = self._kernel_sigma_value() if kernel_sigma is None else float(kernel_sigma)
        n_landmarks = distances.shape[1]
        if n_landmarks < 2:
            return np.full(len(distances), fallback, dtype=float)
        k = min(self.k_neighbors, n_landmarks)
        scales = np.partition(distances, k - 1, axis=1)[:, k - 1]
        return np.where(scales > 1e-12, scales, fallback)

    def _build_anchored_graph(self, anchors):
        K = len(anchors) if self.use_anchors else 0
        anchors = anchors if self.use_anchors else np.empty((0, self.d_orig))
        n_extended = self.n_local + K
        self.X_extended = np.vstack([self.X, anchors])
        distances = cdist(self.X_extended, self.X_extended, metric='euclidean')
        adjacency = np.zeros((n_extended, n_extended))
        sigma = self._kernel_sigma_value(anchors)
        scales = self._local_kernel_scales(distances, n_extended, sigma)
        global_denom = 2.0 * sigma ** 2
        # Kept so the landmark block of these scales can be reused verbatim by
        # the out-of-sample extension.
        self._last_scales = scales

        if self.graph_method == 'knn':
            k = min(self.k_neighbors, n_extended - 1)
            for i in range(n_extended):
                neighbors = [j for j in np.argsort(distances[i]) if j != i][:k]
                for j in neighbors:
                    denom = global_denom if scales is None else scales[i] * scales[j]
                    weight = np.exp(-distances[i, j] ** 2 / denom)
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight
        elif self.graph_method == 'fully_connected':
            for i in range(n_extended):
                for j in range(i + 1, n_extended):
                    denom = global_denom if scales is None else scales[i] * scales[j]
                    weight = np.exp(-distances[i, j] ** 2 / denom)
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight
        else:
            raise ValueError("Invalid graph method: {}".format(self.graph_method))

        anchor_indices = list(range(self.n_local, n_extended))
        for i in anchor_indices:
            for j in anchor_indices:
                if i != j:
                    denom = global_denom if scales is None else scales[i] * scales[j]
                    weight = np.exp(-distances[i, j] ** 2 / denom)
                    adjacency[i, j] = max(adjacency[i, j], weight * self.anchor_weight)
                    adjacency[j, i] = adjacency[i, j]

        for i in range(self.n_local) if K else []:
            anchor_distances = distances[i, self.n_local:]
            nearest_anchor_idx = np.argmin(anchor_distances)
            nearest_anchor = self.n_local + nearest_anchor_idx
            denom = global_denom if scales is None else scales[i] * scales[nearest_anchor]
            weight = np.exp(-anchor_distances[nearest_anchor_idx] ** 2 / denom)
            adjacency[i, nearest_anchor] = max(adjacency[i, nearest_anchor], weight * self.anchor_weight)
            adjacency[nearest_anchor, i] = adjacency[i, nearest_anchor]

        return adjacency

    def _compute_anchored_embedding(self, anchors, n_components, fixed_anchor_emb=None):
        K = len(anchors)
        n_extended = self.n_local + (K if self.use_anchors else 0)
        if n_components < 1 or n_components > n_extended:
            raise ValueError("d_emb=%d does not fit the local graph of %d nodes "
                             "(%d samples + %d anchors)"
                             % (n_components, n_extended, self.n_local,
                                K if self.use_anchors else 0))
        if self.eigen_solver == 'partial' and n_components == n_extended:
            raise ValueError("partial eigsh requires d_emb < local graph size; no dense fallback")
        adjacency = self._build_anchored_graph(anchors)
        laplacian = csgraph.laplacian(adjacency, normed=True)

        if hasattr(laplacian, 'toarray'):
            L = laplacian.toarray()
        else:
            L = laplacian

        try:
            if self.eigen_solver == 'dense':
                eigenvalues, eigenvectors = np.linalg.eigh(L)
            else:
                # Keep the same p lowest eigenvectors, INCLUDING the lowest one.
                # This changes only the solver: graph construction is still dense.
                start = isolated_rng('spectral_eigsh', self.random_state,
                                     self.id, n_extended, n_components).normal(size=n_extended)
                eigenvalues, eigenvectors = eigsh(
                    csr_matrix(L), k=n_components, which='SA', v0=start, tol=1e-8)
            idx = np.argsort(eigenvalues)[:n_components]
            selected_eigenvalues = eigenvalues[idx]
            embedding = eigenvectors[:, idx]
            embedding = canonicalize_eigenvector_signs(embedding)
            # Nystrom needs the EIGENVECTOR COMPONENTS, i.e. the rows before the
            # row-wise L2 normalization: the extension u(x) = (1/mu) sum_j
            # W(x,j)/sqrt(d(x)d_j) u_j is linear in u_j, so rescaling every
            # landmark row to unit norm first made the formula return a vector
            # that was not even parallel to the training embedding. The
            # row-normalized copy is still what clustering consumes; the raw one
            # is kept only to extend new samples into the same coordinates.
            embedding_raw = embedding
            embedding = normalize(embedding, norm='l2', axis=1)

            X_emb = embedding[:self.n_local]
            anchors_emb = embedding[self.n_local:]

            # Landmark (Nystrom) state: the K centers serve as landmarks and the
            # selected eigenvalues drive the out-of-sample extension. The degrees
            # are those of the landmark nodes inside this client's own graph.
            self.last_rotation = None
            self.last_alignment_scale = 1.0
            if self.use_anchors:
                self.landmark_orig = np.array(anchors, dtype=float, copy=True)
                self.landmark_eigenvalues = selected_eigenvalues
                self.landmark_degrees = adjacency.sum(axis=1)[self.n_local:]
                # Local scale of each landmark, taken from the very same scales
                # that built this graph. Without it the out-of-sample kernel
                # would silently switch to one global sigma and disagree with
                # the kernel the embedding was computed from.
                self.landmark_scales = (None if self._last_scales is None
                                        else np.asarray(self._last_scales, dtype=float)[self.n_local:])
                # Raw (pre-row-normalization) landmark rows of the eigenvectors.
                self.landmark_embedding_raw = np.array(
                    embedding_raw[self.n_local:], dtype=float, copy=True)
            else:
                # ours_no_anchors: the graph has no landmark block, so no
                # out-of-sample mapping can be defined.
                self.landmark_orig = None
                self.landmark_eigenvalues = None
                self.landmark_degrees = None
                self.landmark_scales = None
                self.landmark_embedding_raw = None

            if self.use_anchors and fixed_anchor_emb is not None:
                X_emb, anchors_emb = self._align_with_procrustes(X_emb, anchors_emb, fixed_anchor_emb)

            self.landmark_embedding = anchors_emb if self.use_anchors else None
            return X_emb, anchors_emb
        except np.linalg.LinAlgError as error:
            raise ValueError("Spectral eigendecomposition failed; no random fallback") from error

    def _align_with_procrustes(self, X_emb, anchors_emb, target_anchors_emb,
                               match_scale=None, eps=1e-12):
        """Orthogonal Procrustes alignment, optionally rescaled to Frobenius match.

        NOTE: scipy.linalg.orthogonal_procrustes returns (R, scale), where `scale`
        is the sum of the singular values of A^T B. It is NOT a multiplicative
        factor and must never be applied to the embedding. The isotropic rescaling
        used here is ||target||_F / ||aligned||_F instead.
        """
        if match_scale is None:
            match_scale = self.match_procrustes_scale
        R, _unused_scale = orthogonal_procrustes(anchors_emb, target_anchors_emb)
        X_emb_aligned = X_emb @ R
        anchors_emb_aligned = anchors_emb @ R
        factor = 1.0
        if match_scale:
            numerator = float(np.linalg.norm(target_anchors_emb))
            denominator = float(np.linalg.norm(anchors_emb_aligned))
            if numerator > eps and denominator > eps:
                factor = numerator / denominator
                X_emb_aligned = X_emb_aligned * factor
                anchors_emb_aligned = anchors_emb_aligned * factor
        # Remembered so that out-of-sample points are mapped into the same frame.
        self.last_rotation = R
        self.last_alignment_scale = factor
        return X_emb_aligned, anchors_emb_aligned

    def embed_out_of_sample(self, X_new, centers_orig=None, eps=1e-12):
        """Nystrom extension: map new samples into the existing spectral embedding.

        The K cluster centers act as landmarks. For the symmetric normalized
        Laplacian L = I - M with M = D^-1/2 W D^-1/2, an eigenpair (lambda, u) of
        L gives M u = mu u with mu = 1 - lambda, hence

            u(x) = (1 / mu) * sum_j  W(x, j) / sqrt(d(x) * d_j)  * u_j

        where d(x) is the affinity sum of the new sample and d_j the degree of
        landmark j in the client graph. The result is row-normalized and rotated
        exactly like the training embedding, so it shares its coordinates.

        The affinity itself uses the SAME kernel as the training graph. When the
        graph was built in 'self_tuning' mode the training edge (i, j) used the
        denominator sigma_i * sigma_j, so a query point must bring its own local
        scale sigma_x (its distance to the k-th nearest landmark) instead of
        falling back to one global kernel_sigma. Mixing the two made the query
        kernel a different kernel from the one the eigenvectors belong to, which
        is why the extension was unusable.

        OFFLINE EVALUATION ONLY: this runs outside the federated protocol, adds no
        communication, and is not covered by CKKS protection.
        """
        if self.landmark_orig is None or self.landmark_embedding is None:
            raise ValueError("No landmark embedding available; run an embedding round first")
        if self.landmark_eigenvalues is None or self.landmark_degrees is None:
            raise ValueError("Nyström state is incomplete (eigenvalues/degrees missing)")
        if self.landmark_embedding_raw is None:
            raise ValueError("Nyström state is incomplete (raw landmark eigenvectors missing)")
        if centers_orig is not None and not np.array_equal(
                np.asarray(centers_orig, dtype=float), self.landmark_orig):
            raise ValueError("Requested landmarks differ from the stored embedding landmarks")

        X_new = np.asarray(X_new, dtype=float)
        distances = cdist(X_new, self.landmark_orig, metric='euclidean')
        sigma = self._kernel_sigma_value(self.landmark_orig)

        # Plain Gaussian affinity to every landmark. The anchor_weight boost is a
        # graph-construction choice and is intentionally not part of the kernel.
        if self.landmark_scales is not None:
            # Zelnik-Manor & Perona out-of-sample extension: the query point
            # gets its own local scale, exactly as a training node did.
            scale_x = self._oos_local_scales(distances, sigma)
            denom = np.outer(np.maximum(scale_x, eps), np.maximum(self.landmark_scales, eps))
        else:
            denom = np.full(distances.shape, 2.0 * sigma ** 2, dtype=float)
        # Log-space: subtracting the row maximum before exp keeps the largest
        # weight at 1 and makes far-away landmarks underflow to 0 instead of
        # turning the whole row into a denormal/zero sum.
        log_weights = -(distances ** 2) / denom
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        degrees_new = weights.sum(axis=1)
        # Floor instead of raise: a zero row used to abort the whole TEST
        # evaluation. With the stabilized weights it cannot happen anymore, but
        # if it ever did the honest behaviour is to keep the row and let the
        # caller report it, not to drop every TEST label.
        zero_rows = degrees_new <= eps
        if np.any(zero_rows):
            warnings.warn("%d/%d query points have zero affinity to every landmark; "
                          "their degree is floored to eps" % (int(zero_rows.sum()), len(degrees_new)),
                          RuntimeWarning)
        degrees_new = np.maximum(degrees_new, eps)

        mu = 1.0 - self.landmark_eigenvalues
        mu = np.where(np.abs(mu) < eps, np.copysign(eps, mu), mu)
        scaled = weights / np.sqrt(np.outer(degrees_new, np.maximum(self.landmark_degrees, eps)))
        # Linear in the RAW eigenvector components; the row normalization that
        # the training embedding received is applied here, after the sum, so
        # both sides end up in the same coordinates.
        embedding = (scaled @ self.landmark_embedding_raw) / mu
        embedding = normalize(embedding, norm='l2', axis=1)
        if self.last_rotation is not None:
            embedding = embedding @ self.last_rotation
        if self.last_alignment_scale != 1.0:
            embedding = embedding * self.last_alignment_scale
        return embedding

    def compute_dual_statistics(self, centers_orig, centers_emb, fuzziness, use_procrustes=True):
        started = time.perf_counter()
        X_emb, anchors_emb = self._compute_anchored_embedding(
            centers_orig, centers_emb.shape[1],
            fixed_anchor_emb=centers_emb if use_procrustes and self.use_anchors else None)
        self.training_embedding = X_emb
        self.embedding_time = time.perf_counter() - started

        started = time.perf_counter()
        _U_fused, S_orig, S_emb, n_k = fuzzy_sufficient_statistics(
            self.X, X_emb, centers_orig, centers_emb, fuzziness, self.lambda_space,
            normalize_spaces=self.normalize_spaces)
        self.statistics_time = time.perf_counter() - started

        # No per-client anchor coordinates or sample counts are uploaded.
        anchor_stats = {
            'client_id': self.id,
            'alignment_error': (float(np.linalg.norm(anchors_emb - centers_emb))
                                if use_procrustes and self.use_anchors else None),
        }
        return S_orig, S_emb, n_k, anchor_stats

    def get_noisy_statistics(self, centers_orig, centers_emb, fuzziness, use_procrustes=True, add_noise=True):
        """Optional uncalibrated perturbation; this is NOT formal DP."""
        S_orig, S_emb, n_k, anchor_stats = self.compute_dual_statistics(
            centers_orig, centers_emb, fuzziness, use_procrustes)
        if add_noise and self.dp_sigma > 0:
            rng = isolated_rng('dp_noise', self.random_state, self.id, self._dp_draw)
            self._dp_draw += 1
            S_orig = S_orig + rng.standard_normal(S_orig.shape) * self.dp_sigma
            S_emb = S_emb + rng.standard_normal(S_emb.shape) * self.dp_sigma
            n_k = np.maximum(n_k + rng.standard_normal(n_k.shape) * self.dp_sigma * 0.1, 1e-6)
        self.communication_bytes = (
            S_orig.nbytes + S_emb.nbytes + n_k.nbytes
            + len(json.dumps(anchor_stats, sort_keys=True).encode('utf-8')))
        self.encryption_time = 0.0
        return S_orig, S_emb, n_k, anchor_stats

    def get_encrypted_statistics(self, centers_orig, centers_emb, fuzziness,
                                 ckks_manager, use_procrustes=True, add_noise=True):
        """Return serialized packed chunks, public shapes and scalar diagnostics.

        `add_noise` defaults to True, i.e. the CKKS path keeps the same
        perturbation policy as the plaintext path. It is harmless but redundant:
        the perturbation inside `get_noisy_statistics` is additionally gated on
        `dp_sigma > 0`, so with dp_sigma == 0 no noise is drawn on either path
        and the two paths contribute identical numbers. Passing add_noise=False
        makes "encryption only, no perturbation" explicit.
        """
        require_tenseal()
        # Same numerical contribution in Plain and CKKS paths.
        S_orig, S_emb, n_k, anchor_stats = self.get_noisy_statistics(
            centers_orig, centers_emb, fuzziness, use_procrustes, add_noise=add_noise)
        started = time.perf_counter()
        chunks = ckks_manager.encrypt_statistics(S_orig, S_emb, n_k)
        self.encryption_time = time.perf_counter() - started
        self.communication_bytes = (sum(map(len, chunks))
            + len(json.dumps(anchor_stats, sort_keys=True).encode('utf-8')))
        return chunks, S_orig.shape, S_emb.shape, anchor_stats


# ============================================================================
# Modified Server with CKKS Support
# ============================================================================

def _apply_topology_boundary(centers_orig_tensor, centers_emb_tensor, reference_orig,
                             reference_emb, boundary_mode, max_displacement):
    """Post-SGD boundary, applied in place.

    'tanh' squashes every coordinate, which rescales the whole space; 'clamp'
    bounds the displacement from the aggregated centers instead. Both are kept
    because which one is right is an empirical question.
    """
    import torch

    with torch.no_grad():
        if boundary_mode == 'tanh':
            centers_orig_tensor.copy_(torch.tanh(centers_orig_tensor))
            centers_emb_tensor.copy_(torch.tanh(centers_emb_tensor))
        else:
            centers_orig_tensor.copy_(torch.clamp(
                centers_orig_tensor, reference_orig - max_displacement,
                reference_orig + max_displacement))
            centers_emb_tensor.copy_(torch.clamp(
                centers_emb_tensor, reference_emb - max_displacement,
                reference_emb + max_displacement))


class AnchoredSpectralServer:
    #: Accepted by ``initialize_centers``. kmeans_plusplus needs pooled FIT data,
    #: which is why it is benchmark initialization and not a private protocol.
    INIT_METHODS = ('random', 'uniform', 'kmeans_plusplus')

    def __init__(self, num_clusters, d_orig, d_emb, use_ckks=False,
                 ckks_poly_degree=8192, ckks_coeff_mod_bit_sizes=(60, 40, 40, 60),
                 ckks_manager=None):
        # K >= 2 is a real restriction, not an arbitrary guard: with one cluster
        # the separation term of the topology objective and the minimum center
        # distance are both undefined.
        if num_clusters < 2 or d_orig < 1 or d_emb < 1:
            raise ValueError("num_clusters >= 2 and d_orig, d_emb >= 1 are required; "
                             "K=1 has no separation term and no inter-center distance")
        self.K = num_clusters
        self.d_orig = d_orig
        self.d_emb = d_emb
        self.centers_orig = None
        self.centers_emb = None
        self.alignment_errors = []
        self.use_ckks = use_ckks

        # Initialize CKKS encryption if needed
        if self.use_ckks:
            require_tenseal()
            if ckks_manager is None or ckks_manager.context.has_secret_key():
                raise ValueError("Provide a public-only CKKS aggregation context")
            self.ckks_manager = ckks_manager
        else:
            self.ckks_manager = None

        # Performance tracking
        self.aggregation_time = 0
        self.decryption_time = 0

    def initialize_centers(self, method, init_data=None, random_state=42):
        # No np.random.seed() reset here: resetting the global stream made every
        # later global draw depend on whether initialization ran.
        rng = isolated_rng('center_init', random_state)

        if method == 'random':
            self.centers_orig = rng.standard_normal((self.K, self.d_orig))
            self.centers_emb = rng.standard_normal((self.K, self.d_emb))
            self.centers_emb = normalize(self.centers_emb, norm='l2', axis=1)
        elif method == 'uniform':
            self.centers_orig = rng.uniform(0, 1.0, size=(self.K, self.d_orig))
            self.centers_emb = rng.uniform(0, 1.0, size=(self.K, self.d_emb))
            self.centers_emb = normalize(self.centers_emb, norm='l2', axis=1)
        elif method == 'kmeans_plusplus':
            from sklearn.cluster import kmeans_plusplus
            init_data = np.asarray(init_data, dtype=float)
            if init_data.ndim != 2 or init_data.shape[1] != self.d_orig or len(init_data) < self.K:
                raise ValueError("init_method='kmeans_plusplus' needs pooled FIT data of "
                                 "shape (n >= K, d_orig=%d)" % self.d_orig)
            self.centers_orig, _ = kmeans_plusplus(init_data, n_clusters=self.K, random_state=random_state)
            self.centers_emb = rng.standard_normal((self.K, self.d_emb)) * 0.3
            self.centers_emb = normalize(self.centers_emb, norm='l2', axis=1)
        else:
            raise ValueError("Unknown initialization method %r; use one of %s"
                             % (method, ', '.join(self.INIT_METHODS)))

        return self.centers_orig, self.centers_emb

    def aggregate_statistics(self, stats_list):
        """
        Aggregate statistics from clients (backward compatible with DP noise).

        Args:
            stats_list: List of (S_orig, S_emb, n_k, anchor_stats) tuples

        Returns:
            Updated centers
        """
        if not stats_list:
            raise ValueError("At least one client contribution is required")
        start_time = time.time()

        S_orig_total = sum(s[0] for s in stats_list)
        S_emb_total = sum(s[1] for s in stats_list)
        n_k_total = sum(s[2] for s in stats_list)
        n_k_total = np.maximum(n_k_total, 1e-6)

        self.centers_orig = S_orig_total / n_k_total[:, np.newaxis]
        self.centers_emb = S_emb_total / n_k_total[:, np.newaxis]
        self.centers_emb = normalize(self.centers_emb, norm='l2', axis=1)

        # Track alignment errors
        alignment_errors = []
        for s in stats_list:
            anchor_stats = s[3]
            if anchor_stats['alignment_error'] is not None:
                alignment_errors.append(anchor_stats['alignment_error'])

        if alignment_errors:
            avg_error = np.mean(alignment_errors)
            self.alignment_errors.append(avg_error)

        self.aggregation_time = time.time() - start_time

        return self.centers_orig, self.centers_emb

    def aggregate_encrypted_statistics(self, encrypted_stats_list):
        """Sum ciphertexts only; the server does not hold the decryption key."""
        if not self.use_ckks or not encrypted_stats_list:
            raise ValueError("Nonempty CKKS contributions are required")
        require_tenseal()
        started = time.perf_counter()
        count = len(encrypted_stats_list[0][0])
        if any(len(stats[0]) != count or stats[1] != (self.K, self.d_orig)
               or stats[2] != (self.K, self.d_emb) for stats in encrypted_stats_list):
            raise ValueError("Inconsistent packed statistic shapes")
        aggregated = []
        for column in range(count):
            vectors = [ts.ckks_vector_from(self.ckks_manager.context, stats[0][column])
                       for stats in encrypted_stats_list]
            result = vectors[0]
            for vector in vectors[1:]:
                result = result + vector
            aggregated.append(result.serialize())
        self.aggregation_time = time.perf_counter() - started
        return aggregated

    def enforce_topology_consistency(self, num_epochs, learning_rate, min_separation=0.1, random_state=42,
                                     min_sep_adjust_rate=0.1, min_sep_decay=0.9, min_sep_min=0.01,
                                     min_sep_max=1.0, patience=3,
                                     fidelity_weight=1.0, topology_weight=1.0, separation_weight=1.0,
                                     boundary_mode='tanh', max_displacement=1.0,
                                     early_stopping=True, early_stopping_patience=5):
        """
        Enforce topology consistency with adaptive minimum separation.

        The three loss terms were hard-coded at 1:1:1. They are now explicit
        arguments so the balance can be swept and reported instead of being an
        undocumented constant. Defaults reproduce the original behaviour.

        boundary_mode 'tanh' squashes every coordinate through tanh, which
        rescales the whole space rather than limiting how far the optimizer may
        move. 'clamp' instead bounds the displacement from the aggregated
        centers, which is the behaviour the fidelity term actually intends.
        'tanh' remains the default because the correct bound is an empirical
        question, not something to guess here.

        Args:
            num_epochs: Number of optimization epochs
            learning_rate: Learning rate for SGD
            min_separation: Initial minimum separation between centers
            random_state: Random seed for reproducibility
            min_sep_adjust_rate: Rate at which to adjust min_separation
            min_sep_decay: Decay factor for min_separation
            min_sep_min: Minimum allowed value for min_separation
            min_sep_max: Maximum allowed value for min_separation
            patience: Number of epochs to wait before adjusting min_separation
            fidelity_weight: weight of the distance-to-aggregated-centers term
            topology_weight: weight of the pairwise-distance consistency term
            separation_weight: weight of the anti-collapse separation term
            boundary_mode: 'tanh' (original) or 'clamp' (displacement limiter)
            max_displacement: bound used only by boundary_mode='clamp'
            early_stopping: stop the epoch loop once the loss stops improving
            early_stopping_patience: epochs without improvement before stopping

        Returns:
            Updated centers
        """
        if boundary_mode not in ('tanh', 'clamp'):
            raise ValueError("boundary_mode must be 'tanh' or 'clamp'")
        if num_epochs < 0:
            raise ValueError("num_epochs must be nonnegative")
        if self.K < 2:
            raise ValueError("Topology consistency needs K >= 2: the separation term "
                             "is a mean over pairs of centers and K=1 has none")
        import torch
        import torch.nn.functional as F

        # Make torch deterministic
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Convert to tensors. NOTE: these are float32, so the centers returned by
        # this method are float32 too. That is pre-existing behaviour and is left
        # alone on purpose: promoting them back to float64 would change every
        # published number.
        centers_orig_tensor = torch.tensor(self.centers_orig, dtype=torch.float32, requires_grad=True)
        centers_emb_tensor = torch.tensor(self.centers_emb, dtype=torch.float32, requires_grad=True)
        reference_orig = centers_orig_tensor.detach().clone()
        reference_emb = centers_emb_tensor.detach().clone()
        optimizer = torch.optim.SGD([centers_orig_tensor, centers_emb_tensor], lr=learning_rate)

        def topology_loss(orig, emb, separation):
            fidelity = F.mse_loss(orig, reference_orig) + F.mse_loss(emb, reference_emb)
            dist_orig = torch.cdist(orig, orig)
            dist_emb = torch.cdist(emb, emb)
            topology = F.mse_loss(dist_orig / (dist_orig.max() + 1e-6),
                                  dist_emb / (dist_emb.max() + 1e-6))
            mask = torch.triu(torch.ones_like(dist_orig), diagonal=1).bool()
            sep_loss = torch.relu(separation - dist_orig[mask]).mean()
            return (fidelity_weight * fidelity + topology_weight * topology
                    + separation_weight * sep_loss), sep_loss

        def apply_boundary():
            _apply_topology_boundary(centers_orig_tensor, centers_emb_tensor,
                                     reference_orig, reference_emb,
                                     boundary_mode, max_displacement)

        # Track best loss
        best_loss = float('inf')
        best_centers_orig = None
        best_centers_emb = None
        best_sep_loss = float('inf')
        no_improvement = 0
        stalled_epochs = 0
        epochs_run = 0
        current_min_sep = min_separation
        best_epoch = None

        for epoch in range(num_epochs):
            epochs_run = epoch + 1
            optimizer.zero_grad()

            # Keep the original fidelity / topology / separation SGD objective.
            loss, _ = topology_loss(centers_orig_tensor, centers_emb_tensor, current_min_sep)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Apply the boundary in-place, then score the resulting checkpoint
            apply_boundary()
            with torch.no_grad():
                candidate_orig = centers_orig_tensor.detach().clone()
                candidate_emb = torch.from_numpy(normalize(
                    centers_emb_tensor.detach().cpu().numpy(), norm='l2', axis=1))
                # Score exactly the state that would be returned, including normalization.
                candidate_loss, candidate_sep = topology_loss(candidate_orig, candidate_emb, current_min_sep)
                if best_centers_orig is not None:
                    # Adaptive separation changes the objective: compare on the same threshold.
                    incumbent_loss, incumbent_sep = topology_loss(
                        best_centers_orig, best_centers_emb, current_min_sep)
                    best_loss, best_sep_loss = incumbent_loss.item(), incumbent_sep.item()

            # Track best loss
            current_loss = candidate_loss.item()
            if not np.isfinite(current_loss):
                raise ValueError("Nonfinite topology checkpoint loss")
            if current_loss < best_loss:
                best_loss = current_loss
                best_centers_orig = candidate_orig.clone()
                best_centers_emb = candidate_emb.clone()
                best_sep_loss = candidate_sep.item()
                best_epoch = epoch + 1
                no_improvement = 0
                stalled_epochs = 0
            else:
                no_improvement += 1
                stalled_epochs += 1

            # Independent early stop. It keeps its own counter because the
            # min_separation schedule below deliberately resets no_improvement,
            # which would otherwise stall this check forever.
            if early_stopping and stalled_epochs >= early_stopping_patience:
                break

            # Adjust min_separation if separation loss is too low or too high
            if no_improvement >= patience:
                if best_sep_loss < 1e-3:  # Separation too large, can reduce min_sep
                    current_min_sep = max(min_sep_min, current_min_sep * min_sep_decay)
                else:  # Separation too small, increase min_sep
                    current_min_sep = min(min_sep_max, current_min_sep * (1 + min_sep_adjust_rate))
                no_improvement = 0
                best_sep_loss = float('inf')

        # Use best performing tensors
        if best_centers_orig is not None and best_centers_emb is not None:
            self.centers_orig = best_centers_orig.cpu().numpy()
            self.centers_emb = best_centers_emb.cpu().numpy()
        else:
            self.centers_orig = centers_orig_tensor.detach().cpu().numpy()
            self.centers_emb = centers_emb_tensor.detach().cpu().numpy()
            self.centers_emb = normalize(self.centers_emb, norm='l2', axis=1)

        with torch.no_grad():
            final_loss, _ = topology_loss(torch.from_numpy(self.centers_orig),
                                          torch.from_numpy(self.centers_emb), current_min_sep)
        self.topology_diagnostics = {
            'solver_version': 'post_update_projected_checkpoint_v2',
            'selected_epoch': best_epoch, 'epochs': num_epochs, 'epochs_run': epochs_run,
            'selected_loss': final_loss.item(), 'min_separation': current_min_sep,
            'loss_weights': {'fidelity': fidelity_weight, 'topology': topology_weight,
                             'separation': separation_weight},
            'boundary_mode': boundary_mode, 'max_displacement': max_displacement,
            'early_stopping': bool(early_stopping),
            'early_stopped': bool(early_stopping and epochs_run < num_epochs),
            'checkpoint_state': ('post-SGD tanh original centers + normalized spectral centers'
                                 if boundary_mode == 'tanh' else
                                 'post-SGD displacement-clamped original centers '
                                 '+ normalized spectral centers'),
        }
        return self.centers_orig, self.centers_emb


# Structural registration, NOT inheritance: the Ours classes are numerically
# validated and must not change. `register` makes them virtual subclasses of the
# protocol so `isinstance` works and a trainer can pick them by name.
from federated.protocols import register_protocol  # noqa: E402  (kept last: no cycle)

register_protocol('ours', AnchoredSpectralClient, AnchoredSpectralServer)
