"""
FedSC Trainer — Federated Spectral Clustering pipeline.

Two-stage orchestration:
  Stage I  — Federated Similarity Reconstruction (S communication rounds)
  Stage II — Spectral Clustering on reconstructed similarity matrix
"""

import time

import numpy as np

from federated.fedsc.federated_spectral_clustering import FedSCClient, FedSCServer
from federated.fedsc.tracker import FedSCConvergenceTracker


# ============================================================================
# Main Training Function
# ============================================================================

def fedsc_clustering(X_train, y_train, X_test, y_test, X_list, n_clusters,
                     S=10, Q=20, eta=0.01, lambda_reg=0.01,
                     kernel='rbf', sigma=None, d=None,
                     active_ratio=1.0, dp_sigma=0.0,
                     k_neighbors=None, n_eigenvecs=None,
                     random_state=42, record_history=False, use_tracker=False,
                     return_communication=False, eigen_solver='auto',
                     return_similarity=True):
    """
    Federated Spectral Clustering via Similarity Reconstruction (FedSC).

    Implements the two-stage algorithm from:
      "Federated Spectral Clustering via Secure Similarity Reconstruction"
      NeurIPS 2023

    Args:
        X_train, y_train: Metrics data/labels in concatenated X_list order
        X_test, y_test: Test data and labels for metrics
        X_list: List of client data arrays (each Np x M)
        n_clusters: Number of clusters (K)
        S: Communication rounds for Stage I
        Q: Local iterations per client per round
        eta: Learning rate for dictionary update
        lambda_reg: Regularization for coefficient update
        kernel: Kernel type ('rbf')
        sigma: RBF kernel bandwidth (auto-estimated if None)
        d: Dictionary size (default: K * 5)
        active_ratio: Fraction of clients per round
        dp_sigma: Perturbation-only noise; no clipping/accountant or formal DP
        k_neighbors: KNN sparsification neighbors
        n_eigenvecs: Number of eigenvectors
        random_state: Random seed
        record_history: Record training history
        use_tracker: Enable convergence tracker

    Returns:
        labels: Cluster assignments
        (optional) K_reconstructed: Similarity matrix
        (optional) history: Training history
        (optional) tracker: Convergence tracker
    """
    training_start = time.perf_counter()
    if not X_list:
        raise ValueError("X_list must contain at least one client")
    if S < 1 or Q < 1 or n_clusters < 1 or not 0 < active_ratio <= 1:
        raise ValueError('rounds, iterations, clusters and active ratio must be valid')
    if not np.isfinite(dp_sigma) or dp_sigma < 0:
        raise ValueError('dp_sigma must be nonnegative')
    if not np.isfinite(eta) or eta <= 0 or not np.isfinite(lambda_reg) or lambda_reg < 0:
        raise ValueError('eta must be positive and lambda_reg nonnegative')
    if eigen_solver not in ('auto', 'dense', 'arpack'):
        raise ValueError('eigen_solver must be auto, dense or arpack')

    rng = np.random.RandomState(random_state)

    M = X_list[0].shape[1]
    P = len(X_list)
    N_total = sum(X.shape[0] for X in X_list)
    if n_clusters > N_total or any(X.ndim != 2 or X.shape[1] != M or len(X) == 0
                                   or not np.isfinite(X).all() for X in X_list):
        raise ValueError('clients must be nonempty, finite and dimension-compatible; K <= N')
    pooled = np.vstack(X_list)
    if use_tracker and X_train is not None and not np.array_equal(X_train, pooled):
        raise ValueError('FedSC tracker X_train must follow concatenated X_list order')
    if y_train is not None and len(y_train) != N_total:
        raise ValueError('FedSC y_train must follow concatenated X_list order')

    if d is None:
        d = n_clusters * 5
    if d < 1:
        raise ValueError('dictionary size must be positive')

    print(f"FedSC Configuration:")
    print(f"  Clients: {P}, Total samples: {N_total}, Features: {M}")
    print(f"  Clusters: {n_clusters}, Dictionary size: {d}")
    print(f"  Stage I rounds: {S}, Local iterations: {Q}")
    print(f"  Learning rate: {eta}, Regularization: {lambda_reg}")
    print(f"  Kernel: {kernel}, Active ratio: {active_ratio}")
    print(f"  Perturbation sigma (not formal DP): {dp_sigma}")

    # Initialize clients (FedSC uses M x Np transposed format)
    clients = [
        FedSCClient(client_id=i, X=X.T, dp_sigma=dp_sigma,
                     random_state=(random_state + i) % (2 ** 32))
        for i, X in enumerate(X_list)
    ]

    # Initialize server
    server = FedSCServer(K=n_clusters, M=M, d=d)

    # Auto-estimate kernel bandwidth
    auto_sigma = sigma is None
    if auto_sigma:
        sigma = server._estimate_kernel_bandwidth(clients, seed=random_state)
        print(f"  Auto-estimated sigma: {sigma:.4f}")

    # Initialize dictionary
    Z = server.initialize_dictionary(clients, d=d, seed=random_state)

    # Initialize tracker
    tracker = None
    if use_tracker:
        X_train_combined = (
            X_train if X_train is not None else np.vstack(X_list)
        )
        tracker = FedSCConvergenceTracker(
            X_train=X_train_combined, y_train=y_train,
            X_test=X_test, y_test=y_test
        )

    history = {
        'Z': [Z.copy()], 'Z_shift': [],
        'C_matrices': [], 'round_times': [], 'reconstruction_error': [],
        'learning_rate': [], 'label_order': 'concatenated X_list order'
    } if record_history else None
    setup_time = time.perf_counter() - training_start

    # =========================================================================
    # Stage I: Federated Similarity Reconstruction
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("STAGE I: FEDERATED SIMILARITY RECONSTRUCTION")
    print(f"{'=' * 70}")

    P_active = max(1, int(P * active_ratio))
    bandwidth_summary_bytes = sum(client.Np >= 2 for client in clients) * 8 if auto_sigma else 0
    uplink_bytes = bandwidth_summary_bytes  # Only clients with valid distance summaries upload.
    downlink_bytes = 0
    round_times = []
    stage_I_start = time.perf_counter()

    for s in range(S):
        round_start = time.perf_counter()

        # Select active clients
        if active_ratio < 1.0 - 1e-10:
            active_indices = rng.choice(P, P_active, replace=False)
        else:
            active_indices = list(range(P))
        downlink_bytes += len(active_indices) * Z.nbytes

        prev_Z = Z.copy()

        # Client updates
        Z_list = []
        C_list_round = []

        for idx in active_indices:
            client = clients[idx]
            C = client.update_C(Z, lambda_reg, kernel, sigma)
            C_list_round.append(C)
            Z_local = client.update_Z(
                Z_init=Z, C=C, Q=Q, eta=eta,
                kernel=kernel, sigma=sigma, lambda_reg=lambda_reg
            )
            Z_list.append(Z_local)
            uplink_bytes += Z_local.nbytes

        # Algorithm 1 averages participating dictionaries uniformly.
        Z = server.aggregate_Z(Z_list)

        round_time = time.perf_counter() - round_start
        round_times.append(round_time)
        Z_shift = np.linalg.norm(Z - prev_Z, ord='fro')
        objective = (float(np.mean([
            clients[idx].reconstruction_objective(Z, C, lambda_reg, kernel, sigma)
            for idx, C in zip(active_indices, C_list_round)]))
            if use_tracker or record_history else np.nan)

        print(f"  Round {s + 1}/{S}: Z shift={Z_shift:.6f}, Time={round_time:.2f}s")

        if use_tracker:
            comm_bytes = 2 * len(active_indices) * Z.nbytes
            C_active = [clients[idx].C for idx in active_indices
                        if clients[idx].C is not None]
            tracker.update_stage_I(
                round_idx=s + 1, Z=Z, prev_Z=prev_Z,
                clients=[clients[idx] for idx in active_indices],
                round_time=round_time, alpha=len(active_indices),
                communication_bytes=comm_bytes, C_list=C_active,
                reconstruction_error=objective,
                uplink_bytes=len(active_indices) * Z.nbytes,
                downlink_bytes=len(active_indices) * Z.nbytes, learning_rate=eta
            )

        if record_history:
            history['Z'].append(Z.copy())
            history['Z_shift'].append(Z_shift)
            history['round_times'].append(round_time)
            history['reconstruction_error'].append(objective)
            history['learning_rate'].append(float(eta))

    stage_I_time = time.perf_counter() - stage_I_start
    # Final C collection
    final_collection_start = time.perf_counter()
    downlink_bytes += P * Z.nbytes  # All clients need the FINAL aggregated dictionary.
    C_list_final = []
    for client in clients:
        C = client.update_C(Z, lambda_reg, kernel, sigma)
        # The paper's perturbed-factor variant perturbs C at final upload.
        if dp_sigma > 0:
            C = C + client._rng.normal(0, dp_sigma, C.shape)
        C_list_final.append(C)
        uplink_bytes += C.nbytes

    if record_history:
        history['C_matrices'] = C_list_final
    final_collection_time = time.perf_counter() - final_collection_start

    # =========================================================================
    # Stage II: Spectral Clustering
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("STAGE II: SPECTRAL CLUSTERING")
    print(f"{'=' * 70}")

    stage_II_start = time.perf_counter()

    labels, K_reconstructed = server.spectral_clustering(
        clients=clients, Z=Z, C_list=C_list_final,
        kernel=kernel, sigma=sigma,
        k_neighbors=k_neighbors, n_eigenvecs=n_eigenvecs,
        random_state=random_state, eigen_solver=eigen_solver,
        return_similarity=return_similarity
    )

    stage_II_time = time.perf_counter() - stage_II_start
    downlink_bytes += labels.nbytes  # Each label is returned only to its owner.

    print(f"Spectral clustering completed in {stage_II_time:.2f}s")
    print(f"  Unique labels: {np.unique(labels)}")
    print(f"  Label distribution: {np.bincount(labels)}")

    total_time = time.perf_counter() - training_start
    evaluation_start = time.perf_counter()
    if use_tracker:
        tracker.update_stage_II(
            labels=labels, similarity_matrix=K_reconstructed,
            stage_II_time=stage_II_time
        )
    evaluation_time = time.perf_counter() - evaluation_start
    counts = {
        'uplink_bytes': int(uplink_bytes), 'downlink_bytes': int(downlink_bytes),
        'rounds': int(S), 'setup_bytes': int(bandwidth_summary_bytes),
        'final_bytes': int(P * Z.nbytes + sum(C.nbytes for C in C_list_final)
                           + labels.nbytes),
        'total_bytes': int(uplink_bytes + downlink_bytes),
        'bytes_per_round': float(uplink_bytes + downlink_bytes) / S,
        'accounting': 'actual NumPy payload nbytes; no serialization, TLS or network framing',
        'encryption': 'none; kernel-factor reconstruction is not cryptographic aggregation',
    }
    performance = {
        'setup_time': setup_time, 'stage_I_time': stage_I_time,
        'final_collection_time': final_collection_time, 'stage_II_time': stage_II_time,
        'total_training_time': total_time, 'evaluation_time': evaluation_time,
    }
    if tracker is not None:
        tracker.communication = counts
        tracker.performance = performance
        tracker.spectral_diagnostics = {key: value for key, value in
            server.spectral_diagnostics.items() if key != 'eigenvectors'}

    if record_history:
        history['stage_II_time'] = stage_II_time
        history['labels'] = labels
        history['similarity_matrix'] = K_reconstructed
        history['performance'] = performance
        history['communication'] = counts
        history['spectral_diagnostics'] = {key: value for key, value in
            server.spectral_diagnostics.items() if key != 'eigenvectors'}

    # Summary
    print(f"\n{'=' * 70}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Stage I: {stage_I_time:.2f}s")
    print(f"  Stage II: {stage_II_time:.2f}s")
    print(f"  Final dictionary shift: {Z_shift:.6e}")
    print(f"{'=' * 70}\n")

    # Return
    output = (labels, K_reconstructed, history, tracker) if (
        use_tracker and record_history
    ) else ((labels, tracker) if use_tracker else (
        (labels, K_reconstructed, history) if record_history else labels
    ))
    if return_communication:
        return (*output, counts) if isinstance(output, tuple) else (output, counts)
    return output
