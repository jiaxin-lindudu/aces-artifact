import numpy as np

from federated.fkm.federated_kmeans import KMeansClient, KMeansServer
from federated.fkm.tracker import KMeansConvergenceTracker
from federated.utils.center_utils import initialize_centers
import time


def fkm_clustering(X_train, y_train, X_test, y_test, X_list,
                   n_clusters,
                      center_init_method='random',
                      method='ma',
                      n_rounds=15,
                      dp_sigma=0.0,
                      random_state=42,
                      record_history=False,
                      Q=20,
                      B=40,
                      alpha=None,
                      beta=0.8,
                      weighted=True,
                      use_tracker=False):
    """
    Unified federated clustering function supporting three methods with tracking.

    Returns:
        centers: Final cluster centers
        history: Dictionary with training history (if record_history is True)
        tracker: KMeansConvergenceTracker instance (if use_tracker is True, only for MA/GS)
    """
    np.random.seed(random_state)

    P = len(X_list)
    M = X_list[0].shape[1]

    if alpha is None:
        alpha = P

    # Initialize K-means clients
    kmeans_clients = [
        KMeansClient(client_id=i, X=X, dp_sigma=dp_sigma)
        for i, X in enumerate(X_list)
    ]

    # Initialize server
    kmeans_server = KMeansServer(K=n_clusters, M=M)

    # Initialize centers
    if center_init_method == 'federated_kmeans++':
        centers = kmeans_server.federated_kmeans_plusplus_init(
            kmeans_clients, seed=random_state
        )
    else:
        centers = initialize_centers(n_clusters, M,
                                     method=center_init_method,
                                     random_state=random_state)

    # Initialize tracker if requested
    tracker = None
    if use_tracker:
        tracker = KMeansConvergenceTracker(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        # Initial update (round 0)
        tracker.update(
            round_idx=0,
            centers=centers,
            clients=kmeans_clients,
            prev_centers=None,
            round_time=0,
            Q=Q,
            alpha=alpha
        )

    # Prepare history tracking
    if record_history:
        history = {
            'centers': [centers.copy()],
            'inertia': []
        }
    else:
        history = None

    # Federated training loop
    for round_idx in range(1, n_rounds + 1):
        round_start_time = time.time()
        prev_centers = centers.copy()

        # Randomly select active clients
        if alpha < P:
            active_indices = np.random.choice(P, size=alpha, replace=False)
        else:
            active_indices = range(P)

        active_clients = [kmeans_clients[i] for i in active_indices]

        if method == 'ma':
            # ===== Model Averaging =====
            client_results = []
            for client in active_clients:
                centers_p, v_p = client.local_model_averaging(centers, Q=Q, B=B)
                client_results.append((centers_p, v_p))

            centers_bar = kmeans_server.aggregate_model_averaging(
                client_results, weighted=weighted
            )

            centers = centers + beta * (centers_bar - centers)

        elif method == 'gs':
            # ===== Gradient Sharing =====
            all_gradients = []
            for client in active_clients:
                gradients = client.local_gradient_sharing(centers, Q=Q, B=B)
                all_gradients.append(gradients)

            centers = kmeans_server.update_centers_with_gradient(
                centers, all_gradients, beta, B, len(active_clients)
            )

        round_time = time.time() - round_start_time

        # Update tracker
        if use_tracker:
            # Estimate communication (centers: K×M floats, 8 bytes each)
            comm_bytes = n_clusters * M * 8 * len(active_clients)

            tracker.update(
                round_idx=round_idx,
                centers=centers,
                clients=kmeans_clients,
                prev_centers=prev_centers,
                round_time=round_time,
                Q=Q,
                alpha=len(active_clients),
                communication_bytes=comm_bytes
            )

        # Record history
        if record_history:
            history['centers'].append(centers.copy())

            # Compute total inertia
            total_inertia = 0
            total_N = 0
            for client in kmeans_clients:
                distances = np.linalg.norm(
                    client.X[:, None, :] - centers[None, :, :], axis=2
                )
                min_distances = np.min(distances, axis=1)
                total_inertia += np.sum(min_distances ** 2)
                total_N += client.N

            avg_inertia = total_inertia / total_N
            history['inertia'].append(avg_inertia)

    # Return based on what was requested
    if use_tracker and record_history:
        return centers, history, tracker
    elif use_tracker:
        return centers, tracker
    elif record_history:
        return centers, history
    else:
        return centers