import numpy as np
import time
from federated.ffcm.aggregation import average_statistics
from federated.ffcm.ffcm_clustering import Server, Client
from federated.ffcm.tracker import FFCMConvergenceTracker
from federated.utils.center_postprocess import check_and_fix_center_collapse, enforce_minimum_separation
from federated.utils.center_utils import initialize_centers


def ffcm_clustering(X_train, y_train, X_test, y_test, X_list, n_clusters, center_init_method='random', n_rounds=15, m=2.0, dp_sigma=0.0,
                    check_collapse=False, min_separation=0.15, random_state=42,
                    record_history=False,
                      use_tracker=False, use_ckks=False, return_cost=False):
    """
    Federated Fuzzy C-Means clustering using client-server architecture.

    Args:
        X_list: List of numpy arrays, where each array represents one client's data
        n_clusters: Number of clusters to form
        n_rounds: Number of federated learning rounds
        m: Fuzziness parameter (m > 1)
        random_state: Random seed for reproducibility
        dp_sigma: Standard deviation of Gaussian noise for differential privacy (0 for no noise)
        check_collapse: Whether to check for center collapse
        min_separation: Minimum separation between centers
        record_history: Whether to record the history of centers
        use_ckks: Pack and add native (S_k, n_k) statistics under the same
            CKKS parameters and public-aggregator/key-authority roles as Ours.
        return_cost: Return (centers, measured_cost) with history/tracker disabled.
            Existing calls retain their original return shape by default.

    Returns:
        centers: Final cluster centers
        history: List of centers at each round (if record_history is True)
    """
    started = time.perf_counter()
    if return_cost and (use_tracker or record_history):
        raise ValueError('Cost mode uses final-only offline evaluation; disable tracker/history')
    np.random.seed(random_state)
    # Optional measured transport; the native FFCM update below is unchanged.
    setup_started = time.perf_counter()
    setup_bytes = 0
    authority = public = None
    if use_ckks:
        from federated.ours.aces_clustering import CKKSEncryptionManager
        import tenseal as ts
        authority = CKKSEncryptionManager()
        public = authority.public_copy()
        setup_bytes = len(public.context_bytes) * (len(X_list) + 1)
    setup_seconds = time.perf_counter() - setup_started
    total_up, total_down, interservice = 0, setup_bytes, 0
    resources, round_times = [], []

    M = X_list[0].shape[1]

    # Initialize clients
    clients = [
        Client(id=i, X=X, dp_sigma=dp_sigma)
        for i, X in enumerate(X_list)
    ]

    server = Server(aggregation_fn=average_statistics)

    initial_centers = initialize_centers(
        n_clusters, X_list[0].shape[1], method=center_init_method,
        client_data=(np.vstack(X_list)
                     if center_init_method in ('client', 'kmeans_plusplus') else None),
        random_state=random_state)

    # Initialize tracker if requested
    tracker = None
    if use_tracker:
        tracker = FFCMConvergenceTracker(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        # Initial update (round 0)
        tracker.update(
            round_idx=0,
            centers=initial_centers,
            clients=clients,
            prev_centers=None,
            round_time=0,
        )

    # Prepare history tracking
    if record_history:
        history = {
            'centers': [initial_centers.copy()],
            'inertia': []
        }
    else:
        history = None

    centers = initial_centers.copy()

    # Federated training loop
    for round_idx in range(1, n_rounds+1):
        round_start_time = time.perf_counter()
        prev_centers = centers.copy()
        # Local computation at each client
        client_stats = []
        resource = dict(embedding=0., statistics=0., encryption=0., aggregation=0.,
                        decryption=0., topology=0.)
        encrypted_stats = []
        total_down += centers.nbytes * len(clients)
        for client in clients:
            local_started = time.perf_counter()
            S_k, n_k = client.local_computation(
                centers=centers,
                edge_index=None,  # Not used in basic FCM
                use_distance=True,
                fuzziness=m
            )

            # stat_list.append((S_k_noisy, n_k_noisy))
            client_stats.append((S_k, n_k))
            resource['statistics'] += time.perf_counter() - local_started
            if use_ckks:
                encrypt_started = time.perf_counter()
                chunks = public.encrypt_statistics(S_k, np.empty((n_clusters, 0)), n_k)
                resource['encryption'] += time.perf_counter() - encrypt_started
                encrypted_stats.append(chunks)
                total_up += sum(map(len, chunks))
            else:
                total_up += S_k.nbytes + n_k.nbytes

        # Aggregate statistics at the server
        aggregate_started = time.perf_counter()
        if use_ckks:
            chunks = []
            for column in range(len(encrypted_stats[0])):
                vectors = [ts.ckks_vector_from(public.context, value[column])
                           for value in encrypted_stats]
                aggregate = vectors[0]
                for vector in vectors[1:]:
                    aggregate = aggregate + vector
                chunks.append(aggregate.serialize())
            resource['aggregation'] += time.perf_counter() - aggregate_started
            decrypt_started = time.perf_counter()
            S_sum, _, n_sum = authority.decrypt_statistics(chunks, n_clusters, M, 0)
            resource['decryption'] = time.perf_counter() - decrypt_started
            interservice += sum(map(len, chunks)) + S_sum.nbytes + n_sum.nbytes
            aggregate_started = time.perf_counter()
            centers = server.aggregate([(S_sum, n_sum)])
        else:
            centers = server.aggregate(client_stats)
        resource['aggregation'] += time.perf_counter() - aggregate_started

        if check_collapse:
            centers, collapsed = check_and_fix_center_collapse(
                centers,
                min_distance_threshold=min_separation * 0.5,
                data_std=0.3
            )
            if collapsed:
                centers = enforce_minimum_separation(centers, min_separation=min_separation)

        round_time = time.perf_counter() - round_start_time
        resources.append(resource)
        round_times.append(round_time)

        if use_tracker:
            # Estimate communication (centers: K×M floats, 8 bytes each)
            comm_bytes = n_clusters * M * 8 * len(clients)

            tracker.update(
                round_idx=round_idx,
                centers=centers,
                clients=clients,
                prev_centers=prev_centers,
                round_time=round_time,
                alpha=len(clients),
                communication_bytes=comm_bytes
            )

        if record_history:
            history['centers'].append(centers.copy())

            # Compute total inertia
            total_inertia = 0
            total_N = 0
            for client in clients:
                distances = np.linalg.norm(
                    client.X[:, None, :] - centers[None, :, :], axis=2
                )
                min_distances = np.min(distances, axis=1)
                total_inertia += np.sum(min_distances ** 2)
                total_N += len(client.X)

            avg_inertia = total_inertia / total_N
            history['inertia'].append(avg_inertia)

    if return_cost:
        from federated.ours.aces_trainer import _communication_summary
        communication = _communication_summary(total_up, total_down, setup_bytes,
            interservice, use_ckks, n_rounds, n_clusters, M, 0, len(clients), 8192,
            len(public.context_bytes) if use_ckks else 0)
        communication['measurement_basis'] = (
            'serialized CKKS chunks + float64 broadcasts + public setup + aggregate/decryptor transfer'
            if use_ckks else 'NumPy float64 sufficient statistics + float64 broadcasts')
        return centers, dict(communication=communication, round_times=round_times,
            resource_breakdown=resources, execution_cost=dict(
                trainer_seconds=time.perf_counter() - started, setup_seconds=setup_seconds,
                evaluation_seconds=0., round_substage_seconds={
                    key: sum(row[key] for row in resources) for key in resources[0]}))
    if record_history:
        return centers, history
    else:
        return centers
