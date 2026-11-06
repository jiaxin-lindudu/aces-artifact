import numpy as np
import time

from federated.nnfc.nnfc_clustering import NNFCClient, NNFCServer


def federated_nnfc_clustering(
        X_train, y_train, X_test, y_test, X_list,
        num_clusters,
        dp_sigma=0.001,
        k_neighbors=2,
        random_state=42
):
    """
    Federated NNFC (Nearest Neighbor-based Federated Clustering) - One-Shot Method.

    NNFC is a one-shot federated clustering algorithm that:
    1. Each client generates local anchor points (cluster centers) using K-Means
    2. Server aggregates all anchor points and applies SNN clustering to find global centers
    3. No iterative rounds - completes in a single communication round

    Args:
        X_train: Training data (for evaluation reference, not used in training)
        y_train: Training labels (for evaluation, not used in training)
        X_test: Test data (for evaluation, not used in training)
        y_test: Test labels (for evaluation, not used in training)
        X_list: List of client datasets (each element is np.ndarray of shape [n_samples, n_features])
        num_clusters: Number of global clusters to find
        dp_sigma: Differential privacy noise level (epsilon parameter)
                  - If None or 0: no noise is added
                  - If > 0: Laplacian noise is added based on epsilon calculation
        random_state: Random seed for reproducibility

    Returns:
        centers_orig: Final cluster centers in original space (np.ndarray)
        centers_emb: None (NNFC works only in original space, no embedding)
        history: Dictionary with one key 'centers' containing list of centers (one-shot = one entry)
        tracker: None (NNFC doesn't use convergence tracking)
    """
    np.random.seed(random_state)
    training_started = time.perf_counter()

    # Handle edge case: empty client list
    if len(X_list) == 0:
        print("Warning: No client data provided")
        return np.array([]), None, {'centers': []}, None

    # Get data dimension
    d_orig = X_list[0].shape[1]

    # Initialize clients
    clients = []
    for i, X in enumerate(X_list):
        if X.shape[0] == 0:
            print(f"Warning: Client {i} has no data, skipping...")
            continue

        clients.append(
            NNFCClient(
                id=i,
                X=X,
                dp_epsilon=dp_sigma,
                random_state=random_state
            )
        )

    if len(clients) == 0:
        print("Warning: No valid clients after filtering empty datasets")
        return np.array([]), None, {'centers': []}, None

    # Initialize server
    server = NNFCServer(
        clients=clients,
        num_clusters=num_clusters,
        k_neighbors=k_neighbors
    )

    # One-shot training (single round)
    print(f"\n{'=' * 60}")
    print(f"NNFC One-Shot Federated Clustering")
    print(f"{'=' * 60}")
    print(f"Number of clients: {len(clients)}")
    print(f"Number of clusters: {num_clusters}")
    print(f"DP noise level (epsilon): {dp_sigma}")
    print(f"{'=' * 60}\n")

    server.train_one_round(round_num=0)

    # Check results
    if server.global_centers is None or len(server.global_centers) == 0:
        print("Warning: No global centers found by NNFC")
        return np.array([]), None, {'centers': []}, None

    print(f"✓ NNFC completed: Found {len(server.global_centers)} global centers")
    print(
        f"✓ Total anchor points collected: {len(server.all_anchor_points) if server.all_anchor_points is not None else 0}")

    # Create history for compatibility with evaluation pipeline
    # Note: NNFC is one-shot, so history contains only the final centers
    history = {
        'centers': [server.global_centers],
        'completed_rounds': 1,
        'communication': {
            'uplink_bytes': int(server.uplink_bytes),
            'downlink_bytes': int(server.downlink_bytes),
            'total_bytes': int(server.uplink_bytes + server.downlink_bytes),
            'rounds': 1,
            'measurement_basis': 'float64 local anchors uplink + global centers unicast downlink; excludes framing',
        },
        'local_seconds': server.local_seconds,
        'aggregation_seconds': server.aggregation_seconds,
        'training_seconds': time.perf_counter() - training_started,
        'fallback_used': server.fallback_used,
        'fallback_error': server.fallback_error,
        'privacy': {'noise_enabled': bool(dp_sigma is None or dp_sigma > 0),
                    'epsilon': dp_sigma, 'end_to_end_dp': False},
    }

    # Print client privacy information
    print(f"\n{'=' * 60}")
    print("Client Privacy Information")
    print(f"{'=' * 60}")
    for client in clients:
        info = client.get_privacy_info()
        print(f"Client {info['client_id']}: "
              f"epsilon={info['epsilon']:.4f}, "
              f"n_clusters={info['n_clusters']}, "
              f"data_size={info['data_size']}, "
              f"source={info['epsilon_source']}")
    print(f"{'=' * 60}\n")

    return server.global_centers, history
