"""Training entry point for the one-shot k-FED baseline."""
from __future__ import annotations

import time
import numpy as np

from federated.K_Fed.kfed_clustering import KFedClient, KFedServer
from federated.K_Fed.kfed_tracker import KFedTracker
from baselines.paper_common import validate_clients


def federated_kfed_clustering(X_train, y_train, X_test, y_test, X_list,
                              num_clusters, local_clusters=None,
                              random_state=42, use_tracker=True, train_indices=None,
                              implementation="author", max_lloyd_iter=300):
    """Run one communication round, independent of an outer experiment budget.

    ``y_train``/``y_test`` are used only by the tracker, never for local k,
    initialization or clustering. FIT predictions are the induced clustering;
    nearest-global-center prediction is an explicitly separate query proxy.
    """
    if not isinstance(num_clusters, (int, np.integer)):
        raise ValueError("num_clusters must be a positive integer")
    X_list = validate_clients(X_list, num_clusters)
    local_policy = "supplied known/configured local K"
    if local_clusters is None:
        # sqrt(K) is a separation-theorem condition, not an estimator of k(z).
        # Without known local counts use an explicit label-free upper bound;
        # this is a benchmark adaptation and not a claim that k(z) is known.
        local_clusters = num_clusters
        local_policy = "unknown local K: global-K upper bound (benchmark adaptation)"
    requested = ([local_clusters] * len(X_list) if np.isscalar(local_clusters)
                 else list(local_clusters))
    if (len(requested) != len(X_list) or any(not isinstance(k, (int, np.integer))
            or k < 1 or k > num_clusters for k in requested)):
        raise ValueError("supply one integer local K per client, each in [1, global K]")

    # ========================================================================
    # Prepare local problems without consulting target labels
    # ========================================================================
    tracker = KFedTracker() if use_tracker else None
    clients = [KFedClient(i, X, requested[i], random_state + i,
                         max_lloyd_iter=max_lloyd_iter, implementation=implementation)
               for i, X in enumerate(X_list)]
    total_started = time.perf_counter()
    # ========================================================================
    # Local stage and single upload of unweighted prototypes
    # ========================================================================
    local_started = time.perf_counter()
    results = [client.fit() for client in clients]
    local_time = time.perf_counter() - local_started
    upload = sum(r.centers.nbytes for r in results)

    # ========================================================================
    # Server cleanup and induced FIT assignments
    # ========================================================================
    server = KFedServer(num_clusters, random_state, implementation, max_lloyd_iter)
    aggregate_started = time.perf_counter()
    centers = server.aggregate(results)
    aggregation_time = time.perf_counter() - aggregate_started
    download = sum(m.nbytes for m in server.local_center_labels)
    pooled_labels = np.concatenate([server.induced_labels(i, r) for i, r in enumerate(results)])
    native_labels = None
    if X_train is not None:
        pooled_x = np.vstack(X_list)
        order = (np.arange(len(pooled_x)) if train_indices is None
                 else np.concatenate(train_indices))
        if (order.shape != (len(X_train),) or not np.issubdtype(order.dtype, np.integer)
                or not np.array_equal(np.sort(order), np.arange(len(X_train)))
                or not np.array_equal(pooled_x, np.asarray(X_train)[order])):
            raise ValueError("provide correct train_indices for non-pooled FIT order")
        native_labels = np.empty(len(pooled_labels), dtype=int)
        native_labels[order] = pooled_labels
    # Do not charge metric evaluation (including silhouette) as model training.
    training_time = time.perf_counter() - total_started
    communication = {
        "uplink_bytes": int(upload), "downlink_bytes": int(download),
        "total_bytes": int(upload + download), "logical_payload_bytes": int(upload + download),
        "rounds": 1, "setup_bytes": 0, "final_bytes": 0,
        "measurement_basis": "dense float64 centers uplink + int64 center-label mapping downlink; excludes framing/transport",
    }
    if tracker:
        tracker.local_stage_time = local_time
        tracker.server_stage_time = aggregation_time
        tracker.update(
            round_idx=1, centers_orig=centers,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            predictor=server.predict, train_labels=native_labels,
            stage="one_shot_" + implementation + "_server_lloyd",
            uplink_bytes=upload, downlink_bytes=download,
            statistics_time=local_time,
            aggregation_time=aggregation_time,
            round_time=training_time)
        tracker.communication = communication.copy()
        tracker.model = server
        tracker.metrics = tracker.final_metrics()

    history = {
        "centers": [centers.copy()], "centers_orig": [centers.copy()],
        "centers_emb": [], "round_times": [training_time], "completed_rounds": 1,
        "local_stage_time": local_time, "aggregation_time": aggregation_time,
        "local_center_labels": server.local_center_labels,
        "pooled_train_labels": pooled_labels, "train_labels": native_labels,
        "methodology": dict(implementation=implementation, local_k_policy=local_policy,
            requested_local_clusters=list(map(int, requested)),
            effective_local_clusters=[c.local_clusters for c in clients],
            local_lloyd_iterations=[r.iterations for r in results],
            local_converged=[r.converged for r in results],
            empty_stable_sets=[r.empty_stable_sets for r in results],
            server_lloyd_iterations=server.iterations, server_converged=server.converged,
            fit_label_rule="induced local-cluster to global-cluster mapping",
            query_label_rule="nearest global center; adaptation, not paper FIT clustering",
            safeguards="cap local K to distinct records; empty stable set uses Voronoi cell/nearest record",
            initialization="uncentered exact SVD; single D-squared kmeans++ draw sequence; no hidden multi-start",
            max_lloyd_iter=int(max_lloyd_iter),
            source_parity="author recipe, not RNG/bitwise parity; released code uses randomized SVD",
            source_license="no license found; inspected only, no source copied",
            reference="https://github.com/metastableB/kfed",
            scope="Plain one-shot; no cryptographic privacy or formal DP"),
        "communication": communication,
        "execution_cost": dict(training_seconds=training_time, local_seconds=local_time,
            aggregation_seconds=aggregation_time, completed_rounds=1),
    }
    return history, centers, server, tracker
