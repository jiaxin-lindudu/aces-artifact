"""Training orchestration for the "Ours" (FDCC) federated anchored spectral
clustering method.

ALGORITHM (in-protocol): local anchored spectral embedding; Eq. (11) dual-space
fusion via ``fuzzy_sufficient_statistics`` (the single definition); server
aggregation and topology-consistency refinement.

SIMULATION (never a security boundary): CKKS runs as key authority + public
aggregator objects in ONE process -- aggregation-only, NOT threshold CKKS and
NOT a deployment boundary. ``dp_sigma`` is an uncalibrated perturbation, NOT
formal DP. Byte counters are serialized object sizes, not network traffic.

OFFLINE EVALUATION (outside the protocol): FIT scoring is transductive; TEST
scoring uses a Nyström out-of-sample embedding or an original-space proxy.
Neither is uploaded nor covered by CKKS, and both must be reported as offline.

The Client/Server classes come from the ``federated.protocols`` registry;
'ours' is the default and behaves exactly like importing them directly.
"""
import random
import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from evaluation.metrics import compute_clustering_accuracy
from federated.ours.aces_clustering import (
    CKKSEncryptionManager,
    fuzzy_sufficient_statistics,
)
from federated.protocols import get_protocol, register_trainer
from federated.utils.rng import isolated_rng

#: Keys of every ``history['resource_breakdown']`` row. Declared once so the
#: execution-cost summary is defined even if no round ever ran.
RESOURCE_KEYS = ('embedding', 'statistics', 'encryption', 'aggregation',
                 'decryption', 'topology')

TEST_EMBEDDING_MODES = ('nystrom', 'original_space')
LANDMARK_MODES = ('strict', 'stored')


def set_all_seeds(seed):
    """Set initialization randomness without changing the data/partition seed."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cluster_labels(X, centers):
    """Original-space nearest-center proxy, NOT dual-space prediction."""
    return np.argmin(cdist(X, centers), axis=1)


def relative_center_shift(new, previous, eps=1e-12):
    """Frobenius displacement of the centers relative to their own scale.

    The raw displacement grows with the scale of the data, so an absolute
    threshold silently means different things on different datasets (and after
    any rescaling of the features). Normalizing by the norm of the previous
    centers turns the tolerance into one interpretable number.
    """
    new = np.asarray(new, dtype=float)
    previous = np.asarray(previous, dtype=float)
    return float(np.linalg.norm(new - previous) / (np.linalg.norm(previous) + eps))


def nystrom_test_labels(clients, X_test, centers_orig, centers_emb, fuzziness,
                        lambda_space, normalize_spaces=True, landmark_mode='strict'):
    """Offline dual-space TEST labels from per-client Nyström embeddings.

    The per-dimension median across clients is used because every client
    embedding was Procrustes aligned to the same shared centers_emb and
    therefore already lives in a common coordinate frame.

    landmark_mode
        'strict' (default): only clients whose stored landmarks are exactly
            ``centers_orig`` may extend. The trainer evaluates with the CURRENT
            centers while clients embedded against the PREVIOUS ones, so this
            set is normally empty and the caller falls back to the
            original-space proxy. Kept as the default so published numbers do
            not move silently; see the note in the refactor report.
        'stored': every client that holds an embedding extends with its own
            stored landmarks and the result is scored against the current
            centers. This is the out-of-sample extension the method describes.

    Returns (labels, U_test) or (None, None) when no client can extend.

    OFFLINE EVALUATION ONLY: computed outside the federated protocol, no upload,
    no communication, and not covered by CKKS protection.
    """
    if landmark_mode not in LANDMARK_MODES:
        raise ValueError("landmark_mode must be one of %s" % (LANDMARK_MODES,))
    centers_orig = np.asarray(centers_orig, dtype=float)
    usable = [client for client in clients
              if client.landmark_embedding is not None and client.landmark_orig is not None]
    if landmark_mode == 'strict':
        usable = [client for client in usable
                  if np.array_equal(client.landmark_orig, centers_orig)]
    if not usable:
        return None, None
    stacked = np.stack([client.embed_out_of_sample(X_test) for client in usable], axis=0)
    X_test_emb = np.median(stacked, axis=0)
    U_test, _S_orig, _S_emb, _n_k = fuzzy_sufficient_statistics(
        X_test, X_test_emb, centers_orig, centers_emb, fuzziness, lambda_space,
        normalize_spaces=normalize_spaces)
    return U_test.argmax(axis=1), U_test


def _validate_configuration(X_list, num_clusters, d_emb, num_rounds, fuzziness, beta,
                            lambda_space, active_ratio, kernel_sigma, k_neighbors,
                            dp_sigma, anchor_weight, topology_epochs, learning_rate,
                            tol_center_shift, early_stopping_patience, use_anchors,
                            run_privacy_eval, privacy_attack_folds, privacy_max_samples,
                            privacy_ciphertext_trials, test_embedding_mode,
                            topology_boundary_mode, topology_weights):
    """Reject impossible configurations early, with one message per problem."""
    if test_embedding_mode not in TEST_EMBEDDING_MODES:
        raise ValueError("test_embedding_mode must be one of %s" % (TEST_EMBEDDING_MODES,))
    if topology_boundary_mode not in ('tanh', 'clamp'):
        raise ValueError("topology_boundary_mode must be 'tanh' or 'clamp'")
    # None selects the data-adaptive median-landmark-distance heuristic; 5.0 was
    # a magic constant that meant nothing on either the raw or the scaled data.
    if kernel_sigma is not None and (not np.isfinite(kernel_sigma) or kernel_sigma <= 0):
        raise ValueError("kernel_sigma must be positive, or None for the data-adaptive heuristic")
    weights = np.asarray(topology_weights, dtype=float)
    if not np.isfinite(weights).all() or weights.min() < 0:
        raise ValueError("Topology loss weights must be finite and nonnegative")

    lambda_space = beta if lambda_space is None else lambda_space
    arrays = [np.asarray(X, dtype=float) for X in X_list]
    if not arrays or any(X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all() for X in arrays):
        raise ValueError("Clients must be nonempty finite 2D arrays")
    if len({X.shape[1] for X in arrays}) != 1:
        raise ValueError("Clients must have the same feature dimension")

    # K = 1 is a genuine restriction: the topology separation term averages over
    # pairs of centers and the minimum center distance is undefined.
    if num_clusters < 2:
        raise ValueError("num_clusters >= 2 is required: with K=1 the topology "
                         "separation term and the inter-center distance are undefined")
    if num_clusters > sum(map(len, arrays)):
        raise ValueError("num_clusters=%d exceeds the %d FIT samples"
                         % (num_clusters, sum(map(len, arrays))))
    smallest_graph = min(len(X) for X in arrays) + (num_clusters if use_anchors else 0)
    if d_emb < 1 or d_emb > smallest_graph:
        raise ValueError("d_emb=%d does not fit the smallest local graph of %d nodes "
                         "(samples + %s)" % (d_emb, smallest_graph,
                                             'anchors' if use_anchors else 'no anchors'))
    # (name, value, predicate, message) -> one specific message per bad scalar.
    checks = (
        ('num_rounds', num_rounds, lambda v: v >= 1, 'num_rounds must be >= 1'),
        ('fuzziness', fuzziness, lambda v: v > 1, 'fuzziness must be > 1'),
        ('lambda_space', lambda_space, lambda v: 0 <= v <= 1, 'lambda_space must be in [0, 1]'),
        ('active_ratio', active_ratio, lambda v: 0 < v <= 1, 'active_ratio must be in (0, 1]'),
        ('k_neighbors', k_neighbors, lambda v: v >= 1, 'k_neighbors must be >= 1'),
        ('dp_sigma', dp_sigma, lambda v: v >= 0, 'dp_sigma must be >= 0'),
        ('topology_epochs', topology_epochs, lambda v: v >= 0, 'topology_epochs must be >= 0'),
        ('learning_rate', learning_rate, lambda v: v > 0, 'learning_rate must be > 0'),
        ('tol_center_shift', tol_center_shift, lambda v: v >= 0, 'tol_center_shift must be >= 0'),
        ('early_stopping_patience', early_stopping_patience, lambda v: v >= 1,
         'early_stopping_patience must be >= 1'),
    )
    for name, value, predicate, message in checks:
        if not np.isfinite(value) or not predicate(value):
            raise ValueError("%s (got %r)" % (message, value))
    if not np.isfinite(anchor_weight):
        raise ValueError("anchor_weight must be finite (got %r)" % anchor_weight)
    if run_privacy_eval and (privacy_attack_folds < 2 or privacy_max_samples < 10
                             or privacy_ciphertext_trials < 2 * privacy_attack_folds
                             or privacy_ciphertext_trials % 2):
        raise ValueError("Invalid empirical privacy attack budget")
    if dp_sigma > 0:
        warnings.warn("dp_sigma is uncalibrated perturbation, NOT formal DP", UserWarning)
    return arrays, lambda_space


def _fit_memberships(clients, train_indices, centers_orig, centers_emb, fuzziness,
                     lambda_space, normalize_spaces, d_emb, use_procrustes, use_anchors,
                     fallback_centers_orig, fallback_centers_emb):
    """Transductive FIT memberships from each client's cached embedding.

    A client that did not take part in this round still has to be scored; its
    embedding is computed here against the centers of the previous round, which
    is offline evaluation only and triggers no upload.
    """
    total = sum(len(indices) for indices in train_indices)
    U_train = np.empty((total, len(centers_orig)), dtype=float)
    train_labels = np.empty(total, dtype=int)
    for client, indices in zip(clients, train_indices):
        if client.training_embedding is None:
            client.training_embedding, _ = client._compute_anchored_embedding(
                fallback_centers_orig, d_emb,
                fallback_centers_emb if use_procrustes and use_anchors else None)
        U_client, _S_orig, _S_emb, _n_k = fuzzy_sufficient_statistics(
            client.X, client.training_embedding, centers_orig, centers_emb,
            fuzziness, lambda_space, normalize_spaces=normalize_spaces)
        U_train[indices] = U_client
        train_labels[indices] = U_client.argmax(axis=1)
    return U_train, train_labels


def _test_labels(clients, X_test, centers_orig, centers_emb, fuzziness, lambda_space,
                 normalize_spaces, test_embedding_mode, landmark_mode):
    """Offline TEST labels; returns (labels, U_test, dual_space_used)."""
    if X_test is None:
        return None, None, False
    test_labels = U_test = None
    if test_embedding_mode == 'nystrom':
        test_labels, U_test = nystrom_test_labels(
            clients, X_test, centers_orig, centers_emb, fuzziness, lambda_space,
            normalize_spaces=normalize_spaces, landmark_mode=landmark_mode)
    dual_space_used = test_labels is not None
    if test_labels is None:
        # No client exposes usable landmarks (ours_no_anchors, or no client
        # embedded against these centers). Keep the original-space proxy rather
        # than dropping TEST labels; compare such runs with
        # test_embedding_mode='original_space' for a consistent basis.
        test_labels = get_cluster_labels(X_test, centers_orig)
        U_test = None
    return test_labels, U_test, dual_space_used


def _describe_prediction(test_embedding_mode, landmark_mode, dual_space_used):
    """Honest metadata for history['prediction']['test']."""
    if dual_space_used:
        return 'offline Nyström out-of-sample embedding + Eq.11 dual-space fusion'
    if test_embedding_mode == 'nystrom':
        return ('Nyström requested but no usable landmark (landmark_mode=%s): '
                'original-space nearest-center proxy fallback' % landmark_mode)
    return 'original-space proxy only; no dual-space query extension'


def federated_anchored_clustering(
        X_list, num_clusters, d_emb,
        X_train=None, y_train=None,
        X_test=None, y_test=None,
        active_ratio=1.0,
        num_rounds=50, fuzziness=2.0, beta=0.5, kernel_sigma=None,
        dp_sigma=0.0, enforce_topology=True, topology_epochs=20,
        use_procrustes=True, anchor_weight=3.0,
        graph_method='knn', k_neighbors=7,
        init_method='random', learning_rate=0.001,
        use_ckks=False,
        track_convergence=True,
        record_history=True,
        random_state=42,
        lambda_space=None, train_indices=None,
        tol_center_shift=1e-4, early_stopping_patience=3,
        use_anchors=True, run_privacy_eval=False,
        privacy_attack_folds=5, privacy_max_samples=1000, privacy_ciphertext_trials=200,
        ckks_poly_degree=8192, ckks_coeff_mod_bit_sizes=(60, 40, 40, 60),
        ckks_scale_bits=40,
        normalize_spaces=True, kernel_scale_mode='self_tuning',
        match_procrustes_scale=True, test_embedding_mode='nystrom',
        topology_fidelity_weight=1.0, topology_topology_weight=1.0,
        topology_separation_weight=1.0, topology_boundary_mode='tanh',
        topology_max_displacement=1.0, topology_early_stopping=True,
        topology_early_stopping_patience=5,
        test_landmark_mode='stored', protocol='ours', eigen_solver='dense'
):
    """Train the original local anchored graph with corrected Eq. (11) fusion.

    FIT uses cached local spectral coordinates and known ownership. TEST defaults
    to a Nyström out-of-sample embedding (test_embedding_mode='nystrom'); pass
    'original_space' for the previous nearest-center proxy. Either way TEST is
    OFFLINE evaluation: computed outside the federated protocol, no upload, no
    communication, and not covered by CKKS.
    CKKS roles are separate objects in one process, NOT threshold CKKS or a
    deployed security boundary. Pooled k-means++ initialization and diagnostic
    scalars are not protected by this aggregation-only simulation.

    New optional switches:
        test_landmark_mode : 'stored' (default) performs the out-of-sample
                             extension with each client's stored landmarks; this
                             is the extension the method describes. 'strict'
                             keeps the historical landmark match rule, which is
                             normally empty once topology displacement changes
                             the centers every round and therefore silently fell
                             back to the original-space proxy.
        protocol           : name in the ``federated.protocols`` registry used to
                             pick the Client/Server implementation.
        eigen_solver       : 'dense' preserves the original full eigendecomposition;
                             'partial' uses eigsh for the same lowest p coordinates,
                             including the lowest eigenvector. Graph storage remains dense.
    """
    started = time.perf_counter()
    arrays, lambda_space = _validate_configuration(
        X_list, num_clusters, d_emb, num_rounds, fuzziness, beta, lambda_space,
        active_ratio, kernel_sigma, k_neighbors, dp_sigma, anchor_weight,
        topology_epochs, learning_rate, tol_center_shift, early_stopping_patience,
        use_anchors, run_privacy_eval, privacy_attack_folds, privacy_max_samples,
        privacy_ciphertext_trials, test_embedding_mode, topology_boundary_mode,
        (topology_fidelity_weight, topology_topology_weight, topology_separation_weight))
    set_all_seeds(random_state)

    # Training rows retain their original owner and order.
    if X_train is None:
        X_train = np.vstack(arrays)
    X_train = np.asarray(X_train)
    if train_indices is None:
        if not np.array_equal(X_train, np.vstack(arrays)):
            raise ValueError("Provide train_indices for non-stacked FIT row order")
        offsets = np.cumsum([0] + [len(X) for X in arrays])
        train_indices = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(arrays))]
    train_indices = [np.asarray(indices, dtype=int) for indices in train_indices]
    if (len(train_indices) != len(arrays)
            or not np.array_equal(np.sort(np.concatenate(train_indices)), np.arange(len(X_train)))
            or any(not np.array_equal(X_train[indices], X) for indices, X in zip(train_indices, arrays))):
        raise ValueError("FIT ownership must match every local row exactly once")

    P = len(arrays)
    d_orig = arrays[0].shape[1]
    client_cls, server_cls = _resolve_protocol(protocol)
    clients = [
        client_cls(
            id=i, X=X, anchor_weight=anchor_weight, beta=lambda_space,
            dp_sigma=dp_sigma, graph_method=graph_method, k_neighbors=k_neighbors,
            kernel_sigma=kernel_sigma, use_ckks=use_ckks, use_anchors=use_anchors,
            normalize_spaces=normalize_spaces, kernel_scale_mode=kernel_scale_mode,
            match_procrustes_scale=match_procrustes_scale, random_state=random_state,
            eigen_solver=eigen_solver)
        for i, X in enumerate(arrays)]

    # Key authority and public aggregator never share a secret-key context.
    setup_started = time.perf_counter()
    key_authority = None
    public_manager = None
    setup_bytes = 0
    if use_ckks:
        key_authority = CKKSEncryptionManager(
            poly_modulus_degree=ckks_poly_degree,
            coeff_mod_bit_sizes=list(ckks_coeff_mod_bit_sizes),
            global_scale=2 ** ckks_scale_bits)
        public_manager = key_authority.public_copy()
        setup_bytes = len(public_manager.context_bytes) * (P + 1)
    setup_seconds = time.perf_counter() - setup_started
    server = server_cls(num_clusters, d_orig, d_emb, use_ckks=use_ckks,
                        ckks_manager=public_manager)

    # Retain benchmark k-means++ initialization; it is outside secure aggregation.
    all_data = X_train if init_method == 'kmeans_plusplus' else None
    centers_orig, centers_emb = server.initialize_centers(init_method, all_data, random_state)
    if centers_orig is None:
        raise ValueError("Unknown initialization method")

    history = {
        'centers_orig': [centers_orig.copy()], 'centers_emb': [centers_emb.copy()],
        'min_distances_orig': [], 'min_distances_emb': [], 'alignment_errors': [],
        'round_times': [], 'resource_breakdown': [],
        'topology_diagnostics': [],
        'aggregation_shifts_orig': [], 'aggregation_shifts_emb': [],
        'topology_shifts_orig': [], 'topology_shifts_emb': [],
    }
    tracker = None
    if track_convergence:
        from federated.ours.tracker import ConvergenceTracker
        tracker = ConvergenceTracker()

    total_uplink = 0
    total_downlink = setup_bytes
    interservice_bytes = 0
    evaluation_seconds = 0.0
    stable_rounds = 0
    first_stable_round = None
    stop_reason = 'round_limit'
    P_active = max(1, int(P * active_ratio))
    client_upload_totals = np.zeros(P)
    client_local_seconds = np.zeros(P)
    client_observations = np.zeros(P, dtype=int)
    test_dual_space_used = False

    for round_idx in range(num_rounds):
        round_start = time.perf_counter()
        if active_ratio < 1.0:
            # Own stream per round: the previous global-stream draw depended on
            # how many other numpy draws happened earlier in the process.
            active_indices = isolated_rng('active_clients', random_state, round_idx).choice(
                P, P_active, replace=False)
        else:
            active_indices = list(range(P))
        active_clients = [clients[i] for i in active_indices]
        prev_centers_orig = server.centers_orig.copy()
        prev_centers_emb = server.centers_emb.copy()
        downlink_bytes = (centers_orig.nbytes + centers_emb.nbytes) * len(active_clients)

        # =====================================================================
        # Client Phase: Compute sufficient statistics
        # =====================================================================
        stats_list = []
        for client in active_clients:
            if use_ckks:
                stats = client.get_encrypted_statistics(
                    centers_orig, centers_emb, fuzziness, public_manager, use_procrustes)
            else:
                stats = client.get_noisy_statistics(
                    centers_orig, centers_emb, fuzziness, use_procrustes, add_noise=(dp_sigma > 0))
            stats_list.append(stats)
            if run_privacy_eval:
                client_upload_totals[client.id] += client.communication_bytes
                client_local_seconds[
                    client.id] += client.embedding_time + client.statistics_time + client.encryption_time
                client_observations[client.id] += 1
        uplink_bytes = sum(client.communication_bytes for client in active_clients)

        # =====================================================================
        # Server Phase: Aggregate statistics
        # =====================================================================
        decryption_seconds = 0.0
        if use_ckks:
            chunks = server.aggregate_encrypted_statistics(stats_list)
            aggregation_seconds = server.aggregation_time
            decrypt_started = time.perf_counter()
            S_orig, S_emb, n_k = key_authority.decrypt_statistics(
                chunks, num_clusters, d_orig, d_emb)
            decryption_seconds = time.perf_counter() - decrypt_started
            round_interservice = sum(map(len, chunks)) + S_orig.nbytes + S_emb.nbytes + n_k.nbytes
            errors = [stats[3]['alignment_error'] for stats in stats_list
                      if stats[3]['alignment_error'] is not None]
            centers_orig, centers_emb = server.aggregate_statistics([
                (S_orig, S_emb, n_k, {'alignment_error': float(np.mean(errors)) if errors else None})])
            aggregation_seconds += server.aggregation_time
        else:
            centers_orig, centers_emb = server.aggregate_statistics(stats_list)
            aggregation_seconds = server.aggregation_time
            round_interservice = 0
        interservice_bytes += round_interservice
        total_uplink += uplink_bytes
        total_downlink += downlink_bytes
        server.aggregation_time = aggregation_seconds
        server.decryption_time = decryption_seconds

        # =====================================================================
        # Topology Consistency Enforcement
        # =====================================================================
        # Snapshot the aggregated centers BEFORE topology. The topology optimizer
        # can displace the centers far more than aggregation does, and folding
        # that displacement into the criterion is why nearly every run used to
        # stop only on the round limit.
        aggregated_centers_orig = centers_orig.copy()
        aggregated_centers_emb = centers_emb.copy()
        topology_started = time.perf_counter()
        if enforce_topology:
            centers_orig, centers_emb = server.enforce_topology_consistency(
                num_epochs=topology_epochs, learning_rate=learning_rate, random_state=random_state,
                fidelity_weight=topology_fidelity_weight, topology_weight=topology_topology_weight,
                separation_weight=topology_separation_weight, boundary_mode=topology_boundary_mode,
                max_displacement=topology_max_displacement,
                early_stopping=topology_early_stopping,
                early_stopping_patience=topology_early_stopping_patience)
            history['topology_diagnostics'].append(dict(server.topology_diagnostics, round=round_idx + 1))
        topology_seconds = time.perf_counter() - topology_started if enforce_topology else 0.0
        round_time = time.perf_counter() - round_start
        # Relative, and split by cause. Aggregation decides convergence; topology
        # displacement is recorded but must never hold the run open.
        shifts = (relative_center_shift(aggregated_centers_orig, prev_centers_orig),
                  relative_center_shift(aggregated_centers_emb, prev_centers_emb))
        topology_shifts = (relative_center_shift(centers_orig, aggregated_centers_orig),
                           relative_center_shift(centers_emb, aggregated_centers_emb))
        dist_orig = cdist(centers_orig, centers_orig)
        dist_emb = cdist(centers_emb, centers_emb)
        np.fill_diagonal(dist_orig, np.inf)
        np.fill_diagonal(dist_emb, np.inf)
        history['centers_orig'].append(centers_orig.copy())
        history['centers_emb'].append(centers_emb.copy())
        history['min_distances_orig'].append(float(dist_orig.min()))
        history['min_distances_emb'].append(float(dist_emb.min()))
        history['alignment_errors'].append(server.alignment_errors[-1] if server.alignment_errors else np.nan)
        history['round_times'].append(round_time)
        history['aggregation_shifts_orig'].append(shifts[0])
        history['aggregation_shifts_emb'].append(shifts[1])
        history['topology_shifts_orig'].append(topology_shifts[0])
        history['topology_shifts_emb'].append(topology_shifts[1])
        resource = {
            'embedding': sum(client.embedding_time for client in active_clients),
            'statistics': sum(client.statistics_time for client in active_clients),
            'encryption': sum(client.encryption_time for client in active_clients),
            'aggregation': aggregation_seconds, 'decryption': decryption_seconds,
            'topology': topology_seconds,
        }
        history['resource_breakdown'].append(resource)

        # =====================================================================
        # Metrics Computation: cached FIT embeddings; no query graph refitting
        # =====================================================================
        evaluation_started = time.perf_counter()
        U_train, train_labels = _fit_memberships(
            clients, train_indices, centers_orig, centers_emb, fuzziness, lambda_space,
            normalize_spaces, d_emb, use_procrustes, use_anchors,
            prev_centers_orig, prev_centers_emb)
        test_labels, U_test, round_dual_space = _test_labels(
            clients, X_test, centers_orig, centers_emb, fuzziness, lambda_space,
            normalize_spaces, test_embedding_mode, test_landmark_mode)
        test_dual_space_used = test_dual_space_used or round_dual_space

        if tracker is not None:
            tracker.update(
                round_idx=round_idx, server=server, clients=active_clients,
                X_train=X_train, y_train=y_train, U_train=U_train,
                X_test=X_test, y_test=y_test, U_test=U_test,
                compute_accuracy_fn=compute_clustering_accuracy, round_time=round_time,
                prev_centers_orig=prev_centers_orig, prev_centers_emb=prev_centers_emb,
                train_labels=train_labels, test_labels=test_labels,
                fuzziness_m=fuzziness,
                uplink_bytes=uplink_bytes, downlink_bytes=downlink_bytes,
                interservice_bytes=round_interservice, resource=resource)
        evaluation_seconds += time.perf_counter() - evaluation_started
        print(f"Round {round_idx + 1}/{num_rounds}: AggShift={shifts[0]:.6f}, "
              f"TopoShift={topology_shifts[0]:.6f}, "
              f"MinDist={dist_orig.min():.4f}, Time={round_time:.2f}s")
        if max(shifts) <= tol_center_shift:
            if first_stable_round is None:
                first_stable_round = round_idx + 1
            stable_rounds += 1
        else:
            stable_rounds = 0
        if stable_rounds >= early_stopping_patience:
            stop_reason = 'aggregation_relative_center_shift_patience'
            break

    communication = _communication_summary(
        total_uplink, total_downlink, setup_bytes, interservice_bytes, use_ckks,
        round_idx + 1, num_clusters, d_orig, d_emb, P, ckks_poly_degree,
        len(public_manager.context_bytes) if use_ckks else 0)
    history.update(
        completed_rounds=round_idx + 1, communication=communication,
        spectral_solver={'method': eigen_solver, 'includes_lowest_eigenvector': True,
                         'graph_storage': 'dense', 'dense_fallback': False,
                         'partial_which': 'SA', 'partial_tolerance': 1e-8,
                         'partial_start': 'isolated seed/client/graph-size stream'},
        train_labels=train_labels, test_labels=test_labels,
        stopping_configuration={'reason': stop_reason, 'tolerance': tol_center_shift,
                                'tolerance_basis': 'relative aggregation shift; topology displacement excluded',
                                'patience': early_stopping_patience, 'round_limit': num_rounds,
                                'first_stable_round': first_stable_round},
        prediction={'train': 'transductive FIT, known owners, cached local coordinates + Eq.11',
                    'test': _describe_prediction(test_embedding_mode, test_landmark_mode,
                                                 test_dual_space_used),
                    'test_embedding_mode': test_embedding_mode,
                    'test_landmark_mode': test_landmark_mode,
                    'test_dual_space_used': bool(test_dual_space_used),
                    'scope': 'TEST scoring is offline evaluation: no upload, no '
                             'communication, not covered by CKKS'},
        execution_cost={'trainer_seconds': time.perf_counter() - started,
                        'setup_seconds': setup_seconds, 'evaluation_seconds': evaluation_seconds,
                        # Sums over RESOURCE_KEYS, not over row[0]: a run with zero
                        # rounds must not raise IndexError.
                        'round_substage_seconds': {
                            name: sum(row.get(name, 0.0) for row in history['resource_breakdown'])
                            for name in RESOURCE_KEYS}},
        initialization={'method': init_method, 'uses_pooled_fit': init_method == 'kmeans_plusplus',
                        'protected_by_ckks': False,
                        'scope': 'benchmark initialization, NOT an end-to-end private protocol'},
        privacy_results={'formal_dp': 'not provided', 'perturbation_sigma': dp_sigma,
                         'initialization_uses_pooled_fit': init_method == 'kmeans_plusplus',
                         'end_to_end_privacy': False,
                         'model': ('public aggregator + separate key authority; non-collusion assumed; '
                                   'single-process aggregation-only simulation, NOT threshold CKKS'
                                   if use_ckks else 'plaintext aggregation')})
    privacy_started = time.perf_counter()
    if run_privacy_eval:
        from federated.ours.tracker import membership_inference_on_centers, size_leakage_diagnostics
        history['privacy_results']['mia'] = (membership_inference_on_centers(
            X_train, X_test, centers_orig, seed=random_state,
            max_samples=privacy_max_samples, folds=privacy_attack_folds)
                                             if X_test is not None else {'status': 'missing_nonmembers'})
        history['privacy_results']['size_leakage'] = size_leakage_diagnostics(
            list(map(len, arrays)), client_upload_totals, client_local_seconds, client_observations)
        history['privacy_results']['ciphertext'] = (public_manager.evaluate_ciphertext_challenge(
            num_clusters, d_orig, d_emb, trials=privacy_ciphertext_trials,
            seed=random_state, folds=privacy_attack_folds)
                                                    if use_ckks else {'status': 'not_applicable_plaintext'})
        history['privacy_results']['role_check'] = {
            'public_context_has_secret_key': bool(public_manager.context.has_secret_key()) if use_ckks else None,
            'key_authority_has_secret_key': bool(key_authority.context.has_secret_key()) if use_ckks else None,
            'scope': 'object/context capability check only; no threshold or process-isolation test'}
    else:
        history['privacy_results']['mia'] = {'status': 'disabled'}
        history['privacy_results']['ciphertext'] = {'status': 'disabled'}
    history['execution_cost']['privacy_seconds'] = time.perf_counter() - privacy_started
    if tracker is not None:
        tracker.communication = communication
    return centers_orig, centers_emb, history if record_history else None, tracker


def _resolve_protocol(protocol):
    """Client/Server classes for a registered protocol name."""
    spec = get_protocol(protocol)
    return spec.client_cls, spec.server_cls


def _communication_summary(total_uplink, total_downlink, setup_bytes, interservice_bytes,
                           use_ckks, rounds, num_clusters, d_orig, d_emb, num_clients,
                           ckks_poly_degree, public_context_bytes):
    """Byte accounting, with an explicit statement of what it does NOT cover."""
    communication = {
        'uplink_bytes': total_uplink, 'downlink_bytes': total_downlink,
        'setup_bytes': setup_bytes, 'interservice_bytes': interservice_bytes,
        'total_bytes': total_uplink + total_downlink + interservice_bytes,
        'bytes_reproducible': bool(not use_ckks),
        'bytes_reproducibility_note': (
            'TenSEAL key generation draws on system entropy, so ciphertext sizes vary '
            'between identically seeded runs; this is a library property, not '
            'instability of the algorithm' if use_ckks else
            'deterministic: bytes depend only on K, d and the round count'),
        'rounds': rounds, 'packed_values': num_clusters * (d_orig + d_emb + 1),
        'measurement_basis': ('serialized CKKS chunks + UTF-8 scalar diagnostics + '
                              'float64 broadcasts + public setup + aggregate/decryptor transfer'
                              if use_ckks else 'NumPy statistics + UTF-8 scalar diagnostics + float64 broadcasts'),
        'excluded': 'network framing, pooled initialization, final offline model delivery',
    }
    if use_ckks:
        slots = ckks_poly_degree // 2
        communication.update(
            public_context_bytes=public_context_bytes,
            context_recipients=num_clients + 1,
            slots_per_ciphertext=slots,
            ciphertexts_per_client_per_round=(communication['packed_values'] + slots - 1) // slots)
    return communication


register_trainer('ours', federated_anchored_clustering)
