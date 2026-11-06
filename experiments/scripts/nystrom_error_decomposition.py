"""Why can the out-of-sample extension not reproduce the in-sample embedding?

Diagnostic only: nothing here is used by the shipped code path. It answers the
question behind the Nyström acceptance test -- is a large self-consistency error
caused by the query KERNEL, or by the choice of LANDMARKS?

The eigen-relation behind the extension is
    mu * u_i = sum_j W_ij / sqrt(d_i d_j) * u_j,   mu = 1 - lambda
and five ways of evaluating its right-hand side are compared:

    (A) sum over every node, true degree        -> sanity: must be ~1e-15
    (B) sum over the K anchors, true degree     -> cost of the landmark set
    (C) sum over the K anchors, landmark degree -> what the shipped code does
    (D) landmarks = local FIT rows + anchors    -> candidate fix A
    (E) every local node wired to every anchor  -> candidate fix B

Run: python experiments/scripts/nystrom_error_decomposition.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.sparse import csgraph
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import schema  # noqa: E402
from federated.ours.aces_clustering import (  # noqa: E402
    AnchoredSpectralClient, AnchoredSpectralServer, canonicalize_eigenvector_signs)


def median_relative_error(estimate, reference):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    row = (np.linalg.norm(estimate - reference, axis=1)
           / np.maximum(np.linalg.norm(reference, axis=1), 1e-12))
    return float(np.median(row))


def _eigendecomposition(adjacency, d_emb):
    laplacian = csgraph.laplacian(adjacency, normed=True)
    if hasattr(laplacian, 'toarray'):
        laplacian = laplacian.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    idx = np.argsort(eigenvalues)[:d_emb]
    return (eigenvalues[idx], canonicalize_eigenvector_signs(eigenvectors[:, idx]))


def _extend(weights, query_degree, landmark_degrees, landmark_rows, mu):
    scaled = weights / np.sqrt(np.outer(query_degree, landmark_degrees))
    return (scaled @ landmark_rows) / mu


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--partition', default='digits_seed0_beta0.5_clients5')
    parser.add_argument('--client', type=int, default=0)
    parser.add_argument('--d-emb', type=int, default=4)
    parser.add_argument('--init-seed', type=int, default=0)
    args = parser.parse_args(argv)

    directory = (args.partition if os.path.isdir(args.partition)
                 else os.path.join(schema.PARTITIONS_DIR, args.partition))
    meta, arrays = schema.load_partition(directory)
    X_train = np.asarray(arrays['X_train'], dtype=float)
    indices = np.asarray(meta['client_indices'][args.client], dtype=int)
    X = X_train[indices]
    K = int(meta['classes'])

    server = AnchoredSpectralServer(K, X_train.shape[1], args.d_emb)
    centers_orig, centers_emb = server.initialize_centers(
        'kmeans_plusplus', X_train, args.init_seed)
    client = AnchoredSpectralClient(
        id=args.client, X=X, anchor_weight=3.0, beta=0.5, dp_sigma=0.0,
        graph_method='knn', k_neighbors=7, kernel_sigma=None, use_ckks=False,
        use_anchors=True, normalize_spaces=True, kernel_scale_mode='self_tuning',
        match_procrustes_scale=True, random_state=args.init_seed)
    client.compute_dual_statistics(centers_orig, centers_emb, 2.0, use_procrustes=True)

    n_local = client.n_local
    adjacency = client._build_anchored_graph(centers_orig)
    eigvals, U = _eigendecomposition(adjacency, args.d_emb)
    mu = 1.0 - eigvals
    degrees = adjacency.sum(axis=1)
    U_local = U[:n_local]

    est_a = _extend(adjacency[:n_local, :], degrees[:n_local], degrees, U, mu)
    anchor_block = adjacency[:n_local, n_local:]
    landmark_degrees = degrees[n_local:]
    est_b = _extend(anchor_block, degrees[:n_local], landmark_degrees, U[n_local:], mu)
    est_c = _extend(anchor_block, anchor_block.sum(axis=1), landmark_degrees,
                    U[n_local:], mu)
    est_d = _extend(adjacency[:n_local, :], adjacency[:n_local, :].sum(axis=1),
                    degrees, U, mu)

    # (E) the same graph, but every local node is wired to every anchor.
    dense = adjacency.copy()
    distances = cdist(X, centers_orig, metric='euclidean')
    scale_local = client.landmark_scales
    scale_query = client._oos_local_scales(distances, client._kernel_sigma_value())
    for anchor in range(K):
        denom = np.maximum(scale_query, 1e-12) * np.maximum(scale_local[anchor], 1e-12)
        weight = np.exp(-(distances[:, anchor] ** 2) / denom) * 3.0
        dense[:n_local, n_local + anchor] = np.maximum(
            dense[:n_local, n_local + anchor], weight)
        dense[n_local + anchor, :n_local] = dense[:n_local, n_local + anchor]
    dense_degrees = dense.sum(axis=1)
    eigvals_e, U_e = _eigendecomposition(dense, args.d_emb)
    est_e = _extend(dense[:n_local, n_local:], dense[:n_local, n_local:].sum(axis=1),
                    dense_degrees[n_local:], U_e[n_local:], 1.0 - eigvals_e)

    print('client %d | %d local rows | K=%d d_emb=%d' % (args.client, n_local, K, args.d_emb))
    print('median relative error against the in-sample RAW eigenvector rows:')
    print('  (A) full sum, true degree                        = %.3e'
          % median_relative_error(est_a, U_local))
    print('  (B) anchor-only sum, true degree                 = %.3e'
          % median_relative_error(est_b, U_local))
    print('  (C) anchor-only sum, landmark degree  [shipped]  = %.3e'
          % median_relative_error(est_c, U_local))
    print('  (D) landmarks = local rows + anchors             = %.3e'
          % median_relative_error(est_d, U_local))
    print('  (E) every local node wired to every anchor       = %.3e'
          % median_relative_error(est_e, U_e[:n_local]))

    mass = anchor_block.sum(axis=1) / degrees[:n_local]
    print('\nshipped graph: share of a node affinity mass on the anchor block: '
          'median=%.3f p10=%.3f p90=%.3f'
          % (np.median(mass), np.percentile(mass, 10), np.percentile(mass, 90)))
    print('shipped graph: anchor neighbours per node: median=%.1f mean=%.2f'
          % (np.median((anchor_block > 0).sum(axis=1)),
             (anchor_block > 0).sum(axis=1).mean()))
    mass_dense = dense[:n_local, n_local:].sum(axis=1) / dense_degrees[:n_local]
    print('(E) graph:     share of a node affinity mass on the anchor block: median=%.3f'
          % np.median(mass_dense))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
