"""Out-of-sample self-consistency of the Nyström extension.

Acceptance criterion for the Nyström fix: take a point that was PART OF the
training graph, hand it to ``embed_out_of_sample`` as if it were a brand new
sample, and require the returned coordinates to match the in-sample embedding
it was assigned during training. If the query kernel is a different kernel from
the one the eigenvectors were computed with, this test fails loudly.

Two kernels are compared side by side:

* ``fixed_sigma``  the pre-fix query kernel, one global ``kernel_sigma`` for every
  landmark pair (reproduced here verbatim so the regression is quantified);
* ``self_tuning``  the fixed kernel, denom = scale_x * landmark_scales[j], the
  sample-out extension of Zelnik-Manor & Perona.

Run: python experiments/scripts/nystrom_selftest.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import schema  # noqa: E402
from federated.ours.aces_clustering import (  # noqa: E402
    AnchoredSpectralClient, AnchoredSpectralServer)


def _legacy_embedding(client, X_new, kernel_sigma, eps=1e-12):
    """The pre-fix out-of-sample kernel, kept only to quantify the regression.

    denom = 2 * sigma^2 for every landmark pair, plain exp() with no log-space
    stabilisation, and a hard raise when a row underflows to zero.
    """
    distances = cdist(np.asarray(X_new, dtype=float), client.landmark_orig, metric='euclidean')
    weights = np.exp(-distances ** 2 / (2.0 * float(kernel_sigma) ** 2))
    degrees_new = weights.sum(axis=1)
    if np.any(degrees_new <= eps):
        raise ValueError("A new sample has zero affinity to every landmark")
    mu = 1.0 - client.landmark_eigenvalues
    mu = np.where(np.abs(mu) < eps, np.copysign(eps, mu), mu)
    scaled = weights / np.sqrt(np.outer(degrees_new, client.landmark_degrees))
    embedding = (scaled @ client.landmark_embedding) / mu
    embedding = normalize(embedding, norm='l2', axis=1)
    if client.last_rotation is not None:
        embedding = embedding @ client.last_rotation
    if client.last_alignment_scale != 1.0:
        embedding = embedding * client.last_alignment_scale
    return embedding


def _relative_error(reference, estimate):
    """Row-wise, then aggregated, relative error between two embeddings."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    numerator = np.linalg.norm(estimate - reference, axis=1)
    denominator = np.linalg.norm(reference, axis=1)
    row = numerator / np.maximum(denominator, 1e-12)
    return {
        'mean': float(row.mean()),
        'median': float(np.median(row)),
        'p90': float(np.percentile(row, 90)),
        'max': float(row.max()),
        'frobenius': float(np.linalg.norm(estimate - reference)
                           / max(np.linalg.norm(reference), 1e-12)),
    }


def build_client(X, centers_orig, centers_emb, cfg, client_id=0):
    """One client whose landmark state has been populated exactly as at runtime."""
    client = AnchoredSpectralClient(
        id=client_id, X=X, anchor_weight=cfg['anchor_weight'], beta=cfg['lambda_space'],
        dp_sigma=0.0, graph_method=cfg['graph_method'], k_neighbors=cfg['k_neighbors'],
        kernel_sigma=cfg['kernel_sigma'], use_ckks=False, use_anchors=True,
        normalize_spaces=cfg['normalize_spaces'],
        kernel_scale_mode=cfg['kernel_scale_mode'],
        match_procrustes_scale=cfg['match_procrustes_scale'],
        random_state=cfg['random_state'])
    client.compute_dual_statistics(centers_orig, centers_emb, cfg['fuzziness'],
                                   use_procrustes=True)
    return client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--partition', default='digits_seed0_beta0.5_clients5')
    parser.add_argument('--d-emb', type=int, default=4)
    parser.add_argument('--fuzziness', type=float, default=2.0)
    parser.add_argument('--lambda-space', type=float, default=0.5)
    parser.add_argument('--kernel-sigma', type=float, default=None,
                        help='None -> data-adaptive median landmark distance')
    parser.add_argument('--k-neighbors', type=int, default=7)
    parser.add_argument('--kernel-scale-mode', default='self_tuning',
                        choices=['self_tuning', 'global'])
    parser.add_argument('--init-method', default='kmeans_plusplus')
    parser.add_argument('--init-seed', type=int, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-2)
    args = parser.parse_args(argv)

    directory = (args.partition if os.path.isdir(args.partition)
                 else os.path.join(schema.PARTITIONS_DIR, args.partition))
    meta, arrays = schema.load_partition(directory)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    client_indices = [np.asarray(i, dtype=int) for i in meta['client_indices']]
    num_clusters = int(meta['classes'])
    cfg = {
        'anchor_weight': 3.0, 'lambda_space': args.lambda_space, 'graph_method': 'knn',
        'k_neighbors': args.k_neighbors, 'kernel_sigma': args.kernel_sigma,
        'normalize_spaces': True, 'kernel_scale_mode': args.kernel_scale_mode,
        'match_procrustes_scale': True, 'fuzziness': args.fuzziness,
        'random_state': args.init_seed,
    }

    server = AnchoredSpectralServer(num_clusters, X_train.shape[1], args.d_emb, use_ckks=False)
    centers_orig, centers_emb = server.initialize_centers(
        args.init_method, X_train if args.init_method == 'kmeans_plusplus' else None,
        args.init_seed)

    print('partition   = %s' % os.path.basename(directory))
    print('K=%d d_orig=%d d_emb=%d | kernel_scale_mode=%s | kernel_sigma=%s'
          % (num_clusters, X_train.shape[1], args.d_emb, args.kernel_scale_mode,
             'adaptive' if args.kernel_sigma is None else args.kernel_sigma))

    rows = []
    for client_id, indices in enumerate(client_indices):
        X = X_train[indices]
        client = build_client(X, centers_orig, centers_emb, cfg, client_id=client_id)
        sigma_resolved = client._kernel_sigma_value(centers_orig)
        in_sample = client.training_embedding
        new_self_tuning = client.embed_out_of_sample(X)
        scale_x = client._oos_local_scales(
            cdist(X, client.landmark_orig, metric='euclidean'), sigma_resolved)
        rows.append({
            'client': client_id,
            'n': len(X),
            'sigma': sigma_resolved,
            'landmark_scale_median': (np.nan if client.landmark_scales is None
                                      else float(np.median(client.landmark_scales))),
            'scale_x_median': float(np.median(scale_x)),
            'fixed_sigma': _relative_error(in_sample, _legacy_embedding(
                client, X, sigma_resolved)),
            'self_tuning': _relative_error(in_sample, new_self_tuning),
        })

    print('\nresolved adaptive sigma = %.4f (median landmark pairwise distance)'
          % rows[0]['sigma'])
    print('landmark local scales (median) = %.4f | query local scale (median) = %.4f'
          % (np.median([r['landmark_scale_median'] for r in rows]),
             np.median([r['scale_x_median'] for r in rows])))
    print('\nrelative error of embed_out_of_sample(x_train) vs the in-sample embedding')
    print('%-7s %-6s | %-28s | %-28s'
          % ('client', 'n', 'fixed_sigma kernel (pre-fix)', 'self_tuning kernel (fixed)'))
    print('%-7s %-6s | %-28s | %-28s'
          % ('', '', 'mean/median/p90/max', 'mean/median/p90/max'))
    for row in rows:
        print('%-7d %-6d | %-28s | %-28s'
              % (row['client'], row['n'],
                 '%.3e/%.3e/%.3e/%.3e' % (row['fixed_sigma']['mean'], row['fixed_sigma']['median'],
                                          row['fixed_sigma']['p90'], row['fixed_sigma']['max']),
                 '%.3e/%.3e/%.3e/%.3e' % (row['self_tuning']['mean'], row['self_tuning']['median'],
                                          row['self_tuning']['p90'], row['self_tuning']['max'])))

    worst_mean = max(row['self_tuning']['mean'] for row in rows)
    worst_median = max(row['self_tuning']['median'] for row in rows)
    legacy_mean = max(row['fixed_sigma']['mean'] for row in rows)
    print('\nworst-client mean relative error: fixed_sigma=%.3e -> self_tuning=%.3e'
          % (legacy_mean, worst_mean))
    verdict = 'PASS' if worst_mean <= args.tolerance else 'FAIL'
    print('[%s] self-consistency tolerance %.1e (worst-client mean %.3e, median %.3e)'
          % (verdict, args.tolerance, worst_mean, worst_median))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
