"""Single-factor accuracy ablation on digits: why is ours below fkm?

One factor is changed at a time against a fixed base configuration, on one
frozen partition, averaged over the same init seeds. Every row carries the
stop_reason and the number of rounds actually executed, so "it stopped early"
can be told apart from "it converged to a bad solution".

The point is to produce NUMBERS for each of the following, not advice:

    a) topology_boundary_mode : tanh vs clamp   (+ the data scale it acts on)
    b) d_emb                  : 4 vs K vs 2K
    c) init_method            : random vs kmeans_plusplus
    d) lambda_space           : 0 (orig) vs 0.5 (fused) vs 1.0 (embedding only)
    e) enforce_topology       : True vs False
    f) num_rounds / tol       : 10 vs 25, and tol=0 (early stop disabled)

Run: python experiments/scripts/ablate_accuracy.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import schema  # noqa: E402
from experiments.baselines_adapter import run_baseline  # noqa: E402
from federated.ours.aces_trainer import (  # noqa: E402
    federated_anchored_clustering, set_all_seeds)
from federated.ours.tracker import clustering_quality  # noqa: E402

# The configuration the published sweep used; every variant differs in ONE key.
BASE = {
    'num_rounds': 10,
    'd_emb': 4,
    'fuzziness': 2.0,
    'lambda_space': 0.5,
    'init_method': 'kmeans_plusplus',
    'topology_boundary_mode': 'tanh',
    'enforce_topology': True,
    'tol_center_shift': 1e-4,
    'kernel_sigma': None,
    'test_embedding_mode': 'nystrom',
    'test_landmark_mode': 'stored',
}

DIAGNOSTIC_COLUMNS = (
    'study', 'variant', 'changed_key', 'changed_value', 'data_seed', 'init_seed',
    'acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari',
    'n_distinct_labels', 'center_spread_ratio',
    'stop_reason', 'rounds_run', 'first_stable_round', 'test_basis',
    'd_emb', 'lambda_space', 'fuzziness', 'init_method', 'boundary_mode',
    'enforce_topology', 'num_rounds', 'tol_center_shift', 'kernel_sigma',
    'kernel_scale_mode', 'test_embedding_mode', 'test_landmark_mode',
    'wall_seconds')


def data_scale(arrays):
    """Scale of the FIT block the topology/graph hyperparameters act on."""
    X = np.asarray(arrays['X_train'], dtype=float)
    return {
        'n': int(X.shape[0]), 'd': int(X.shape[1]),
        'min': float(X.min()), 'max': float(X.max()), 'mean': float(X.mean()),
        'std': float(X.std()), 'p99': float(np.percentile(X, 99)),
        'median_pairwise_distance': float(np.median(pdist(X))),
    }


def collapse_diagnostics(centers, train_labels, data_median_distance):
    """Detect the degenerate solution: centers collapsing onto the data mean.

    Fuzzy c-means with a large fuzziness in a high-dimensional space has a
    uniform-membership fixed point; once the memberships are uniform every
    center becomes the same weighted mean and the clustering stops separating
    anything. Two cheap proxies expose it: the number of distinct predicted
    labels, and the mean pairwise center distance relative to the spread of the
    data itself.
    """
    centers = np.asarray(centers, dtype=float)
    pairwise = (np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
                if len(centers) > 1 else np.array([0.0]))
    spread = float(pairwise[np.triu_indices(len(centers), 1)].mean()) if len(centers) > 1 else 0.0
    return {
        'n_distinct_labels': int(len(np.unique(train_labels))),
        'center_spread_ratio': float(spread / max(data_median_distance, 1e-12)),
    }


def run_ours(partition_dir, cfg, init_seed, data_median_distance=1.0):
    """One ours run; returns (metrics dict)."""
    meta, arrays = schema.load_partition(partition_dir)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    X_test, y_test = arrays['X_test'], arrays['y_test']
    client_indices = [np.asarray(i, dtype=int) for i in meta['client_indices']]
    X_list = [X_train[i] for i in client_indices]
    K = int(meta['classes'])

    set_all_seeds(init_seed)
    started = time.perf_counter()
    centers, _centers_emb, history, _tracker = federated_anchored_clustering(
        X_list, K, cfg['d_emb'], X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test, num_rounds=cfg['num_rounds'],
        fuzziness=cfg['fuzziness'], lambda_space=cfg['lambda_space'], beta=0.5,
        kernel_sigma=cfg['kernel_sigma'], k_neighbors=7, anchor_weight=3.0,
        topology_epochs=20, init_method=cfg['init_method'], use_ckks=False,
        track_convergence=True, random_state=init_seed, train_indices=client_indices,
        enforce_topology=cfg['enforce_topology'],
        topology_boundary_mode=cfg['topology_boundary_mode'],
        tol_center_shift=cfg['tol_center_shift'],
        kernel_scale_mode=cfg.get('kernel_scale_mode', 'self_tuning'),
        test_embedding_mode=cfg['test_embedding_mode'],
        test_landmark_mode=cfg['test_landmark_mode'],
        run_privacy_eval=False)
    wall = time.perf_counter() - started

    fit_metrics = clustering_quality(X_train, y_train, history['train_labels'])
    test_metrics = clustering_quality(X_test, y_test, history['test_labels'])
    prediction = history.get('prediction', {})
    stopping = history.get('stopping_configuration', {})
    result = {
        'acc': fit_metrics['acc'], 'nmi': fit_metrics['nmi'], 'ari': fit_metrics['ari'],
        'test_acc': test_metrics['acc'], 'test_nmi': test_metrics['nmi'],
        'test_ari': test_metrics['ari'],
        'stop_reason': stopping.get('reason'),
        'rounds_run': int(history['completed_rounds']),
        'first_stable_round': stopping.get('first_stable_round'),
        'test_basis': ('nystrom' if prediction.get('test_dual_space_used')
                       else ('original_space_proxy'
                             if prediction.get('test_embedding_mode') == 'nystrom'
                             else 'original_space')),
        'wall_seconds': wall,
    }
    result.update(collapse_diagnostics(centers, history['train_labels'],
                                       data_median_distance))
    return result


def run_reference(partition_dir, method, init_seed, num_rounds, fuzziness):
    """Federated baselines on the same frozen split, for the ours-vs-fkm gap."""
    meta, arrays = schema.load_partition(partition_dir)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    X_test, y_test = arrays['X_test'], arrays['y_test']
    client_indices = [np.asarray(i, dtype=int) for i in meta['client_indices']]
    X_list = [X_train[i] for i in client_indices]
    result = run_baseline(method, X_list, X_train, y_train, X_test, y_test,
                          int(meta['classes']), num_rounds=num_rounds,
                          random_state=init_seed, fuzziness=fuzziness)
    fit = clustering_quality(X_train, y_train, result['train_labels'])
    test = clustering_quality(X_test, y_test, result['test_labels'])
    return {'acc': fit['acc'], 'nmi': fit['nmi'], 'ari': fit['ari'],
            'test_acc': test['acc'], 'test_nmi': test['nmi'],
            'test_ari': test['ari'],
            'n_distinct_labels': int(len(np.unique(result['train_labels']))),
            'center_spread_ratio': np.nan,
            'stop_reason': 'n/a',
            'rounds_run': int(result['rounds']), 'first_stable_round': np.nan,
            'test_basis': 'original_space', 'wall_seconds': np.nan}


def build_variants(K):
    """One factor at a time; the base config is repeated as the reference row."""
    variants = [('base', 'none', 'base', {})]
    for value in ('tanh', 'clamp'):
        variants.append(('a_boundary_mode', 'topology_boundary_mode', value,
                         {'topology_boundary_mode': value}))
    for value in (4, K, 2 * K):
        variants.append(('b_d_emb', 'd_emb', value, {'d_emb': value}))
    for value in ('random', 'kmeans_plusplus'):
        variants.append(('c_init_method', 'init_method', value, {'init_method': value}))
    for value in (0.0, 0.5, 1.0):
        variants.append(('d_lambda_space', 'lambda_space', value, {'lambda_space': value}))
    for value in (True, False):
        variants.append(('e_enforce_topology', 'enforce_topology', value,
                         {'enforce_topology': value}))
    variants.append(('f_rounds', 'num_rounds', 25, {'num_rounds': 25}))
    variants.append(('f_tol', 'tol_center_shift', 0.0, {'tol_center_shift': 0.0}))
    # Extra g: does the Nyström branch ever actually fire?
    variants.append(('g_landmark_mode', 'test_landmark_mode', 'stored',
                     {'test_landmark_mode': 'stored'}))
    # Extra h: fuzziness. Centralized FCM with m=2 degenerates to a uniform
    # membership on this data (see sanity_ceiling), so m is a prime suspect for
    # "ours < fkm" and it is not covered by a-f.
    for value in (1.1, 1.5, 3.0):
        variants.append(('h_fuzziness', 'fuzziness', value, {'fuzziness': value}))
    # Extra i: the two suspects interact -- a lower m may only help if the
    # embedding space is also removed, so cross them.
    for fuzz in (1.1, 1.5):
        for lam in (0.0, 0.5, 1.0):
            variants.append(('i_m_x_lambda', 'm=%.1f,lambda=%.1f' % (fuzz, lam),
                             '%.1f/%.1f' % (fuzz, lam),
                             {'fuzziness': fuzz, 'lambda_space': lam}))
    return variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--partition', default='digits_seed0_beta0.5_clients5')
    parser.add_argument('--init-seeds', default='0,1,2')
    parser.add_argument('--references', default='fkm',
                        help='comma separated federated baselines to add for scale')
    parser.add_argument('--out-dir', default=os.path.join(schema.RESULTS_DIR, 'ablation'))
    parser.add_argument('--only', default=None,
                        help='comma separated variant prefixes to run (e.g. a,b)')
    args = parser.parse_args(argv)

    directory = (args.partition if os.path.isdir(args.partition)
                 else os.path.join(schema.PARTITIONS_DIR, args.partition))
    meta, arrays = schema.load_partition(directory)
    K = int(meta['classes'])
    init_seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    scale = data_scale(arrays)

    print('partition = %s | K=%d | FIT %sx%s | init seeds=%s'
          % (os.path.basename(directory), K, scale['n'], scale['d'], init_seeds))
    print('FIT scale after preprocessing: min=%.3f max=%.3f mean=%.3f std=%.3f '
          'p99=%.3f | median pairwise distance=%.3f'
          % (scale['min'], scale['max'], scale['mean'], scale['std'],
             scale['p99'], scale['median_pairwise_distance']))
    print('(raw digits pixels are 0..16; these numbers show which preprocessing '
          'actually reached the frozen file)')

    variants = build_variants(K)
    if args.only:
        keep = {v.strip() for v in args.only.split(',') if v.strip()}
        variants = [v for v in variants if v[0].split('_')[0] in keep]

    rows = []
    for method in [m.strip() for m in args.references.split(',') if m.strip()]:
        for init_seed in init_seeds:
            metrics = run_reference(directory, method, init_seed,
                                    BASE['num_rounds'], BASE['fuzziness'])
            rows.append(dict(study='reference', variant=method, changed_key='none',
                             changed_value='base', data_seed=meta['data_seed'],
                             init_seed=init_seed, d_emb=np.nan, lambda_space=np.nan,
                             fuzziness=BASE['fuzziness'], init_method='n/a',
                             boundary_mode='n/a', enforce_topology='n/a',
                             num_rounds=BASE['num_rounds'], tol_center_shift=np.nan,
                             kernel_sigma=np.nan, kernel_scale_mode='n/a',
                             test_embedding_mode='n/a', test_landmark_mode='n/a',
                             **metrics))
            print('  reference %-5s init=%d | acc=%.4f nmi=%.4f ari=%.4f'
                  % (method, init_seed, metrics['acc'], metrics['nmi'], metrics['ari']))

    for study, changed_key, changed_value, override in variants:
        cfg = dict(BASE)
        cfg.update(override)
        for init_seed in init_seeds:
            metrics = run_ours(directory, cfg, init_seed,
                               scale['median_pairwise_distance'])
            rows.append(dict(study=study, variant='%s=%s' % (changed_key, changed_value),
                             changed_key=changed_key, changed_value=changed_value,
                             data_seed=meta['data_seed'], init_seed=init_seed,
                             d_emb=cfg['d_emb'], lambda_space=cfg['lambda_space'],
                             fuzziness=cfg['fuzziness'], init_method=cfg['init_method'],
                             boundary_mode=cfg['topology_boundary_mode'],
                             enforce_topology=cfg['enforce_topology'],
                             num_rounds=cfg['num_rounds'],
                             tol_center_shift=cfg['tol_center_shift'],
                             kernel_sigma=('adaptive' if cfg['kernel_sigma'] is None
                                           else cfg['kernel_sigma']),
                             kernel_scale_mode=cfg.get('kernel_scale_mode', 'self_tuning'),
                             test_embedding_mode=cfg['test_embedding_mode'],
                             test_landmark_mode=cfg['test_landmark_mode'],
                             **metrics))
            print('  %-22s init=%d | acc=%.4f nmi=%.4f ari=%.4f | rounds=%d/%d (%s) '
                  'labels=%d spread=%.3f test=%s'
                  % (study, init_seed, metrics['acc'], metrics['nmi'], metrics['ari'],
                     metrics['rounds_run'], cfg['num_rounds'], metrics['stop_reason'],
                     metrics['n_distinct_labels'], metrics['center_spread_ratio'],
                     metrics['test_basis']))

    import pandas as pd
    os.makedirs(args.out_dir, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(DIAGNOSTIC_COLUMNS))
    out_path = os.path.join(args.out_dir, 'ablation_accuracy.csv')
    frame.to_csv(out_path, index=False)

    print('\n--- mean over init seeds ---')
    grouped = (frame.groupby(['study', 'variant'])[['acc', 'nmi', 'ari', 'test_acc',
                                                    'n_distinct_labels',
                                                    'center_spread_ratio',
                                                    'rounds_run']]
               .mean().round(4))
    print(grouped.to_string())
    print('\nstop reasons / test basis per variant:')
    print(frame.groupby(['study', 'variant'])[['stop_reason', 'test_basis']]
          .agg(lambda s: '; '.join(sorted(set(map(str, s))))).to_string())
    print('\nwrote %s (%d rows)' % (out_path, len(frame)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
