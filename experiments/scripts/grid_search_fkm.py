"""Grid search for federated K-means (FKM) hyperparameters on one federated partition.

STRUCTURE & FAIRNESS
--------------------
Mirrors grid_search_ours.py for fair comparison:
  * Same partition loading / FIT-block reconstruction / shared TEST block
  * Same selection signal: FIT acc/nmi/ari (TEST reported only)
  * Same grid modes: OAT (one-at-a-time) and cartesian
  * Same init-seed averaging and CSV schema

NOTE ON NAMING
--------------
The FKM trainer takes the learning rate as its ``beta`` argument, but the PARTITION
identity already owns the column name ``beta`` (Dirichlet concentration, from --beta).
To avoid a duplicate-column collision in the result frame, the tunable axis is named
``learning_rate`` here and mapped to ``beta=`` when the trainer is called.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.datasets_and_utils import load_federated_partition  # noqa: E402
from experiments.partition_checks import check_partition, PROVENANCE_COLUMNS
from experiments import schema  # noqa: E402
from federated.fkm.fkm_trainer import fkm_clustering  # noqa: E402
from federated.fkm.tracker import KMeansConvergenceTracker  # noqa: E402


# ======================================================================================
# HELPERS
# ======================================================================================

def set_all_seeds(seed):
    np.random.seed(seed)


def clustering_quality(X, y_true, y_pred):
    """Accuracy (Hungarian), NMI, ARI."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'acc': 0.0, 'nmi': 0.0, 'ari': 0.0}
    n_clusters = max(len(np.unique(y_true)), len(np.unique(y_pred))) + 1
    cm = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / len(y_true)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return {'acc': float(accuracy), 'nmi': float(nmi), 'ari': float(ari)}


# ======================================================================================
# THE GRID -- every tunable FKM hyper-parameter and its sweep range.
#   NOTE: learning rate is 'learning_rate' (mapped to the trainer's `beta=`),
#         so it never collides with the partition-identity column 'beta'.
# ======================================================================================
GRID = {
    # ---- method selection ----
    'method':              ['ma', 'gs'],          # Model Averaging vs Gradient Sharing

    # ---- federated optimization ----
    'n_rounds':            [50, 100, 150],          # Training rounds
    'Q':                   [5, 10, 20],           # Local iterations per client per round
    'B':                   [32, 64],              # Mini-batch size for local computation
    'alpha':               [None],                # None = all clients participate
    'learning_rate':       [0.5, 0.8, 1.0],       # -> trainer's `beta` (aggregation step)
    'weighted':            [True, False],         # cluster-count-weighted averaging

    # ---- initialization ----
    'center_init_method':  ['random', 'uniform'], # pooled kmeans++ excluded for privacy
}

ALLOW_POOLED_INIT = False
if ALLOW_POOLED_INIT:
    GRID['center_init_method'] = GRID['center_init_method'] + ['federated_kmeans++']

#: Result columns (same as ours for fair comparison)
RESULT_COLUMNS = (
    'acc', 'nmi', 'ari',
    'test_acc', 'test_nmi', 'test_ari',
    'n_distinct_labels', 'center_spread_ratio',
    'rounds_run', 'final_inertia',
    'init_uses_pooled_fit', 'wall_seconds',
)

GRID_PARAMS = tuple(GRID.keys())
GRID_COLUMNS = list(GRID_PARAMS) + list(RESULT_COLUMNS) + list(PROVENANCE_COLUMNS)

#: External fixed inputs, written to CSV for provenance only (NOT grid axes).
IDENTITY_COLUMNS = ('split', 'dataset', 'data_seed', 'beta', 'clients', 'init_seed')

FIXED_EVAL = {
    'dp_sigma': 0.0,
    'record_history': False,
    'use_tracker': True,
    # 'protocol': 'fkm',
}

SHOW_TRAINER_LOGS = False
MAX_RUNS_WITHOUT_FORCE = 2000


# ======================================================================================
# Small helpers
# ======================================================================================

def _slug(name):
    return ''.join(c if (c.isalnum() or c in '._-') else '_' for c in str(name))


def _freeze(cfg):
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()))


SHORT = {
    'method': 'meth', 'n_rounds': 'r', 'Q': 'Q', 'B': 'B', 'alpha': 'alpha',
    'learning_rate': 'lr', 'weighted': 'wgt', 'center_init_method': 'init',
}


def describe(cfg):
    return ' '.join('%s=%s' % (SHORT.get(k, k), cfg[k]) for k in GRID_PARAMS)


def print_grid_definition(mode):
    print('grid mode : %s' % mode)
    print('grid axes : %d tunable hyper-parameters' % len(GRID))
    for k in GRID_PARAMS:
        shown = GRID[k]
        print('  %-30s %-50s (default=%s)' % (k, shown, shown[0]))


# ======================================================================================
# Partition loading (same frozen Stage 2 partition as ours)
# ======================================================================================

_PARTITION_CACHE = {}


def load_partition_bundle(dataset, data_seed, beta, num_clients, base_dir):
    key = (dataset, int(data_seed), float(beta), int(num_clients), os.path.abspath(base_dir))
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    try:
        partition = load_federated_partition(dataset, data_seed, beta, num_clients,
                                             base_dir=base_dir, verbose=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            '%s\nGenerate it first:\n'
            '  python data/prepare_centralized_splits.py --datasets %s --data-seeds %d\n'
            '  python data/prepare_federated_partitions.py --datasets %s '
            '--data-seeds %d --betas %s --clients %d'
            % (error, dataset, data_seed, dataset, data_seed, float(beta), num_clients)
        ) from None

    metadata = partition.metadata
    n_classes = metadata.get('split', {}).get('n_classes')
    X_train = np.asarray(partition.X_train, dtype=float)
    X_test = np.asarray(partition.X_test, dtype=float)
    y_train = np.asarray(partition.y_train)
    client_indices = [np.asarray(indices, dtype=int) for indices in partition.client_indices]

    X_list = [X_train[indices] for indices in client_indices]

    X_fit = np.vstack(X_list)
    y_fit = np.concatenate([y_train[indices] for indices in client_indices])
    sizes = [len(X) for X in X_list]
    offsets = np.cumsum([0] + sizes)
    train_indices = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(sizes))]

    assert len(train_indices) == len(X_list)
    assert np.array_equal(np.sort(np.concatenate(train_indices)), np.arange(len(X_fit)))
    assert all(np.array_equal(X_fit[ind], X) for ind, X in zip(train_indices, X_list))

    bundle = {
        'dataset': dataset,
        'data_seed': int(data_seed),
        'beta': float(beta),
        'clients': int(num_clients),
        'label': '%s_seed_%d_beta_%s_clients_%d'
                 % (dataset, int(data_seed), float(beta), int(num_clients)),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': np.asarray(partition.y_test),
        'client_indices': client_indices,
        'X_list': X_list,
        'X_fit': X_fit,
        'y_fit': y_fit,
        'train_indices': train_indices,
        'K': int(n_classes) if n_classes else int(len(np.unique(y_train))),
        'median_distance': float(np.median(pdist(X_train))),
        'metadata': metadata,
    }
    bundle['provenance'] = check_partition(dataset, data_seed, beta, num_clients, base_dir)
    _PARTITION_CACHE[key] = bundle
    return bundle


def partition_summary(bundle):
    federated = bundle['metadata'].get('federated', {})
    split_meta = bundle['metadata'].get('split', {})
    sizes = [len(indices) for indices in bundle['client_indices']]
    return ('%s | K=%d | clients=%d | FIT=%d TEST=%d | client sizes [%d, %d] avg %.1f'
            % (bundle['label'], bundle['K'], bundle['clients'],
               split_meta.get('n_train', len(bundle['X_train'])),
               split_meta.get('n_test', len(bundle['X_test'])),
               federated.get('min_size', min(sizes)),
               federated.get('max_size', max(sizes)),
               federated.get('mean_size', float(np.mean(sizes)))))


# ======================================================================================
# One FKM run driven by a single grid config
# ======================================================================================

def run_fkm(bundle, cfg, init_seed):
    X_fit, y_fit = bundle['X_fit'], bundle['y_fit']
    X_test, y_test = bundle['X_test'], bundle['y_test']
    X_list = bundle['X_list']
    K = bundle['K']
    median_distance = bundle['median_distance']

    # Map the grid axis 'learning_rate' -> trainer's `beta`; never leak 'learning_rate'.
    kwargs = dict(cfg)
    learning_rate = kwargs.pop('learning_rate')
    kwargs.update(FIXED_EVAL)
    kwargs.update({'random_state': init_seed, 'n_clusters': K, 'beta': learning_rate})

    set_all_seeds(init_seed)
    started = time.perf_counter()
    sink = io.StringIO() if not SHOW_TRAINER_LOGS else sys.stdout
    with contextlib.redirect_stdout(sink):
        result = fkm_clustering(
            X_train=X_fit, y_train=y_fit,
            X_test=X_test, y_test=y_test,
            X_list=X_list,
            **kwargs
        )
    wall = time.perf_counter() - started

    if kwargs.get('use_tracker') and kwargs.get('record_history'):
        centers, history, tracker = result
    elif kwargs.get('use_tracker'):
        centers, tracker = result
    elif kwargs.get('record_history'):
        centers, history = result
    else:
        centers = result
        tracker = None

    fit_labels = np.argmin(
        np.linalg.norm(X_fit[:, None, :] - centers[None, :, :], axis=2), axis=1)
    fit = clustering_quality(X_fit, y_fit, fit_labels)

    test_labels = np.argmin(
        np.linalg.norm(X_test[:, None, :] - centers[None, :, :], axis=2), axis=1)
    test = clustering_quality(X_test, y_test, test_labels)

    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    spread = float(pairwise[np.triu_indices(len(centers), 1)].mean()) if len(centers) > 1 else 0.0

    final_inertia = (np.sum((X_fit - centers[fit_labels]) ** 2) / len(X_fit)
                     if len(X_fit) > 0 else 0.0)

    init_uses_pooled = cfg.get('center_init_method') == 'federated_kmeans++'

    return {
        'acc': fit['acc'], 'nmi': fit['nmi'], 'ari': fit['ari'],
        'test_acc': test['acc'], 'test_nmi': test['nmi'], 'test_ari': test['ari'],
        'n_distinct_labels': int(len(np.unique(fit_labels))),
        'center_spread_ratio': float(spread / max(median_distance, 1e-12)),
        'rounds_run': int(cfg.get('n_rounds', 10)),
        'final_inertia': float(final_inertia),
        'init_uses_pooled_fit': bool(init_uses_pooled),
        'wall_seconds': wall,
    }


def _failed_metrics(error):
    out = {column: np.nan for column in RESULT_COLUMNS}
    out['init_uses_pooled_fit'] = False
    return out


# ======================================================================================
# Grid construction
# ======================================================================================

def build_grid(mode):
    baseline = {k: GRID[k][0] for k in GRID_PARAMS}
    if mode == 'cartesian':
        combos = [dict(zip(GRID_PARAMS, values))
                  for values in itertools.product(*(GRID[k] for k in GRID_PARAMS))]
        return baseline, combos

    combos = [dict(baseline)]
    seen = {_freeze(baseline)}
    for axis in GRID_PARAMS:
        for value in GRID[axis]:
            candidate = dict(baseline)
            candidate[axis] = value
            key = _freeze(candidate)
            if key in seen:
                continue
            seen.add(key)
            combos.append(candidate)
    return baseline, combos


# ======================================================================================
# Main
# ======================================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Dirichlet concentration of the PARTITION (identity, not a grid axis)')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--init-seeds', default='0,1,2')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data'))
    parser.add_argument('--mode', choices=('oat', 'cartesian'), default='oat')
    parser.add_argument('--select-on', default='acc', choices=('acc', 'nmi', 'ari'))
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--tag', default='fkm')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--trainer-logs', action='store_true')
    args = parser.parse_args(argv)

    global SHOW_TRAINER_LOGS
    SHOW_TRAINER_LOGS = args.trainer_logs

    init_seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    baseline, combos = build_grid(args.mode)

    print('=' * 78)
    print('GRID DEFINITION (FKM)')
    print('=' * 78)
    print_grid_definition(args.mode)
    swept = [k for k in GRID_PARAMS if len(GRID[k]) > 1]
    est_runs = len(combos) * len(init_seeds)
    print('\nconfigurations : %d  (swept axes: %s)'
          % (len(combos), swept or ['none -- baseline only']))
    print('init seeds     : %s  -> estimated runs = %d' % (init_seeds, est_runs))

    if args.dry_run:
        print('\n[dry-run] no training performed.')
        return 0

    if est_runs > MAX_RUNS_WITHOUT_FORCE and not args.force:
        print('\n! Estimated %d runs > %d. Re-run with --force, or switch --mode oat, '
              'or trim the ranges in GRID.' % (est_runs, MAX_RUNS_WITHOUT_FORCE))
        return 1

    if args.out_dir is None:
        args.out_dir = os.path.join(schema.RESULTS_DIR, _slug(args.dataset),
                                    'federated_%d' % args.grid_split)
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                   args.clients, args.data_dir)
    print('\ndata dir   : %s' % os.path.abspath(args.data_dir))
    print('out dir    : %s' % os.path.relpath(args.out_dir, ROOT))
    print('grid split : %s' % partition_summary(bundle))
    print('fixed eval : %s' % ', '.join('%s=%s' % (k, v) for k, v in FIXED_EVAL.items()))

    rows = []
    started_all = time.perf_counter()
    for index, cfg in enumerate(combos, start=1):
        for seed in init_seeds:
            try:
                metrics = run_fkm(bundle, cfg, seed)
            except Exception as error:  # noqa: BLE001
                if not args.force:
                    raise
                metrics = _failed_metrics(error)
                print('  ! config %d seed=%d failed: %s' % (index, seed, error))

            row = dict(split=bundle['label'], dataset=args.dataset,
                       data_seed=args.grid_split, beta=args.beta,
                       clients=args.clients, init_seed=seed)
            for k, v in cfg.items():
                row[k] = v          # cfg now carries 'learning_rate', not 'beta'
            row.update(metrics)
            row.update({key: bundle['provenance'][key] for key in PROVENANCE_COLUMNS})
            rows.append(row)

            print('  [%3d/%3d] seed=%d | FIT acc=%.4f nmi=%.4f | TEST=%.4f | '
                  'inertia=%.2f (%.1fs)'
                  % (index, len(combos), seed, metrics['acc'], metrics['nmi'],
                     metrics['test_acc'], metrics['final_inertia'], metrics['wall_seconds']))
    grid_seconds = time.perf_counter() - started_all

    frame = pd.DataFrame(rows, columns=list(IDENTITY_COLUMNS) + list(GRID_COLUMNS))
    numeric = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari',
               'n_distinct_labels', 'center_spread_ratio', 'rounds_run',
               'final_inertia', 'wall_seconds']
    mean = (frame.groupby(list(GRID_PARAMS), dropna=False)[numeric]
            .mean().reset_index())
    std = (frame.groupby(list(GRID_PARAMS), dropna=False)[['acc', 'nmi', 'ari']]
           .std(ddof=1).reset_index())
    mean = mean.merge(std, on=list(GRID_PARAMS), suffixes=('', '_std'))
    mean['config'] = [describe(dict(zip(GRID_PARAMS, values)))
                      for values in mean[list(GRID_PARAMS)].to_numpy()]
    mean = mean.sort_values(args.select_on, ascending=False)

    print('\n--- grid, mean over %d init seeds (sorted by FIT %s) ---'
          % (len(init_seeds), args.select_on))
    show_cols = (['config'] + swept
                 + ['acc', 'acc_std', 'nmi', 'ari', 'test_acc',
                    'final_inertia', 'wall_seconds'])
    show_cols = [c for c in show_cols if c in mean.columns]
    print(mean[show_cols].round(4).to_string(index=False))

    best = mean.iloc[0]
    best_cfg = {key: best[key] for key in GRID_PARAMS}
    print('\nBEST: %s | FIT acc=%.4f (+-%.4f) nmi=%.4f | TEST acc=%.4f'
          % (describe(best_cfg), best['acc'], best['acc_std'], best['nmi'], best['test_acc']))

    stem = 'grid_%s_beta%s_clients%d' % (args.tag, float(args.beta), args.clients)
    out_path = os.path.join(args.out_dir, stem + '.csv')
    mean_path = os.path.join(args.out_dir, stem + '_mean.csv')
    frame.to_csv(out_path, index=False)
    mean.to_csv(mean_path, index=False)

    print('\ngrid wall seconds: %.1f' % grid_seconds)
    print('wrote %s (%d rows) and %s (%d configs)'
          % (os.path.relpath(out_path, ROOT), len(frame),
             os.path.relpath(mean_path, ROOT), len(mean)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# python experiments/scripts/grid_search_fkm.py --dataset digits     --beta 0.5 --clients 20 --init-seeds 0,1,2 --tag fkm