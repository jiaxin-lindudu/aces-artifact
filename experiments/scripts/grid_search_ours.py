"""Grid search for every tunable `ours` hyperparameter on one federated partition.

WHAT CHANGED
------------
* Every TUNABLE hyper-parameter now lives in the module-level ``GRID`` and is
  published in ``GRID_COLUMNS``. The sweep ranges are defined HERE, not on the
  command line -- there is no ``--fuzziness`` / ``--rounds`` / ... flag any more.
* Fixed EXTERNAL inputs (``--dataset``, ``--grid-split`` data seed, ``--beta``,
  ``--clients``, ``--init-seeds``) are looped outside the grid and are NOT
  printed as grid axes. They are kept in the CSV only for provenance.
* ``--dry-run`` prints the whole grid definition (every axis + its candidate
  values + the baseline) and exits without training.

GRID MODES
----------
* ``oat`` (default): baseline + one axis varied at a time. ``N = 1 + Sum(len(axis)-1)``
  configurations, which stays tractable even with ~25 tunable axes.
* ``cartesian``: the full Cartesian product. Only use with deliberately small
  ranges -- the script warns and asks for ``--force`` above a run threshold.

DATA / SELECTION
----------------
One frozen Stage 2 partition, addressed by (dataset, grid-split, beta, clients):

    data/{dataset}/partitions/{dataset}_seed_{s}_beta_{beta}_clients_{M}/

Client FIT data and the shared TEST block both come from that partition. Every
configuration is averaged over ``--init-seeds``. FIT ``acc/nmi/ari`` is the ONLY
selection signal (``--select-on``); TEST metrics are reported for every config
but never enter the argmax.

PRIVACY NOTE ON init_method
---------------------------
``kmeans_plusplus`` requires the pooled FIT matrix, i.e. sample-level access.
``GRID['init_method']`` therefore defaults to ``('random','uniform')``. Setting
``ALLOW_POOLED_INIT = True`` appends ``kmeans_plusplus`` and marks every such
row with ``init_uses_pooled_fit=True`` so it can never be confused with the
privacy-consistent runs.

Run:
  python experiments/scripts/grid_search_ours.py --dataset digits \
      --beta 0.5 --clients 20 --init-seeds 0,1,2
  python experiments/scripts/grid_search_ours.py --dataset digits --dry-run
  python experiments/scripts/grid_search_ours.py --dataset digits --mode cartesian --force
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.datasets_and_utils import load_federated_partition  # noqa: E402
from experiments.partition_checks import check_partition, PROVENANCE_COLUMNS
from experiments import schema  # noqa: E402  (RESULTS_DIR only; partitions live in data/)
from federated.ours.aces_trainer import (  # noqa: E402
    federated_anchored_clustering, set_all_seeds)
from federated.ours.tracker import clustering_quality  # noqa: E402


# ======================================================================================
# THE GRID -- every tunable hyper-parameter and its sweep range.
# Edit the lists to widen / narrow a sweep; nothing here comes from the CLI.
# Single-valued axes are still published in GRID_COLUMNS but collapse in `oat`/`cartesian`.
# ======================================================================================
GRID = {
    # Swiss Roll: FIT-guided neighborhoods; only these five axes differ from round one.
    # ---- dual-space fuzzy model (Eq. 11) -------------------------------------------
    'fuzziness':          [1.05, 1.01, 1.025, 1.075, 1.10, 1.20, 1.30],
    'lambda_space':       [0.9, 0.8, 0.85, 0.875, 0.925, 0.95, 0.975,
                           1.0, 0.75, 0.65, 0.5],
    'd_emb':              [2, 3, 4, 6, 8],      # unchanged from the completed Swiss Roll scan

    # ---- local anchored graph ------------------------------------------------------
    'kernel_sigma':       [None],             # restore the adaptive scale used by high-FIT runs
    'kernel_scale_mode':  ['self_tuning', 'global'],
    'k_neighbors':        [5, 4, 6, 7],
    'anchor_weight':      [1.0, 0.5, 0.75, 1.25, 1.5, 2.0],
    'graph_method':       ['knn'],
    'use_anchors':        [True],           # anchors off -> no Nystrom OOS, TEST falls back

    # ---- federated optimization ----------------------------------------------------
    'num_rounds':         [20, 10, 40],
    'active_ratio':       [1.0],                   # client participation fraction
    'init_method':        ['random', 'uniform'],   # kmeans_plusplus excluded (pooled data)

    # ---- topology-consistency refinement ------------------------------------------
    'enforce_topology':               [True],
    'topology_epochs':                [20],
    'learning_rate':                  [1e-3, 5e-3],
    'topology_fidelity_weight':       [1.0],
    'topology_topology_weight':       [1.0],
    'topology_separation_weight':     [1.0],
    'topology_boundary_mode':         ['tanh', 'clamp'],
    'topology_max_displacement':      [1.0],       # only used by boundary_mode='clamp'
    'topology_early_stopping':        [True],
    'topology_early_stopping_patience': [5],

    # ---- fusion / alignment switches (ablation-ready) ------------------------------
    'normalize_spaces':       [True, False],
    'match_procrustes_scale': [True],
    'use_procrustes':         [True],

    # ---- convergence criterion -----------------------------------------------------
    'tol_center_shift':        [1e-4],
    'early_stopping_patience': [3],
}

# Append kmeans_plusplus only if you explicitly accept pooled (non-private) init.
ALLOW_POOLED_INIT = False
if ALLOW_POOLED_INIT:
    GRID['init_method'] = GRID['init_method'] + ['kmeans_plusplus']

#: Column order of the grid definition itself: tunable axes + result metrics.
#: External inputs (dataset / data_seed / beta / clients / init_seed) are NOT here.
RESULT_COLUMNS = (
    'acc', 'nmi', 'ari',
    'test_acc', 'test_nmi', 'test_ari',
    'n_distinct_labels', 'center_spread_ratio',
    'stop_reason', 'rounds_run', 'final_center_shift_orig',
    'test_basis', 'init_uses_pooled_fit', 'wall_seconds',
)
GRID_PARAMS = tuple(GRID.keys())
GRID_COLUMNS = list(GRID_PARAMS) + list(RESULT_COLUMNS) + list(PROVENANCE_COLUMNS)

#: External fixed inputs, written to the CSV for provenance only (NOT grid axes).
IDENTITY_COLUMNS = ('split', 'dataset', 'data_seed', 'beta', 'clients', 'init_seed')

#: Non-tunable evaluation / protocol settings pinned per run (recorded in metadata).
FIXED_EVAL = {
    'use_ckks': False,
    'track_convergence': True,
    'dp_sigma': 0.0,                 # privacy knob, not a utility hyper-parameter
    'run_privacy_eval': False,
    'test_embedding_mode': 'nystrom',
    'test_landmark_mode': 'stored',
    'protocol': 'ours',
    'beta': 0.5,                     # trainer's lambda alias; inert (lambda_space passed)
}

SHOW_TRAINER_LOGS = False        # toggled by --trainer-logs
MAX_RUNS_WITHOUT_FORCE = 2000    # guard against an accidental cartesian explosion


# ======================================================================================
# Small helpers
# ======================================================================================

def _slug(name):
    return ''.join(c if (c.isalnum() or c in '._-') else '_' for c in str(name))


def sigma_label(value):
    if value is None or str(value).strip().lower() in ('adaptive', 'none', ''):
        return 'adaptive'
    return '%g' % float(value)


def _freeze(cfg):
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()))


SHORT = {
    'fuzziness': 'm', 'lambda_space': 'lam', 'd_emb': 'd', 'kernel_sigma': 'sig',
    'kernel_scale_mode': 'ksm', 'k_neighbors': 'k', 'anchor_weight': 'aw',
    'graph_method': 'graph', 'use_anchors': 'anchors', 'num_rounds': 'r',
    'active_ratio': 'act', 'init_method': 'init', 'enforce_topology': 'topo',
    'topology_epochs': 'te', 'learning_rate': 'lr',
    'topology_fidelity_weight': 'wf', 'topology_topology_weight': 'wt',
    'topology_separation_weight': 'ws', 'topology_boundary_mode': 'bnd',
    'topology_max_displacement': 'md', 'topology_early_stopping': 'tes',
    'topology_early_stopping_patience': 'tesp',
    'normalize_spaces': 'norm', 'match_procrustes_scale': 'pscale',
    'use_procrustes': 'procr', 'tol_center_shift': 'tol',
    'early_stopping_patience': 'pat',
}


def describe(cfg):
    parts = []
    for k in GRID_PARAMS:
        v = cfg[k]
        if k == 'kernel_sigma':
            v = sigma_label(v)
        parts.append('%s=%s' % (SHORT.get(k, k), v))
    return ' '.join(parts)


def print_grid_definition(mode):
    print('grid mode : %s' % mode)
    print('grid axes : %d tunable hyper-parameters (ranges fixed in the script)' % len(GRID))
    for k in GRID_PARAMS:
        shown = [sigma_label(v) if k == 'kernel_sigma' else v for v in GRID[k]]
        print('  %-34s %-42s (default=%s)' % (k, shown, shown[0]))


# ======================================================================================
# Partition loading (one frozen Stage 2 partition)
# ======================================================================================

_PARTITION_CACHE = {}

def load_partition_bundle(dataset, data_seed, beta, num_clients, base_dir):
    beta_value = 'subject' if beta == 'subject' else float(beta)
    key = (dataset, int(data_seed), beta_value, int(num_clients), os.path.abspath(base_dir))
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
            % (error, dataset, data_seed, dataset, data_seed, beta_value, num_clients)
        ) from None

    metadata = partition.metadata
    n_classes = metadata.get('split', {}).get('n_classes')
    X_train = np.asarray(partition.X_train, dtype=float)
    X_test = np.asarray(partition.X_test, dtype=float)
    y_train = np.asarray(partition.y_train)
    client_indices = [np.asarray(indices, dtype=int) for indices in partition.client_indices]
    # Order matters: the trainer verifies X_list[i] == X_train[client_indices[i]].
    X_list = [X_train[indices] for indices in client_indices]

    # Rebuild the global FIT block in client-concatenated order, with contiguous
    # ownership: len(train_indices) == len(X_list), concat == arange(len(X_fit))
    # and X_fit[train_indices[i]] == X_list[i] row for row, i.e. the trainer's
    # ownership check holds by construction. Row order changes from the saved
    # partition, but every clustering metric is permutation invariant. Trainer
    # labels come back in this same index space (U_train[indices]), so X_fit and
    # y_fit must be the pair handed to it -- and they are, once, here.
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
        'beta': beta_value,
        'clients': int(num_clients),
        'label': '%s_seed_%d_beta_%s_clients_%d'
                 % (dataset, int(data_seed), beta_value, int(num_clients)),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': np.asarray(partition.y_test),
        'client_indices': client_indices,
        'X_list': X_list,
        # The self-consistent FIT pair the trainer is fed; built once, cached.
        'X_fit': X_fit,
        'y_fit': y_fit,
        'train_indices': train_indices,
        'K': int(n_classes) if n_classes else int(len(np.unique(y_train))),
        'median_distance': float(np.median(pdist(X_train), overwrite_input=True)),
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
# One run of `ours` on one partition, driven entirely by a single grid config
# ======================================================================================

def run_ours(bundle, cfg, init_seed, eval_overrides=None, return_details=False):
    X_test, y_test = bundle['X_test'], bundle['y_test']
    X_list = bundle['X_list']
    K = bundle['K']
    median_distance = bundle['median_distance']
    # Built once by load_partition_bundle: client-concatenated FIT block plus the
    # contiguous ownership the trainer checks, in one index space.
    X_fit, y_fit, train_indices = bundle['X_fit'], bundle['y_fit'], bundle['train_indices']

    kwargs = dict(cfg)
    kwargs.update(FIXED_EVAL)
    kwargs.update(eval_overrides or {})
    kwargs.update({'random_state': init_seed, 'train_indices': train_indices})

    set_all_seeds(init_seed)
    started = time.perf_counter()
    sink = io.StringIO() if not SHOW_TRAINER_LOGS else sys.stdout
    with contextlib.redirect_stdout(sink):
        centers, _centers_emb, history, tracker = federated_anchored_clustering(
            X_list, K, X_train=X_fit, y_train=y_fit,  # ← 用重建的自洽数组
            X_test=X_test, y_test=y_test, **kwargs)
    wall = time.perf_counter() - started

    fit = clustering_quality(X_fit, y_fit, history['train_labels'])  # ← 与训练同一套 X/y

    test = clustering_quality(X_test, y_test, history['test_labels'])
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    spread = float(pairwise[np.triu_indices(len(centers), 1)].mean())
    shifts = ((tracker.history.get('center_shift_orig') or []) if tracker is not None else [])
    prediction = history.get('prediction', {})
    init_meta = history.get('initialization', {})
    metrics = {
        'acc': fit['acc'], 'nmi': fit['nmi'], 'ari': fit['ari'],
        'test_acc': test['acc'], 'test_nmi': test['nmi'], 'test_ari': test['ari'],
        'n_distinct_labels': int(len(np.unique(history['train_labels']))),
        'center_spread_ratio': float(spread / max(median_distance, 1e-12)),
        'stop_reason': history.get('stopping_configuration', {}).get('reason'),
        'rounds_run': int(history['completed_rounds']),
        'final_center_shift_orig': float(shifts[-1]) if shifts else np.nan,
        'test_basis': ('nystrom' if prediction.get('test_dual_space_used') else 'original_space'),
        'init_uses_pooled_fit': bool(init_meta.get('uses_pooled_fit', False)),
        'wall_seconds': wall,
    }
    return (metrics, history, tracker) if return_details else metrics


def _failed_metrics(error):
    out = {column: np.nan for column in RESULT_COLUMNS}
    out['stop_reason'] = 'error: %s' % str(error)[:160]
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

    # one-at-a-time
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- external fixed inputs (NOT grid axes) ----
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--clients', type=int, default=20,
                        help='20 is what data/{dataset}/partitions ships here')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Dirichlet concentration of the partition')
    parser.add_argument('--grid-split', type=int, default=0,
                        help='data_seed of the partition (FIT and TEST both come from it)')
    parser.add_argument('--init-seeds', default='0,1,2')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data'))
    # ---- grid / run control ----
    parser.add_argument('--mode', choices=('oat', 'cartesian'), default='oat',
                        help="'oat' = one axis at a time (default); 'cartesian' = full product")
    parser.add_argument('--select-on', default='acc', choices=('acc', 'nmi', 'ari'))
    parser.add_argument('--out-dir', default=None,
                        help='default: results/{dataset}/federated_{grid_split}')
    parser.add_argument('--tag', default='ours')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the grid definition and exit without training')
    parser.add_argument('--force', action='store_true',
                        help='run even if the estimated run count is large')
    parser.add_argument('--trainer-logs', action='store_true')
    args = parser.parse_args(argv)

    global SHOW_TRAINER_LOGS
    SHOW_TRAINER_LOGS = args.trainer_logs

    init_seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    baseline, combos = build_grid(args.mode)

    # ---- always publish the grid definition (auto-listed ranges) ----
    print('=' * 78)
    print('GRID DEFINITION')
    print('=' * 78)
    print_grid_definition(args.mode)
    swept = [k for k in GRID_PARAMS if len(GRID[k]) > 1]
    est_runs = len(combos) * len(init_seeds)
    print('\nconfigurations : %d  (swept axes: %s)'
          % (len(combos), swept or ['none -- single baseline config']))
    print('init seeds     : %s  -> estimated runs = %d' % (init_seeds, est_runs))

    if args.dry_run:
        print('\n[dry-run] no training performed.')
        return 0

    if est_runs > MAX_RUNS_WITHOUT_FORCE and not args.force:
        print('\n! Estimated %d runs > %d. Re-run with --force, or switch --mode oat, '
              'or trim the ranges in GRID.' % (est_runs, MAX_RUNS_WITHOUT_FORCE))
        return 1

    # ---- resolve outputs and load the single frozen partition ----
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

    # ---- sweep ----
    rows = []
    started_all = time.perf_counter()
    for index, cfg in enumerate(combos, start=1):
        for seed in init_seeds:
            try:
                metrics = run_ours(bundle, cfg, seed)
            except Exception as error:  # noqa: BLE001
                if not args.force:
                    raise
                metrics = _failed_metrics(error)
                print('  ! config %d seed=%d failed: %s' % (index, seed, error))

            row = dict(split=bundle['label'], dataset=args.dataset,
                       data_seed=args.grid_split, beta=args.beta,
                       clients=args.clients, init_seed=seed)
            row['kernel_sigma'] = sigma_label(cfg['kernel_sigma'])
            for k, v in cfg.items():
                if k != 'kernel_sigma':
                    row[k] = v
            row.update(metrics)
            row.update({key: bundle['provenance'][key] for key in PROVENANCE_COLUMNS})
            rows.append(row)

            print('  [%3d/%3d] seed=%d | FIT acc=%.4f nmi=%.4f | TEST=%.4f | '
                  'shift=%.4f %s (%.1fs)'
                  % (index, len(combos), seed, metrics['acc'], metrics['nmi'],
                     metrics['test_acc'], metrics['final_center_shift_orig'],
                     metrics['stop_reason'], metrics['wall_seconds']))
    grid_seconds = time.perf_counter() - started_all

    # ---- aggregate over init seeds (group by the tunable axes only) ----
    frame = pd.DataFrame(rows, columns=list(IDENTITY_COLUMNS) + list(GRID_COLUMNS))
    numeric = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari',
               'n_distinct_labels', 'center_spread_ratio', 'rounds_run',
               'final_center_shift_orig', 'wall_seconds']
    mean = (frame.groupby(list(GRID_PARAMS), dropna=False)[numeric]
            .mean().reset_index())
    std = (frame.groupby(list(GRID_PARAMS), dropna=False)[['acc', 'nmi', 'ari']]
           .std(ddof=1).reset_index())
    mean = mean.merge(std, on=list(GRID_PARAMS), suffixes=('', '_std'))
    # list(...) matters: a tuple would be read as ONE column key, not a column subset.
    mean['config'] = [describe(dict(zip(GRID_PARAMS, values)))
                      for values in mean[list(GRID_PARAMS)].to_numpy()]
    mean = mean.sort_values(args.select_on, ascending=False)

    print('\n--- grid, mean over %d init seeds (sorted by FIT %s) ---'
          % (len(init_seeds), args.select_on))
    show_cols = (['config'] + swept
                 + ['acc', 'acc_std', 'nmi', 'ari', 'test_acc',
                    'final_center_shift_orig', 'stop_reason'])
    show_cols = [c for c in show_cols if c in mean.columns]
    print(mean[show_cols].round(4).to_string(index=False))

    best = mean.iloc[0]
    best_cfg = {key: best[key] for key in GRID_PARAMS}
    print('\nBEST: %s | FIT acc=%.4f (+-%.4f) nmi=%.4f | TEST acc=%.4f'
          % (describe(best_cfg), best['acc'], best['acc_std'], best['nmi'], best['test_acc']))

    # ---- save ----
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
