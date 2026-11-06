"""Grid search for nnfc on one frozen Stage 2 partition.

Same CLI, OAT/cartesian expansion, FIT-only seed-mean selection and two CSV
outputs as grid_search_ffcm.py. Algorithm initialization stays method-native.
No raw TEST rows/labels or FIT labels are passed to numerical training.
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
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data.datasets_and_utils import load_federated_partition
from experiments import schema
from experiments.partition_checks import check_partition, PROVENANCE_COLUMNS

METHOD_TAG = 'nnfc'
GRID = {'k_neighbors': [2, 5, 10, 20]}
FIXED_EVAL = {'dp_sigma': 0.0}

RESULT_COLUMNS = (
    'acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari',
    'n_distinct_labels', 'center_spread_ratio', 'rounds_run', 'final_inertia',
    'init_uses_pooled_fit', 'wall_seconds',
    'uplink_bytes', 'downlink_bytes', 'total_bytes', 'bytes_per_round',
    'training_seconds', 'local_seconds', 'aggregation_seconds',
    'fit_basis', 'test_basis', 'initialization', 'communication_basis',
    'privacy_status', 'effective_epsilon', 'effective_delta',
    'status', 'error', 'hit_cap',
)
GRID_PARAMS = tuple(GRID)
GRID_COLUMNS = list(GRID_PARAMS) + list(RESULT_COLUMNS)
IDENTITY_COLUMNS = ('split', 'dataset', 'data_seed', 'beta', 'clients', 'init_seed')
SHOW_TRAINER_LOGS = False
MAX_RUNS_WITHOUT_FORCE = 2000

def _slug(name):
    return ''.join(c if (c.isalnum() or c in '._-') else '_' for c in str(name))


def _freeze(cfg):
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()))


SHORT = {}


def describe(cfg):
    return ' '.join('%s=%s' % (SHORT.get(k, k), cfg[k]) for k in GRID_PARAMS)


def print_grid_definition(mode):
    print('grid mode : %s' % mode)
    print('grid axes : %d tunable hyper-parameters' % len(GRID))
    for k in GRID_PARAMS:
        shown = GRID[k]
        print('  %-30s %-50s (default=%s)' % (k, shown, shown[0]))


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



def configure_grid(K):
    global GRID_PARAMS, GRID_COLUMNS
    GRID_PARAMS = tuple(GRID)
    GRID_COLUMNS = list(GRID_PARAMS)+list(RESULT_COLUMNS)


def run_model(b, cfg, seed):
    from federated.nnfc.nnfc_trainer import federated_nnfc_clustering
    centers, history = federated_nnfc_clustering(None, None, None, None, b['X_list'], b['K'],
        random_state=seed, **cfg, **FIXED_EVAL)
    if history.get('fallback_used'):
        raise ValueError('NNFC SNN fallback: '+history.get('fallback_error',''))
    fit = cdist(b['X_fit'],centers).argmin(axis=1)
    test = cdist(b['X_test'],centers).argmin(axis=1)
    return score_outcome(b, centers, fit, test, 1, history['communication'], history,
        'nearest_released_center', 'nearest_released_center_proxy',
        'client-local KMeans k-means++ / n_init=10; local K=min(n_i//3,50)',
        'noise_disabled; epsilon=0 is a non-private ablation', epsilon=0.)

def score_outcome(bundle, centers, labels, test_labels, rounds, communication,
                  cost, fit_basis, test_basis, initialization, privacy_status,
                  hit_cap=False, epsilon=np.nan, delta=np.nan):
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (bundle['K'], bundle['X_fit'].shape[1]) or not np.isfinite(centers).all():
        raise ValueError('Trainer did not return K finite centers; no synthetic KMeans fallback')
    fit = clustering_quality(bundle['X_fit'], bundle['y_fit'], labels)
    test = clustering_quality(bundle['X_test'], bundle['y_test'], test_labels)
    spread = pdist(centers).mean() if len(centers) > 1 else 0.
    up, down = communication['uplink_bytes'], communication['downlink_bytes']
    return dict(fit, test_acc=test['acc'], test_nmi=test['nmi'], test_ari=test['ari'],
        n_distinct_labels=len(np.unique(labels)),
        center_spread_ratio=float(spread / max(bundle['median_distance'], 1e-12)),
        rounds_run=rounds, final_inertia=float(np.mean(np.sum((bundle['X_fit']-centers[labels])**2, axis=1))),
        init_uses_pooled_fit=False, uplink_bytes=up, downlink_bytes=down,
        total_bytes=up+down, bytes_per_round=(up+down)/rounds,
        training_seconds=cost.get('training_seconds', np.nan),
        local_seconds=cost.get('local_seconds', np.nan),
        aggregation_seconds=cost.get('aggregation_seconds', np.nan),
        fit_basis=fit_basis, test_basis=test_basis, initialization=initialization,
        communication_basis=communication.get('measurement_basis', communication.get('accounting', 'NumPy payload nbytes; excludes network framing')),
        privacy_status=privacy_status, effective_epsilon=epsilon, effective_delta=delta,
        status='completed', error='', hit_cap=bool(hit_cap))


def build_grid(mode):
    baseline = {key: values[0] for key, values in GRID.items()}
    if mode == 'cartesian':
        return baseline, [dict(zip(GRID, vals)) for vals in itertools.product(*GRID.values())]
    combos, seen = [dict(baseline)], {_freeze(baseline)}
    for key, values in GRID.items():
        for value in values:
            cfg = dict(baseline, **{key: value})
            if _freeze(cfg) not in seen:
                combos.append(cfg)
                seen.add(_freeze(cfg))
    return baseline, combos


def aggregate(frame, init_seeds, select_on):
    numeric = [k for k in RESULT_COLUMNS if k in frame and pd.api.types.is_numeric_dtype(frame[k])]
    groups = frame.groupby(list(GRID_PARAMS), dropna=False, sort=False)
    mean = groups[numeric].mean().reset_index()
    std = groups[['acc','nmi','ari','test_acc','test_nmi','test_ari']].std(ddof=1).reset_index()
    mean = mean.merge(std, on=list(GRID_PARAMS), suffixes=('', '_std'), sort=False)
    complete = []
    for _, part in groups:
        complete.append(len(part)==len(init_seeds) and sorted(part.init_seed.tolist())==sorted(init_seeds)
            and part.status.eq('completed').all()
            and np.isfinite(part[['acc','nmi','ari','test_acc','test_nmi','test_ari']].to_numpy(float)).all())
    mean['complete'] = complete
    mean['n_seeds'] = len(init_seeds)
    mean['config'] = [describe(cfg) for cfg in mean[list(GRID_PARAMS)].to_dict('records')]
    return mean.sort_values(['complete', select_on], ascending=[False, False], kind='stable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--init-seeds', default='0,1,2')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data'))
    parser.add_argument('--mode', choices=('oat','cartesian'), default='oat')
    parser.add_argument('--select-on', choices=('acc','nmi','ari'), default='acc')
    parser.add_argument('--out-dir')
    parser.add_argument('--tag', default=METHOD_TAG)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--trainer-logs', action='store_true')
    args = parser.parse_args(argv)
    global SHOW_TRAINER_LOGS
    SHOW_TRAINER_LOGS = args.trainer_logs
    seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    if not seeds or len(set(seeds))!=len(seeds) or any(not 0<=s<2**32 for s in seeds):
        parser.error('init-seeds must be distinct nonnegative uint32 integers')
    # K-dependent axes use the frozen dataset metadata; never use labels to infer local K.
    if args.dry_run:
        configure_grid(None)
    else:
        bundle = load_partition_bundle(args.dataset, args.grid_split, args.beta, args.clients, args.data_dir)
        configure_grid(bundle['K'])
    _, combos = build_grid(args.mode)
    print_grid_definition(args.mode)
    print('configurations=%d seeds=%s runs=%d' % (len(combos), seeds, len(combos)*len(seeds)))
    print('fixed evaluation:', FIXED_EVAL)
    if args.dry_run:
        return 0
    if len(combos)*len(seeds)>MAX_RUNS_WITHOUT_FORCE and not args.force:
        parser.error('run count exceeds 2000; reduce GRID or use --force')
    out = args.out_dir or os.path.join(schema.RESULTS_DIR, _slug(args.dataset), 'federated_%d'%args.grid_split)
    os.makedirs(out, exist_ok=True)
    stem = 'grid_%s_beta%s_clients%d' % (args.tag, float(args.beta), args.clients)
    raw_path, mean_path = os.path.join(out, stem+'.csv'), os.path.join(out, stem+'_mean.csv')
    if os.path.exists(raw_path) or os.path.exists(mean_path):
        raise FileExistsError('Preserve existing results: choose a new --out-dir or --tag')
    print(partition_summary(bundle))
    rows = []
    columns = list(IDENTITY_COLUMNS)+GRID_COLUMNS+list(PROVENANCE_COLUMNS)
    for index, cfg in enumerate(combos, 1):
        for seed in seeds:
            started = time.perf_counter()
            error = None
            try:
                set_all_seeds(seed)
                sink = sys.stdout if SHOW_TRAINER_LOGS else io.StringIO()
                # Same sequential CPU budget; avoid BLAS oversubscription changing wall times.
                with contextlib.redirect_stdout(sink), threadpool_limits(limits=1):
                    result = run_model(bundle, cfg, seed)
            except Exception as exc:
                error = exc
                result = {k: np.nan for k in RESULT_COLUMNS}
                result.update(status='failed', error='%s: %s'%(type(exc).__name__,exc), init_uses_pooled_fit=False)
            result['wall_seconds'] = time.perf_counter()-started
            row = dict(split=bundle['label'], dataset=args.dataset, data_seed=args.grid_split,
                       beta=args.beta, clients=args.clients, init_seed=seed, **cfg, **result)
            row.update({k:bundle['provenance'][k] for k in PROVENANCE_COLUMNS})
            rows.append(row)
            pd.DataFrame(rows, columns=columns).to_csv(raw_path, index=False)
            print('[%d/%d] seed=%d FIT=%.4f TEST=%.4f status=%s' %
                  (index,len(combos),seed,result['acc'],result['test_acc'],result['status']), flush=True)
            if error is not None and not args.force:
                raise error
    frame = pd.DataFrame(rows, columns=columns)
    mean = aggregate(frame, seeds, args.select_on)
    mean.to_csv(mean_path, index=False)
    complete = mean[mean.complete]
    if complete.empty:
        print('No complete configuration; inspect failed rows in the raw CSV')
        return 1
    best = complete.iloc[0]
    print('BEST on mean FIT %s: %s | FIT=%.4f TEST=%.4f' %
          (args.select_on,best['config'],best['acc'],best['test_acc']))
    print('wrote', raw_path, 'and', mean_path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
