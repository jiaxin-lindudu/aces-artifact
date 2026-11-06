"""Tune and score the centralized clustering ceilings on a single frozen split.

WORKFLOW (one dataset, one data_seed)
-------------------------------------
Everything happens on that one frozen FIT/TEST split:

    load the frozen split ONCE;
    for each init_seed:                          # tuning is per initialization
        for each method:
            (a) BUDGET   fix the iteration count on FIT -- sweep convergence
                         budgets and keep the smallest budget where the solver
                         stops moving (FIT accuracy equals that at the largest
                         budget AND no run hits the cap);
            (b) TUNE     sweep the hyper-parameter grid AT that budget on FIT and
                         keep the configuration with the best FIT <select_on>;
            (c) SCORE    re-train the chosen configuration on FIT and score the
                         held-out TEST block (TEST never touches selection).

Selection uses the FIT block only. During tuning the TEST block is not even
scored, so there is no route for test information to influence the choice.

SAVE LAYOUT
-----------
    results/{dataset}/seed_{data_seed}/
        centralized_grid_{dataset}_seed{data_seed}.csv          # every grid cell (FIT only)
        centralized_ceiling_{dataset}_seed{data_seed}.csv       # chosen cfg: FIT + TEST per init_seed
        centralized_convergence_{dataset}_seed{data_seed}.csv   # iteration-budget sweep
        centralized_run_metadata_{dataset}_seed{data_seed}.json # config + provenance + summary

Run:
  python centralized/run_ceiling_grid.py --dataset digits --data-seed 0 \
      --init-seeds 40,41,42,43,44
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from centralized import METHODS  # noqa: E402
from centralized.grids import GRIDS, describe  # noqa: E402
from data.datasets_and_utils import load_centralized_split  # noqa: E402
from experiments import schema  # noqa: E402
from federated.ours.tracker import clustering_quality  # noqa: E402

DEFAULT_BUDGETS = (1, 2, 5, 10, 30, 100, 300, 1000)
_QUALITY_FALLBACK = ('acc', 'nmi', 'ari', 'silhouette',
                     'davies_bouldin', 'calinski_harabasz')


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------

def _parse_kv(text: str) -> Dict:
    """`fuzziness=1.3,max_iter=50` -> dict; numbers stay numbers."""
    parsed = {}
    for item in (text or '').split(','):
        item = item.strip()
        if not item:
            continue
        key, _, raw = item.partition('=')
        try:
            value = float(raw)
            if value == int(value):
                value = int(value)
        except ValueError:
            value = raw
        parsed[key.strip()] = value
    return parsed


def _slug(name: str) -> str:
    """Filesystem-safe dataset slug that keeps the full name (cifar10_vit, ...)."""
    return ''.join(c if (c.isalnum() or c in '._-') else '_' for c in str(name))


def _sha256(array) -> str:
    if array is None:
        return ''
    return hashlib.sha256(np.ascontiguousarray(array, dtype=float).tobytes()).hexdigest()


def _load_split(dataset: str, data_seed: int, base_dir=None) -> Dict:
    """Load the frozen split, tolerating BOTH loader signatures.

    Accepts the attribute/namedtuple form (`split.X_train`, `split.centers`,
    `split.metadata`) or the historical `(meta, arrays)` tuple form. This keeps the
    script working whichever shape ``load_centralized_split`` currently returns.
    """
    raw = load_centralized_split(dataset, data_seed,
                                 base_dir=base_dir or os.path.join(ROOT, 'data'), verbose=False)

    def _normalise(X_train, X_test, y_train, y_test, centers, metadata):
        centers_arr = None
        if centers is not None:
            centers_arr = np.asarray(centers, dtype=float)
            if centers_arr.ndim != 2 or centers_arr.size == 0 or not np.isfinite(centers_arr).all():
                centers_arr = None
        return {
            'X_train': np.asarray(X_train, dtype=float),
            'X_test': np.asarray(X_test, dtype=float),
            'y_train': np.asarray(y_train, dtype=int),
            'y_test': np.asarray(y_test, dtype=int),
            'centers': centers_arr,
            'metadata': metadata or {},
        }

    if all(hasattr(raw, attr) for attr in
           ('X_train', 'X_test', 'y_train', 'y_test')):
        return _normalise(
            raw.X_train, raw.X_test, raw.y_train, raw.y_test,
            getattr(raw, 'centers', None), getattr(raw, 'metadata', {}) or {},
        )
    if isinstance(raw, tuple) and len(raw) == 2:
        meta, arrays = raw
        return _normalise(
            arrays['X_train'], arrays['X_test'], arrays['y_train'], arrays['y_test'],
            arrays.get('centers'), meta or {},
        )
    raise TypeError('Unrecognised split object from load_centralized_split: %r'
                    % (type(raw),))


def _num_classes(arrays: Dict) -> int:
    centers = arrays.get('centers')
    if centers is not None and np.asarray(centers).size:
        return int(len(centers))
    return int(len(np.unique(arrays['y_train'])))


def _preprocess_name(metadata: Dict) -> str:
    pre = metadata.get('preprocessing') if isinstance(metadata, dict) else None
    if isinstance(pre, dict):
        return str(pre.get('name') or pre.get('scaler') or pre.get('source')
                   or 'frozen_split')
    if isinstance(pre, str) and pre:
        return pre
    if isinstance(metadata, dict) and metadata.get('preprocess'):
        return str(metadata['preprocess'])
    return 'frozen_split'


def _score(metrics: Dict, key: str) -> float:
    """FIT selection score; NaN/missing sorts to the bottom."""
    value = metrics.get(key, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -np.inf
    return value if np.isfinite(value) else -np.inf


def _run(method: str, arrays: Dict, init_seed: int, cfg: Dict,
         compute_test: bool) -> Dict:
    """One (method, initialization, configuration) run on the frozen split.

    ``compute_test=False`` is used during tuning: the model still runs (the runner
    needs X_test), but the TEST block is never scored, so a configuration cannot be
    chosen using TEST information.
    """
    X_train, X_test = arrays['X_train'], arrays['X_test']
    y_train, y_test = arrays['y_train'], arrays['y_test']
    K = _num_classes(arrays)

    started = time.perf_counter()
    outcome = METHODS[method](X_train,X_test,K,init_seed,cfg)
    wall = time.perf_counter() - started

    fit = clustering_quality(X_train, y_train, outcome['train_labels'])
    if compute_test and X_test is not None and len(X_test):
        test = clustering_quality(X_test, y_test, outcome['test_labels'])
    else:
        test = {key: np.nan for key in _QUALITY_FALLBACK}

    return {
        'fit': fit,
        'test': test,
        'wall': wall,
        'test_basis': outcome.get('test_basis', 'unknown'),
        'extra': outcome.get('extra', {}) or {},
        'K': K,
    }


# ----------------------------------------------------------------------------------
# Row builders (centralized: no clients / beta)
# ----------------------------------------------------------------------------------

def _common_row(dataset, method, data_seed, init_seed, cfg, label, K, study,
                wall, test_basis, extra, preprocess_name, partition_sha):
    row = schema.empty_row()
    row.update({
        'dataset': dataset,
        'method': method,
        'data_seed': int(data_seed),
        'clusters': int(K),
        'study': study,
        'study_value': 1.0,
        'init_seed': int(init_seed),
        'config': label,
        'max_iter': int(cfg.get('max_iter', -1)),
        'n_iter': int(extra.get('n_iter', -1)),
        'hit_cap': bool(extra.get('hit_cap', False)),
        'partition_sha256': partition_sha,
        'test_basis': test_basis,
        'oos_status': 'not_applicable',
        'use_ckks': False,
        'ckks_available': False,
        'fuzziness': cfg.get('fuzziness', np.nan),
        'lambda_space': np.nan,
        'rounds': np.nan,
        'wall_seconds': wall,
        'preprocess': preprocess_name,
        'note': ('centralized %s | cfg: %s | max_iter=%d iters=%d cap=%s'
                 % (str(method).split('_')[-1].upper(), label,
                    int(cfg.get('max_iter', -1)), int(extra.get('n_iter', -1)),
                    extra.get('hit_cap'))),
    })
    return row


def _grid_row(dataset, method, data_seed, init_seed, cfg, label, result,
              preprocess_name, partition_sha):
    row = _common_row(dataset, method, data_seed, init_seed, cfg, label,
                      result['K'], 'ceiling_grid', result['wall'],
                      result['test_basis'], result['extra'], preprocess_name,
                      partition_sha)
    for key in schema.QUALITY_KEYS:
        row[key] = schema.as_float(result['fit'].get(key, np.nan))
        row['all_fit_' + key] = schema.as_float(result['fit'].get(key, np.nan))
        row['test_proxy_' + key] = np.nan  # TEST not scored during tuning
    row['note'] += ' | tuning uses FIT only; TEST not scored'
    return row


def _ceiling_row(dataset, method, data_seed, init_seed, cfg, label, result,
                 preprocess_name, partition_sha):
    row = _common_row(dataset, method, data_seed, init_seed, cfg, label,
                      result['K'], 'ceiling', result['wall'],
                      result['test_basis'], result['extra'], preprocess_name,
                      partition_sha)
    for key in schema.QUALITY_KEYS:
        row[key] = schema.as_float(result['fit'].get(key, np.nan))
        row['all_fit_' + key] = schema.as_float(result['fit'].get(key, np.nan))
        row['test_proxy_' + key] = schema.as_float(result['test'].get(key, np.nan))
    row['note'] += ' | selected on FIT; TEST is held out'
    if 'crispness' in result['extra']:
        row['note'] += (' | mean max membership=%.3f (uniform %.3f)'
                        % (result['extra']['crispness'],
                           result['extra']['uniform_crispness']))
    return row


# ----------------------------------------------------------------------------------
# Per-(method, init_seed) iteration budget
# ----------------------------------------------------------------------------------

def determine_budget(method: str, arrays: Dict, init_seed: int,
                     budgets: Tuple[int, ...], extra_cfg: Dict,
                     verbose: bool = True) -> Tuple[int, List[Dict]]:
    """Smallest budget at which this (method, init_seed) run stops moving.

    Convergence rule (as in tune_iterations.py, but per initialization): the budget is
    accepted when its FIT accuracy equals that at the largest tested budget exactly
    (|delta| < 1e-12) AND its hit_cap fraction is zero. Otherwise fall back to the
    largest budget (with a warning if it still hits the cap).
    """
    rows = []
    for budget in budgets:
        cfg = dict(extra_cfg, max_iter=int(budget))
        result = _run(method, arrays, init_seed, cfg, compute_test=False)
        rows.append({
            'method': method,
            'init_seed': int(init_seed),
            'max_iter': int(budget),
            'fit_acc': float(result['fit'].get('acc', np.nan)),
            'n_iter': int(result['extra'].get('n_iter', -1)),
            'hit_cap': bool(result['extra'].get('hit_cap', False)),
        })

    reference = rows[-1]['fit_acc']
    settled = None
    for row in rows:
        if (abs(row['fit_acc'] - reference) < 1e-12) and (not row['hit_cap']):
            settled = row['max_iter']
            break
    if settled is None:
        settled = int(budgets[-1])
        if rows[-1]['hit_cap']:
            print('  ! %s init=%d: even max_iter=%d hits the cap; budget may be '
                  'unreliable' % (method, init_seed, settled))

    if verbose:
        print('    %-16s init=%d -> budget=%d (FIT acc@max=%.6f)'
              % (method, init_seed, settled, reference))
    return settled, rows


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--data-seed', type=int, default=0,
                        help='the single frozen split to tune AND score on')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data'),
                        help='Stage 1 data root, independent of the current working directory')
    parser.add_argument('--init-seeds', default='0,1,2,3,4',
                        help='initialization seeds; each is tuned and scored separately')
    parser.add_argument('--max-iter', type=int, default=None,
                        help='optional FIXED iteration budget for every method/init; '
                             'if omitted the budget is determined per (method, init_seed)')
    parser.add_argument('--recheck-max-iter', type=int, default=1000,
                        help='Re-run a selected capped configuration at this budget before TEST; 0 disables')
    parser.add_argument('--budgets', default=','.join(str(b) for b in DEFAULT_BUDGETS),
                        help='candidate iteration budgets for the auto sweep')
    parser.add_argument('--select-on', default='acc', choices=('acc', 'nmi', 'ari'),
                        help='FIT metric used to pick the configuration')
    parser.add_argument('--extra-cfg', default='',
                        help='key=value,key2=value2 merged into every method config')
    parser.add_argument('--out-dir', default=None,
                        help='default: results/{dataset}/seed_{data_seed}')
    parser.add_argument('--force', action='store_true',
                        help='continue past per-cell errors instead of aborting')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if args.recheck_max_iter < 0:
        parser.error('recheck-max-iter must be nonnegative')

    verbose = not args.quiet
    init_seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    budgets = tuple(int(v) for v in args.budgets.split(',') if v.strip())
    extra_cfg = _parse_kv(args.extra_cfg)

    dataset = args.dataset
    dataset_slug = _slug(dataset)
    if args.out_dir is None:
        args.out_dir = os.path.join(schema.RESULTS_DIR, dataset_slug,
                                    'seed_%d' % args.data_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load the single frozen split ----
    try:
        arrays = _load_split(dataset, args.data_seed, args.data_dir)
    except FileNotFoundError as exc:
        print('Cannot load split for %s seed=%d. Run prepare_centralized_splits.py '
              'first.\n%s' % (dataset, args.data_seed, exc), file=sys.stderr)
        return 1

    metadata = arrays['metadata']
    K = _num_classes(arrays)
    preprocess_name = _preprocess_name(metadata)
    n_train, n_test = len(arrays['X_train']), len(arrays['X_test'])
    n_features = arrays['X_train'].shape[1]
    partition_sha = _sha256(arrays['X_train'])
    split_data_seed = int(metadata.get('data_seed', args.data_seed)) \
        if isinstance(metadata, dict) else int(args.data_seed)

    print('=' * 70)
    print('CENTRALIZED CLUSTERING CEILING (single frozen split)')
    print('=' * 70)
    print('dataset=%s | data_seed=%d | FIT=%d TEST=%d K=%d d=%d'
          % (dataset, args.data_seed, n_train, n_test, K, n_features))
    print('preprocess=%s | X_train sha256=%s' % (preprocess_name, partition_sha[:12]))
    print('init_seeds=%s | budget=%s | select_on=FIT/%s'
          % (init_seeds,
             ('fixed %d' % args.max_iter) if args.max_iter is not None else 'auto',
             args.select_on))
    if split_data_seed != args.data_seed:
        print('! metadata data_seed=%d differs from --data-seed=%d'
              % (split_data_seed, args.data_seed))

    # ---- main loop: per init_seed -> per method ----
    convergence_rows: List[Dict] = []
    grid_rows: List[Dict] = []
    ceiling_rows: List[Dict] = []
    per_method: Dict[str, Dict] = {}

    started = time.perf_counter()

    for init_seed in init_seeds:
        if verbose:
            print('\n' + '-' * 70)
            print('INIT SEED %d' % init_seed)
            print('-' * 70)

        for method, grid in GRIDS.items():
            method_key = method
            entry = per_method.setdefault(
                method_key, {'budgets': {}, 'winners': {}})

            # (a) iteration budget
            if args.max_iter is not None:
                budget = int(args.max_iter)
                if verbose:
                    print('    %-16s init=%d -> budget=%d (fixed)' %
                          (method, init_seed, budget))
            else:
                budget, sweep = determine_budget(method, arrays, init_seed,
                                                 budgets, extra_cfg, verbose)
                convergence_rows.extend(sweep)
            entry['budgets'][str(init_seed)] = int(budget)

            # (b) grid tuning on FIT only
            best = None
            for cfg in grid:
                cell = dict(cfg, max_iter=int(budget))
                label = describe(method, cfg)
                try:
                    result = _run(method, arrays, init_seed, cell,
                                  compute_test=False)
                except Exception as exc:  # noqa: BLE001
                    print('  x %s init=%d cfg=%s: %s'
                          % (method, init_seed, label, exc), file=sys.stderr)
                    if not args.force:
                        return 1
                    continue
                grid_rows.append(_grid_row(dataset, method, args.data_seed,
                                           init_seed, cell, label, result,
                                           preprocess_name, partition_sha))
                score = _score(result['fit'], args.select_on)
                if best is None or score > best['score']:
                    best = {'score': score, 'cfg': cell, 'label': label,
                            'hit_cap': bool(result['extra'].get('hit_cap', False))}

            if best is None:
                print('  x %s init=%d: no usable configuration' %
                      (method, init_seed), file=sys.stderr)
                if not args.force:
                    return 1
                continue

            # Preserve FIT selection. A capped winner gets a larger budget,
            # rather than being described as converged at the default-config budget.
            if best['hit_cap'] and args.recheck_max_iter > best['cfg']['max_iter']:
                entry.setdefault('cap_rechecks', {})[str(init_seed)] = {
                    'previous_max_iter': best['cfg']['max_iter'],
                    'recheck_max_iter': args.recheck_max_iter,
                }
                best['cfg'] = dict(best['cfg'], max_iter=args.recheck_max_iter)
            # (c) re-train chosen config and score TEST exactly once
            result = _run(method, arrays, init_seed, best['cfg'],
                          compute_test=True)
            ceiling_rows.append(_ceiling_row(
                dataset, method, args.data_seed, init_seed, best['cfg'],
                best['label'], result, preprocess_name, partition_sha))

            entry['winners'][str(init_seed)] = {
                'config': best['label'],
                'max_iter': int(best['cfg'].get('max_iter', budget)),
                'fit_acc': float(result['fit'].get('acc', np.nan)),
                'test_acc': float(result['test'].get('acc', np.nan)),
                'hit_cap': bool(result['extra'].get('hit_cap', False)),
            }
            if verbose:
                print('    -> %-16s chosen: %-28s FIT acc=%.4f  TEST acc=%.4f'
                      % (method, best['label'],
                         result['fit'].get('acc', np.nan),
                         result['test'].get('acc', np.nan)))

    # ---- assemble frames ----
    grid_df = pd.DataFrame(grid_rows)
    ceiling_df = pd.DataFrame(ceiling_rows)
    convergence_df = pd.DataFrame(convergence_rows)

    # ---- summary across init seeds ----
    print('\n' + '=' * 70)
    print('SUMMARY (chosen config per init_seed, aggregated across init seeds)')
    print('=' * 70)

    metric_cols = ['acc', 'nmi', 'ari',
                   'test_proxy_acc', 'test_proxy_nmi', 'test_proxy_ari']
    metric_cols = [c for c in metric_cols if c in ceiling_df.columns]

    if len(ceiling_df):
        caps = int(ceiling_df['hit_cap'].sum()) if 'hit_cap' in ceiling_df else 0
        print('convergence: %d/%d chosen-config runs hit the cap'
              % (caps, len(ceiling_df)))

        table = (ceiling_df.groupby('method')[metric_cols]
                 .agg(['mean', 'std']).round(4))
        print(table.to_string())

        for method, entry in per_method.items():
            distinct = sorted({w['config'] for w in entry['winners'].values()})
            print('  %-16s distinct winners across init_seeds: %d %s'
                  % (method, len(distinct), distinct if len(distinct) <= 3 else ''))
    else:
        print('no ceiling rows produced')

    # ---- save ----
    stem = 'centralized_%s_%s_seed%d'
    grid_file = os.path.join(args.out_dir, stem % ('grid', dataset_slug, args.data_seed) + '.csv')
    ceiling_file = os.path.join(args.out_dir, stem % ('ceiling', dataset_slug, args.data_seed) + '.csv')
    conv_file = os.path.join(args.out_dir, stem % ('convergence', dataset_slug, args.data_seed) + '.csv')
    meta_file = os.path.join(args.out_dir, stem % ('run_metadata', dataset_slug, args.data_seed) + '.json')

    grid_df.to_csv(grid_file, index=False)
    ceiling_df.to_csv(ceiling_file, index=False)
    if len(convergence_df):
        convergence_df.to_csv(conv_file, index=False)

    run_metadata = {
        'dataset': dataset,
        'data_seed': int(args.data_seed),
        'split_data_seed': split_data_seed,
        'n_train': int(n_train),
        'n_test': int(n_test),
        'n_features': int(n_features),
        'n_classes': int(K),
        'preprocess': preprocess_name,
        'init_seeds': [int(s) for s in init_seeds],
        'select_on': args.select_on,
        'budget_mode': 'fixed' if args.max_iter is not None else 'auto',
        'fixed_max_iter': (int(args.max_iter) if args.max_iter is not None else None),
        'budgets': [int(b) for b in budgets],
        'extra_cfg': extra_cfg,
        'selection_scope': 'FIT block only; TEST scored once per init_seed after selection',
        'arrays_sha256': {
            'X_train': _sha256(arrays['X_train']),
            'y_train': _sha256(arrays['y_train']),
            'X_test': _sha256(arrays['X_test']),
            'y_test': _sha256(arrays['y_test']),
        },
        'per_method': per_method,
        'files': {
            'grid': os.path.basename(grid_file),
            'ceiling': os.path.basename(ceiling_file),
            'convergence': os.path.basename(conv_file) if len(convergence_df) else None,
        },
    }
    if isinstance(metadata, dict):
        for key in ('dataset_source', 'split'):
            if key in metadata:
                run_metadata[key] = metadata[key]

    with open(meta_file, 'w', encoding='utf-8') as stream:
        json.dump(run_metadata, stream, indent=2, sort_keys=True, default=str)

    print('\nResults saved to %s' % os.path.relpath(args.out_dir, ROOT))
    print('  grid:        %s' % os.path.relpath(grid_file, ROOT))
    print('  ceiling:     %s' % os.path.relpath(ceiling_file, ROOT))
    if len(convergence_df):
        print('  convergence: %s' % os.path.relpath(conv_file, ROOT))
    print('  metadata:    %s' % os.path.relpath(meta_file, ROOT))
    print('\nwrote %d grid rows, %d ceiling rows, %d convergence rows'
          % (len(grid_df), len(ceiling_df), len(convergence_df)))
    print('total wall seconds: %.1f' % (time.perf_counter() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
