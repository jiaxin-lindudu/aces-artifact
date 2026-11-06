"""Small, predeclared reference comparison on frozen Covtype-10k or HAR data.

By default each method retains at most four fixed candidates. --expand-ours adds
eight FIT-guided neighbors to the original four Ours candidates. Select by mean
FIT over the same initialization seeds. Only raw and _mean CSVs are written. Prepare
the README Stage 1/2 files first; --dry-run never loads or downloads data.
HAR runs the five baselines here; its Ours entry is tune_covtype_ours.py --dataset har.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.partition_checks import PROVENANCE_COLUMNS
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete

METHODS = ('ours', 'ffcm', 'fedsc', 'kfed', 'fkm', 'nnfc')
SCORES = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', choices=('covtype_10000', 'har'), default='covtype_10000')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--methods', default=None,
                        help='Comma-separated methods; HAR defaults to the five baselines, excluding Ours')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--out-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--trainer-logs', action='store_true')
    parser.add_argument('--expand-ours', action='store_true',
                        help='Optional 12-point Ours grid; the adaptive TPE entry is tune_covtype_ours.py')
    parser.add_argument('--eigen-solver', choices=('dense', 'partial'), default='dense',
                        help='Ours only; keep dense for comparison with historical main results')
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    default_methods = METHODS if args.dataset == 'covtype_10000' else METHODS[1:]
    methods = [value.strip() for value in (args.methods or ','.join(default_methods)).split(',')
               if value.strip()]
    if (not seeds or len(set(seeds)) != len(seeds) or not methods
            or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS)
            or args.clients < 1 or not np.isfinite(args.beta) or args.beta <= 0):
        parser.error('Use distinct seeds, known methods, positive clients and beta')
    if args.dataset == 'har' and ('ours' in methods or args.expand_ours):
        parser.error('HAR Ours uses adaptive TPE: tune_covtype_ours.py --dataset har')

    # Dataset contracts determine candidate counts without reading true centers.
    K, n_fit, n_test, features = ((7, 8000, 2000, 54) if args.dataset == 'covtype_10000'
                                 else (6, 8239, 2060, 64))
    jobs = []
    for method in methods:
        module = importlib.import_module('experiments.scripts.grid_search_' + method)
        module.SHOW_TRAINER_LOGS = args.trainer_logs
        if method in ('fedsc', 'kfed', 'nnfc'):
            module.configure_grid(K)
        base = {key: values[0] for key, values in module.GRID.items()}
        if method == 'ours':
            base.update(fuzziness=1.1, d_emb=4, num_rounds=20,
                        kernel_sigma=None, kernel_scale_mode='self_tuning',
                        k_neighbors=5, anchor_weight=1., normalize_spaces=True,
                        learning_rate=.001, active_ratio=1.)
            configs = [{**base, 'lambda_space': lam, 'init_method': init}
                       for lam in (.5, .9) for init in ('random', 'uniform')]
            # The original candidate_0 (lambda=.5, random) leads FIT in all five
            # completed splits. Keep candidate_0..3 unchanged for --resume, then
            # vary one axis at a time around that configuration; no TEST tuning.
            if args.expand_ours:
                neighborhood = configs[0]
                configs += [{**neighborhood, **change} for change in (
                    {'fuzziness': 1.05}, {'fuzziness': 1.2},
                    {'lambda_space': .2}, {'lambda_space': .35},
                    {'d_emb': 2}, {'d_emb': 6},
                    {'k_neighbors': 10}, {'anchor_weight': .5},
                )]
        elif method == 'ffcm':
            base.update(n_rounds=20, check_collapse=True, min_separation=.05)
            configs = [{**base, 'fuzziness': m, 'center_init_method': init}
                       for m in (1.1, 1.3) for init in ('random', 'uniform')]
        elif method == 'fedsc':
            configs = [{**base, 'S': rounds, 'sigma': sigma, 'd': 3 * K}
                       for rounds in (10, 20) for sigma in (1., 5.)]
        elif method == 'kfed':
            configs = [{**base, 'local_clusters': count}
                       for count in dict.fromkeys((int(np.sqrt(K)), int(np.ceil(K / 2)), K))]
        elif method == 'fkm':
            base.update(n_rounds=50, Q=5, B=32, alpha=None,
                        learning_rate=.5, weighted=True)
            configs = [{**base, 'method': variant, 'center_init_method': init}
                       for variant in ('ma', 'gs') for init in ('random', 'uniform')]
        else:
            configs = [{**base, 'k_neighbors': count} for count in (2, 5, 10, 20)]
        jobs.append((method, module, configs))
        print(f'{method}: {len(configs)} fixed candidates x {len(seeds)} initializations')
        for index, config in enumerate(configs):
            print(f'  candidate_{index}: {module.describe(config)}')
    print(f'Total: {sum(len(configs) for _, _, configs in jobs) * len(seeds)} fits on one frozen split; '
          f'select on mean FIT/{args.select_on}, never TEST.')
    if 'ours' in methods:
        print(f'Ours eigen solver: {args.eigen_solver}; quality study uses plaintext aggregation.')
    if args.dry_run:
        return 0

    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    if (bundle['K'] != K or bundle['X_fit'].shape != (n_fit, features)
            or bundle['X_test'].shape != (n_test, features)):
        raise ValueError(f'Expected frozen {args.dataset}: FIT {n_fit}x{features}, '
                         f'TEST {n_test}x{features}, K={K}')
    print(ours.partition_summary(bundle))
    out = Path(args.out_dir or ROOT / 'experiments/results' / args.dataset
               / f'federated_{args.grid_split}_reference')
    identity = dict(split=bundle['label'], dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    proof = {key: bundle['provenance'][key] for key in PROVENANCE_COLUMNS}
    failed = False
    for method, module, configs in jobs:
        axes = list(module.GRID_PARAMS)
        condition_configs = {f'candidate_{index}': json.dumps(cfg, sort_keys=True)
                             for index, cfg in enumerate(configs)}
        expected = {(condition, seed) for condition in condition_configs for seed in seeds}
        raw_path = out / f'grid_{method}_beta{float(args.beta)}_clients{args.clients}.csv'
        mean_path = raw_path.with_name(raw_path.stem + '_mean.csv')
        if (raw_path.exists() or mean_path.exists()) and not args.resume:
            raise FileExistsError('Preserve existing results: use --resume or another --out-dir')
        if mean_path.exists() and not raw_path.exists():
            raise FileNotFoundError('Cannot resume without the raw CSV')
        existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
        evaluation = json.dumps({**module.FIXED_EVAL, **(
            {'eigen_solver': args.eigen_solver} if method == 'ours' else {})}, sort_keys=True)
        if len(existing):
            keys = list(zip(existing.condition, existing.init_seed))
            if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
                raise ValueError('Existing CSV has duplicate or unexpected candidate/seed rows')
            for key, value in {**identity, **proof, 'select_on': args.select_on,
                               'evaluation_config': evaluation}.items():
                if not existing[key].eq(value).all():
                    raise ValueError('Existing CSV differs from this study: ' + key)
            if not existing.condition_config.eq(existing.condition.map(condition_configs)).all():
                raise ValueError('Candidate parameters changed; use another --out-dir')
        rows = existing.to_dict('records')
        out.mkdir(parents=True, exist_ok=True)
        for index, config in enumerate(configs):
            condition = f'candidate_{index}'
            for seed in seeds:
                previous = next((row for row in rows if row['condition'] == condition
                                 and row['init_seed'] == seed), None)
                if (previous is not None and previous['status'] == 'completed'
                        and np.isfinite([previous[key] for key in SCORES]).all()):
                    continue
                attempt = int(previous.get('attempt_count', 1)) + 1 if previous else 1
                if previous is not None:
                    rows.remove(previous)
                started = time.perf_counter()
                try:
                    if method == 'ours':
                        metrics = module.run_ours(bundle, config, seed,
                                                  eval_overrides={'eigen_solver': args.eigen_solver})
                    elif method == 'ffcm':
                        metrics = module.run_ffcm(bundle, config, seed)
                    elif method == 'fkm':
                        metrics = module.run_fkm(bundle, config, seed)
                    else:
                        metrics = module.run_model(bundle, config, seed)
                    if not np.isfinite([metrics[key] for key in SCORES]).all():
                        raise ValueError('Nonfinite FIT/TEST quality score')
                    metrics.update(status='completed', error='')
                except Exception as error:
                    metrics = {key: np.nan for key in module.RESULT_COLUMNS}
                    metrics.update(status='failed', error=str(error), init_uses_pooled_fit=False)
                    print(f'  FAILED {method}/{condition}/init={seed}: {error}')
                metrics['wall_seconds'] = time.perf_counter() - started
                metrics.setdefault('fit_basis', 'native_dual_space' if method == 'ours' else 'nearest_released_center')
                metrics.setdefault('test_basis', 'nearest_released_center_proxy')
                row_config = {key: ours.sigma_label(value) if key == 'kernel_sigma' else value
                              for key, value in config.items()}
                rows.append({**identity, 'init_seed': seed, **row_config, **metrics, **proof,
                             'condition': condition, 'condition_config': condition_configs[condition],
                             'study': 'nonimage_fixed_candidates', 'select_on': args.select_on,
                             'evaluation_config': evaluation, 'attempt_count': attempt,
                             'eigen_solver': args.eigen_solver if method == 'ours' else 'not_applicable'})
                pd.DataFrame(rows).sort_values(['condition', 'init_seed'], kind='stable').to_csv(raw_path, index=False)
                print(f'  {method}/{condition}/init={seed}: {metrics["status"]}, '
                      f'FIT={metrics.get("acc", np.nan):.4f}, TEST={metrics.get("test_acc", np.nan):.4f}')

        frame = pd.DataFrame(rows).sort_values(['condition', 'init_seed'], kind='stable')
        groups = frame.groupby(axes, dropna=False, sort=False)
        numeric = [key for key in module.RESULT_COLUMNS if key in frame
                   and pd.api.types.is_numeric_dtype(frame[key])]
        mean = groups[numeric].mean().reset_index()
        std = groups[SCORES].std(ddof=1).reset_index()
        mean = mean.merge(std, on=axes, suffixes=('', '_std'), sort=False)
        mean['complete'] = [len(part) == len(seeds) and sorted(part.init_seed) == sorted(seeds)
                            and part.status.eq('completed').all()
                            and np.isfinite(part[SCORES].to_numpy(float)).all()
                            for _, part in groups]
        mean['n_seeds'] = len(seeds)
        mean['eigen_solver'] = args.eigen_solver if method == 'ours' else 'not_applicable'
        mean['config'] = [module.describe(cfg) for cfg in mean[axes].to_dict('records')]
        mean = mean.sort_values(['complete', args.select_on], ascending=[False, False], kind='stable')
        mean.to_csv(mean_path, index=False)
        failed |= not frame.status.eq('completed').all()
        if mean.complete.any():
            chosen, best = select_complete(frame, mean, axes, seeds, args.select_on)
            print(f'BEST {method}: {module.describe(best)} | '
                  f'FIT={chosen[args.select_on].mean():.4f}, TEST ACC={chosen.test_acc.mean():.4f}')
        else:
            print(f'No complete {method} candidate; failures remain in {raw_path.name}')
            failed = True
        print(f'wrote {raw_path} and {mean_path}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
