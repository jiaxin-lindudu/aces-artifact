"""Frozen-parameter HAR validation on natural subject clients and unseen subjects.

Transfer one configuration per method from the existing record-level HAR split
0 to all five subject folds. No parameter search or TEST-based selection.
Dense/partial/process-CKKS Ours, five federated baselines and three centralized references
are scored on the same frozen FIT/TEST. Only raw and mean CSVs are saved.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete
from centralized import METHODS as CENTRAL_METHODS
from centralized.grids import GRIDS, describe

CONDITIONS = ('ours_dense', 'ours_partial', 'ours_process_partial', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc',
              'central_kmeans', 'central_fcm', 'central_spectral')
SCORES = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-root', default=str(ROOT / 'experiments/results/har'))
    parser.add_argument('--out-dir')
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--methods', default=','.join(CONDITIONS))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.init_seeds.split(',')]
    conditions = args.methods.split(',')
    if (args.fold not in range(5) or not seeds or len(set(seeds)) != len(seeds)
            or not conditions or len(set(conditions)) != len(conditions)
            or not set(conditions).issubset(CONDITIONS) or not set(seeds).issubset(range(40, 45))):
        parser.error('Use fold 0--4, distinct init seeds 40--44, and declared method names')
    bundle = ours.load_partition_bundle('har_subject', args.fold, 'subject', 24, args.data_dir)
    proof = {k: bundle['provenance'][k] for k in ('data_sha256', 'partition_sha256')}
    source_root = Path(args.source_root)
    reference_bundle = ours.load_partition_bundle('har', 0, .5, 50, args.data_dir)
    reference_identity = dict(dataset='har', data_seed=0, beta=.5, clients=50,
        **{k: reference_bundle['provenance'][k] for k in ('data_sha256', 'partition_sha256')})
    configurations, modules, sources = {}, {}, {}
    for method in dict.fromkeys('ours' if name.startswith('ours_') else name
                                for name in conditions if not name.startswith('central_')):
        module = importlib.import_module('experiments.scripts.grid_search_' + method)
        if method in ('nnfc', 'kfed', 'fedsc'):
            module.configure_grid(bundle['K'])
        suffix = 'ours_tpe' if method == 'ours' else 'reference'
        source = source_root / f'federated_0_clients50_{suffix}' / f'grid_{method}_beta0.5_clients50.csv'
        raw = pd.read_csv(source); mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
        selected, config = select_complete(raw, mean, module.GRID_PARAMS, list(range(40, 45)), 'acc')
        for key, value in reference_identity.items():
            if not selected[key].eq(value).all():
                raise ValueError('Record-level source differs from frozen HAR: ' + key)
        for key, value in config.items():
            if key == 'kernel_sigma':
                config[key] = None if module.sigma_label(value) == 'adaptive' else float(value)
            elif pd.isna(value):
                config[key] = None
            elif module.GRID[key][0] is None:
                config[key] = int(value)
            else:
                config[key] = type(module.GRID[key][0])(value)
        configurations[method] = config; modules[method] = module; sources[method] = str(source.resolve())
    central_source = source_root / 'seed_0_full/centralized_ceiling_har_seed0.csv'
    if any(method.startswith('central_') for method in conditions):
        central = pd.read_csv(central_source)
        for method in conditions:
            if not method.startswith('central_'):
                continue
            selected = central[central.method.eq(method) & central.init_seed.isin(seeds)]
            if (len(selected) != len(seeds) or selected.duplicated('init_seed').any()
                    or not selected.partition_sha256.eq(reference_bundle['provenance']['arrays_sha256']['X_train']).all()):
                raise ValueError('Incomplete/mismatched centralized HAR reference')
            for _, row in selected.iterrows():
                matches = [config for config in GRIDS[method] if describe(method, config) == row.config]
                if len(matches) != 1:
                    raise ValueError('Unknown centralized reference configuration: ' + str(row.config))
                configurations[f'{method}:{int(row.init_seed)}'] = dict(matches[0], max_iter=int(row.max_iter))
            sources[method] = str(central_source.resolve())
    split_meta = bundle['metadata']['split']
    identity = dict(dataset='har_subject', data_seed=args.fold, beta='subject', clients=24,
        fit_rows=len(bundle['X_fit']), test_rows=len(bundle['X_test']),
        training_subjects=json.dumps(split_meta['train_subjects']),
        test_subjects=json.dumps(split_meta['test_subjects']),
        partition_scope='one training subject per client; test subjects disjoint from FIT and across folds',
        statistical_scope='five held-out-subject folds share FIT records; not five independent trials',
        selection_scope='fixed transfer from record-level HAR split0; existing source used the same cohort, not an untouched confirmation',
        source_selection='mean FIT acc over init40--44; centralized per-init frozen source config',
        reference_spec=json.dumps(configurations, sort_keys=True), **proof)
    out = Path(args.out_dir or ROOT / 'experiments/results/har_subject' / f'fold_{args.fold}_clients24_fixed')
    raw_path = out / 'grid_har_subject_validation_clients24.csv'
    mean_path = out / 'grid_har_subject_validation_clients24_mean.csv'
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Use --resume to preserve existing subject results')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    expected = {(name, seed) for name in conditions for seed in seeds}
    if len(existing):
        if existing.duplicated(['condition', 'init_seed']).any() or not set(zip(
                existing.condition, existing.init_seed)).issubset(expected):
            raise ValueError('Duplicate/unexpected saved subject condition')
        for key, value in identity.items():
            if not existing[key].eq(value).all():
                raise ValueError('Existing subject output differs: ' + key)
    rows = existing[existing.status.eq('completed')].to_dict('records') if len(existing) else []
    if rows and not np.isfinite(pd.DataFrame(rows)[SCORES].to_numpy(float)).all():
        raise ValueError('Saved completed subject results contain nonfinite quality')
    done = {(row['condition'], row['init_seed']) for row in rows}
    print(ours.partition_summary(bundle))
    print(f'{len(expected)} fixed fits; {len(done)} saved. No search; no synthetic extra clients.')
    for name in conditions:
        key = 'ours' if name.startswith('ours_') else name
        print(' ', name, 'source=', sources[key])
    if args.dry_run:
        print('[dry-run] no model fitting or result files.'); return 0
    out.mkdir(parents=True, exist_ok=True)
    for name in conditions:
        method = 'ours' if name.startswith('ours_') else name
        for seed in seeds:
            if (name, seed) in done:
                continue
            config = configurations[f'{method}:{seed}' if method.startswith('central_') else method]
            row = dict(**identity, condition=name, method=method, init_seed=seed,
                       config=json.dumps(config, sort_keys=True), reference_source_csv=sources[method],
                       **{k: np.nan for k in SCORES})
            tick = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    if method == 'ours':
                        if name == 'ours_process_partial':
                            from experiments.scripts.run_ours_process_training import run_case
                            metrics = run_case(dict(dataset='har_subject', data_seed=args.fold,
                                beta='subject', clients=24, data_dir=args.data_dir,
                                K=bundle['K'], d_orig=bundle['X_fit'].shape[1], config=config,
                                init_seed=seed, eigen_solver='partial'), 'process')
                        else:
                            metrics = ours.run_ours(bundle, config, seed,
                                eval_overrides=dict(use_ckks=False, run_privacy_eval=False,
                                                    eigen_solver=name.split('_')[1]))
                    elif method in ('fkm', 'ffcm'):
                        metrics = getattr(modules[method], 'run_' + method)(bundle, config, seed)
                    elif not method.startswith('central_'):
                        metrics = modules[method].run_model(bundle, config, seed)
                    else:
                        actual = dict(config)
                        result = CENTRAL_METHODS[method](bundle['X_fit'], bundle['X_test'], bundle['K'], seed, actual)
                        rechecked = bool(result['extra'].get('hit_cap', False) and actual['max_iter'] < 1000)
                        if rechecked:
                            actual['max_iter'] = 1000
                            result = CENTRAL_METHODS[method](bundle['X_fit'], bundle['X_test'], bundle['K'], seed, actual)
                        fit = ours.clustering_quality(bundle['X_fit'], bundle['y_fit'], result['train_labels'])
                        test = ours.clustering_quality(bundle['X_test'], bundle['y_test'], result['test_labels'])
                        metrics = {k: fit[k] for k in ('acc', 'nmi', 'ari')}
                        metrics.update({'test_' + k: test[k] for k in ('acc', 'nmi', 'ari')})
                        metrics.update(test_basis=result['test_basis'], fit_basis='centralized_native',
                            n_distinct_labels=len(np.unique(result['train_labels'])),
                            actual_config=json.dumps(actual, sort_keys=True), budget_rechecked=rechecked,
                            n_iter=result['extra'].get('n_iter'), hit_cap=result['extra'].get('hit_cap', False))
                row.update(metrics, wall_seconds=time.perf_counter() - tick,
                           status='completed', result_origin='trained', error='')
                if not np.isfinite([row[k] for k in SCORES]).all():
                    raise ValueError('Subject validation produced nonfinite quality')
            except Exception:
                row.update(status='failed', error=traceback.format_exc(), wall_seconds=time.perf_counter() - tick)
                print(f'  FAILED {name} init={seed}: {row["error"].splitlines()[-1]}', flush=True)
            rows.append(row); pd.DataFrame(rows).to_csv(raw_path, index=False)
            if row['status'] == 'completed':
                print(f'  {name} init={seed} FIT={row["acc"]:.4f} TEST={row["test_acc"]:.4f} '
                      f'time={row["wall_seconds"]:.1f}s', flush=True)
    frame = pd.DataFrame(rows); good = frame[frame.status.eq('completed')]
    groups = good.groupby('condition', sort=False)
    stats = groups[SCORES + ['wall_seconds']].mean().join(
        groups[SCORES + ['wall_seconds']].std(ddof=1).add_suffix('_std')).reset_index()
    summary = frame[['condition']].drop_duplicates().merge(stats, on='condition', how='left')
    for key, value in identity.items():
        summary[key] = value
    summary['n_seeds'] = summary.condition.map(good.groupby('condition').size()).fillna(0).astype(int)
    summary['n_failed'] = summary.condition.map(frame[frame.status.eq('failed')].groupby('condition').size()).fillna(0).astype(int)
    summary['n_expected'] = len(seeds)
    summary['complete'] = summary.n_seeds.eq(len(seeds)) & summary.n_failed.eq(0)
    summary.to_csv(mean_path, index=False)
    print('wrote', raw_path, 'and', mean_path)
    return 0 if summary.complete.all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
