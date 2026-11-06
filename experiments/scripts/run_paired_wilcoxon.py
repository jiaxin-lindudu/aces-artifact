"""Fixed-configuration initialization validation, followed by paired Wilcoxon.

Train init seeds 100--109 using configurations selected on FIT/init40--44.
The statistical unit is one new seed averaged over the five frozen splits.
No parameter search, new partition, encryption experiment or paper edit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.partition_checks import check_partition
from experiments.scripts.compare_grid_results import select_complete

CLIENTS = dict(digits=20, usps=50, coil20=20, mnist_10000=20, har=50)
METHODS = ('ours', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc')
REFERENCE_SEEDS = tuple(range(40, 45))
EVAL_SEEDS = tuple(range(100, 110))
SCORES = ('acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari')
PROTOCOL = 'aces_fixed_partition_new_initializations_v1'
SCOPE = 'initialization variability conditional on five fixed partitions and frozen FIT-selected configurations'


def result_path(results_root, dataset, data_seed, method):
    clients = CLIENTS[dataset]
    return (Path(results_root) / dataset / f'federated_{data_seed}_clients{clients}_paired_validation'
            / f'grid_{method}_paired_validation_beta0.5_clients{clients}.csv')


def read_reference(dataset, data_seed, method, proof, source_dir=None):
    """Validate the external reference once, then return its native configuration."""
    clients = CLIENTS[dataset]
    if source_dir is None:
        if dataset == 'digits':
            folder = f'federated_{data_seed}_heterogeneity'
        elif dataset == 'usps':
            folder = f'federated_{data_seed}_clients50_heterogeneity'
        elif dataset == 'har':
            suffix = 'ours_tpe' if method == 'ours' else 'reference'
            folder = f'federated_{data_seed}_clients50_{suffix}'
        else:
            folder = f'federated_{data_seed}_ours'
        source_dir = ROOT / 'experiments/results' / dataset / folder
    source = Path(source_dir) / f'grid_{method}_beta0.5_clients{clients}.csv'
    mean_source = source.with_name(source.stem + '_mean.csv')
    raw, mean = pd.read_csv(source), pd.read_csv(mean_source)
    module = importlib.import_module('experiments.scripts.grid_search_' + method)
    identity = dict(dataset=dataset, data_seed=data_seed, beta=.5, clients=clients)
    for key, value in identity.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Reference identity differs: {source} / {key}')
    recorded = all(key in raw for key in ('data_sha256', 'partition_sha256'))
    for key in ('data_sha256', 'partition_sha256'):
        if key in raw and not raw[key].eq(proof[key]).all():
            raise ValueError(f'Reference differs from frozen data: {source} / {key}')
    selected, config = select_complete(raw, mean, module.GRID_PARAMS, REFERENCE_SEEDS, 'acc')
    if 'init_uses_pooled_fit' in selected and selected.init_uses_pooled_fit.any():
        raise ValueError('Reference uses pooled sample initialization')
    for key, value in config.items():
        if key == 'kernel_sigma':
            config[key] = None if module.sigma_label(value) == 'adaptive' else float(value)
        elif pd.isna(value):
            config[key] = None
        elif module.GRID[key][0] is None:
            config[key] = int(value)
        else:
            config[key] = type(module.GRID[key][0])(value)
    initializer = 'init_method' if method == 'ours' else 'center_init_method'
    if method in ('ours', 'fkm', 'ffcm') and config[initializer] not in ('random', 'uniform'):
        raise ValueError('Keep the selected random/uniform initializer')
    evaluation = dict(module.FIXED_EVAL)
    if method == 'ours':
        evaluation.update(use_ckks=False, run_privacy_eval=False, eigen_solver='dense')
    source = source.resolve()
    source_id = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    metadata = dict(
        config=json.dumps(config, sort_keys=True, allow_nan=False),
        evaluation_config=json.dumps(evaluation, sort_keys=True, allow_nan=False),
        reference_source_csv=source_id.as_posix(),
        reference_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_mean_sha256=hashlib.sha256(mean_source.read_bytes()).hexdigest(),
        reference_provenance='recorded run hashes' if recorded else 'legacy reference without complete run hashes',
        reference_select_on='mean FIT acc over init40--44',
        reference_fit_acc=float(selected.acc.mean()),
    )
    return module, config, metadata


def write_results(rows, path, identity, write_raw=True):
    frame = pd.DataFrame(rows)
    if write_raw:
        frame.to_csv(path, index=False)
    good = frame[frame.status.eq('completed')]
    metrics = list(SCORES) + ['wall_seconds']
    record = dict(identity, n_seeds=len(good), n_expected=len(EVAL_SEEDS),
                  n_missing=len(EVAL_SEEDS) - len(good), n_attempts=len(frame),
                  n_failed_attempts=int(frame.status.eq('failed').sum()),
                  complete=len(good) == len(EVAL_SEEDS), uncertainty_basis='sample SD across new initializations on one frozen split')
    record.update({key: good[key].mean() for key in metrics})
    record.update({key + '_std': good[key].std(ddof=1) for key in metrics})
    pd.DataFrame([record]).to_csv(path.with_name(path.stem + '_mean.csv'), index=False)


def run(args):
    from experiments.scripts import grid_search_ours as ours
    clients = CLIENTS[args.dataset]
    if args.clients is not None and args.clients != clients:
        raise ValueError(f'The canonical {args.dataset} client count is {clients}')
    if args.beta != .5:
        raise ValueError('This fixed protocol uses beta=0.5')
    seeds = [int(value.strip()) for value in args.init_seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(EVAL_SEEDS):
        raise ValueError('Use distinct evaluation seeds from 100--109')
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, .5, clients, args.data_dir)
    proof = {key: bundle['provenance'][key] for key in ('data_sha256', 'partition_sha256')}
    module, config, reference = read_reference(args.dataset, args.grid_split, args.method, proof, args.source_dir)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split, beta=.5, clients=clients,
                    method=args.method, protocol=PROTOCOL, **proof, **reference)
    path = result_path(args.results_root, args.dataset, args.grid_split, args.method)
    if args.out_dir is not None:
        path = Path(args.out_dir) / path.name
    mean_path = path.with_name(path.stem + '_mean.csv')
    if (path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Use --resume to preserve existing results')
    if mean_path.exists() and not path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(existing):
        for key, value in identity.items():
            matches = (np.isclose(existing[key].to_numpy(float), value, atol=1e-12, rtol=0)
                       if key == 'reference_fit_acc' else existing[key].eq(value))
            if not matches.all():
                raise ValueError('Saved run identity/configuration changed: ' + key)
        if (not existing.init_seed.isin(EVAL_SEEDS).all()
                or not existing.status.isin(['completed', 'failed']).all()
                or existing.duplicated(['init_seed', 'attempt']).any()):
            raise ValueError('Unexpected seed, status or duplicate attempt in saved results')
        good = existing[existing.status.eq('completed')]
        if good.init_seed.duplicated().any() or not np.isfinite(good[list(SCORES)].to_numpy(float)).all():
            raise ValueError('Saved successes contain duplicates or nonfinite quality')
    rows = existing.to_dict('records')
    done = {int(row['init_seed']) for row in rows if row['status'] == 'completed'}
    display = 'ACES' if args.method == 'ours' else args.method
    print(ours.partition_summary(bundle), flush=True)
    print(f'{display}: {len(seeds)} requested seeds; {len(done.intersection(seeds))} already complete.', flush=True)
    print('Frozen FIT-selected configuration:', identity['config'], flush=True)
    print('Reference:', reference['reference_source_csv'], flush=True)
    print('Output:', path, flush=True)
    if args.dry_run:
        print('[dry-run] Reference and partition checked; no training or files.')
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = False
    for seed in seeds:
        if seed in done:
            continue
        row = dict(identity, init_seed=seed,
                   attempt=1 + sum(int(old['init_seed']) == seed for old in rows),
                   result_origin='trained', **{key: np.nan for key in SCORES})
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if args.method == 'ours':
                    metrics = module.run_ours(bundle, config, seed, eval_overrides=dict(
                        use_ckks=False, run_privacy_eval=False, eigen_solver='dense'))
                elif args.method in ('fkm', 'ffcm'):
                    metrics = getattr(module, 'run_' + args.method)(bundle, config, seed)
                else:
                    metrics = module.run_model(bundle, config, seed)
            row.update(metrics)
            if (not np.isfinite([row[key] for key in SCORES]).all()
                    or row.get('init_uses_pooled_fit', False)
                    or row.get('status', 'completed') != 'completed'):
                raise ValueError('Nonfinite quality, pooled initialization or failed native outcome')
            row.update(status='completed', error='')
            done.add(seed)
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
        row['wall_seconds'] = time.perf_counter() - started
        rows.append(row)
        write_results(rows, path, identity)
        updated = True
        if row['status'] == 'completed':
            print(f'  init={seed}: FIT={row["acc"]:.4f} TEST={row["test_acc"]:.4f} '
                  f'time={row["wall_seconds"]:.2f}s', flush=True)
        else:
            print(f'  FAILED init={seed}: {row["error"]}', flush=True)
    # Also rebuild a missing/stale companion mean when every requested fit is reused.
    if rows and not updated:
        write_results(rows, path, identity, write_raw=False)
    return 0 if set(seeds).issubset(done) else 1


def paired_statistics(differences):
    """Two-sided signed-rank test using every sign assignment, including ties."""
    from scipy.stats import PermutationMethod, wilcoxon
    differences = np.round(np.asarray(differences, dtype=float), 12)
    if not np.any(differences):
        return 0., 1.
    result = wilcoxon(differences, zero_method='wilcox', alternative='two-sided',
                      method=PermutationMethod(n_resamples=np.inf, batch=1024))
    return float(result.statistic), float(result.pvalue)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind='stable')
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def summarize(args):
    blocks, records = [], []
    digest = hashlib.sha256()
    for dataset, clients in CLIENTS.items():
        for data_seed in range(5):
            proof = check_partition(dataset, data_seed, .5, clients, args.data_dir)
            for method in METHODS:
                path = result_path(args.results_root, dataset, data_seed, method)
                frame = pd.read_csv(path)
                _, _, reference = read_reference(dataset, data_seed, method, proof)
                identity = dict(dataset=dataset, data_seed=data_seed, beta=.5, clients=clients,
                                method=method, protocol=PROTOCOL, **reference,
                                **{key: proof[key] for key in ('data_sha256', 'partition_sha256')})
                for key, value in identity.items():
                    matches = (np.isclose(frame[key].to_numpy(float), value, atol=1e-12, rtol=0)
                               if key == 'reference_fit_acc' else frame[key].eq(value))
                    if not matches.all():
                        raise ValueError(f'Summary input differs: {path} / {key}')
                if (not frame.status.isin(['completed', 'failed']).all()
                        or not frame.init_seed.isin(EVAL_SEEDS).all()
                        or frame.duplicated(['init_seed', 'attempt']).any()):
                    raise ValueError(f'Invalid attempts/seeds: {path}')
                good = frame[frame.status.eq('completed')]
                if (len(good) != len(EVAL_SEEDS) or sorted(good.init_seed) != list(EVAL_SEEDS)
                        or not np.isfinite(good[list(SCORES)].to_numpy(float)).all()):
                    raise ValueError(f'Incomplete comparison: {path}; need every init100--109 exactly once')
                blocks.append(good)
                digest.update(str(path.relative_to(Path(args.results_root))).encode())
                digest.update(path.read_bytes())
    all_runs = pd.concat(blocks, ignore_index=True)
    # The split axis is averaged inside each new-seed unit; it never becomes n=50.
    per_seed = all_runs.groupby(['dataset', 'method', 'init_seed'])[list(SCORES)].mean()
    for dataset, clients in CLIENTS.items():
        ours = per_seed.loc[(dataset, 'ours')].reindex(EVAL_SEEDS)
        for baseline in METHODS[1:]:
            other = per_seed.loc[(dataset, baseline)].reindex(EVAL_SEEDS)
            delta = np.round((ours.test_acc - other.test_acc).to_numpy(float), 12)
            statistic, pvalue = paired_statistics(delta)
            record = dict(dataset=dataset, clients=clients, beta=.5, method='ours', display_method='ACES',
                          baseline=baseline, primary_metric='TEST ACC', n_pairs=len(EVAL_SEEDS),
                          n_data_seeds=5, runs_per_method=50, protocol=PROTOCOL,
                          init_seeds=','.join(map(str, EVAL_SEEDS)), data_seeds='0,1,2,3,4',
                          paired_mean_diff_pp=float(delta.mean() * 100),
                          paired_median_diff_pp=float(np.median(delta) * 100),
                          wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
                          wilcoxon_statistic=statistic, p_value=pvalue,
                          paired_unit='new initialization seed, mean of five fixed splits',
                          test='two-sided Wilcoxon; exhaustive sign permutations; wilcox zero policy',
                          difference_round_decimals=12, statistical_scope=SCOPE,
                          source_raw_sha256=digest.hexdigest())
            for metric in ('test_acc', 'test_nmi', 'test_ari'):
                record['ours_' + metric + '_mean'] = float(ours[metric].mean())
                record['baseline_' + metric + '_mean'] = float(other[metric].mean())
                record['ours_' + metric + '_seed_std'] = float(ours[metric].std(ddof=1))
                record['baseline_' + metric + '_seed_std'] = float(other[metric].std(ddof=1))
            records.append(record)
    result = pd.DataFrame(records)
    result['p_holm'] = holm_adjust(result.p_value.to_numpy())
    result['reject_0_05'] = result.p_holm < .05
    result['correction_family'] = '25 predeclared dataset-by-baseline TEST ACC comparisons'
    output = Path(args.results_root) / 'paired_wilcoxon.csv'
    print(f'Complete: {len(all_runs)} fits; {len(result)} comparisons; 10 paired seed units each.')
    print('Scope:', SCOPE)
    print(result[['dataset', 'baseline', 'paired_mean_diff_pp', 'p_value', 'p_holm']].to_string(index=False))
    if args.dry_run:
        print('[dry-run] Complete inputs checked; no summary file written.')
    else:
        result.to_csv(output, index=False)
        print('Wrote', output)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(CLIENTS), default='digits')
    parser.add_argument('--method', choices=METHODS, default='ours')
    parser.add_argument('--grid-split', type=int, choices=range(5), default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, help='Must match the canonical count for this dataset')
    parser.add_argument('--init-seeds', default=','.join(map(str, EVAL_SEEDS)),
                        help='Distinct seeds from 100--109; the final summary always requires all ten')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', help='Override the historical grid directory for this one task')
    parser.add_argument('--results-root', default=str(ROOT / 'experiments/results'))
    parser.add_argument('--out-dir', help='Override this task output directory; not used by the full summary')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true',
                        help='Check all 1500 successes and write one 25-comparison CSV; no training')
    args = parser.parse_args(argv)
    if args.summarize_only and (args.source_dir or args.out_dir):
        parser.error('--summarize-only uses --results-root and the canonical references')
    return summarize(args) if args.summarize_only else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
