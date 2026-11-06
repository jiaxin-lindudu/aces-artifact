"""Fixed-parameter dense/partial solver and nested large-silo cost studies.

Each fit uses a fresh process; only raw and mean CSVs are written. Historical
results and default dense training stay unchanged. Partial eigsh keeps the same
lowest p eigenvectors, including the lowest one; graph construction stays dense.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.partition_checks import array_hash
from experiments.scripts import grid_search_ours as ours
from experiments.scripts import grid_search_ffcm as ffcm
from experiments.scripts.compare_grid_results import select_complete

SCORES = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']


def cost_view(bundle, clients, local_samples):
    """Disjoint seeded reservoirs; every silo's prefixes are nested across n."""
    permutation = np.random.RandomState(bundle['data_seed']).permutation(len(bundle['X_fit']))
    reservoirs = np.array_split(permutation, clients)
    indices = [part[:local_samples] for part in reservoirs]
    X_list = [bundle['X_fit'][part] for part in indices]
    X_fit = np.vstack(X_list)
    y_fit = np.concatenate([bundle['y_fit'][part] for part in indices])
    offsets = np.cumsum([0] + [len(part) for part in indices])
    owners = [np.arange(offsets[i], offsets[i + 1]) for i in range(clients)]
    view = dict(bundle, clients=clients, X_list=X_list, X_fit=X_fit, y_fit=y_fit,
                X_train=X_fit, y_train=y_fit, train_indices=owners,
                client_indices=owners, median_distance=float(np.median(pdist(X_fit))))
    source_indices = np.concatenate(indices).astype('<i8')
    proof = dict(fit_view_sha256=array_hash(X_fit), fit_label_sha256=array_hash(y_fit),
                 ownership_view_sha256=hashlib.sha256(source_indices.tobytes()).hexdigest(),
                 fit_rows=len(X_fit), view_clients=clients, local_samples=local_samples,
                 partition_family='nested_equal_silo_cost_view',
                 view_rule='seeded source-client-concatenated FIT permutation; disjoint equal reservoirs; nested prefixes')
    return view, proof


def run_worker(job):
    bundle = ours.load_partition_bundle(job['dataset'], job['grid_split'], job['beta'],
                                       job['source_clients'], job['data_dir'])
    if job['study'] == 'large-silo':
        bundle, view_proof = cost_view(bundle, job['view_clients'], job['local_samples'])
    else:
        ownership = np.concatenate(bundle['client_indices']).astype('<i8')
        view_proof = dict(fit_view_sha256=array_hash(bundle['X_fit']),
                         fit_label_sha256=array_hash(bundle['y_fit']),
                         ownership_view_sha256=hashlib.sha256(ownership.tobytes()).hexdigest(),
                         fit_rows=len(bundle['X_fit']), view_clients=bundle['clients'],
                         local_samples=None, partition_family='frozen_dirichlet')
    cfg, seed = dict(job['config']), job['init_seed']
    method, solver = job['method'], job['eigen_solver']
    if job['study'] == 'large-silo':
        cfg['num_rounds' if method == 'ours' else 'n_rounds'] = job['benchmark_rounds']
    process = psutil.Process()
    baseline = process.memory_info().rss
    started = time.perf_counter()
    if method == 'ours':
        metrics, history, _ = ours.run_ours(bundle, cfg, seed, eval_overrides=dict(
            use_ckks=job['use_ckks'], run_privacy_eval=False, eigen_solver=solver),
            return_details=True)
    else:
        kwargs = dict(cfg)
        kwargs['m'] = kwargs.pop('fuzziness')
        centers, history = ffcm.ffcm_clustering(
            X_train=bundle['X_fit'], y_train=bundle['y_fit'],
            X_test=bundle['X_test'], y_test=bundle['y_test'], X_list=bundle['X_list'],
            n_clusters=bundle['K'], random_state=seed, dp_sigma=0.,
            use_tracker=False, record_history=False, use_ckks=job['use_ckks'],
            return_cost=True, **kwargs)
        fit_labels = np.argmin(np.linalg.norm(bundle['X_fit'][:, None] - centers[None], axis=2), axis=1)
        test_labels = np.argmin(np.linalg.norm(bundle['X_test'][:, None] - centers[None], axis=2), axis=1)
        fit = ffcm.clustering_quality(bundle['X_fit'], bundle['y_fit'], fit_labels)
        test = ffcm.clustering_quality(bundle['X_test'], bundle['y_test'], test_labels)
        metrics = dict(**fit, **{'test_' + key: value for key, value in test.items()},
                       rounds_run=cfg['n_rounds'], stop_reason='round_limit',
                       test_basis='nearest_center', init_uses_pooled_fit=False)
    wall = time.perf_counter() - started
    memory = process.memory_info()
    if sys.platform == 'win32':
        peak = memory.peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == 'darwin' else 1024
    cost, comm = history['execution_cost'], history['communication']
    metrics['wall_seconds'] = wall
    row = dict(**metrics,
        trainer_seconds=cost['trainer_seconds'], evaluation_seconds=cost['evaluation_seconds'],
        training_core_seconds=cost['trainer_seconds'] - cost['evaluation_seconds'],
        setup_seconds=cost['setup_seconds'], peak_rss_bytes=int(peak),
        baseline_rss_bytes=int(baseline), actual_config=json.dumps(cfg, sort_keys=True),
        memory_basis='fresh-worker lifetime peak RSS; imports/source data/view/evaluation included; not incremental training RAM',
        timing_basis='single-host serial; core includes setup and excludes offline evaluation; no attacks',
        budget_basis=('fixed source FIT-selected parameters and native round budget' if job['study'] == 'solver'
                      else 'fixed source FIT-selected parameters; equal maximum round budget; retain actual completed rounds'),
        hardware=platform.platform() + ' | ' + platform.processor(),
        logical_cpus=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
        python_version=platform.python_version(), numpy_version=np.__version__,
        scipy_version=scipy.__version__, blas_omp_threads=1,
        graph_storage='dense', eigen_includes_lowest=True, eigen_tolerance=1e-8,
        eigen_start='isolated seed/client/graph-size stream',
        protection_scope=('single-process aggregate-only CKKS; public context and separate key-authority object'
                          if job['use_ckks'] else 'plaintext aggregation'),
        **view_proof,
        **{key: bundle['provenance'][key] for key in ('data_sha256', 'partition_sha256')})
    for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes', 'interservice_bytes',
                'total_bytes', 'packed_values', 'public_context_bytes',
                'slots_per_ciphertext', 'ciphertexts_per_client_per_round'):
        row[key] = comm.get(key, np.nan)
    row.update(communication_basis=comm['measurement_basis'],
               communication_excluded=comm['excluded'],
               bytes_per_round=(comm['total_bytes'] - comm['setup_bytes']) / metrics['rounds_run'])
    row.update({key + '_seconds': value for key, value in cost['round_substage_seconds'].items()})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', choices=('solver', 'large-silo'), default='solver')
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5, help='source partition concentration')
    parser.add_argument('--clients', type=int, default=20, help='source frozen client count; use 50 for USPS')
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir')
    parser.add_argument('--ffcm-source-dir',
                        help='FFCM reference directory for large-silo; defaults to --source-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--aggregation', choices=('plain', 'ckks'), default='ckks')
    parser.add_argument('--solvers', nargs='+', choices=('dense', 'partial'), default=['dense', 'partial'])
    parser.add_argument('--local-samples', nargs='+', type=int, default=[128, 512, 1024, 2048])
    parser.add_argument('--view-clients', type=int, default=3)
    parser.add_argument('--benchmark-rounds', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        with contextlib.redirect_stdout(io.StringIO()):
            row = run_worker(json.load(sys.stdin))
        print(json.dumps(row))
        return 0
    if not args.source_dir or not args.out_dir:
        parser.error('--source-dir and --out-dir are required')
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    if len(args.solvers) != len(set(args.solvers)):
        parser.error('--solvers must be distinct')
    if (args.clients < 1 or args.view_clients < 1 or args.benchmark_rounds < 1
            or not args.local_samples or min(args.local_samples) < 1
            or len(args.local_samples) != len(set(args.local_samples))):
        parser.error('client counts, round budget and distinct local sample counts must be positive')
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta, args.clients, args.data_dir)
    proof = {key: bundle['provenance'][key] for key in ('data_sha256', 'partition_sha256')}
    source_identity = dict(dataset=args.dataset, data_seed=args.grid_split, beta=args.beta, clients=args.clients)
    methods = ('ours', 'ffcm') if args.study == 'large-silo' else ('ours',)
    configs, sources = {}, {}
    for method in methods:
        module = ours if method == 'ours' else ffcm
        source_dir = args.ffcm_source_dir if method == 'ffcm' and args.ffcm_source_dir else args.source_dir
        source = Path(source_dir) / f'grid_{method}_beta{args.beta}_clients{args.clients}.csv'
        raw = pd.read_csv(source)
        mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
        for key, value in {**source_identity, **proof}.items():
            if not raw[key].eq(value).all():
                raise ValueError(f'{method} source differs from frozen partition: {key}')
        # Select on the complete source grid, even when a smaller run seed subset is requested.
        selection_seeds = sorted(raw.init_seed.unique().tolist())
        selected, cfg = select_complete(raw, mean, module.GRID_PARAMS, selection_seeds, args.select_on)
        if selected.init_uses_pooled_fit.any():
            raise ValueError('Pooled initialization is excluded')
        for key, value in cfg.items():
            if key == 'kernel_sigma':
                cfg[key] = None if ours.sigma_label(value) == 'adaptive' else float(value)
            else:
                cfg[key] = type(module.GRID[key][0])(value)
        init = cfg['init_method' if method == 'ours' else 'center_init_method']
        if init not in ('random', 'uniform') or (method == 'ours' and cfg['active_ratio'] != 1.):
            raise ValueError('Require random/uniform initialization and full participation')
        if method == 'ours' and 'eigen_solver' in selected and not selected.eigen_solver.eq('dense').all():
            raise ValueError('This paired study requires a dense reference configuration')
        configs[method], sources[method] = cfg, str(source.resolve())
        print(f'{method}: {module.describe(cfg)}')
    if args.study == 'large-silo':
        if max(args.local_samples) > len(bundle['X_fit']) // args.view_clients:
            parser.error('Each cost view must use unique source FIT rows; insufficient rows for the largest n')
        if configs['ours']['d_emb'] >= min(args.local_samples) + bundle['K']:
            parser.error('partial eigsh requires d_emb < the smallest benchmark local graph')
        views = {}
        for count in args.local_samples:
            _, view_proof = cost_view(bundle, args.view_clients, count)
            views[count] = view_proof
        arms = [('ours', 'dense'), ('ours', 'partial'), ('ffcm', 'not_applicable')]
        conditions = [(f'{method}_{solver}_n{count}', method, solver, count)
                      for count in args.local_samples for method, solver in arms]
        stem = (f'grid_ours_large_silo_{args.aggregation}_sourcebeta{args.beta}_'
                f'sourceclients{args.clients}_viewclients{args.view_clients}_rounds{args.benchmark_rounds}')
        actual_clients = args.view_clients
    else:
        if ('partial' in args.solvers and configs['ours']['d_emb']
                >= min(map(len, bundle['X_list'])) + bundle['K']):
            parser.error('partial eigsh requires d_emb < every local graph')
        conditions = [(solver, 'ours', solver, None) for solver in args.solvers]
        stem = f'grid_ours_solver_{args.aggregation}_beta{args.beta}_clients{args.clients}'
        views = {}
        actual_clients = args.clients
    identity = dict(dataset=args.dataset, data_seed=args.grid_split, beta=args.beta,
                    clients=actual_clients, source_clients=args.clients, study=args.study,
                    use_ckks=args.aggregation == 'ckks', select_on=args.select_on, **proof)
    raw_path = Path(args.out_dir) / (stem + '.csv')
    mean_path = raw_path.with_name(stem + '_mean.csv')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    expected = {(condition, seed) for condition, *_ in conditions for seed in seeds}
    if len(existing):
        keys = list(zip(existing.condition, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
            raise ValueError('Output contains unexpected/duplicate conditions or seeds; use another out-dir')
        for key, value in identity.items():
            if not existing[key].eq(value).all():
                raise ValueError(f'Existing results mismatch: {key}')
        for method, cfg in configs.items():
            if not existing.loc[existing.method.eq(method), 'config'].eq(json.dumps(cfg, sort_keys=True)).all():
                raise ValueError('Source selected configuration changed; use another out-dir')
        for count, view_proof in views.items():
            subset = existing[existing.local_samples.eq(count)]
            for key in ('fit_view_sha256', 'fit_label_sha256', 'ownership_view_sha256'):
                if not subset[key].eq(view_proof[key]).all():
                    raise ValueError('Derived cost view changed; use another out-dir')
        complete = existing.status.eq('completed')
        if not np.isfinite(existing.loc[complete, SCORES].to_numpy(float)).all():
            raise ValueError('Completed results contain nonfinite scores')
    done = (set(zip(existing.loc[existing.status.eq('completed'), 'condition'],
                    existing.loc[existing.status.eq('completed'), 'init_seed'])) if len(existing) else set())
    print(ours.partition_summary(bundle))
    print(f'{len(conditions)} conditions x {len(seeds)} seeds; {len(done)} complete. No hyperparameter search.')
    if args.study == 'large-silo':
        print(f'Cost views: {args.view_clients} equal silos; unique nested FIT; unchanged TEST; '
              f'maximum {args.benchmark_rounds} rounds; source beta is not a new Dirichlet setting.')
    if args.dry_run:
        print('[dry-run] No training or output files.')
        return 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    rows = existing.to_dict('records')
    failed = False
    for index, seed in enumerate(seeds):
        offset = index % len(conditions)
        for condition, method, solver, count in conditions[offset:] + conditions[:offset]:
            if (condition, seed) in done:
                continue
            job = dict(dataset=args.dataset, grid_split=args.grid_split, beta=args.beta,
                       source_clients=args.clients, data_dir=args.data_dir, init_seed=seed,
                       study=args.study, method=method, eigen_solver=solver, config=configs[method],
                       use_ckks=args.aggregation == 'ckks', local_samples=count,
                       view_clients=args.view_clients, benchmark_rounds=args.benchmark_rounds)
            print(f'Running {condition}, init={seed}', flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                input=json.dumps(job), text=True, encoding='utf-8', capture_output=True,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            row = dict(**identity, condition=condition, method=method, eigen_solver=solver,
                       init_seed=seed, local_samples=count, benchmark_rounds=(
                           args.benchmark_rounds if args.study == 'large-silo' else configs[method]['num_rounds']),
                       config=json.dumps(configs[method], sort_keys=True),
                       reference_source_csv=sources[method], error='')
            if count is not None:
                row.update(views[count])
            if result.returncode:
                row.update({key: np.nan for key in SCORES})
                row.update(status='failed', error=(result.stderr + '\n' + result.stdout)[-4000:])
                failed = True
                print(f'  FAILED: {row["error"]}', flush=True)
            else:
                row.update(json.loads(result.stdout))
                if np.isfinite([row[key] for key in SCORES]).all():
                    row['status'] = 'completed'
                    print(f"  FIT={row['acc']:.4f} TEST={row['test_acc']:.4f} "
                          f"core={row['training_core_seconds']:.2f}s", flush=True)
                else:
                    row.update(status='failed', error='Nonfinite quality score')
                    failed = True
            # A rerun retries failed entries, replacing that condition/seed row only.
            rows = [old for old in rows if (old['condition'], old['init_seed']) != (condition, seed)]
            rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_csv(raw_path, index=False)
            groups = frame.groupby('condition', sort=False)
            complete = frame[frame.status.eq('completed')]
            numeric = [key for key in complete.select_dtypes(include=['number']).columns
                       if key not in ('data_seed', 'init_seed', 'beta', 'clients', 'source_clients')]
            if len(complete):
                means = complete.groupby('condition')[numeric].mean()
                deviation = complete.groupby('condition')[numeric].std(ddof=1).add_suffix('_std')
                summary = means.join(deviation).reset_index()
            else:
                summary = pd.DataFrame(columns=['condition'])
            summary = pd.DataFrame({'condition': [item[0] for item in conditions]}).merge(
                summary, on='condition', how='left')
            summary['n_completed'] = summary.condition.map(groups.status.apply(lambda values: values.eq('completed').sum())).fillna(0).astype(int)
            summary['n_failed'] = summary.condition.map(groups.status.apply(lambda values: values.eq('failed').sum())).fillna(0).astype(int)
            summary['n_expected'] = len(seeds)
            summary['status'] = np.where(summary.n_completed.eq(len(seeds)), 'completed', 'incomplete')
            for key in ('method', 'eigen_solver', 'config', 'reference_source_csv',
                        'partition_family', 'fit_view_sha256', 'fit_label_sha256',
                        'ownership_view_sha256', 'test_basis', 'memory_basis', 'timing_basis',
                        'budget_basis', 'graph_storage'):
                if key in frame:
                    summary[key] = summary.condition.map(groups[key].first())
            for key, value in identity.items():
                summary[key] = value
            summary['uncertainty_scope'] = 'initialization variability within one source frozen split/view'
            summary.to_csv(mean_path, index=False)
    if not len(rows):
        return 1
    # Previously completed runs already have a mean file from their last save.
    print(f'wrote {raw_path} and {mean_path}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
