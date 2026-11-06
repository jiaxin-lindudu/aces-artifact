"""Four fixed-best conditions: Ours/FFCM x plaintext/CKKS, on one frozen split.

Each fit runs in a fresh process. Only raw and mean CSVs are written; completed
runs resume by condition and initialization seed. No attacks or parameter search.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Apply before importing numerical libraries, including in fresh workers.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import grid_search_ours as ours
from experiments.scripts import grid_search_ffcm as ffcm
from experiments.scripts.compare_grid_results import select_complete

CONDITIONS = ('ours_plain', 'ours_ckks', 'ffcm_plain', 'ffcm_ckks')
SCORES = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']


def run_worker(job):
    bundle = ours.load_partition_bundle(job['dataset'], job['grid_split'], job['beta'],
                                       job['clients'], job['data_dir'])
    process = psutil.Process()
    baseline = process.memory_info().rss
    cfg, seed = job['config'], job['init_seed']
    ckks = job['condition'].endswith('_ckks')
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if job['condition'].startswith('ours_'):
            metrics, history, _ = ours.run_ours(bundle, cfg, seed,
                eval_overrides=dict(use_ckks=ckks, run_privacy_eval=False), return_details=True)
        else:
            kwargs = dict(cfg)
            kwargs['m'] = kwargs.pop('fuzziness')
            centers, history = ffcm.ffcm_clustering(
                X_train=bundle['X_fit'], y_train=bundle['y_fit'],
                X_test=bundle['X_test'], y_test=bundle['y_test'],
                X_list=bundle['X_list'], n_clusters=bundle['K'], random_state=seed,
                dp_sigma=0., use_tracker=False, record_history=False,
                use_ckks=ckks, return_cost=True, **kwargs)
            fit_labels = np.argmin(np.linalg.norm(
                bundle['X_fit'][:, None] - centers[None], axis=2), axis=1)
            test_labels = np.argmin(np.linalg.norm(
                bundle['X_test'][:, None] - centers[None], axis=2), axis=1)
            fit = ffcm.clustering_quality(bundle['X_fit'], bundle['y_fit'], fit_labels)
            test = ffcm.clustering_quality(bundle['X_test'], bundle['y_test'], test_labels)
            metrics = dict(**fit, **{'test_' + key: value for key, value in test.items()},
                           rounds_run=cfg['n_rounds'], stop_reason='round_limit',
                           test_basis='nearest_center', init_uses_pooled_fit=False)
    wall = time.perf_counter() - started
    memory = process.memory_info()
    if sys.platform == 'win32':
        peak = memory.peak_wset
        memory_basis = 'Windows fresh-worker lifetime peak working set; imports/data/evaluation included'
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == 'darwin' else 1024
        memory_basis = 'fresh-worker lifetime peak RSS; imports/data/evaluation included'
    cost, comm = history['execution_cost'], history['communication']
    metrics['wall_seconds'] = wall
    row = dict(**metrics, condition=job['condition'], method=job['condition'].split('_')[0],
        use_ckks=ckks,
        trainer_seconds=cost['trainer_seconds'], evaluation_seconds=cost['evaluation_seconds'],
        training_core_seconds=cost['trainer_seconds'] - cost['evaluation_seconds'],
        setup_seconds=cost['setup_seconds'],
        final_scoring_and_wrapper_seconds=wall - cost['trainer_seconds'],
        peak_rss_bytes=int(peak), baseline_rss_bytes=int(baseline), memory_basis=memory_basis,
        timing_basis='single-host serial, fresh worker; core excludes offline evaluation; no attacks',
        budget_basis='each method retains its own FIT-selected round budget; not equal total budgets',
        protection_scope='aggregate-only CKKS simulation; public aggregator, separate key-authority object; not DP or isolated services',
        hardware=platform.platform() + ' | ' + platform.processor(),
        logical_cpus=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
        python_version=platform.python_version(), numpy_version=np.__version__,
        blas_omp_threads=1, ckks_threading='TenSEAL default context pool, same for both methods',
        **{key: bundle['provenance'][key]
                               for key in ('data_sha256', 'partition_sha256')})
    for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes', 'interservice_bytes',
                'total_bytes', 'packed_values', 'public_context_bytes', 'context_recipients',
                'slots_per_ciphertext', 'ciphertexts_per_client_per_round'):
        row[key] = comm.get(key, np.nan)
    row.update(communication_basis=comm['measurement_basis'], communication_excluded=comm['excluded'],
        bytes_reproducible=comm['bytes_reproducible'],
        bytes_per_round=(comm['total_bytes'] - comm['setup_bytes']) / metrics['rounds_run'],
        core_seconds_per_round=(row['training_core_seconds'] - row['setup_seconds']) / metrics['rounds_run'],
        packing_efficiency=(comm['packed_values'] / (comm['slots_per_ciphertext'] *
            comm['ciphertexts_per_client_per_round']) if ckks else np.nan),
        round_times_trace=json.dumps(history['round_times']),
        round_resources_trace=json.dumps(history['resource_breakdown']))
    row.update({key + '_seconds': value for key, value in cost['round_substage_seconds'].items()})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--ffcm-source-dir',
                        help='FFCM reference directory; defaults to --source-dir')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        # JSON travels through pipes only; no intermediate files.
        with contextlib.redirect_stdout(io.StringIO()):
            row = run_worker(json.load(sys.stdin))
        print(json.dumps(row))
        return 0
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('--init-seeds must contain distinct seeds')
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split, beta=args.beta, clients=args.clients)
    proof = {key: bundle['provenance'][key] for key in ('data_sha256', 'partition_sha256')}
    configs, sources = {}, {}
    for name, module in (('ours', ours), ('ffcm', ffcm)):
        source_dir = args.ffcm_source_dir if name == 'ffcm' and args.ffcm_source_dir else args.source_dir
        source = Path(source_dir) / f'grid_{name}_beta{args.beta}_clients{args.clients}.csv'
        raw = pd.read_csv(source)
        mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
        for key, value in {**identity, **proof}.items():
            if not raw[key].eq(value).all():
                raise ValueError(f'{name} source differs from frozen partition: {key}')
        selected, cfg = select_complete(raw, mean, module.GRID_PARAMS, seeds, args.select_on)
        if selected.init_uses_pooled_fit.any():
            raise ValueError('Pooled initialization is excluded')
        for key, value in cfg.items():
            cfg[key] = (None if ours.sigma_label(value) == 'adaptive' else float(value)) if key == 'kernel_sigma' else type(module.GRID[key][0])(value)
        init = cfg['init_method' if name == 'ours' else 'center_init_method']
        if init not in ('random', 'uniform') or (name == 'ours' and cfg['active_ratio'] != 1.):
            raise ValueError('Require random/uniform initialization and full participation')
        configs[name], sources[name] = cfg, str(source.resolve())
        print(f'{name}: {module.describe(cfg)}')
    out = Path(args.out_dir)
    stem = f'grid_ckks_cost_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    expected = {(condition, seed) for condition in CONDITIONS for seed in seeds}
    if len(existing):
        keys = list(zip(existing.condition, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
            raise ValueError('Output contains duplicate/unexpected conditions or seeds; use another out-dir')
        for key, value in {**identity, **proof, 'select_on': args.select_on}.items():
            if not existing[key].eq(value).all():
                raise ValueError(f'Existing results mismatch: {key}')
        for name, cfg in configs.items():
            if not existing.loc[existing.method.eq(name), 'config'].eq(json.dumps(cfg, sort_keys=True)).all():
                raise ValueError('Selected configuration changed; use another out-dir')
        if not np.isfinite(existing[SCORES].to_numpy(float)).all():
            raise ValueError('Existing results have nonfinite scores')
    done = set(zip(existing.condition, existing.init_seed)) if len(existing) else set()
    print(ours.partition_summary(bundle))
    print(f'4 conditions x {len(seeds)} seeds; {len(done)} complete. No grid search.')
    if args.dry_run:
        print('[dry-run] No training or output files.')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    rows = existing.to_dict('records')
    # Rotate condition order across seeds, avoiding a fixed warm/cold order.
    for index, seed in enumerate(seeds):
        order = CONDITIONS[index % 4:] + CONDITIONS[:index % 4]
        for condition in order:
            if (condition, seed) in done:
                continue
            name = condition.split('_')[0]
            job = dict(dataset=args.dataset, grid_split=args.grid_split, beta=args.beta,
                clients=args.clients, data_dir=args.data_dir, condition=condition,
                init_seed=seed, config=configs[name])
            print(f'Running {condition}, init={seed}', flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                *sys.argv[1:], '--worker'], input=json.dumps(job), text=True,
                encoding='utf-8', capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            if result.returncode:
                raise RuntimeError(f'{condition} init={seed}\n{result.stderr}\n{result.stdout}')
            row = json.loads(result.stdout)
            row.update(**identity, init_seed=seed, config=json.dumps(configs[name], sort_keys=True),
                       reference_source_csv=sources[name], select_on=args.select_on, status='completed')
            if not np.isfinite([row[key] for key in SCORES]).all():
                raise ValueError(f'{condition} produced nonfinite scores')
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"  FIT={row['acc']:.4f} TEST={row['test_acc']:.4f} core={row['training_core_seconds']:.2f}s bytes={row['total_bytes']}", flush=True)
    frame = pd.DataFrame(rows)
    numeric = [key for key in frame.select_dtypes(include=['number']).columns
               if key not in ('data_seed', 'init_seed', 'beta', 'clients')]
    groups = frame.groupby('condition', sort=False)
    summary = groups[numeric].mean().reset_index()
    deviation = groups[numeric].std(ddof=1).reset_index()
    summary = summary.merge(deviation, on='condition', suffixes=('', '_std'))
    for key in ('method', 'config', 'reference_source_csv', 'test_basis', 'timing_basis',
                'budget_basis', 'communication_basis', 'communication_excluded', 'memory_basis',
                'hardware', 'python_version', 'numpy_version', 'protection_scope'):
        summary[key] = summary.condition.map(groups[key].agg(lambda values: '; '.join(map(str, pd.unique(values)))))
    for key, value in {**identity, **proof, 'select_on': args.select_on}.items():
        summary[key] = value
    summary['n_seeds'] = len(seeds)
    summary['uncertainty_scope'] = 'initialization variability within one frozen split'
    summary.to_csv(mean_path, index=False)
    print(summary[['condition', 'test_acc', 'training_core_seconds', 'total_bytes']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
