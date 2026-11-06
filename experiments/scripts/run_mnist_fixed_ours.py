"""Repeat the existing MNIST Ours studies with one screenshot-specified configuration.

Use the frozen mnist_10000 data and the existing experiment entries/CSV schemas.
No grid search, baseline training, paper update or diagnostic files are included.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Set before importing NumPy/SciPy, as in run_remaining_experiments.ps1.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.partition_checks import check_partition
from experiments.scripts import grid_search_ours as ours


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-seeds', default='0,1,2,3,4')
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--out-root', default=str(ROOT / 'experiments/results/mnist_10000'))
    parser.add_argument('--studies', nargs='+', default=['all'], choices=(
        'all', 'main', 'mechanism', 'space', 'anchors', 'participation', 'privacy',
        'privacy_control', 'partition', 'quantity', 'feature', 'samples', 'clients', 'clusters'))
    parser.add_argument('--dry-run', action='store_true',
                        help='check frozen partitions and print the schedule; no training or files')
    args = parser.parse_args(argv)
    data_seeds = [int(value) for value in args.data_seeds.split(',') if value.strip()]
    init_seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if (not data_seeds or len(data_seeds) != len(set(data_seeds))
            or not set(data_seeds).issubset(range(5))
            or not init_seeds or len(init_seeds) != len(set(init_seeds))):
        parser.error('use distinct data seeds in 0..4 and distinct initialization seeds')

    # Same full-model configuration on all five splits. Only the interventions
    # in the existing ablation/robustness entries change their named factor.
    config = {key: values[0] for key, values in ours.GRID.items()}
    config.update(
        anchor_weight=10.0, d_emb=10, lambda_space=0.75, fuzziness=1.1,
        k_neighbors=30, kernel_sigma=6.3170086712, kernel_scale_mode='global',
        num_rounds=10, learning_rate=0.001, init_method='random', active_ratio=1.0,
        topology_boundary_mode='tanh', normalize_spaces=True,
    )
    # Reuse the original single-file grid pipeline with singleton axes. This
    # modifies only this process, never grid_search_ours.py or its default grid.
    ours.GRID.update({key: [value] for key, value in config.items()})

    # (study, module, output stem, partition beta, extra CLI arguments).
    # main must precede dependent studies; privacy precedes its attack controls.
    jobs = [('main', 'grid_search_ours', 'grid_ours_beta0.5_clients20', '0.5',
             ['--mode', 'oat'])]
    for study in ('mechanism', 'space', 'anchors'):
        tag = 'ablation' if study == 'mechanism' else study
        jobs.append((study, 'ablate_ours_best', f'grid_ours_{tag}_beta0.5_clients20',
                     '0.5', ['--study', study]))
    for study in ('participation', 'privacy'):
        jobs.append((study, 'run_ours_privacy_participation',
                     f'grid_ours_{study}_beta0.5_clients20', '0.5', ['--study', study]))
    jobs.append(('privacy_control', 'run_ours_privacy_control',
                 'grid_ours_privacy_control_beta0.5_clients20', '0.5', []))
    for beta in ('0.1', '1.0'):
        jobs.append(('partition', 'run_federated_best', f'grid_ours_fixed_beta{beta}_clients20',
                     beta, ['--method', 'ours', '--study', 'partition', '--reference-beta', '0.5']))
    for study in ('quantity', 'feature'):
        jobs.append((study, 'run_ours_heterogeneity',
                     f'grid_ours_{study}_sourcebeta0.5_clients20', '0.5',
                     ['--method', 'ours', '--study', study]))
    for study in ('samples', 'clients', 'clusters'):
        jobs.append((study, 'run_ours_scale', f'grid_ours_scale_{study}_beta0.5_clients20',
                     '0.5', ['--study', study]))
    jobs = [job for job in jobs if 'all' in args.studies or job[0] in args.studies]

    print('MNIST fixed 10,000-row subset: FIT=8000, TEST=2000; source clients=20.', flush=True)
    print('One full-model configuration; no hyperparameter search.', flush=True)
    print(ours.describe(config), flush=True)
    print('Screenshot fusion_alpha=.75 -> lambda_space=.75; partition beta is separate.', flush=True)
    print('global kernel mode makes the specified kernel_sigma effective.', flush=True)
    for data_seed in data_seeds:
        out = Path(args.out_root) / f'federated_{data_seed}_clients20_mnist_fixed_v2'
        print(f'\nDATA SEED {data_seed} -> {out}', flush=True)
        if args.dry_run:
            for beta in sorted({float(job[3]) for job in jobs}):
                proof = check_partition('mnist_10000', data_seed, beta, 20, args.data_dir)
                print(f'  frozen beta={beta}: data={proof["data_sha256"][:12]} '
                      f'ownership={proof["partition_sha256"][:12]}', flush=True)
        for study, module_name, stem, beta, extra in jobs:
            raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
            if raw_path.exists() and mean_path.exists():
                print(f'  SKIP existing raw/mean pair: {stem}', flush=True)
                continue
            if raw_path.exists() or mean_path.exists():
                raise FileExistsError(f'Incomplete output pair: {stem}; preserve it and choose another --out-root')
            cli = ['--dataset', 'mnist_10000', '--grid-split', str(data_seed),
                   '--beta', beta, '--clients', '20', '--init-seeds', ','.join(map(str, init_seeds)),
                   '--data-dir', args.data_dir, '--out-dir', str(out)]
            if study != 'main':
                cli += ['--source-dir', str(out)]
            if study != 'privacy_control':
                cli += ['--select-on', 'acc']
            cli += extra
            print(f'  {"PLAN" if args.dry_run else "RUN"}: {study} / {stem}', flush=True)
            if args.dry_run:
                continue
            module = importlib.import_module('experiments.scripts.' + module_name)
            result = module.main(cli)
            if result != 0:
                raise RuntimeError(f'MNIST failed: seed={data_seed} / {study}, exit={result}')

    if args.dry_run:
        print('\n[dry-run] no training, attacks or output files.', flush=True)
    else:
        print('\nAll requested MNIST Ours studies finished; only raw/mean CSVs were written.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
