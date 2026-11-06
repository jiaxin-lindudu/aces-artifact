"""FastLloyd epsilon curve with FIT-selected rounds/radius fixed at epsilon=3.

All four budgets are trained afresh. Initialization seeds do not seed the
method's SystemRandom DP noise; preprocessing remains outside its DP claim.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts import grid_search_fastlloyd as fastlloyd
from experiments.scripts.compare_grid_results import select_complete


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
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    source = Path(args.source_dir) / f'grid_fastlloyd_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    bundle = fastlloyd.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                           args.clients, args.data_dir)
    fastlloyd.configure_grid(bundle['K'], args.dataset)
    fastlloyd.FIXED_EVAL.update(epsilon=3., delta=1e-5)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    expected = dict(identity, effective_epsilon=3., effective_delta=1e-5)
    expected.update({key: bundle['provenance'][key]
                     for key in ('data_sha256', 'partition_sha256')})
    for key, value in expected.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Source CSV differs from the epsilon=3 reference: {key}')

    domain = {key: fastlloyd.FIXED_EVAL[key] for key in ('public_bound', 'public_offset')}
    if args.dataset == 'mnist_10000' and not set(domain).intersection(raw.columns):
        # This historical protocol was recorded before domain columns were added.
        source_domain_basis = 'legacy MNIST documented public_bound=3; public_offset=0'
    else:
        for key, value in domain.items():
            if not raw[key].eq(value).all():
                raise ValueError(f'Source public domain differs from the current protocol: {key}')
        source_domain_basis = 'recorded source public_bound/public_offset'

    _, best = select_complete(raw, mean, fastlloyd.GRID_PARAMS, seeds, args.select_on)
    cfg = {key: type(fastlloyd.GRID[key][0])(value) for key, value in best.items()}
    budgets = [.5, 1., 3., 10.]
    out = Path(args.out_dir)
    stem = f'grid_fastlloyd_budget_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if raw_path.exists() or mean_path.exists():
        raise FileExistsError('Preserve existing results: choose another --out-dir')
    print(fastlloyd.partition_summary(bundle))
    print(f'Fixed at mean FIT/{args.select_on} best epsilon=3 configuration: {cfg}')
    print(f'epsilon={budgets}; delta=1e-5; public domain={domain}; {source_domain_basis}')
    print(f'Train {len(budgets) * len(seeds)} runs; no budget-specific parameter search.')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0

    rows = []
    for epsilon in budgets:
        fastlloyd.FIXED_EVAL['epsilon'] = epsilon
        for seed in seeds:
            fastlloyd.set_all_seeds(seed)
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
                result = fastlloyd.run_model(bundle, cfg, seed)
            result['wall_seconds'] = time.perf_counter() - started
            row = dict(split=bundle['label'], **identity, init_seed=seed, **cfg, **result,
                       epsilon=epsilon, delta=1e-5, reference_source_csv=str(source.resolve()),
                       source_domain_basis=source_domain_basis, select_on=args.select_on)
            row.update({key: bundle['provenance'][key]
                        for key in ('data_sha256', 'partition_sha256')})
            rows.append(row)
            print(f'epsilon={epsilon:g} init={seed} FIT={row["acc"]:.4f} '
                  f'TEST={row["test_acc"]:.4f} clusters={row["n_distinct_labels"]}', flush=True)

    frame = pd.DataFrame(rows)
    scores = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
    if not np.isfinite(frame[scores].to_numpy(float)).all():
        raise ValueError('FastLloyd produced nonfinite FIT/TEST scores')
    numeric = [key for key in fastlloyd.RESULT_COLUMNS
               if pd.api.types.is_numeric_dtype(frame[key])]
    groups = frame.groupby(['epsilon', *fastlloyd.GRID_PARAMS], dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    summary = summary.merge(groups[scores].std(ddof=1).reset_index(),
                            on=['epsilon', *fastlloyd.GRID_PARAMS], suffixes=('', '_std'))
    for key in ('fit_basis', 'test_basis', 'initialization', 'communication_basis', 'privacy_status'):
        summary[key] = groups[key].first().to_numpy()
    summary['delta'] = 1e-5
    summary['n_seeds'] = len(seeds)
    summary['config'] = fastlloyd.describe(cfg)
    summary['reference_source_csv'] = str(source.resolve())
    summary['source_domain_basis'] = source_domain_basis
    summary['select_on'] = args.select_on
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    summary.to_csv(mean_path, index=False)
    print(summary[['epsilon', 'acc', 'test_acc', 'test_nmi', 'test_ari']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
