"""Transfer a mean-FIT-selected federated configuration without another grid.

Partition studies transfer to frozen Dirichlet assignments, optionally changing
the client count while retaining canonical FIT/TEST. Participation studies use
the native FKM/FedSC client-selection controls and four fixed ratios.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.compare_grid_results import select_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--method', choices=('ours', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc', 'fastlloyd'), required=True)
    parser.add_argument('--study', choices=('partition', 'participation'), required=True)
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--reference-beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--reference-clients', type=int,
                        help='source client count; defaults to --clients')
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='save each successful run and skip completed condition/seed rows')
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    if args.study == 'participation' and args.method not in ('fkm', 'fedsc'):
        parser.error('Native partial participation is supported here only for FKM/FedSC')
    reference_clients = args.clients if args.reference_clients is None else args.reference_clients
    if args.clients < 1 or reference_clients < 1:
        parser.error('client counts must be positive')
    if args.study == 'participation' and reference_clients != args.clients:
        parser.error('Changing client count is supported only for --study partition')

    module = importlib.import_module('experiments.scripts.grid_search_' + args.method)
    reference = module.load_partition_bundle(args.dataset, args.grid_split,
                                             args.reference_beta, reference_clients, args.data_dir)
    target = module.load_partition_bundle(args.dataset, args.grid_split,
                                          args.beta, args.clients, args.data_dir)
    sizes = np.asarray([len(indices) for indices in target['client_indices']])
    classes, counts = np.unique(target['y_train'], return_counts=True)
    global_label_distribution = counts / counts.sum()
    local_label_tv = [0.5 * np.abs(np.asarray([
        np.count_nonzero(target['y_train'][indices] == label) for label in classes
    ]) / len(indices) - global_label_distribution).sum() for indices in target['client_indices']]
    partition_stats = dict(fit_rows=len(target['X_fit']), test_rows=len(target['X_test']),
        local_samples_min=int(sizes.min()), local_samples_max=int(sizes.max()),
        local_samples_mean=float(sizes.mean()), local_size_ratio=float(sizes.max() / sizes.min()),
        mean_label_tv=float(np.mean(local_label_tv)),
        client_count_scope='fixed total FIT, frozen Dirichlet assignments; realized label/quantity skew may vary with M')
    if args.method == 'fastlloyd':
        module.configure_grid(reference['K'], args.dataset)
        module.FIXED_EVAL.update(epsilon=3., delta=1e-5)
    elif args.method in ('nnfc', 'kfed', 'fedsc'):
        module.configure_grid(reference['K'])
    source = Path(args.source_dir) / f'grid_{args.method}_beta{args.reference_beta}_clients{reference_clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    expected = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.reference_beta, clients=reference_clients,
                    data_sha256=reference['provenance']['data_sha256'],
                    partition_sha256=reference['provenance']['partition_sha256'])
    # Check the external reference at entry; both views must share canonical FIT/TEST.
    for key, value in expected.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Source CSV differs from its frozen reference: {key}')
    if target['provenance']['data_sha256'] != reference['provenance']['data_sha256']:
        raise ValueError('Target and reference must use the same canonical FIT/TEST arrays')
    selected, best = select_complete(raw, mean, module.GRID_PARAMS, seeds, args.select_on)
    if selected.init_uses_pooled_fit.any():
        raise ValueError('The reference must not use pooled sample initialization')
    if args.method == 'fastlloyd':
        for key, value in dict(effective_epsilon=3., effective_delta=1e-5).items():
            if not raw[key].eq(value).all():
                raise ValueError(f'FastLloyd source differs from the fixed privacy protocol: {key}')
        for key in ('public_bound', 'public_offset'):
            if key in raw and not raw[key].eq(module.FIXED_EVAL[key]).all():
                raise ValueError(f'FastLloyd public domain differs from its reference: {key}')
    for key, value in best.items():
        if key == 'kernel_sigma':
            best[key] = None if module.sigma_label(value) == 'adaptive' else float(value)
        elif pd.isna(value):
            best[key] = None
        elif module.GRID[key][0] is None:
            best[key] = int(value)
        else:
            best[key] = type(module.GRID[key][0])(value)

    reference_config = json.dumps(best, sort_keys=True)
    reuse_reference = (args.study == 'partition' and args.reference_beta == args.beta
                       and reference_clients == args.clients)
    result_origin = 'reused' if reuse_reference else 'trained'
    if (args.method == 'fkm' and reference_clients != args.clients
            and best['alpha'] is not None and best['alpha'] != reference_clients):
        raise ValueError('Client-count transfer requires FKM alpha=None or full source participation')
    if args.study == 'participation':
        conditions = [(f'active_{ratio:g}', ratio) for ratio in (1., .75, .5, .25)]
        tag = 'participation'
        scope = ('FKM native alpha clients per communication round'
                 if args.method == 'fkm' else
                 'FedSC partial Stage I dictionary learning; final dictionary/C collection uses all clients')
    else:
        conditions = [(f'partition_beta_{args.beta:g}', 1.)]
        tag = 'fixed'
        scope = ('reference-beta/client-count mean-FIT configuration transferred to target frozen '
                 'partition; canonical FIT/TEST unchanged; no retuning')
    out = Path(args.out_dir)
    stem = f'grid_{args.method}_{tag}_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if not args.resume and (raw_path.exists() or mean_path.exists()):
        raise FileExistsError('Preserve existing results: choose another --out-dir')
    if args.resume and mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume a mean CSV without its raw runs')
    scores = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
    condition_configs = {}
    for condition, ratio in conditions:
        cfg = dict(best)
        if args.method == 'fkm':
            if args.study == 'participation':
                cfg['alpha'] = max(1, int(args.clients * ratio))
            elif reference_clients != args.clients and cfg['alpha'] is not None:
                cfg['alpha'] = args.clients
        condition_configs[condition] = cfg
    existing = pd.read_csv(raw_path) if args.resume and raw_path.exists() else pd.DataFrame()
    done = set()
    if len(existing):
        keys = list(zip(existing.condition, existing.init_seed))
        allowed = {(condition, seed) for condition, _ in conditions for seed in seeds}
        if len(keys) != len(set(keys)) or not set(keys).issubset(allowed):
            raise ValueError('Resume CSV has duplicate or unexpected condition/seed rows')
        resume_identity = dict(dataset=args.dataset, data_seed=args.grid_split,
            beta=args.beta, clients=args.clients, method_tag=args.method, study=args.study,
            reference_beta=args.reference_beta, reference_clients=reference_clients,
            result_origin=result_origin,
            reference_config=reference_config, reference_source_csv=str(source.resolve()),
            reference_data_sha256=reference['provenance']['data_sha256'],
            reference_partition_sha256=reference['provenance']['partition_sha256'],
            data_sha256=target['provenance']['data_sha256'],
            partition_sha256=target['provenance']['partition_sha256'], select_on=args.select_on)
        for key, value in resume_identity.items():
            if not existing[key].eq(value).all():
                raise ValueError(f'Resume CSV differs from the current experiment: {key}')
        for condition, cfg in condition_configs.items():
            rows_for_condition = existing[existing.condition.eq(condition)]
            for key, value in cfg.items():
                if key == 'kernel_sigma':
                    value = module.sigma_label(value)
                same = rows_for_condition[key].isna() if value is None else rows_for_condition[key].eq(value)
                if not same.all():
                    raise ValueError(f'Resume CSV configuration differs: {condition}/{key}')
        if not existing.status.eq('completed').all() or not np.isfinite(existing[scores].to_numpy(float)).all():
            raise ValueError('Resume CSV contains failed or nonfinite runs')
        done = set(keys)
    print(module.partition_summary(target))
    print(f'{args.method}: selected on reference mean FIT/{args.select_on}: {best}')
    print(scope)
    print(f'source clients={reference_clients}; target clients={args.clients}')
    print(f'{len(conditions)} conditions x {len(seeds)} seeds; {len(done)} complete; no parameter search.')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0

    runner = (module.run_ours if args.method == 'ours' else
              module.run_fkm if args.method == 'fkm' else
              module.run_ffcm if args.method == 'ffcm' else module.run_model)
    rows = existing.to_dict('records')
    out.mkdir(parents=True, exist_ok=True)
    for condition, ratio in conditions:
        cfg = condition_configs[condition]
        active_clients = max(1, int(args.clients * ratio))
        if args.method == 'fedsc':
            module.FIXED_EVAL['active_ratio'] = ratio
        for seed in seeds:
            if (condition, seed) in done:
                continue
            started = time.perf_counter()
            if reuse_reference:
                source_row = selected.loc[selected.init_seed.eq(seed)].iloc[0]
                result = {key: source_row[key] for key in module.RESULT_COLUMNS if key in source_row}
                # Keep native assignments/prediction metadata from the original run.
                for key in ('fit_basis', 'test_basis', 'status'):
                    if key in source_row:
                        result[key] = source_row[key]
            else:
                result = runner(target, cfg, seed)
            result.setdefault('wall_seconds', time.perf_counter() - started)
            row = dict(split=target['label'], dataset=args.dataset, data_seed=args.grid_split,
                       beta=args.beta, clients=args.clients, init_seed=seed, **cfg, **result,
                       method_tag=args.method, study=args.study, condition=condition,
                       active_clients_per_round=active_clients,
                       participation_scope=scope, reference_beta=args.reference_beta,
                       reference_clients=reference_clients,
                       result_origin=result_origin,
                       reference_config=reference_config, reference_source_csv=str(source.resolve()),
                       reference_data_sha256=reference['provenance']['data_sha256'],
                       reference_partition_sha256=reference['provenance']['partition_sha256'],
                       select_on=args.select_on)
            row.update(partition_stats)
            row['active_ratio'] = ratio
            if args.method == 'ours':
                row['kernel_sigma'] = module.sigma_label(cfg['kernel_sigma'])
            row.update({key: target['provenance'][key]
                        for key in ('data_sha256', 'partition_sha256')})
            row.setdefault('fit_basis', 'native_dual_space' if args.method == 'ours' else 'nearest_center')
            row.setdefault('test_basis', 'nearest_center')
            row.setdefault('status', 'completed')
            if not np.isfinite([row[key] for key in scores]).all() or row['status'] != 'completed':
                raise ValueError(f'{condition} init={seed} failed or produced nonfinite FIT/TEST scores')
            rows.append(row)
            if args.resume:
                pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'{condition} init={seed} FIT={row["acc"]:.4f} TEST={row["test_acc"]:.4f}', flush=True)

    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[scores].to_numpy(float)).all():
        raise ValueError('A fixed-configuration run produced nonfinite FIT/TEST scores')
    group_keys = ['condition', *module.GRID_PARAMS]
    numeric = [key for key in frame.select_dtypes(include=['number', 'bool']).columns
               if key not in group_keys and key not in ('init_seed', 'data_seed', 'beta', 'clients')]
    groups = frame.groupby(group_keys, dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    summary = summary.merge(groups[scores].std(ddof=1).reset_index(),
                            on=group_keys, suffixes=('', '_std'))
    text_columns = [key for key in frame.select_dtypes(include=['object', 'string']).columns
                    if key not in group_keys and key not in ('split', 'dataset', 'study', 'method_tag')]
    states = groups[text_columns].agg(
        lambda values: '; '.join(str(value) for value in pd.unique(values.dropna()))).reset_index()
    summary = summary.merge(states, on=group_keys)
    summary['dataset'] = args.dataset
    summary['data_seed'] = args.grid_split
    summary['beta'] = args.beta
    summary['clients'] = args.clients
    summary['reference_clients'] = reference_clients
    summary['method_tag'] = args.method
    summary['study'] = args.study
    summary['n_seeds'] = len(seeds)
    if not args.resume:
        frame.to_csv(raw_path, index=False)
    summary.to_csv(mean_path, index=False)
    print(summary[['condition', 'acc', 'test_acc', 'test_nmi', 'test_ari']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
