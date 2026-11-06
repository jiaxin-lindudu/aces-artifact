"""Fixed-best normalization or round-budget checks on one frozen partition.

Only the named factor changes. Historical reference runs are reused, without
inventing their missing trajectories. Each completed run checkpoints the raw
CSV; the companion mean CSV is written after all conditions finish.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete

SCORES = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
TRACE_KEYS = ('aggregation_shifts_orig', 'aggregation_shifts_emb',
              'topology_shifts_orig', 'topology_shifts_emb', 'center_shift_orig',
              'alignment_error', 'train_accuracy', 'test_accuracy',
              'round_times', 'round_resources')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='usps')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=50)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--study', choices=('normalization', 'budget'), required=True)
    round_options = parser.add_mutually_exclusive_group()
    round_options.add_argument('--rounds', nargs='+', type=int,
                               help='budget conditions; default: 10 20 40')
    round_options.add_argument('--round-multipliers', nargs='+', type=float,
                               help='multiply the selected reference round budget')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    if args.study == 'normalization' and (args.rounds or args.round_multipliers):
        parser.error('round options apply only to --study budget')

    source = Path(args.source_dir) / f'grid_ours_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    proof = {key: bundle['provenance'][key]
             for key in ('data_sha256', 'partition_sha256')}
    # The saved reference is external input: validate its identity once here.
    for key, value in {**identity, **proof}.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Source CSV differs from the frozen partition: {key}')
    selected, best = select_complete(raw, mean, ours.GRID_PARAMS, seeds, args.select_on)
    if selected.init_uses_pooled_fit.any():
        raise ValueError('The reference must not use pooled FIT initialization')
    for key, value in best.items():
        best[key] = ((None if ours.sigma_label(value) == 'adaptive' else float(value))
                     if key == 'kernel_sigma' else type(ours.GRID[key][0])(value))
    if best['init_method'] not in ('random', 'uniform'):
        raise ValueError('The reference must use random/uniform initialization')
    reference_config = json.dumps(best, sort_keys=True)
    if args.study == 'normalization':
        if not best['normalize_spaces']:
            raise ValueError('Normalization ablation requires a normalized full-model reference')
        conditions = [('full', dict(best)),
                      ('no_normalization', {**best, 'normalize_spaces': False})]
        tag = 'normalization'
    else:
        values = (np.asarray(args.round_multipliers) * best['num_rounds']
                  if args.round_multipliers is not None else
                  np.asarray(args.rounds if args.rounds is not None else [10, 20, 40]))
        if (not 1 <= len(values) <= 4 or not np.isfinite(values).all()
                or (values <= 0).any() or not np.equal(values, np.floor(values)).all()
                or len(set(values.tolist())) != len(values)):
            parser.error('provide one to four distinct positive integer round budgets')
        conditions = [(f'rounds_{int(value)}', {**best, 'num_rounds': int(value)})
                      for value in values]
        tag = 'budget_check'
    cfg_by_condition = {name: json.dumps(cfg, sort_keys=True) for name, cfg in conditions}
    out = Path(args.out_dir)
    stem = f'grid_ours_{tag}_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Preserve existing results: use --resume or another --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    expected = {(name, seed) for name, _ in conditions for seed in seeds}
    if len(existing):
        keys = list(zip(existing.condition, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
            raise ValueError('Existing CSV contains duplicate/unexpected condition-seed pairs')
        for key, value in {**identity, **proof, 'study': args.study,
                           'reference_config': reference_config,
                           'select_on': args.select_on}.items():
            if not existing[key].eq(value).all():
                raise ValueError(f'Existing CSV differs from this study: {key}')
        if not existing.condition_config.eq(existing.condition.map(cfg_by_condition)).all():
            raise ValueError('Condition parameters changed; choose another --out-dir')
        if not existing.status.eq('completed').all() or not np.isfinite(
                existing[SCORES].to_numpy(float)).all():
            raise ValueError('Existing CSV contains incomplete/nonfinite runs')
    done = set(zip(existing.condition, existing.init_seed)) if len(existing) else set()
    reused = sum(cfg == best for _, cfg in conditions) * len(seeds)
    print(ours.partition_summary(bundle))
    print(f'Reference selected on mean FIT/{args.select_on}: {ours.describe(best)}')
    print(f'{len(conditions)} conditions x {len(seeds)} seeds; {reused} reference records reused; '
          f'{len(done)} already complete. No parameter search or TEST-based budget selection.')
    for name, cfg in conditions:
        print(f'  {name}: {"reuse" if cfg == best else "train"}; rounds={cfg["num_rounds"]}; '
              f'normalize_spaces={cfg["normalize_spaces"]}')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rows = existing.to_dict('records')
    for name, cfg in conditions:
        for seed in seeds:
            if (name, seed) in done:
                continue
            if cfg == best:
                row = selected.loc[selected.init_seed.eq(seed)].iloc[0].to_dict()
                # Historical grid CSVs contain no complete per-round histories.
                row.update({key + '_trace': np.nan for key in TRACE_KEYS})
                row['trajectory_status'] = 'not recorded in historical reference CSV'
            else:
                metrics, history, tracker = ours.run_ours(bundle, cfg, seed,
                    eval_overrides=dict(use_ckks=False, run_privacy_eval=False), return_details=True)
                row = dict(split=bundle['label'], **identity, init_seed=seed,
                           **cfg, **metrics, **proof)
                for key in ('aggregation_shifts_orig', 'aggregation_shifts_emb',
                            'topology_shifts_orig', 'topology_shifts_emb', 'round_times'):
                    row[key + '_trace'] = json.dumps([float(value) for value in history[key]])
                for key in ('center_shift_orig', 'alignment_error', 'train_accuracy', 'test_accuracy'):
                    row[key + '_trace'] = json.dumps([float(value) for value in tracker.history[key]])
                row['round_resources_trace'] = json.dumps(history['resource_breakdown'])
                cost, communication = history['execution_cost'], history['communication']
                row.update(trainer_seconds=cost['trainer_seconds'],
                           evaluation_seconds=cost['evaluation_seconds'],
                           training_core_seconds=cost['trainer_seconds'] - cost['evaluation_seconds'],
                           setup_seconds=cost['setup_seconds'],
                           **{key + '_seconds': value for key, value in cost['round_substage_seconds'].items()})
                for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes',
                            'interservice_bytes', 'total_bytes'):
                    row[key] = communication[key]
                row['communication_basis'] = communication['measurement_basis']
                row['communication_excluded'] = communication['excluded']
                row['trajectory_status'] = 'recorded; evaluation after each completed round'
            row.update(**cfg)
            row['kernel_sigma'] = ours.sigma_label(cfg['kernel_sigma'])
            row.update(study=args.study, condition=name, variant=name, status='completed',
                       round_budget=cfg['num_rounds'], reference_num_rounds=best['num_rounds'],
                       result_origin='reused' if cfg == best else 'trained',
                       use_ckks=False, run_privacy_eval=False,
                       reference_config=reference_config, condition_config=cfg_by_condition[name],
                       reference_source_csv=str(source.resolve()), select_on=args.select_on,
                       config=ours.describe(cfg),
                       stopping_shift_basis='relative pre-topology aggregate shift; orig and emb separately',
                       center_shift_trace_basis='post-topology absolute original-space displacement; not the stopping threshold',
                       budget_selection='predeclared budgets; report all; TEST does not select a budget')
            if not np.isfinite([row[key] for key in SCORES]).all():
                raise ValueError(f'{name} init={seed} produced nonfinite FIT/TEST scores')
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'  {name:20s} init={seed} FIT={row["acc"]:.4f} TEST={row["test_acc"]:.4f} '
                  f'rounds={row["rounds_run"]}/{row["round_budget"]} {row["result_origin"]}', flush=True)

    frame = pd.DataFrame(rows)
    group_keys = ['condition', *ours.GRID_PARAMS]
    numeric = [key for key in frame.select_dtypes(include=['number', 'bool']).columns
               if key not in group_keys and key not in ('init_seed', 'data_seed', 'beta', 'clients')]
    groups = frame.groupby(group_keys, dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    summary = summary.merge(groups[SCORES].std(ddof=1).reset_index(),
                            on=group_keys, suffixes=('', '_std'))
    text_columns = [key for key in frame.select_dtypes(include=['object', 'string']).columns
                    if key not in group_keys and key not in ('split', 'dataset', 'study')
                    and not key.endswith('_trace')]
    states = groups[text_columns].agg(
        lambda values: '; '.join(str(value) for value in pd.unique(values.dropna()))).reset_index()
    summary = summary.merge(states, on=group_keys)
    for key, value in {**identity, **proof, 'study': args.study,
                       'n_seeds': len(seeds),
                       'uncertainty_scope': 'initialization variability within one frozen split'}.items():
        summary[key] = value
    summary.to_csv(mean_path, index=False)
    print(summary[['condition', 'acc', 'test_acc', 'rounds_run']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
