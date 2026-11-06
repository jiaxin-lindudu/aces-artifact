"""Four fixed-best CKKS scale conditions using derived views of one frozen split.

Sample-count views, equal-size client views and model-cluster counts are separate
studies. Derived FIT views are not replacements for the main-table partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.partition_checks import array_hash
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='usps')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=50,
                        help='client count of the frozen source partition')
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--study', choices=('samples', 'clients', 'clusters'), required=True)
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    source = Path(args.source_dir) / f'grid_ours_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    original = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                         args.clients, args.data_dir)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    proof = {key: original['provenance'][key]
             for key in ('data_sha256', 'partition_sha256')}
    # Validate the external reference once; internal views retain these origins.
    for key, value in {**identity, **proof}.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Source CSV differs from the frozen partition: {key}')
    selected, best = select_complete(raw, mean, ours.GRID_PARAMS, seeds, args.select_on)
    if selected.init_uses_pooled_fit.any():
        raise ValueError('The reference must use random/uniform initialization')
    for key, value in best.items():
        if key == 'kernel_sigma':
            best[key] = None if ours.sigma_label(value) == 'adaptive' else float(value)
        else:
            best[key] = type(ours.GRID[key][0])(value)

    rng = np.random.RandomState(args.sampling_seed)
    source_ownership = original['client_indices']
    if args.study == 'samples':
        local_orders = [rng.permutation(len(indices)) for indices in source_ownership]
        values = [.25, .5, .75, 1.]
        view_note = 'nested subsets within source clients; unchanged TEST'
    elif args.study == 'clients':
        n_local = len(original['X_train']) // 100
        if n_local < 1:
            parser.error('The clients study requires at least 100 source FIT rows')
        order = rng.permutation(len(original['X_train']))
        values = [10, 20, 50, 100]
        view_note = ('equal-size random client views at fixed n_i; total FIT grows with M; '
                     'beta identifies the source, not a new Dirichlet partition')
    else:
        values = [max(2, original['K'] // 2), original['K'],
                  2 * original['K'], 4 * original['K']]
        view_note = 'unchanged source clients and FIT; only model K changes; true labels unchanged'

    out = Path(args.out_dir)
    stem = f'grid_ours_scale_{args.study}_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Preserve existing results: use --resume or another --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    scores = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
    print(ours.partition_summary(original))
    print(f'Selected on mean FIT/{args.select_on}: {ours.describe(best)}')
    print(f'{args.study}={values}; CKKS enabled; fixed sampling seed={args.sampling_seed}')
    print(view_note)
    print(f'{len(values)} conditions x {len(seeds)} initialization seeds; no parameter search.')

    views = []
    for value in values:
        model_k = value if args.study == 'clusters' else original['K']
        if args.study == 'samples':
            ownership = [indices[np.sort(local_order[:max(1, int(len(indices) * value))])]
                         for indices, local_order in zip(source_ownership, local_orders)]
        elif args.study == 'clients':
            ownership = [order[i * n_local:(i + 1) * n_local] for i in range(value)]
        else:
            ownership = source_ownership
        X_list = [original['X_train'][indices] for indices in ownership]
        X_fit = np.vstack(X_list)
        y_fit = np.concatenate([original['y_train'][indices] for indices in ownership])
        sizes = [len(X) for X in X_list]
        offsets = np.cumsum([0, *sizes])
        train_indices = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(sizes))]
        condition = f'{args.study}_{value:g}'
        bundle = {**original, 'label': original['label'] + '_scale_' + condition,
                  'clients': len(X_list), 'K': int(model_k), 'X_train': X_fit,
                  'y_train': y_fit, 'X_list': X_list, 'X_fit': X_fit, 'y_fit': y_fit,
                  'client_indices': train_indices, 'train_indices': train_indices}
        actual_fit_hash = array_hash(X_fit)
        actual_ownership_hash = hashlib.sha256(json.dumps(
            [np.asarray(indices, dtype=int).tolist() for indices in ownership]).encode()).hexdigest()
        print(f'  {condition}: FIT={len(X_fit)} M={len(X_list)} K={model_k} '
              f'n_i=[{min(sizes)}, {max(sizes)}]')
        facts = dict(dataset=args.dataset, data_seed=args.grid_split,
                       beta=args.beta, clients=len(X_list),
                       study=args.study, study_value=value, condition=condition,
                       model_clusters=int(model_k), source_true_clusters=original['K'],
                       source_clients=args.clients, source_fit_rows=len(original['X_train']),
                       fit_rows=len(X_fit), local_samples_min=min(sizes),
                       local_samples_max=max(sizes), local_samples_mean=float(np.mean(sizes)),
                       sampling_seed=args.sampling_seed, use_ckks=True, run_privacy_eval=False,
                       source_data_sha256=proof['data_sha256'],
                       source_partition_sha256=proof['partition_sha256'],
                       actual_fit_sha256=actual_fit_hash,
                       actual_ownership_sha256=actual_ownership_hash,
                       actual_ownership_basis='per-client row indices in the source canonical FIT',
                       source_spread_normalizer='source FIT median pairwise distance',
                       view_note=view_note, reference_source_csv=str(source.resolve()),
                       select_on=args.select_on)
        facts['condition_config'] = json.dumps(dict(config=best, model_clusters=int(model_k),
            clients=len(X_list), study=args.study, study_value=value,
            sampling_seed=args.sampling_seed), sort_keys=True)
        views.append((bundle, facts))

    reference = dict(dataset=args.dataset, data_seed=args.grid_split, beta=args.beta,
                     source_clients=args.clients, study=args.study, sampling_seed=args.sampling_seed,
                     source_data_sha256=proof['data_sha256'],
                     source_partition_sha256=proof['partition_sha256'],
                     reference_source_csv=str(source.resolve()), select_on=args.select_on)
    expected = {(facts['condition'], seed) for _, facts in views for seed in seeds}
    view_keys = ('study_value', 'clients', 'model_clusters', 'fit_rows',
                 'actual_fit_sha256', 'actual_ownership_sha256', 'condition_config')
    if len(existing):
        required = set(reference) | set(view_keys) | set(scores) | {'condition', 'init_seed', 'status'}
        if not required <= set(existing):
            raise ValueError('Existing output lacks resume metadata; use a new out-dir')
        keys = list(zip(existing.condition, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
            raise ValueError('Existing output has duplicate or unexpected condition/seed rows')
        for key, value in reference.items():
            if not existing[key].eq(value).all():
                raise ValueError('Existing output differs from the selected reference: ' + key)
        for key in view_keys:
            expected_values = {facts['condition']: facts[key] for _, facts in views}
            if not existing[key].eq(existing.condition.map(expected_values)).all():
                raise ValueError('Existing derived scale view differs: ' + key)
        if (not existing.status.eq('completed').all()
                or not np.isfinite(existing[scores].to_numpy(float)).all()):
            raise ValueError('Existing output contains failed or nonfinite measurements')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0
    rows = existing.to_dict('records')
    done = set(zip(existing.condition, existing.init_seed)) if len(existing) else set()
    out.mkdir(parents=True, exist_ok=True)
    for bundle, facts in views:
        for seed in seeds:
            if (facts['condition'], seed) in done:
                continue
            metrics, history, _tracker = ours.run_ours(
                bundle, best, seed, eval_overrides=dict(use_ckks=True, run_privacy_eval=False),
                return_details=True)
            communication, cost = history['communication'], history['execution_cost']
            row = dict(split=bundle['label'], **facts, init_seed=seed, **best, **metrics,
                       status='completed')
            row['kernel_sigma'] = ours.sigma_label(best['kernel_sigma'])
            row['active_clients_per_round'] = max(1, int(bundle['clients'] * best['active_ratio']))
            for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes', 'interservice_bytes',
                        'total_bytes', 'packed_values', 'public_context_bytes', 'context_recipients',
                        'slots_per_ciphertext', 'ciphertexts_per_client_per_round'):
                row[key] = communication.get(key, np.nan)
            row['bytes_per_round'] = ((communication['total_bytes'] - communication['setup_bytes'])
                                      / history['completed_rounds'])
            row['packing_efficiency'] = (communication['packed_values']
                                         / (communication['slots_per_ciphertext']
                                            * communication['ciphertexts_per_client_per_round']))
            row['communication_basis'] = communication['measurement_basis']
            row['communication_excluded'] = communication['excluded']
            row['bytes_reproducible'] = communication['bytes_reproducible']
            row['bytes_reproducibility_note'] = communication['bytes_reproducibility_note']
            for key in ('trainer_seconds', 'setup_seconds', 'evaluation_seconds', 'privacy_seconds'):
                row[key] = cost.get(key, np.nan)
            row.update({key + '_seconds': seconds
                        for key, seconds in cost['round_substage_seconds'].items()})
            if not np.isfinite([row[key] for key in scores]).all():
                raise ValueError('Scale study produced nonfinite FIT/TEST scores')
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'    init={seed} FIT={row["acc"]:.4f} TEST={row["test_acc"]:.4f} '
                  f'bytes/round={row["bytes_per_round"]:.0f}', flush=True)

    frame = pd.DataFrame(rows)
    group_keys = ['condition', 'study_value', 'clients', 'model_clusters',
                  *ours.GRID_PARAMS, 'use_ckks']
    numeric = [key for key in frame.select_dtypes(include=['number', 'bool']).columns
               if key not in group_keys and key not in ('init_seed', 'data_seed', 'beta')]
    groups = frame.groupby(group_keys, dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    std = groups[scores].std(ddof=1).reset_index()
    summary = summary.merge(std, on=group_keys, suffixes=('', '_std'))
    text_columns = [key for key in frame.select_dtypes(include=['object', 'string']).columns
                    if key not in group_keys and key not in ('split', 'dataset', 'study')]
    states = groups[text_columns].agg(
        lambda entries: '; '.join(str(entry) for entry in pd.unique(entries.dropna()))).reset_index()
    summary = summary.merge(states, on=group_keys)
    summary['dataset'], summary['data_seed'] = args.dataset, args.grid_split
    summary['beta'], summary['study'] = args.beta, args.study
    summary['sampling_seed'] = args.sampling_seed
    summary['n_seeds'] = [len(part) for _, part in groups]
    summary['complete'] = [len(part) == len(seeds)
                           and sorted(part.init_seed.tolist()) == sorted(seeds)
                           and part.status.eq('completed').all()
                           and np.isfinite(part[scores].to_numpy(float)).all()
                           for _, part in groups]
    summary['config'] = [ours.describe(cfg) for cfg in summary.to_dict('records')]
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    summary.to_csv(mean_path, index=False)
    print(summary[['condition', 'fit_rows', 'clients', 'model_clusters',
                   'bytes_per_round', 'test_acc']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
