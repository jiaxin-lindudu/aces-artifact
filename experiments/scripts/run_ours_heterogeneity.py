"""Fixed-best federated quantity imbalance or FIT-only feature shift.

Derive reproducible in-memory views of one README frozen split. The source
TEST stays unchanged. Each invocation writes only raw and mean result CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.partition_checks import array_hash
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='usps')
    parser.add_argument('--method', choices=('ours', 'ffcm', 'fedsc'), default='ours')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5,
                        help='Identity of the source frozen partition, not the derived view')
    parser.add_argument('--clients', type=int, default=50)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--study', choices=('quantity', 'feature'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    module = importlib.import_module('experiments.scripts.grid_search_' + args.method)
    source = Path(args.source_dir) / f'grid_{args.method}_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    if args.method == 'fedsc':
        module.configure_grid(bundle['K'])
    axes = module.GRID_PARAMS
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    proof = bundle['provenance']
    for key, value in {**identity, **{key: proof[key]
                                    for key in ('data_sha256', 'partition_sha256')}}.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Source CSV differs from the frozen partition: {key}')
    selected, best = select_complete(raw, mean, axes, seeds, args.select_on)
    if selected.init_uses_pooled_fit.any():
        raise ValueError('The reference must not use pooled sample initialization')
    for key, value in best.items():
        if key == 'kernel_sigma':
            best[key] = None if ours.sigma_label(value) == 'adaptive' else float(value)
        elif module.GRID[key][0] is None:
            best[key] = None if pd.isna(value) else value
        else:
            best[key] = type(module.GRID[key][0])(value)
    if args.method == 'ours':
        best['active_ratio'] = 1.0

    strengths = (1., 2., 5., 10.) if args.study == 'quantity' else (0., .25, .5, 1.)
    out = Path(args.out_dir)
    stem = f'grid_{args.method}_{args.study}_sourcebeta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Preserve existing results: use --resume or another --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    scores = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
    condition_config = json.dumps(best, sort_keys=True)
    print(ours.partition_summary(bundle))
    print(f'{args.method} selected on source mean FIT/{args.select_on}: {module.describe(best)}')
    print(f'{len(strengths)} conditions x {len(seeds)} seeds; fixed parameters, plaintext, full participation.')
    print('Quantity changes ownership and approximately preserves common class proportions.'
          if args.study == 'quantity' else
          'Feature shift changes FIT only; unchanged TEST is a common target domain (cross-domain heldout).')

    X_source, y_train = bundle['X_train'], bundle['y_train']
    classes, encoded = np.unique(y_train, return_inverse=True)
    global_class_proportions = np.bincount(encoded, minlength=len(classes)) / len(encoded)
    feature_std = X_source.std(axis=0, ddof=0)
    nonconstant = feature_std > 0
    views = []
    for strength in strengths:
        prepared = time.perf_counter()
        # This RNG controls the data view only. All initialization seeds see the
        # same view, and feature strengths use the same Gaussian directions.
        rng = np.random.RandomState(args.grid_split)
        X_view = X_source
        repaired = 0
        if args.study == 'quantity':
            weights = np.linspace(1., strength, args.clients)
            weights /= weights.sum()
            ownership = [[] for _ in range(args.clients)]
            for label in range(len(classes)):
                indices = np.flatnonzero(encoded == label)
                rng.shuffle(indices)
                quotas = len(indices) * weights
                counts = np.floor(quotas).astype(int)
                # Random, reproducible tie order avoids privileging one client
                # for the remainder of every class, especially at ratio=1.
                tie_order = rng.permutation(args.clients)
                order = tie_order[np.argsort(-(quotas - counts)[tie_order], kind='stable')]
                counts[order[:len(indices) - int(counts.sum())]] += 1
                offset = 0
                for cid, count in enumerate(counts):
                    ownership[cid].extend(indices[offset:offset + count].tolist())
                    offset += count
            ownership = [np.asarray(indices, dtype=int) for indices in ownership]
            for cid, indices in enumerate(ownership):
                if len(indices) == 0:
                    donor = int(np.argmax([len(part) for part in ownership]))
                    ownership[cid] = ownership[donor][-1:].copy()
                    ownership[donor] = ownership[donor][:-1]
                    repaired += 1
            for indices in ownership:
                rng.shuffle(indices)
            family = 'class_proportional_quantity_allocation'
            test_domain = 'unchanged_common_heldout'
        else:
            ownership = [np.asarray(indices, dtype=int).copy()
                         for indices in bundle['client_indices']]
            directions = rng.normal(size=(args.clients, X_source.shape[1]))
            directions /= np.sqrt(np.mean(directions ** 2, axis=1, keepdims=True))
            X_view = X_source.copy()
            for cid, indices in enumerate(ownership):
                X_view[indices] += strength * directions[cid] * feature_std
            family = 'source_ownership_fit_feature_shift'
            test_domain = 'unchanged_common_target_cross_domain_heldout'

        sizes = [len(indices) for indices in ownership]
        local_tv = [.5 * np.abs(np.bincount(encoded[indices], minlength=len(classes))
                                / len(indices) - global_class_proportions).sum()
                    for indices in ownership]
        normalized_shift_rms = (float(np.sqrt(np.mean(
            ((X_view[:, nonconstant] - X_source[:, nonconstant])
             / feature_std[nonconstant]) ** 2))) if nonconstant.any() else 0.)
        arrays_hashes = {key: array_hash(X_view if key == 'X_train' else bundle[key])
                        for key in ('X_train', 'X_test', 'y_train', 'y_test')}
        derived_data_hash = hashlib.sha256(
            json.dumps(arrays_hashes, sort_keys=True).encode()).hexdigest()
        derived_ownership_hash = hashlib.sha256(
            json.dumps([indices.tolist() for indices in ownership]).encode()).hexdigest()
        view = dict(bundle)
        X_list = [X_view[indices] for indices in ownership]
        offsets = np.cumsum([0] + sizes)
        train_indices = [np.arange(offsets[cid], offsets[cid + 1])
                         for cid in range(args.clients)]
        view.update(X_train=X_view, client_indices=ownership, X_list=X_list,
                    X_fit=np.vstack(X_list),
                    y_fit=np.concatenate([y_train[indices] for indices in ownership]),
                    train_indices=train_indices,
                    label=f'{args.dataset}_seed_{args.grid_split}_{args.study}_{strength:g}'
                          f'_clients_{args.clients}_sourcebeta_{args.beta}')
        view['provenance'] = dict(data_sha256=derived_data_hash,
                                  partition_sha256=derived_ownership_hash,
                                  arrays_sha256=arrays_hashes)
        if args.study == 'feature' and strength:
            view['median_distance'] = float(np.median(pdist(X_view), overwrite_input=True))
        facts = dict(method=args.method, study=args.study, condition=f'{args.study}_{strength:g}',
                     strength=strength, dataset=args.dataset, data_seed=args.grid_split,
                     source_beta=args.beta, clients=args.clients,
                     partition_family=family, sampling_seed=args.grid_split,
                     data_view_version='class_largest_remainder_or_standardized_gaussian_v1',
                     requested_quantity_ratio=strength if args.study == 'quantity' else np.nan,
                     requested_feature_shift=strength if args.study == 'feature' else np.nan,
                     actual_quantity_ratio=max(sizes) / min(sizes),
                     client_size_min=min(sizes), client_size_max=max(sizes),
                     client_size_mean=float(np.mean(sizes)),
                     client_label_tv_mean=float(np.mean(local_tv)),
                     normalized_shift_rms=normalized_shift_rms,
                     empty_clients_repaired=repaired, test_domain=test_domain,
                     source_data_sha256=proof['data_sha256'],
                     source_partition_sha256=proof['partition_sha256'],
                     source_fit_sha256=proof['arrays_sha256']['X_train'],
                     data_sha256=derived_data_hash, partition_sha256=derived_ownership_hash,
                     feature_sha256=arrays_hashes['X_train'],
                     test_sha256=arrays_hashes['X_test'],
                     test_unchanged=all(arrays_hashes[key] == proof['arrays_sha256'][key]
                                        for key in ('X_test', 'y_test')),
                     reference_source_csv=str(source.resolve()), select_on=args.select_on,
                     selection_scope='source mean FIT; no tuning on derived views',
                     view_preparation_seconds=time.perf_counter() - prepared)
        print(f'  {facts["condition"]}: sizes=[{min(sizes)}, {max(sizes)}], '
              f'ratio={facts["actual_quantity_ratio"]:.3f}, '
              f'mean label TV={facts["client_label_tv_mean"]:.4f}, '
              f'normalized shift RMS={normalized_shift_rms:.4f}', flush=True)
        views.append((view, facts))

    reference = dict(dataset=args.dataset, data_seed=args.grid_split, source_beta=args.beta,
                     clients=args.clients, method=args.method, study=args.study,
                     source_data_sha256=proof['data_sha256'],
                     source_partition_sha256=proof['partition_sha256'],
                     source_fit_sha256=proof['arrays_sha256']['X_train'],
                     reference_source_csv=str(source.resolve()), select_on=args.select_on,
                     condition_config=condition_config)
    expected = {(facts['condition'], seed) for _, facts in views for seed in seeds}
    view_keys = ('strength', 'data_sha256', 'partition_sha256', 'feature_sha256',
                 'test_sha256', 'test_unchanged', 'partition_family', 'data_view_version')
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
                raise ValueError('Existing derived data view differs: ' + key)
        if (not existing.status.eq('completed').all()
                or not np.isfinite(existing[scores].to_numpy(float)).all()):
            raise ValueError('Existing output contains failed or nonfinite measurements')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0
    rows = existing.to_dict('records')
    done = set(zip(existing.condition, existing.init_seed)) if len(existing) else set()
    out.mkdir(parents=True, exist_ok=True)
    for view, facts in views:
        for seed in seeds:
            if (facts['condition'], seed) in done:
                continue
            started = time.perf_counter()
            if args.method == 'ours':
                metrics, history, _tracker = ours.run_ours(
                    view, best, seed,
                    eval_overrides=dict(use_ckks=False, dp_sigma=0., run_privacy_eval=False),
                    return_details=True)
            elif args.method == 'ffcm':
                metrics = module.run_ffcm(view, best, seed)
            else:
                metrics = module.run_model(view, best, seed)
                metrics['wall_seconds'] = time.perf_counter() - started
            row = dict(split=view['label'], **facts, init_seed=seed, **best, **metrics)
            row.update(use_ckks=False, dp_sigma=0., run_privacy_eval=False,
                       formal_dp=False, active_ratio=1.)
            row.setdefault('status', 'completed')
            if args.method == 'ours':
                row['kernel_sigma'] = ours.sigma_label(best['kernel_sigma'])
                communication, cost = history['communication'], history['execution_cost']
                for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes',
                            'interservice_bytes', 'total_bytes', 'packed_values'):
                    row[key] = communication.get(key, np.nan)
                row['bytes_per_round'] = ((communication['total_bytes'] - communication['setup_bytes'])
                                          / history['completed_rounds'])
                row['communication_basis'] = communication['measurement_basis']
                row['communication_excluded'] = communication['excluded']
                for key in ('trainer_seconds', 'setup_seconds', 'evaluation_seconds', 'privacy_seconds'):
                    row[key] = cost.get(key, np.nan)
                row.update({key + '_seconds': value
                            for key, value in cost['round_substage_seconds'].items()})
            elif args.method == 'ffcm':
                for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes',
                            'interservice_bytes', 'total_bytes', 'bytes_per_round',
                            'training_seconds', 'local_seconds', 'aggregation_seconds'):
                    row[key] = np.nan
                row.update(communication_basis='not recorded by current FFCM adapter',
                           fit_basis='nearest_center', test_basis='nearest_center',
                           stop_reason='fixed_round_budget')
            else:
                row['stop_reason'] = 'fixed_round_budget'
            row['condition_config'] = condition_config
            if (row['status'] != 'completed'
                    or not np.isfinite([row[key] for key in scores]).all()):
                raise ValueError('Study produced failed or nonfinite FIT/TEST measurements')
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'    init={seed} FIT={row["acc"]:.4f} TEST={row["test_acc"]:.4f} '
                  f'{row["stop_reason"]}', flush=True)
    frame = pd.DataFrame(rows)
    keys = ['condition', *axes]
    numeric = [key for key in frame.select_dtypes(include=['number', 'bool']).columns
               if key not in keys and key not in ('init_seed', 'data_seed', 'source_beta', 'clients')]
    groups = frame.groupby(keys, dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    std = groups[scores].std(ddof=1).reset_index()
    summary = summary.merge(std, on=keys, suffixes=('', '_std'))
    text = [key for key in frame.select_dtypes(include=['object', 'string']).columns
            if key not in keys and key not in ('split', 'dataset', 'study')]
    states = groups[text].agg(
        lambda values: '; '.join(str(value) for value in pd.unique(values.dropna()))).reset_index()
    summary = summary.merge(states, on=keys)
    summary['n_seeds'] = [len(part) for _, part in groups]
    summary['complete'] = [len(part) == len(seeds)
                           and sorted(part.init_seed.tolist()) == sorted(seeds)
                           and part.status.eq('completed').all()
                           and np.isfinite(part[scores].to_numpy(float)).all()
                           for _, part in groups]
    summary['dataset'] = args.dataset
    summary['data_seed'] = args.grid_split
    summary['source_beta'] = args.beta
    summary['clients'] = args.clients
    summary['study'] = args.study
    summary['method'] = args.method
    summary['config'] = [module.describe(cfg) for cfg in summary.to_dict('records')]
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    summary.to_csv(mean_path, index=False)
    print(summary[['condition', 'acc', 'test_acc', 'actual_quantity_ratio',
                   'client_label_tv_mean', 'normalized_shift_rms']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
