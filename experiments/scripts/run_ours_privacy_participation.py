"""Fixed-best Ours privacy or client-participation conditions on one frozen split.

Select one complete configuration by mean FIT, then run two privacy conditions
or four participation conditions. Each invocation writes only raw and mean CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='usps')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=50)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--study', choices=('privacy', 'participation'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse-full', action='store_true',
                        help='Participation only: reuse active=1 quality; reference resource metrics are unavailable')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')
    if args.reuse_full and args.study != 'participation':
        parser.error('--reuse-full is only available for participation')
    source = Path(args.source_dir) / f'grid_ours_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    proof = {key: bundle['provenance'][key]
             for key in ('data_sha256', 'partition_sha256')}
    # Validate the external source once, before using its selected configuration.
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

    if args.study == 'privacy':
        conditions = [('plain', False, 1.0), ('ckks', True, 1.0)]
    else:
        conditions = [(f'active_{ratio:g}', False, ratio) for ratio in (1., .75, .5, .25)]
    out = Path(args.out_dir)
    stem = f'grid_ours_{args.study}_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Preserve existing results: use --resume or another --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    condition_configs = {name: json.dumps({**best, 'active_ratio': ratio}, sort_keys=True)
                         for name, _, ratio in conditions}
    expected = {(name, seed) for name, _, _ in conditions for seed in seeds}
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    scores = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
    reference = dict(reference_source_csv=str(source.resolve()), select_on=args.select_on,
                     study=args.study, reuse_full=args.reuse_full)
    if len(existing):
        required = set(identity) | set(proof) | set(reference) | set(scores) | {
            'condition', 'init_seed', 'condition_config', 'status'}
        if not required <= set(existing):
            raise ValueError('Existing output lacks resume metadata; use a new out-dir')
        keys = list(zip(existing.condition, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
            raise ValueError('Existing output has duplicate or unexpected condition/seed rows')
        for key, value in {**identity, **proof, **reference}.items():
            if not existing[key].eq(value).all():
                raise ValueError('Existing output differs from this study: ' + key)
        if (not existing.condition_config.eq(existing.condition.map(condition_configs)).all()
                or not existing.status.eq('completed').all()
                or not np.isfinite(existing[scores].to_numpy(float)).all()):
            raise ValueError('Existing condition parameters or quality scores are invalid')
    rows = existing.to_dict('records')
    done = set(zip(existing.condition, existing.init_seed)) if len(existing) else set()

    print(ours.partition_summary(bundle))
    print(f'Selected on mean FIT/{args.select_on}: {ours.describe(best)}')
    print(f'{len(conditions)} conditions x {len(seeds)} seeds; {len(done)} saved; no hyperparameter search.')
    if args.reuse_full:
        print('active=1 reuses quality only; reference communication, resources and traces were not recorded.')
    for name, ckks, ratio in conditions:
        print(f'  {name}: use_ckks={ckks}, active_ratio={ratio}, '
              f'active clients/round={max(1, int(args.clients * ratio))}')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for name, ckks, ratio in conditions:
        cfg = {**best, 'active_ratio': ratio}
        evaluation = dict(use_ckks=ckks, run_privacy_eval=args.study == 'privacy',
                          privacy_attack_folds=5, privacy_max_samples=1000,
                          privacy_ciphertext_trials=200)
        for seed in seeds:
            if (name, seed) in done:
                continue
            if args.reuse_full and ratio == 1.:
                original = selected.loc[selected.init_seed.eq(seed)].iloc[0]
                metrics = {key: original[key] for key in ours.RESULT_COLUMNS if key in original}
                reference_wall = metrics.pop('wall_seconds', np.nan)
                row = dict(split=bundle['label'], **identity, init_seed=seed, **cfg,
                           **metrics, **proof, **reference, condition=name,
                           condition_config=condition_configs[name], status='completed',
                           result_origin='reused_quality_only',
                           resource_metrics_status='not_recorded_in_reference',
                           wall_seconds=np.nan, reference_wall_seconds=reference_wall,
                           wall_seconds_basis='reference_quality_run',
                           use_ckks=False, run_privacy_eval=False,
                           active_clients_per_round=args.clients)
                row['kernel_sigma'] = ours.sigma_label(cfg['kernel_sigma'])
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                print(f'  {name:12s} init={seed} FIT={row["acc"]:.4f} '
                      f'TEST={row["test_acc"]:.4f} reused quality only', flush=True)
                continue
            metrics, history, tracker = ours.run_ours(
                bundle, cfg, seed, eval_overrides=evaluation, return_details=True)
            communication = history['communication']
            cost = history['execution_cost']
            privacy = history['privacy_results']
            mia = privacy['mia']
            ciphertext = privacy['ciphertext']
            role = privacy.get('role_check', {})
            leakage = privacy.get('size_leakage', {})
            row = dict(split=bundle['label'], **identity, init_seed=seed, **cfg,
                       condition=name, study=args.study, use_ckks=ckks,
                       run_privacy_eval=evaluation['run_privacy_eval'], **metrics, **proof)
            row['kernel_sigma'] = ours.sigma_label(cfg['kernel_sigma'])
            row['active_clients_per_round'] = max(1, int(args.clients * ratio))
            for key in ('uplink_bytes', 'downlink_bytes', 'setup_bytes',
                        'interservice_bytes', 'total_bytes', 'packed_values',
                        'public_context_bytes', 'context_recipients',
                        'slots_per_ciphertext', 'ciphertexts_per_client_per_round'):
                row[key] = communication.get(key, np.nan)
            # Downlink already includes setup; exclude it only for iteration means.
            row['bytes_per_round'] = ((communication['total_bytes'] - communication['setup_bytes'])
                                      / history['completed_rounds'])
            row['communication_basis'] = communication['measurement_basis']
            row['communication_excluded'] = communication['excluded']
            row['bytes_reproducible'] = communication['bytes_reproducible']
            row['bytes_reproducibility_note'] = communication['bytes_reproducibility_note']
            row['packing_efficiency'] = (
                communication['packed_values'] / (communication['slots_per_ciphertext']
                                                  * communication['ciphertexts_per_client_per_round'])
                if ckks else np.nan)
            for key in ('trainer_seconds', 'setup_seconds', 'evaluation_seconds', 'privacy_seconds'):
                row[key] = cost.get(key, np.nan)
            row.update({key + '_seconds': value
                        for key, value in cost['round_substage_seconds'].items()})
            row['privacy_model'] = privacy['model']
            row['formal_dp'] = privacy['formal_dp']
            row['end_to_end_privacy'] = privacy['end_to_end_privacy']
            row['mia_status'] = mia['status']
            row['mia_prediction_status'] = mia.get('prediction_status', np.nan)
            row['mia_valid'] = (mia['status'] == 'evaluated'
                                and mia.get('prediction_status') == 'evaluated'
                                and mia.get('degenerate_folds') == 0)
            for key in ('acc', 'balanced_accuracy', 'auc', 'tpr', 'tnr', 'cohens_d',
                        'degenerate_folds', 'folds', 'per_class', 'attack_test_samples'):
                row['mia_' + key] = mia.get(key, np.nan)
            row['mia_effect_status'] = mia.get('effect_status', np.nan)
            row['mia_attacker_view'] = mia.get('attacker_view', np.nan)
            row['ciphertext_status'] = ciphertext['status']
            row['ciphertext_prediction_status'] = ciphertext.get('prediction_status', np.nan)
            for key in ('acc', 'balanced_accuracy', 'auc', 'advantage', 'degenerate_folds',
                        'folds', 'trials', 'message_values'):
                row['ciphertext_' + key] = ciphertext.get(key, np.nan)
            row['ciphertext_attacker_view'] = ciphertext.get('attacker_view', np.nan)
            row['ciphertext_limitation'] = ciphertext.get('limitation', np.nan)
            for key in ('public_context_has_secret_key', 'key_authority_has_secret_key'):
                row[key] = float(role[key]) if role.get(key) is not None else np.nan
            for key in ('clients_observed', 'upload_size_correlation', 'local_time_size_correlation',
                        'upload_status', 'local_time_status'):
                row[key] = leakage.get(key, np.nan)
            row.update(**reference, condition_config=condition_configs[name], status='completed',
                       result_origin='trained', resource_metrics_status='recorded',
                       wall_seconds_basis='current_condition_training')
            # Keep convergence data inside the experiment CSV, not separate traces.
            for key in ('center_shift_orig', 'alignment_error', 'train_accuracy', 'test_accuracy'):
                row[key + '_trace'] = json.dumps([float(value) for value in tracker.history[key]])
            row['round_times_trace'] = json.dumps(history['round_times'])
            row['round_resources_trace'] = json.dumps(history['resource_breakdown'])
            if not np.isfinite([row[key] for key in scores]).all():
                raise ValueError('Study produced nonfinite FIT/TEST scores')
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'  {name:12s} init={seed} FIT={row["acc"]:.4f} '
                  f'TEST={row["test_acc"]:.4f} bytes={row["total_bytes"]} '
                  f'MIA={row["mia_status"]}/{row["mia_prediction_status"]}', flush=True)

    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[scores].to_numpy(float)).all():
        raise ValueError('Study produced nonfinite FIT/TEST scores')
    group_keys = ['condition', *ours.GRID_PARAMS, 'use_ckks', 'run_privacy_eval']
    numeric = [key for key in frame.select_dtypes(include=['number', 'bool']).columns
               if key not in group_keys and key not in ('init_seed', 'data_seed', 'beta', 'clients')]
    groups = frame.groupby(group_keys, dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    std = groups[scores].std(ddof=1).reset_index()
    summary = summary.merge(std, on=group_keys, suffixes=('', '_std'))
    # Preserve readable state labels instead of replacing missing attacks with zeros.
    text_columns = [key for key in frame.select_dtypes(include=['object', 'string']).columns
                    if key not in group_keys and key not in ('split', 'dataset', 'study')
                    and not key.endswith('_trace')]
    states = groups[text_columns].agg(
        lambda values: '; '.join(str(value) for value in pd.unique(values.dropna()))).reset_index()
    summary = summary.merge(states, on=group_keys)
    summary['n_seeds'] = [len(part) for _, part in groups]
    summary['complete'] = [len(part) == len(seeds)
                           and sorted(part.init_seed.tolist()) == sorted(seeds)
                           and part.status.eq('completed').all()
                           and np.isfinite(part[scores].to_numpy(float)).all()
                           for _, part in groups]
    summary['dataset'] = args.dataset
    summary['data_seed'] = args.grid_split
    summary['beta'] = args.beta
    summary['clients'] = args.clients
    summary['study'] = args.study
    summary['config'] = [ours.describe(cfg) for cfg in summary.to_dict('records')]

    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(mean_path, index=False)
    print(summary[['condition', 'acc', 'test_acc', 'total_bytes', 'mia_auc']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
