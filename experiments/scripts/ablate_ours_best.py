"""Fixed-parameter Ours mechanism, space or anchor ablations on one frozen partition.

Parameters come from the full model's mean-FIT-selected configuration.
Anchor comparisons retrain both conditions with a shared dimension and TEST
prediction basis. No hyperparameter search is performed.
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
    parser.add_argument('--study', choices=('mechanism', 'space', 'anchors'), default='mechanism')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--include-normalization', action='store_true',
                        help='Add no_normalization to the mechanism conditions')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')

    source = Path(args.source_dir) / f'grid_ours_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source)
    mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    # Validate the saved source at this entry boundary, then reuse its typed config.
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    for key, value in {**identity, **{key: bundle['provenance'][key]
                                    for key in ('data_sha256', 'partition_sha256')}}.items():
        if not raw[key].eq(value).all():
            raise ValueError(f'Source CSV differs from the frozen partition: {key}')

    selected, best = select_complete(raw, mean, ours.GRID_PARAMS, seeds, args.select_on)
    if selected.init_uses_pooled_fit.any():
        raise ValueError('The full-model reference must use random/uniform initialization')
    for key, value in best.items():
        if key == 'kernel_sigma':
            best[key] = None if ours.sigma_label(value) == 'adaptive' else float(value)
        else:
            best[key] = type(ours.GRID[key][0])(value)

    selected_description = ours.describe(best)
    selected_d_emb = best['d_emb']
    eval_overrides = {}
    if args.study == 'mechanism':
        variants = {
            'full': {},
            'no_topology': {'enforce_topology': False},
            'no_procrustes': {'use_procrustes': False},
            'original_membership': {'lambda_space': 1.0},
        }
        if args.include_normalization:
            variants['no_normalization'] = {'normalize_spaces': False}
    elif args.study == 'space':
        # lambda=1 is already in mechanism; do not train that condition twice.
        variants = {'full': {}, 'spectral_membership': {'lambda_space': 0.0}}
    else:
        best['d_emb'] = min(selected_d_emb, min(map(len, bundle['X_list'])))
        eval_overrides = {'test_embedding_mode': 'original_space'}
        variants = {'full_matched': {}, 'no_anchors': {'use_anchors': False}}
    out = Path(args.out_dir)
    tag = 'ablation' if args.study == 'mechanism' else args.study
    stem = f'grid_ours_{tag}_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Use --resume or another --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    scores = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']
    proof = {key: bundle['provenance'][key]
             for key in ('data_sha256', 'partition_sha256')}
    configs = {name: json.dumps({**best, **changes}, sort_keys=True)
               for name, changes in variants.items()}
    evaluation = json.dumps({**ours.FIXED_EVAL, 'eigen_solver': 'dense',
                             **eval_overrides}, sort_keys=True)
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    if len(existing):
        expected = {(name, seed) for name in variants for seed in seeds}
        keys = list(zip(existing.variant, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
            raise ValueError('Existing output contains unexpected or duplicate variant/seed rows')
        for key, value in {**identity, **proof, 'study': args.study,
                           'selected_d_emb': selected_d_emb, 'select_on': args.select_on,
                           'reference_source_csv': str(source.resolve()),
                           'evaluation_config': evaluation}.items():
            if not existing[key].eq(value).all():
                raise ValueError('Existing output differs from this study: ' + key)
        if (not existing.condition_config.eq(existing.variant.map(configs)).all()
                or not existing.status.eq('completed').all()
                or not np.isfinite(existing[scores].to_numpy(float)).all()):
            raise ValueError('Existing output has changed parameters or incomplete quality scores')

    print(ours.partition_summary(bundle))
    print(f'Selected on mean FIT/{args.select_on}: {selected_description}')
    reuse_count = len(seeds) if 'full' in variants else 0
    print(f'Reuse {reuse_count} full-model runs; train {len(variants) * len(seeds) - reuse_count} ablation runs.')
    if args.study == 'anchors':
        print(f'Anchor comparison: selected_d_emb={selected_d_emb}; common_d_emb={best["d_emb"]}; '
              'TEST=original_space for both newly trained conditions.')
        print('no_anchors also removes anchor-based Procrustes alignment; this is a joint removal.')
    for name, changes in variants.items():
        action = 'reuse selected full-model rows' if name == 'full' else (changes or 'train matched full model')
        print(f'  {name}: {action}')
    if args.dry_run:
        print('[dry-run] no training or output files.')
        return 0

    rows = existing.to_dict('records')
    done = set(zip(existing.variant, existing.init_seed)) if len(existing) else set()
    out.mkdir(parents=True, exist_ok=True)
    for name, changes in variants.items():
        cfg = {**best, **changes}
        for seed in seeds:
            if (name, seed) in done:
                continue
            if name == 'full':
                reference = selected.loc[selected.init_seed.eq(seed)].iloc[0]
                row = {key: reference[key] for key in ours.RESULT_COLUMNS if key in reference}
                row.update(split=bundle['label'], **identity, init_seed=seed, **cfg, **proof)
                row['kernel_sigma'] = ours.sigma_label(cfg['kernel_sigma'])
            else:
                metrics = ours.run_ours(bundle, cfg, seed, eval_overrides=eval_overrides)
                row = dict(split=bundle['label'], **identity, init_seed=seed, **cfg)
                row['kernel_sigma'] = ours.sigma_label(cfg['kernel_sigma'])
                row.update(metrics)
                row.update({key: bundle['provenance'][key]
                            for key in ('data_sha256', 'partition_sha256')})
            row.update(study=args.study, selected_d_emb=selected_d_emb,
                       variant=name, result_origin='reused' if name == 'full' else 'trained',
                       reference_source_csv=str(source.resolve()), select_on=args.select_on,
                       condition_config=configs[name], evaluation_config=evaluation, status='completed')
            if not np.isfinite([row[key] for key in scores]).all():
                raise ValueError('Ablation produced nonfinite FIT/TEST scores')
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'  {name:22s} init={seed} FIT={row["acc"]:.4f} '
                  f'TEST={row["test_acc"]:.4f} {row["stop_reason"]} '
                  f'({row["result_origin"]})', flush=True)

    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[scores].to_numpy(float)).all():
        raise ValueError('Ablation produced nonfinite FIT/TEST scores')
    numeric = scores + ['n_distinct_labels', 'center_spread_ratio', 'rounds_run',
                        'final_center_shift_orig', 'wall_seconds']
    groups = frame.groupby(['variant', *ours.GRID_PARAMS], dropna=False, sort=False)
    summary = groups[numeric].mean().reset_index()
    std = groups[scores].std(ddof=1).reset_index()
    summary = summary.merge(std, on=['variant', *ours.GRID_PARAMS], suffixes=('', '_std'))
    summary['n_seeds'] = groups.size().to_numpy()
    summary['complete'] = [len(part) == len(seeds)
                           and sorted(part.init_seed.tolist()) == sorted(seeds)
                           for _, part in groups]
    summary['study'] = args.study
    summary['selected_d_emb'] = selected_d_emb
    summary['config'] = [ours.describe(cfg) for cfg in summary.to_dict('records')]
    summary['test_basis'] = summary.variant.map(
        frame.groupby('variant').test_basis.agg(lambda values: '; '.join(values.unique())))
    summary['result_origin'] = summary.variant.map(
        {name: 'reused' if name == 'full' else 'trained' for name in variants})
    summary['reference_source_csv'] = str(source.resolve())
    summary['select_on'] = args.select_on

    frame.to_csv(raw_path, index=False)
    summary.to_csv(mean_path, index=False)
    print(summary[['variant', 'acc', 'test_acc', 'test_nmi', 'test_ari']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
