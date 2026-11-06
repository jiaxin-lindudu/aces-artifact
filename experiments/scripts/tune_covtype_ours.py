"""Sequential FIT-only TPE tuning of Ours on frozen Covtype-10k or HAR splits.

Covtype reuses four completed references; HAR starts with two declared random/
uniform configurations counted in its total trial budget. Resume solely from
raw/_mean CSVs; no database, auxiliary state or training on reused rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.partition_checks import PROVENANCE_COLUMNS
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete

SCORES = ('acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari')
SEARCH_SPACE = {
    'fuzziness': dict(low=1.025, high=1.2, step=.025),
    'lambda_space': dict(low=.2, high=.9, step=.05),
    'd_emb': [2, 4, 6],
    'k_neighbors': [5, 10],
    'anchor_weight': dict(low=.5, high=1.5, step=.25),
    'init_method': ['random', 'uniform'],
}


def config_json(config):
    """Canonical parameters also make warm floating-point values cache hits."""
    out = {}
    for key in ours.GRID_PARAMS:
        value = config[key]
        if key == 'kernel_sigma':
            value = None if ours.sigma_label(value) == 'adaptive' else float(value)
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 12)
        out[key] = value
    return json.dumps(out, sort_keys=True)


def summarize(rows, seeds, stop_reason='in_progress'):
    if not rows:
        return pd.DataFrame(columns=['trial_index', 'complete', 'acc'])
    frame = pd.DataFrame(rows)
    output = []
    for index, part in frame.groupby('trial_index', sort=True):
        good = part[part.status.eq('completed')]
        complete = (len(part) == len(seeds)
                    and sorted(part.init_seed.tolist()) == sorted(seeds)
                    and part.status.eq('completed').all()
                    and np.isfinite(part[list(SCORES)].to_numpy(float)).all())
        row = {key: part.iloc[0][key] for key in ours.GRID_PARAMS}
        row.update(trial_index=int(index), condition=part.condition.iloc[0],
                   result_origin=part.result_origin.iloc[0], complete=complete,
                   n_seeds=len(good), n_expected=len(seeds),
                   n_failed=int(part.status.eq('failed').sum()),
                   config=ours.describe(row), condition_config=part.condition_config.iloc[0],
                   objective='mean FIT acc over all five initialization seeds',
                   search_stop_reason=stop_reason)
        for key in ours.RESULT_COLUMNS:
            if key in good and pd.api.types.is_numeric_dtype(good[key]):
                row[key] = good[key].mean()
                if key in SCORES:
                    row[key + '_std'] = good[key].std(ddof=1)
        for key in ('split', 'dataset', 'data_seed', 'beta', 'clients', *PROVENANCE_COLUMNS,
                    'method', 'select_on', 'eigen_solver', 'reference_source_csv',
                    'reference_raw_sha256', 'reference_mean_sha256',
                    'evaluation_config', 'search_spec', 'fit_basis', 'test_basis'):
            if key in part:
                row[key] = part[key].iloc[0]
        output.append(row)
    return pd.DataFrame(output).sort_values(
        ['complete', 'acc'], ascending=[False, False], kind='stable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', choices=('covtype_10000', 'har'), default='covtype_10000')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--select-on', choices=('acc',), default='acc')
    parser.add_argument('--eigen-solver', choices=('dense',), default='dense')
    parser.add_argument('--source-dir', help='Required for Covtype; HAR uses a declared cold start')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--out-dir')
    parser.add_argument('--max-new-trials', type=int, default=8,
                        help='Cumulative budget: Covtype excludes four reused references; HAR includes both starts')
    parser.add_argument('--patience', type=int, default=3,
                        help='Stop after this many complete new trials without meaningful FIT gain')
    parser.add_argument('--min-improvement', type=float, default=.002,
                        help='Absolute FIT ACC gain; .002 means 0.2 percentage points')
    parser.add_argument('--target-fit-acc', type=float)
    parser.add_argument('--search-seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--trainer-logs', action='store_true')
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if (len(seeds) != 5 or len(set(seeds)) != 5 or args.max_new_trials < 0
            or args.patience < 1 or not np.isfinite(args.min_improvement)
            or args.min_improvement < 0 or args.clients < 1
            or not np.isfinite(args.beta) or args.beta <= 0
            or (args.target_fit_acc is not None and not 0 <= args.target_fit_acc <= 1)):
        parser.error('Use five distinct initialization seeds and valid budgets, tolerances and target')
    cold_start = args.dataset == 'har'
    if cold_start and args.max_new_trials < 2:
        parser.error('HAR requires --max-new-trials >= 2, including both declared starts')
    if not cold_start and not args.source_dir:
        parser.error('Covtype requires --source-dir with four completed reference configurations')
    if args.dry_run:
        print('TPE, mean FIT ACC only; ' + ('HAR starts with m=1.05, lambda=.5, '
              'd=4, k=5, aw=1, random then uniform (both count toward the budget).'
              if cold_start else 'four completed reference configurations are reused.'))
        print(json.dumps(SEARCH_SPACE, sort_keys=True))
        print(f'At most {args.max_new_trials} new configurations x 5 seeds; '
              f'patience={args.patience}, minimum FIT improvement={args.min_improvement}.')
        print('Fixed: dense eigensolver, 20 rounds, learning rate .001; no per-seed pruning.')
        print('[dry-run] No data/model load, Optuna import, training or output files.')
        return 0

    import optuna
    if optuna.__version__ != '4.5.0':
        raise RuntimeError('This reproducible TPE entry requires optuna==4.5.0')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ours.SHOW_TRAINER_LOGS = args.trainer_logs
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta,
                                       args.clients, args.data_dir)
    expected = (6, (8239, 64), (2060, 64)) if cold_start else (7, (8000, 54), (2000, 54))
    if (bundle['K'], bundle['X_fit'].shape, bundle['X_test'].shape) != expected:
        raise ValueError('Frozen dataset shape/class contract differs for ' + args.dataset)
    identity = dict(split=bundle['label'], dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients, method='ours', select_on='acc',
                    eigen_solver='dense')
    proof = {key: bundle['provenance'][key] for key in PROVENANCE_COLUMNS}
    evaluation = json.dumps({**ours.FIXED_EVAL, 'eigen_solver': 'dense'}, sort_keys=True)
    warm = []
    starts = []
    if cold_start:
        base = {key: values[0] for key, values in ours.GRID.items()}
        base.update(fuzziness=1.05, lambda_space=.5, d_emb=4, num_rounds=20,
                    kernel_sigma=None, kernel_scale_mode='self_tuning', k_neighbors=5,
                    anchor_weight=1., normalize_spaces=True, learning_rate=.001,
                    active_ratio=1., init_method='random')
        base = json.loads(config_json(base))
        starts = [{**base, 'init_method': init} for init in ('random', 'uniform')]
        reference = dict(reference_source_csv='not_applicable',
                         reference_raw_sha256='not_applicable',
                         reference_mean_sha256='not_applicable')
    else:
        source = Path(args.source_dir) / f'grid_ours_beta{float(args.beta)}_clients{args.clients}.csv'
        source_mean = source.with_name(source.stem + '_mean.csv')
        raw, mean = pd.read_csv(source), pd.read_csv(source_mean)
        for key, value in {**identity, **proof}.items():
            if key in ('split', 'method', 'select_on', 'eigen_solver') and key not in raw:
                continue
            if not raw[key].eq(value).all():
                raise ValueError('Reference identity differs from frozen data or evaluation: ' + key)
        chosen, best = select_complete(raw, mean, ours.GRID_PARAMS, seeds, 'acc')
        base = json.loads(config_json(best))
        if (base['num_rounds'] != 20 or base['learning_rate'] != .001
                or base['kernel_scale_mode'] != 'self_tuning'
                or chosen.init_uses_pooled_fit.any()):
            raise ValueError('Require the 20-round, lr=.001, self-tuning, nonpooled reference')
        if 'evaluation_config' in raw and not raw.evaluation_config.eq(evaluation).all():
            raise ValueError('Reference evaluation differs from the declared plaintext/dense protocol')
        reference = dict(reference_source_csv=str(source.resolve()),
            reference_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            reference_mean_sha256=hashlib.sha256(source_mean.read_bytes()).hexdigest())
    fixed = {key: value for key, value in base.items() if key not in SEARCH_SPACE}
    for _, part in ([] if cold_start else raw.groupby('condition', sort=False)):
        if (len(part) != 5 or sorted(part.init_seed.tolist()) != sorted(seeds)
                or not part.status.eq('completed').all()
                or not np.isfinite(part[list(SCORES)].to_numpy(float)).all()):
            raise ValueError('Each of the four warm configurations needs all five successful seeds')
        cfg = json.loads(config_json(part.iloc[0]))
        if {key: value for key, value in cfg.items() if key not in SEARCH_SPACE} != fixed:
            raise ValueError('Warm configurations change a fixed, nonsearch parameter')
        if part.init_uses_pooled_fit.any():
            raise ValueError('Only native random/uniform initialization is permitted')
        warm.append((cfg, part))
    if not cold_start and (len(warm) != 4 or len({config_json(cfg) for cfg, _ in warm}) != 4):
        raise ValueError('This entry expects the four distinct completed reference configurations')

    reference_count = len(warm)
    startup_count = 2 if cold_start else 4
    search_spec = dict(space=SEARCH_SPACE, fixed=fixed, sampler='TPESampler',
        optuna_version=optuna.__version__, n_startup_trials=startup_count, search_seed=args.search_seed,
        proposal_seed_rule='search_seed + data_seed * 1000 + trial_index',
        patience=args.patience, min_improvement=args.min_improvement,
        target_fit_acc=args.target_fit_acc, pruning=False)
    if cold_start:
        search_spec.update(cold_start=starts, budget_includes_cold_start=True,
                           patience_starts_after_trial_index=1)
    spec = json.dumps(search_spec, sort_keys=True)
    provenance = dict(**identity, **proof, **reference,
        evaluation_config=evaluation, search_spec=spec)
    out = Path(args.out_dir or ROOT / 'experiments/results' / args.dataset
               / f'federated_{args.grid_split}_ours_tpe')
    raw_path = out / f'grid_ours_beta{float(args.beta)}_clients{args.clients}.csv'
    mean_path = raw_path.with_name(raw_path.stem + '_mean.csv')
    if not cold_start and out.resolve() == source.parent.resolve():
        raise ValueError('Use a separate output directory; reference results are read-only')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Use --resume or a new --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')

    rows = []
    if raw_path.exists():
        previous = pd.read_csv(raw_path)
        for key, value in provenance.items():
            if not previous[key].eq(value).all():
                raise ValueError('Resume identity, source or search specification changed: ' + key)
        if previous.duplicated(['trial_index', 'init_seed']).any() or not set(previous.init_seed) <= set(seeds):
            raise ValueError('Unexpected or duplicate trial/seed rows')
        indices = sorted(previous.trial_index.unique().tolist())
        if (indices != list(range(max(indices) + 1))
                or not set(range(reference_count)) <= set(indices)):
            raise ValueError('Incomplete warm-start bookkeeping or missing trial index')
        known = []
        for index, part in previous.groupby('trial_index', sort=True):
            values = part.condition_config.unique()
            if len(values) != 1 or not part.result_origin.eq(
                    'reference_reused' if index < reference_count else 'trained').all():
                raise ValueError('Inconsistent trial parameters or origin')
            cfg = json.loads(values[0])
            if (config_json(cfg) != values[0]
                    or {key: value for key, value in cfg.items() if key not in SEARCH_SPACE} != fixed):
                raise ValueError('Invalid recorded parameter configuration')
            if index < reference_count and values[0] != config_json(warm[int(index)][0]):
                raise ValueError('Warm reference ordering changed')
            if cold_start and index < 2 and values[0] != config_json(starts[int(index)]):
                raise ValueError('Declared HAR start ordering changed')
            known.append(values[0])
        if len(set(known)) != len(known):
            raise ValueError('A parameter configuration was duplicated across trials')
        rows = previous.to_dict('records')
    else:
        for index, (cfg, part) in enumerate(warm):
            for original in part.to_dict('records'):
                rows.append({**original, **provenance, 'trial_index': index,
                    'condition': f'warm_{index}', 'condition_config': config_json(cfg),
                    'result_origin': 'reference_reused', 'objective': 'mean_fit_acc',
                    'requested_max_new_trials': args.max_new_trials, 'attempt_count': 0,
                    'status': 'completed', 'error': '', 'study': 'nonimage_adaptive_tpe'})

    distributions = {
        key: (optuna.distributions.FloatDistribution(**value) if isinstance(value, dict)
              else optuna.distributions.CategoricalDistribution(value))
        for key, value in SEARCH_SPACE.items()}
    # Optuna validates warm values against the predeclared search space once.
    for cfg, part in warm:
        optuna.trial.create_trial(params={key: cfg[key] for key in SEARCH_SPACE},
            distributions=distributions, value=round(float(part.acc.mean()), 12))
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        summarize(rows, seeds).to_csv(mean_path, index=False)
    print(ours.partition_summary(bundle))
    print(('HAR: two declared starts are included. ' if cold_start else
           'Reusing four configurations / 20 measurements. ') +
          f'New trial budget is cumulative: {args.max_new_trials}.')
    print('Selection: mean FIT ACC across all five seeds. TEST is reported only.')
    attempted = set()
    stop_reason = ''
    while True:
        frame = pd.DataFrame(rows)
        summary = summarize(rows, seeds)
        groups = list(frame.groupby('trial_index', sort=True)) if rows else []
        partial = next((int(index) for index, part in groups if index >= reference_count
            and index not in attempted and (len(part) != 5 or not part.status.eq('completed').all()
                or not np.isfinite(part[list(SCORES)].to_numpy(float)).all())), None)
        if partial is not None:
            index = partial
            cfg = json.loads(frame[frame.trial_index.eq(index)].condition_config.iloc[0])
        elif cold_start and len(groups) < 2:
            index = len(groups)
            cfg = starts[index]
            print(f'HAR declared start {index + 1}/2: {ours.describe(cfg)}', flush=True)
        else:
            complete = summary[summary.complete]
            best_fit = float(complete.acc.max()) if len(complete) else -np.inf
            baseline = complete[complete.trial_index.lt(startup_count)]
            meaningful_best = float(baseline.acc.max()) if len(baseline) else -np.inf
            stale = 0
            for _, trial in complete[complete.trial_index.ge(startup_count)].sort_values('trial_index').iterrows():
                value = float(trial.acc)
                if value > meaningful_best and value - meaningful_best >= args.min_improvement - 1e-12:
                    meaningful_best, stale = value, 0
                else:
                    stale += 1
            issued = len(summary[summary.trial_index.ge(reference_count)])
            if cold_start and not len(baseline):
                stop_reason = 'no_completed_startup_configuration'
            elif args.target_fit_acc is not None and best_fit >= args.target_fit_acc:
                stop_reason = 'target_fit_acc_reached'
            elif stale >= args.patience:
                stop_reason = 'patience_exhausted'
            elif issued >= args.max_new_trials:
                stop_reason = 'new_trial_budget_exhausted'
            if stop_reason:
                break
            index = int(frame.trial_index.max()) + 1
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
                seed=args.search_seed + args.grid_split * 1000 + index, n_startup_trials=startup_count))
            for old_index, part in groups:
                old_cfg = json.loads(part.condition_config.iloc[0])
                complete_row = summary[summary.trial_index.eq(old_index)].iloc[0]
                study.add_trial(optuna.trial.create_trial(
                    params={key: old_cfg[key] for key in SEARCH_SPACE},
                    distributions=distributions,
                    state=optuna.trial.TrialState.COMPLETE if complete_row.complete else optuna.trial.TrialState.FAIL,
                    value=round(float(complete_row.acc), 12) if complete_row.complete else None))
            known = set(frame.condition_config)
            for _ in range(64):
                proposal = study.ask(fixed_distributions=distributions)
                # Canonical rounding makes, e.g., 1.099999999999 equal to warm 1.1.
                cfg = json.loads(config_json({**base, **proposal.params}))
                if config_json(cfg) not in known:
                    break
                study.tell(proposal, state=optuna.trial.TrialState.FAIL)
            else:
                stop_reason = 'no_unique_proposal_after_64_attempts'
                break
            print(f'TPE trial {index - reference_count + 1}/{args.max_new_trials}: {ours.describe(cfg)}', flush=True)

        for seed in seeds:
            previous = next((row for row in rows if row['trial_index'] == index
                             and row['init_seed'] == seed), None)
            if (previous is not None and previous['status'] == 'completed'
                    and np.isfinite([previous[key] for key in SCORES]).all()):
                continue
            started = time.perf_counter()
            try:
                metrics = ours.run_ours(bundle, cfg, seed, eval_overrides={'eigen_solver': 'dense'})
                if not np.isfinite([metrics[key] for key in SCORES]).all():
                    raise ValueError('Nonfinite FIT/TEST score')
                metrics.update(status='completed', error='')
            except Exception as error:
                metrics = {key: np.nan for key in ours.RESULT_COLUMNS}
                metrics.update(status='failed', error=f'{type(error).__name__}: {error}',
                               init_uses_pooled_fit=False, test_basis='failed_not_evaluated')
            metrics['wall_seconds'] = time.perf_counter() - started
            row_cfg = {key: ours.sigma_label(value) if key == 'kernel_sigma' else value
                       for key, value in cfg.items()}
            rows = [row for row in rows if not (row['trial_index'] == index and row['init_seed'] == seed)]
            rows.append({**provenance, 'init_seed': seed, **row_cfg, **metrics,
                'trial_index': index, 'condition': (f'start_{cfg["init_method"]}'
                    if cold_start and index < 2 else f'tpe_{index - startup_count}'),
                'condition_config': config_json(cfg), 'result_origin': 'trained',
                'objective': 'mean_fit_acc', 'study': 'nonimage_adaptive_tpe',
                'requested_max_new_trials': args.max_new_trials,
                'attempt_count': int(previous.get('attempt_count', 0)) + 1 if previous else 1,
                'fit_basis': 'native_dual_space'})
            pd.DataFrame(rows).sort_values(['trial_index', 'init_seed']).to_csv(raw_path, index=False)
            summarize(rows, seeds).to_csv(mean_path, index=False)
            print(f'  init={seed}: {metrics["status"]}, FIT={metrics["acc"]:.4f}, '
                  f'TEST={metrics["test_acc"]:.4f}', flush=True)
        attempted.add(index)

    summary = summarize(rows, seeds, stop_reason)
    summary.to_csv(mean_path, index=False)
    completed = summary[summary.complete]
    print(f'STOP: {stop_reason}; new trials already issued='
          f'{len(summary[summary.trial_index.ge(reference_count)])}.')
    if len(completed):
        best_row = completed.sort_values('acc', ascending=False, kind='stable').iloc[0]
        print(f'BEST on FIT: {best_row["config"]} | FIT={best_row.acc:.4f}, '
              f'TEST ACC={best_row.test_acc:.4f}')
    print(f'wrote {raw_path} and {mean_path}')
    return int(any(row['status'] != 'completed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())

