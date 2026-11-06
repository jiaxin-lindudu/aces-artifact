"""Compare complete FIT-selected configurations on the README frozen split.

Produces one comparison CSV alongside the existing grid CSVs. No diagnostic
files or new data partitions are created. Legacy rows without recorded hashes
are visible as legacy_unverified, never silently certified after the fact.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import schema
from experiments.partition_checks import check_partition

METHODS = ('ours', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc', 'fastlloyd')
METRICS = ('acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari',
           'rounds_run', 'wall_seconds', 'uplink_bytes', 'downlink_bytes',
           'total_bytes', 'bytes_per_round', 'training_seconds')


def exact_seeds(frame, seeds):
    return (len(frame) == len(seeds)
            and sorted(frame.init_seed.tolist()) == sorted(seeds))


def select_complete(raw, mean, axes, seeds, select_on):
    """Keep mean-FIT selection; never let a missing/failed seed win."""
    for _, cfg in mean.sort_values(select_on, ascending=False, kind='stable').iterrows():
        chosen = raw
        for key in axes:
            value = cfg[key]
            mask = chosen[key].isna() if pd.isna(value) else chosen[key].eq(value)
            chosen = chosen[mask]
        if not exact_seeds(chosen, seeds):
            continue
        if 'status' in chosen and not chosen.status.eq('completed').all():
            continue
        if not np.isfinite(chosen[list(METRICS[:6])].to_numpy(float)).all():
            continue
        if not np.isclose(chosen[select_on].mean(), cfg[select_on], atol=1e-12, rtol=0):
            raise ValueError('Mean CSV selection score differs from selected raw runs')
        return chosen, {key: cfg[key] for key in axes}
    raise ValueError('No configuration has the full expected seed set and finite FIT/TEST scores')


def summarize(chosen):
    out = {'n_seeds': len(chosen)}
    for key in METRICS:
        if key in chosen:
            values = pd.to_numeric(chosen[key], errors='coerce')
            out[key + '_mean'] = values.mean()
            out[key + '_std'] = values.std(ddof=1)
    for key in ('fit_basis', 'test_basis', 'initialization', 'communication_basis', 'privacy_status'):
        if key in chosen:
            out[key] = '; '.join(map(str, chosen[key].dropna().unique()))
    if 'hit_cap' in chosen:
        out['capped_runs'] = int(chosen.hit_cap.eq(True).sum())
    if 'n_distinct_labels' in chosen:
        out['single_cluster_runs'] = int(chosen.n_distinct_labels.le(1).sum())
    return out


def comparison(dataset, data_seed, beta, clients, seeds, result_dir, central_dir,
               data_dir, methods=METHODS, select_on='acc'):
    proof = check_partition(dataset, data_seed, beta, clients, data_dir)
    rows = []
    for method in methods:
        row = dict(method=method, dataset=dataset, data_seed=data_seed, beta=beta,
                   clients=clients, data_sha256=proof['data_sha256'],
                   partition_sha256=proof['partition_sha256'], select_on=select_on,
                   uncertainty_scope='initialization/run variability on one frozen split')
        raw_path = Path(result_dir) / f'grid_{method}_beta{float(beta)}_clients{clients}.csv'
        mean_path = raw_path.with_name(raw_path.stem + '_mean.csv')
        try:
            raw, mean = pd.read_csv(raw_path), pd.read_csv(mean_path)
            if (not raw.dataset.eq(dataset).all() or not raw.data_seed.eq(data_seed).all()
                    or not raw.beta.eq(beta).all() or not raw.clients.eq(clients).all()):
                raise ValueError('CSV identity differs from requested partition')
            recorded = all(key in raw and raw[key].notna().all()
                           for key in ('data_sha256', 'partition_sha256'))
            if not recorded and any(key in raw for key in ('data_sha256', 'partition_sha256')):
                raise ValueError('Run provenance columns exist but are incomplete')
            if recorded:
                for key in ('data_sha256', 'partition_sha256'):
                    if not raw[key].eq(proof[key]).all():
                        raise ValueError('Recorded ' + key + ' differs from current frozen data')
            module = importlib.import_module('experiments.scripts.grid_search_' + method)
            chosen, cfg = select_complete(raw, mean, tuple(module.GRID), seeds, select_on)
            # Score the selected raw runs, rather than trusting copied means.
            row.update(summarize(chosen), config=json.dumps(cfg, default=str, sort_keys=True),
                       status='completed' if recorded else 'legacy_unverified',
                       provenance='recorded run hashes' if recorded else 'legacy CSV lacks run hashes',
                       source_csv=str(raw_path.resolve()), error='')
            if 'fit_basis' not in row:
                row['fit_basis'] = 'native_dual_space' if method == 'ours' else 'nearest_center'
            if 'test_basis' not in row:
                row['test_basis'] = 'nearest_center'
        except FileNotFoundError:
            row.update(status='missing', error='Required grid CSV is absent')
        except (ValueError, KeyError) as exc:
            row.update(status='incomplete_or_mismatched', error=str(exc))
        rows.append(row)

    path = Path(central_dir) / f'centralized_ceiling_{dataset}_seed{data_seed}.csv'
    meta_path = Path(central_dir) / f'centralized_run_metadata_{dataset}_seed{data_seed}.json'
    for method in ('central_kmeans', 'central_fcm', 'central_spectral'):
        row = dict(method=method, dataset=dataset, data_seed=data_seed,
                   data_sha256=proof['data_sha256'], select_on=select_on,
                   uncertainty_scope='initialization variability on one frozen split')
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('arrays_sha256') != proof['arrays_sha256']:
                raise ValueError('Centralized arrays differ from federated canonical arrays')
            if meta.get('select_on') != select_on:
                raise ValueError('Centralized selection metric differs')
            all_rows = pd.read_csv(path)
            chosen = all_rows[all_rows.method.eq(method)].copy()
            if not exact_seeds(chosen, seeds):
                raise ValueError('Centralized result has missing, extra or duplicated seeds')
            if (not chosen.dataset.eq(dataset).all() or not chosen.data_seed.eq(data_seed).all()
                    or not chosen.partition_sha256.eq(proof['arrays_sha256']['X_train']).all()):
                raise ValueError('Centralized row identity or FIT hash differs')
            for key in ('acc', 'nmi', 'ari'):
                chosen['test_' + key] = chosen['test_proxy_' + key]
            if not np.isfinite(chosen[list(METRICS[:6])].to_numpy(float)).all():
                raise ValueError('Centralized result contains nonfinite scores')
            row.update(summarize(chosen), status='completed', error='',
                       config='; '.join(chosen.config.unique()),
                       fit_basis='centralized_native', source_csv=str(path.resolve()),
                       provenance='recorded centralized array hashes')
            row['status'] = 'capped' if row.get('capped_runs', 0) else 'completed'
        except FileNotFoundError:
            row.update(status='missing', error='Centralized result/metadata is absent')
        except (ValueError, KeyError) as exc:
            row.update(status='incomplete_or_mismatched', error=str(exc))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', default='digits')
    p.add_argument('--grid-split', type=int, default=0)
    p.add_argument('--beta', type=float, default=.5)
    p.add_argument('--clients', type=int, default=20)
    p.add_argument('--init-seeds', default='40,41,42,43,44')
    p.add_argument('--methods', default=','.join(METHODS))
    p.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    p.add_argument('--data-dir', default=str(ROOT / 'data'))
    p.add_argument('--results-dir')
    p.add_argument('--central-dir')
    p.add_argument('--out-dir')
    args = p.parse_args(argv)
    seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    methods = args.methods.split(',')
    if not seeds or len(set(seeds)) != len(seeds) or not set(methods) <= set(METHODS):
        p.error('Use distinct init seeds and known methods')
    result_dir = args.results_dir or str(Path(schema.RESULTS_DIR) / args.dataset / f'federated_{args.grid_split}')
    central_dir = args.central_dir or str(Path(schema.RESULTS_DIR) / args.dataset / f'seed_{args.grid_split}')
    frame = comparison(args.dataset, args.grid_split, args.beta, args.clients,
                       seeds, result_dir, central_dir, args.data_dir, methods, args.select_on)
    out = Path(args.out_dir or result_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'comparison_beta{float(args.beta)}_clients{args.clients}.csv'
    if path.exists():
        raise FileExistsError('Preserve existing comparison: choose a new --out-dir')
    frame.to_csv(path, index=False)
    print(frame.reindex(columns=['method', 'status', 'n_seeds', 'acc_mean', 'test_acc_mean', 'capped_runs']).to_string(index=False))
    print('wrote', path)
    return int(frame.status.isin(['missing', 'incomplete_or_mismatched']).any())


if __name__ == '__main__':
    raise SystemExit(main())
