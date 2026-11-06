"""Calibrate the existing privacy attacks with exposed-output and shuffled-label controls.

Reuse frozen FIT/TEST and saved privacy references. No clustering or CKKS training
is repeated. Exposed FIT records/plaintext statistics are artificial positive
controls, not outputs released by ACES or claims about CKKS security.
Save every completed condition/seed; --resume reuses those diagnostic records.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets_and_utils import load_federated_partition
from experiments.partition_checks import check_partition
from federated.ours.tracker import evaluate_attack_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir', required=True,
                        help='Directory containing the completed grid_ours_privacy CSV')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed condition/seed records from the raw CSV')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error('--init-seeds must contain distinct seeds')

    source = Path(args.source_dir) / f'grid_ours_privacy_beta{args.beta}_clients{args.clients}.csv'
    reference = pd.read_csv(source)
    reference = reference[reference.condition.isin(['plain', 'ckks'])
                          & reference.init_seed.isin(seeds)].copy()
    partition = load_federated_partition(args.dataset, args.grid_split, args.beta,
                                         args.clients, base_dir=args.data_dir, verbose=False)
    proof = check_partition(args.dataset, args.grid_split, args.beta, args.clients, args.data_dir)
    identity = dict(dataset=args.dataset, data_seed=args.grid_split,
                    beta=args.beta, clients=args.clients)
    # Validate external references once before using their configuration/metrics.
    for key, value in {**identity, **{k: proof[k] for k in ('data_sha256', 'partition_sha256')}}.items():
        if not reference[key].eq(value).all():
            raise ValueError(f'Reference differs from the frozen partition: {key}')
    if (len(reference) != 2 * len(seeds)
            or reference.duplicated(['condition', 'init_seed']).any()
            or not reference.mia_status.eq('evaluated').all()
            or not reference.mia_folds.eq(5).all()):
        raise ValueError('References must contain one evaluated 5-fold plain/CKKS MIA per seed')
    X_fit = np.vstack([partition.X_train[index] for index in partition.client_indices]).astype(float)
    X_test = np.asarray(partition.X_test, dtype=float)
    K = int(partition.metadata['split']['n_classes'])
    count = min(len(X_fit), len(X_test), 1000)
    if count < 10 or len(X_fit) < K or not reference.mia_per_class.eq(count).all():
        raise ValueError('FIT/TEST or saved MIA sample counts do not match the control protocol')
    reference = reference.set_index(['condition', 'init_seed'])
    out = Path(args.out_dir)
    stem = f'grid_ours_privacy_control_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    source_proof = dict(reference_source_csv=str(source.resolve()),
                        reference_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    conditions = ('mia_exposed_fit', 'mia_shuffled_labels', 'bytes_plaintext', 'bytes_shuffled_labels')
    expected = {(condition, seed) for condition in conditions for seed in seeds}
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Preserve existing results: use --resume or another --out-dir')
    if mean_path.exists() and not raw_path.exists():
        raise FileNotFoundError('Cannot resume without the raw CSV')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    if len(existing):
        required = set(identity) | {'condition', 'init_seed', 'status', 'prediction_status',
            'data_sha256', 'partition_sha256', *source_proof}
        if not required <= set(existing):
            raise ValueError('Existing output lacks resume metadata; use a new out-dir')
        keys = list(zip(existing.condition, existing.init_seed))
        if len(set(keys)) != len(keys) or not set(keys) <= expected:
            raise ValueError('Existing output has duplicate or unexpected condition/seed rows')
        for key, value in {**identity, **{k: proof[k] for k in ('data_sha256', 'partition_sha256')},
                           **source_proof}.items():
            if not existing[key].eq(value).all():
                raise ValueError('Existing output differs from this study: ' + key)
    rows = existing.to_dict('records')
    done = set(zip(existing.condition, existing.init_seed)) if len(existing) else set()
    print(f'{args.dataset} seed={args.grid_split} clients={args.clients} | '
          f'FIT={len(X_fit)} TEST={len(X_test)} K={K}')
    print(f'4 attack controls x {len(seeds)} seeds; {len(done)} saved; no training, search or encryption.')
    print('Artificial exposure controls only; chance AUC is not a security proof.')
    if args.dry_run:
        print('[dry-run] no attacks or output files.')
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if all((condition, seed) in done for condition in conditions):
            continue
        plain = reference.loc[('plain', seed)]
        encrypted = reference.loc[('ckks', seed)]
        controls = []
        if any((condition, seed) not in done for condition in conditions[:2]):
            started = time.perf_counter()
            rng = np.random.RandomState(seed)
            # Same sampling rule and K-dimensional sorted-distance feature family as
            # the original MIA, but deliberately expose all FIT records as landmarks.
            probes = np.vstack((X_fit[rng.choice(len(X_fit), count, replace=False)],
                                X_test[rng.choice(len(X_test), count, replace=False)]))
            target = np.r_[np.ones(count, dtype=int), np.zeros(count, dtype=int)]
            features = np.empty((2 * count, K))
            for begin in range(0, len(probes), 128):
                distances = cdist(probes[begin:begin + 128], X_fit)
                nearest = np.partition(distances, K - 1, axis=1)[:, :K]
                features[begin:begin + len(nearest)] = np.sort(nearest, axis=1)
            feature_seconds = time.perf_counter() - started
            controls.extend([
                ('mia_exposed_fit', 'center_distance', features, target,
                 'all FIT records deliberately exposed; sorted K-nearest distances', feature_seconds),
                ('mia_shuffled_labels', 'center_distance', features,
                 np.random.RandomState(seed + 1000003).permutation(target),
                 'same exposed-FIT features; membership targets permuted', feature_seconds),
            ])
        if any((condition, seed) not in done for condition in conditions[2:]):
            started = time.perf_counter()
            bits = np.random.RandomState(seed).permutation(np.tile([0, 1], 100))
            message_values = K * (X_fit.shape[1] + int(plain.d_emb) + 1)
            byte_features = []
            # Same length/histogram/mean/std feature family as the ciphertext probe;
            # this positive control deliberately exposes raw float64 bytes, not CKKS.
            for bit in bits:
                serialized = np.full(message_values, float(bit), dtype=np.float64).tobytes()
                values = np.frombuffer(serialized, dtype=np.uint8)
                histogram = np.histogram(values, bins=16, range=(0, 256))[0] / len(values)
                byte_features.append(np.r_[len(serialized), histogram, values.mean(), values.std()])
            byte_features = np.asarray(byte_features)
            feature_seconds = time.perf_counter() - started
            controls.extend([
                ('bytes_plaintext', 'serialized_byte', byte_features, bits,
                 'raw float64 bytes of equal-shape all-zero versus all-one statistics', feature_seconds),
                ('bytes_shuffled_labels', 'serialized_byte', byte_features,
                 np.random.RandomState(seed + 1000033).permutation(bits),
                 'same plaintext-byte features; challenge targets permuted', feature_seconds),
            ])
        for condition, attack, values, labels, view, preparation in controls:
            if (condition, seed) in done:
                continue
            started = time.perf_counter()
            result = evaluate_attack_features(values, labels, seed=seed, folds=5)
            row = dict(**identity, init_seed=seed, study='privacy_control', condition=condition,
                       attack=attack, K=K, d_orig=X_fit.shape[1], d_emb=int(plain.d_emb),
                       status=result['status'], attacker_view=view, n_samples=len(labels),
                       feature_seconds=preparation, attack_seconds=time.perf_counter() - started,
                       **source_proof, result_origin='attack_control_only',
                       data_sha256=proof['data_sha256'], partition_sha256=proof['partition_sha256'],
                       reference_plain_auc=float(plain.mia_auc) if attack == 'center_distance' else np.nan,
                       reference_ckks_auc=float(encrypted.mia_auc if attack == 'center_distance'
                                                else encrypted.ciphertext_auc),
                       formal_dp='not provided', is_security_proof=False)
            for key in ('auc', 'acc', 'balanced_accuracy', 'tpr', 'tnr', 'folds',
                        'degenerate_folds', 'prediction_status'):
                row[key] = result.get(key, np.nan)
            rows.append(row)
            done.add((condition, seed))
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f'  {condition:22s} init={seed} AUC={row["auc"]:.4f} '
                  f'balanced_acc={row["balanced_accuracy"]:.4f} {row["prediction_status"]}', flush=True)

    frame = pd.DataFrame(rows)
    metrics = ['auc', 'acc', 'balanced_accuracy', 'tpr', 'tnr', 'degenerate_folds',
               'feature_seconds', 'attack_seconds', 'reference_plain_auc', 'reference_ckks_auc']
    groups = frame.groupby(['attack', 'condition'], sort=False)
    mean = groups[metrics].mean()
    mean = mean.join(groups[metrics].std(ddof=1).add_suffix('_std')).reset_index()
    for key, value in {**identity, 'n_seeds': len(seeds), 'study': 'privacy_control',
                       **source_proof,
                       'data_sha256': proof['data_sha256'], 'partition_sha256': proof['partition_sha256'],
                       'is_security_proof': False}.items():
        mean[key] = value
    for key in ('status', 'prediction_status', 'attacker_view'):
        mean[key] = groups[key].agg(lambda v: '; '.join(pd.unique(v))).to_numpy()
    mean.to_csv(mean_path, index=False)
    print(mean[['condition', 'auc', 'auc_std', 'balanced_accuracy']].to_string(index=False))
    print(f'wrote {raw_path} and {mean_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
