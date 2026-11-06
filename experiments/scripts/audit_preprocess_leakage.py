"""Audit: does the frozen FIT/TEST preprocessing ever see TEST statistics?

A scaler fitted on train+test leaks the test set into every published number.
``data.datasets_and_utils.get_dataset`` has two paths, both leak-free by
construction, and this script recomputes each of them from the raw source data
and checks the frozen arrays byte-for-byte:

* ``standardized``  X_scaled = (X - mean(FIT)) / std(FIT); the leak hypothesis is
                    the same transform with mean/std of FIT+TEST.
* ``divide_by_16``   X_scaled = X / 16.0. No learned statistics at all, so no
                    leak is even expressible; the check still recomputes it and
                    shows a learned FULL-data standardization would differ.

If H_fit reproduces the frozen arrays and H_leak does not, the claim in
partition.json is true. If both reproduce them, the split is degenerate
(FIT == ALL) and the check is inconclusive.

Run: python experiments/scripts/audit_preprocess_leakage.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import schema  # noqa: E402
from experiments.scripts.make_partition import preprocess_meta, refresh_partition_meta  # noqa: E402


def raw_source(dataset):
    """The exact array ``get_dataset`` starts from, before any splitting."""
    if dataset == 'digits':
        from sklearn.datasets import load_digits
        data = load_digits()
        return np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=int)
    raise ValueError('this audit currently reconstructs the digits path only '
                     '(got %r)' % dataset)


def reconstruct(dataset, X_raw, y_raw, data_seed):
    """Reproduce the exact (train, test) blocks get_dataset would write."""
    from sklearn.model_selection import train_test_split
    # Row order is part of the identity of the split: train_test_split returns
    # shuffled blocks, so the indices must NOT be sorted.
    idx_train, idx_test = train_test_split(
        np.arange(len(X_raw)), test_size=0.2, random_state=data_seed, stratify=y_raw)
    if dataset == 'digits':
        return X_raw[idx_train] / 16.0, X_raw[idx_test] / 16.0
    fit_mean = X_raw[idx_train].mean(axis=0)
    fit_std = X_raw[idx_train].std(axis=0)
    return ((X_raw[idx_train] - fit_mean) / np.where(fit_std > 0, fit_std, 1.0),
            (X_raw[idx_test] - fit_mean) / np.where(fit_std > 0, fit_std, 1.0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', default='digits')
    parser.add_argument('--data-seeds', default='0,1,2,3,4')
    parser.add_argument('--betas', default='0.1,0.5,1.0')
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--refresh-meta', action='store_true',
                        help='rewrite partition.json with the audited provenance')
    args = parser.parse_args(argv)

    failures = 0
    for dataset in [name.strip() for name in args.datasets.split(',') if name.strip()]:
        X_raw, y_raw = raw_source(dataset)
        print('raw %s: shape=%s range=[%.3f, %.3f] | claimed preprocess=%s'
              % (dataset, X_raw.shape, X_raw.min(), X_raw.max(),
                 preprocess_meta(dataset)['scaler']))
        for beta in [float(v) for v in args.betas.split(',') if v.strip()]:
            for data_seed in [int(v) for v in args.data_seeds.split(',') if v.strip()]:
                directory = schema.partition_path(dataset, data_seed, beta, args.clients)
                if not os.path.isfile(os.path.join(directory, 'partition.json')):
                    continue
                meta, arrays = schema.load_partition(directory)
                frozen_train = np.asarray(arrays['X_train'], dtype=float)
                frozen_test = np.asarray(arrays['X_test'], dtype=float)

                h_fit_train, h_fit_test = reconstruct(dataset, X_raw, y_raw, data_seed)
                # The leak hypothesis: a learned scaler fitted on train+test.
                all_mean = X_raw.mean(axis=0)
                all_std = X_raw.std(axis=0)
                h_leak_train = (X_raw - all_mean) / np.where(all_std > 0, all_std, 1.0)
                from sklearn.model_selection import train_test_split
                idx_train, idx_test = train_test_split(
                    np.arange(len(X_raw)), test_size=0.2, random_state=data_seed,
                    stratify=y_raw)
                h_leak_train = h_leak_train[idx_train]
                h_leak_test = (X_raw - all_mean) / np.where(all_std > 0, all_std, 1.0)
                h_leak_test = h_leak_test[idx_test]

                err = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max())
                e_fit = max(err(frozen_train, h_fit_train), err(frozen_test, h_fit_test))
                e_leak = max(err(frozen_train, h_leak_train), err(frozen_test, h_leak_test))
                ok = e_fit < 1e-9 and e_leak > 1e-6
                print('%-34s | max|frozen - claimed| = %.3e | '
                      'max|frozen - ALL-fitted| = %.3e | %s'
                      % (os.path.basename(directory), e_fit, e_leak,
                         'PASS no leak' if ok else 'FAIL'))
                if not ok:
                    failures += 1
                if args.refresh_meta:
                    refresh_partition_meta(directory)
    if args.refresh_meta:
        print('\npartition.json rewritten with per-dataset provenance')
    print('\nfailures: %d' % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
