"""Freeze five held-out-subject HAR folds; each training subject is one client.

Uses the existing Stage 1/2 files and loaders. The partition token is 'subject',
not a numeric Dirichlet beta. No model fitting or result files are produced.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.datasets_and_utils import apply_pca_preprocessing
from experiments.partition_checks import array_hash, check_partition

SOURCE_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
              '00240/UCI%20HAR%20Dataset.zip')
ALLOCATION_SEED = 20261104
DATASET = 'har_subject'


def read_source(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        names = outer.namelist()
        nested = [name for name in names if name.endswith('UCI HAR Dataset.zip')]
        archive_bytes = outer.read(nested[0]) if nested else payload
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        blocks = []
        for split in ('train', 'test'):
            columns = []
            for filename in ('X', 'y', 'subject'):
                suffix = f'/{split}/{filename}_{split}.txt'
                matches = [name for name in names if name.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError('Official HAR archive lacks a unique ' + suffix)
                columns.append(np.loadtxt(io.BytesIO(archive.read(matches[0]))))
            blocks.append(columns)
    X = np.vstack([part[0] for part in blocks])
    y = np.concatenate([part[1] for part in blocks]).astype(np.int32) - 1
    subjects = np.concatenate([part[2] for part in blocks]).astype(np.int32)
    if (X.shape != (10299, 561) or len(y) != len(X) or len(subjects) != len(X)
            or not np.isfinite(X).all() or not np.array_equal(np.unique(y), np.arange(6))
            or not np.array_equal(np.unique(subjects), np.arange(1, 31))):
        raise ValueError('Unexpected official HAR rows, features, labels or subject IDs')
    return X, y, subjects


def prepare_subject_folds(X, y, subjects, output_dir, folds, source):
    """Shared with the temporary smoke test; labels never determine ownership."""
    root = Path(output_dir) / DATASET
    subject_order = np.random.RandomState(ALLOCATION_SEED).permutation(np.unique(subjects))
    held_out = np.array_split(subject_order, 5)
    for fold in folds:
        test_subjects = np.sort(held_out[fold])
        fit_subjects = np.setdiff1d(subject_order, test_subjects)
        clients = len(fit_subjects)
        stage1 = root / 'splits' / f'{DATASET}_centralized_seed_{fold}.npz'
        meta1 = stage1.with_name(stage1.stem + '_metadata.json')
        stage2 = root / 'partitions' / f'{DATASET}_seed_{fold}_beta_subject_clients_{clients}'
        files = (stage1, meta1, stage2 / 'partition.npz',
                 stage2 / 'client_splits.json', stage2 / 'partition_metadata.json')
        if all(path.exists() for path in files):
            saved = json.loads(meta1.read_text(encoding='utf-8'))
            if (saved['dataset_source'] != source or saved['split']['test_subjects'] != test_subjects.tolist()
                    or saved['split']['allocation_seed'] != ALLOCATION_SEED):
                raise ValueError('Existing subject folds use a different source or allocation')
            check_partition(DATASET, fold, 'subject', clients, output_dir)
            print(f'Reuse subject fold {fold}: {clients} natural clients')
            continue
        test_rows = np.flatnonzero(np.isin(subjects, test_subjects))
        fit_rows = np.flatnonzero(~np.isin(subjects, test_subjects))
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X[fit_rows])
        X_test = scaler.transform(X[test_rows])
        y_fit, y_test = y[fit_rows], y[test_rows]
        centers = np.array([X_fit[y_fit == label].mean(axis=0) for label in range(6)])
        # Same HAR train-only PCA routine as the record-level benchmark.
        with contextlib.redirect_stdout(io.StringIO()):
            X_fit, X_test, centers = apply_pca_preprocessing(
                X_fit, X_test, centers, dict(dataset='har', apply_pca=True, random_state=fold))
        arrays = dict(X_train=np.ascontiguousarray(X_fit, dtype=np.float32),
                      X_test=np.ascontiguousarray(X_test, dtype=np.float32),
                      y_train=np.asarray(y_fit, dtype=np.int32),
                      y_test=np.asarray(y_test, dtype=np.int32),
                      centers=np.ascontiguousarray(centers, dtype=np.float32),
                      raw_indices_train=fit_rows, raw_indices_test=test_rows,
                      subjects_train=subjects[fit_rows], subjects_test=subjects[test_rows])
        split = dict(source='five held-out-subject folds from pooled official HAR',
            n_train=len(fit_rows), n_test=len(test_rows), n_classes=6,
            stratified=False, allocation_seed=ALLOCATION_SEED,
            train_subjects=fit_subjects.tolist(), test_subjects=test_subjects.tolist(),
            fold_count=5, test_subjects_disjoint_across_folds=True,
            training_folds_overlap=True, original_official_split_preserved=False)
        pre = dict(source='data.datasets_and_utils.apply_pca_preprocessing',
            scaler_fit_on='FIT subjects only; TEST is transformed, never fitted',
            apply_pca=True, original_features=X.shape[1], final_features=arrays['X_train'].shape[1])
        metadata = dict(dataset=DATASET, data_seed=fold, split=split,
            preprocessing=pre, dataset_source=source,
            arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_hash(value))
                    for key, value in arrays.items()})
        ownership = [np.flatnonzero(subjects[fit_rows] == subject).astype(np.int32)
                     for subject in fit_subjects]
        client_info = {str(i): dict(subject_id=int(subject), indices=indices.tolist(),
            size=len(indices), class_distribution=np.bincount(y_fit[indices], minlength=6).tolist())
            for i, (subject, indices) in enumerate(zip(fit_subjects, ownership))}
        sizes = list(map(len, ownership))
        partition_meta = dict(dataset=DATASET, data_seed=fold, split=split,
            preprocessing=pre, dataset_source=source,
            federated=dict(num_clients=clients, dirichlet_beta=None, partition_strategy='subject',
                client_subject_ids=fit_subjects.tolist(), client_sizes=sizes,
                min_size=min(sizes), max_size=max(sizes), mean_size=float(np.mean(sizes))),
            client_indices_file='client_splits.json', partition_arrays_file='partition.npz',
            scope='one FIT subject per client; held-out TEST subjects are never assigned to clients')
        stage1.parent.mkdir(parents=True, exist_ok=True)
        stage2.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stage1, **arrays)
        meta1.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
        np.savez_compressed(stage2 / 'partition.npz', **arrays)
        (stage2 / 'client_splits.json').write_text(json.dumps(client_info, indent=2), encoding='utf-8')
        (stage2 / 'partition_metadata.json').write_text(
            json.dumps(partition_meta, indent=2, sort_keys=True), encoding='utf-8')
        check_partition(DATASET, fold, 'subject', clients, output_dir)
        print(f'Frozen subject fold {fold}: FIT={len(fit_rows)} TEST={len(test_rows)} '
              f'M={clients} | held-out subjects={test_subjects.tolist()}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-zip')
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    folds = [int(value) for value in args.folds.split(',')]
    if not folds or len(set(folds)) != len(folds) or any(value not in range(5) for value in folds):
        parser.error('--folds must be distinct values from 0 to 4')
    path = Path(args.source_zip) if args.source_zip else Path(args.output_dir) / DATASET / 'source' / 'uci_har.zip'
    if args.dry_run:
        print(f'Plan only: {SOURCE_URL} -> {path}; folds={folds}; 24 subject clients per fold.')
        return 0
    payload = path.read_bytes() if path.exists() else urllib.request.urlopen(SOURCE_URL, timeout=120).read()
    X, y, subjects = read_source(payload)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    source = dict(provider='UCI official archive', source_url=SOURCE_URL,
        source_sha256=hashlib.sha256(payload).hexdigest(), raw_samples=10299,
        raw_features=561, raw_subjects=30, subject_id_is_feature=False)
    prepare_subject_folds(X, y, subjects, args.output_dir, folds, source)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
