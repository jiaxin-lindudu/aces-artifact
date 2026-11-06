"""
Stage 1: prepare_centralized_splits.py

Prepare centralized FIT/TEST splits for all datasets across different seeds.
Each split is independent of federated partitions and can be used directly
for centralized ceiling benchmarks.

Structure:
  data/{dataset}/
    splits/
      {dataset}_seed_{seed}.npz              # Preprocessed FIT/TEST arrays
      {dataset}_seed_{seed}_metadata.json    # Preprocessing provenance & hashes
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets_and_utils import get_dataset, apply_pca_preprocessing, DATASET_SOURCES


def _compute_digest(array):
    """Compute SHA-256 digest of a NumPy array."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=float).tobytes()).hexdigest()


def prepare_centralized_split(dataset, data_seed, output_dir, apply_pca=True,
                              samples_per_cluster=200, num_clusters=10,
                              n_features=None, force=False, verbose=True):
    """
    Prepare and save a centralized FIT/TEST split.

    Stage 1 of the pipeline. The split depends only on (dataset, data_seed) and is
    independent of any federated partitioning. Both FIT and TEST are preprocessed
    (scaler fitted on FIT only) and saved together.

    Args:
        dataset (str): Dataset name (e.g., 'digits', 'cifar10_vit', 'usps')
        data_seed (int): Random seed for train/test split
        output_dir (str): Root output directory for splits
        apply_pca (bool): Whether to apply PCA for high-dimensional datasets
        samples_per_cluster (int): Samples per cluster for synthetic datasets
        num_clusters (int): Number of clusters for dataset generation
        n_features (int, optional): Target feature dimension (None = auto)
        force (bool): Overwrite existing splits
        verbose (bool): Print progress

    Returns:
        tuple: (split_path, metadata)
    """
    dataset = 'bank' if dataset == 'banknote' else dataset
    split_dir = Path(output_dir) / dataset / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)

    # Determine cache suffix to uniquely identify this split
    seed_suffix = '' if data_seed == 42 else f'_seed_{data_seed}'
    split_filename = f'{dataset}_centralized{seed_suffix}'
    split_path = split_dir / f'{split_filename}.npz'
    meta_path = split_dir / f'{split_filename}_metadata.json'

    # Check if already exists
    if split_path.exists() and meta_path.exists() and not force:
        if verbose:
            print(f'✓ Split exists: {split_path.relative_to(ROOT)}')
        with open(meta_path, encoding='utf-8') as f:
            return str(split_path), json.load(f)

    if verbose:
        print(f'\n=== Preparing Centralized Split: {dataset} seed={data_seed} ===')

    # Load and preprocess data
    requested_samples = num_clusters * samples_per_cluster
    if verbose:
        print(f'Loading {dataset} (n_samples={requested_samples}, n_clusters={num_clusters})...')

    X_train, X_test, y_train, y_test, centers = get_dataset(
        dataset,
        n_samples=requested_samples,
        n_clusters=num_clusters,
        n_features=n_features,
        random_state=data_seed
    )
    original_features = int(X_train.shape[1])

    # Preprocess: standardization + optional PCA
    cfg = {
        'dataset': dataset,
        'apply_pca': apply_pca,
        'random_state': data_seed,
        'num_features': n_features or X_train.shape[1]
    }
    X_train, X_test, centers = apply_pca_preprocessing(X_train, X_test, centers, cfg)

    # Ensure contiguous arrays and convert to float32 for storage efficiency
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)

    # Compute digests for reproducibility verification
    arrays = {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'centers': centers
    }

    # Save arrays
    if verbose:
        print(f'Saving arrays to {split_path.name}...')
    np.savez_compressed(split_path, **arrays)

    # Prepare metadata
    n_classes = len(np.unique(y_train))
    metadata = {
        'dataset': dataset,
        'data_seed': int(data_seed),
        'split': {
            'source': 'data.datasets_and_utils.get_dataset',
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_classes': n_classes,
            'test_fraction': round(len(X_test) / (len(X_train) + len(X_test)), 6),
            'stratified': True,
        },
        'preprocessing': {
            'source': 'data.datasets_and_utils.apply_pca_preprocessing',
            'scaler_fit_on': 'FIT block only; TEST is transformed, never fitted',
            'apply_pca': apply_pca,
            'final_features': int(X_train.shape[1]),
            'original_features': original_features,
        },
        'arrays': {
            name: {
                'shape': tuple(map(int, array.shape)),
                'dtype': str(array.dtype),
                'sha256': _compute_digest(array)
            }
            for name, array in arrays.items()
        }
    }

    # Add dataset-specific metadata if available
    if dataset in DATASET_SOURCES:
        metadata['dataset_source'] = DATASET_SOURCES[dataset]

    # Save metadata
    if verbose:
        print(f'Saving metadata to {meta_path.name}...')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if verbose:
        print(f'✓ Split prepared: {len(X_train)} train + {len(X_test)} test samples')
        print(f'  Classes: {n_classes}, Features: {X_train.shape[1]}')

    return str(split_path), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Stage 1: Prepare centralized FIT/TEST splits for clustering experiments'
    )
    parser.add_argument('--datasets', default='digits',
                        help='comma-separated dataset names')
    parser.add_argument('--data-seeds', default='0,1,2,3,4',
                        help='comma-separated data seeds')
    parser.add_argument('--output-dir', default=None,
                        help='root output directory (default: PROJECT/data)')
    parser.add_argument('--samples-per-cluster', type=int, default=200)
    parser.add_argument('--num-clusters', type=int, default=10)
    parser.add_argument('--apply-pca', action='store_true', default=True,
                        help='apply PCA for high-dimensional datasets')
    parser.add_argument('--force', action='store_true',
                        help='overwrite existing splits')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = str(ROOT / 'data')

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s) for s in args.data_seeds.split(',') if s.strip()]

    count = 0
    for dataset in datasets:
        for seed in seeds:
            try:
                prepare_centralized_split(
                    dataset, seed, args.output_dir,
                    apply_pca=args.apply_pca,
                    samples_per_cluster=args.samples_per_cluster,
                    num_clusters=args.num_clusters,
                    force=args.force,
                    verbose=not args.quiet
                )
                count += 1
            except Exception as e:
                print(f'✗ Failed: {dataset} seed={seed}: {e}', file=sys.stderr)
                return 1

    print(f'\n✓ Centralized splits prepared: {count} | root={args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
