"""
Stage 2: prepare_federated_partitions.py

Build federated partitions on top of centralized FIT/TEST splits.
Each partition assigns FIT samples to clients via Dirichlet label-skew distribution,
while keeping the TEST block intact.

Structure:
  data/{dataset}/
    partitions/
      {dataset}_seed_{seed}_beta_{beta}_clients_{M}/
        partition.npz           # All arrays (X_train, X_test, y_train, y_test)
        client_splits.json      # Client ownership indices + metadata
        partition_metadata.json # Complete partition provenance
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.non_iid_loader import dirichlet_split
from data.datasets_and_utils import DATASET_SOURCES


def load_centralized_split(split_path):
    """Load preprocessed FIT/TEST arrays from centralized split."""
    with np.load(split_path) as archive:
        return {key: archive[key] for key in archive.files}


def prepare_federated_partition(split_path, beta, num_clients, output_dir,
                                force=False, verbose=True):
    """
    Stage 2: Build a federated partition from a centralized split.

    Assigns FIT samples to clients using Dirichlet label-skew distribution.
    TEST block and centers are kept intact.

    Args:
        split_path (str): Path to centralized split .npz
        beta (float): Dirichlet concentration parameter (controls Non-IID)
        num_clients (int): Number of clients
        output_dir (str): Output directory root
        force (bool): Overwrite existing partitions
        verbose (bool): Print progress

    Returns:
        tuple: (partition_dir, metadata)
    """
    # Parse split metadata
    meta_path = str(split_path).replace('.npz', '_metadata.json')
    with open(meta_path, encoding='utf-8') as f:
        split_meta = json.load(f)

    dataset = split_meta['dataset']
    data_seed = split_meta['data_seed']

    # Load split data
    arrays = load_centralized_split(split_path)
    X_train = np.asarray(arrays['X_train'], dtype=float)
    y_train = np.asarray(arrays['y_train'], dtype=int)
    X_test = np.asarray(arrays['X_test'], dtype=float)
    y_test = np.asarray(arrays['y_test'], dtype=int)
    centers = np.asarray(arrays['centers'], dtype=float)

    # Create partition directory
    partition_dir = Path(output_dir) / dataset / 'partitions' / \
                    f'{dataset}_seed_{data_seed}_beta_{beta}_clients_{num_clients}'
    partition_dir.mkdir(parents=True, exist_ok=True)

    partition_file = partition_dir / 'partition.npz'
    splits_file = partition_dir / 'client_splits.json'
    meta_file = partition_dir / 'partition_metadata.json'

    # Check if already exists
    if partition_file.exists() and splits_file.exists() and not force:
        if verbose:
            print(f'✓ Partition exists: {partition_dir.name}')
        with open(meta_file, encoding='utf-8') as f:
            return str(partition_dir), json.load(f)

    if verbose:
        print(f'\n=== Building Federated Partition ===')
        print(f'Dataset: {dataset}, seed={data_seed}, beta={beta}, clients={num_clients}')

    # Perform Dirichlet split on FIT data
    client_indices = dirichlet_split(
        labels=y_train,
        n_clients=num_clients,
        alpha=beta,
        random_state=data_seed
    )

    # Repair empty clients (min-size = 1 policy)
    for i in range(len(client_indices)):
        if len(client_indices[i]) == 0:
            donor = int(np.argmax([len(idx) for idx in client_indices]))
            if len(client_indices[donor]) > 1:
                client_indices[i] = np.append(client_indices[i],
                                              client_indices[donor][-1])
                client_indices[donor] = client_indices[donor][:-1]

    # Verify partition covers all FIT samples exactly once
    flattened = np.sort(np.concatenate([
        np.asarray(idx, dtype=int) for idx in client_indices
    ]))
    assert np.array_equal(flattened, np.arange(len(y_train))), \
        "Client partition does not cover FIT rows exactly once"

    # Convert to proper format
    client_indices = [np.asarray(idx, dtype=np.int32) for idx in client_indices]

    # Save partition arrays (FIT + TEST + centers)
    if verbose:
        print(f'Saving partition arrays...')
    partition_arrays = {
        'X_train': np.ascontiguousarray(X_train, dtype=np.float32),
        'y_train': np.ascontiguousarray(y_train, dtype=np.int32),
        'X_test': np.ascontiguousarray(X_test, dtype=np.float32),
        'y_test': np.ascontiguousarray(y_test, dtype=np.int32),
        'centers': np.ascontiguousarray(centers, dtype=np.float32),
    }
    np.savez_compressed(partition_file, **partition_arrays)

    # Save client indices and class counts
    if verbose:
        print(f'Saving client ownership...')
    class_labels, encoded_labels = np.unique(y_train, return_inverse=True)
    client_info = {}
    for cid, indices in enumerate(client_indices):
        client_info[str(cid)] = {
            'indices': indices.tolist(),
            'size': int(len(indices)),
            'class_distribution': np.bincount(
                encoded_labels[indices],
                minlength=len(class_labels)
            ).tolist()
        }

    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(client_info, f, indent=2)

    # Prepare comprehensive metadata
    n_classes = len(np.unique(y_train))
    client_sizes = [len(idx) for idx in client_indices]
    metadata = {
        'dataset': dataset,
        'data_seed': int(data_seed),
        'federated': {
            'num_clients': num_clients,
            'dirichlet_beta': float(beta),
            'client_sizes': client_sizes,
            'mean_size': float(np.mean(client_sizes)),
            'min_size': int(np.min(client_sizes)),
            'max_size': int(np.max(client_sizes)),
        },
        'split': {
            'source': 'centralized split (Stage 1)',
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_classes': n_classes,
            'class_labels': class_labels.tolist(),
            'test_fraction': round(len(X_test) / (len(X_train) + len(X_test)), 6),
        },
        'preprocessing': split_meta.get('preprocessing', {}),
        'client_indices_file': 'client_splits.json',
        'partition_arrays_file': 'partition.npz',
        'scope': (
            'FIT rows assigned to clients; TEST block and centers shared across all clients'
        )
    }

    # Add dataset source if available
    if dataset in DATASET_SOURCES:
        metadata['dataset_source'] = DATASET_SOURCES[dataset]

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if verbose:
        print(f'✓ Partition built:')
        print(f'  Clients: {num_clients}, Beta: {beta}')
        print(f'  FIT sizes: min={min(client_sizes)}, max={max(client_sizes)}, avg={np.mean(client_sizes):.1f}')
        print(f'  TEST: {len(X_test)} samples (shared)')

    return str(partition_dir), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Stage 2: Build federated partitions from centralized splits'
    )
    parser.add_argument('--datasets', default='digits',
                        help='comma-separated dataset names')
    parser.add_argument('--data-seeds', default='0,1,2,3,4')
    parser.add_argument('--betas', default='0.1,0.5,1.0',
                        help='comma-separated Dirichlet beta values')
    parser.add_argument('--clients', type=int, default=10,
                        help='number of clients')
    parser.add_argument('--input-dir', default=None,
                        help='directory containing Stage 1 splits (default: PROJECT/data)')
    parser.add_argument('--output-dir', default=None,
                        help='root output directory for partitions (default: PROJECT/data)')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    if args.input_dir is None:
        args.input_dir = str(ROOT / 'data')
    if args.output_dir is None:
        args.output_dir = str(ROOT / 'data')

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s) for s in args.data_seeds.split(',') if s.strip()]
    betas = [float(b) for b in args.betas.split(',') if b.strip()]

    count = 0
    for dataset in datasets:
        for seed in seeds:
            # Locate corresponding centralized split
            split_path = (Path(args.input_dir) / dataset / 'splits' /
                          f"{dataset}_centralized{'' if seed == 42 else f'_seed_{seed}'}.npz")

            if not split_path.exists():
                print(f'✗ Split not found: {split_path}', file=sys.stderr)
                print(f'  Run: python data/prepare_centralized_splits.py --datasets {dataset} --data-seeds {seed}')
                return 1

            for beta in betas:
                try:
                    prepare_federated_partition(
                        str(split_path), beta, args.clients,
                        args.output_dir,
                        force=args.force,
                        verbose=not args.quiet
                    )
                    count += 1
                except Exception as e:
                    print(f'✗ Failed: {dataset} seed={seed} beta={beta}: {e}', file=sys.stderr)
                    return 1

    print(f'\n✓ Federated partitions built: {count} | root={args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
