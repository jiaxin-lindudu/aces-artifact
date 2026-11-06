"""Read-only checks on the existing Stage 1/2 format, not another data pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = ('X_train', 'X_test', 'y_train', 'y_test')
PROVENANCE_COLUMNS = ('data_sha256', 'partition_sha256')


def array_hash(array):
    # Match Stage 1 metadata and the centralized runner, including saved float32.
    return hashlib.sha256(np.ascontiguousarray(array, dtype=float).tobytes()).hexdigest()


def check_partition(dataset, data_seed, beta, clients, base_dir):
    """Validate all four canonical arrays, metadata and exact FIT ownership."""
    from data.datasets_and_utils import load_centralized_split, load_federated_partition
    central = load_centralized_split(dataset, data_seed, base_dir=base_dir, verbose=False)
    fed = load_federated_partition(dataset, data_seed, beta, clients,
                                  base_dir=base_dir, verbose=False)
    name = 'bank' if dataset == 'banknote' else dataset
    for metadata in (central.metadata, fed.metadata):
        if metadata.get('dataset') != name or metadata.get('data_seed') != int(data_seed):
            raise ValueError('Saved dataset/data_seed identity differs from the requested split')
    hashes = {}
    for key in ARRAY_KEYS:
        a, b = getattr(central, key), getattr(fed, key)
        if not np.array_equal(a, b):
            raise ValueError(f'Stage 1/2 mismatch: {key}')
        hashes[key] = array_hash(a)
        expected = central.metadata.get('arrays', {}).get(key, {}).get('sha256')
        if expected and hashes[key] != expected:
            raise ValueError(f'Stage 1 metadata hash mismatch: {key}')
    ownership = [np.asarray(idx, dtype=int).tolist() for idx in fed.client_indices]
    if len(ownership) != clients or any(not idx for idx in ownership):
        raise ValueError('Each configured client must own at least one FIT row')
    if not np.array_equal(np.sort(np.concatenate(ownership)), np.arange(len(fed.X_train))):
        raise ValueError('FIT ownership must cover every row exactly once')
    return {
        'data_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'partition_sha256': hashlib.sha256(json.dumps(ownership).encode()).hexdigest(),
        'arrays_sha256': hashes,
    }


def check_all_partitions(base_dir, dataset='digits'):
    records = []
    for path in sorted((Path(base_dir) / dataset / 'partitions').glob('*/partition_metadata.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        fed = meta['federated']
        records.append(dict(dataset=dataset, data_seed=meta['data_seed'],
            beta=fed['dirichlet_beta'], clients=fed['num_clients'],
            **check_partition(dataset, meta['data_seed'], fed['dirichlet_beta'],
                              fed['num_clients'], base_dir)))
    return records


def require_spectral_memory(samples, available_bytes=None, *, dense_arrays=6,
                            working_set_bytes=None):
    """Check the caller's working set; dense solvers keep the original estimate."""
    estimated = (int(dense_arrays) * int(samples) ** 2 * 8
                 if working_set_bytes is None else int(working_set_bytes))
    if available_bytes is None:
        import os
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ('total_phys', 'available_phys', 'total_page', 'available_page',
                     'total_virtual', 'available_virtual', 'extended')]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                available_bytes = status.available_phys
        elif hasattr(os, 'sysconf'):
            available_bytes = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    if available_bytes is not None and estimated > available_bytes // 2:
        kind = 'Dense spectral' if working_set_bytes is None else 'Spectral'
        raise MemoryError('%s working-set estimate %.2f GiB exceeds half available RAM %.2f GiB '
                          '(available RAM %.2f GiB); '
                          'FIT is unchanged. Prepare and disclose a separate common split if needed.'
                          % (kind, estimated / 2**30, (available_bytes // 2) / 2**30,
                             available_bytes / 2**30))
    return estimated
