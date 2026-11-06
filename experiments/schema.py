"""Frozen-partition format and results-table contract, in one place.

Two things must stay identical across every method and every script, otherwise
the paired comparison is meaningless:

* the on-disk partition (``partition.json`` + ``arrays.npz``) that defines the
  FIT rows, the held-out TEST rows and the client ownership;
* the ordered result columns, because ``federated.ours.tracker.summarize_results``
  and ``two_factor_summary`` group on ``GROUP_KEYS`` and silently drop any metric
  column that is missing.

Global index convention: rows are addressed in a stacked space where the FIT
block occupies ``[0, n_train)`` and the TEST block ``[n_train, n_train+n_test)``.
The FIT block comes first, so a client index is simultaneously a global index
and a row position inside ``X_train`` -- which is exactly what
``federated_anchored_clustering(train_indices=...)`` validates.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTITIONS_DIR = os.path.join(ROOT, 'experiments', 'partitions')
#: The FIT/TEST split itself, frozen BEFORE any client assignment. A partition is
#: this split plus a client ownership map; the ceilings read the split, so that a
#: federated run can be re-partitioned on exactly the same training set.
SPLITS_DIR = os.path.join(ROOT, 'experiments', 'splits')
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results')
FIGURES_DIR = os.path.join(ROOT, 'experiments', 'figures')
CONFIGS_DIR = os.path.join(ROOT, 'experiments', 'configs')
TRACES_DIR = os.path.join(RESULTS_DIR, 'traces')

# Grouping keys read by summarize_results() / two_factor_summary().
GROUP_KEYS = ('dataset', 'method', 'data_seed', 'dirichlet_beta', 'clients',
              'clusters', 'study', 'study_value')

# Provenance of one run; never used for grouping. The diagnostic columns added
# for the accuracy ablation (stop_reason, boundary_mode, init_method, d_emb,
# enforce_topology, use_anchors, kernel_scale_mode, test_embedding_mode,
# preprocess) live here too: they are per-run labels, so summarize_results() and
# two_factor_summary() keep their grouping keys intact while any downstream table
# can still group by them to reproduce the diagnostics.
LABEL_KEYS = ('init_seed', 'partition_sha256', 'test_basis', 'oos_status',
              'use_ckks', 'ckks_available', 'fuzziness', 'lambda_space', 'note',
              'stop_reason', 'rounds_run', 'boundary_mode', 'init_method',
              'd_emb', 'enforce_topology', 'use_anchors', 'kernel_scale_mode',
              'test_embedding_mode', 'preprocess')

QUALITY_KEYS = ('acc', 'nmi', 'ari', 'silhouette', 'davies_bouldin', 'calinski_harabasz')
ALL_FIT_KEYS = tuple('all_fit_' + key for key in QUALITY_KEYS)
TEST_PROXY_KEYS = tuple('test_proxy_' + key for key in QUALITY_KEYS)

COMM_KEYS = ('total_bytes', 'uplink_bytes', 'downlink_bytes', 'setup_bytes',
             'bytes_per_round', 'interservice_bytes')
TIME_KEYS = ('wall_seconds', 'embedding_seconds', 'statistics_seconds',
             'encryption_seconds', 'aggregation_seconds', 'decryption_seconds',
             'topology_seconds', 'setup_seconds', 'evaluation_seconds',
             'privacy_seconds')
PRIVACY_KEYS = ('epsilon', 'delta', 'mia_valid', 'mia_acc', 'mia_balanced_accuracy',
                'mia_auc', 'mia_tpr', 'mia_tnr', 'mia_cohens_d',
                'mia_degenerate_folds', 'ciphertext_acc', 'ciphertext_auc',
                'ciphertext_advantage', 'ciphertext_degenerate_folds',
                'public_context_has_secret_key', 'upload_size_correlation',
                'local_time_size_correlation')

RESULT_COLUMNS = (list(GROUP_KEYS) + list(LABEL_KEYS) + ['rounds']
                  + list(QUALITY_KEYS) + list(ALL_FIT_KEYS) + list(TEST_PROXY_KEYS)
                  + list(COMM_KEYS) + list(TIME_KEYS) + list(PRIVACY_KEYS))

# Metrics that must be produced for every method; used by the smoke assertions.
REQUIRED_METRICS = QUALITY_KEYS + ALL_FIT_KEYS + TEST_PROXY_KEYS


def empty_row():
    """A row with every declared column present and NaN-filled."""
    return {column: np.nan for column in RESULT_COLUMNS}


def as_float(value):
    """float() that survives None / NaN / arrays without raising."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return result


# ---------------------------------------------------------------- partitions
def partition_name(dataset, data_seed, beta, clients):
    return '%s_seed%d_beta%g_clients%d' % (dataset, int(data_seed), float(beta), int(clients))


def partition_path(dataset, data_seed, beta, clients):
    return os.path.join(PARTITIONS_DIR, partition_name(dataset, data_seed, beta, clients))


def split_name(dataset, data_seed):
    """Identity of a FIT/TEST split: dataset and split seed only."""
    return '%s_seed%d' % (dataset, int(data_seed))


def split_path(dataset, data_seed):
    return os.path.join(SPLITS_DIR, split_name(dataset, data_seed))


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


def write_partition(directory, meta, arrays):
    """Write meta + arrays so that any later run reads the exact same bytes."""
    os.makedirs(directory, exist_ok=True)
    payload = dict(meta)
    payload['arrays_sha256'] = {name: _digest(value) for name, value in arrays.items()}
    flattened = [list(map(int, indices)) for indices in meta['client_indices']]
    payload['partition_sha256'] = hashlib.sha256(
        json.dumps(flattened, sort_keys=True).encode('utf-8')).hexdigest()
    np.savez_compressed(os.path.join(directory, 'arrays.npz'), **arrays)
    with open(os.path.join(directory, 'partition.json'), 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=False)
    return payload


def load_partition(directory):
    """Return (meta, arrays) and verify the recorded array digests."""
    with open(os.path.join(directory, 'partition.json'), encoding='utf-8') as stream:
        meta = json.load(stream)
    with np.load(os.path.join(directory, 'arrays.npz')) as stored:
        arrays = {name: stored[name] for name in stored.files}
    for name, expected in meta.get('arrays_sha256', {}).items():
        if name in arrays and _digest(arrays[name]) != expected:
            raise ValueError('Frozen array %r in %s does not match its recorded sha256; '
                             'regenerate the partition instead of editing it' % (name, directory))
    return meta, arrays


# ------------------------------------------------------------------- results
def read_results(csv_path):
    import pandas as pd
    if not os.path.isfile(csv_path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(csv_path)
    return frame.reindex(columns=RESULT_COLUMNS) if len(frame.columns) else frame


def append_row(csv_path, row):
    """Append one run; the header is written only for a new file."""
    import pandas as pd
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    frame = pd.DataFrame([{column: row.get(column, np.nan) for column in RESULT_COLUMNS}],
                         columns=RESULT_COLUMNS)
    header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    frame.to_csv(csv_path, mode='a', header=header, index=False)
    return frame
