"""Run one (partition, method, init_seed) triple and append one row to results.csv.

Stage 2 of the pipeline. The partition is read from disk and never rebuilt here,
so every method in a comparison sees byte-identical FIT rows, the same client
ownership and the same K.

Column semantics (fixed by federated/ours/tracker.py):
    acc / nmi / ari / silhouette / ...   transductive FIT labels, all FIT rows
    all_fit_*                            the same FIT numbers, explicit name
    test_proxy_*                         offline TEST scoring; see test_basis
``test_basis`` records whether TEST really used the Nyström query extension or
fell back to the original-space nearest-center proxy -- without it two rows with
different ``test_proxy_*`` cannot be compared. Baselines set ``oos_status='N/A'``
because none of them defines an out-of-sample rule.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import schema  # noqa: E402
from experiments.baselines_adapter import BASELINE_METHODS, run_baseline  # noqa: E402
from federated.ours.aces_trainer import (  # noqa: E402
    federated_anchored_clustering, set_all_seeds)
from federated.ours.tracker import clustering_quality  # noqa: E402

DEFAULT_CFG = {
    'rounds': 10,
    'd_emb': 4,
    # fuzziness=1.1 / lambda_space=1.0 are the frozen paper config decided by the
    # accuracy ablation: m=2.0 collapses the memberships in a 64-d space (uniform
    # membership fixed point -> centers collapse onto the data mean), and at m=1.1
    # the pure embedding fusion is the strongest arm. See experiments/results/ablation.
    'fuzziness': 1.1,
    'lambda_space': 1.0,
    'graph_beta': 0.5,
    # None -> data-adaptive median landmark distance; pass a float to pin it.
    'kernel_sigma': None,
    'k_neighbors': 7,
    'anchor_weight': 3.0,
    'topology_epochs': 20,
    'init_method': 'kmeans_plusplus',
    'topology_boundary_mode': 'tanh',
    'enforce_topology': True,
    'use_anchors': True,
    'kernel_scale_mode': 'self_tuning',
    'test_embedding_mode': 'nystrom',
    'test_landmark_mode': 'stored',
    'use_ckks': False,
    'run_privacy_eval': False,
    'privacy_attack_folds': 5,
    'privacy_max_samples': 1000,
}


def tenseal_available():
    try:
        import tenseal  # noqa: F401
        return True
    except ImportError:
        return False


def _test_basis(prediction):
    """Map the trainer's honest prediction metadata onto one CSV cell."""
    if prediction.get('test_dual_space_used'):
        return 'nystrom'
    if prediction.get('test_embedding_mode') == 'nystrom':
        return 'original_space_proxy'
    return 'original_space'


def _round_substage(cost):
    stages = cost.get('round_substage_seconds', {}) if cost else {}
    return {name: schema.as_float(stages.get(name)) for name in schema.TIME_KEYS}


def _trace_payload(history):
    """Numeric per-round series of a tracker history, ready for np.savez."""
    payload = {}
    for key, values in (history or {}).items():
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
            continue
        try:
            payload[key] = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
    return payload


def _save_trace(path, payload, extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload, **{k: schema.as_float(v) for k, v in extra.items()})
    return path


def run_once(partition_dir, method, init_seed, cfg=None, study='quality'):
    """Execute one run and return a fully populated result row (no I/O).

    ``study`` labels the experiment the row belongs to. It is a grouping key, so
    a diagnostic run must NOT reuse 'quality': dropping a handful of extra rows
    into the main study would unbalance the D x S design that
    two_factor_summary() asserts on.
    """
    settings = dict(DEFAULT_CFG)
    settings.update(cfg or {})
    meta, arrays = schema.load_partition(partition_dir)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    X_test, y_test = arrays['X_test'], arrays['y_test']
    client_indices = [np.asarray(indices, dtype=int) for indices in meta['client_indices']]
    X_list = [X_train[indices] for indices in client_indices]
    num_clusters = int(meta['classes'])

    ckks_available = tenseal_available()
    use_ckks = bool(settings['use_ckks'])
    note = ''
    if use_ckks and not ckks_available:
        use_ckks, note = False, 'CKKS requested but tenseal is missing: ran plaintext'

    set_all_seeds(init_seed)
    started = time.perf_counter()
    trace = None
    history = None
    tracker = None
    if method == 'ours':
        centers, centers_emb, history, tracker = federated_anchored_clustering(
            X_list, num_clusters, settings['d_emb'],
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            num_rounds=settings['rounds'], fuzziness=settings['fuzziness'],
            lambda_space=settings['lambda_space'], beta=settings['graph_beta'],
            kernel_sigma=settings['kernel_sigma'], k_neighbors=settings['k_neighbors'],
            anchor_weight=settings['anchor_weight'], topology_epochs=settings['topology_epochs'],
            init_method=settings['init_method'], use_ckks=use_ckks,
            track_convergence=True, random_state=init_seed, train_indices=client_indices,
            enforce_topology=settings['enforce_topology'],
            topology_boundary_mode=settings['topology_boundary_mode'],
            use_anchors=settings['use_anchors'],
            kernel_scale_mode=settings['kernel_scale_mode'],
            test_embedding_mode=settings['test_embedding_mode'],
            test_landmark_mode=settings['test_landmark_mode'],
            run_privacy_eval=settings['run_privacy_eval'],
            privacy_attack_folds=settings['privacy_attack_folds'],
            privacy_max_samples=settings['privacy_max_samples'])
        train_labels, test_labels = history['train_labels'], history['test_labels']
        stop_reason = history.get('stopping_configuration', {}).get('reason', 'unknown')
        rounds = int(history['completed_rounds'])
        test_basis = _test_basis(history['prediction'])
        oos_status = 'available' if test_basis == 'nystrom' else 'proxy_fallback'
        communication = history.get('communication', {})
        cost = history.get('execution_cost', {})
        substage = _round_substage(cost)
        wall_seconds = schema.as_float(cost.get('trainer_seconds'))
        setup_seconds = schema.as_float(cost.get('setup_seconds'))
        evaluation_seconds = schema.as_float(cost.get('evaluation_seconds'))
        privacy_seconds = schema.as_float(cost.get('privacy_seconds'))
        privacy = history.get('privacy_results', {})
        if tracker is not None and history:
            trace = _trace_payload({k: v for k, v in tracker.history.items()})
    elif method in BASELINE_METHODS:
        result = run_baseline(
            method, X_list, X_train, y_train, X_test, y_test, num_clusters,
            num_rounds=settings['rounds'], random_state=init_seed,
            fuzziness=settings['fuzziness'], k_neighbors=settings['k_neighbors'])
        train_labels, test_labels = result['train_labels'], result['test_labels']
        rounds, test_basis, oos_status = result['rounds'], 'original_space', result['oos_status']
        note = result['note'] if not note else note + '; ' + result['note']
        stop_reason = 'n/a'
        communication, substage, privacy = {}, {}, {}
        wall_seconds = time.perf_counter() - started
        setup_seconds = evaluation_seconds = privacy_seconds = np.nan
    else:
        raise ValueError('Unknown method %r' % method)
    if not wall_seconds or not np.isfinite(wall_seconds):
        wall_seconds = time.perf_counter() - started

    fit_metrics = clustering_quality(X_train, y_train, train_labels)
    test_metrics = clustering_quality(X_test, y_test, test_labels)

    is_ours = method == 'ours'
    row = schema.empty_row()
    row.update({
        'dataset': meta['dataset'], 'method': method, 'data_seed': meta['data_seed'],
        'dirichlet_beta': meta['federated']['dirichlet_beta'],
        'clients': meta['clients'], 'clusters': num_clusters,
        'study': study, 'study_value': 1.0,
        'init_seed': int(init_seed), 'partition_sha256': meta['partition_sha256'],
        'test_basis': test_basis, 'oos_status': oos_status,
        'use_ckks': bool(use_ckks), 'ckks_available': ckks_available,
        'fuzziness': settings['fuzziness'], 'lambda_space': settings['lambda_space'],
        'rounds': rounds, 'wall_seconds': wall_seconds, 'note': note,
        'stop_reason': stop_reason, 'rounds_run': rounds,
        'boundary_mode': settings['topology_boundary_mode'] if is_ours else 'n/a',
        'init_method': settings['init_method'] if is_ours else 'n/a',
        'd_emb': settings['d_emb'] if is_ours else np.nan,
        'enforce_topology': settings['enforce_topology'] if is_ours else 'n/a',
        'use_anchors': settings['use_anchors'] if is_ours else 'n/a',
        'kernel_scale_mode': settings['kernel_scale_mode'] if is_ours else 'n/a',
        'test_embedding_mode': settings['test_embedding_mode'] if is_ours else 'n/a',
        'preprocess': (meta.get('preprocess') or {}).get('name', 'standardized'),
    })
    for key in schema.QUALITY_KEYS:
        row[key] = schema.as_float(fit_metrics[key])
        row['all_fit_' + key] = schema.as_float(fit_metrics[key])
        row['test_proxy_' + key] = schema.as_float(test_metrics[key])
    for key in schema.COMM_KEYS:
        row[key] = schema.as_float(communication.get(key))
    row['setup_seconds'] = setup_seconds
    row['evaluation_seconds'] = evaluation_seconds
    row['privacy_seconds'] = privacy_seconds
    for key in schema.TIME_KEYS:
        if key in substage and not np.isfinite(row.get(key, np.nan)):
            row[key] = substage[key]

    mia = privacy.get('mia', {}) or {}
    ciphertext = privacy.get('ciphertext', {}) or {}
    role = privacy.get('role_check', {}) or {}
    leakage = privacy.get('size_leakage', {}) or {}
    row['mia_valid'] = mia.get('status') == 'evaluated'
    row['mia_acc'] = schema.as_float(mia.get('acc'))
    row['mia_balanced_accuracy'] = schema.as_float(mia.get('balanced_accuracy'))
    row['mia_auc'] = schema.as_float(mia.get('auc'))
    row['mia_tpr'] = schema.as_float(mia.get('tpr'))
    row['mia_tnr'] = schema.as_float(mia.get('tnr'))
    row['mia_cohens_d'] = schema.as_float(mia.get('cohens_d'))
    row['mia_degenerate_folds'] = schema.as_float(mia.get('degenerate_folds'))
    row['ciphertext_acc'] = schema.as_float(ciphertext.get('acc'))
    row['ciphertext_auc'] = schema.as_float(ciphertext.get('auc'))
    row['ciphertext_advantage'] = schema.as_float(ciphertext.get('advantage'))
    row['ciphertext_degenerate_folds'] = schema.as_float(ciphertext.get('degenerate_folds'))
    row['public_context_has_secret_key'] = role.get('public_context_has_secret_key')
    row['upload_size_correlation'] = schema.as_float(leakage.get('upload_size_correlation'))
    row['local_time_size_correlation'] = schema.as_float(leakage.get('local_time_size_correlation'))
    # dp_sigma is an uncalibrated perturbation, not formal DP: epsilon/delta stay NaN.
    row['epsilon'], row['delta'] = np.nan, np.nan
    return row, trace, (tracker.communication if tracker is not None else {})


def trace_path(partition_dir, method, init_seed):
    name = os.path.basename(os.path.normpath(partition_dir))
    return os.path.join(schema.TRACES_DIR, name, '%s_init%d.npz' % (method, int(init_seed)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--partition', required=True,
                        help='partition directory, or a name under experiments/partitions')
    parser.add_argument('--method', default='ours',
                        help='ours | ' + ' | '.join(BASELINE_METHODS))
    parser.add_argument('--init-seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=DEFAULT_CFG['rounds'])
    parser.add_argument('--d-emb', type=int, default=DEFAULT_CFG['d_emb'])
    parser.add_argument('--fuzziness', type=float, default=DEFAULT_CFG['fuzziness'])
    parser.add_argument('--lambda-space', type=float, default=DEFAULT_CFG['lambda_space'])
    parser.add_argument('--kernel-sigma', type=float, default=DEFAULT_CFG['kernel_sigma'],
                        help='Gaussian kernel sigma; the default (None) is the '
                             'data-adaptive median landmark distance')
    parser.add_argument('--test-embedding-mode', default=DEFAULT_CFG['test_embedding_mode'],
                        choices=['nystrom', 'original_space'])
    parser.add_argument('--test-landmark-mode', default=DEFAULT_CFG['test_landmark_mode'],
                        choices=['strict', 'stored'])
    parser.add_argument('--study', default='quality',
                        help="row label; use anything but 'quality' for a diagnostic "
                             "run so the main D x S design stays balanced")
    parser.add_argument('--use-ckks', action='store_true')
    parser.add_argument('--run-privacy-eval', action='store_true')
    parser.add_argument('--results', default=os.path.join(schema.RESULTS_DIR, 'results.csv'))
    parser.add_argument('--traces', action='store_true', default=True,
                        help='save per-round series for the cross-seed bands')
    args = parser.parse_args(argv)

    directory = (args.partition if os.path.isdir(args.partition)
                 else schema.partition_path(*_parse_name(args.partition)))
    cfg = {'rounds': args.rounds, 'd_emb': args.d_emb, 'fuzziness': args.fuzziness,
           'lambda_space': args.lambda_space, 'kernel_sigma': args.kernel_sigma,
           'use_ckks': args.use_ckks, 'run_privacy_eval': args.run_privacy_eval,
           'test_embedding_mode': args.test_embedding_mode,
           'test_landmark_mode': args.test_landmark_mode}
    row, trace, communication = run_once(directory, args.method, args.init_seed, cfg,
                                         study=args.study)
    if args.traces and trace:
        _save_trace(trace_path(directory, args.method, args.init_seed), trace,
                    {'total_bytes': communication.get('total_bytes', np.nan)})
    schema.append_row(args.results, row)
    print('%s | %s init=%d | acc=%.4f nmi=%.4f ari=%.4f test=%s(%s) rounds=%s -> %s'
          % (row['dataset'], row['method'], row['init_seed'], row['acc'], row['nmi'],
             row['ari'], row['test_proxy_acc'], row['test_basis'], row['rounds'],
             args.results))
    return 0


def _parse_name(name):
    """'digits_seed0_beta0.5_clients5' -> (dataset, seed, beta, clients)."""
    parts = name.split('_')
    if len(parts) != 4:
        raise ValueError('Partition name %r must look like '
                         'dataset_seed0_beta0.5_clients5' % name)
    return (parts[0], int(parts[1][len('seed'):]), float(parts[2][len('beta'):]),
            int(parts[3][len('clients'):]))


if __name__ == '__main__':
    raise SystemExit(main())
