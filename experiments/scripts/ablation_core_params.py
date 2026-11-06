"""Three-parameter ablation for `ours`: d_emb x lambda_space x boundary_mode.

Five combinations, one factor added at a time so the marginal effect of each is
readable (and `all-three` shows whether they compound):

    1 baseline   d_emb=4  lambda_space=1.0  boundary=tanh   (= current DEFAULT_CFG)
    2 +d_emb     d_emb=8  lambda_space=1.0  boundary=tanh
    3 +fusion    d_emb=4  lambda_space=0.7  boundary=tanh
    4 +boundary  d_emb=4  lambda_space=1.0  boundary=clamp
    5 all-three  d_emb=8  lambda_space=0.7  boundary=clamp

fuzziness stays at 1.1 for every arm: the accuracy ablation and the 10-client
grid both showed m >= 1.5 collapses to a uniform-membership fixed point, so m is
not a free parameter here, it is a precondition.

The design is 5 data_seed x 2 init_seed x 5 configs = 50 runs on 10 clients, and
rows carry the full GROUP_KEYS so summarize_results() / two_factor_summary()
work on the appended table unchanged.

Run:
  python \
      experiments/scripts/ablation_core_params.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import schema  # noqa: E402
from experiments.scripts.run_single import DEFAULT_CFG  # noqa: E402
from federated.ours.aces_trainer import (  # noqa: E402
    federated_anchored_clustering, set_all_seeds)
from federated.ours.tracker import clustering_quality, summarize_results, two_factor_summary  # noqa: E402

# One factor at a time against the configuration the 10-client grid confirmed.
CONFIGS = (
    ('baseline', dict(d_emb=4, lambda_space=1.0, topology_boundary_mode='tanh')),
    ('+d_emb', dict(d_emb=8, lambda_space=1.0, topology_boundary_mode='tanh')),
    ('+fusion', dict(d_emb=4, lambda_space=0.7, topology_boundary_mode='tanh')),
    ('+boundary', dict(d_emb=4, lambda_space=1.0, topology_boundary_mode='clamp')),
    ('all-three', dict(d_emb=8, lambda_space=0.7, topology_boundary_mode='clamp')),
)

STUDY = 'core_ablation'

COLUMNS = (
    ['dataset', 'method', 'data_seed', 'dirichlet_beta', 'clients', 'clusters',
     'study', 'study_value', 'arm']
    + ['init_seed', 'fuzziness', 'lambda_space', 'd_emb', 'boundary_mode',
       'init_method', 'enforce_topology', 'use_anchors', 'kernel_scale_mode',
       'test_embedding_mode', 'stop_reason', 'rounds_run']
    + ['acc', 'nmi', 'ari', 'test_proxy_acc', 'test_proxy_nmi', 'test_proxy_ari',
       'rounds', 'wall_seconds']
    + ['n_distinct_labels', 'center_spread_ratio',
       'final_min_distance_orig', 'final_center_shift_orig'])


def run_arm(directory, arm_name, arm_cfg, init_seed, fuzziness, median_distance):
    """One ours run under one arm; returns the row dict."""
    meta, arrays = schema.load_partition(directory)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    X_test, y_test = arrays['X_test'], arrays['y_test']
    client_indices = [np.asarray(i, dtype=int) for i in meta['client_indices']]
    X_list = [X_train[i] for i in client_indices]
    K = int(meta['classes'])

    set_all_seeds(init_seed)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        centers, _centers_emb, history, tracker = federated_anchored_clustering(
            X_list, K, arm_cfg['d_emb'], X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, num_rounds=DEFAULT_CFG['rounds'],
            fuzziness=fuzziness, lambda_space=arm_cfg['lambda_space'],
            beta=DEFAULT_CFG['graph_beta'], kernel_sigma=DEFAULT_CFG['kernel_sigma'],
            k_neighbors=DEFAULT_CFG['k_neighbors'],
            anchor_weight=DEFAULT_CFG['anchor_weight'],
            topology_epochs=DEFAULT_CFG['topology_epochs'],
            init_method=DEFAULT_CFG['init_method'], use_ckks=False,
            track_convergence=True, random_state=init_seed,
            train_indices=client_indices,
            enforce_topology=DEFAULT_CFG['enforce_topology'],
            topology_boundary_mode=arm_cfg['topology_boundary_mode'],
            use_anchors=DEFAULT_CFG['use_anchors'],
            kernel_scale_mode=DEFAULT_CFG['kernel_scale_mode'],
            test_embedding_mode=DEFAULT_CFG['test_embedding_mode'],
            test_landmark_mode=DEFAULT_CFG['test_landmark_mode'],
            run_privacy_eval=False)
    wall = time.perf_counter() - started

    fit = clustering_quality(X_train, y_train, history['train_labels'])
    test = clustering_quality(X_test, y_test, history['test_labels'])
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    spread = float(pairwise[np.triu_indices(len(centers), 1)].mean())
    history_dict = tracker.history
    return {
        'dataset': meta['dataset'], 'method': 'ours',
        'data_seed': int(meta['data_seed']),
        'dirichlet_beta': float(meta['federated']['dirichlet_beta']),
        'clients': int(meta['clients']), 'clusters': int(K),
        'study': STUDY, 'study_value': float(arm_index(arm_name)), 'arm': arm_name,
        'init_seed': int(init_seed), 'fuzziness': fuzziness,
        'lambda_space': arm_cfg['lambda_space'], 'd_emb': arm_cfg['d_emb'],
        'boundary_mode': arm_cfg['topology_boundary_mode'],
        'init_method': DEFAULT_CFG['init_method'],
        'enforce_topology': DEFAULT_CFG['enforce_topology'],
        'use_anchors': DEFAULT_CFG['use_anchors'],
        'kernel_scale_mode': DEFAULT_CFG['kernel_scale_mode'],
        'test_embedding_mode': DEFAULT_CFG['test_embedding_mode'],
        'stop_reason': history.get('stopping_configuration', {}).get('reason'),
        'rounds_run': int(history['completed_rounds']),
        'acc': fit['acc'], 'nmi': fit['nmi'], 'ari': fit['ari'],
        'test_proxy_acc': test['acc'], 'test_proxy_nmi': test['nmi'],
        'test_proxy_ari': test['ari'],
        'rounds': int(history['completed_rounds']), 'wall_seconds': wall,
        'n_distinct_labels': int(len(np.unique(history['train_labels']))),
        'center_spread_ratio': float(spread / max(median_distance, 1e-12)),
        'final_min_distance_orig': float(history_dict['min_distances_orig'][-1])
        if history_dict.get('min_distances_orig') else np.nan,
        'final_center_shift_orig': float(history_dict['center_shift_orig'][-1])
        if history_dict.get('center_shift_orig') else np.nan,
    }


def arm_index(name):
    for index, (arm_name, _cfg) in enumerate(CONFIGS, start=1):
        if arm_name == name:
            return index
    raise ValueError('unknown arm %r' % name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--data-seeds', default='0,1,2,3,4')
    parser.add_argument('--init-seeds', default='0,1')
    parser.add_argument('--fuzziness', type=float, default=None,
                        help='default: DEFAULT_CFG fuzziness (1.1)')
    parser.add_argument('--out', default=os.path.join(schema.RESULTS_DIR, 'ablation',
                                                      'ablation.csv'))
    args = parser.parse_args(argv)

    fuzziness = DEFAULT_CFG['fuzziness'] if args.fuzziness is None else args.fuzziness
    data_seeds = [int(v) for v in args.data_seeds.split(',') if v.strip()]
    init_seeds = [int(v) for v in args.init_seeds.split(',') if v.strip()]
    total = len(CONFIGS) * len(data_seeds) * len(init_seeds)
    print('arms=%d x data_seed=%d x init_seed=%d = %d runs | %d clients, beta=%g, '
          'fuzziness=%.1f'
          % (len(CONFIGS), len(data_seeds), len(init_seeds), total, args.clients,
             args.beta, fuzziness))

    rows = []
    done = 0
    started_all = time.perf_counter()
    for arm_name, arm_cfg in CONFIGS:
        for data_seed in data_seeds:
            directory = schema.partition_path(args.dataset, data_seed, args.beta,
                                              args.clients)
            _, arrays = schema.load_partition(directory)
            median_distance = float(np.median(pdist(
                np.asarray(arrays['X_train'], dtype=float))))
            for init_seed in init_seeds:
                row = run_arm(directory, arm_name, arm_cfg, init_seed, fuzziness,
                              median_distance)
                rows.append(row)
                done += 1
                print('  [%2d/%2d] %-10s seed=%d init=%d | acc=%.4f nmi=%.4f | '
                      'TEST=%.4f | spread=%.3f'
                      % (done, total, arm_name, data_seed, init_seed, row['acc'],
                         row['nmi'], row['test_proxy_acc'],
                         row['center_spread_ratio']))
    print('runs finished in %.1f s' % (time.perf_counter() - started_all))

    frame = pd.DataFrame(rows, columns=COLUMNS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.isfile(args.out):
        previous = pd.read_csv(args.out)
        frame = pd.concat([previous, frame], ignore_index=True, sort=False)
    frame.to_csv(args.out, index=False)
    print('appended %d rows -> %s (%d rows total)' % (len(rows), args.out, len(frame)))

    fresh = frame[frame.study == STUDY]
    fresh = fresh[fresh.data_seed.isin(data_seeds)]

    print('\n--- summarize_results (per split, n = %d init seeds) ---' % len(init_seeds))
    summary = summarize_results(fresh)
    if summary.empty:
        print('summary is empty -- check the group keys')
    else:
        acc_summary = summary[summary.metric == 'acc']
        print(acc_summary.groupby('study_value')['mean'].agg(['mean', 'std'])
              .round(4).to_string())

    print('\n--- ranking: mean over all %d runs per arm ---'
          % (len(data_seeds) * len(init_seeds)))
    ranking = (fresh.groupby('arm')[['acc', 'nmi', 'ari', 'test_proxy_acc',
                                     'test_proxy_nmi', 'test_proxy_ari']]
               .agg(['mean', 'std']).round(4))
    ranking = ranking.reindex([name for name, _ in CONFIGS])
    print(ranking.to_string())

    cross = two_factor_summary(fresh)
    if not cross.empty:
        print('\n--- two_factor (D=%d splits x S=%d seeds, authoritative CI) ---'
              % (len(data_seeds), len(init_seeds)))
        acc_cross = cross[cross.metric == 'acc'].copy()
        acc_cross['arm'] = [CONFIGS[int(v) - 1][0] for v in acc_cross['study_value']]
        print(acc_cross[['arm', 'mean', 'sd_split', 'sd_init', 'ci95_low',
                         'ci95_high']].round(4).to_string(index=False))

    base = fresh[fresh.arm == 'baseline']['acc'].mean()
    best_row = fresh.groupby('arm')['acc'].mean().idxmax()
    best = fresh[fresh.arm == best_row]['acc'].mean()
    gain = best - base
    print('\nbaseline acc = %.4f | best arm = %s acc = %.4f | gain = %+.4f (%+.2f%%)'
          % (base, best_row, best, gain, 100.0 * gain))
    if gain > 0.01:
        print('DECISION: gain > 1%% -> fold %s into DEFAULT_CFG and re-run the '
              'full grid.' % best_row)
    elif gain < 0.005:
        print('DECISION: gain < 0.5%% -> keep DEFAULT_CFG, skip the P3 change, '
              'the current numbers are good enough.')
    else:
        print('DECISION: gain in [0.5%%, 1%%] -> borderline, no automatic change.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
