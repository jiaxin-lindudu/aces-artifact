"""Sweep (data_seed x init_seed x method x dirichlet_beta) into results.csv.

Stage 2 driver. One row per run, append-only, then the two summaries that
federated/ours/tracker.py defines:

    summary.csv     summarize_results  -- mean/SD/t-CI across INIT seeds on one
                                        frozen split; answers "how much does the
                                        initialization matter here?"
    two_factor.csv  two_factor_summary -- split as the sampling unit; the
                                        AUTHORITATIVE interval whenever more than
                                        one data_seed is present.

The per-split CI must never be quoted as a method-level interval, which is why
both files are always produced together.
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
from experiments.scripts import make_partition, run_single  # noqa: E402
from federated.ours.tracker import summarize_results, two_factor_summary  # noqa: E402


def split_list(text, cast=str):
    return [cast(item.strip()) for item in str(text).split(',') if item.strip()]


def ensure_partition(dataset, data_seed, beta, clients, features):
    directory = schema.partition_path(dataset, data_seed, beta, clients)
    if os.path.isfile(os.path.join(directory, 'partition.json')):
        return directory
    return make_partition.build_partition(dataset, data_seed, beta, clients,
                                          features=features)[0]


def sweep(datasets, betas, data_seeds, init_seeds, methods, cfg, results_path,
          clients=10, features=64, fresh=False, max_retries=3):
    """Run the full grid, appending every row; returns the reloaded frame.

    A (split, init_seed) cell either succeeds and appends exactly one row, or the
    whole sweep raises: a failed cell is retried ``max_retries`` times and, if it
    still fails, the run aborts instead of leaving a half-written, unbalanced
    design that two_factor_summary() would refuse to summarize.
    """
    if fresh and os.path.isfile(results_path):
        os.remove(results_path)
    for dataset in datasets:
        for beta in betas:
            for data_seed in data_seeds:
                directory = ensure_partition(dataset, data_seed, beta, clients, features)
                for method in methods:
                    for init_seed in init_seeds:
                        row = trace = communication = None
                        last_exc = None
                        for attempt in range(max_retries + 1):
                            try:
                                row, trace, communication = run_single.run_once(
                                    directory, method, init_seed, cfg)
                                break
                            except Exception as exc:  # noqa: BLE001 - retry, then abort
                                last_exc = exc
                                print('  retry %-5s init=%-2d (attempt %d/%d): %s: %s'
                                      % (method, init_seed, attempt + 1, max_retries,
                                         type(exc).__name__, exc))
                        if row is None:
                            raise RuntimeError(
                                'aborting: %s %s init=%d failed after %d attempts'
                                % (dataset, method, init_seed, max_retries + 1)) from last_exc
                        if trace:
                            run_single._save_trace(
                                run_single.trace_path(directory, method, init_seed), trace,
                                {'total_bytes': communication.get('total_bytes', np.nan)})
                        schema.append_row(results_path, row)
                        print('%-6s %-5s beta=%-4g data_seed=%-2d init=%-2d | '
                              'acc=%.4f nmi=%.4f ari=%.4f test=%.4f (%s)'
                              % (dataset, method, beta, data_seed, init_seed,
                                 row['acc'], row['nmi'], row['ari'],
                                 row['test_proxy_acc'], row['test_basis']))
    return schema.read_results(results_path)


def write_tables(frame, out_dir):
    """Split the primary transductive table from the Nyström appendix table.

    The two answer different questions and must not be merged: ``all_fit_*`` is
    the in-protocol transductive result, ``test_proxy_*`` is offline evaluation
    whose basis (Nyström or original-space fallback) is recorded per row.
    """
    keys = list(schema.GROUP_KEYS) + ['init_seed']
    main = frame[keys + list(schema.ALL_FIT_KEYS)].copy()
    main.to_csv(os.path.join(out_dir, 'main_transductive.csv'), index=False)
    appendix = frame[keys + ['test_basis', 'oos_status'] + list(schema.TEST_PROXY_KEYS)].copy()
    appendix.to_csv(os.path.join(out_dir, 'appendix_nystrom.csv'), index=False)
    return main, appendix


def write_summaries(frame, out_dir, verbose=True):
    """Write summary.csv, two_factor.csv and the two result tables."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    main, appendix = write_tables(frame, out_dir)
    summary = summarize_results(frame)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    cross = pd.DataFrame()
    # Group the same way two_factor_summary does, i.e. WITHOUT data_seed.
    grouping = [key for key in schema.GROUP_KEYS if key != 'data_seed']
    splits_per_group = frame.groupby(grouping, dropna=False)['data_seed'].nunique()
    countable = int((splits_per_group >= 2).sum())
    if countable:
        cross = two_factor_summary(frame)
        cross.to_csv(os.path.join(out_dir, 'two_factor.csv'), index=False)
    elif verbose:
        print('two_factor.csv not written: every group has a single data split '
              '(sigma_split is not estimable and no method-level CI exists)')
    if verbose:
        print('summary rows=%d | two_factor rows=%d | out=%s'
              % (len(summary), len(cross), out_dir))
    return summary, cross


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', default='digits')
    parser.add_argument('--betas', default='0.1,0.5,1.0')
    parser.add_argument('--data-seeds', default='0,1,2,3,4')
    parser.add_argument('--init-seeds', default='0,1,2,3,4')
    parser.add_argument('--methods', default='ours,fkm,ffcm,nnfc')
    parser.add_argument('--clients', type=int, default=10,
                        help='every experiment in this checkout runs at 10 clients')
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=run_single.DEFAULT_CFG['rounds'])
    parser.add_argument('--d-emb', type=int, default=run_single.DEFAULT_CFG['d_emb'])
    parser.add_argument('--fuzziness', type=float, default=run_single.DEFAULT_CFG['fuzziness'])
    parser.add_argument('--lambda-space', type=float,
                        default=run_single.DEFAULT_CFG['lambda_space'])
    parser.add_argument('--use-ckks', action='store_true')
    parser.add_argument('--run-privacy-eval', action='store_true')
    parser.add_argument('--results', default=os.path.join(schema.RESULTS_DIR, 'results.csv'))
    parser.add_argument('--out-dir', default=schema.RESULTS_DIR)
    parser.add_argument('--fresh', action='store_true', help='drop an existing results.csv first')
    parser.add_argument('--figures', action='store_true', help='also render the aggregate figures')
    args = parser.parse_args(argv)

    cfg = {'rounds': args.rounds, 'd_emb': args.d_emb, 'fuzziness': args.fuzziness,
           'lambda_space': args.lambda_space, 'use_ckks': args.use_ckks,
           'run_privacy_eval': args.run_privacy_eval}
    frame = sweep(split_list(args.datasets), split_list(args.betas, float),
                  split_list(args.data_seeds, int), split_list(args.init_seeds, int),
                  split_list(args.methods), cfg, args.results,
                  clients=args.clients, features=args.features, fresh=args.fresh)
    write_summaries(frame, args.out_dir)
    if args.figures:
        from experiments.scripts import plot_aggregate
        plot_aggregate.render_all(args.results, split_list(args.datasets)[0])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
