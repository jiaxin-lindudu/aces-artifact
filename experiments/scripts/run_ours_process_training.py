"""Fixed-parameter full training in native and three-process CKKS execution.

C owns all logical clients and offline evaluation; A owns prototype updates;
D owns the secret key. Every training round uses authenticated local sockets.
Only raw/mean CSVs are saved. This is a single-host prototype, not a security
proof or a claim that released aggregates protect output membership.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import queue
import secrets
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import grid_search_ours as ours
from experiments.scripts.compare_grid_results import select_complete
from experiments.scripts.run_ckks_role_validation import send_message, receive_message

SCORES = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']


def decryptor(job, reports, authkey):
    import tenseal as ts
    tick = time.perf_counter()
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2 ** 40
    public = context.serialize(save_public_key=True, save_secret_key=False,
                               save_galois_keys=False, save_relin_keys=False)
    setup = time.perf_counter() - tick
    length = job['K'] * (job['d_orig'] + job['config']['d_emb'] + 1)
    expected_chunks = (length + 4095) // 4096
    rounds = rejected = sent = probe_reply_bytes = 0
    seconds = 0.
    with Listener(('127.0.0.1', 0), authkey=authkey) as listener:
        reports.put(dict(kind='ready', role='D', address=listener.address))
        with listener.accept() as connection:
            if receive_message(connection) != dict(op='hello', role='A'):
                raise ValueError('Expected aggregator handshake')
            sent += send_message(connection, dict(public_context=public))
            while True:
                message = receive_message(connection)
                if message['op'] == 'close':
                    break
                if (message['op'] != 'aggregate' or message.get('round_id') != rounds
                        or message.get('active_clients') != list(range(job['clients']))
                        or len(message.get('chunks', [])) != expected_chunks):
                    rejected += 1
                    count = send_message(connection, dict(status='rejected'))
                    sent += count; probe_reply_bytes += count
                    continue
                tick = time.perf_counter()
                values = np.concatenate([ts.ckks_vector_from(context, chunk).decrypt()
                                         for chunk in message['chunks']])[:length]
                seconds += time.perf_counter() - tick
                sent += send_message(connection, dict(status='aggregate', round_id=rounds, values=values))
                rounds += 1
    return dict(D_pid=os.getpid(), D_has_secret_key=bool(context.has_secret_key()),
        D_setup_seconds=setup, D_decryption_seconds=seconds, D_sent_bytes=sent,
        D_decryption_batches=rounds, D_rejected_requests=rejected,
        D_probe_reply_bytes=probe_reply_bytes,
        public_context_bytes=len(public))


def aggregator(job, reports, keys, address):
    import tenseal as ts
    from federated.ours.aces_clustering import AnchoredSpectralServer
    from federated.ours.aces_trainer import set_all_seeds, relative_center_shift
    set_all_seeds(job['init_seed'])
    cfg = job['config']; K, d, p = job['K'], job['d_orig'], cfg['d_emb']
    batches = (K * (d + p + 1) + 4095) // 4096
    sent_d = sent_c = 0
    with Client(address, authkey=keys['AD']) as D:
        sent_d += send_message(D, dict(op='hello', role='A'))
        public = receive_message(D)['public_context']
        tick = time.perf_counter()
        context = ts.context_from(public)
        if context.has_secret_key():
            raise ValueError('Aggregator must receive public context only')
        server = AnchoredSpectralServer(K, d, p, use_ckks=False)
        cx, ch = server.initialize_centers(cfg['init_method'], None, job['init_seed'])
        setup = time.perf_counter() - tick
        stable = 0
        round_times, shifts, crypto_times, topology_times = [], [], [], []
        stop_reason = 'round_limit'
        with Listener(('127.0.0.1', 0), authkey=keys['CA']) as listener:
            reports.put(dict(kind='ready', role='A', address=listener.address))
            with listener.accept() as C:
                sent_c += send_message(C, dict(op='setup', public_context=public))
                if receive_message(C) != dict(op='ready'):
                    raise ValueError('Client setup did not complete')
                for round_id in range(cfg['num_rounds']):
                    tick_round = time.perf_counter()
                    previous_x, previous_h = cx.copy(), ch.copy()
                    sent_c += send_message(C, dict(op='round', round_id=round_id,
                                                   centers_orig=cx, centers_emb=ch))
                    sums = None; alignment = []; received = set()
                    tick_crypto = 0.
                    for _ in range(job['clients']):
                        message = receive_message(C)
                        cid = message['client_id']
                        if (message['round_id'] != round_id or cid in received
                                or cid not in range(job['clients']) or len(message['chunks']) != batches):
                            raise ValueError('Unexpected client contribution')
                        received.add(cid)
                        tick = time.perf_counter()
                        vectors = [ts.ckks_vector_from(context, item) for item in message['chunks']]
                        sums = vectors if sums is None else [a + b for a, b in zip(sums, vectors)]
                        tick_crypto += time.perf_counter() - tick
                        if message['alignment_error'] is not None:
                            alignment.append(message['alignment_error'])
                    tick = time.perf_counter()
                    chunks = [item.serialize() for item in sums]
                    tick_crypto += time.perf_counter() - tick
                    request = dict(op='aggregate', round_id=round_id,
                                   active_clients=sorted(received), chunks=chunks)
                    sent_d += send_message(D, request)
                    decoded = receive_message(D)
                    if decoded['status'] != 'aggregate' or decoded['round_id'] != round_id:
                        raise ValueError('Authorized aggregate was rejected')
                    vector = decoded['values']; first, second = K * d, K * (d + p)
                    cx, ch = server.aggregate_statistics([(vector[:first].reshape(K, d),
                        vector[first:second].reshape(K, p), vector[second:],
                        dict(alignment_error=float(np.mean(alignment)) if alignment else None))])
                    shift_orig = relative_center_shift(cx, previous_x)
                    shift = max(shift_orig, relative_center_shift(ch, previous_h))
                    tick = time.perf_counter()
                    if cfg['enforce_topology']:
                        cx, ch = server.enforce_topology_consistency(
                            num_epochs=cfg['topology_epochs'], learning_rate=cfg['learning_rate'],
                            random_state=job['init_seed'], fidelity_weight=cfg['topology_fidelity_weight'],
                            topology_weight=cfg['topology_topology_weight'],
                            separation_weight=cfg['topology_separation_weight'],
                            boundary_mode=cfg['topology_boundary_mode'],
                            max_displacement=cfg['topology_max_displacement'],
                            early_stopping=cfg['topology_early_stopping'],
                            early_stopping_patience=cfg['topology_early_stopping_patience'])
                    topology_times.append(time.perf_counter() - tick)
                    round_times.append(time.perf_counter() - tick_round)
                    crypto_times.append(tick_crypto); shifts.append(shift)
                    stable = stable + 1 if shift <= cfg['tol_center_shift'] else 0
                    if stable >= cfg['early_stopping_patience']:
                        stop_reason = 'aggregation_relative_center_shift_patience'
                        break
                sent_c += send_message(C, dict(op='finish', centers_orig=cx, centers_emb=ch,
                    previous_orig=previous_x, previous_emb=previous_h))
        # Two policy probes after training; not included in round core time.
        probe_request_bytes = send_message(D, request)
        sent_d += probe_request_bytes
        replay = receive_message(D)['status'] == 'rejected'
        count = send_message(D, dict(op='decrypt_individual'))
        sent_d += count; probe_request_bytes += count
        unsupported = receive_message(D)['status'] == 'rejected'
        sent_d += send_message(D, dict(op='close'))
    return dict(A_pid=os.getpid(), A_has_secret_key=bool(context.has_secret_key()),
        A_setup_seconds=setup, A_sent_D_bytes=sent_d, A_sent_C_bytes=sent_c,
        A_probe_request_bytes=probe_request_bytes,
        rounds_run=len(round_times), stop_reason=stop_reason,
        final_center_shift_orig=shift_orig, A_round_seconds=sum(round_times),
        aggregation_seconds=sum(crypto_times), topology_seconds=sum(topology_times),
        round_times_trace=json.dumps(round_times), aggregation_max_shifts_trace=json.dumps(shifts),
        replay_rejected=replay, unsupported_operation_rejected=unsupported,
        centers_orig=cx, centers_emb=ch)


def clients(job, address, authkey):
    import tenseal as ts
    from federated.ours.aces_clustering import AnchoredSpectralClient
    from federated.ours.aces_trainer import set_all_seeds, _fit_memberships, _test_labels
    set_all_seeds(job['init_seed'])
    bundle = job['synthetic_bundle'] if job.get('synthetic') else ours.load_partition_bundle(
        job['dataset'], job['data_seed'], job['beta'], job['clients'], job['data_dir'])
    cfg = job['config']; sent = 0; local_seconds = crypto_seconds = 0.
    with Client(address, authkey=authkey) as A:
        public = receive_message(A)['public_context']
        tick = time.perf_counter()
        context = ts.context_from(public)
        if context.has_secret_key():
            raise ValueError('Clients must receive public context only')
        local_clients = [AnchoredSpectralClient(id=i, X=X, anchor_weight=cfg['anchor_weight'],
            beta=cfg['lambda_space'], dp_sigma=0., graph_method=cfg['graph_method'],
            k_neighbors=cfg['k_neighbors'], kernel_sigma=cfg['kernel_sigma'], use_ckks=False,
            use_anchors=cfg['use_anchors'], normalize_spaces=cfg['normalize_spaces'],
            kernel_scale_mode=cfg['kernel_scale_mode'], match_procrustes_scale=cfg['match_procrustes_scale'],
            random_state=job['init_seed'], eigen_solver=job['eigen_solver'])
            for i, X in enumerate(bundle['X_list'])]
        setup = time.perf_counter() - tick
        sent += send_message(A, dict(op='ready'))
        rounds = 0
        while True:
            message = receive_message(A)
            if message['op'] == 'finish':
                final = message
                break
            if message['op'] != 'round' or message['round_id'] != rounds:
                raise ValueError('Unexpected prototype round')
            for client in local_clients:
                tick = time.perf_counter()
                sx, sh, masses, info = client.compute_dual_statistics(
                    message['centers_orig'], message['centers_emb'], cfg['fuzziness'], cfg['use_procrustes'])
                local_seconds += time.perf_counter() - tick
                vector = np.r_[sx.ravel(), sh.ravel(), masses]
                tick = time.perf_counter()
                chunks = [ts.ckks_vector(context, vector[start:start + 4096].tolist()).serialize()
                          for start in range(0, len(vector), 4096)]
                crypto_seconds += time.perf_counter() - tick
                sent += send_message(A, dict(round_id=rounds, client_id=client.id, chunks=chunks,
                                             alignment_error=info['alignment_error']))
            rounds += 1
    tick = time.perf_counter()
    _u, fit_labels = _fit_memberships(local_clients, bundle['train_indices'],
        final['centers_orig'], final['centers_emb'], cfg['fuzziness'], cfg['lambda_space'],
        cfg['normalize_spaces'], cfg['d_emb'], cfg['use_procrustes'], cfg['use_anchors'],
        final['previous_orig'], final['previous_emb'])
    test_labels, _u_test, dual = _test_labels(local_clients, bundle['X_test'],
        final['centers_orig'], final['centers_emb'], cfg['fuzziness'], cfg['lambda_space'],
        cfg['normalize_spaces'], 'nystrom', 'stored')
    fit = ours.clustering_quality(bundle['X_fit'], bundle['y_fit'], fit_labels)
    test = ours.clustering_quality(bundle['X_test'], bundle['y_test'], test_labels)
    return dict(C_pid=os.getpid(), C_has_secret_key=bool(context.has_secret_key()),
        C_setup_seconds=setup, C_sent_bytes=sent, C_completed_rounds=rounds,
        local_statistics_seconds=local_seconds, encryption_seconds=crypto_seconds,
        evaluation_seconds=time.perf_counter() - tick,
        test_basis='nystrom' if dual else 'original_space', init_uses_pooled_fit=False,
        n_distinct_labels=int(len(np.unique(fit_labels))),
        **{k: float(fit[k]) for k in ('acc', 'nmi', 'ari')},
        **{'test_' + k: float(test[k]) for k in ('acc', 'nmi', 'ari')})


def worker(role, job, reports, keys=None, address=None):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if role == 'D':
                result = decryptor(job, reports, keys['AD'])
            elif role == 'A':
                result = aggregator(job, reports, keys, address)
            elif role == 'C':
                result = clients(job, address, keys['CA'])
            else:
                bundle = job['synthetic_bundle'] if job.get('synthetic') else ours.load_partition_bundle(
                    job['dataset'], job['data_seed'], job['beta'], job['clients'], job['data_dir'])
                metrics, history, _ = ours.run_ours(bundle, job['config'], job['init_seed'],
                    eval_overrides=dict(use_ckks=True, run_privacy_eval=False,
                                        eigen_solver=job['eigen_solver']), return_details=True)
                cost = history['execution_cost']; comm = history['communication']
                result = dict(**metrics,
                    training_core_seconds=cost['trainer_seconds'] - cost['evaluation_seconds'],
                    setup_seconds=cost['setup_seconds'], evaluation_seconds=cost['evaluation_seconds'],
                    logical_total_bytes=comm['total_bytes'],
                    round_times_trace=json.dumps(history['round_times']),
                    native_pid=os.getpid(), centers_orig=history['centers_orig'][-1],
                    centers_emb=history['centers_emb'][-1])
        memory = psutil.Process().memory_info()
        result[role + '_peak_rss_bytes'] = int(getattr(memory, 'peak_wset', memory.rss))
        reports.put(dict(kind='done', role=role, result=result))
    except Exception:
        reports.put(dict(kind='error', role=role, error=traceback.format_exc()))


def run_case(job, execution, timeout_seconds=1800.):
    tick = time.perf_counter(); ctx = mp.get_context('spawn'); reports = ctx.Queue()
    processes = []; results = {}
    def next_report():
        try:
            message = reports.get(timeout=timeout_seconds)
        except queue.Empty:
            raise TimeoutError('Full-training worker did not report before the timeout') from None
        if message['kind'] == 'error':
            raise RuntimeError(message['role'] + ' worker failed:\n' + message['error'])
        return message
    try:
        if execution == 'native':
            process = ctx.Process(target=worker, args=('native', job, reports))
            processes.append(process); process.start()
            report = next_report(); results['native'] = report['result']
        else:
            keys = {'AD': secrets.token_bytes(32), 'CA': secrets.token_bytes(32)}
            public_job = {k: job[k] for k in ('K', 'd_orig', 'clients', 'config', 'init_seed')}
            D = ctx.Process(target=worker, args=('D', public_job, reports, {'AD': keys['AD']}))
            processes.append(D); D.start(); address_d = next_report()['address']
            A = ctx.Process(target=worker, args=('A', public_job, reports, keys, address_d))
            processes.append(A); A.start(); address_a = next_report()['address']
            C = ctx.Process(target=worker, args=('C', job, reports, {'CA': keys['CA']}, address_a))
            processes.append(C); C.start()
            while len(results) < 3:
                report = next_report(); results[report['role']] = report['result']
    finally:
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate(); process.join()
        reports.close(); reports.join_thread()
    if execution == 'native':
        result = results['native']
        result.update(distinct_role_processes=1, role_boundary_passed=np.nan,
                      socket_application_bytes=np.nan,
                      peak_role_rss_sum_bytes=result['native_peak_rss_bytes'])
    else:
        result = {k: value for record in results.values() for k, value in record.items()}
        result['distinct_role_processes'] = len({result['C_pid'], result['A_pid'], result['D_pid']})
        result['role_boundary_passed'] = bool(result['distinct_role_processes'] == 3
            and result['D_has_secret_key'] and not result['C_has_secret_key']
            and not result['A_has_secret_key'] and result['D_decryption_batches'] == result['rounds_run']
            and result['C_completed_rounds'] == result['rounds_run']
            and result['replay_rejected'] and result['unsupported_operation_rejected'])
        if not result['role_boundary_passed']:
            raise ValueError('Full-training role, round or key check failed')
        result['setup_seconds'] = sum(result[k + '_setup_seconds'] for k in ('C', 'A', 'D'))
        result['training_core_seconds'] = result['setup_seconds'] + result['A_round_seconds']
        result['socket_application_bytes'] = sum(result[k] for k in (
            'C_sent_bytes', 'A_sent_C_bytes', 'A_sent_D_bytes', 'D_sent_bytes'))
        result['policy_probe_application_bytes'] = result['A_probe_request_bytes'] + result['D_probe_reply_bytes']
        result['protocol_application_bytes'] = result['socket_application_bytes'] - result['policy_probe_application_bytes']
        result['peak_role_rss_sum_bytes'] = sum(result[k + '_peak_rss_bytes'] for k in ('C', 'A', 'D'))
        result['logical_total_bytes'] = np.nan
    result['centers_orig_sha256'] = hashlib.sha256(np.asarray(result.pop('centers_orig')).tobytes()).hexdigest()
    result['centers_emb_sha256'] = hashlib.sha256(np.asarray(result.pop('centers_emb')).tobytes()).hexdigest()
    result.update(wall_seconds=time.perf_counter() - tick, execution=execution,
        resource_basis='wall includes fresh worker startup/load/evaluation; peak sum is a sum of role lifetime peaks, not simultaneous peak',
        timing_basis='native core excludes per-round evaluation; socket core sums setup-service work and A round elapsed including transport, excludes final offline evaluation',
        communication_basis='native bytes are modeled logical M-client traffic; socket bytes are actual pickled application payloads with two public-context deliveries',
        communication_excluded='socket authentication, TCP/IP framing, coordinator queues and process bootstrap',
        protection_scope=('full iterative C/A/D role separation on one host; one C process multiplexes clients; released aggregates are visible to A/D'
                          if execution == 'process' else
                          'single-process reference; public/key roles are cooperating objects in one address space'),
        tenseal_context_threads='library default in both execution arms')
    if not np.isfinite([result[k] for k in SCORES]).all():
        raise ValueError('Full-training run returned nonfinite quality')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='har')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=50)
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--eigen-solvers', nargs='+', choices=('dense', 'partial'), default=['dense', 'partial'])
    parser.add_argument('--timeout-seconds', type=float, default=1800.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args(argv)
    if args.smoke_test:
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=120, n_features=8, centers=3, random_state=8)
        X = X / X.std(axis=0); X_fit, X_test = X[:90], X[90:]
        pieces = np.array_split(X_fit, 3); ownership = np.array_split(np.arange(90), 3)
        bundle = dict(X_list=pieces, X_fit=X_fit, y_fit=y[:90], X_train=X_fit,
            y_train=y[:90], X_test=X_test, y_test=y[90:], train_indices=ownership,
            K=3, median_distance=3.)
        cfg = {k: values[0] for k, values in ours.GRID.items()}
        cfg.update(num_rounds=3, topology_epochs=2, d_emb=2, k_neighbors=3, fuzziness=1.5)
        job = dict(synthetic=True, synthetic_bundle=bundle, K=3, d_orig=8, clients=3,
                   config=cfg, init_seed=40, eigen_solver='partial')
        if args.dry_run:
            print('[dry-run] Native and three-process three-round CKKS test; no files.'); return 0
        for execution in ('native', 'process'):
            result = run_case(job, execution, args.timeout_seconds)
            print(json.dumps({k: result[k] for k in ('execution', 'rounds_run', 'test_acc',
                'distinct_role_processes', 'role_boundary_passed', 'wall_seconds')}, default=str))
        return 0
    if not args.source_dir or not args.out_dir:
        parser.error('--source-dir and --out-dir are required')
    seeds = [int(value) for value in args.init_seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or len(set(args.eigen_solvers)) != len(args.eigen_solvers):
        parser.error('Use distinct seeds and solvers')
    source = Path(args.source_dir) / f'grid_ours_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source); mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    selected, cfg = select_complete(raw, mean, ours.GRID_PARAMS, list(range(40, 45)), 'acc')
    for k, value in cfg.items():
        cfg[k] = ((None if ours.sigma_label(value) == 'adaptive' else float(value))
                  if k == 'kernel_sigma' else type(ours.GRID[k][0])(value))
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta, args.clients, args.data_dir)
    proof = {k: bundle['provenance'][k] for k in ('data_sha256', 'partition_sha256')}
    identity = dict(dataset=args.dataset, data_seed=args.grid_split, beta=args.beta, clients=args.clients)
    for k, v in {**identity, **proof}.items():
        if not selected[k].eq(v).all():
            raise ValueError('Reference differs from frozen data: ' + k)
    if cfg['init_method'] not in ('random', 'uniform') or cfg['active_ratio'] != 1.:
        raise ValueError('This full-role validation uses data-independent initialization and full participation')
    reference = dict(reference_config=json.dumps(cfg, sort_keys=True),
                     reference_source_csv=str(source.resolve()), protocol='full-round-sockets-v1')
    out = Path(args.out_dir); stem = f'grid_ours_full_roles_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    if (raw_path.exists() or mean_path.exists()) and not args.resume:
        raise FileExistsError('Use --resume to preserve completed runs')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    expected = {(f'{execution}_{solver}', seed) for execution in ('native', 'process')
                for solver in args.eigen_solvers for seed in seeds}
    if len(existing):
        if existing.duplicated(['condition', 'init_seed']).any() or not set(zip(
                existing.condition, existing.init_seed)).issubset(expected):
            raise ValueError('Unexpected/duplicate saved condition or seed')
        for k, v in {**identity, **proof, **reference}.items():
            if not existing[k].eq(v).all():
                raise ValueError('Existing full-role output differs: ' + k)
    rows = existing[existing.status.eq('completed')].to_dict('records') if len(existing) else []
    if rows and not np.isfinite(pd.DataFrame(rows)[SCORES].to_numpy(float)).all():
        raise ValueError('Saved completed runs contain nonfinite quality')
    done = {(row['condition'], row['init_seed']) for row in rows}
    print(ours.partition_summary(bundle))
    print(f'{len(expected)} fixed runs; {len(done)} saved. Full native/process training, no search.')
    if args.dry_run:
        print('[dry-run] no workers or result files.'); return 0
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        conditions = [(execution, solver) for solver in args.eigen_solvers for execution in ('native', 'process')]
        if seed % 2:
            conditions.reverse()
        for execution, solver in conditions:
            condition = execution + '_' + solver
            if (condition, seed) in done:
                continue
            job = dict(**identity, K=bundle['K'], d_orig=bundle['X_fit'].shape[1],
                config=cfg, init_seed=seed, eigen_solver=solver, data_dir=args.data_dir)
            row = dict(**identity, **proof, **reference, condition=condition,
                       init_seed=seed, eigen_solver=solver, execution=execution,
                       method='ours', **{k: np.nan for k in SCORES})
            try:
                row.update(run_case(job, execution, args.timeout_seconds), status='completed', error='')
            except Exception:
                row.update(status='failed', error=traceback.format_exc())
                print(f'  FAILED {condition} init={seed}: {row["error"].splitlines()[-1]}', flush=True)
            rows.append(row); pd.DataFrame(rows).to_csv(raw_path, index=False)
            if row['status'] == 'completed':
                print(f'  {condition} init={seed} TEST={row["test_acc"]:.4f} '
                      f'rounds={row["rounds_run"]} wall={row["wall_seconds"]:.1f}s', flush=True)
    frame = pd.DataFrame(rows)
    good = frame[frame.status.eq('completed')]
    group = good.groupby(['condition', 'execution', 'eigen_solver'], sort=False)
    numeric = [c for c in good.select_dtypes(include=['number', 'bool']).columns
               if c not in ('data_seed', 'init_seed', 'beta', 'clients') and not c.endswith('_pid')]
    stats = group[numeric].mean().join(group[numeric].std(ddof=1).add_suffix('_std')).reset_index()
    summary = frame[['condition', 'execution', 'eigen_solver']].drop_duplicates().merge(
        stats, on=['condition', 'execution', 'eigen_solver'], how='left')
    for k, v in {**identity, **proof, **reference}.items():
        summary[k] = v
    summary['n_seeds'] = summary.condition.map(good.groupby('condition').size()).fillna(0).astype(int)
    summary['n_failed'] = summary.condition.map(frame[frame.status.eq('failed')].groupby('condition').size()).fillna(0).astype(int)
    summary['n_expected'] = len(seeds)
    summary['complete'] = summary.n_seeds.eq(len(seeds)) & summary.n_failed.eq(0)
    summary.to_csv(mean_path, index=False)
    print('wrote', raw_path, 'and', mean_path)
    return 0 if summary.complete.all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
