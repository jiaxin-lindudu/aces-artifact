"""Validate one native first-round CKKS aggregation across three spawned roles.

C multiplexes the saved logical clients and retains individual plaintext
statistics. A adds ciphertexts and receives aggregate plaintext for the native
center update. D alone holds the secret key. C--A and A--D use authenticated
loopback sockets. This is a transport snapshot, not isolated end-to-end training,
a cryptographic proof, or a mitigation of CKKS decryption-output leakage.
Only raw and mean CSVs are written. --smoke-test writes no files.
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
import pickle
import platform
import queue
import secrets
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDITION = 'isolated_ckks_first_aggregate'


def send_message(connection, value):
    payload = pickle.dumps(value, protocol=5)
    connection.send_bytes(payload)
    return len(payload)


def receive_message(connection):
    return pickle.loads(connection.recv_bytes())


def decryptor_role(job, reports, authkey):
    import tenseal as ts
    started = time.perf_counter()
    key_started = time.perf_counter()
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1)
    context.global_scale = 2 ** 40
    public = context.serialize(save_public_key=True, save_secret_key=False,
                               save_galois_keys=False, save_relin_keys=False)
    key_seconds = time.perf_counter() - key_started
    length = job['K'] * (job['d_orig'] + job['config']['d_emb'] + 1)
    chunks_expected = (length + 4095) // 4096
    roster = list(range(job['clients']))
    completed_rounds = set()
    decrypt_seconds = 0.0
    decrypt_calls = rejected = 0
    probe_bytes = 0
    with Listener(('127.0.0.1', 0), family='AF_INET', authkey=authkey) as listener:
        reports.put(dict(kind='ready', role='D', address=listener.address))
        with listener.accept() as connection:
            hello = receive_message(connection)
            if hello != {'op': 'hello', 'role': 'A'}:
                raise ValueError('Expected the authenticated aggregator handshake')
            setup_bytes = send_message(connection, dict(public_context=public,
                decryptor_pid=os.getpid(), decryptor_has_secret_key=context.has_secret_key()))
            aggregate_reply_bytes = 0
            while True:
                message = receive_message(connection)
                if message['op'] == 'close':
                    break
                reason = None
                if message['op'] != 'aggregate':
                    reason = 'unsupported_operation'
                elif message['round_id'] in completed_rounds:
                    reason = 'round_replay'
                elif message['round_id'] != 0:
                    reason = 'unexpected_round'
                elif message['active_clients'] != roster:
                    reason = 'unexpected_roster'
                elif len(message['chunks']) != chunks_expected:
                    reason = 'unexpected_chunk_count'
                if reason is not None:
                    rejected += 1
                    probe_bytes += send_message(connection, dict(status='rejected', reason=reason))
                    continue
                tick = time.perf_counter()
                aggregate = np.concatenate([np.asarray(
                    ts.ckks_vector_from(context, chunk).decrypt(), dtype=float)
                    for chunk in message['chunks']])[:length]
                decrypt_seconds += time.perf_counter() - tick
                decrypt_calls += 1
                completed_rounds.add(message['round_id'])
                aggregate_reply_bytes += send_message(connection, dict(
                    status='decoded_aggregate', round_id=message['round_id'], aggregate=aggregate))
    return dict(D_pid=os.getpid(), D_has_secret_key=bool(context.has_secret_key()),
        key_generation_seconds=key_seconds, decryption_seconds=decrypt_seconds,
        public_context_bytes=len(public), D_setup_application_bytes=setup_bytes,
        D_aggregate_reply_application_bytes=aggregate_reply_bytes,
        D_probe_reply_application_bytes=probe_bytes, decryption_batches=decrypt_calls,
        ciphertexts_decrypted=decrypt_calls * chunks_expected, rejected_requests=rejected,
        D_seconds=time.perf_counter() - started)


def aggregator_role(job, reports, authkeys, decryptor_address):
    import tenseal as ts
    from federated.ours.aces_clustering import AnchoredSpectralServer
    started = time.perf_counter()
    K, d, p = job['K'], job['d_orig'], job['config']['d_emb']
    b = (K * (d + p + 1) + 4095) // 4096
    with Client(decryptor_address, family='AF_INET', authkey=authkeys['AD']) as decryptor:
        setup_bytes = send_message(decryptor, {'op': 'hello', 'role': 'A'})
        hello = receive_message(decryptor)
        context = ts.context_from(hello['public_context'], n_threads=1)
        if context.has_secret_key():
            raise ValueError('Aggregator received a secret key')
        server = AnchoredSpectralServer(K, d, p, use_ckks=False)
        centers_x, centers_h = server.initialize_centers(job['config']['init_method'],
                                                       None, job['init_seed'])
        sums = None
        received = set()
        addition_seconds = 0.0
        with Listener(('127.0.0.1', 0), family='AF_INET', authkey=authkeys['CA']) as listener:
            reports.put(dict(kind='ready', role='A', address=listener.address))
            with listener.accept() as clients:
                setup_bytes += send_message(clients, dict(public_context=hello['public_context'],
                    centers_orig=centers_x, centers_emb=centers_h))
                for _ in range(job['clients']):
                    message = receive_message(clients)
                    client_id = message['client_id']
                    if (client_id not in range(job['clients']) or client_id in received
                            or message['round_id'] != 0 or len(message['chunks']) != b):
                        raise ValueError('Invalid or duplicate client snapshot')
                    received.add(client_id)
                    tick = time.perf_counter()
                    vectors = [ts.ckks_vector_from(context, chunk) for chunk in message['chunks']]
                    sums = vectors if sums is None else [a + value for a, value in zip(sums, vectors)]
                    addition_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        chunks = [value.serialize() for value in sums]
        addition_seconds += time.perf_counter() - tick
        request = dict(op='aggregate', round_id=0, active_clients=sorted(received), chunks=chunks)
        aggregate_request_bytes = send_message(decryptor, request)
        decoded = receive_message(decryptor)
        if decoded['status'] != 'decoded_aggregate':
            raise ValueError('Authorized aggregate was rejected')
        aggregate = decoded['aggregate']
        first, second = K * d, K * (d + p)
        tick = time.perf_counter()
        centers_x, centers_h = server.aggregate_statistics([(
            aggregate[:first].reshape(K, d), aggregate[first:second].reshape(K, p),
            aggregate[second:], {'alignment_error': None})])
        center_seconds = time.perf_counter() - tick
        # These are policy checks, not proofs that a malicious A submitted a sum.
        probe_bytes = send_message(decryptor, dict(op='decrypt_individual', round_id=0))
        unsupported = receive_message(decryptor)
        probe_bytes += send_message(decryptor, request)
        replay = receive_message(decryptor)
        close_bytes = send_message(decryptor, {'op': 'close'})
    return dict(A_pid=os.getpid(), A_has_secret_key=bool(context.has_secret_key()),
        A_setup_application_bytes=setup_bytes,
        A_aggregate_request_application_bytes=aggregate_request_bytes,
        A_probe_application_bytes=probe_bytes, A_close_application_bytes=close_bytes,
        aggregate_ciphertext_bytes=sum(map(len, chunks)), addition_seconds=addition_seconds,
        center_update_seconds=center_seconds,
        unsupported_operation_rejected=(unsupported.get('reason') == 'unsupported_operation'),
        replay_rejected=(replay.get('reason') == 'round_replay'),
        received_client_count=len(received), aggregate=aggregate,
        centers_orig=centers_x, centers_emb=centers_h,
        A_seconds=time.perf_counter() - started)


def client_role(job, reports, authkey, aggregator_address):
    import tenseal as ts
    from federated.ours.aces_clustering import (
        AnchoredSpectralClient, AnchoredSpectralServer)
    from experiments.partition_checks import array_hash
    started = time.perf_counter()
    if job.get('synthetic'):
        rng = np.random.default_rng(job['init_seed'])
        arrays = [rng.standard_normal((12, job['d_orig'])) + i * .4
                  for i in range(job['clients'])]
        proof = dict(data_sha256='synthetic_smoke', partition_sha256='synthetic_smoke')
    else:
        from data.datasets_and_utils import load_federated_partition
        partition = load_federated_partition(job['dataset'], job['grid_split'], job['beta'],
            job['clients'], base_dir=job['data_dir'], verbose=False)
        ownership = [np.asarray(indices, dtype=int).tolist() for indices in partition.client_indices]
        hashes = {name: array_hash(getattr(partition, name))
                  for name in ('X_train', 'X_test', 'y_train', 'y_test')}
        proof = dict(data_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            partition_sha256=hashlib.sha256(json.dumps(ownership).encode()).hexdigest())
        if proof != job['proof']:
            raise ValueError('Client snapshot differs from the selected frozen partition')
        arrays = [np.asarray(partition.X_train[indices], dtype=float) for indices in ownership]
    cfg = job['config']
    K, d, p = job['K'], job['d_orig'], cfg['d_emb']
    reference = np.zeros(K * (d + p + 1))
    local_seconds = encryption_seconds = 0.0
    upload_bytes = ciphertext_bytes = 0
    with Client(aggregator_address, family='AF_INET', authkey=authkey) as connection:
        initial = receive_message(connection)
        context = ts.context_from(initial['public_context'], n_threads=1)
        if context.has_secret_key():
            raise ValueError('Client process received a secret key')
        for i, X in enumerate(arrays):
            tick = time.perf_counter()
            client = AnchoredSpectralClient(id=i, X=X, anchor_weight=cfg['anchor_weight'],
                beta=cfg['lambda_space'], dp_sigma=0., graph_method=cfg['graph_method'],
                k_neighbors=cfg['k_neighbors'], kernel_sigma=cfg['kernel_sigma'], use_ckks=False,
                use_anchors=cfg['use_anchors'], normalize_spaces=cfg['normalize_spaces'],
                kernel_scale_mode=cfg['kernel_scale_mode'],
                match_procrustes_scale=cfg['match_procrustes_scale'], random_state=job['init_seed'])
            Sx, Sh, masses, diagnostics = client.compute_dual_statistics(
                initial['centers_orig'], initial['centers_emb'], cfg['fuzziness'], cfg['use_procrustes'])
            vector = np.r_[Sx.ravel(), Sh.ravel(), masses]
            reference += vector
            local_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            chunks = [ts.ckks_vector(context, vector[offset:offset + 4096].tolist()).serialize()
                      for offset in range(0, len(vector), 4096)]
            encryption_seconds += time.perf_counter() - tick
            ciphertext_bytes += sum(map(len, chunks))
            upload_bytes += send_message(connection, dict(round_id=0, client_id=i,
                chunks=chunks, alignment_error=diagnostics['alignment_error']))
        first, second = K * d, K * (d + p)
        reference_server = AnchoredSpectralServer(K, d, p, use_ckks=False)
        centers_x, centers_h = reference_server.aggregate_statistics([(
            reference[:first].reshape(K, d), reference[first:second].reshape(K, p),
            reference[second:], {'alignment_error': None})])
    return dict(C_pid=os.getpid(), C_has_secret_key=bool(context.has_secret_key()),
        C_upload_application_bytes=upload_bytes, client_ciphertext_bytes=ciphertext_bytes,
        local_statistics_seconds=local_seconds, encryption_seconds=encryption_seconds,
        C_seconds=time.perf_counter() - started, reference_aggregate=reference,
        reference_centers_orig=centers_x, reference_centers_emb=centers_h,
        minimum_reference_cluster_mass=float(reference[-K:].min()),
        negligible_mass_clusters=int(np.count_nonzero(reference[-K:] < 1e-6)), **proof)


def role_worker(role, job, reports, authkeys, address=None):
    # Exceptions cross this process boundary as text, never as local statistics.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if role == 'D':
                result = decryptor_role(job, reports, authkeys['AD'])
            elif role == 'A':
                result = aggregator_role(job, reports, authkeys, address)
            else:
                result = client_role(job, reports, authkeys['CA'], address)
        reports.put(dict(kind='done', role=role, result=result))
    except Exception:
        reports.put(dict(kind='error', role=role, error=traceback.format_exc()))


def run_case(job, timeout_seconds, atol, center_atol, rtol):
    started = time.perf_counter()
    context = mp.get_context('spawn')
    reports = context.Queue()
    authkeys = {'AD': secrets.token_bytes(32), 'CA': secrets.token_bytes(32)}
    public_job = {key: job[key] for key in ('clients', 'K', 'd_orig', 'config', 'init_seed')}
    workers = []
    def report():
        try:
            message = reports.get(timeout=timeout_seconds)
        except queue.Empty:
            raise TimeoutError('A role did not complete the transport snapshot in time') from None
        if message['kind'] == 'error':
            raise RuntimeError(message['role'] + ' role failed:\n' + message['error'])
        return message
    try:
        decryptor = context.Process(target=role_worker,
            args=('D', public_job, reports, {'AD': authkeys['AD']}))
        workers.append(decryptor); decryptor.start()
        address_d = report()['address']
        aggregator = context.Process(target=role_worker,
            args=('A', public_job, reports, authkeys, address_d))
        workers.append(aggregator); aggregator.start()
        address_a = report()['address']
        clients = context.Process(target=role_worker,
            args=('C', job, reports, {'CA': authkeys['CA']}, address_a))
        workers.append(clients); clients.start()
        results = {}
        while len(results) < 3:
            message = report()
            results[message['role']] = message['result']
    finally:
        for worker in workers:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate(); worker.join()
        reports.close(); reports.join_thread()
    D, A, C = (results[role] for role in ('D', 'A', 'C'))
    reference = C.pop('reference_aggregate')
    actual = A.pop('aggregate')
    orig_reference, emb_reference = C.pop('reference_centers_orig'), C.pop('reference_centers_emb')
    orig_actual, emb_actual = A.pop('centers_orig'), A.pop('centers_emb')
    if not all(np.isfinite(x).all() for x in (reference, actual, orig_actual, emb_actual)):
        raise ValueError('Nonfinite aggregate or centers')
    row = {**D, **A, **C}
    row['distinct_role_processes'] = len({row['D_pid'], row['A_pid'], row['C_pid']})
    row['role_boundary_passed'] = bool(row['distinct_role_processes'] == 3 and
        row['D_has_secret_key'] and not row['A_has_secret_key'] and not row['C_has_secret_key'] and
        row['decryption_batches'] == 1 and row['received_client_count'] == job['clients'] and
        row['unsupported_operation_rejected'] and row['replay_rejected'])
    if not row['role_boundary_passed']:
        raise ValueError('Role/key/routing checks failed')
    row.update(aggregate_max_abs_error=float(np.abs(reference - actual).max()),
        aggregate_relative_linf_error=float(np.abs(reference - actual).max() / max(1., np.abs(reference).max())),
        center_max_abs_error_orig=float(np.abs(orig_reference - orig_actual).max()),
        center_max_abs_error_emb=float(np.abs(emb_reference - emb_actual).max()),
        aggregate_check_passed=bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
        centers_check_passed=bool(np.allclose(orig_actual, orig_reference, atol=center_atol, rtol=rtol)
            and np.allclose(emb_actual, emb_reference, atol=center_atol, rtol=rtol)),
        aggregate_atol=atol, center_atol=center_atol, validation_rtol=rtol,
        packed_values=len(reference), ciphertexts_per_client=(len(reference) + 4095) // 4096,
        offline_reference_bytes=reference.nbytes + orig_reference.nbytes + emb_reference.nbytes,
        wall_seconds=time.perf_counter() - started)
    row['numeric_check_passed'] = row['aggregate_check_passed'] and row['centers_check_passed']
    row['numeric_status'] = 'within_declared_tolerances' if row['numeric_check_passed'] else 'mismatch_retained'
    row['setup_application_bytes'] = row['D_setup_application_bytes'] + row['A_setup_application_bytes']
    row['normal_application_bytes'] = (row['setup_application_bytes'] + row['C_upload_application_bytes']
        + row['A_aggregate_request_application_bytes'] + row['D_aggregate_reply_application_bytes'])
    row['probe_application_bytes'] = row['A_probe_application_bytes'] + row['D_probe_reply_application_bytes']
    row['total_socket_application_bytes'] = (row['normal_application_bytes'] + row['probe_application_bytes']
        + row['A_close_application_bytes'])
    row.update(status='completed', method='ours', condition=CONDITION, snapshot_rounds=1,
        snapshot_phase='first aggregate before topology; canonical data-independent initialization',
        protection_scope='three-process aggregation transport snapshot; one C process multiplexes logical clients; not full isolated training or a security proof',
        host_access_scope='separate process address spaces under the same OS user; no shared-filesystem access-control hardening',
        released_objects='aggregate plaintext to A/D; public prototypes; individual clear diagnostics; offline aggregate reference to validation coordinator',
        policy_scope='authenticated endpoints; op/round/roster/chunk checks; replay rejected; no proof a malicious aggregator supplied the sum',
        decryption_output_scope='approximate aggregates released without noise flooding; no IND-CPA-D guarantee',
        communication_basis='actual pickled application payloads on authenticated AF_INET loopback C-A and A-D sockets',
        communication_excluded='authentication handshake, connection framing, TCP/IP, process-control queues and offline aggregate-reference checks',
        setup_basis='two physical public-context deliveries to A and the multiplexed C, not M independent client endpoints',
        timing_basis='single-host snapshot; wall includes process startup; native local stats, crypto and update stages separately timed',
        python_version=platform.python_version(), numpy_version=np.__version__,
        hardware=platform.platform(), blas_omp_threads=1, tenseal_context_threads=1)
    return row


def write_results(rows, raw_path, mean_path, identity, expected_seeds):
    frame = pd.DataFrame(rows)
    frame.to_csv(raw_path, index=False)
    done = frame[frame.status.eq('completed')]
    summary = dict(identity, condition=CONDITION, n_seeds=len(done), n_expected_seeds=expected_seeds,
                   n_failed=len(frame) - len(done),
                   status='completed' if len(done) == expected_seeds else 'incomplete',
                   uncertainty_scope='initialization variability within one frozen split')
    for key in done.select_dtypes(include=['number', 'bool']).columns:
        if key not in ('init_seed', 'data_seed', 'beta', 'clients') and not key.endswith('_pid'):
            summary[key] = done[key].mean()
            summary[key + '_std'] = done[key].std(ddof=1)
    for key in ('config', 'reference_source_csv', 'data_sha256', 'partition_sha256',
                'protection_scope', 'communication_basis', 'communication_excluded',
                'timing_basis', 'released_objects', 'policy_scope', 'decryption_output_scope',
                'host_access_scope', 'native_core_sha256'):
        summary[key] = '; '.join(map(str, frame[key].dropna().unique())) if key in frame else ''
    pd.DataFrame([summary]).to_csv(mean_path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='digits')
    parser.add_argument('--grid-split', type=int, default=0)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--init-seeds', default='40,41,42,43,44')
    parser.add_argument('--selection-seeds', default='40,41,42,43,44',
                        help='Seeds used by the saved FIT selection; validation may use a subset')
    parser.add_argument('--select-on', choices=('acc', 'nmi', 'ari'), default='acc')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--source-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--timeout-seconds', type=float, default=900.)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--center-atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Completed rows are always reused')
    parser.add_argument('--smoke-test', action='store_true', help='Synthetic three-role check; writes no files')
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or min(args.atol, args.center_atol, args.rtol) < 0:
        parser.error('Require a positive timeout and nonnegative numerical tolerances')
    if args.smoke_test:
        cfg = dict(d_emb=2, init_method='random', anchor_weight=1., lambda_space=.7,
            graph_method='knn', k_neighbors=3, kernel_sigma=None, use_anchors=True,
            normalize_spaces=True, kernel_scale_mode='self_tuning', match_procrustes_scale=True,
            fuzziness=2., use_procrustes=True)
        job = dict(synthetic=True, clients=3, K=3, d_orig=4, init_seed=40, config=cfg)
        if args.dry_run:
            print('[dry-run] Synthetic three-process snapshot; no files.')
            return 0
        result = run_case(job, args.timeout_seconds, args.atol, args.center_atol, args.rtol)
        print(json.dumps({key: result[key] for key in ('status', 'distinct_role_processes',
            'role_boundary_passed', 'numeric_check_passed', 'aggregate_max_abs_error',
            'center_max_abs_error_orig', 'center_max_abs_error_emb', 'replay_rejected',
            'unsupported_operation_rejected', 'total_socket_application_bytes')}))
        return 0 if result['numeric_check_passed'] else 1
    if not args.source_dir or not args.out_dir:
        parser.error('--source-dir and --out-dir are required unless --smoke-test is used')
    seeds = [int(value) for value in args.init_seeds.split(',') if value.strip()]
    selection_seeds = [int(value) for value in args.selection_seeds.split(',') if value.strip()]
    if (not seeds or len(set(seeds)) != len(seeds) or not selection_seeds
            or len(set(selection_seeds)) != len(selection_seeds)):
        parser.error('Initialization and selection seed lists must be nonempty and distinct')
    from experiments.scripts import grid_search_ours as ours
    from experiments.scripts.compare_grid_results import select_complete
    bundle = ours.load_partition_bundle(args.dataset, args.grid_split, args.beta, args.clients, args.data_dir)
    proof = {key: bundle['provenance'][key] for key in ('data_sha256', 'partition_sha256')}
    identity = dict(dataset=args.dataset, data_seed=args.grid_split, beta=args.beta, clients=args.clients)
    source = Path(args.source_dir) / f'grid_ours_beta{args.beta}_clients{args.clients}.csv'
    raw = pd.read_csv(source); mean = pd.read_csv(source.with_name(source.stem + '_mean.csv'))
    for key, value in {**identity, **proof}.items():
        if not raw[key].eq(value).all():
            raise ValueError('Reference differs from the frozen partition: ' + key)
    selected, cfg = select_complete(raw, mean, ours.GRID_PARAMS, selection_seeds, args.select_on)
    for key, value in cfg.items():
        cfg[key] = (None if ours.sigma_label(value) == 'adaptive' else float(value)) if key == 'kernel_sigma' else type(ours.GRID[key][0])(value)
    if selected.init_uses_pooled_fit.any() or cfg['init_method'] not in ('random', 'uniform') or cfg['active_ratio'] != 1.:
        raise ValueError('Require native random/uniform initialization and full participation')
    config = json.dumps(cfg, sort_keys=True)
    core_hash = hashlib.sha256((ROOT / 'federated/ours/aces_clustering.py').read_bytes()).hexdigest()
    validation = dict(aggregate_atol=args.atol, center_atol=args.center_atol, validation_rtol=args.rtol)
    out = Path(args.out_dir)
    stem = f'grid_ckks_role_validation_beta{args.beta}_clients{args.clients}'
    raw_path, mean_path = out / (stem + '.csv'), out / (stem + '_mean.csv')
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    if len(existing):
        if existing.init_seed.duplicated().any() or not set(existing.init_seed).issubset(seeds):
            raise ValueError('Unexpected/duplicate existing seeds; use another output directory')
        for key, value in {**identity, **proof, **validation, 'config': config,
                           'condition': CONDITION, 'native_core_sha256': core_hash}.items():
            if not existing[key].eq(value).all():
                raise ValueError('Existing output identity/configuration differs: ' + key)
    rows = existing[existing.status.eq('completed')].to_dict('records') if len(existing) else []
    completed = {row['init_seed'] for row in rows}
    print(ours.partition_summary(bundle))
    print('Selected on FIT:', ours.describe(cfg))
    print(f'{len(seeds)} first-round transport snapshots, {len(completed)} complete; no search or quality refit.')
    if args.dry_run:
        print('[dry-run] No processes, training, partitions or output files created.')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for seed in seeds:
        if seed in completed:
            continue
        job = dict(dataset=args.dataset, grid_split=args.grid_split, beta=args.beta, clients=args.clients,
            data_dir=args.data_dir, K=bundle['K'], d_orig=bundle['X_fit'].shape[1], init_seed=seed,
            config=cfg, proof=proof)
        row = dict(**identity, **proof, **validation, init_seed=seed, condition=CONDITION, config=config,
            reference_source_csv=str(source.resolve()), select_on=args.select_on,
            selection_seeds=','.join(map(str, selection_seeds)), reference_round_budget=cfg['num_rounds'],
            native_core_sha256=core_hash)
        try:
            row.update(run_case(job, args.timeout_seconds, args.atol, args.center_atol, args.rtol))
            print(f"  init={seed} roles={row['distinct_role_processes']} aggregate error={row['aggregate_max_abs_error']:.3g} centers={row['numeric_status']}", flush=True)
        except Exception as error:
            failures += 1
            row.update(status='failed', error=str(error))
            print(f'  init={seed} FAILED: {error}', flush=True)
        rows.append(row)
        write_results(rows, raw_path, mean_path, identity, len(seeds))
    write_results(rows, raw_path, mean_path, identity, len(seeds))
    print(f'wrote {raw_path} and {mean_path}')
    return 1 if failures else 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
