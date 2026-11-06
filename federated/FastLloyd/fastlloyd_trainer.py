import secrets
import time
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from baselines.paper_common import PaperResult, validate_clients, timed_record
from .fastlloyd_clustering import FastLloydClient, sphere_packing_centers, update_centers
from .fastlloyd_tracker import FastLloydTracker
from .privacy import allocated_noise, calibrate_gdp
from .protocol import MSAServer, MSAClients


def fastlloyd_clustering(x_list, n_clusters, *, num_rounds=50, random_state=42,
                         epsilon=3.0, delta=1e-5, public_bound=3.0,
                         radius=None, radius_policy="step", radius_alpha=.8,
                         fractional_bits=16, enable_dp=True, initial_centers=None,
                         X_train=None, y_train=None, X_test=None, y_test=None):
    # DP adjacency includes deleting the last record of a fixed participant.
    # Public initialization does not require even one private record per center.
    arrays = validate_clients(x_list, n_clusters, allow_empty=True,
                             require_samples_for_clusters=False)
    if (not isinstance(num_rounds, (int, np.integer)) or isinstance(num_rounds, bool)
            or num_rounds < 1 or not isinstance(fractional_bits, (int, np.integer))
            or isinstance(fractional_bits, bool) or not 0 < fractional_bits <= 30 or public_bound <= 0
            or not np.isfinite(public_bound) or radius_alpha <= 0
            or not np.isfinite(radius_alpha) or not isinstance(enable_dp, (bool, np.bool_))
            or radius_policy not in ("constant", "step")):
        raise ValueError("invalid FastLloyd schedule/domain/precision")
    d = arrays[0].shape[1]
    target_radius = (radius_alpha * public_bound * np.sqrt(d) / n_clusters**(1/d)
                     if radius is None else radius)
    if target_radius <= 0 or not np.isfinite(target_radius):
        raise ValueError("radius must be positive and finite")
    centers = (sphere_packing_centers(n_clusters, d, public_bound, random_state)
               if initial_centers is None else np.asarray(initial_centers, float).copy())
    if (centers.shape != (n_clusters, d) or not np.isfinite(centers).all()
            or np.any(np.abs(centers) > public_bound)):
        raise ValueError("initial centers must lie in the public domain")
    clients = [FastLloydClient(x, public_bound) for x in arrays]
    group, server, tracker = MSAClients(len(clients), fractional_bits), MSAServer(), FastLloydTracker()
    # Never reuse the public initialization seed for DP randomness.
    rng = secrets.SystemRandom()
    initial_sha256 = hashlib.sha256(np.ascontiguousarray(centers).tobytes()).hexdigest()
    noise_scales = {}
    if enable_dp:
        for eta in {target_radius, public_bound * np.sqrt(d) if radius_policy == "step" else target_radius}:
            noise_scales[eta] = allocated_noise(epsilon, delta, num_rounds, d, eta, fractional_bits)
    evaluation = dict(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
    # Fixed public T; no data-dependent early stopping or radius selection.
    for r in range(1, num_rounds + 1):
        start = time.perf_counter()
        previous = centers.copy()
        eta = public_bound * np.sqrt(d) if r == 1 and radius_policy == "step" else target_radius
        stats = [client.statistics(previous, eta) for client in clients]
        stats_time = time.perf_counter() - start
        t = time.perf_counter()
        messages, masks = group.mask([v for v, _ in stats])
        mask_time = time.perf_counter() - t
        if enable_dp:
            sd_r, sd_c, mu = noise_scales[eta]
            # Two independent draws avoid the single-draw Gaussian reconstruction issue.
            standard = np.asarray([(rng.normalvariate(0, 1) + rng.normalvariate(0, 1)) / np.sqrt(2)
                                   for _ in range(n_clusters * (d + 1))])
            noise = standard * np.r_[np.full(n_clusters*d, sd_r), np.full(n_clusters, sd_c)]
        else:
            sd_r = sd_c = 0.0
            noise = np.zeros(n_clusters * (d+1))
        t = time.perf_counter()
        released_masked = server.aggregate(messages, noise, fractional_bits)
        agg_time = time.perf_counter() - t
        t = time.perf_counter()
        released = group.reveal_to_clients(released_masked, masks)
        centers = update_centers(previous, released, eta, public_bound,
                                 count_floor=1.0 if enable_dp else 0.0)
        unmask_time = time.perf_counter() - t
        payload = sum(len(v.tobytes()) for v in messages)
        timed_record(tracker, r, centers, previous, start, evaluation,
            uplink=payload, downlink=len(clients) * len(released_masked.tobytes()), logical=payload,
            times=dict(statistics_time=stats_time, encryption_time=mask_time,
                       aggregation_time=agg_time, decryption_time=unmask_time),
            predictor=lambda x: cdist(np.clip(np.asarray(x), -public_bound, public_bound), centers).argmin(axis=1),
            diagnostics=dict(radius=eta, noise_std_relative=sd_r, noise_std_count=sd_c))
        # Accepted counts are deliberately NOT published: they are private queries.
    tracker.privacy_results = dict(
        status="ideal_gaussian_dp_with_finite_precision_caveat" if enable_dp else "dp_disabled",
        epsilon=float(epsilon) if enable_dp else None, delta=float(delta) if enable_dp else None,
        level="record", adjacency="add/remove one supplied transformed record",
        accountant="analytic Gaussian GDP; joint count/relative query; fixed-T composition" if enable_dp else None,
        mu_total=float(calibrate_gdp(epsilon, delta)) if enable_dp else None,
        steps=num_rounds, public_bound=public_bound, fractional_bits=fractional_bits,
        caveat="floating Gaussian and deterministic quantization are not a certified discrete DP sampler",
        preprocessing_scope="upstream fitted scaling/PCA is NOT covered by this accountant",
        release_scope="noisy aggregates/model only; raw evaluation scores/timing are NOT DP releases")
    tracker.privacy_results.update(
        end_to_end_dp=False,
        composition_scope="all fixed-T intermediate count and relative-sum queries in THIS run",
        benchmark_scope="fixed transformed input; private preprocessing, label-assisted tuning and multiple experiment releases are NOT jointly accounted",
        noise_randomness="SystemRandom; independent of public initialization seed",
        finite_precision_scope="ideal Gaussian calibration plus conservative local-rounding sensitivity; NOT certified discrete-DP")
    tracker.communication["setup_bytes"] = None
    tracker.communication["setup_note"] = "client-only confidential group-key provisioning not simulated"
    tracker.communication["measurement_basis"] = (
        "serialized uint64 masked application payload; uplink M*K*(d+1)*8 and "
        "unicast downlink M*K*(d+1)*8 per round; excludes key provisioning, public configuration and network framing")
    return PaperResult(centers, tracker, dict(
        reference="Diaa et al., FastLloyd, USENIX Security 2025",
        protocol_scope="MSA same-process application transcript; no MPI transport/dropout support",
        threat_model="non-colluding server and clients; all clients participate",
        numerical_rules="zero raw empty counts; DP noisy count<=1 retains center (explicit stabilization); no-DP count>0 updates; radius clipping and reflection",
        initial_centers_sha256=initial_sha256,
        training_seconds=float(sum(tracker.history['round_time'])),
        radius_policy=radius_policy, radius_alpha=radius_alpha, target_radius=float(target_radius),
        public_bound=float(public_bound), fractional_bits=fractional_bits,
        stop_reason="fixed_public_round_limit",
        source_parity="independent paper Eq.5-10 reproduction; NOT author MPI deployment; no source copied",
        initialization=("public sphere packing" if initial_centers is None else
                        "caller supplied; must be public or separately privacy-accounted"),
        prediction="nearest centroid after the SAME fixed public-domain projection as training"),
        predictor=lambda x: cdist(np.clip(np.asarray(x), -public_bound, public_bound), centers).argmin(axis=1))
