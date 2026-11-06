"""Analytic Gaussian/GDP calibration for the fixed FastLloyd round schedule.

Finite-precision sampling is an experimental implementation, not a certified
discrete-DP sampler. The result reports the ideal Gaussian guarantee separately.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr


def gdp_delta(epsilon, mu):
    if mu <= 0:
        return 0.0
    a = log_ndtr(-epsilon / mu + mu / 2)
    b = epsilon + log_ndtr(-epsilon / mu - mu / 2)
    return float(np.exp(a) * (-np.expm1(min(0.0, b - a))))


def calibrate_gdp(epsilon, delta):
    if epsilon <= 0 or not 0 < delta < 1 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and delta in (0,1)")
    upper = 1.0
    while gdp_delta(epsilon, upper) < delta:
        upper *= 2
    return brentq(lambda mu: gdp_delta(epsilon, mu) - delta, 1e-10, upper)


def allocated_noise(epsilon, delta, rounds, dimensions, radius, fractional_bits=16):
    mu = calibrate_gdp(epsilon, delta)
    sigma = 1 / mu
    root = np.sqrt(4 * dimensions)
    sigma_r = sigma * np.sqrt(1 + root) / (4 * dimensions)**.25
    sigma_c = sigma * np.sqrt(1 + root)
    # Local SUM is rounded once. Adjacent rounded sums can differ by an
    # extra 1/scale per coordinate; counts are exactly representable.
    sensitivity_r = radius + np.sqrt(dimensions) / 2**fractional_bits
    return sigma_r * sensitivity_r * np.sqrt(rounds), sigma_c * np.sqrt(rounds), mu
