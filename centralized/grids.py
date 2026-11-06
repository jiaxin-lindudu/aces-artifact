"""Hyper-parameter grids for the centralized ceilings.

Each grid is deliberately small but has to contain a genuinely different
setting, otherwise "tuned" would be a lie:

    KMeans    init and n_init -- n_init=1 with a random seed separates how much
              of the score is owed to seeding rather than to the data.
    FCM       the fuzziness exponent. This is the one that actually matters:
              m >= 1.5 sits on the uniform-membership fixed point in a 64-d
              space, so those cells are kept in the grid on purpose, as the
              evidence that the collapse is a property of m and not of the run.
    Spectral  bandwidth heuristic (1/d vs median) and whether the embedding rows
              are renormalised (Ng-Jordan-Weiss vs raw eigenvectors).
"""
from __future__ import annotations

KMEANS_GRID = [
    {'init': 'k-means++', 'n_init': 10},
    {'init': 'k-means++', 'n_init': 1},
    {'init': 'random', 'n_init': 10},
    {'init': 'random', 'n_init': 1},
]

FCM_GRID = [
    {'fuzziness': 1.1},
    {'fuzziness': 1.3},
    {'fuzziness': 1.5},
    {'fuzziness': 2.0},
]

SPECTRAL_GRID = [
    {'gamma_mode': 'one_over_d', 'normalize_rows': True},
    {'gamma_mode': 'one_over_d', 'normalize_rows': False},
    {'gamma_mode': 'median', 'normalize_rows': True},
    {'gamma_mode': 'median', 'normalize_rows': False},
]

GRIDS = {
    'central_kmeans': KMEANS_GRID,
    'central_fcm': FCM_GRID,
    'central_spectral': SPECTRAL_GRID,
}


def describe(method, cfg):
    """Stable one-line label for a configuration, for tables and CSV."""
    if method == 'central_fcm':
        return 'm=%.2g' % cfg['fuzziness']
    if method == 'central_kmeans':
        return 'init=%s n_init=%d' % (cfg['init'], cfg['n_init'])
    return 'gamma=%s rows_norm=%s' % (cfg['gamma_mode'], cfg['normalize_rows'])
