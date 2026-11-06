"""Centralized ceilings: K-Means, fuzzy c-means, spectral clustering.

These are the no-federation upper bounds that every federated number is judged
against. They run on the POOLED FIT block of a frozen partition -- no clients,
no anchor graph, no Nyström landmark approximation error -- and are scored on
both blocks:

    train (FIT)  the labels the method produces on the data it clustered
    test  (TEST) the same model applied out-of-sample to the frozen TEST block

The TEST side is what makes these comparable to the federated rows instead of
merely optimistic: a ceiling that only ever reports its own training block
cannot be used to judge a method that is scored on held-out data.

Out-of-sample rule per method -- this is the part that is easy to get wrong, so
it is stated explicitly:

    K-Means    sklearn's own ``predict`` (nearest center). Exact.
    FCM        nearest center in the original space, same centers the FIT
               memberships were built from. Exact for the hard-label reading.
    Spectral    Nyström extension of the affinity eigenvectors, i.e. the TEST
               point is embedded through its affinity to the FIT points and
               assigned to the nearest embedding center. This is the same
               out-of-sample mechanism `ours` uses (``test_basis='nystrom'``),
               so the two are on one footing. NOTE: a k-nearest-neighbour
               affinity is deliberately NOT offered -- building that graph
               requires the TEST points, which would make the reported test
               score transductive and no longer an honest ceiling.

Every method is grid-tuned (see ``centralized.grids``) before it is used as a
ceiling: an untuned ceiling is not a bound, it is just one arbitrary setting.

WHAT THESE NUMBERS DO AND DO NOT DEPEND ON
------------------------------------------
A centralized number is a function of (dataset, data_seed, init_seed) only. The
client count and dirichlet_beta decide how the FIT rows are handed out to
clients, and a centralized method never sees that assignment -- so its scores
must be identical across every (clients, beta). That is asserted, not assumed:
``centralized/verify_invariance.py`` compares the pooled blocks bitwise across
(client, beta) combinations and re-runs all three methods on the two most
different partitions. It passes, so a single centralized ceiling is valid for
every federated configuration, and the ``clients``/``dirichlet_beta`` columns
carried on the output rows identify WHICH frozen partition the pooled block was
read from -- they are provenance, not factors.

Consequence for the paper: client count and heterogeneity are knobs that act on
the FEDERATED methods only. Any claim of the form "more clients / more skew
hurts" has to be measured on the federated side; the ceiling does not move.
"""
from __future__ import annotations

from .fuzzy_cmeans import run as run_fcm
from .kmeans import run as run_kmeans
from .spectral import run as run_spectral

METHODS = {
    'central_kmeans': run_kmeans,
    'central_fcm': run_fcm,
    'central_spectral': run_spectral,
}

__all__ = ['METHODS', 'run_kmeans', 'run_fcm', 'run_spectral']
