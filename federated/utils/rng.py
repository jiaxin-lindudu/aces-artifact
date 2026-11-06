"""Named, isolated random streams for the Ours training path.

Before this module existed, the Ours path drew active-client sampling, DP
perturbation and center initialization from ONE global numpy stream, and reset
that stream repeatedly (`np.random.seed(...)`). That is fragile: any unrelated
numpy draw anywhere in the process silently shifts every later draw, so results
depend on call order rather than on the configured seed.

`isolated_rng` derives an independent child stream per (purpose, seed, ...) from
a numpy SeedSequence. Two consumers can no longer disturb each other, and each
one is reproducible on its own.

The purpose tag is folded in with zlib.crc32, NOT with the builtin hash():
Python's string hashing is salted per process (PYTHONHASHSEED), so hash() would
make the "reproducible" stream differ between runs.
"""
import zlib

import numpy as np


def isolated_rng(purpose, seed, *counters):
    """Return a Generator dedicated to a single purpose.

    purpose  : stable short name, e.g. 'active_clients', 'dp_noise', 'center_init'
    seed     : base seed of the run
    counters : optional integers that further separate the stream, e.g. a round
               index, so that consecutive rounds do not draw identical values
    """
    tag = zlib.crc32(str(purpose).encode('utf-8')) & 0xFFFFFFFF
    entropy = [int(seed) & 0xFFFFFFFF, tag]
    entropy.extend(int(counter) & 0xFFFFFFFF for counter in counters)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=entropy)))
