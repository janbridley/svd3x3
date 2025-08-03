"""Fast SVD for 3x3 matrices.

This is an implementation of the method described in [Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations](http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf), adapted from
code by Eric Jang. This version of the library includes Python bindings for ease of use
and testing.
"""

import numpy as np
from ._c import svd as _svd


def svd(a):
    a = np.asarray(a)
    if a.shape != (3,3):
        raise ValueError("Input matrix does not have the correct shape.")
    return _svd(a)

def _bench(N = 1_000_000, N_REPS=100):
    import numpy as np
    import timeit
    from ._c import svd

    S_TO_US = 1e6

    def bench(fn):
        matrices = np.random.rand(N, 3, 3)
        def svd_all():
            for A in matrices:
                fn(A)
        return timeit.repeat(svd_all, number=1, repeat=N_REPS)

    for mode, fn in [("np ", np.linalg.svd), ("jen", svd)]:
        times = bench(fn)

        avg_times = [(t / N) * S_TO_US for t in times]
        mean = np.mean(avg_times)
        stdev = np.std(avg_times)

        print(f"Average SVD time {mode}: {mean:.4f} ± {stdev:.4f} μs")
