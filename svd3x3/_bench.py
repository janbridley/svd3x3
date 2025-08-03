def _bench(N=500_000, N_REPS=20):
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
