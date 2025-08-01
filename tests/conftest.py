import numpy as np
from hypothesis.strategies import composite, floats
import numpy as np
from hypothesis.extra.numpy import arrays

RNG = np.random.default_rng(seed=0)


def generate_random_matrixes(n=1000, s=(3, 3)):
    buffer_fraction = 0.25
    singular_tolerance = 1e-8
    sample_n = int((buffer_fraction + 1) * n)
    m = RNG.uniform(size=(sample_n, *s)) - 0.5
    m *= RNG.uniform(-100, 100, size=sample_n)[:, None, None]
    m = m[np.abs(np.linalg.det(m)) > singular_tolerance][:n]
    assert len(m) == n
    return m


@composite
def nonsingular_3x3_matrices(draw):
    while True:
        matrix = draw(arrays(np.float64, (3, 3), elements=floats(-10, 10)))
        if np.linalg.matrix_rank(matrix) == 3:
            return matrix
