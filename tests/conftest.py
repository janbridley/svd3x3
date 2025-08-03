import numpy as np
from hypothesis.strategies import composite, floats
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


def generate_crosscorrelation_matrixes(n=1000):
    mats = []
    for _ in range(n):
        y = RNG.uniform(-10, 10, (4096, 3))
        x = y + RNG.normal(loc=RNG.uniform(-10, 10), scale=RNG.uniform(0, 5))
        tx, ty = x.mean(axis=0), y.mean(axis=0).round(12)
        x0, y0 = x - tx, y - ty

        # Compute cross-covariance between test and reference sets
        H = (x0.T @ y0) / 4096
        mats.append(H)
    return np.array(mats)


@composite
def nonsingular_3x3_matrices(draw):
    while True:
        matrix = draw(arrays(np.float64, (3, 3), elements=floats(-10, 10)))
        if np.linalg.matrix_rank(matrix) == 3:
            return matrix


@composite
def rotmat2x2(draw):
    """NOTE: this uses the reduced range [-π/4, π/4] for compliance with the paper."""
    theta = draw(floats(min_value=-np.pi / 4, max_value=np.pi / 4))
    return (
        theta,
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
    )
