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


def generate_point_set(n=4096):
    y = RNG.uniform(-10, 10, (n, 3))
    x = y + RNG.normal(loc=RNG.uniform(-10, 10), scale=RNG.uniform(0, 1))
    return x, y


def generate_crosscorrelation_matrixes(n=1000):
    mats = []
    for _ in range(n):
        x, y = generate_point_set()

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


def kabsch_umeyama(x, y, umeyama=True, svd=np.linalg.svd):
    N = len(y)
    tx, ty = x.mean(axis=0), y.mean(axis=0)
    x0, y0 = x - tx, y - ty

    # Compute cross-covariance between test and reference sets
    H = (x0.T @ y0) / (N if umeyama else 1)

    # Decompose singular values and reconstruct rotation and shear matrixes
    U, Σ, Vt = svd(H)

    if Σ.shape != (3, 3):
        Σ = np.diag(Σ)
    else:
        Vt = Vt.T

    d = np.sign(np.linalg.det(U @ Vt))
    S = np.diag([1, 1, d])
    R = (U @ S @ Vt).T

    # Compute scale factor to align sets
    if umeyama:
        σx = np.einsum("ij, ij->", x0, x0) / N
        c = np.dot(Σ, [1, 1, d]) / σx  # Equivalent to tr(diag(Σ)@S) # TODO: accruact?
    else:
        c = 1

    t = ty - c * R @ tx

    return (R, t, c, d)


def compute_rmsd(x, y, axis: int | None = None, normalize=False):
    """Compute the root mean squared deviation between to matrixes.

    Source: https://userguide.mdanalysis.org/stable/examples/analysis/alignment_and_rms/rmsd.html#Background
    """
    assert len(x) == len(y), f"len(x) = {len(x)}, len(y) = {len(y)}"
    rmsd = np.sqrt(np.square(x - y).sum(axis=axis) / len(x))
    return rmsd if not normalize else (rmsd / np.square(x - y.mean(axis=0)).sum())
