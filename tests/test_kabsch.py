import pytest
import rowan
import numpy as np
from conftest import RNG, generate_point_set, kabsch_umeyama, compute_rmsd
from svd3x3 import svd3

ATOL_LOOSE = 1e-3
N_KABSCH = 2_000
ANGLE_TOLERANCE = np.deg2rad(0.005)
np.random.seed(0)


@pytest.mark.parametrize("pts", [generate_point_set() for _ in range(N_KABSCH)])
@pytest.mark.parametrize("umeyama", [False, True], ids=["no_umeyama", "umeyama"])
def test_kabsch_unaligned(pts, umeyama):
    x, y = pts
    R, t, c, _ = kabsch_umeyama(x, y, umeyama=umeyama, svd=svd3)
    R_ref, t_ref, c_ref, _ = kabsch_umeyama(x, y, umeyama=umeyama, svd=np.linalg.svd)

    np.testing.assert_allclose(t, t_ref, atol=ATOL_LOOSE)
    np.testing.assert_allclose(c, c_ref, ATOL_LOOSE)
    np.testing.assert_allclose(
        compute_rmsd(c * (x - t) @ R, y),
        compute_rmsd(c_ref * (x - t_ref) @ R_ref, y),
        atol=ATOL_LOOSE,
    )

    assert (
        rowan.geometry.sym_intrinsic_distance(
            rowan.from_matrix(R), rowan.from_matrix(R_ref)
        )
        < ANGLE_TOLERANCE
    )


@pytest.mark.parametrize("pts", [generate_point_set() for _ in range(N_KABSCH)])
@pytest.mark.parametrize("umeyama", [False, True], ids=["no_umeyama", "umeyama"])
def test_kabsch_aligned(pts, umeyama):
    y, _ = pts

    q = rowan.random.rand()
    x = rowan.rotate(q, y + RNG.uniform(-10, 10, size=3))
    R, t, c, _ = kabsch_umeyama(x, y, umeyama=umeyama, svd=svd3)
    R_ref, t_ref, c_ref, _ = kabsch_umeyama(x, y, umeyama=umeyama, svd=np.linalg.svd)

    np.testing.assert_allclose(t, t_ref, atol=ATOL_LOOSE)
    np.testing.assert_allclose(c, c_ref, ATOL_LOOSE)
    np.testing.assert_allclose(
        compute_rmsd(c * (x - t) @ R, y),
        compute_rmsd(c_ref * (x - t_ref) @ R_ref, y),
        atol=ATOL_LOOSE,
    )

    assert (
        rowan.geometry.sym_intrinsic_distance(
            rowan.from_matrix(R), rowan.from_matrix(R_ref)
        )
        < ANGLE_TOLERANCE
    )
