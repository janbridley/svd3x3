from copy import deepcopy
from sympy import Matrix, atan2, cos, sin, simplify, symbols
import pytest
import warnings
import rowan
from hypothesis import given, settings, HealthCheck, strategies as st
from conftest import generate_random_matrixes, nonsingular_3x3_matrices, rotmat2x2
import numpy as np
from svd3x3._c import (
    mul_a_b,
    mul_at_b,
    rsqrt,
    q2mat3,
    norm2,
    qr,
    jacobi_eigenanalysis,
    # approximate_givens_quat,
)
from svd3x3 import (
    svd3,
)

# TODO: generate meaningful test matrixes
N = 20
ATOL = 1e-12
ATOL_SP = 1e-6
ATOL_TIGHT = 4e-16
GAMMA = 3 + 2 * np.sqrt(2)
CSTAR = np.cos(np.pi / 8)
SSTAR = np.cos(np.pi / 8)

QUAT_LOWER_DISTANCE_TOL = np.pi / 8 + ATOL_TIGHT
QUAT_UPPER_DISTANCE_TOL = np.pi - (np.pi / 8 + ATOL_TIGHT)


@given(st.floats(min_value=0.0))
def test_rsqrt(x):
    warnings.filterwarnings(
        action="ignore",
        message="divide by zero encountered in scalar divide",
        category=RuntimeWarning,
    )
    np.testing.assert_allclose(rsqrt(x), 1 / np.sqrt(x))


@given(st.floats(), st.floats(), st.floats())
def test_dist_squared(x, y, z):
    np.testing.assert_allclose(norm2(x, y, z), np.einsum("i,i->", [x, y, z], [x, y, z]))


@pytest.mark.parametrize("q", rowan.random.rand(N**2))
def test_q_to_mat(q):
    q_xyzw = np.array([*q[1:], q[0]])
    np.testing.assert_allclose(q2mat3(q_xyzw), rowan.to_matrix(q))


@pytest.mark.parametrize("a", generate_random_matrixes(N))
@pytest.mark.parametrize("b", generate_random_matrixes(N))
def test_matmul(a, b):
    np.testing.assert_allclose(mul_a_b(a[:], b[:]), a @ b)


@pytest.mark.parametrize("a", generate_random_matrixes(N))
@pytest.mark.parametrize("b", generate_random_matrixes(N))
def test_matmul_transposed(a, b):
    np.testing.assert_allclose(mul_at_b(a[:], b[:]), a.T @ b)


@pytest.mark.parametrize("a", generate_random_matrixes(N))
def test_aTa(a):
    b = deepcopy(a)
    np.testing.assert_allclose(mul_at_b(a[:], a[:]), a.T @ a)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("a", generate_random_matrixes(N**2))
def test_qr(a):
    b = deepcopy(a)
    Q, R = qr(a)
    np.testing.assert_array_equal(a, b)

    np.linalg.inv(a)  # Will raise LinAlgError if a is not invertible

    # Validate properties: q should be mutually perpendicular unit vectors (Q.T=inv(Q))
    np.testing.assert_allclose(Q.T, np.linalg.inv(Q))

    # r should be upper triangular
    np.testing.assert_allclose(R, np.triu(R), atol=ATOL)

    # Validate the matrixes themselves are a correct decomposition
    np.testing.assert_allclose(Q @ R, a)

    # Convert QR decomposition to unique form and compare to numpy
    def make_qr_unique(q, r):
        """Make a QR decomposition unique by asserting all(diag(R)>0)."""
        d = np.sign(np.diag(r))
        return (q @ np.diag(d), r * d[:, None])

    Q, R = make_qr_unique(Q, R)

    q_ref, r_ref = make_qr_unique(*np.linalg.qr(a))
    np.testing.assert_allclose(Q, q_ref, atol=ATOL)
    np.testing.assert_allclose(R, r_ref, atol=ATOL)


# @given(rotmat2x2())
# def test_approximate_givens_angle(angle_and_mat):
#     _, m = angle_and_mat
#     c, _, _, s = approximate_givens_angle(m)

#     # This is the condition we require for Jacobi iteration to converge (Section 2.1)
#     np.testing.assert_allclose(
#         c * s * (m[1, 1] - m[0, 0]) + (c**2 - s**2) * m[1, 0], 0, atol=ATOL_TIGHT
#     )


# @pytest.mark.parametrize("a", generate_random_matrixes(N**2))
# def test_jacobi_eigenanalysis(a):
#     b = deepcopy(a)
#     q_diag = jacobi_eigenanalysis(a)

#     # Validate b != a (a should be diagonalized in place)
#     with np.testing.assert_raises(AssertionError):
#         np.testing.assert_array_equal(b, a)
#         assert not np.allclose(np.diag(np.diag(a)), a)

#     # a should now be diagonalized
#     np.testing.assert_allclose(a, np.diag(np.diag(a)), atol=ATOL)


def test_svd_matches_original():
    m = np.array([
        [-0.558253, -0.0461681, -0.505735],
        [-0.411397, 0.0365854, 0.199707],
        [0.285389, -0.313789, 0.200189],
    ])

    # RESULTS USING FAST INVERSE SQUARE ROOT: S is not diagonal
    # u_ref = [
    #     [-0.847391, -0.343132, -0.387926],
    #     [-0.269368, 0.922143, -0.234785],
    #     [0.448557, -0.095353, -0.882110],
    # ]
    # s_ref = [
    #     [0.848985, -0.012012, 0.003219],
    #     [0.000416, 0.404733, -0.001072],
    #     [-0.001275, -0.000150, -0.293733],
    # ]
    # v_ref = [
    #     [0.814780, -0.521873, -0.208476],
    #     [-0.127013, 0.189629, -0.973017],
    #     [0.548463, 0.819371, 0.092894],
    # ]
    # u_ref = [
    #     [-0.849310, -0.354882, -0.390809],
    #     [-0.278376, 0.930100, -0.239627],
    #     [0.448531, -0.094725, -0.888734],
    # ]
    # s_ref = [
    #     [0.860883, -0.000000, 0.000000],
    #     [0.000000, 0.413613, -0.000000],
    #     [0.000000, -0.000000, -0.296320],
    # ]
    # v_ref = [
    #     [0.832469, -0.511493, -0.213002],
    #     [-0.129771, 0.193746, -0.972431],
    #     [0.538660, 0.837160, 0.094911],
    # ]
    # usv_ref = [
    #     [-0.558253, -0.046168, -0.505735],
    #     [-0.411397, 0.036585, 0.199707],
    #     [0.285389, -0.313789, 0.200189],
    # ]
    u_ref = [
        [-0.8493101885475047, -0.3548815183929819, -0.3908087886492650],
        [-0.2783757569859232, 0.9301001264726573, -0.2396262311997496],
        [0.4485302160939998, -0.0947252827035972, -0.8887336950477889],
    ]
    s_ref = [
        [0.8608828113372722, -0.0000000485205244, -0.0000000130628518],
        [-0.0000000182295021, 0.4136132119692853, 0.0000000003743927],
        [0.0000000112557583, -0.0000000027352566, -0.2963202037851793],
    ]
    v_ref = [
        [0.8324691649454764, -0.5114929566004334, -0.2130021948052509],
        [-0.1297706377407898, 0.1937463028621116, -0.9724309463980638],
        [0.5386599737577430, 0.8371602091022577, 0.0949110810207865],
    ]
    usv_ref = [
        [-0.5582530991794445, -0.0461680664427804, -0.5057350527163749],
        [-0.4113970118077264, 0.0365854019305501, 0.1997069505182104],
        [0.2853889783740473, -0.3137889412046463, 0.2001889901756999],
    ]

    u, s, v = svd3(m)

    # Validate our outputs are orthogonal
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=ATOL_SP)
    np.testing.assert_allclose(v.T @ v, np.eye(3), atol=ATOL_SP)

    np.testing.assert_allclose(np.array(v_ref).T @ v_ref, np.eye(3), atol=ATOL_SP)
    np.testing.assert_allclose(np.array(u_ref).T @ u_ref, np.eye(3), atol=ATOL_SP)
    np.testing.assert_allclose(s_ref, np.diag(np.diag(s_ref)), atol=ATOL_SP)
    np.testing.assert_allclose(usv_ref, m, atol=ATOL_SP)
    # np.testing.assert_allclose(np.array(u_ref) @ s_ref @ v_ref, usv_ref)

    # np.testing.assert_allclose(u, u_ref)
    np.testing.assert_allclose(s, s_ref, atol=ATOL_SP)
    # np.testing.assert_allclose(v, v_ref)


@pytest.mark.parametrize(
    "bad_m", [0.9, np.eye(2), np.eye(4), np.random.rand(3, 2), [[1]], None]
)
def test_svd3_raises(bad_m):
    with pytest.raises(ValueError):
        svd3(bad_m)
# TODO: generate real covariances, NOT random matrixes. This will be a better test of
# numerical stability
@pytest.mark.parametrize("a", generate_random_matrixes(1000))
def test_svd3(a):
    b = deepcopy(a)
    ref_u, ref_s, ref_v = np.linalg.svd(a)
    U, S, VT = svd3(a)
    np.testing.assert_array_equal(a, b)

    # Validate we've sorted our singular values in descending order
    np.testing.assert_array_equal(np.diag(S), sorted(np.diag(S))[::-1])

    # ISSUE: this result can be arbitrarily(?) bad?
    np.testing.assert_allclose(S, np.diag(np.diag(S)), atol=1e-1)

    # Validation checks - do we recover the input?
    np.testing.assert_allclose(U @ S @ VT.T, a)
    np.testing.assert_allclose(ref_u @ np.diag(ref_s) @ ref_v, a)

    # U and VT are orthogonal
    np.testing.assert_allclose(U.T @ U, np.eye(3), atol=ATOL)
    np.testing.assert_allclose(VT.T @ VT, np.eye(3), atol=ATOL)
