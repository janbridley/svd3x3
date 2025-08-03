from copy import deepcopy
import pytest
import warnings
import rowan
from hypothesis import given, settings, HealthCheck, strategies as st
from conftest import generate_random_matrixes, nonsingular_3x3_matrices
import numpy as np
from svd3x3._c import (
    mul_a_b,
    mul_at_b,
    svd,
    rsqrt,
    q2mat3,
    norm2,
    qr,
    jacobi_eigenanalysis,
)

# TODO: generate meaningful test matrixes
N = 20
ATOL = 1e-12
ATOL_TIGHT = 4e-16


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


@pytest.mark.parametrize("a", generate_random_matrixes(N**2))
def test_jacobi_eigenanalysis(a):
    b = deepcopy(a)
    q_diag = jacobi_eigenanalysis(a)

    # Validate b != a (a should be diagonalized in place)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(b, a)
        assert not np.allclose(np.diag(np.diag(a)), a)

    # a should now be diagonalized
    np.testing.assert_allclose(a,np.diag(np.diag(a)), atol=ATOL)

# @pytest.mark.parametrize("a", generate_random_matrixes(N**2))
# def test_svd(a):
#     b = deepcopy(a)
#     ref_u, ref_s, ref_v = np.linalg.svd(a)
#     U, S, V = svd(a)
#     np.testing.assert_array_equal(a, b)
#     # Validate we've sorted our singular values in descending order
#     np.testing.assert_array_equal(np.diag(S), sorted(np.diag(S))[::-1])
#     np.testing.assert_allclose(U, ref_u)
#     np.testing.assert_allclose(S.round(13), np.diag(ref_s))
#     np.testing.assert_allclose(V, ref_v)
