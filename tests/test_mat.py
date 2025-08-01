import pytest
import warnings
import rowan
from hypothesis import given, settings, HealthCheck, strategies as st
from conftest import generate_random_matrixes, nonsingular_3x3_matrices
import numpy as np
from svd3x3._c import mul_a_b, mul_at_b, svd, rsqrt, q2mat3, norm2, qr

# TODO: generate meaningful test matrixes
N = 20
ATOL = 1e-12


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


@pytest.mark.parametrize("a", generate_random_matrixes(N**2))
def test_qr(a):
    q_ref, r_ref = np.linalg.qr(a[:], mode="complete")
    q, r = qr(a[:])

    # Validate properties: q should be mutually perpendicular unit vectors
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1)
    np.testing.assert_allclose(np.cross(q[0, :], q[1, :]), q[2, :])
    np.testing.assert_allclose(np.cross(q[1, :], q[2, :]), q[0, :])
    np.testing.assert_allclose(np.cross(q[2, :], q[0, :]), q[1, :])

    # r should be upper triangular
    np.testing.assert_allclose(r, np.triu(r), atol=ATOL)

    np.testing.assert_allclose(q, q_ref)
    np.testing.assert_allclose(r, r_ref)


@pytest.mark.parametrize("a", generate_random_matrixes(N**2))
def test_svd(a):
    ref_u, ref_s, ref_v = np.linalg.svd(a[:])
    u, s, v = svd(a[:])
    np.testing.assert_allclose(u, ref_u)
    np.testing.assert_allclose(s.round(13), np.diag(ref_s))
    np.testing.assert_allclose(v, ref_v)
