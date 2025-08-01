import pytest
from hypothesis import given, settings, HealthCheck
from conftest import generate_random_matrixes, nonsingular_3x3_matrices
import numpy as np
from svd3x3._c import mul_a_b, mul_at_b, svd

N = 20


@pytest.mark.parametrize("a", generate_random_matrixes(N))
@pytest.mark.parametrize("b", generate_random_matrixes(N))
def test_matmul(a, b):
    np.testing.assert_allclose(mul_a_b(a[:], b[:]), a @ b)


@pytest.mark.parametrize("a", generate_random_matrixes(N))
@pytest.mark.parametrize("b", generate_random_matrixes(N))
def test_matmul_transposed(a, b):
    np.testing.assert_allclose(mul_at_b(a[:], b[:]), a.T @ b)

@pytest.mark.parametrize("a", generate_random_matrixes(N**2))
def test_svd(a):
    ref_u, ref_s, ref_v = np.linalg.svd(a[:])
    u, s, v = svd(a[:])
    # np.testing.assert_allclose(u, ref_u)
    np.testing.assert_allclose(s.round(13), np.diag(ref_s))
    # np.testing.assert_allclose(u, ref_u)
