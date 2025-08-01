from hypothesis import given, settings, HealthCheck
from conftest import nonsingular_3x3_matrices
import numpy as np

@settings(suppress_health_check=(HealthCheck.large_base_example,))
@given(nonsingular_3x3_matrices())
def test_svd(m):
    np.linalg.svd(m)
