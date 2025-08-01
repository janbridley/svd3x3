"""Fast SVD for 3x3 matrices.

This is an implementation of the method described in [Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations](http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf), adapted from
code by Eric Jang. This version of the library includes Python bindings for ease of use
and testing.
"""

import math, numpy as np
from ._c import add, rsqrt

np.testing.assert_allclose(rsqrt(2.1), 1 / math.sqrt(2.1))
