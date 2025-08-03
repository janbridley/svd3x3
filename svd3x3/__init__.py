"""Fast SVD for 3x3 matrices.

This is an implementation of the method described in [Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations](http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf), adapted from
code by Eric Jang. This version of the library includes Python bindings for ease of use
and testing.
"""

import numpy as np
from numpy.typing import ArrayLike
from ._c import svd as _svd
from ._bench import _bench


def svd3(a: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the singular value decomposition `U @ S @ VT` of a 3x3 matrix.

    The calculation is performed in double precision, and is typically ~65% faster than
    the same operation using Numpy.
    """
    a = np.asarray(a)
    if a.shape != (3, 3):
        raise ValueError("Input matrix does not have the correct shape.")
    return _svd(a)


__all__ = ["svd3", "_bench"]
