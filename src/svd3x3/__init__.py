"""Fast SVD for 3x3 matrices.

This is an implementation of the method described in [Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations](http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf), adapted from
code by Eric Jang. This version of the library includes Python bindings for ease of use
and testing.
"""

# import _c
import math, numpy as np
from ._c import add, rsqrt,  return_m33, mul_a_b

print(add(2, 3))
np.testing.assert_allclose(rsqrt(2.1), 1 / math.sqrt(2.1))

u, s, v = np.eye(3)*2, np.ones((3, 3)), np.eye(3)
# print(svd)
# print(
#     svd(
#         np.eye(3)
#         # a11,a12,a13,a21,a22,a23,a31,a32,a33,
#         # u11,u12,u13,u21,u22,u23,u31,u32,u33,# // output S
#         # s11,s12,s13,s21,s22,s23,s31,s32,s33,# // output V
#         # v11,v12,v13,v21,v22,v23,v31,v32,v33 # // output U
#     )
# )
# print(np.array([u11,u12,u13,u21,u22,u23,u31,u32,u33]).reshape(3,3))
# print(np.array([s11,s12,s13,s21,s22,s23,s31,s32,s33]).reshape(3,3))
# print(np.array([v11,v12,v13,v21,v22,v23,v31,v32,v33]).reshape(3,3))

# print(s)
# print(matmul(u, s))
# print(s)
print(return_m33())
print()
print()
print(s)
print(u)
print()
# print(take_and_return_m33(u))
print(mul_a_b(s, u))
