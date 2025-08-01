#include "../svd3.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <tuple>

namespace nb = nanobind;

using namespace nb::literals;

using Matrix33 = nb::ndarray<double, nb::numpy, nb::shape<3, 3>>;

NB_MODULE(_c, m) {
  m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
  m.def("rsqrt", &rsqrt);
  m.def("mul_a_b", [](const Matrix33 a, const Matrix33 b) {
    double m[3][3];
    matmul(a.data(), b.data(), m);
    return Matrix33(m).cast();
  });
  m.def("mul_at_b", [](const Matrix33 a, const Matrix33 b) {
    double m[3][3];
    matTmul(a.data(), b.data(), m);
    return Matrix33(m).cast();
  });
  m.def(
      "svd",
      [](const Matrix33 a) -> std::tuple<Matrix33, Matrix33, Matrix33> {
        // TODO: remove, this is just for debugging
        double u[3][3] = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        double s[3][3] = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        double v[3][3] = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        svd(a.data(), u, s, v);
        return std::make_tuple(Matrix33(u), Matrix33(s), Matrix33(v));
      },
      nb::rv_policy::automatic);
}
