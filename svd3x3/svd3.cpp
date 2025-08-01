#include "../svd3.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <tuple>

namespace nb = nanobind;

using namespace nb::literals;

using Matrix33d = nb::ndarray<double, nb::numpy, nb::shape<3, 3>>;
using Quaternion = nb::ndarray<double, nb::numpy, nb::shape<4>>;

NB_MODULE(_c, m) {
  m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
  m.def("rsqrt", &rsqrt);
  m.def("mul_a_b", [](const Matrix33d a, const Matrix33d b) {
    double m[3][3];
    matmul(reinterpret_cast<double(*)[3]>(a.data()),
           reinterpret_cast<double(*)[3]>(b.data()), m);
    return Matrix33d(m).cast();
  });
  m.def("mul_at_b", [](const Matrix33d a, const Matrix33d b) {
    double m[3][3];
    matTmul(reinterpret_cast<double(*)[3]>(a.data()),
            reinterpret_cast<double(*)[3]>(b.data()), m);
    return Matrix33d(m).cast();
  });
  m.def("q2mat3", [](const Quaternion q) {
    double m[3][3];
    quatToMat3(q.data(), m);
    return Matrix33d(m).cast();
  });
  m.def(
      "svd",
      [](const Matrix33d a) -> std::tuple<Matrix33d, Matrix33d, Matrix33d> {
        // TODO: remove, this is just for debugging
        double u[3][3] = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        double s[3][3] = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        double v[3][3] = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        svd(reinterpret_cast<double(*)[3]>(a.data()), u, s, v);
        return std::make_tuple(Matrix33d(u), Matrix33d(s), Matrix33d(v));
      },
      nb::rv_policy::automatic);
}
