#include "svd3.hpp"
#include <array>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <tuple>

namespace nb = nanobind;

using namespace nb::literals;

using NBMatrix33 = nb::ndarray<double, nb::numpy, nb::shape<3, 3>>;

NB_MODULE(_c, m) {
  m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
  m.def("rsqrt", &rsqrt);
  m.def("mul_a_b", [](const NBMatrix33 a, const NBMatrix33 b) {
    double m[3][3];
    matmul(a.data(), b.data(), m);
    return NBMatrix33(m).cast();
  });
  m.def("mul_at_b", [](const NBMatrix33 a, const NBMatrix33 b) {
    double m[3][3];
    matTmul(a.data(), b.data(), m);
    return NBMatrix33(m).cast();
  });
}
