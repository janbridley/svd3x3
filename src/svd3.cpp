#include "svd3.hpp"
#include <array>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <tuple>

namespace nb = nanobind;

using namespace nb::literals;

using Vector3f = nb::ndarray<float, nb::numpy, nb::shape<3>>;
using Vector9d = nb::ndarray<double, nb::numpy, nb::shape<9>>;
using NBMatrix33 = nb::ndarray<double, nb::numpy, nb::shape<3, 3>>;

namespace wrap {


// std::tuple<Matrix33, Matrix33, Matrix33> svd(const Matrix33 a) {
void pysvd(const NBMatrix33 &a) {
  // NBMatrix33 u;
  // NBMatrix33 s;
  // NBMatrix33 v;
  printf("Input (%zu, %zu):\n", a.shape(0), a.shape(1));
  for (int i = 0; i < a.shape(0); ++i) {
    for (int j = 0; j < a.shape(1); ++j) {
      printf("%f ", a(j, i));
      // printf("%d, %d", j, i);
    }
    printf("\n");
  }
  // svd( // input A
  //      a(0, 0),  a(0, 1),  a(0, 2),  a(1, 0),  a(1, 1),  a(1, 2),
  //      a(2, 0),  a(2, 1),  a(2, 2),
  //      u(0, 0),  u(0, 1),  u(0, 2),  u(1, 0),  u(1, 1),
  //      u(1, 2),  u(2, 0),  u(2, 1),  u(2, 2),
  //      s(0, 0),  s(0, 1),  s(0, 2),  s(1, 0),  s(1, 1),
  //      s(1, 2),  s(2, 0),  s(2, 1),  s(2, 2),
  //      v(0, 0),  v(0, 1),  v(0, 2),  v(1, 0),  v(1, 1),
  //      v(1, 2),  v(2, 0),  v(2, 1),  v(2, 2)
  //     // a.data(), u.data(), s.data(), v.data()
  // );
  // printf("Matrix U:\n");
  // printf("U (%zu, %zu):\n", u.shape(0), u.shape(1));
  // for (int i = 0; i < u.shape(0); ++i) {
  //   for (int j = 0; j < u.shape(1); ++j) {
  //     printf("%f ", u(j, i));
  //     // printf("%d, %d", j, i);
  //   }
  // }
}

}; // End namespace wrap

NB_MODULE(_c, m) {
  m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
  m.def("rsqrt", &rsqrt);
  // m.def("svd", &wrap::pysvd);
  // m.def("matmul", &wrap::pymatmul);
  m.def("return_m33", [] {
    double data[3][3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    return NBMatrix33(data).cast();
  });
  m.def("take_and_return_m33", [](const NBMatrix33 inpt) {
    double data[3][3];
    inpt(1, 1) += 3.0; // MODIFIES INOUPT
    // return NBMatrix33(inpt.data()).cast();
    return NBMatrix33(data).cast();
  });
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
