/**************************************************************************
**
**  svd3
**
**  Quick singular value decomposition as described by:
**  A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
**  "Computing the Singular Value Decomposition of 3x3 matrices
**  with minimal branching and elementary floating point operations",
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
**  OPTIMIZED CPU VERSION
**  Implementation by: Eric Jang
**
**  13 Apr 2014
**
**  Updated by: Jenna Bradley
**
**  01 Aug 2025
**************************************************************************/

#pragma once

#include <cmath>
#include <cstdio>
#include <utility>
#define _gamma 5.82842712474619010 /* 3 + sqrt(8) = 3 + 2 * sqrt(2) */
#define _cstar 0.92387953251128676 /* cos(π/8) */
#define _sstar 0.38268343236508977 /* sin(π/8) */
// #define EPSILON 1e-6
#define EPSILON 1e-13

#include <math.h>


inline void printMat3(double a[3][3]) {

  double a11 = a[0][0];
  double a12 = a[0][1];
  double a13 = a[0][2];
  double a21 = a[1][0];
  double a22 = a[1][1];
  double a23 = a[1][2];
  double a31 = a[2][0];
  double a32 = a[2][1];
  double a33 = a[2][2];
  printf("[[%.16f, %.16f, %.16f], \n", a11, a12, a13);
  printf(" [%.16f, %.16f, %.16f], \n", a21, a22, a23);
  printf(" [%.16f, %.16f, %.16f]] \n", a31, a32, a33);
}

/**
 * @brief Compute the inverse square root of a number.
 *
 * Although the fast inverse square root algorithm is absolutely deserving of
 * its fame, on modern architectures it simply does not make sense. Regardless
 * of its precision (which for this algorithm does not matter, see section 2.4
 * of the original paper), fast rsqrt is never actually faster than sqrt[f|d] on
 * modern hardware. The paper
 * [here](https://apps.dtic.mil/sti/pdfs/AD1014930.pdf), from 2016, indicates
 * this was not the case for single precision at the time, so it is possible
 * that a performance gain is still possible on some hardware. For now, we use
 * the simple formula.
 */
inline double rsqrt(double x) { return 1.0 / sqrt(x); }

/**
 * @brief Compute the inverse square root of a number.
 *
 * @param c[in] A boolean indicating whether the values should be swapped
 * @param X[in, out] The first value to consider swapping
 * @param Y[in, out] The second value to consider swapping
 */
inline void condSwap(bool c, double &X, double &Y) {
  // used in step 2
  double Z = X;
  X = c ? Y : X;
  Y = c ? Z : Y;
} // TODO: replace with std::swap

inline void condNegSwap(bool c, double &X, double &Y) {
  // used in step 2 and 3
  double Z = -X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}

// matrix multiplication M = A * B
inline void matmul(const double a[3][3], const double b[3][3], double m[3][3]) {
  m[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
  m[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
  m[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
  m[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
  m[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
  m[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
  m[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
  m[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
  m[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}
// matrix multiplication M = A.T * B
inline void matTmul(const double a[3][3], const double b[3][3],
                    double m[3][3]) {
  m[0][0] = a[0][0] * b[0][0] + a[1][0] * b[1][0] + a[2][0] * b[2][0];
  m[0][1] = a[0][0] * b[0][1] + a[1][0] * b[1][1] + a[2][0] * b[2][1];
  m[0][2] = a[0][0] * b[0][2] + a[1][0] * b[1][2] + a[2][0] * b[2][2];
  m[1][0] = a[0][1] * b[0][0] + a[1][1] * b[1][0] + a[2][1] * b[2][0];
  m[1][1] = a[0][1] * b[0][1] + a[1][1] * b[1][1] + a[2][1] * b[2][1];
  m[1][2] = a[0][1] * b[0][2] + a[1][1] * b[1][2] + a[2][1] * b[2][2];
  m[2][0] = a[0][2] * b[0][0] + a[1][2] * b[1][0] + a[2][2] * b[2][0];
  m[2][1] = a[0][2] * b[0][1] + a[1][2] * b[1][1] + a[2][2] * b[2][1];
  m[2][2] = a[0][2] * b[0][2] + a[1][2] * b[1][2] + a[2][2] * b[2][2];
}

inline void quatToMat3(const double *qV, double m[3][3]) {
  double w = qV[3];
  double x = qV[0];
  double y = qV[1];
  double z = qV[2];

  double qxx = x * x;
  double qyy = y * y;
  double qzz = z * z;
  double qxz = x * z;
  double qxy = x * y;
  double qyz = y * z;
  double qwx = w * x;
  double qwy = w * y;
  double qwz = w * z;

  m[0][0] = 1 - 2 * (qyy + qzz);
  m[0][1] = 2 * (qxy - qwz);
  m[0][2] = 2 * (qxz + qwy);
  m[1][0] = 2 * (qxy + qwz);
  m[1][1] = 1 - 2 * (qxx + qzz);
  m[1][2] = 2 * (qyz - qwx);
  m[2][0] = 2 * (qxz - qwy);
  m[2][1] = 2 * (qyz + qwx);
  m[2][2] = 1 - 2 * (qxx + qyy);
}

inline void approxGivensAngle(double a11, double a12, double a22, double &ch,
                              double &sh) {
  // if ((a22 - a11) < EPSILON) {
  //   ch = sh = sqrt(2);
  //   return;
  // }
  // double theta = std::atan(2 * a12 / (a11 - a22)) / 2;

  // ch = std::cos(theta);
  // sh = std::sin(theta);
  double sq = (a11 - a22) * (a11 - a22);
  bool b = (a12 * a12 < sq);
  double omega = rsqrt(a12 * a12 + sq);
  ch = b ? omega * a12 : 0.70710678118654752;
  sh = b ? omega * (a11 - a22) : 0.70710678118654752;
}

/**
 * @brief Approximate the quaternion representing a 2x2 Givens rotation
 *
 * @param a11[in] The upper left of the Givens rotation
 * @param a12[in] The upper right of the Givens rotation
 * @param a22[in] The lower right of the Givens rotation
 * @param ch,sh[out] Approximations for the unnormalized Givens quaternion elements.
 */
inline void approximateGivensQuaternion(double a11, double a12, double a22,
                                        double &ch, double &sh) {

  // TODO: This seems to be wrong? Did I break something?
  ch = 2 * (a11 - a22);
  sh = a12;
  bool b = _gamma * sh * sh < ch * ch;
  double ω = rsqrt(ch * ch + sh * sh); // Normalize the vector [c^2, s^2]
  ch = b ? ω * ch : (double)_cstar;
  sh = b ? ω * sh : (double)_sstar;
}

inline void jacobiConjugation(const int x, const int y, const int z,
                              double &s11, double &s21, double &s22,
                              double &s31, double &s32, double &s33,
                              double *qV) {
  double ch, sh;
  approximateGivensQuaternion(s11, s21, s22, ch, sh);

  double scale = ch * ch + sh * sh;
  double a = (ch * ch - sh * sh) / scale;
  double b = (2 * sh * ch) / scale;

  // make temp copy of S
  double _s11 = s11;
  double _s21 = s21;
  double _s22 = s22;
  double _s31 = s31;
  double _s32 = s32;
  double _s33 = s33;

  // perform conjugation S = Q'*S*Q
  // Q already implicitly solved from a, b
  s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
  s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
  s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
  s31 = a * _s31 + b * _s32;
  s32 = -b * _s31 + a * _s32;
  s33 = _s33;

  // update cumulative rotation qV
  double tmp[3];
  tmp[0] = qV[0] * sh;
  tmp[1] = qV[1] * sh;
  tmp[2] = qV[2] * sh;
  sh *= qV[3];

  qV[0] *= ch;
  qV[1] *= ch;
  qV[2] *= ch;
  qV[3] *= ch;

  // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
  // for (p,q) = ((0,1),(1,2),(0,2))
  qV[z] += sh;
  qV[3] -= tmp[z]; // w
  qV[x] += tmp[y];
  qV[y] -= tmp[x];

  // re-arrange matrix for next iteration
  _s11 = s22;
  _s21 = s32;
  _s22 = s33;
  _s31 = s21;
  _s32 = s31;
  _s33 = s11;
  s11 = _s11;
  s21 = _s21;
  s22 = _s22;
  s31 = _s31;
  s32 = _s32;
  s33 = _s33;
}

inline double dist2(double x, double y, double z) {
  return x * x + y * y + z * z;
}

/**
 * @brief Find the quaternion that diagonalizes a symmetric matrix
 *
 *
 * @param s[in, out] Input symmetric 3x3 matrix.
 * @param qV[out]    The quaternion that diagonalizes the input matrix.
 */
inline void jacobiEigenanalysis(
    // We don't need the full matrix, but we pass it for clarity
    double s[3][3], double *qV) {
  // xyzw convention
  qV[0] = 0;
  qV[1] = 0;
  qV[2] = 0;
  qV[3] = 1;
  for (int i = 0; i < 4; i++) {
    // we wish to eliminate the maximum off-diagonal element
    // on every iteration, but cycling over all 3 possible rotations
    // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
    //  asymptotic convergence
    jacobiConjugation(0, 1, 2, s[0][0], s[1][0], s[1][1], s[2][0], s[2][1],
                      s[2][2], qV); // p,q = 0,1
    jacobiConjugation(1, 2, 0, s[0][0], s[1][0], s[1][1], s[2][0], s[2][1],
                      s[2][2], qV); // p,q = 1,2
    jacobiConjugation(2, 0, 1, s[0][0], s[1][0], s[1][1], s[2][0], s[2][1],
                      s[2][2], qV); // p,q = 0,2
  }
}

inline void sortSingularValues(double b[3][3], double v[3][3]) {
  // TODO: should be able to standard swap entire rows!
  // Let's do this and add tests to this fn
  // std::swap(b[0], v[0]);
  double rho1 = dist2(b[0][0], b[1][0], b[2][0]);
  double rho2 = dist2(b[0][1], b[1][1], b[2][1]);
  double rho3 = dist2(b[0][2], b[1][2], b[2][2]);
  bool c;
  c = rho1 < rho2;
  condNegSwap(c, b[0][0], b[0][1]);
  condNegSwap(c, v[0][0], v[0][1]);
  condNegSwap(c, b[1][0], b[1][1]);
  condNegSwap(c, v[1][0], v[1][1]);
  condNegSwap(c, b[2][0], b[2][1]);
  condNegSwap(c, v[2][0], v[2][1]);
  condSwap(c, rho1, rho2);
  c = rho1 < rho3;
  condNegSwap(c, b[0][0], b[0][2]);
  condNegSwap(c, v[0][0], v[0][2]);
  condNegSwap(c, b[1][0], b[1][2]);
  condNegSwap(c, v[1][0], v[1][2]);
  condNegSwap(c, b[2][0], b[2][2]);
  condNegSwap(c, v[2][0], v[2][2]);
  condSwap(c, rho1, rho3);
  c = rho2 < rho3;
  condNegSwap(c, b[0][1], b[0][2]);
  condNegSwap(c, v[0][1], v[0][2]);
  condNegSwap(c, b[1][1], b[1][2]);
  condNegSwap(c, v[1][1], v[1][2]);
  condNegSwap(c, b[2][1], b[2][2]);
  condNegSwap(c, v[2][1], v[2][2]);
}

inline void QRGivensQuaternion(double a1, double a2, double &ch, double &sh) {
  // a1 = pivot point on diagonal
  // a2 = lower triangular entry we want to annihilate
  double epsilon = (double)EPSILON;
  double rho = sqrt(a1 * a1 + a2 * a2);

  sh = rho > epsilon ? a2 : 0;
  ch = abs(a1) + fmax(rho, epsilon);
  bool b = a1 < 0;
  condSwap(b, sh, ch);
  double w = rsqrt(ch * ch + sh * sh);
  ch *= w;
  sh *= w;
}

inline void QRDecomposition(double b[3][3], double q[3][3], double r[3][3]) {

  double ch1, sh1, ch2, sh2, ch3, sh3;
  double aa, bb;

  // first givens rotation (ch,0,0,sh)
  QRGivensQuaternion(b[0][0], b[1][0], ch1, sh1);
  aa = 1 - 2 * sh1 * sh1;
  bb = 2 * ch1 * sh1;
  // apply B = Q' * B
  r[0][0] = aa * b[0][0] + bb * b[1][0];
  r[0][1] = aa * b[0][1] + bb * b[1][1];
  r[0][2] = aa * b[0][2] + bb * b[1][2];
  r[1][0] = -bb * b[0][0] + aa * b[1][0];
  r[1][1] = -bb * b[0][1] + aa * b[1][1];
  r[1][2] = -bb * b[0][2] + aa * b[1][2];
  r[2][0] = b[2][0];
  r[2][1] = b[2][1];
  r[2][2] = b[2][2];

  // second givens rotation (ch,0,-sh,0)
  QRGivensQuaternion(r[0][0], r[2][0], ch2, sh2);
  aa = 1 - 2 * sh2 * sh2;
  bb = 2 * ch2 * sh2;
  // apply B = Q' * B;
  double qprimeb[3][3];
  qprimeb[0][0] = aa * r[0][0] + bb * r[2][0];
  qprimeb[0][1] = aa * r[0][1] + bb * r[2][1];
  qprimeb[0][2] = aa * r[0][2] + bb * r[2][2];
  qprimeb[1][0] = r[1][0];
  qprimeb[1][1] = r[1][1];
  qprimeb[1][2] = r[1][2];
  qprimeb[2][0] = -bb * r[0][0] + aa * r[2][0];
  qprimeb[2][1] = -bb * r[0][1] + aa * r[2][1];
  qprimeb[2][2] = -bb * r[0][2] + aa * r[2][2];

  // third givens rotation (ch,sh,0,0)
  QRGivensQuaternion(qprimeb[1][1], qprimeb[2][1], ch3, sh3);
  aa = 1 - 2 * sh3 * sh3;
  bb = 2 * ch3 * sh3;
  // R is now set to desired value
  r[0][0] = qprimeb[0][0];
  r[0][1] = qprimeb[0][1];
  r[0][2] = qprimeb[0][2];
  r[1][0] = aa * qprimeb[1][0] + bb * qprimeb[2][0];
  r[1][1] = aa * qprimeb[1][1] + bb * qprimeb[2][1];
  r[1][2] = aa * qprimeb[1][2] + bb * qprimeb[2][2];
  r[2][0] = -bb * qprimeb[1][0] + aa * qprimeb[2][0];
  r[2][1] = -bb * qprimeb[1][1] + aa * qprimeb[2][1];
  r[2][2] = -bb * qprimeb[1][2] + aa * qprimeb[2][2];

  // construct the cumulative rotation Q=Q1 * Q2 * Q3
  // the number of doubleing point operations for three quaternion
  // multiplications is more or less comparable to the explicit form of the
  // joined matrix. certainly more memory-efficient!
  double sh12 = sh1 * sh1;
  double sh22 = sh2 * sh2;
  double sh32 = sh3 * sh3;

  q[0][0] = (-1 + 2 * sh12) * (-1 + 2 * sh22);
  q[0][1] = 4 * ch2 * ch3 * (-1 + 2 * sh12) * sh2 * sh3 +
            2 * ch1 * sh1 * (-1 + 2 * sh32);
  q[0][2] = 4 * ch1 * ch3 * sh1 * sh3 -
            2 * ch2 * (-1 + 2 * sh12) * sh2 * (-1 + 2 * sh32);

  q[1][0] = 2 * ch1 * sh1 * (1 - 2 * sh22);
  q[1][1] = -8 * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 +
            (-1 + 2 * sh12) * (-1 + 2 * sh32);
  q[1][2] = -2 * ch3 * sh3 +
            4 * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1 + 2 * sh32));

  q[2][0] = 2 * ch2 * sh2;
  q[2][1] = 2 * ch3 * (1 - 2 * sh22) * sh3;
  q[2][2] = (-1 + 2 * sh22) * (-1 + 2 * sh32);
}

inline void svd(double a[3][3], double u[3][3], double s[3][3],
                double v[3][3]) {
  // normal equations matrix
  double ATA[3][3];

  matTmul(a, a, ATA); // TODO: gives same result?

  // symmetric eigenalysis
  double qV[4];
  jacobiEigenanalysis(ATA, qV);
  // printf("qV: [%f, %f, %f, %f]\n", qV[0], qV[1], qV[2], qV[3]); // qV matches prev
  // printMat3(ATA); // This also matches
  quatToMat3(qV, v);
  // printMat3(v); // v matches

  double b[3][3];
  matmul(a, v, b);
  
  printMat3(b);


  // sort singular values and find V
  sortSingularValues(b, v);

  // QR decomposition
  QRDecomposition(b, u, s); // ISSUE: we pass in 
}

/// polar decomposition can be reconstructed trivially from SVD result
// A = UP
// inline void pd(double a11, double a12, double a13, double a21, double a22,
// double a23,
//         double a31, double a32, double a33,
//         // output U
//         double &u11, double &u12, double &u13, double &u21, double &u22,
//         double &u23, double &u31, double &u32, double &u33,
//         // output P
//         double &p11, double &p12, double &p13, double &p21, double &p22,
//         double &p23, double &p31, double &p32, double &p33) {
//   double w11, w12, w13, w21, w22, w23, w31, w32, w33;
//   double s11, s12, s13, s21, s22, s23, s31, s32, s33;
//   double v11, v12, v13, v21, v22, v23, v31, v32, v33;

//   svd(a11, a12, a13, a21, a22, a23, a31, a32, a33, w11, w12, w13, w21, w22,
//   w23,
//       w31, w32, w33, s11, s12, s13, s21, s22, s23, s31, s32, s33, v11, v12,
//       v13, v21, v22, v23, v31, v32, v33);

//   // P = VSV'
//   double t11, t12, t13, t21, t22, t23, t31, t32, t33;
//   multAB(v11, v12, v13, v21, v22, v23, v31, v32, v33, s11, s12, s13, s21,
//   s22,
//          s23, s31, s32, s33, t11, t12, t13, t21, t22, t23, t31, t32, t33);

//   multAB(t11, t12, t13, t21, t22, t23, t31, t32, t33, v11, v21, v31, v12,
//   v22,
//          v32, v13, v23, v33, p11, p12, p13, p21, p22, p23, p31, p32, p33);

//   // U = WV'
//   multAB(w11, w12, w13, w21, w22, w23, w31, w32, w33, v11, v21, v31, v12,
//   v22,
//          v32, v13, v23, v33, u11, u12, u13, u21, u22, u23, u31, u32, u33);
// }
