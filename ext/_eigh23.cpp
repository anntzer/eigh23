#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

// Formulas and notations from SelfAdjointEigenSolver.h's direct_selfadjoint_eigenvalues.

inline double sq(double x) {
  return x * x;
}

std::pair<py::array_t<double>, py::array_t<double>> eigh22(
  py::array_t<double> m00_, py::array_t<double> m11_, py::array_t<double> m01_)
{
  auto const& m00 = m00_.unchecked<1>(),
            & m11 = m11_.unchecked<1>(),
            & m01 = m01_.unchecked<1>();
  auto const n = m00.shape(0);
  if (n != m11.shape(0) || n != m01.shape(0)) {
    throw std::invalid_argument{"mismatched array sizes"};
  }
  auto eval_ = py::array_t<double>{{n, ssize_t{2}}};
  auto eval = eval_.mutable_data();
  auto evec_ = py::array_t<double>{{n, ssize_t{2}, ssize_t{2}}};
  auto evec = evec_.mutable_data();
  for (auto i = 0; i < n; ++i) {
    auto shift = 0.5 * (m00(i) + m11(i)),
         scale = std::max({std::abs(m00(i)), std::abs(m11(i)), std::abs(m01(i))}),
         invscale = scale ? 1. / scale : 0,
         a00 = invscale * (m00(i) - shift),
         a11 = invscale * (m11(i) - shift),
         a01 = invscale * m01(i),
         t0 = 0.5 * std::sqrt(sq(a00 - a11) + 4 * sq(a01)),
         t1 = 0.5 * (a00 + a11),
         e0 = t1 - t0,
         e1 = t1 + t0;
    *eval++ = scale * e0 + shift;
    *eval++ = scale * e1 + shift;
    if (t0 <= std::numeric_limits<double>::epsilon() * t1) {
      *evec++ = 1.; *evec++ = 0.;
      *evec++ = 0.; *evec++ = 1.;
    } else {
      a00 -= e1;
      a11 -= e1;
      auto a00_2 = a00 * a00,
           a11_2 = a11 * a11,
           a01_2 = a01 * a01;
      if (a00_2 > a11_2) {
        auto n = 1. / std::sqrt(a00_2 + a01_2);
        *evec++ = -a00 * n; *evec++ = -a01 * n;
        *evec++ = -a01 * n; *evec++ = a00 * n;
      } else {
        auto n = 1. / std::sqrt(a11_2 + a01_2);
        *evec++ = -a01 * n; *evec++ = -a11 * n;
        *evec++ = -a11 * n; *evec++ = a01 * n;
      }
    }
  }
  return {eval_, evec_};
}

using V3 = std::tuple<double, double, double>;

inline double dot(V3 a, V3 b)
{
  auto [a0, a1, a2] = a;
  auto [b0, b1, b2] = b;
  return a0 * b0 + a1 * b1 + a2 * b2;
}

inline V3 cross(V3 a, V3 b)
{
  auto [a0, a1, a2] = a;
  auto [b0, b1, b2] = b;
  return {a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0};
}

inline double normalize(V3& a)
{
  auto n2 = dot(a, a);
  if (n2) {
    auto& [a0, a1, a2] = a;
    auto f = 1. / std::sqrt(n2);
    a0 *= f;
    a1 *= f;
    a2 *= f;
  }
  return n2;
}

inline std::pair<V3, V3>
  extract_kernel(double a00, double a11, double a22, double a01, double a12, double a20)
{
  auto aa00 = std::abs(a00),
       aa11 = std::abs(a11),
       aa22 = std::abs(a22);
  auto col0 = V3{a00, a01, a20},
       col1 = V3{a01, a11, a12},
       col2 = V3{a20, a12, a22};
  V3 rep, c0, c1;
  if (aa00 >= aa11 && aa00 >= aa22) {  // Must be >=, not >, to handle tied cases!
    rep = col0;
    c0 = cross(col0, col1);
    c1 = cross(col0, col2);
  } else if (aa11 >= aa22 && aa11 >= aa00) {
    rep = col1;
    c0 = cross(col1, col2);
    c1 = cross(col1, col0);
  } else {
    rep = col2;
    c0 = cross(col2, col0);
    c1 = cross(col2, col1);
  }
  auto n0 = normalize(c0),
       n1 = normalize(c1);
  return {n0 > n1 ? c0 : c1, rep};
}

std::pair<py::array_t<double>, py::array_t<double>> eigh33(
  py::array_t<double> m00_, py::array_t<double> m11_, py::array_t<double> m22_,
  py::array_t<double> m01_, py::array_t<double> m12_, py::array_t<double> m20_)
{
  auto const& m00 = m00_.unchecked<1>(),
            & m11 = m11_.unchecked<1>(),
            & m22 = m22_.unchecked<1>(),
            & m01 = m01_.unchecked<1>(),
            & m12 = m12_.unchecked<1>(),
            & m20 = m20_.unchecked<1>();
  auto const n = m00.shape(0);
  if (n != m11.shape(0) || n != m22.shape(0)
      || n != m01.shape(0) || n != m12.shape(0) || n != m20.shape(0)) {
    throw std::invalid_argument{"mismatched array sizes"};
  }
  auto eval_ = py::array_t<double>{{n, ssize_t{3}}};
  auto eval = eval_.mutable_data();
  auto evec_ = py::array_t<double>{{n, ssize_t{3}, ssize_t{3}}};
  auto evec = evec_.mutable_data();
  for (auto i = 0; i < n; ++i) {
    auto shift = (m00(i) + m11(i) + m22(i)) / 3.,
         scale = std::max({std::abs(m00(i)), std::abs(m11(i)), std::abs(m22(i)),
                           std::abs(m01(i)), std::abs(m12(i)), std::abs(m20(i))}),
         invscale = scale ? 1. / scale : 0,
         a00 = invscale * (m00(i) - shift),
         a11 = invscale * (m11(i) - shift),
         a22 = invscale * (m22(i) - shift),
         a01 = invscale * m01(i),
         a12 = invscale * m12(i),
         a20 = invscale * m20(i),
         c0 = a00 * a11 * a22 + 2 * a01 * a12 * a20
            - a00 * a12 * a12 - a11 * a20 * a20 - a22 * a01 * a01,
         c1 = a00 * a11 - a01 * a01
            + a11 * a22 - a12 * a12
            + a22 * a00 - a20 * a20,
         c2 = a00 + a11 + a22,
         c2o3 = c2 * (1. / 3),
         ao3 = std::max((c2 * c2o3 - c1) * (1. / 3), 0.),
         hb = 0.5 * (c0 + c2o3 * (2 * c2o3 * c2o3 - c1)),
         q = std::max(ao3 * ao3 * ao3 - hb * hb, 0.),
         rho = std::sqrt(ao3),
         theta = std::atan2(std::sqrt(q), hb) * (1. / 3),
         cos = std::cos(theta),
         sin = std::sin(theta),
         e0 = c2o3 - rho * (cos + std::sqrt(3) * sin),
         e1 = c2o3 - rho * (cos - std::sqrt(3) * sin),
         e2 = c2o3 + 2 * rho * cos;
    *eval++ = scale * e0 + shift;
    *eval++ = scale * e1 + shift;
    *eval++ = scale * e2 + shift;
    if (e2 - e0 <= std::numeric_limits<double>::epsilon()) {
      *evec++ = 1.; *evec++ = 0.; *evec++ = 0.;
      *evec++ = 0.; *evec++ = 1.; *evec++ = 0.;
      *evec++ = 0.; *evec++ = 0.; *evec++ = 1.;
    } else {
      auto d12 = e2 - e1, d01 = e1 - e0;
      V3 col0, col2;
      if (d12 <= d01) {  // k = 0, l = 2
        auto k0k2 = extract_kernel(a00 - e0, a11 - e0, a22 - e0, a01, a12, a20);
        col0 = k0k2.first;
        if (d12 <= 2 * std::numeric_limits<double>::epsilon() * d01) {
          col2 = k0k2.second;  // FIXME I don't understand line 679 in eigen here.
          normalize(col2);
        } else {
          auto k2k0 = extract_kernel(a00 - e2, a11 - e2, a22 - e2, a01, a12, a20);
          col2 = k2k0.first;
        }
      } else {  // k = 2, l = 0
        auto k2k0 = extract_kernel(a00 - e2, a11 - e2, a22 - e2, a01, a12, a20);
        col2 = k2k0.first;
        if (d01 <= 2 * std::numeric_limits<double>::epsilon() * d12) {
          col0 = k2k0.second;  // FIXME I don't understand line 679 in eigen here.
          normalize(col0);
        } else {
          auto k0k2 = extract_kernel(a00 - e0, a11 - e0, a22 - e0, a01, a12, a20);
          col0 = k0k2.first;
        }
      }
      auto col1 = cross(col2, col0);
      normalize(col1);
      *evec++ = std::get<0>(col0);
      *evec++ = std::get<0>(col1);
      *evec++ = std::get<0>(col2);
      *evec++ = std::get<1>(col0);
      *evec++ = std::get<1>(col1);
      *evec++ = std::get<1>(col2);
      *evec++ = std::get<2>(col0);
      *evec++ = std::get<2>(col1);
      *evec++ = std::get<2>(col2);
    }
  }
  return {eval_, evec_};
}

PYBIND11_MODULE(_eigh23, m) {
    m
      .def("eigh22", eigh22, "m00"_a, "m11"_a, "m01"_a,
           R"__doc__(
Compute the eigenvalues and eigenvectors of ``N`` 2x2 real symmetric matrices.

Parameters
----------
m00, m11, m01 : (N,) arrays
    The corresponding elements of the matrices.

Returns
-------
eigenvalues : (N, 2) array
    The eigenvalues in ascending order.
eigenvectors : (N, 2, 2) array
    The eigenvectors: the columns of each 2x2 matrix are the eigenvectors, in
    the same order as the eigenvalues.
)__doc__")
      .def("eigh33", eigh33, "m00"_a, "m11"_a, "m22"_a, "m01"_a, "m12"_a, "m20"_a,
           R"__doc__(
Compute the eigenvalues and eigenvectors of ``N`` 3x3 real symmetric matrices.

Parameters
----------
m00, m11, m22, m01, m12, m20 : (N,) arrays
    The corresponding elements of the matrices.

Returns
-------
eigenvalues : (N, 3) array
    The eigenvalues in ascending order.
eigenvectors : (N, 3, 3) array
    The eigenvectors: the columns of each 3x3 matrix are the eigenvectors, in
    the same order as the eigenvalues.
)__doc__")
      ;
}
