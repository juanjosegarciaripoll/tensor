// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#pragma once
#if !defined(TENSOR_TENSOR_H)
#error "This header cannot be included manually"
#endif
#ifndef TENSOR_DETAIL_TENSOR_OPS_HPP
#define TENSOR_DETAIL_TENSOR_OPS_HPP

#include <cassert>
#include <type_traits>
#include <algorithm>

namespace tensor {

//
// Unary operations
//
template <typename t>
Tensor<t> operator-(const Tensor<t> &a) {
  Tensor<t> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [](const t &x) { return -x; });
  return output;
}

//
// Binary operations
//

bool verify_tensor_dimensions_match(const Indices &d1, const Indices &d2);

//
// TENSOR <OP> TENSOR
//
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator+(const Tensor<t1> &a,
                                                          const Tensor<t2> &b) {
  // This should be: assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  assert(a.size() == b.size());
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](const t1 &x, const t2 &y) { return x + y; });
  return output;
}

template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator-(const Tensor<t1> &a,
                                                          const Tensor<t2> &b) {
  assert(a.size() == b.size());
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](const t1 &x, const t2 &y) { return x - y; });
  return output;
}

template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator*(const Tensor<t1> &a,
                                                          const Tensor<t2> &b) {
  assert(a.size() == b.size());
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](const t1 &x, const t2 &y) { return x * y; });
  return output;
}

template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator/(const Tensor<t1> &a,
                                                          const Tensor<t2> &b) {
  assert(a.size() == b.size());
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](const t1 &x, const t2 &y) { return x / y; });
  return output;
}

//
// TENSOR <OP> NUMBER
//
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator+(const Tensor<t1> &a,
                                                          const t2 &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](const t1 &x) { return x + b; });
  return output;
}
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator-(const Tensor<t1> &a,
                                                          const t2 &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](const t1 &x) { return x - b; });
  return output;
}
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator*(const Tensor<t1> &a,
                                                          const t2 &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](const t1 &x) { return x * b; });
  return output;
}
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator/(const Tensor<t1> &a,
                                                          const t2 &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](const t1 &x) { return x / b; });
  return output;
}

//
// NUMBER <OP> TENSOR
//
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator+(const t1 &a,
                                                          const Tensor<t2> &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](const t2 &x) { return a + x; });
  return output;
}
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator-(const t1 &a,
                                                          const Tensor<t2> &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](const t2 &x) { return a - x; });
  return output;
}
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator*(const t1 &a,
                                                          const Tensor<t2> &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](const t2 &x) { return a * x; });
  return output;
}
template <typename t1, typename t2>
Tensor<typename std::common_type<t1, t2>::type> operator/(const t1 &a,
                                                          const Tensor<t2> &b) {
  Tensor<typename std::common_type<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](const t2 &x) { return a / x; });
  return output;
}

//
// TENSOR <OP=> TENSOR
//
template <typename t1, typename t2>
Tensor<t1> &operator+=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, const t2 &y) { return x + y; });
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator-=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, const t2 &y) { return x - y; });
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator*=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, const t2 &y) { return x * y; });
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator/=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, const t2 &y) { return x / y; });
  return a;
}

//
// TENSOR <OP=> NUMBER
//
template <typename t1, typename t2>
Tensor<t1> &operator+=(Tensor<t1> &a, const t2 &b) {
  for (auto &an : a) {
    an += b;
  }
  return a;
}
template <typename t1, typename t2>
Tensor<t1> &operator-=(Tensor<t1> &a, const t2 &b) {
  for (auto &an : a) {
    an -= b;
  }
  return a;
}
template <typename t1, typename t2>
Tensor<t1> operator*=(Tensor<t1> &a, const t2 &b) {
  for (auto &an : a) {
    an *= b;
  }
  return a;
}
template <typename t1, typename t2>
Tensor<t1> operator/=(Tensor<t1> &a, const t2 &b) {
  for (auto &an : a) {
    an /= b;
  }
  return a;
}

}  // namespace tensor

#endif  // !TENSOR_DETAIL_TENSOR_OPS_H
