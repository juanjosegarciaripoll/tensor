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
#include <functional>

namespace tensor {

//
// Unary operations
//
template <typename t>
Tensor<t> operator-(const Tensor<t> &a) {
  Tensor<t> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), std::negate<t>());
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
Tensor<typename Binop<t1, t2>::type> operator+(const Tensor<t1> &a,
                                               const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(), plus<t1, t2>());
  return output;
}

template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator-(const Tensor<t1> &a,
                                               const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 minus<t1, t2>());
  return output;
}

template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator*(const Tensor<t1> &a,
                                               const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 times<t1, t2>());
  return output;
}

template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator/(const Tensor<t1> &a,
                                               const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 divided<t1, t2>());
  return output;
}

//
// TENSOR <OP> NUMBER
//
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator+(const Tensor<t1> &a,
                                               const t2 &b) {
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), plus_constant<t1, t2>(b));
  return output;
}
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator-(const Tensor<t1> &a,
                                               const t2 &b) {
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), minus_constant<t1, t2>(b));
  return output;
}
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator*(const Tensor<t1> &a,
                                               const t2 &b) {
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), times_constant<t1, t2>(b));
  return output;
}
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator/(const Tensor<t1> &a,
                                               const t2 &b) {
  Tensor<typename Binop<t1, t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 divided_constant<t1, t2>(b));
  return output;
}

//
// NUMBER <OP> TENSOR
//
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator+(const t1 &a,
                                               const Tensor<t2> &b) {
  Tensor<typename Binop<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), plus_constant<t2, t1>(a));
  return output;
}
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator-(const t1 &a,
                                               const Tensor<t2> &b) {
  Tensor<typename Binop<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), constant_minus<t1, t2>(a));
  return output;
}
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator*(const t1 &a,
                                               const Tensor<t2> &b) {
  Tensor<typename Binop<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), times_constant<t2, t1>(a));
  return output;
}
template <typename t1, typename t2>
Tensor<typename Binop<t1, t2>::type> operator/(const t1 &a,
                                               const Tensor<t2> &b) {
  Tensor<typename Binop<t1, t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 constant_divided<t1, t2>(a));
  return output;
}

//
// TENSOR <OP=> TENSOR
//
template <typename t1, typename t2>
Tensor<t1> &operator+=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(), plus<t1, t2>());
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator-=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(), minus<t1, t2>());
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator*=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(), times<t1, t2>());
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator/=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(), divided<t1, t2>());
  return a;
}

//
// TENSOR <OP=> NUMBER
//
template <typename t1, typename t2>
Tensor<t1> &operator+=(Tensor<t1> &a, const t2 &b) {
  std::transform(a.begin(), a.end(), a.begin(), plus_constant<t1, t2>(b));
  return a;
}
template <typename t1, typename t2>
Tensor<t1> &operator-=(Tensor<t1> &a, const t2 &b) {
  std::transform(a.begin(), a.end(), a.begin(), minus_constant<t1, t2>(b));
  return a;
}
template <typename t1, typename t2>
Tensor<t1> operator*=(Tensor<t1> &a, const t2 &b) {
  std::transform(a.begin(), a.end(), a.begin(), times_constant<t1, t2>(b));
  return a;
}
template <typename t1, typename t2>
Tensor<t1> operator/=(Tensor<t1> &a, const t2 &b) {
  std::transform(a.begin(), a.end(), a.begin(), divided_constant<t1, t2>(b));
  return a;
}

}  // namespace tensor

#endif  // !TENSOR_DETAIL_TENSOR_OPS_H
