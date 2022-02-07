// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
#pragma once
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
#ifndef TENSOR_TENSOR_OPERATORS_H
#define TENSOR_TENSOR_OPERATORS_H

#include <tensor/tensor/types.h>
#include <tensor/traits.h>

/*!\addtogroup Tensors*/
/* @{ */

namespace tensor {

/**Return a Tensor with same data and given dimensions.*/
template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, const Dimensions &d) {
  return Tensor<elt_t>(d, t);
}

/**Return a RTensor with same data and given dimensions, specified separately.*/
template <typename elt_t, typename... index_like>
inline Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1,
                             index_like... dnext) {
  return Tensor<elt_t>({d1, static_cast<index>(dnext)...}, t);
}

/**Convert a tensor to a 1D vector with the same elements.*/
template <typename elt_t>
Tensor<elt_t> flatten(const Tensor<elt_t> &t) {
  return reshape(t, t.size());
}

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
tensor_common_t<t1, t2> operator+(const Tensor<t1> &a, const Tensor<t2> &b) {
  // This should be: assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  assert(a.size() == b.size());
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](t1 x, t2 y) { return x + y; });
  return output;
}

template <typename t1, typename t2>
tensor_common_t<t1, t2> operator-(const Tensor<t1> &a, const Tensor<t2> &b) {
  assert(a.size() == b.size());
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](t1 x, t2 y) { return x - y; });
  return output;
}

template <typename t1, typename t2>
tensor_common_t<t1, t2> operator*(const Tensor<t1> &a, const Tensor<t2> &b) {
  assert(a.size() == b.size());
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](t1 x, t2 y) { return x * y; });
  return output;
}

template <typename t1, typename t2>
tensor_common_t<t1, t2> operator/(const Tensor<t1> &a, const Tensor<t2> &b) {
  assert(a.size() == b.size());
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 [](t1 x, t2 y) { return x / y; });
  return output;
}

//
// TENSOR <OP> NUMBER
//
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator+(const Tensor<t1> &a, t2 b) {
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](t1 x) { return x + b; });
  return output;
}
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator-(const Tensor<t1> &a, t2 b) {
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](t1 x) { return x - b; });
  return output;
}
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator*(const Tensor<t1> &a, t2 b) {
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](t1 x) { return x * b; });
  return output;
}
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator/(const Tensor<t1> &a, t2 b) {
  tensor_common_t<t1, t2> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(),
                 [&](t1 x) { return x / b; });
  return output;
}

//
// NUMBER <OP> TENSOR
//
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator+(t1 a, const Tensor<t2> &b) {
  tensor_common_t<t1, t2> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](t2 x) { return a + x; });
  return output;
}
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator-(t1 a, const Tensor<t2> &b) {
  tensor_common_t<t1, t2> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](t2 x) { return a - x; });
  return output;
}
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator*(t1 a, const Tensor<t2> &b) {
  tensor_common_t<t1, t2> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](t2 x) { return a * x; });
  return output;
}
template <typename t1, typename t2>
tensor_common_t<t1, t2> operator/(t1 a, const Tensor<t2> &b) {
  tensor_common_t<t1, t2> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(),
                 [&](t2 x) { return a / x; });
  return output;
}

//
// TENSOR <OP=> TENSOR
//
template <typename t1, typename t2>
Tensor<t1> &operator+=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, t2 y) { return x + y; });
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator-=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, t2 y) { return x - y; });
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator*=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, t2 y) { return x * y; });
  return a;
}

template <typename t1, typename t2>
Tensor<t1> &operator/=(Tensor<t1> &a, const Tensor<t2> &b) {
  assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                 [](t1 &x, t2 y) { return x / y; });
  return a;
}

//
// TENSOR <OP=> NUMBER
//
template <typename t1, typename t2>
Tensor<t1> &operator+=(Tensor<t1> &a, t2 b) {
  for (auto &an : a) {
    an += b;
  }
  return a;
}
template <typename t1, typename t2>
Tensor<t1> &operator-=(Tensor<t1> &a, t2 b) {
  for (auto &an : a) {
    an -= b;
  }
  return a;
}
template <typename t1, typename t2>
Tensor<t1> &operator*=(Tensor<t1> &a, t2 b) {
  for (auto &an : a) {
    an *= b;
  }
  return a;
}
template <typename t1, typename t2>
Tensor<t1> &operator/=(Tensor<t1> &a, t2 b) {
  for (auto &an : a) {
    an /= b;
  }
  return a;
}

}  // namespace tensor

/* @} */

#endif  // TENSOR_TENSOR_OPERATORS_H