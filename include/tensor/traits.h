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
#ifndef TENSOR_TRAITS_H
#define TENSOR_TRAITS_H

#include <type_traits>
#include <tensor/tensor/types.h>

namespace tensor {

template <typename t>
struct is_scalar : public std::is_floating_point<t> {};

template <typename t>
struct is_scalar<std::complex<t>> : public std::true_type {};

template <typename elt_t>
struct is_tensor : public std::false_type {};

template <typename elt_t>
struct is_tensor<Tensor<elt_t>> : public std::true_type {};

template <typename elt_t>
struct is_tensor<TensorView<elt_t>> : public std::true_type {};

template <typename elt_t>
struct is_tensor<MutableTensorView<elt_t>> : public std::true_type {};

template <typename t>
struct tensor_scalar_type_inner {
  using type = t;
};

template <typename t>
struct tensor_scalar_type_inner<Tensor<t>> {
  using type = t;
};

template <typename t>
struct tensor_scalar_type_inner<TensorView<t>> {
  using type = t;
};

template <typename t>
struct tensor_scalar_type_inner<MutableTensorView<t>> {
  using type = t;
};

template <typename t>
using tensor_scalar_t = typename tensor_scalar_type_inner<t>::type;

template <typename t1, typename t2>
using tensor_common_t =
    Tensor<std::common_type_t<tensor_scalar_t<t1>, tensor_scalar_t<t2>>>;

}  // namespace tensor

#endif  // TENSOR_TRAITS_H
