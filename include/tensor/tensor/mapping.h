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
#ifndef TENSOR_TENSOR_MAPPING_H
#define TENSOR_TENSOR_MAPPING_H

#include <algorithm>
#include <tensor/tensor/types.h>

namespace tensor {
namespace mapping {

using namespace tensor;

template <class elt_t, typename op>
inline Tensor<elt_t> ufunc1(const Tensor<elt_t> &a, op b) {
  auto output = Tensor<elt_t>::empty(a.dimensions());
  std::transform(std::begin(a), std::end(a), std::begin(output), b);
  return output;
}

}  // namespace mapping
}  // namespace tensor

#endif  // TENSOR_TENSOR_MAPPING_H