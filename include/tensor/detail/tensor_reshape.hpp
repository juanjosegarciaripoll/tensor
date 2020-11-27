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

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_RESHAPE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_RESHAPE_HPP

namespace tensor {

//
// DIMENSIONS
//

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, const Indices &new_dims) {
  return Tensor<elt_t>(new_dims, t);
}

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index length) {
  Indices new_dims(1);
  new_dims.at(0) = length;
  return Tensor<elt_t>(new_dims, t);
}

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index rows, index cols) {
  Indices new_dims(2);
  new_dims.at(0) = rows;
  new_dims.at(1) = cols;
  return Tensor<elt_t>(new_dims, t);
}

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3) {
  Indices new_dims(3);
  new_dims.at(0) = d1;
  new_dims.at(1) = d2;
  new_dims.at(2) = d3;
  return Tensor<elt_t>(new_dims, t);
}

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3,
                      index d4) {
  Indices new_dims(4);
  new_dims.at(0) = d1;
  new_dims.at(1) = d2;
  new_dims.at(2) = d3;
  new_dims.at(3) = d4;
  return Tensor<elt_t>(new_dims, t);
}

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3,
                      index d4, index d5) {
  Indices new_dims(5);
  new_dims.at(0) = d1;
  new_dims.at(1) = d2;
  new_dims.at(2) = d3;
  new_dims.at(3) = d4;
  new_dims.at(4) = d5;
  return Tensor<elt_t>(new_dims, t);
}

template <typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3,
                      index d4, index d5, index d6) {
  Indices new_dims(6);
  new_dims.at(0) = d1;
  new_dims.at(1) = d2;
  new_dims.at(2) = d3;
  new_dims.at(3) = d4;
  new_dims.at(4) = d5;
  new_dims.at(5) = d6;
  return Tensor<elt_t>(new_dims, t);
}

}  // namespace tensor

#endif  // !#define TENSOR_DETAIL_TENSOR_RESHAPE_HPP
