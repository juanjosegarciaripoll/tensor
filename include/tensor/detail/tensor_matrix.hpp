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

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_MATRIX_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_MATRIX_HPP

namespace tensor {

template<typename elt_t>
Tensor<elt_t> Tensor<elt_t>::eye(index rows, index cols) {
  Tensor<elt_t> output(rows, cols);
  output.fill_with_zeros();
  for (index i = 0; i < rows && i < cols; ++i) {
    output.at(i, i) = number_one<elt_t>();
  }
  return output;
}

template<typename elt_t>
Tensor<elt_t> Tensor<elt_t>::zeros(index rows, index cols) {
  Tensor<elt_t> output(rows, cols);
  output.fill_with_zeros();
  return output;
}

template<typename elt_t>
Tensor<elt_t> Tensor<elt_t>::zeros(const Indices &dimensions) {
  Tensor<elt_t> output(dimensions);
  output.fill_with_zeros();
  return output;
}

template<typename elt_t>
Tensor<elt_t> Tensor<elt_t>::ones(index rows, index cols) {
  Tensor<elt_t> output(rows, cols);
  output.fill_with(number_one<elt_t>());
  return output;
}

} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_MATRIX_H
