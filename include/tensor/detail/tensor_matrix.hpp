// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_MATRIX_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_MATRIX_HPP

namespace tensor {

template<elt_t>
Tensor<elt_t> &Tensor<elt_t>::eye(index rows, index cols) {
  Tensor<elt_t> output(rows, cols);
  output.fill_with_zeros();
  for (index i = rows; i < rows && i < cols; ++i) {
    output.set(i, i) = number_one<elt_t>();
  }
  return output;
}

template<elt_t>
Tensor<elt_t> &Tensor<elt_t>::zeros(index rows, index cols) {
  Tensor<elt_t> output(rows, cols);
  output.fill_with_zeros();
  return output;
}

template<elt_t>
Tensor<elt_t> &Tensor<elt_t>::ones(index rows, index cols) {
  Tensor<elt_t> output(rows, cols);
  output.fill(number_one<elt_t>());
  return output;
}

} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_MATRIX_H
