// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_BASE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_BASE_HPP

#include <cassert>

bool verify_tensor_dims(const Indices &i, index total_size);
index multiply_indices(const Indices &i);

//
// CONSTRUCTORS
//
//
// Vector of dimensions
//
template<typename elt_t>
Tensor<elt_t>::Tensor(const Indices &new_dims) :
  dims_(new_dims), data_(multiply_indices(new_dims))
{
  assert(verify_tensor_dims(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(const Indices &new_dims, const Tensor<elt_t> &data) :
  dims_(new_dims), data_(data)
{
  assert(verify_tensor_dims(dims_, size()));
}

//
// Integer dimensions
//

template<typename elt_t>
Tensor<elt_t>::Tensor(index length) :
  data_(length), dims_(1)
{
  dims_.at(0) = length;
  assert(verify_tensor_dims(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index rows, index cols) :
  data_(rows * cols), dims_(2)
{
  dims_.at(0) = rows;
  dims_.at(1) = cols;
  assert(verify_tensor_dims(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3) :
  data_(d1 * d2 * d3), dims_(3)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  assert(verify_tensor_dims(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3, index d4) :
  data_(d1 * d2 * d3 * d4), dims_(4)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  dims_.at(3) = d4;
  assert(verify_tensor_dims(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3, index d4, index d5) :
  data_(d1 * d2 * d3 * d4 * d5), dims_(5)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  dims_.at(3) = d4;
  dims_.at(4) = d5;
  assert(verify_tensor_dims(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3, index d4, index d5,
                      index d6) :
  data_(d1 * d2 * d3 * d4 * d5 * d6), dims_(6)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  dims_.at(3) = d4;
  dims_.at(4) = d5;
  dims_.at(5) = d6;
  assert(verify_tensor_dims(dims_, size()));
}

#endif // !TENSOR_DETAIL_TENSOR_INL_H
