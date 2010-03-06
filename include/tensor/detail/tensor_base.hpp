// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_BASE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_BASE_HPP

#include <cassert>
#include <algorithm>
#include <tensor/rand.h>

namespace tensor {

bool verify_tensor_dimensions(const Indices &i, index total_size);

inline index normalize_index(index i, index dimension) {
  if (i < 0)
    i += dimension;
  assert((i < dimension) && (i >= 0));
  return i;
}

//
// CONSTRUCTORS
//
//
// Vector of dimensions
//
template<typename elt_t>
Tensor<elt_t>::Tensor(const Indices &new_dims) :
  dims_(new_dims), data_(new_dims.total_size())
{
  assert(verify_tensor_dimensions(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(const Indices &new_dims, const Tensor<elt_t> &other) :
  dims_(new_dims), data_(other.data_)
{
  assert(verify_tensor_dimensions(dims_, size()));
}

//
// Integer dimensions
//

template<typename elt_t>
Tensor<elt_t>::Tensor(index length) :
  data_(length), dims_(1)
{
  dims_.at(0) = length;
  assert(verify_tensor_dimensions(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index rows, index cols) :
  data_(rows * cols), dims_(2)
{
  dims_.at(0) = rows;
  dims_.at(1) = cols;
  assert(verify_tensor_dimensions(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3) :
  data_(d1 * d2 * d3), dims_(3)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  assert(verify_tensor_dimensions(dims_, size()));
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3, index d4) :
  data_(d1 * d2 * d3 * d4), dims_(4)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  dims_.at(3) = d4;
  assert(verify_tensor_dimensions(dims_, size()));
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
  assert(verify_tensor_dimensions(dims_, size()));
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
  assert(verify_tensor_dimensions(dims_, size()));
}

//
// DIMENSIONS
//

template<typename elt_t>
index Tensor<elt_t>::dimension(int which) const {
  assert(rank() >= which);
  assert(which >= 0);
  return dims_[which];
}

template<typename elt_t>
void Tensor<elt_t>::get_dimensions(index *length) const {
  assert(rank() == 1);
  *length = dims_[0];
}

template<typename elt_t>
void Tensor<elt_t>::get_dimensions(index *rows, index *cols) const {
  assert(rank() == 2);
  *rows = dims_[0];
  *cols = dims_[1];
}

template<typename elt_t>
void Tensor<elt_t>::get_dimensions(index *d0, index *d1, index *d2) const {
  assert(rank() == 3);
  *d0 = dims_[0];
  *d1 = dims_[1];
  *d2 = dims_[2];
}

template<typename elt_t>
void Tensor<elt_t>::get_dimensions(index *d0, index *d1, index *d2,
                                   index *d3) const {
  assert(rank() == 4);
  *d0 = dims_[0];
  *d1 = dims_[1];
  *d2 = dims_[2];
  *d3 = dims_[3];
}

template<typename elt_t>
void Tensor<elt_t>::get_dimensions(index *d0, index *d1, index *d2, index *d3,
                                   index *d4) const {
  assert(rank() == 5);
  *d0 = dims_[0];
  *d1 = dims_[1];
  *d2 = dims_[2];
  *d3 = dims_[3];
  *d4 = dims_[4];
}

template<typename elt_t>
void Tensor<elt_t>::get_dimensions(index *d0, index *d1, index *d2, index *d3,
                                   index *d4, index *d5) const {
  assert(rank() == 6);
  *d0 = dims_[0];
  *d1 = dims_[1];
  *d2 = dims_[2];
  *d3 = dims_[3];
  *d4 = dims_[4];
  *d5 = dims_[5];
}

template<typename elt_t>
void Tensor<elt_t>::reshape(const Indices &new_dims)
{
  assert(verify_tensor_dimensions(new_dims, size()));
  dims_ = new_dims;
}

//
// GETTERS
//

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator[](index i) const {
  return data_[i];
}

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator()(index i) const {
  index length;
  get_dimensions(&length);
  i = normalize_index(i, length);
  return data_[i];
}

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator()(index row, index col) const {
  index rows, cols;
  get_dimensions(&rows, &cols);
  row = normalize_index(row, rows);
  col = normalize_index(col, cols);
  return data_[col * rows + row];
}

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator()(index i1, index i2, index i3) const {
  index d1, d2, d3;
  get_dimensions(&d1, &d2, &d3);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  return data_[((i3 * d2) + i2) * d1 + i1];
}

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator()(index i1, index i2, index i3,
                                       index i4) const {
  index d1, d2, d3, d4;
  get_dimensions(&d1, &d2, &d3, &d4);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  i4 = normalize_index(i4, d4);
  return data_[(((i4 * d3 + i3) * d2) + i2) * d1 + i1];
}

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator()(index i1, index i2, index i3,
                                       index i4, index i5) const {
  index d1, d2, d3, d4, d5;
  get_dimensions(&d1, &d2, &d3, &d4, &d5);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  i4 = normalize_index(i4, d4);
  i5 = normalize_index(i5, d5);
  return data_[((((i5 * d4 + i4) * d3 + i3) * d2) + i2) * d1 + i1];
}

template<typename elt_t>
const elt_t &Tensor<elt_t>::operator()(index i1, index i2, index i3,
                                       index i4, index i5, index i6) const {
  index d1, d2, d3, d4, d5, d6;
  get_dimensions(&d1, &d2, &d3, &d4, &d5, &d6);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  i4 = normalize_index(i4, d4);
  i5 = normalize_index(i5, d5);
  i6 = normalize_index(i6, d6);
  return data_[(((((i6 * d5 + i5) * d4 + i4) * d3 + i3) * d2) + i2) * d1 + i1];
}

//
// DESTRUCTIVE ACCESSORS
//

template<typename elt_t>
elt_t &Tensor<elt_t>::at_seq(index i) {
  return data_.at(i);
}

template<typename elt_t>
elt_t &Tensor<elt_t>::at(index i) {
  i = normalize_index(i, size());
  return data_.at(i);
}

template<typename elt_t>
elt_t &Tensor<elt_t>::at(index row, index col) {
  index d0, d1;
  get_dimensions(&d0, &d1);
  row = normalize_index(row, d0);
  col = normalize_index(col, d1);
  return data_.at(col * d0 + row);
}

template<typename elt_t>
elt_t &Tensor<elt_t>::at(index i1, index i2, index i3) {
  index d1, d2, d3;
  get_dimensions(&d1, &d2, &d3);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  return data_.at(((i3 * d2) + i2) * d1 + i1);
}

template<typename elt_t>
elt_t &Tensor<elt_t>::at(index i1, index i2, index i3, index i4) {
  index d1, d2, d3, d4;
  get_dimensions(&d1, &d2, &d3, &d4);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  i4 = normalize_index(i4, d4);
  return data_.at((((i4 * d3 + i3) * d2) + i2) * d1 + i1);
}

template<typename elt_t>
elt_t &Tensor<elt_t>::at(index i1, index i2, index i3, index i4, index i5) {
  index d1, d2, d3, d4, d5;
  get_dimensions(&d1, &d2, &d3, &d4, &d5);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  i4 = normalize_index(i4, d4);
  i5 = normalize_index(i5, d5);
  return data_.at(((((i5 * d4 + i4) * d3 + i3) * d2) + i2) * d1 + i1);
}

template<typename elt_t>
elt_t &Tensor<elt_t>::at(index i1, index i2, index i3, index i4, index i5,
                         index i6) {
  index d1, d2, d3, d4, d5, d6;
  get_dimensions(&d1, &d2, &d3, &d4, &d5);
  i1 = normalize_index(i1, d1);
  i2 = normalize_index(i2, d2);
  i3 = normalize_index(i3, d3);
  i4 = normalize_index(i4, d4);
  i5 = normalize_index(i5, d5);
  i6 = normalize_index(i6, d6);
  return data_.at((((((i6 * d5 + i5) * d4 + i4) * d3 + i3) * d2) + i2) * d1 + i1);
}


//
// SETTERS
//

template<typename elt_t>
void Tensor<elt_t>::fill_with(const elt_t &e) {
  std::fill(begin(), end(), e);
}

template<typename elt_t>
void Tensor<elt_t>::randomize() {
  for (iterator it = begin(); it != end(); it++) {
    *it = rand<elt_t>();
  }
}

} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_BASE_H
