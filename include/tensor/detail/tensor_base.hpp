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

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_BASE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_BASE_HPP

#include <cassert>
#include <algorithm>
#include <tensor/rand.h>
#include <tensor/detail/common.h>

namespace tensor {

//
// CONSTRUCTORS
//

template<typename elt_t>
Tensor<elt_t>::Tensor() : data_(), dims_()
{}

template<typename elt_t>
Tensor<elt_t>::Tensor(const Indices &new_dims) :
  dims_(new_dims), data_(new_dims.total_size())
{
}

template<typename elt_t>
Tensor<elt_t>::Tensor(const Indices &new_dims, const Tensor<elt_t> &other) :
  dims_(new_dims), data_(other.data_)
{
  assert(dims_.total_size() == size());
}

//
// Integer dimensions
//

template<typename elt_t>
Tensor<elt_t>::Tensor(index length) :
  data_(length), dims_(1)
{
  dims_.at(0) = length;
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index rows, index cols) :
  data_(rows * cols), dims_(2)
{
  dims_.at(0) = rows;
  dims_.at(1) = cols;
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3) :
  data_(d1 * d2 * d3), dims_(3)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
}

template<typename elt_t>
Tensor<elt_t>::Tensor(index d1, index d2, index d3, index d4) :
  data_(d1 * d2 * d3 * d4), dims_(4)
{
  dims_.at(0) = d1;
  dims_.at(1) = d2;
  dims_.at(2) = d3;
  dims_.at(3) = d4;
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
}

template<typename elt_t>
Tensor<elt_t>::Tensor(const Tensor<elt_t> &other) :
  dims_(other.dims_), data_(other.data_)
{}

template<typename elt_t>
Tensor<elt_t>::Tensor(const Vector<elt_t> &data) : dims_(1), data_(data) {
  dims_.at(0) = data.size();
}

template<typename elt_t>
const Tensor<elt_t> &Tensor<elt_t>::operator=(const Tensor<elt_t> &other)
{
  data_ = other.data_;
  dims_ = other.dims_;
  return *this;
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
void Tensor<elt_t>::reshape(const Indices &new_dimensions)
{
  assert(new_dimensions.total_size() == size());
  dims_ = new_dimensions;
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
  get_dimensions(&d1, &d2, &d3, &d4, &d5, &d6);
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

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(index length) {
  Tensor<elt_t> t(length);
  t.randomize();
  return t;
}

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(index rows, index cols) {
  Tensor<elt_t> t(rows, cols);
  t.randomize();
  return t;
}

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(index d1, index d2, index d3) {
  Tensor<elt_t> t(d1, d2, d3);
  t.randomize();
  return t;
}

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(index d1, index d2, index d3, index d4) {
  Tensor<elt_t> t(d1, d2, d3, d4);
  t.randomize();
  return t;
}

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(index d1, index d2, index d3, index d4, index d5) {
  Tensor<elt_t> t(d1, d2, d3, d4, d5);
  t.randomize();
  return t;
}

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(index d1, index d2, index d3, index d4, index d5, index d6) {
  Tensor<elt_t> t(d1, d2, d3, d4, d5, d6);
  t.randomize();
  return t;
}

template<typename elt_t>
const Tensor<elt_t>
Tensor<elt_t>::random(const Indices &dims) {
  Tensor<elt_t> t(dims);
  t.randomize();
  return t;
}

} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_BASE_H
