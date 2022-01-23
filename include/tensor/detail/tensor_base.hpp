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
#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_BASE_HPP)
#error "This header cannot be included manually"
#endif
#ifndef TENSOR_DETAIL_TENSOR_BASE_HPP
#define TENSOR_DETAIL_TENSOR_BASE_HPP

#include <utility>
#include <cassert>
#include <algorithm>
#include <tensor/rand.h>
#include <tensor/detail/common.h>

namespace tensor {

//
// CONSTRUCTORS
//

template <typename elt_t>
Tensor<elt_t>::Tensor() : data_(), dims_() {}

template <typename elt_t>
Tensor<elt_t>::Tensor(const Dimensions &new_dims)
    : data_(new_dims.total_size()), dims_(new_dims) {}

template <typename elt_t>
Tensor<elt_t>::Tensor(const Dimensions &new_dims, const Tensor<elt_t> &other)
    : data_(other.data_), dims_(new_dims) {
  assert(dims_.total_size() == size());
}

//
// Integer dimensions
//

template <typename elt_t>
Tensor<elt_t>::Tensor(const Vector<elt_t> &data)
    : dims_{data_.size()}, data_(data) {}

template <typename elt_t>
Tensor<elt_t>::Tensor(Vector<elt_t> &&data)
    : dims_({data_.size()}), data_(std::move(data)) {}

//
// DIMENSIONS
//

template <typename elt_t>
index Tensor<elt_t>::dimension(int which) const {
  assert(rank() > which);
  assert(which >= 0);
  return dims_[which];
}

template <typename elt_t>
void Tensor<elt_t>::reshape(const Dimensions &new_dimensions) {
  assert(new_dimensions.total_size() == size());
  dims_ = new_dimensions;
}

//
// SETTERS
//

template <typename elt_t>
Tensor<elt_t> &Tensor<elt_t>::fill_with(const elt_t &e) {
  std::fill(begin(), end(), e);
  return *this;
}

template <typename elt_t>
Tensor<elt_t> &Tensor<elt_t>::randomize() {
  for (iterator it = begin(); it != end(); it++) {
    *it = rand<elt_t>();
  }
  return *this;
}

}  // namespace tensor

#endif  // !TENSOR_DETAIL_TENSOR_BASE_H
