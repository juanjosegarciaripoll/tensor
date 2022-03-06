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

#ifndef TENSOR_TENSOR_ITERATOR_H
#define TENSOR_TENSOR_ITERATOR_H

#include <tensor/ranges.h>

namespace tensor {

template <typename elt_t>
class TensorConstIterator {
 public:
  typedef const elt_t value_type;
  typedef index difference_type;
  typedef const elt_t &reference;
  typedef const elt_t *pointer;
  typedef std::input_iterator_tag iterator_category;

  TensorConstIterator(RangeIterator it, const elt_t *base)
      : iterator_{std::move(it)}, base_{base} {}
  const elt_t &operator*() {
    index tensor_iterator_position = iterator_.get_position();
    return base_[tensor_iterator_position];
  }
  const elt_t &operator->() { return this->operator*(); }
  TensorConstIterator<elt_t> &operator++() {
    ++iterator_;
    return *this;
  }
  bool operator!=(const TensorConstIterator<elt_t> &other) const {
    return iterator_ != other.iterator_;
  }

 private:
  RangeIterator iterator_;
  const elt_t *base_;
};

template <typename elt_t>
class TensorIterator {
 public:
  typedef elt_t value_type;
  typedef index difference_type;
  typedef elt_t &reference;
  typedef elt_t *pointer;
  typedef std::input_iterator_tag iterator_category;

  TensorIterator(RangeIterator it, elt_t *base)
      : iterator_{std::move(it)}, base_{base} {}

  elt_t &operator*() tensor_noexcept {
    index tensor_iterator_position = iterator_.get_position();
    return base_[tensor_iterator_position];
  }

  elt_t &operator->() { return this->operator*(); }

  TensorIterator<elt_t> &operator++() {
    ++iterator_;
    return *this;
  }

  bool operator!=(const TensorIterator<elt_t> &other) const {
    return iterator_ != other.iterator_;
  }

 private:
  RangeIterator iterator_;
  elt_t *base_;
};

}  // namespace tensor

#endif  // TENSOR_TENSOR_ITERATOR_H