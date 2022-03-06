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
class TensorIterator;

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

  friend class TensorIterator<elt_t>;

  template <typename it>
  void copy_to_contiguous_iterator(it destination) {
    if (iterator_.contiguous()) {
      for (const auto size = iterator_.limit(); iterator_.counter() != size;
           iterator_.advance_next()) {
        auto origin_begin = base_ + iterator_.offset();
        std::copy(origin_begin, origin_begin + size, destination);
        destination += size;
      }
    } else {
      for (const auto limit = iterator_.limit(); iterator_.counter() != limit;
           ++iterator_, ++destination) {
        *destination = base_[iterator_.get_position()];
      }
    }
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

  elt_t &operator*() noexcept {
    index tensor_iterator_position = iterator_.get_position();
    return base_[tensor_iterator_position];
  }

  elt_t &operator->() noexcept { return this->operator*(); }

  TensorIterator<elt_t> &operator++() noexcept {
    ++iterator_;
    return *this;
  }

  bool operator!=(const TensorIterator<elt_t> &other) const noexcept {
    return iterator_ != other.iterator_;
  }

  void copy_from(TensorConstIterator<elt_t> origin) noexcept {
    auto &origin_it = origin.iterator_;
    if (iterator_.contiguous() && origin_it.contiguous() &&
        iterator_.limit() == origin_it.limit()) {
      for (const auto size = iterator_.limit(); iterator_.counter() != size;
           iterator_.advance_next(), origin_it.advance_next()) {
        auto destination_begin = base_ + iterator_.offset();
        auto origin_begin = origin.base_ + origin_it.offset();
        std::copy(origin_begin, origin_begin + size, destination_begin);
      }
    } else {
      for (const auto limit = iterator_.limit(); iterator_.counter() != limit;
           ++iterator_, ++origin) {
        base_[iterator_.get_position()] = *origin;
      }
    }
  }

  template <typename it>
  void copy_from_contiguous_iterator(it origin) noexcept {
    if (iterator_.contiguous()) {
      for (const auto size = iterator_.limit(); iterator_.counter() != size;
           iterator_.advance_next()) {
        auto destination_begin = base_ + iterator_.offset();
        auto origin_end = origin + size;
        std::copy(origin, origin_end, destination_begin);
        origin = origin_end;
      }
    } else {
      for (const auto limit = iterator_.limit(); iterator_.counter() != limit;
           ++iterator_, ++origin) {
        base_[iterator_.get_position()] = *origin;
      }
    }
  }

  void fill(elt_t value) noexcept {
    if (iterator_.contiguous()) {
      for (const auto size = iterator_.limit(); iterator_.counter() != size;
           iterator_.advance_next()) {
        auto destination_begin = base_ + iterator_.offset();
        std::fill(destination_begin, destination_begin + size, value);
      }
    } else {
      for (const auto limit = iterator_.limit(); iterator_.counter() != limit;
           ++iterator_) {
        base_[iterator_.get_position()] = value;
      }
    }
  }

 private:
  RangeIterator iterator_;
  elt_t *base_;
};

}  // namespace tensor

#endif  // TENSOR_TENSOR_ITERATOR_H