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
#ifndef TENSOR_VECTOR_H
#define TENSOR_VECTOR_H

#include <memory>
#include <cstring>
#include <tensor/numbers.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// VECTOR CLASS
//

typedef std::ptrdiff_t index;

template <typename elt>
class Vector {
 public:
  typedef elt elt_t;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  Vector() : size_{0}, data_{} {}

  explicit Vector(size_t size)
      : size_{size}, data_(new elt_t[size], std::default_delete<elt_t[]>()) {}

  /* Copy constructor and copy operator */
  Vector(const Vector<elt_t> &v) = default;
  Vector &operator=(const Vector<elt_t> &v) {
    size_ = v.size_;
    data_ = v.data_;
    return *this;
  }

  /* Move semantics. */
  Vector(Vector<elt_t> &&v) = default;
  Vector &operator=(Vector<elt_t> &&v) {
    std::swap(size_, v.size_);
    std::swap(data_, v.data_);
    return *this;
  }

  /* Create a vector that references data we do not own (own=false in the
     RefPointer constructor. */
  //Vector(size_t size, elt_t *data) : size_{size}, data_{data} {}

  constexpr size_t size() const { return size_; }

  const elt_t &operator[](size_t pos) const { return *(begin() + pos); }
  elt_t &at(size_t pos) { return *(begin() + pos); }

  iterator begin() { return appropriate(); }
  const_iterator begin() const { return data_.get(); }
  const_iterator begin_const() const { return data_.get(); }
  const_iterator end_const() const { return data_.get() + size_; }
  const_iterator end() const { return data_.get() + size_; }
  iterator end() { return begin() + size_; }

  // Only for testing purposes
  size_t ref_count() const { return data_.use_count(); }

 private:
  size_t size_;
  std::shared_ptr<elt_t> data_;

  elt_t *appropriate() {
    if (data_.use_count() > 1) {
      std::shared_ptr<elt_t> tmp(new elt_t[size_],
                                 std::default_delete<elt_t[]>());
      memcpy(tmp.get(), data_.get(), size_ * sizeof(elt_t));
      std::swap(data_, tmp);
    }
    return data_.get();
  }
};

typedef Vector<double> RVector;
typedef Vector<cdouble> CVector;

template <typename t1, typename t2>
bool operator==(const Vector<t1> &v1, const Vector<t2> &v2) {
  return (v1.size() == v2.size()) &&
         std::equal(v1.begin(), v1.end(), v2.begin());
}

};  // namespace tensor

#endif  // !TENSOR_VECTOR_H
