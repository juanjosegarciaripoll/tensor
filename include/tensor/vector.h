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

#ifndef TENSOR_VECTOR_H
#define TENSOR_VECTOR_H

#include <tensor/refcount.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// VECTOR CLASS
//

typedef long index;

template<typename elt> class VectorView;

template<typename elt>
class Vector {
 public:
  typedef tensor::index index;
  typedef elt elt_t;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  Vector() : data_() {}

  explicit Vector(index size) : data_(size) {}

  Vector(const Vector<elt_t> &v) : data_(v.data_) {}
  Vector(index size, elt_t *data) : data_(data, size) {}
  Vector &operator=(const Vector<elt_t> &v) { data_ = v.data_; return *this; }

  index size() const {
    return data_.size();
  }
  void resize(index new_size) {
    data_.reallocate(new_size);
  }

  const elt_t &operator[](index pos) const {
    return *(data_.begin_const() + pos);
  }
  elt_t &at(index pos) {
    return *(data_.begin() + pos);
  }

  iterator begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin_const(); }
  const_iterator begin_const() const { return data_.begin_const(); }
  const_iterator end_const() const { return data_.end_const(); }
  const_iterator end() const { return data_.end_const(); }
  iterator end() { return data_.end(); }

  // Only for testing purposes
  int ref_count() const { return data_.ref_count(); }

 private:
  friend class VectorView<elt_t>;

  RefPointer<elt_t> data_;
};

template<typename t1, typename t2>
bool operator==(const Vector<t1> &v1, const Vector<t2> &v2) {
  return (v1.size() == v2.size()) && std::equal(v1.begin(), v1.end(), v2.begin());
}

template<typename elt>
class VectorView {
 public:
  typedef tensor::index index;
  typedef elt elt_t;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  explicit VectorView(const Vector<elt_t> &other) : data_(other.data_) {}

  index size() const {
    return data_.size();
  }

  const elt_t &operator[](index pos) const {
    return *(data_.begin_const() + pos);
  }
  elt_t &at(index pos) {
    return *(data_.begin() + pos);
  }

  iterator begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin_const(); }
  const_iterator end() const { return data_.end_const(); }
  iterator end() { return data_.end(); }

  // Only for testing purposes
  int ref_count() const { return data_.ref_count(); }

 private:
  VectorView<elt_t>(); /* Deactivated */
  VectorView<elt_t>(const VectorView<elt_t> &); /* Deactivated */
  VectorView<elt_t> &operator=(const Vector<elt_t> &v);

  RefPointerView<elt_t> data_;
};

}; // namespace

#endif // !TENSOR_VECTOR_H
