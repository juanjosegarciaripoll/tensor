// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_VECTOR_H
#define TENSOR_VECTOR_H

#include <tensor/refcount.h>

namespace tensor {

template<typename elt>
class Vector {
 public:
  typedef size_t index;
  typedef elt elt_t;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  explicit Vector(index size) : data_(size) {}

  Vector(const Vector<elt_t> &v) : data_(v.data_) {}
  Vector &operator=(const Vector<elt_t> &v) { data_ = v.data_; }

  index size() const {
    return data_.size();
  }

  const elt_t &operator[](index pos) const {
    return data_.constant_pointer()[pos];
  }
  const elt_t &at(index pos) const {
    return data_.pointer()[pos];
  }
  elt_t &at(index pos) {
    return data_.pointer()[pos];
  }

  iterator begin() {
    return data_.pointer();
  }
  const_iterator begin() const {
    return data_.pointer();
  }
  const_iterator begin_const() const {
    return data_.constant_pointer();
  }
  const_iterator end_const() const {
    return data_.constant_pointer() + data_.size();
  }
  const_iterator end() const {
    return data_.pointer() + data_.size();
  }
  iterator end() {
    return data_.pointer() + data_.size();
  }

 private:
  RefPointer<elt_t> data_;
};

}; // namespace

#endif // !TENSOR_VECTOR_H
