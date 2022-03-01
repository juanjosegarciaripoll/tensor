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

#ifndef TENSOR_VECTOR_H
#define TENSOR_VECTOR_H

#include <algorithm>
#include <memory>
#include <cstring>
#include <tensor/numbers.h>
#include <tensor/exceptions.h>

namespace tensor {

template <typename elt_t, size_t n>
class StaticVector;

//////////////////////////////////////////////////////////////////////
// VECTOR CLASS WITH SHARED DATA
//

template <typename elt_t>
class SimpleVector;

inline size_t safe_size_t(index s) {
  if (s < 0) {
    throw std::length_error("Negative length");
  }
  return static_cast<size_t>(s);
}

template <typename sequence>
index ssize(const sequence &s) {
  return static_cast<index>(s.size());
}

template <typename elt>
class Vector {
 public:
  typedef elt elt_t;
  typedef elt value_type;
  typedef size_t size_type;
  typedef index difference_type;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  Vector() noexcept = default;

  explicit Vector(size_type size)
      : size_{size},
        base_{size ? new elt_t[size] : nullptr},
        data_(base_, std::default_delete<elt_t[]>()) {}

  static inline Vector<elt_t> empty(size_type size) { return Vector(size); }

  static inline Vector<elt_t> empty(difference_type size) {
    return Vector(safe_size_t(size));
  }

  template <size_t n>
  Vector(const StaticVector<elt_t, n> &v) : Vector(v.size()) {
    v.push(base_);
  }

  template <typename other_elt>
  Vector(const std::initializer_list<other_elt> &l) : Vector(l.size()) {
    std::copy(l.begin(), l.end(), base_);
  }

  Vector(const SimpleVector<elt_t> &v) : Vector(v.size()) {
    std::copy(v.begin(), v.end(), base_);
  }

  /* Copy constructor and copy operator */
  Vector(const Vector<elt_t> &v) = default;
  Vector &operator=(const Vector<elt_t> &v) = default;

  /* Move semantics. */
  Vector(Vector<elt_t> &&v) noexcept = default;
  Vector &operator=(Vector<elt_t> &&v) noexcept = default;

  /* Create a vector that references data we do not own (own=false in the
     RefPointer constructor. */
  Vector(size_type size, elt_t *data) : size_{size}, base_{data}, data_{} {}

  constexpr size_type size() const noexcept { return size_; }
  constexpr difference_type ssize() const noexcept {
    return static_cast<difference_type>(size_);
  }

  /* TODO: Add operator[] for non-const once we remove copy-on-write */

  const elt_t &operator[](difference_type pos) const noexcept {
    return *(cbegin() + pos);
  }
  elt_t &at(difference_type pos) { return *(begin() + pos); }
  const elt_t &at(difference_type pos) const noexcept {
    return *(cbegin() + pos);
  }

  elt_t *data() { return begin(); }
  const elt_t *data() const noexcept { return cbegin(); }

  /* TODO: Make begin() and end() no except when we remove copy-on-write. */
  iterator begin() { return appropriate(); }
  const_iterator begin() const noexcept { return base_; }
  const_iterator cbegin() const noexcept { return base_; }
  const_iterator cend() const noexcept { return cbegin() + size_; }
  const_iterator end() const noexcept { return cbegin() + size_; }
  iterator end() { return begin() + size_; }

  // Only for testing purposes
  difference_type ref_count() const noexcept { return data_.use_count(); }

 private:
  size_type size_{0};
  elt_t *base_{nullptr};
  std::shared_ptr<elt_t> data_{};

  elt_t *appropriate() {
    if (data_.use_count() > 1) {
      std::shared_ptr<elt_t> tmp(new elt_t[size_],
                                 std::default_delete<elt_t[]>());
      std::copy(base_, base_ + size_, tmp.get());
      std::swap(data_, tmp);
      return base_ = data_.get();
    }
    return base_;
  }
};

typedef Vector<double> RVector;
typedef Vector<cdouble> CVector;

template <typename t1, typename t2>
bool operator==(const Vector<t1> &v1, const Vector<t2> &v2) {
  return (v1.size() == v2.size()) &&
         std::equal(v1.begin(), v1.end(), v2.begin());
}

//////////////////////////////////////////////////////////////////////
// VECTOR CLASS
//

template <typename elt>
class SimpleVector {
 public:
  typedef elt elt_t;
  typedef elt value_type;
  typedef size_t size_type;
  typedef index difference_type;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  SimpleVector() noexcept = default;

  explicit SimpleVector(size_type size) : size_{size}, base_(new elt_t[size]) {}

  explicit SimpleVector(difference_type size)
      : size_{safe_size_t(size)}, base_(new elt_t[size]) {}

  template <size_t n>
  SimpleVector(const StaticVector<elt_t, n> &v) : SimpleVector(v.size()) {
    v.push(base_.get());
  }

  template <typename other_elt>
  SimpleVector(const std::initializer_list<other_elt> &l)
      : SimpleVector(l.size()) {
    std::copy(l.begin(), l.end(), begin());
  }

  SimpleVector(const Vector<elt_t> &v) : SimpleVector<elt_t>(v.size()) {
    std::copy(v.begin(), v.end(), base_.get());
  }

  /* Copy constructor and copy operator */
  SimpleVector(const SimpleVector<elt_t> &v) : SimpleVector<elt_t>(v.size()) {
    std::copy(v.begin(), v.end(), begin());
  };
  SimpleVector &operator=(const SimpleVector<elt_t> &v) {
    size_ = v.size_;
    base_ = std::make_unique<elt_t[]>(v.size_);
    std::copy(v.begin(), v.end(), base_.get());
    return *this;
  };

  /* Move semantics. */
  SimpleVector(SimpleVector<elt_t> &&v) noexcept = default;
  SimpleVector &operator=(SimpleVector<elt_t> &&v) noexcept = default;

  static inline SimpleVector<elt_t> empty(size_type size) {
    return SimpleVector<elt_t>(size);
  }

  constexpr size_type size() const noexcept { return size_; }
  constexpr difference_type ssize() const noexcept {
    return static_cast<difference_type>(size_);
  }

  elt_t &operator[](difference_type pos) noexcept { return *(begin() + pos); }
  const elt_t &operator[](difference_type pos) const noexcept {
    return *(cbegin() + pos);
  }
  const elt_t &at(difference_type pos) const noexcept {
    return *(cbegin() + pos);
  }
  elt_t &at(difference_type pos) noexcept { return *(begin() + pos); }

  elt_t *data() noexcept { return begin(); }
  const elt_t *data() const noexcept { return cbegin(); }

  iterator begin() noexcept { return base_.get(); }
  const_iterator begin() const noexcept { return base_.get(); }
  const_iterator end() const noexcept { return base_.get() + size_; }
  const_iterator cbegin() const noexcept { return base_.get(); }
  const_iterator cend() const noexcept { return base_.get() + size_; }
  iterator end() noexcept { return base_.get() + size_; }

 private:
  size_t size_{0};
  std::unique_ptr<elt_t[]> base_{};
};

}  // namespace tensor

#endif  // !TENSOR_VECTOR_H
