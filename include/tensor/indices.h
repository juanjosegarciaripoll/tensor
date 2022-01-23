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
#ifndef TENSOR_INDICES_H
#define TENSOR_INDICES_H

#include <list>
#include <cassert>
#include <tensor/vector.h>
#include <tensor/gen.h>
#include <iostream>

/*!\addtogroup Tensors */
/*@{*/
namespace tensor {

extern template class Vector<index>;
extern template class SimpleVector<index>;
class Dimensions;

/** Vector of 'index' type, where 'index' fits the indices of a tensor.*/
class Indices : public Vector<index> {
 public:
  Indices() : Vector<index>() {}
  Indices(const Vector<index> &v) : Vector<index>(v) {}
  template <size_t n>
  Indices(StaticVector<index, n> v) : Vector<index>(v) {}
  Indices(const Dimensions &dims);

  template <typename other_elt>
  Indices(const std::initializer_list<other_elt> &l) : Vector<index>(l) {}

  explicit Indices(index size) : Vector<index>(size) {}

  static const Indices range(index min, index max, index step = 1);
};

class Dimensions {
 public:
  typedef index *iterator;
  typedef const index *const_iterator;

  Dimensions() : dimensions_(), total_size_{0} {}

  Dimensions(const SimpleVector<index> &dims)
      : dimensions_(dims), total_size_{compute_total_size(dimensions_)} {}

  Dimensions(const Indices &dims)
      : dimensions_(dims), total_size_{compute_total_size(dimensions_)} {};

  template <typename other_elt>
  Dimensions(const std::initializer_list<other_elt> &l)
      : dimensions_(l), total_size_{compute_total_size(dimensions_)} {}

  template <size_t n>
  Dimensions(const StaticVector<index, n> &v)
      : dimensions_(v), total_size_{compute_total_size(dimensions_)} {}

  index total_size() const { return total_size_; }
  index rank() const { return dimensions_.size(); }

  index operator[](index pos) const { return dimensions_[pos]; }
  const_iterator begin() const { return dimensions_.begin(); }
  const_iterator end() const { return dimensions_.end(); }
  const SimpleVector<index> &get_vector() const { return dimensions_; }

  static inline index normalize_index(index i, index dimension) {
    if (i < 0) i += dimension;
    assert((i < dimension) && (i >= 0));
    return i;
  }

  template <typename... index_like>
  index column_major_position(index i0, index_like... in) const {
    assert(rank() == sizeof...(in) + 1);
    return column_major_inner(0, i0, in...);
  }

  template <typename... index_like>
  void get_values(index_like *...in) const {
    assert(rank() == sizeof...(in));
    index n = 0;
    auto ignored = {(*(in) = dimensions_[n++], 1)...};
  }

 private:
  SimpleVector<index> dimensions_;
  index total_size_;

  template <typename... index_like>
  index column_major_inner(index n, index in, index_like... irest) const {
    index dn = dimensions_[n];
    in = normalize_index(in, dn);
    return in + dn * column_major_inner(n + 1, irest...);
  }

  inline index column_major_inner(index n, index in) const {
    index dn = dimensions_[n];
    in = normalize_index(in, dn);
    return in;
  }

  static index compute_total_size(const SimpleVector<index> &dims);
};

void surrounding_dimensions(const Indices &d, index ndx, index *d1, index *d2,
                            index *d3);

const Indices operator<<(const Indices &a, const Indices &b);

extern template class Vector<bool>;

/** Vector of boolean values. */
class Booleans : public Vector<bool> {
 public:
  Booleans() : Vector<bool>() {}
  Booleans(const Booleans &b) : Vector<bool>(b) {}
  explicit Booleans(index size) : Vector<bool>(size) {}
};

Booleans operator!(const Booleans &b);
Booleans operator&&(const Booleans &a, const Booleans &b);
Booleans operator||(const Booleans &a, const Booleans &b);
const Indices which(const Booleans &b);

bool all_of(const Booleans &b);
bool any_of(const Booleans &b);
inline bool none_of(const Booleans &b) { return !any_of(b); }

bool all_equal(const Indices &a, const Indices &b);
inline bool some_unequal(const Indices &a, const Indices &b) {
  return !all_equal(a, b);
}
Booleans operator==(const Indices &a, const Indices &b);
Booleans operator<(const Indices &a, const Indices &b);
Booleans operator>(const Indices &a, const Indices &b);
Booleans operator<=(const Indices &a, const Indices &b);
Booleans operator>=(const Indices &a, const Indices &b);
Booleans operator!=(const Indices &a, const Indices &b);

Booleans operator==(const Indices &a, index b);
Booleans operator<(const Indices &a, index b);
Booleans operator>(const Indices &a, index b);
Booleans operator<=(const Indices &a, index b);
Booleans operator>=(const Indices &a, index b);
Booleans operator!=(const Indices &a, index b);

inline Booleans operator==(index a, const Indices &b) { return b == a; }
inline Booleans operator<(index a, const Indices &b) { return b >= a; }
inline Booleans operator>(index a, const Indices &b) { return b <= a; }
inline Booleans operator<=(index a, const Indices &b) { return b > a; }
inline Booleans operator>=(index a, const Indices &b) { return b < a; }
inline Booleans operator!=(index a, const Indices &b) { return b != a; }

//////////////////////////////////////////////////////////////////////
// RANGE OF INTEGERS
//

/** Range of indices. This class should never be used by public functions, but
      only as the output of the function range() and only to access segments of
      a tensors, as in
      \code
      b = a(range(1,2),range())
      \endcode
  */
class Range {
 public:
  Range();
  virtual ~Range();
  virtual index pop();
  virtual void set_factor(index new_factor);
  virtual void set_limit(index new_limit);
  virtual index size() const;
  virtual void reset();
  index nomore() const { return ~(index)0; }
  index get_offset() const { return base_; }
  index get_limit() const { return limit_; }
  index get_factor() const { return factor_; }
  void set_offset(index new_base) { base_ = new_base; }

 private:
  index base_, limit_, factor_;
};

class FullRange : public Range {
 public:
  FullRange();
  virtual index pop();
  virtual void set_factor(index new_factor);
  virtual void set_limit(index new_limit);
  virtual index size() const;
  virtual void reset();

 private:
  index counter_, counter_end_;
};

class StepRange : public Range {
 public:
  StepRange(index start, index end, index step = 1);
  virtual index pop();
  virtual void set_factor(index new_factor);
  virtual void set_limit(index new_limit);
  virtual index size() const;
  virtual void reset();

 private:
  index ndx_, start_, end_, step_;
};

class SingleRange : public Range {
 public:
  SingleRange(index ndx);
  virtual index pop();
  virtual void set_factor(index new_factor);
  virtual void set_limit(index new_limit);
  virtual index size() const;
  virtual void reset();

 private:
  index ndx_, counter_;
};

class IndexRange : public Range {
 public:
  IndexRange(const Indices &i);
  virtual index pop();
  virtual void set_factor(index new_factor);
  virtual void set_limit(index new_limit);
  virtual index size() const;
  virtual void reset();

 private:
  Indices indices_;
  index counter_;
};

class ProductRange : public Range {
 public:
  ProductRange(Range *r1, Range *r2);
  ~ProductRange();
  virtual index pop();
  virtual void set_factor(index new_factor);
  virtual void set_limit(index new_limit);
  virtual index size() const;
  virtual void reset();

 private:
  Range *r1_, *r2_;
  index base_;
};

class PRange {
 public:
  PRange(Range *r) : ptr_(r){};
  operator Range *() const { return ptr_; }
  Range &operator*() { return *ptr_; }
  Range *operator->() { return ptr_; }

 private:
  PRange();
  Range *ptr_;
};

/**Create a Range which only contains one index. \sa \ref sec_tensor_view*/
inline PRange range(index ndx) { return new SingleRange(ndx); }
/**Create a Range start:end (Matlab notation). \sa \ref sec_tensor_view*/
inline PRange range(index start, index end) {
  return new StepRange(start, end);
}
/**Create a Range start:step:end (Matlab notation). \sa \ref sec_tensor_view*/
inline PRange range(index start, index end, index step) {
  return new StepRange(start, end, step);
}
/**Create a Range with the give set of indices. \sa \ref sec_tensor_view*/
inline PRange range(Indices i) { return new IndexRange(i); }
/**Create a Range which covers all indices. \ref sec_tensor_view*/
inline PRange range() { return new FullRange(); }

/**Return a vector of integers from 'start' to 'end' (included) in 'steps'. */
inline Indices iota(index start, index end, index step = 1) {
  return Indices::range(start, end, step);
}

template <size_t n>
bool operator==(const tensor::Indices &v2,
                const tensor::StaticVector<tensor::index, n> &v1) {
  tensor::Indices v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

template <size_t n>
bool operator==(const tensor::StaticVector<tensor::index, n> &v1,
                const tensor::Indices &v2) {
  tensor::Indices v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

};  // namespace tensor

/*@}*/
#endif  // !TENSOR_H
