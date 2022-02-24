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

#ifndef TENSOR_INDICES_H
#define TENSOR_INDICES_H

#include <cassert>
#include <iterator>
#include <algorithm>
#include <tensor/vector.h>
#include <tensor/exceptions.h>
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

  explicit Indices(size_t size) : Vector<index>(size) {}

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
  index rank() const { return dimensions_.ssize(); }

  index operator[](index pos) const { return dimensions_[pos]; }
  const_iterator begin() const { return dimensions_.begin(); }
  const_iterator end() const { return dimensions_.end(); }
  const SimpleVector<index> &get_vector() const { return dimensions_; }

  static inline index normalize_index(index i, index dimension) {
    if (i < 0) i += dimension;
#if 1
    if ((i > dimension) || (i < 0)) {
      throw out_of_bounds_index();
    }
#endif
    return i;
  }

  static inline index normalize_index_safe(index i, index dimension) {
    if (i < 0) i += dimension;
    if ((i > dimension) || (i < 0)) {
      throw out_of_bounds_index();
    }
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

  bool operator==(const Dimensions &other) const {
    return (rank() == other.rank()) &&
           std::equal(begin(), end(), other.begin());
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
void surrounding_dimensions(const Dimensions &d, index ndx, index *d1,
                            index *d2, index *d3);
Dimensions squeeze_dimensions(const Dimensions &d);

const Indices operator<<(const Indices &a, const Indices &b);

extern template class Vector<bool>;

/** Vector of boolean values. */
class Booleans : public Vector<bool> {
 public:
  Booleans() : Vector<bool>() {}
  Booleans(const Booleans &b) : Vector<bool>(b) {}
  explicit Booleans(size_t size) : Vector<bool>(size) {}
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
    only as the output of the function range and only to access segments of
    a tensors, as in
    \code
    b = a(range(1,2),_)
    \endcode
  */
class Range {
  /* The logic of ranges is that we define the start, the step and one past the
     last element we visit (the limit). We combine this with the wraparound
     semantics from Numpy, by which negative numbers are associated positions
     relative to the dimension of the coordinate we are running through. Thus,
    
     (first, last) = (0, 0)   -> runs over [0]
                   = (0, -1)  -> runs over [0, 1, 2] for dimension = 3
                   = (-2, -1) -> runs over [1, 2] for dimension = 3

     Since we can store negative numbers, the `limit=end+1` can also take
     negative or zero values.
   */
 public:
  Range(index position) : Range(position, position, 1) { squeezed_ = true; }
  Range(index first, index last) : Range(first, last, 1) {}
  Range(index first, index last, index step)
      : first_{first}, step_{step}, last_{last}, dimension_{-1} {
    if (step == 0) {
      throw std::invalid_argument("Invalid step sie in Range()");
    }
  }
  Range(index first, index last, index step, index dimension)
      : first_{first}, step_{step}, last_{last}, dimension_{-1} {
    if (step == 0) {
      throw std::invalid_argument("Invalid step sie in Range()");
    }
    set_dimension(dimension);
  }
  Range(Indices indices);
  Range() = default;
  Range(const Range &r) = default;
  Range(Range &&r) = default;
  Range &operator=(const Range &r) = default;
  Range &operator=(Range &&r) = default;

  index first() const { return first_; }
  index step() const { return step_; }
  index last() const { return last_; }
  index dimension() const { return dimension_; }
  void set_dimension(index dimension);
  bool has_indices() const { return indices().size() != 0; }
  const Indices &indices() const { return indices_; }
  index get_index(index pos) const { return indices()[pos]; }
  index size() const {
    return std::max<index>(0, 1 + (last_ - first_) / step_);
  }
  index is_full() const {
    return first_ == 0 && step_ == 1 &&
           (last_ == -1 || last_ == dimension_ - 1);
  }
  bool squeezed() const { return squeezed_; }
  bool maybe_combine(const Range &other);

  static Range empty();
  static Range empty(index dimension);
  static Range full(index first = 0, index step = 1);

 private:
  index first_{-1}, step_{1}, last_{-2}, dimension_{-1};
  Indices indices_;
  bool squeezed_{false};
};

extern const Range _;

Dimensions dimensions_from_ranges(SimpleVector<Range> &ranges,
                                  const Dimensions &parent_dimensions);

class RangeIterator {
 public:
  static const Range empty_range;
  typedef std::shared_ptr<RangeIterator> next_t;
  typedef Range *range_ptr_t;
  typedef enum { range_begin = 0, range_end = 1 } end_flag_t;

  static RangeIterator make_range_iterators(const SimpleVector<Range> &ranges,
                                            end_flag_t flag = range_begin);
  RangeIterator(const Range &r, index factor = 1, end_flag_t flag = range_begin,
                next_t next = nullptr);
  index operator*() const { return get_position(); };
  RangeIterator &operator++() {
    if (++counter_ >= limit_) {
      advance_next();
    } else {
      offset_ += step_;
    }
    return *this;
  }
  bool finished() const { return counter_ >= limit_; }
  bool operator!=(const RangeIterator &other) const {
    return other.counter_ != counter_;
  }
  bool operator==(const RangeIterator &other) const {
    return other.counter_ == counter_;
  }
  static RangeIterator begin(const SimpleVector<Range> &ranges) {
    return make_range_iterators(ranges, range_begin);
  }
  static RangeIterator end(const SimpleVector<Range> &ranges) {
    return make_range_iterators(ranges, range_end);
  }
  index get_position() const;
  index counter() const { return counter_; }
  index offset() const { return offset_; }
  index limit() const { return limit_; }
  index step() const { return step_; }
  bool has_next() const { return next_ != nullptr; }
  const RangeIterator &next() const { return *next_; }
  bool has_indices() const { return indices_.size() != 0; }
  const Indices &indices() const { return indices_; }

 private:
  index counter_, limit_, step_, offset_, factor_, start_;
  Indices indices_;
  next_t next_;
  void advance_next();
  static RangeIterator make_next_iterator(
      const Range *ranges, index left, index factor,
      end_flag_t end_flagflag = range_begin);
};

template <typename elt_t>
class TensorIterator {
 public:
  typedef elt_t value_type;
  typedef index difference_type;
  typedef elt_t &reference;
  typedef elt_t *pointer;
  typedef std::input_iterator_tag iterator_category;

  TensorIterator(RangeIterator &&it, elt_t *base, index size = 0)
      : iterator_{std::move(it)}, base_{base}, size_{size} {}
#ifdef NDEBUG
  elt_t &operator*() { return base_[iterator_.get_position()]; }
#else
  elt_t &operator*() {
    index n = iterator_.get_position();
    if (n < 0 || n >= size_) {
      throw iterator_overflow();
    }
    return base_[n];
  }
#endif
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
  index size_;
};

template <typename elt_t>
class TensorConstIterator {
 public:
  typedef const elt_t value_type;
  typedef index difference_type;
  typedef const elt_t &reference;
  typedef const elt_t *pointer;
  typedef std::input_iterator_tag iterator_category;

  TensorConstIterator(RangeIterator &&it, const elt_t *base)
      : iterator_{std::move(it)}, base_{base} {}
  const elt_t &operator*() { return base_[iterator_.get_position()]; }
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

/**Create a Range which only contains one index. \sa \ref sec_tensor_view*/
inline Range range(index ndx) { return Range(ndx); }
/**Create a Range start:end (Matlab notation). \sa \ref sec_tensor_view*/
inline Range range(index start, index end) { return Range(start, end); }
/**Create a Range start:step:end (Matlab notation). \sa \ref sec_tensor_view*/
inline Range range(index start, index end, index step) {
  return Range(start, end, step);
}
/**Create a Range with the give set of indices. \sa \ref sec_tensor_view*/
inline Range range(Indices i) { return Range(i); }
/**Create a Range which covers all indices. \ref sec_tensor_view*/
inline Range range() { return Range::full(); }

/**Return a vector of integers from 'start' to 'end' (included) in 'steps'. */
inline Indices iota(index start, index end, index step = 1) {
  return Indices::range(start, end, step);
}

template <size_t n>
bool operator==(const tensor::Indices &v2,
                const tensor::StaticVector<tensor::index, n> &v1) {
  tensor::Indices v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.cbegin(), v0.cend(), v2.cbegin());
}

template <size_t n>
bool operator==(const tensor::StaticVector<tensor::index, n> &v1,
                const tensor::Indices &v2) {
  tensor::Indices v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.cbegin(), v0.cend(), v2.cbegin());
}

}  // namespace tensor

/*@}*/
#endif  // !TENSOR_H
