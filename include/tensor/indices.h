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

#include <iterator>
#include <algorithm>
#include <tensor/vector.h>
#include <tensor/exceptions.h>
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

  index total_size() const { return total_size_; }
  index rank() const { return dimensions_.ssize(); }

  index operator[](index pos) const { return dimensions_[pos]; }
  const_iterator begin() const { return dimensions_.begin(); }
  const_iterator end() const { return dimensions_.end(); }
  const SimpleVector<index> &get_vector() const { return dimensions_; }

  static inline index normalize_index(index i,
                                      index dimension) tensor_noexcept {
    if (i < 0) i += dimension;
    tensor_assert2((i < dimension) && (i >= 0), out_of_bounds_index());
    return i;
  }

  static inline index normalize_index_safe(index i, index dimension) {
    if (i < 0) i += dimension;
    tensor_assert2((i < dimension) && (i >= 0), out_of_bounds_index());
    return i;
  }

  template <typename... index_like>
  index column_major_position(index i0, index_like... in) const {
    tensor_assert(rank() == sizeof...(in) + 1);
    return column_major_inner(0, i0, in...);
  }

  template <typename... index_like>
  void get_values(index_like *...in) const {
    tensor_assert(rank() == sizeof...(in));
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
  Booleans(const std::initializer_list<bool> &l) : Vector<bool>(l) {}
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

/**Return a vector of integers from 'start' to 'end' (included) in 'steps'. */
inline Indices iota(index start, index end, index step = 1) {
  return Indices::range(start, end, step);
}

}  // namespace tensor

/*@}*/
#endif  // !TENSOR_INDICES_H
