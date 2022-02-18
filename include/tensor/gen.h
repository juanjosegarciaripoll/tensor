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
#ifndef TENSOR_GEN_H
#define TENSOR_GEN_H

#include <tensor/numbers.h>
#include <tensor/vector.h>

/*!\addtogroup Tensors */
/*@{*/
namespace tensor {

/** Compile-time generator of vectors, tensors and indices. This placeholder
      can be used to statically create arrays of data that can then be coerced
      to Vector, Tensor and Indices types. There are three generators tensor::igen,
      tensor::rgen and tensor::cgen of types ListGenerator<t> with t = tensor::index,
      double and cdouble, which can be used as shown in \ref sec_tensor_gen
\code
RTensor a = rgen << 2 << 3 << 4;
Indices d = igen << 1 << 2 << 3 << 5;
\endcode

  */
template <typename elt_t>
class ListGenerator {};

/** Placeholder for statically creating a vector of indices. For example
\code
Indices d = igen << 2 << 2;
RTensor a(igen << 1 << 3);
\endcode
\sa ListGenerator */
extern const ListGenerator<index> igen;
/** Placeholder for statically creating a vector of reals.
\code
RTensor a = rgen << 1 << 3;
\endcode
\sa ListGenerator */
extern const ListGenerator<double> rgen;
/** Placeholder for statically creating a vector of complex numbers.
\code
CTensor a = cgen << 1 << 3;
\endcode
\sa ListGenerator */
extern const ListGenerator<cdouble> cgen;
/** Placeholder for statically creating a vector of complex numbers.
\code
Booleans a = bgen << true << false;
\endcode
\sa ListGenerator */
extern const ListGenerator<bool> bgen;

/** Placeholder for statically creating a vector of elements, whose type
is determined by the first element.
\code
RTensor a = gen << (double)1.0 << 3;
\endcode
\sa ListGenerator */
extern class FlexiListGenerator {
} xgen;

template <typename elt_t, size_t n>
class StaticVector {
 public:
  StaticVector(const StaticVector<elt_t, n - 1> &other, elt_t x)
      : inner(other), extra(x){};
  void push(elt_t *v) const {
    inner.push(v);
    v[n - 1] = extra;
  }
  size_t size() const { return n; }
  index ssize() const { return static_cast<index>(n); }

 protected:
  StaticVector<elt_t, n - 1> inner;
  elt_t extra;
};

template <typename elt_t>
class StaticVector<elt_t, static_cast<size_t>(1)> {
 public:
  StaticVector(elt_t x) : extra(x){};
  operator Vector<elt_t>() const {
    Vector<elt_t> output(1);
    push(output.begin());
    return output;
  }
  void push(elt_t *v) const { v[0] = extra; }
  size_t size() const { return 1; }
  index ssize() const { return 1; }

 private:
  elt_t extra;
};

template <typename t1, typename t2>
StaticVector<t1, 1> operator<<(const ListGenerator<t1> &, const t2 x) {
  return StaticVector<t1, 1>(x);
}

template <typename t1>
StaticVector<t1, 1> operator<<(const FlexiListGenerator &, const t1 x) {
  return StaticVector<t1, 1>(x);
}

template <typename t1, typename t2, size_t n>
StaticVector<t1, n + 1> operator<<(const StaticVector<t1, n> &g, const t2 x) {
  return StaticVector<t1, n + 1>(g, x);
}

template <typename t1>
StaticVector<t1, 1> gen(t1 r) {
  return StaticVector<t1, 1>(r);
}

template <typename elt_t, size_t n>
bool operator==(const tensor::StaticVector<elt_t, n> &v1,
                const tensor::Vector<elt_t> &v2) {
  tensor::Vector<elt_t> v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

template <typename elt_t, size_t n>
bool operator==(const tensor::Vector<elt_t> &v2,
                const tensor::StaticVector<elt_t, n> &v1) {
  tensor::Vector<elt_t> v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

}  // namespace tensor
/*@}*/

#endif  // TENSOR_GEN_H
