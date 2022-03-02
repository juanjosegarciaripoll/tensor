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

#if !defined(TENSOR_SPARSE_H)
#error "This header cannot be included manually"
#endif
#ifndef TENSOR_DETAIL_SPARSE_OPS_HPP
#define TENSOR_DETAIL_SPARSE_OPS_HPP

#include <functional>
#include <algorithm>
#include <type_traits>
#include <tensor/io.h>
#include <tensor/exceptions.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// SPARSE MATRIX NEGATION
//
template <typename T>
const Sparse<T> operator-(const Sparse<T> &s) {
  return Sparse<T>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                   -s.priv_data());
}

//////////////////////////////////////////////////////////////////////
// MINIMAL ARITHMETICS WITH NUMBERS
//

template <typename T1, typename T2>
const Sparse<typename std::common_type<T1, T2>::type> operator*(
    const Sparse<T1> &s, T2 b) {
  return Sparse<T3>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                    s.priv_data() * b);
}

template <typename T1, typename T2>
const Sparse<typename std::common_type<T1, T2>::type> operator/(
    const Sparse<T1> &s, T2 b) {
  return Sparse<T3>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                    s.priv_data() / a);
}

template <typename T1, typename T2>
const Sparse<typename std::common_type<T1, T2>::type> operator*(
    T1 b, const Sparse<T2> &s) {
  return s * b;
}

//////////////////////////////////////////////////////////////////////
// ARITHMETIC BETWEEN MATRICES
//

template <typename T1, typename T2, class binop>
const Sparse<typename std::common_type<T1, T2>::type> sparse_binop(
    const Sparse<T1> &m1, const Sparse<T2> &m2, binop op) {
  typedef typename std::common_type<T1, T2>::type T3;

  size_t rows = m1.rows();
  size_t cols = m1.columns();

  tensor_assert(rows == m2.rows() && cols == m2.columns());

  if (rows == 0 || cols == 0) return Sparse<T3>(rows, cols);

  index max_size = m1.priv_data().size() + m2.priv_data().size();
  Tensor<T3> data(max_size);
  Indices column(max_size);
  Indices row_start(rows + 1);

  typename Tensor<T3>::iterator out_data = data.begin();
  typename Indices::iterator out_column = column.begin();
  typename Indices::iterator out_row_start = row_start.begin();
  typename Tensor<T3>::iterator out_begin = out_data;

  typename Tensor<T1>::const_iterator m1_data = m1.priv_data().begin();
  typename Indices::const_iterator m1_row_start = m1.priv_row_start().begin();
  typename Indices::const_iterator m1_column = m1.priv_column().begin();

  typename Tensor<T2>::const_iterator m2_data = m2.priv_data().begin();
  typename Indices::const_iterator m2_row_start = m2.priv_row_start().begin();
  typename Indices::const_iterator m2_column = m2.priv_column().begin();

  index j1 = *(m1_row_start++);     // data start for this row in M1
  index l1 = (*m1_row_start) - j1;  // # elements in this row in M1
  index j2 = *(m2_row_start++);     // data start for this row in M2
  index l2 = (*m2_row_start) - j2;  // # elements in this row in M2
  *out_row_start = 0;
  while (1) {
    // We look for the next unprocessed matrix element on this row,
    // for both matrices. c1 and c2 are the columns associated to
    // each element on each matrix.
    index c1 = l1 ? *m1_column : cols;
    index c2 = l2 ? *m2_column : cols;
    T3 value;
    index c;
    if (c1 < c2) {
      // There is an element a column c1 on matrix m1, but the
      // same element at m2 is zero
      value = op(*m1_data, number_zero<T2>());
      c = c1;
      l1--;
      m1_column++;
      m1_data++;
    } else if (c2 < c1) {
      // There is an element a column c2 on matrix m2, but the
      // same element at m1 is zero
      value = op(number_zero<T1>(), *m2_data);
      c = c2;
      l2--;
      m2_column++;
      m2_data++;
    } else if (c2 < cols) {
      // Both elements in m1 and m2 are nonzero.
      value = op(*m1_data, *m2_data);
      c = c1;
      l1--;
      l2--;
      m1_column++;
      m1_data++;
      m2_column++;
      m2_data++;
    } else {
      // We have processed all elements in this row.
      out_row_start++;
      *out_row_start = out_data - out_begin;
      if (--rows == 0) {
        break;
      }
      j1 = *m1_row_start;
      m1_row_start++;
      l1 = (*m1_row_start) - j1;
      j2 = *m2_row_start;
      m2_row_start++;
      l2 = (*m2_row_start) - j2;
      continue;
    }
    if (!(value == number_zero<T3>())) {
      *(out_data++) = value;
      *(out_column++) = c;
    }
  }
  index j = out_data - out_begin;
  Indices the_column(j);
  std::copy(column.begin(), column.begin() + j, the_column.begin());
  Tensor<T3> the_data(j);
  std::copy(data.begin(), data.begin() + j, the_data.begin());
  return Sparse<T3>(m1.dimensions(), row_start, the_column, the_data);
}

template <typename T1, typename T2>
const Sparse<typename std::common_type<T1, T2>::type> operator+(
    const Sparse<T1> &m1, const Sparse<T2> &m2) {
  typedef typename std::common_type<T1, T2>::type T3;
  return sparse_binop(m1, m2, std::plus<T3>());
}

template <typename T1, typename T2>
const Sparse<typename std::common_type<T1, T2>::type> operator-(
    const Sparse<T1> &m1, const Sparse<T2> &m2) {
  typedef typename std::common_type<T1, T2>::type T3;
  return sparse_binop(m1, m2, std::minus<T3>());
}

template <typename T1, typename T2>
const Sparse<typename std::common_type<T1, T2>::type> operator*(
    const Sparse<T1> &m1, const Sparse<T2> &m2) {
  typedef typename std::common_type<T1, T2>::type T3;
  return sparse_binop(m1, m2, std::multiplies<T3>());
}

}  // namespace tensor

#endif  // !TENSOR_DETAIL_SPARSE_OPS_HPP
