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

#if !defined(TENSOR_SPARSE_H) || defined(TENSOR_DETAIL_SPARSE_OPS_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_SPARSE_OPS_HPP

#include <cassert>
#include <functional>
#include <algorithm>
#include <tensor/detail/functional.h>

namespace tensor {

  //////////////////////////////////////////////////////////////////////
  // SPARSE MATRIX NEGATION
  //
  template<typename T>
  const Sparse<T> operator-(const Sparse<T> &s)
  {
    Vector<T> data(s.priv_data().size());
    std::for_each(data.begin(), data.end(), std::negate<T>());
    return Sparse<T>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                     data);
  }

  //////////////////////////////////////////////////////////////////////
  // MINIMAL ARITHMETICS WITH NUMBERS
  //

  template<typename T>
  const Sparse<T> operator*(const Sparse<T> &s, T b)
  {
    Vector<T> data(s.priv_data().size());
    std::for_each(data.begin(), data.end(), times_constant<T,T>(b));
    return Sparse<T>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                     data);
  }

  template<typename T>
  const Sparse<T> operator/(const Sparse<T> &s, T b)
  {
    Vector<T> data(s.priv_data().size());
    std::for_each(data.begin(), data.end(), divided_constant<T,T>(b));
    return Sparse<T>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                     data);
  }

  template<typename T>
  const Sparse<T> operator*(T b, const Sparse<T> &s)
  {
    return s * b;
  }

  //////////////////////////////////////////////////////////////////////
  // ARITHMETIC BETWEEN MATRICES
  //

  template<typename T, class binop>
  const Sparse<T> sparse_binop(const Sparse<T> &m1, const Sparse<T> &m2,
                               binop op)
  {
    size_t rows = m1.rows();
    size_t cols = m1.columns();

    assert(rows == m2.rows() && cols == m2.columns());

    if (rows == 0)
      return m1;

    index max_size = m1.priv_data().size() + m2.priv_data().size();
    Vector<T> data(max_size);
    Indices column(max_size);
    Indices row_start(rows);

    typename Vector<T>::iterator out_data = data.begin();
    typename Indices::iterator out_column = column.begin();
    typename Indices::iterator out_row_start = row_start.begin();
    typename Vector<T>::iterator out_begin = out_data;

    typename Vector<T>::const_iterator m1_data = m1.priv_data().begin();
    typename Indices::const_iterator m1_row_start = m1.priv_row_start().begin();
    typename Indices::const_iterator m1_column = m1.priv_column().begin();

    typename Vector<T>::const_iterator m2_data = m2.priv_data().begin();
    typename Indices::const_iterator m2_row_start = m2.priv_row_start().begin();
    typename Indices::const_iterator m2_column = m2.priv_column().begin();

    size_t j1 = *(m1_row_start++), j2 = *(m2_row_start++);
    size_t l1 = (*m1_row_start) - j1;
    size_t l2 = (*m2_row_start) - j2;
    while (1) {
      // We look for the next unprocessed matrix element on this row,
      // for both matrices. c1 and c2 are the columns associated to
      // each element on each matrix.
      index c1 = l1 ? *m1_column : cols;
      index c2 = l2 ? *m2_column : cols;
      T value;
      index c;
      if (c1 < c2) {
        // There is an element a column c1 on matrix m1, but the
        // same element at m2 is zero
        value = op(*m1_data, number_zero<T>());
        c = c1;
        l1--; m1_column++; m1_data++;
      } else if (c2 < c1) {
        // There is an element a column c2 on matrix m2, but the
        // same element at m1 is zero
        value = op(number_zero<T>(), *m2_data);
        c = c2;
        l2--; m2_column++; m2_data++;
      } else if (c2 < cols) {
        // Both elements in m1 and m2 are nonzero.
        value = op(*m1_data, *m2_data);
        c = c1;
        l1--; l2--;
        m1_column++; m1_data++;
        m2_column++; m2_data++;
      } else {
        // We have processed all elements in this row.
        out_row_start++;
        *out_row_start = out_data - out_begin;
        if (--rows == 0) {
          break;
        }
        j1 = *m1_row_start; m1_row_start++; l1 = (*m1_row_start) - j1;
        j2 = *m2_row_start; m2_row_start++; l2 = (*m2_row_start) - j2;
        continue;
      }
      if (!(value == number_zero<T>())) {
        *(out_data++) = value;
        *(out_column++) = c2;
      }
    }
    size_t j = out_data - out_begin;
    Indices the_column(j);
    std::copy(the_column.begin(), the_column.end(), column.begin());
    Vector<T> the_data(j);
    std::copy(the_data.begin(), the_data.end(), data.begin());
    return Sparse<T>(m1.dimensions(), row_start, the_column, the_data);
  }

  template<typename T>
  const Sparse<T> operator+(const Sparse<T> &m1, const Sparse<T> &m2)
  {
    return sparse_binop(m1, m2, std::plus<T>());
  }

  template<typename T>
  const Sparse<T> operator-(const Sparse<T> &m1, const Sparse<T> &m2)
  {
    return sparse_binop(m1, m2, std::minus<T>());
  }

  template<typename T>
  const Sparse<T> operator*(const Sparse<T> &m1, const Sparse<T> &m2)
  {
    return sparse_binop(m1, m2, std::multiplies<T>());
  }

}

#endif // !TENSOR_DETAIL_SPARSE_OPS_HPP
