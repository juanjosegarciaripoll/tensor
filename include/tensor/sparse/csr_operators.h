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
#ifndef TENSOR_SPARSE_CSR_OPERATORS_H
#define TENSOR_SPARSE_CSR_OPERATORS_H

#include <functional>
#include <algorithm>
#include <type_traits>
#include <tensor/exceptions.h>
#include <tensor/sparse/types.h>
#include <tensor/sparse/csr_matrix.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// SPARSE MATRIX NEGATION
//
template <typename T>
const CSRMatrix<T> operator-(const CSRMatrix<T> &s) {
  return CSRMatrix<T>(s.dimensions(), s.priv_row_start(), s.priv_column(),
                      -s.priv_data());
}

//////////////////////////////////////////////////////////////////////
// MINIMAL ARITHMETICS WITH NUMBERS
//

template <typename T1, typename T2>
const CSRMatrix<std::common_type_t<T1, T2>> operator*(const CSRMatrix<T1> &s,
                                                      T2 b) {
  return CSRMatrix<std::common_type_t<T1, T2>>(
      s.dimensions(), s.priv_row_start(), s.priv_column(), s.priv_data() * b);
}

template <typename T1, typename T2>
const CSRMatrix<std::common_type_t<T1, T2>> operator/(const CSRMatrix<T1> &s,
                                                      T2 b) {
  return CSRMatrix<std::common_type_t<T1, T2>>(
      s.dimensions(), s.priv_row_start(), s.priv_column(), s.priv_data() / b);
}

template <typename T1, typename T2>
const CSRMatrix<std::common_type_t<T1, T2>> operator*(T1 b,
                                                      const CSRMatrix<T2> &s) {
  return s * b;
}

//////////////////////////////////////////////////////////////////////
// ARITHMETIC BETWEEN MATRICES
//

template <typename T1, typename T2, class binop>
Sparse<std::common_type_t<T1, T2>> sparse_binop(const Sparse<T1> &m1,
                                                const Sparse<T2> &m2,
                                                binop op) {
  using result_type = std::common_type_t<T1, T2>;

  index rows = m1.rows();
  index cols = m1.columns();

  tensor_assert(rows == m2.rows() && cols == m2.columns());

  if (rows == 0 || cols == 0) return Sparse<result_type>(rows, cols);

  size_t max_size = m1.priv_data().size() + m2.priv_data().size();
  Tensor<result_type> data(Dimensions{max_size});
  Indices column(max_size);
  Indices row_start(static_cast<size_t>(rows) + 1);

  auto out_data = data.begin();
  auto out_column = column.begin();
  auto out_row_start = row_start.begin();
  auto out_begin = out_data;

  auto m1_data = m1.priv_data().begin();
  auto m1_row_start = m1.priv_row_start().begin();
  auto m1_column = m1.priv_column().begin();

  auto m2_data = m2.priv_data().begin();
  auto m2_row_start = m2.priv_row_start().begin();
  auto m2_column = m2.priv_column().begin();

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
    result_type value;
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
    if (!(value == number_zero<result_type>())) {
      *(out_data++) = value;
      *(out_column++) = c;
    }
  }
  index j = out_data - out_begin;
  Indices the_column(safe_size_t(j));
  std::copy(column.begin(), column.begin() + j, the_column.begin());
  Tensor<result_type> the_data(Dimensions{j});
  std::copy(data.begin(), data.begin() + j, the_data.begin());
  return Sparse<result_type>(m1.dimensions(), row_start, the_column, the_data);
}

template <typename T1, typename T2>
const CSRMatrix<std::common_type_t<T1, T2>> operator+(const CSRMatrix<T1> &m1,
                                                      const CSRMatrix<T2> &m2) {
  using result_type = std::common_type_t<T1, T2>;
  return sparse_binop(m1, m2, std::plus<result_type>());
}

template <typename T1, typename T2>
const CSRMatrix<std::common_type_t<T1, T2>> operator-(const CSRMatrix<T1> &m1,
                                                      const CSRMatrix<T2> &m2) {
  using result_type = std::common_type_t<T1, T2>;
  return sparse_binop(m1, m2, std::minus<result_type>());
}

template <typename T1, typename T2>
const CSRMatrix<std::common_type_t<T1, T2>> operator*(const CSRMatrix<T1> &m1,
                                                      const CSRMatrix<T2> &m2) {
  using result_type = std::common_type_t<T1, T2>;
  return sparse_binop(m1, m2, std::multiplies<result_type>());
}

}  // namespace tensor

#endif  // !TENSOR_SPARSE_CSR_OPERATORS_H
