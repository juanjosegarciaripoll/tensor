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

#ifndef TENSOR_SPARSE_CSR_MATRIX_H
#define TENSOR_SPARSE_CSR_MATRIX_H

#include <algorithm>
#include <tensor/exceptions.h>
#include <tensor/rand.h>
#include <tensor/sparse/types.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// CONSTRUCTORS
//

template <typename elt_t>
CSRMatrix<elt_t>::CSRMatrix()
    : dims_{0, 0}, row_start_({0}), column_(0), data_{Vector<elt_t>()} {}

template <typename elt_t>
CSRMatrix<elt_t>::CSRMatrix(index rows, index cols, index nonzero)
    : dims_{rows, cols},
      row_start_(static_cast<index_t>(safe_size_t(rows + 1))),
      column_(static_cast<index_t>(safe_size_t(nonzero))),
      data_(Vector<elt_t>(column_.size())) {
  std::fill(row_start_.begin(), row_start_.end(), 0);
}

template <typename elt_t>
CSRMatrix<elt_t>::CSRMatrix(Indices dims, Indices row_start, Indices column,
                            Tensor<elt_t> data)
    : dims_(std::move(dims)),
      row_start_(std::move(row_start)),
      column_(std::move(column)),
      data_(std::move(data)) {
  tensor_assert(row_start.ssize() == dims[0] + 1);
}

template <typename elt_t>
CSRMatrix<elt_t>::CSRMatrix(const Indices &row_indices,
                            const Indices &column_indices,
                            const Tensor<elt_t> &data, index rows,
                            index columns)
    : CSRMatrix(
          make_sparse(make_sparse_triplets(row_indices, column_indices, data),
                      rows, columns)) {}

template <typename elt_t>
CSRMatrix<elt_t>::CSRMatrix(std::vector<SparseTriplet<elt_t>> coordinates,
                            index_t rows, index_t columns)
    : CSRMatrix(make_sparse(coordinates, rows, columns)) {}

template <typename elt_t>
std::vector<SparseTriplet<elt_t>> CSRMatrix<elt_t>::make_sparse_triplets(
    const Indices &rows, const Indices &cols, const Tensor<elt_t> &data) {
  index l = rows.ssize();
  tensor_assert(cols.ssize() == l);
  tensor_assert(data.ssize() == l);
  std::vector<SparseTriplet<elt_t>> output;
  output.reserve(static_cast<size_t>(l));
  for (index_t i = 0; i < l; ++i) {
    auto d = data[i];
    if (d != number_zero<elt_t>()) {
      output.emplace_back(rows[i], cols[i], d);
    }
  }
  return output;
}

template <typename elt_t>
CSRMatrix<elt_t> CSRMatrix<elt_t>::make_sparse(
    std::vector<SparseTriplet<elt_t>> sorted_data, index nrows, index ncols) {
  if (sorted_data.size()==0) {
	return CSRMatrix<elt_t>(nrows, ncols, 0);
  }
  std::sort(sorted_data.begin(), sorted_data.end());
  nrows = std::max(nrows, sorted_data.back().row);

  auto row_start_ = Indices(nrows + 1);
  auto column_ = Indices(ssize(sorted_data));
  index_t l = column_.ssize();
  auto data_ = Tensor<elt_t>::empty(l);

  std::fill(row_start_.begin(), row_start_.end(), 0);
  index_t j = 0, last_row = -1, last_col = 0;
  for (const auto &d : sorted_data) {
    if (d.row == last_row) {
      if (d.col == last_col) continue;
    } else {
      while (last_row < d.row) {
        row_start_.at(++last_row) = j;
      }
    }
    column_.at(j) = d.col;
    data_.at(j) = d.value;
    j++;
  }
  while (last_row < nrows) {
    row_start_.at(++last_row) = j;
  }
  ncols = std::max(ncols, *std::max_element(column_.begin(), column_.end()));
  return CSRMatrix<elt_t>(Dimensions{nrows, ncols}, row_start_, column_, data_);
}

//////////////////////////////////////////////////////////////////////
// CONSTRUCTOR FROM FULL TENSOR TO SPARSE AND VICEVERSA
//

template <typename elt_t>
static index number_of_nonzero(const Tensor<elt_t> &data) {
  auto counter = std::count_if(
      data.cbegin(), data.cend(),
      [](const elt_t &value) { return value != number_zero<elt_t>(); });
  return static_cast<index>(counter);
}

template <typename elt_t>
CSRMatrix<elt_t>::CSRMatrix(const Tensor<elt_t> &t)
    : dims_(t.dimensions()),
      row_start_(t.rows() + 1),
      column_(number_of_nonzero<elt_t>(t)),
      data_(Tensor<elt_t>::empty(column_.size())) {
  index nrows = rows();
  index ncols = columns();

  Indices::iterator row_it = row_start_.begin();
  Indices::iterator col_it = column_.begin();
  Indices::iterator col_begin = col_it;
  typename Tensor<elt_t>::iterator data_it = data_.begin();

  *(row_it++) = 0;
  for (index r = 0; r < nrows; r++) {
    for (index c = 0; c < ncols; c++) {
      elt_t v = t(r, c);
      if (!(v == number_zero<elt_t>())) {
        *(data_it++) = v;
        *(col_it++) = c;
      }
    }
    *(row_it++) = col_it - col_begin;
  }
}

template <typename elt_t>
const Tensor<elt_t> full(const CSRMatrix<elt_t> &s) {
  index nrows = s.rows();
  index ncols = s.columns();
  auto output = Tensor<elt_t>::empty(nrows, ncols);
  if (nrows && ncols) {
    output.fill_with_zeros();

    Indices::const_iterator row_start = s.priv_row_start().begin();
    Indices::const_iterator column = s.priv_column().begin();
    typename Tensor<elt_t>::const_iterator data = s.priv_data().begin();

    for (index i = 0; i < nrows; i++) {
      for (index l = row_start[i + 1] - row_start[i]; l; l--) {
        output.at(i, *(column++)) = *(data++);
      }
    }
  }
  return output;
}

template <typename elt_t>
index CSRMatrix<elt_t>::dimension(int which) const {
  tensor_assert(which < 2);
  return which ? columns() : rows();
}

//////////////////////////////////////////////////////////////////////
// SPECIAL MATRICES CONSTRUCTORS
//

template <typename elt_t>
CSRMatrix<elt_t> CSRMatrix<elt_t>::eye(index rows, index columns) {
  index nel = std::min(rows, columns);
  auto data = Tensor<elt_t>::empty(safe_size_t(nel));
  std::fill(data.begin(), data.end(), number_one<elt_t>());
  Indices row_start(static_cast<index_t>(safe_size_t(rows + 1)));
  for (index i = 0; i <= rows; i++) {
    row_start.at(i) = std::min(i, nel);
  }
  return CSRMatrix({rows, columns},
                   row_start,                   // row_start
                   Indices::range(0, nel - 1),  // columns
                   data);
}

template <typename elt_t>
CSRMatrix<elt_t> CSRMatrix<elt_t>::random(index rows, index columns,
                                          double density) {
  auto output = Tensor<elt_t>::random(rows, columns);
  for (auto &value : output) {
    if (abs(value) > density) {
      value = number_zero<elt_t>();
    } else {
      value /= density;
    }
  }
  return CSRMatrix<elt_t>(output);
}

//////////////////////////////////////////////////////////////////////
// ACCESSING ELEMENTS
//
template <typename elt_t>
elt_t CSRMatrix<elt_t>::operator()(index row, index col) const {
  row = Dimensions::normalize_index(row, rows());
  col = Dimensions::normalize_index(col, columns());
  for (index ndx1 = row_start_[row], ndx2 = row_start_[row + 1]; ndx1 < ndx2;
       ndx1++) {
    index this_col = column_[ndx1];
    if (this_col == col) {
      return data_[ndx1];
    } else if (this_col > col) {
      break;
    }
  }
  return number_zero<elt_t>();
}

}  // namespace tensor

#endif  // !TENSOR_SPARSE_CSR_MATRIX_H
