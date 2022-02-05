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
#if !defined(TENSOR_SPARSE_H)
#error "This header cannot be included manually"
#endif
#ifndef TENSOR_DETAIL_SPARSE_BASE_HPP
#define TENSOR_DETAIL_SPARSE_BASE_HPP

#include <cassert>
#include <algorithm>
#include <tensor/rand.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// CONSTRUCTORS
//

static inline size_t safe_size(index nonzero, index rows, index cols) {
  /* The product rows*cols might overflow the word size of this machine */
  if (rows == 0 || cols == 0) {
    return 0;
  }
  assert((nonzero / rows) <= cols);
  return safe_size_t(nonzero);
}

template <typename elt_t>
Sparse<elt_t>::Sparse()
    : dims_{0, 0}, row_start_({0}), column_(0), data_{Vector<elt_t>()} {}

template <typename elt_t>
Sparse<elt_t>::Sparse(index rows, index cols, index nonzero)
    : dims_{rows, cols},
      row_start_(safe_size_t(rows + 1)),
      column_(safe_size(nonzero, rows, cols)),
      data_(Vector<elt_t>(safe_size(nonzero, rows, cols))) {
  std::fill(row_start_.begin(), row_start_.end(), 0);
}

template <typename elt_t>
struct sparse_triplet {
  index row, col;
  elt_t value;
  sparse_triplet(){};
  sparse_triplet(index r, index c, elt_t v) : row(r), col(c), value(v) {}
  bool operator<(const sparse_triplet &other) const {
    if (row < other.row) return 1;
    if (row == other.row) return col < other.col;
    return 0;
  }
};

template <typename elt_t>
Sparse<elt_t>::Sparse(const Indices &dims, const Indices &row_start,
                      const Indices &column, const Tensor<elt_t> &data)
    : dims_(dims), row_start_(row_start), column_(column), data_(data) {
  assert(row_start.ssize() == dims[0] + 1);
}

template <typename elt_t>
Sparse<elt_t>::Sparse(const Indices &rows, const Indices &cols,
                      const Tensor<elt_t> &data, index nrows, index ncols)
    : Sparse(make_sparse(rows, cols, data, nrows, ncols)) {}

template <typename elt_t>
Sparse<elt_t> make_sparse(const Indices &rows, const Indices &cols,
                          const Tensor<elt_t> &data, index nrows, index ncols) {
  index i, j, last_row, last_col, l = rows.ssize();
  assert(cols.ssize() == l);
  assert(data.ssize() == l);

  /* Organize data in sparse_triplets (row,column,value), sorted in the
     * same order in which we store data in Sparse
     */
  std::vector<sparse_triplet<elt_t> > sorted_data;
  sorted_data.reserve(safe_size_t(l));
  for (i = 0; i < l; i++) {
    index r = rows[i];
    index c = cols[i];
    nrows = std::max(nrows, r);
    ncols = std::max(ncols, c);
    if (data[i] != number_zero<elt_t>()) {
      sorted_data.push_back(sparse_triplet<elt_t>(r, c, data[i]));
    }
  }
  std::sort(sorted_data.begin(), sorted_data.end());
  auto row_start_ = Indices(static_cast<size_t>(nrows) + 1);
  auto column_ = Indices(sorted_data.size());
  l = column_.ssize();
  auto data_ = Tensor<elt_t>::empty(l);

  /* Fill in the Sparse structure.
     */
  std::fill(row_start_.begin(), row_start_.end(), 0);
  for (last_col = 0, last_row = -1, j = i = 0; i < l; i++) {
    const sparse_triplet<elt_t> &d = sorted_data[static_cast<size_t>(i)];
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
  return Sparse<elt_t>(Dimensions{nrows, ncols}, row_start_, column_, data_);
}

//////////////////////////////////////////////////////////////////////
// CONSTRUCTOR FROM FULL TENSOR TO SPARSE AND VICEVERSA
//

template <typename elt_t>
static index number_of_nonzero(const Tensor<elt_t> &data) {
  index counter = 0;
  for (typename Tensor<elt_t>::const_iterator it = data.begin();
       it != data.end(); it++) {
    if (!(*it == number_zero<elt_t>())) counter++;
  }
  return counter;
}

template <typename elt_t>
Sparse<elt_t>::Sparse(const Tensor<elt_t> &t)
    : dims_(t.dimensions()),
      row_start_(static_cast<size_t>(t.rows()) + 1),
      column_(),
      data_() {
  column_ = Indices(static_cast<size_t>(number_of_nonzero<elt_t>(t)));
  data_ = Tensor<elt_t>::empty(column_.size());

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
const Tensor<elt_t> full(const Sparse<elt_t> &s) {
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
index Sparse<elt_t>::dimension(int dimension) const {
  assert(dimension < 2);
  return dimension ? columns() : rows();
}

//////////////////////////////////////////////////////////////////////
// SPECIAL MATRICES CONSTRUCTORS
//

template <typename elt_t>
Sparse<elt_t> Sparse<elt_t>::eye(index rows, index columns) {
  index nel = std::min(rows, columns);
  auto data = Tensor<elt_t>::empty(safe_size_t(nel));
  std::fill(data.begin(), data.end(), number_one<elt_t>());
  Indices row_start(safe_size_t(rows) + 1);
  for (index i = 0; i <= rows; i++) {
    row_start.at(i) = std::min(i, nel);
  }
  return Sparse(igen << rows << columns,
                row_start,                   // row_start
                Indices::range(0, nel - 1),  // columns
                data);
}

template <typename elt_t>
Sparse<elt_t> Sparse<elt_t>::random(index rows, index columns, double density) {
  auto output = Tensor<elt_t>::empty(rows * columns);
  output.randomize();
  for (typename Tensor<elt_t>::iterator it = output.begin(), end = output.end();
       it < end; it++) {
    if (abs(*it) > density)
      *it = number_zero<elt_t>();
    else
      *it /= density;
  }
  return Sparse<elt_t>(reshape(output, rows, columns));
}

//////////////////////////////////////////////////////////////////////
// ACCESSING ELEMENTS
//
template <typename elt_t>
elt_t Sparse<elt_t>::operator()(index row, index col) const {
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

#endif  // !TENSOR_DETAIL_SPARSE_BASE_HPP
