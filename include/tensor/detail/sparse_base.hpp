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

#if !defined(TENSOR_SPARSE_H) || defined(TENSOR_DETAIL_SPARSE_BASE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_SPARSE_BASE_HPP

#include <cassert>
#include <tensor/rand.h>
#include <tensor/detail/common.h>

namespace tensor {

  //////////////////////////////////////////////////////////////////////
  // CONSTRUCTORS
  //

  static inline index
  safe_size(index nonzero, index rows, index cols)
  {
    /* The product rows*cols might overflow the word size of this machine */
    if (rows == 0 || cols == 0)
      return 0;
    assert((nonzero / rows) <= cols);
    return nonzero;
  }

  template<typename elt_t>
  Sparse<elt_t>::Sparse() :
    dims_(2), row_start_(1), column_(0), data_(0)
  {
    dims_.at(0) = dims_.at(1) = row_start_.at(0) = 0;
  }

  template<typename elt_t>
  Sparse<elt_t>::Sparse(index rows, index cols, index nonzero) :
    dims_(2), row_start_(rows+1), column_(safe_size(nonzero, rows, cols)),
    data_(safe_size(nonzero, rows, cols))
  {
    dims_.at(0) = rows;
    dims_.at(1) = cols;
    std::fill(row_start_.begin(), row_start_.end(), 0);
  }

  template<typename elt_t>
  Sparse<elt_t>::Sparse(const Sparse<elt_t> &s) :
    dims_(s.dims_), row_start_(s.row_start_), column_(s.column_),
    data_(s.data_)
  {
  }

  template<typename elt_t>
  Sparse<elt_t> &Sparse<elt_t>::operator=(const Sparse<elt_t> &s)
  {
    row_start_ = s.row_start_;
    column_ = s.column_;
    data_ = s.data_;
    dims_ = s.dims_;
    return *this;
  }

  //
  // CONSTRUCTOR FROM FULL TENSOR 
  //

  template<typename elt_t>
  static index
  number_of_nonzero(const Tensor<elt_t> &data)
  {
    ;
    index counter = 0;
    for (typename Tensor<elt_t>::const_iterator it = data.begin();
         it != data.end();
         it++)
    {
      if (!(*it == number_zero<elt_t>())) counter++;
    }
    return counter;
  }

  template<typename elt_t>
  Sparse<elt_t>::Sparse(const Tensor<elt_t> &t) :
    dims_(2), row_start_(), column_(), data_()
  {
    if (t.is_empty()) {
      dims_.at(0) = 0;
      dims_.at(1) = 0;
      return;
    }

    dims_ = t.dimensions();
    row_start_ = Indices(t.rows()+1);
    column_ = Indices(number_of_nonzero<elt_t>(t));
    data_ = Vector<elt_t>(column_.size());

    index nrows = rows();
    index ncols = columns();

    Indices::iterator row_it = row_start_.begin();
    Indices::iterator col_it = column_.begin();
    Indices::iterator col_begin = col_it;
    typename Vector<elt_t>::iterator data_it = data_.begin();

    *(row_it++) = 0;
    for (index r = 0; r < nrows; r++) {
      for (index c = 0; c < ncols; c++) {
        elt_t v = t(r,c);
        if (!(v == number_zero<elt_t>())) {
          *(data_it++) = v;
          *(col_it++) = c;
        }
      }
      *(row_it++) = col_it - col_begin;
    }
  }

  //////////////////////////////////////////////////////////////////////
  // SPECIAL MATRICES CONSTRUCTORS
  //

  template<typename elt_t>
  Sparse<elt_t> Sparse<elt_t>::eye(index rows, index columns)
  {
    index nel = std::min(rows, columns);
    Sparse<elt_t> output(rows, columns, nel);
    for (index k = 0; k < nel; ) {
      output.data_.at(k) = number_one<elt_t>();
      output.column_.at(k) = k;
      output.row_start_.at(++k) = k;
    }
    for (index k = nel; k < rows; k++) {
      output.row_start_.at(++k) = nel;
    }
    return output;
  }

  template<typename elt_t> Sparse<elt_t>
  Sparse<elt_t>::random(index rows, index columns, double density)
  {
    Tensor<elt_t> output(rows * columns);
    output.randomize();
    for (typename Tensor<elt_t>::iterator it = output.begin(),
         end = output.end();
	 it < end;
	 it++)
    {
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
  template<typename elt_t>
  elt_t Sparse<elt_t>::operator()(index row, index col) const
  {
    row = normalize_index(row, rows());
    col = normalize_index(col, columns());
    for (index ndx1 = row_start_[row], ndx2 = row_start_[row+1];
         ndx1 < ndx2; ndx1++)
    {
      index this_col = column_[ndx1];
      if (this_col == col) {
        return data_[ndx1];
      } else if (this_col > col) {
        break;
      }
    }
    return number_zero<elt_t>();
  }

} // namespace tensor

#endif // !TENSOR_DETAIL_SPARSE_BASE_HPP
