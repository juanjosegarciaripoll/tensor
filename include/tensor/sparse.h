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


#ifndef TENSOR_SPARSE_H
#define TENSOR_SPARSE_H

#include <tensor/tensor.h>

namespace tensor {

  /**A sparse matrix. A sparse matrix is a compact representation of
     two-dimensional tensors that have a lot of zero elements. Our
     implementation behaves much like Matlab's sparse matrices, in the sense
     that one can build them up from 2D tensors, preallocate them, perform
     matrix multiplication with mmult(), etc, etc.

     \ingroup Tensors
  */
  template<typename elt>
  class Sparse {
  public:
    typedef elt elt_t;
    typedef Tensor<elt> tensor;

    /**Build an empty matrix.*/
    Sparse();
    /**Create a matrix with all elements set to zero.*/
    Sparse(index rows, index cols, index nonzero = 0);
    /**Convert a tensor to sparse form.*/
    explicit Sparse(const Tensor<elt_t> &tensor);
    /**Copy constructor.*/
    Sparse(const Sparse<elt_t> &s);
    /**Assignment operator.*/
    Sparse &operator=(const Sparse<elt_t> &s);

    /**Return an element of the sparse matrix.*/
    elt_t operator()(index row, index col) const;

    /**Return Sparse matrix dimensions.*/
    const Indices &dimensions() const { return dims_; }
    /**Length of a given Sparse matrix index.*/
    index dimension(int which) const;
    /**Number of rows.*/
    index rows() const { return dims_[0]; }
    /**Number of columns*/
    index columns() const { return dims_[1]; }
    /**Number of nonzero elements.*/
    index length() const { index r = rows(); return r? row_start_[r] : 0; }

    /**Empty matrix?*/
    bool is_empty() const { return (rows() == 0)||(columns() == 0); }

    /**Identity matrix in sparse form.*/
    static Sparse<elt_t> eye(index rows, index cols);
    /**Identity matrix in sparse form.*/
    static Sparse<elt_t> eye(index rows) { return eye(rows,rows); }
    /**Return a random sparse matrix.*/
    static Sparse<elt_t> random(index rows, index columns, double density = 0.2);

    template<typename t> friend const Tensor<t> full(const Sparse<t> &s);

  public:
    Indices dims_;
    Indices row_start_;
    Indices column_;
    Vector<elt_t> data_;
  };

  typedef Sparse<double> RSparse;
  typedef Sparse<cdouble> CSparse;
  const CSparse to_complex(const RSparse &s);
  inline const CSparse to_complex(const CSparse &c) { return c; }

} // namespace tensor

#ifdef TENSOR_LOAD_IMPL
#include <tensor/detail/sparse_base.hpp>
#endif

#endif // !TENSOR_SPARSE_H
