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
    /**Create a sparse matrix from the coordinates and values. */
    Sparse(const Indices &row_indices, const Indices &column_indices, const Vector<elt_t> &data,
           index rows = 0, index columns = 0);
    /* Create a sparse matrix from its internal representation. */
    Sparse(const Indices &dims, const Indices &row_start,
           const Indices &column, const Vector<elt_t> &data);
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

    const Indices &priv_dims() const { return dims_; }
    const Indices &priv_row_start() const { return row_start_; }
    const Indices &priv_column() const { return column_; }
    const Vector<elt> &priv_data() const { return data_; }

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

  //////////////////////////////////////////////////////////////////////
  //
  // Unary operations
  //

  template<typename t>
  const Sparse<t> operator-(const Sparse<t> &);

  //
  // Binary operations
  //

  template<typename t>
  const Sparse<t> operator*(t b, const Sparse<t> &s);
  template<typename t>
  const Sparse<t> operator*(const Sparse<t> &s, t b);
  template<typename t>
  const Sparse<t> operator/(const Sparse<t> &s, t b);

  template<typename t>
  const Sparse<t> operator+(const Sparse<t> &m1, const Sparse<t> &m2);
  template<typename t>
  const Sparse<t> operator-(const Sparse<t> &m1, const Sparse<t> &m2);
  template<typename t>
  const Sparse<t> operator*(const Sparse<t> &m1, const Sparse<t> &m2);

  /**Kronecker product between matrices, in Matlab order.*/
  template<typename t>
  const Sparse<t> kron(const Sparse<t> &s1, const Sparse<t> &s2);
  /**Kronecker product between matrices, opposite to Matlab order.*/
  template<typename t>
  const Sparse<t> kron2(const Sparse<t> &s1, const Sparse<t> &s2);

  //
  // Comparison
  //
  template<typename t1, typename t2>
  inline bool all_equal(const Sparse<t1> &s1, const Sparse<t2> &s2) {
    return all_equal(s1.dimensions(), s2.dimensions()) &&
      all_equal(s1.priv_row_start(), s2.priv_row_start()) &&
      all_equal(s1.priv_column(), s2.priv_column()) &&
      all_equal(s1.priv_data(), s2.priv_data());
  }

  template<typename t1, typename t2>
  inline bool all_equal(const Sparse<t1> &s1, const Tensor<t2> &s2) {
    return all_equal(full(s1), s2);
  }

  template<typename t1, typename t2>
  inline bool all_equal(const Tensor<t1> &s1, const Sparse<t2> &s2) {
    return all_equal(full(s2), s1);
  }

  const RTensor mmult(const RTensor &m1, const RSparse &m2);
  const CTensor mmult(const CTensor &m1, const CSparse &m2);
  const RTensor mmult(const RSparse &m1, const RTensor &m2);
  const CTensor mmult(const CSparse &m1, const CTensor &m2);

  const RSparse &real(const RSparse &A) { return A; }
  const RSparse real(const CSparse &A);
  const RSparse &conj(const RSparse &A) { return A; }
  const CSparse conj(const CSparse &A);
  const RSparse imag(const RSparse &A) { return RSparse(A.rows(), A.columns()); }
  const RSparse imag(const CSparse &A);

} // namespace tensor

#ifdef TENSOR_LOAD_IMPL
#include <tensor/detail/sparse_base.hpp>
#include <tensor/detail/sparse_kron.hpp>
#include <tensor/detail/sparse_ops.hpp>
#endif

#endif // !TENSOR_SPARSE_H
