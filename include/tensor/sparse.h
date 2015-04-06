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
    Sparse(const Indices &row_indices, const Indices &column_indices,
           const Tensor<elt_t> &data,
           index rows = 0, index columns = 0);
    /* Create a sparse matrix from its internal representation. */
    Sparse(const Indices &dims, const Indices &row_start,
           const Indices &column, const Tensor<elt_t> &data);
    /**Convert a tensor to sparse form.*/
    explicit Sparse(const Tensor<elt_t> &tensor);
    /**Copy constructor.*/
    Sparse(const Sparse<elt_t> &s);
    /**Assignment operator.*/
    Sparse &operator=(const Sparse<elt_t> &s);
    /**Implicit conversion from other sparse types.*/
    template<typename e2> Sparse(const Sparse<e2> &other) :
      dims_(other.dims_), row_start_(other.row_start_),
      column_(other.column_), data_(other.data_)
    {}

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
    const Tensor<elt> &priv_data() const { return data_; }

  public:
    /** The dimensions (rows and columns) of the sparse matrix. */
    Indices dims_;
    /** Gives for each row of the matrix at which index the column_/data_ entries start. */
    Indices row_start_;
    /** Gives for each data_ entry the column in the matrix. */
    Indices column_;
    /** The single data entries. */
    Tensor<elt_t> data_;
  };

  typedef Sparse<double> RSparse;
  typedef Sparse<cdouble> CSparse;
  const CSparse to_complex(const RSparse &s);
  inline const CSparse to_complex(const CSparse &c) { return c; }

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

  /* Adjoint of a sparse matrix. */
  const RSparse adjoint(const RSparse &s);
  /* Transpose of a sparse matrix. */
  const RSparse transpose(const RSparse &s);

  /* Adjoint of a sparse matrix. */
  const CSparse adjoint(const CSparse &s);
  /* Transpose of a sparse matrix. */
  const CSparse transpose(const CSparse &s);

  /* Matrix multiplication between tensor and sparse matrix. */
  const RTensor mmult(const RTensor &m1, const RSparse &m2);
  /* Matrix multiplication between tensor and sparse matrix. */
  const CTensor mmult(const CTensor &m1, const CSparse &m2);
  /* Matrix multiplication between tensor and sparse matrix. */
  const RTensor mmult(const RSparse &m1, const RTensor &m2);
  /* Matrix multiplication between tensor and sparse matrix. */
  const CTensor mmult(const CSparse &m1, const CTensor &m2);

  /* Real part of a sparse matrix.*/
  inline const RSparse &real(const RSparse &A) { return A; }
  /* Conjugate of a sparse matrix.*/
  inline const RSparse &conj(const RSparse &A) { return A; }
  /* Imaginary part of a sparse matrix.*/
  inline const RSparse imag(const RSparse &A) {
    return RSparse(A.rows(), A.columns());
  }

  /* Real part of a sparse matrix.*/
  const RSparse real(const CSparse &A);
  /* Conjugate of a sparse matrix.*/
  const CSparse conj(const CSparse &A);
  /* Imaginary part of a sparse matrix.*/
  const RSparse imag(const CSparse &A);

  const RSparse operator-(const RSparse &a);
  const RSparse operator+(const RSparse &a, const RSparse &b);
  const RSparse operator-(const RSparse &a, const RSparse &b);
  const RSparse operator*(const RSparse &a, const RSparse &b);
  const RSparse operator*(const RSparse &a, double b);
  const RSparse operator/(const RSparse &a, double b);
  const RSparse operator*(double a, const RSparse &b);

  const CSparse operator-(const CSparse &a);
  const CSparse operator+(const CSparse &a, const CSparse &b);
  const CSparse operator-(const CSparse &a, const CSparse &b);
  const CSparse operator*(const CSparse &a, const CSparse &b);
  const CSparse operator*(const CSparse &a, cdouble b);
  const CSparse operator/(const CSparse &a, cdouble b);
  const CSparse operator*(cdouble a, const CSparse &b);

  const CSparse operator+(const CSparse &a, const RSparse &b);
  const CSparse operator-(const CSparse &a, const RSparse &b);
  const CSparse operator*(const CSparse &a, const RSparse &b);
  const CSparse operator+(const RSparse &a, const CSparse &b);
  const CSparse operator-(const RSparse &a, const CSparse &b);
  const CSparse operator*(const RSparse &a, const CSparse &b);

  const CSparse operator*(const CSparse &a, double b);
  const CSparse operator/(const CSparse &a, double b);
  const CSparse operator*(double a, const CSparse &b);

  const CSparse operator*(const RSparse &a, cdouble b);
  const CSparse operator/(const RSparse &a, cdouble b);
  const CSparse operator*(cdouble a, const RSparse &b);

  /**Kronecker product between matrices, in Matlab order.*/
  const RSparse kron(const RSparse &s1, const RSparse &s2);
  /**Kronecker product between matrices, opposite to Matlab order.*/
  const RSparse kron2(const RSparse &s1, const RSparse &s2);
  /**Implements A+B where A and B act on different spaces of a tensor product.*/
  const RSparse kron2_sum(const RSparse &s1, const RSparse &s2);

  /**Kronecker product between matrices, in Matlab order.*/
  const CSparse kron(const CSparse &s1, const CSparse &s2);
  /**Kronecker product between matrices, opposite to Matlab order.*/
  const CSparse kron2(const CSparse &s1, const CSparse &s2);
  /**Implements A+B where A and B act on different spaces of a tensor product.*/
  const CSparse kron2_sum(const CSparse &s1, const CSparse &s2);

} // namespace tensor

#ifdef TENSOR_LOAD_IMPL
#include <tensor/detail/sparse_base.hpp>
#endif

#endif // !TENSOR_SPARSE_H
