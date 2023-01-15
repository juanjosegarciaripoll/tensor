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

#ifndef TENSOR_SPARSE_H
#define TENSOR_SPARSE_H

#include <tensor/tensor.h>
#include <tensor/sparse/types.h>
#include <tensor/sparse/csr_matrix.h>

namespace tensor {

CSparse to_complex(const RSparse &s);
inline const CSparse &to_complex(const CSparse &c) { return c; }

//
// Comparison
//
template <typename t1, typename t2>
inline bool all_equal(const Sparse<t1> &s1, const Sparse<t2> &s2) {
  return all_equal(s1.dimensions(), s2.dimensions()) &&
         all_equal(s1.priv_row_start(), s2.priv_row_start()) &&
         all_equal(s1.priv_column(), s2.priv_column()) &&
         all_equal(s1.priv_data(), s2.priv_data());
}

template <typename t1, typename t2>
inline bool all_equal(const Sparse<t1> &s1, const Tensor<t2> &s2) {
  return all_equal(full(s1), s2);
}

template <typename t1, typename t2>
inline bool all_equal(const Tensor<t1> &s1, const Sparse<t2> &s2) {
  return all_equal(full(s2), s1);
}

/* Adjoint of a sparse matrix. */
RSparse adjoint(const RSparse &s);
/* Transpose of a sparse matrix. */
RSparse transpose(const RSparse &s);

/* Adjoint of a sparse matrix. */
CSparse adjoint(const CSparse &s);
/* Transpose of a sparse matrix. */
CSparse transpose(const CSparse &s);

/* Matrix multiplication between tensor and sparse matrix. */
RTensor mmult(const RTensor &m1, const RSparse &m2);
/* Matrix multiplication between tensor and sparse matrix. */
CTensor mmult(const CTensor &m1, const CSparse &m2);
/* Matrix multiplication between tensor and sparse matrix. */
RTensor mmult(const RSparse &m1, const RTensor &m2);
/* Matrix multiplication between tensor and sparse matrix. */
CTensor mmult(const CSparse &m1, const CTensor &m2);

/* Matrix multiplication between tensor and sparse matrix. */
void mmult_into(RTensor &output, const RTensor &m1, const RSparse &m2);
/* Matrix multiplication between tensor and sparse matrix. */
void mmult_into(CTensor &output, const CTensor &m1, const CSparse &m2);
/* Matrix multiplication between tensor and sparse matrix. */
void mmult_into(RTensor &output, const RSparse &m1, const RTensor &m2);
/* Matrix multiplication between tensor and sparse matrix. */
void mmult_into(CTensor &output, const CSparse &m1, const CTensor &m2);

/* Real part of a sparse matrix.*/
inline const RSparse &real(const RSparse &A) { return A; }
/* Conjugate of a sparse matrix.*/
inline const RSparse &conj(const RSparse &A) { return A; }
/* Imaginary part of a sparse matrix.*/
inline RSparse imag(const RSparse &A) { return RSparse(A.rows(), A.columns()); }

/* Real part of a sparse matrix.*/
RSparse real(const CSparse &A);
/* Conjugate of a sparse matrix.*/
CSparse conj(const CSparse &A);
/* Imaginary part of a sparse matrix.*/
RSparse imag(const CSparse &A);

RSparse operator-(const RSparse &a);
RSparse operator+(const RSparse &a, const RSparse &b);
RSparse operator-(const RSparse &a, const RSparse &b);
RSparse operator*(const RSparse &a, const RSparse &b);
RSparse operator*(const RSparse &a, double b);
RSparse operator/(const RSparse &a, double b);
RSparse operator*(double a, const RSparse &b);

CSparse operator-(const CSparse &a);
CSparse operator+(const CSparse &a, const CSparse &b);
CSparse operator-(const CSparse &a, const CSparse &b);
CSparse operator*(const CSparse &a, const CSparse &b);
CSparse operator*(const CSparse &a, cdouble b);
CSparse operator/(const CSparse &a, cdouble b);
CSparse operator*(cdouble a, const CSparse &b);

CSparse operator+(const CSparse &a, const RSparse &b);
CSparse operator-(const CSparse &a, const RSparse &b);
CSparse operator*(const CSparse &a, const RSparse &b);
CSparse operator+(const RSparse &a, const CSparse &b);
CSparse operator-(const RSparse &a, const CSparse &b);
CSparse operator*(const RSparse &a, const CSparse &b);

CSparse operator*(const CSparse &a, double b);
CSparse operator/(const CSparse &a, double b);
CSparse operator*(double a, const CSparse &b);

CSparse operator*(const RSparse &a, cdouble b);
CSparse operator/(const RSparse &a, cdouble b);
CSparse operator*(cdouble a, const RSparse &b);

/**Kronecker product between matrices, in Matlab order.*/
RSparse kron(const RSparse &a, const RSparse &b);
/**Kronecker product between matrices, opposite to Matlab order.*/
RSparse kron2(const RSparse &a, const RSparse &b);
/**Implements A+B where A and B act on different spaces of a tensor product.*/
RSparse kron2_sum(const RSparse &a, const RSparse &b);

/**Kronecker product between matrices, in Matlab order.*/
CSparse kron(const CSparse &a, const CSparse &b);
/**Kronecker product between matrices, opposite to Matlab order.*/
CSparse kron2(const CSparse &a, const CSparse &b);
/**Implements A+B where A and B act on different spaces of a tensor product.*/
CSparse kron2_sum(const CSparse &a, const CSparse &b);

}  // namespace tensor

#endif  // !TENSOR_SPARSE_H
