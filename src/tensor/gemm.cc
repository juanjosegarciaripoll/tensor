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

#ifndef TENSOR_GEMM_CC
#define TENSOR_GEMM_CC

#include <limits>
#ifdef TENSOR_USE_ESSL
#include <essl.h>
#endif
#include <tensor/tensor_blas.h>

namespace blas {

using tensor::index;

inline void gemm(char op1, char op2, index m, index n, index k, double alpha,
                 const double *A, index lda, const double *B, index ldb,
                 double beta, double *C, index ldc) {
#ifdef _MSC_VER
#pragma warning(disable : 4127)
#endif
#ifdef TENSOR_DEBUG
  if (sizeof(blas::integer) < sizeof(tensor::index)) {
    constexpr auto limit = std::numeric_limits<blas::integer>::max();
    tensor_assert2(m <= limit && n <= limit && lda <= limit && ldb <= limit &&
                       ldc <= limit,
                   blas_integer_overflow());
  }
#endif
#ifdef TENSOR_USE_ESSL
  dgemm(&op1, &op2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef TENSOR_USE_ACML
  dgemm(op1, op2, m, n, k, alpha, const_cast<double *>(A), lda,
        const_cast<double *>(B), ldb, beta, C, ldc);
#endif
#if !defined(TENSOR_USE_ESSL) && !defined(TENSOR_USE_ACML)
  cblas_dgemm(CblasColMajor, char_to_op(op1), char_to_op(op2),
              static_cast<blas::integer>(m), static_cast<blas::integer>(n),
              static_cast<blas::integer>(k), alpha, A,
              static_cast<blas::integer>(lda), B,
              static_cast<blas::integer>(ldb), beta, C,
              static_cast<blas::integer>(ldc));
#endif
}

inline void gemm(char op1, char op2, index m, index n, index k,
                 const tensor::cdouble &alpha, const tensor::cdouble *A,
                 index lda, const tensor::cdouble *B, index ldb,
                 const tensor::cdouble &beta, tensor::cdouble *C, index ldc) {
#ifdef _MSC_VER
#pragma warning(disable : 4127)
#endif
#ifdef TENSOR_DEBUG
  if (sizeof(blas::integer) < sizeof(tensor::index)) {
    constexpr auto limit = std::numeric_limits<blas::integer>::max();
    tensor_assert2(m <= limit && n <= limit && lda <= limit && ldb <= limit &&
                       ldc <= limit,
                   blas_integer_overflow());
  }
#endif
#ifdef TENSOR_USE_ESSL
  zgemm(&op1, &op2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef TENSOR_USE_ACML
  zgemm(
      op1, op2, m, n, k,
      reinterpret_cast<doublecomplex *>(const_cast<tensor::cdouble *>(&alpha)),
      reinterpret_cast<doublecomplex *>(const_cast<tensor::cdouble *>(A)), lda,
      reinterpret_cast<doublecomplex *>(const_cast<tensor::cdouble *>(B)), ldb,
      reinterpret_cast<doublecomplex *>(const_cast<tensor::cdouble *>(&beta)),
      reinterpret_cast<doublecomplex *>(C), ldc);
#endif
#ifdef TENSOR_USE_OPENBLAS
  cblas_zgemm(CblasColMajor, char_to_op(op1), char_to_op(op2),
              static_cast<blas::integer>(m), static_cast<blas::integer>(n),
              static_cast<blas::integer>(k),
              reinterpret_cast<double *>(const_cast<tensor::cdouble *>(&alpha)),
              reinterpret_cast<double *>(const_cast<tensor::cdouble *>(A)),
              static_cast<blas::integer>(lda),
              reinterpret_cast<double *>(const_cast<tensor::cdouble *>(B)),
              static_cast<blas::integer>(ldb),
              reinterpret_cast<double *>(const_cast<tensor::cdouble *>(&beta)),
              reinterpret_cast<double *>(C), static_cast<blas::integer>(ldc));
#endif
#if !defined(TENSOR_USE_ESSL) && !defined(TENSOR_USE_ACML) && \
    !defined(TENSOR_USE_OPENBLAS)
  cblas_zgemm(CblasColMajor, char_to_op(op1), char_to_op(op2),
              static_cast<blas::integer>(m), static_cast<blas::integer>(n),
              static_cast<blas::integer>(k), &alpha, A,
              static_cast<blas::integer>(lda), B,
              static_cast<blas::integer>(ldb), &beta, C,
              static_cast<blas::integer>(ldc));
#endif
}

}  // namespace blas

#endif
