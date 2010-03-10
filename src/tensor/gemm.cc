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

#ifndef TENSOR_GEMM_CC
#define TENSOR_GEMM_CC

#include <tensor/cblas.h>

namespace blas {

  inline void gemm(char op1, char op2, integer m, integer n, integer k,
                   double alpha, const double *A, integer lda, const double *B,
                   integer ldb, double beta, double *C, integer ldc)
  {
    cblas_dgemm(CblasColMajor, char_to_op(op1), char_to_op(op2),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void gemm(char op1, char op2, integer m, integer n, integer k,
                   const tensor::cdouble &alpha, const tensor::cdouble *A, integer lda,
                   const tensor::cdouble *B, integer ldb, const tensor::cdouble &beta,
                   tensor::cdouble *C, integer ldc)
  {
    cblas_zgemm(CblasColMajor, char_to_op(op1), char_to_op(op2),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

}

#endif
