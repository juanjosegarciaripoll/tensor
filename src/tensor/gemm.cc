// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_GEMM_CC
#define TENSOR_GEMM_CC

#include <tensor/cblas.h>

namespace blas {

  inline void gemm(char op1, char op2, integer m, integer n, integer k,
                   double alpha, const double *A, integer lda, const double *B,
                   integer ldb, double beta, double *C, integer ldc)
  {
    cblas_dgemm(CblasRowMajor,
                op1=='T'? CblasTrans : CblasNoTrans,
                op2=='T'? CblasTrans : CblasNoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void gemm(char op1, char op2, integer m, integer n, integer k,
                   const tensor::cdouble &alpha, const tensor::cdouble *A, integer lda,
                   const tensor::cdouble *B, integer ldb, const tensor::cdouble &beta,
                   tensor::cdouble *C, integer ldc)
  {
    cblas_zgemm(CblasRowMajor,
                op1=='T'? CblasTrans : CblasNoTrans,
                op2=='T'? CblasTrans : CblasNoTrans,
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

}

#endif
