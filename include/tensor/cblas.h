// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_CBLAS_H
#define TENSOR_CBLAS_H

#include <tensor/config.h>

#ifdef TENSOR_USE_VECLIB
#include <vecLib/cblas.h>
#include <vecLib/clapack.h>
#define F77NAME(x) x##_
#endif
#ifdef TENSOR_USE_ATLAS
#include <cblas.h>
#define F77NAME(x) x##_
#endif
#if !defined(TENSOR_USE_VECLIB) && !defined(TENSOR_USE_ATLAS)
# error "We need to use one of these libraries: VecLib, Atlas"
#endif

namespace blas {

#ifdef TENSOR_USE_VECLIB
  typedef __CLPK_integer integer;
  typedef __CLPK_doublecomplex cdouble;
#endif
#ifdef TENSOR_USE_ATLAS
  typedef int integer;
  typedef struct { double re, im; } cdouble;
  typedef int __CLPK_integer;
  typedef double __CLPK_doublereal;
  typedef cdouble __CLPK_doublecomplex;
#endif

#if defined(TENSOR_USE_VECLIB) || defined(TENSOR_USE_ATLAS)
  inline CBLAS_TRANSPOSE char_to_op(char op)
  {
    if (op == 'T')
      return CblasTrans;
    if (op == 'C')
      return CblasConjTrans;
    return CblasNoTrans;
  }
#endif

  inline const double *tensor_pointer(const tensor::RTensor &A) {
    return static_cast<const double*>(A.begin());
  }

  inline const cdouble *tensor_pointer(const tensor::CTensor &A) {
    return static_cast<const cdouble *>((void*)A.begin());
  }

  inline double *tensor_pointer(tensor::RTensor &A) {
    return static_cast<double*>(A.begin());
  }

  inline cdouble *tensor_pointer(tensor::CTensor &A) {
    return static_cast<cdouble *>((void*)A.begin());
  }

  inline double real(cdouble &z) {
    return tensor::real(*static_cast<tensor::cdouble *>((void*)&z));
  }

}

#endif // TENSOR_CLAPACK_H
