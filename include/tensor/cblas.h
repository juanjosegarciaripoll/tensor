// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_CBLAS_H
#define TENSOR_CBLAS_H

namespace blas {

#ifdef __APPLE__
#include <vecLib/cblas.h>
#include <vecLib/clapack.h>
#define F77NAME(x) x##_
  typedef __CLPK_integer integer;
  typedef __CLPK_doublecomplex cdouble;
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
