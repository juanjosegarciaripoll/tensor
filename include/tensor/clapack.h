// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_CLAPACK_H
#define TENSOR_CLAPACK_H

namespace lapack {

#ifdef __APPLE__
#include <vecLib/cblas.h>
#include <vecLib/clapack.h>
#define F77NAME(x) x##_
  typedef __CLPK_integer integer;
  typedef __CLPK_doublecomplex double_complex;
#endif

  double *tensor_pointer(tensor::RTensor &A) {
    return static_cast<double*>(A.begin());
  }

  double_complex *tensor_pointer(tensor::CTensor &A) {
    return static_cast<double_complex *>((void*)A.begin());
  }

}

#endif // TENSOR_CLAPACK_H
