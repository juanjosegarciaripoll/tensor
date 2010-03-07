// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_CLAPACK_H
#define TENSOR_CLAPACK_H

#include <tensor/cblas.h>

namespace lapack {

  using namespace blas;

#ifdef TENSOR_USE_VECLIB
#include <vecLib/clapack.h>
#endif
#ifdef TENSOR_USE_ATLAS
#include <clapack.h>
extern "C" {
  int F77NAME(dgeev)
    (char *jobvl, char *jobvr, __CLPK_integer *n, __CLPK_doublereal *
     a, __CLPK_integer *lda, __CLPK_doublereal *wr, __CLPK_doublereal *wi,
     __CLPK_doublereal *vl, __CLPK_integer *ldvl, __CLPK_doublereal *vr,
     __CLPK_integer *ldvr, __CLPK_doublereal *work,
     __CLPK_integer *lwork, __CLPK_integer *info);
  int F77NAME(zgeev)
    (char *jobvl, char *jobvr, __CLPK_integer *n, 
     __CLPK_doublecomplex *a, __CLPK_integer *lda, __CLPK_doublecomplex *w,
     __CLPK_doublecomplex *vl, __CLPK_integer *ldvl, __CLPK_doublecomplex *vr,
     __CLPK_integer *ldvr, __CLPK_doublecomplex *work,
     __CLPK_integer *lwork, __CLPK_doublereal *rwork, __CLPK_integer *info);
  int F77NAME(dgesvd)
    (char *jobu, char *jobvt, __CLPK_integer *m, __CLPK_integer *n, 
     __CLPK_doublereal *a, __CLPK_integer *lda, __CLPK_doublereal *s,
     __CLPK_doublereal *u, __CLPK_integer *ldu, __CLPK_doublereal *vt,
     __CLPK_integer *ldvt, __CLPK_doublereal *work, __CLPK_integer *lwork, 
     __CLPK_integer *info);
  int F77NAME(zgesvd)
    (char *jobu, char *jobvt, __CLPK_integer *m, __CLPK_integer *n, 
     __CLPK_doublecomplex *a, __CLPK_integer *lda, __CLPK_doublereal *s,
     __CLPK_doublecomplex *u, __CLPK_integer *ldu, __CLPK_doublecomplex *vt,
     __CLPK_integer *ldvt, __CLPK_doublecomplex *work, 
     __CLPK_integer *lwork, __CLPK_doublereal *rwork, __CLPK_integer *info);
}
#endif

}

#endif // TENSOR_CLAPACK_H
