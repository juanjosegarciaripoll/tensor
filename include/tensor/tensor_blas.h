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

#ifndef TENSOR_TENSOR_BLAS_H
#define TENSOR_TENSOR_BLAS_H

#include <tensor/config.h>
#include <tensor/tensor.h>

#ifdef TENSOR_USE_MKL
# include <mkl_cblas.h>
# include <mkl_lapack.h>
# define F77NAME(x) x
#endif

#ifdef TENSOR_USE_ACML
# include <acml.h>
#endif

#ifdef TENSOR_USE_VECLIB
# include <Accelerate/accelerate.h>
# define F77NAME(x) x##_
#endif

#ifdef TENSOR_USE_ATLAS
extern "C" {
# ifdef HAVE_ATLAS_CBLAS_H
#  include <atlas/cblas.h>
#  include <atlas/clapack.h>
# elif defined(HAVE_CBLAS_H)
#  include <cblas.h>
#  include <clapack.h>
# else
#  error "Header cblas.h not found"
# endif
}
# define F77NAME(x) x##_
#endif

#ifdef TENSOR_USE_CBLAPACK
# include <cblapack.h>
# define F77NAME(x) x##_
#endif

#ifdef TENSOR_USE_ESSL
# include <essl.h>
# define F77NAME(x) x
#endif

#if !defined(TENSOR_USE_VECLIB) && !defined(TENSOR_USE_ATLAS) && !defined(TENSOR_USE_MKL) && !defined(TENSOR_USE_ESSL) && !defined(TENSOR_USE_CBLAPACK) && !defined(TENSOR_USE_ACML)
# error "We need to use one of these libraries: VecLib, Atlas, MKL, ESSL, CBLAPACK"
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
#ifdef TENSOR_USE_MKL
  typedef MKL_INT integer;
  typedef MKL_Complex16 cdouble;
#endif
#ifdef TENSOR_USE_ACML
  typedef int integer;
  typedef doublecomplex cdouble;
  typedef int __CLPK_integer;
  typedef double __CLPK_doublereal;
  typedef cdouble __CLPK_doublecomplex;
#endif
#ifdef TENSOR_USE_ESSL
  typedef _ESVINT integer;
  typedef _ESVCOM cdouble;
  typedef _ESVINT __CLPK_integer;
  typedef double __CLPK_doublereal;
  typedef _ESVCOM __CLPK_doublecomplex;
#endif
#ifdef TENSOR_USE_CBLAPACK
  typedef ::integer integer;
  typedef doublecomplex cdouble;
  typedef integer __CLPK_integer;
  typedef double __CLPK_doublereal;
  typedef cdouble __CLPK_doublecomplex;
#endif

#if defined(TENSOR_USE_VECLIB) || defined(TENSOR_USE_ATLAS) || defined(TENSOR_USE_MKL) || defined(TENSOR_USE_CBLAPACK)
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

#endif // TENSOR_TENSOR_BLAS_H
