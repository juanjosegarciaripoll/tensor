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

#ifndef TENSOR_TENSOR_LAPACK_H
#define TENSOR_TENSOR_LAPACK_H

#include <tensor/tensor_blas.h>

namespace lapack {

using namespace blas;

#ifdef TENSOR_USE_MKL
#include <mkl_lapack.h>
#endif
#ifdef TENSOR_USE_ESSL
#undef dgesv
#undef zgesv
#undef dgeev
#undef zgeev
#undef zgesvd
#undef dgesvd
#undef dsyev
#undef zheev
#endif
#if defined(TENSOR_USE_ATLAS) || defined(TENSOR_USE_ESSL) || \
    defined(TENSOR_USE_OPENBLAS)
extern "C" {
int F77NAME(dgesv)(__CLPK_integer *n, __CLPK_integer *nrhs,
                   __CLPK_doublereal *a, __CLPK_integer *lda,
                   __CLPK_integer *ipiv, __CLPK_doublereal *b,
                   __CLPK_integer *ldb, __CLPK_integer *info);
int F77NAME(zgesv)(__CLPK_integer *n, __CLPK_integer *nrhs,
                   __CLPK_doublecomplex *a, __CLPK_integer *lda,
                   __CLPK_integer *ipiv, __CLPK_doublecomplex *b,
                   __CLPK_integer *ldb, __CLPK_integer *info);
int F77NAME(dgeev)(char *jobvl, char *jobvr, __CLPK_integer *n,
                   __CLPK_doublereal *a, __CLPK_integer *lda,
                   __CLPK_doublereal *wr, __CLPK_doublereal *wi,
                   __CLPK_doublereal *vl, __CLPK_integer *ldvl,
                   __CLPK_doublereal *vr, __CLPK_integer *ldvr,
                   __CLPK_doublereal *work, __CLPK_integer *lwork,
                   __CLPK_integer *info);
void F77NAME(dsyev)(char *jobz, char *uplo, __CLPK_integer *n,
                    __CLPK_doublereal *a, __CLPK_integer *lda,
                    __CLPK_doublereal *w, __CLPK_doublereal *work,
                    __CLPK_integer *lwork, __CLPK_integer *info);
int F77NAME(zgeev)(char *jobvl, char *jobvr, __CLPK_integer *n,
                   __CLPK_doublecomplex *a, __CLPK_integer *lda,
                   __CLPK_doublecomplex *w, __CLPK_doublecomplex *vl,
                   __CLPK_integer *ldvl, __CLPK_doublecomplex *vr,
                   __CLPK_integer *ldvr, __CLPK_doublecomplex *work,
                   __CLPK_integer *lwork, __CLPK_doublereal *rwork,
                   __CLPK_integer *info);
int F77NAME(dgesvd)(char *jobu, char *jobvt, __CLPK_integer *m,
                    __CLPK_integer *n, __CLPK_doublereal *a,
                    __CLPK_integer *lda, __CLPK_doublereal *s,
                    __CLPK_doublereal *u, __CLPK_integer *ldu,
                    __CLPK_doublereal *vt, __CLPK_integer *ldvt,
                    __CLPK_doublereal *work, __CLPK_integer *lwork,
                    __CLPK_integer *info);
int F77NAME(zgesvd)(char *jobu, char *jobvt, __CLPK_integer *m,
                    __CLPK_integer *n, __CLPK_doublecomplex *a,
                    __CLPK_integer *lda, __CLPK_doublereal *s,
                    __CLPK_doublecomplex *u, __CLPK_integer *ldu,
                    __CLPK_doublecomplex *vt, __CLPK_integer *ldvt,
                    __CLPK_doublecomplex *work, __CLPK_integer *lwork,
                    __CLPK_doublereal *rwork, __CLPK_integer *info);
void F77NAME(zheev)(char *jobz, char *uplo, __CLPK_integer *n,
                    __CLPK_doublecomplex *a, __CLPK_integer *lda,
                    __CLPK_doublereal *w, __CLPK_doublecomplex *work,
                    __CLPK_integer *lwork, __CLPK_doublereal *rwork,
                    __CLPK_integer *info);
}
#endif

}  // namespace lapack

#endif  // TENSOR_TENSOR_LAPACK_H
