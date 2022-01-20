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
/*
  ARPACK++ v1.0 8/1/1997
  c++ interface to ARPACK code.

  MODULE arpackf.h
  ARPACK FORTRAN routines.

  ARPACK Authors
     Richard Lehoucq
     Danny Sorensen
     Chao Yang
     Dept. of Computational & Applied Mathematics
     Rice University
     Houston, Texas
*/

#ifndef ARPACKF_H
#define ARPACKF_H

#include <tensor/config.h>
#include <tensor/tensor_blas.h>

using namespace blas;

typedef blas::integer logical;

#ifdef __cplusplus
extern "C"
{
#endif

  // debug "common" statement.

  extern struct { 
    blas::integer logfil, ndigit, mgetv0;
    blas::integer msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd;
    blas::integer mnaupd, mnaup2, mnaitr, mneigt, mnapps, mngets, mneupd;
    blas::integer mcaupd, mcaup2, mcaitr, mceigt, mcapps, mcgets, mceupd;
  } F77NAME(debug);


  // double precision symmetric routines.

  void F77NAME(dsaupd)
    (blas::integer *ido, const char *bmat, blas::integer *n, const char *which,
     blas::integer *nev, double *tol, double *resid,
     blas::integer *ncv, double *V, blas::integer *ldv,
     blas::integer *iparam, blas::integer *ipntr, double *workd,
     double *workl, blas::integer *lworkl, blas::integer *info);

  void F77NAME(dseupd)
    (logical *rvec, const char *HowMny, logical *select,
     double *d, double *Z, blas::integer *ldz,
     double *sigma, const char *bmat, blas::integer *n,
     const char *which, blas::integer *nev, double *tol,
     double *resid, blas::integer *ncv, double *V,
     blas::integer *ldv, blas::integer *iparam, blas::integer *ipntr,
     double *workd, double *workl,
     blas::integer *lworkl, blas::integer *info);

  // double precision nonsymmetric routines.

  void F77NAME(dnaupd)
    (blas::integer *ido, const char *bmat, blas::integer *n, const char *which,
     blas::integer *nev, double *tol, double *resid,
     blas::integer *ncv, double *V, blas::integer *ldv,
     blas::integer *iparam, blas::integer *ipntr, double *workd,
     double *workl, blas::integer *lworkl, blas::integer *info);

  void F77NAME(dneupd)
    (logical *rvec, const char *HowMny, logical *select,
     double *dr, double *di, double *Z,
     blas::integer *ldz, double *sigmar,
     double *sigmai, double *workev,
     const char *bmat, blas::integer *n, const char *which,
     blas::integer *nev, double *tol, double *resid,
     blas::integer *ncv, double *V, blas::integer *ldv,
     blas::integer *iparam, blas::integer *ipntr,
     double *workd, double *workl,
     blas::integer *lworkl, blas::integer *info);

  // double precision complex routines.

  void F77NAME(znaupd)
    (blas::integer *ido,
     const char *bmat, blas::integer *n, const char *which, blas::integer *nev,
     double *tol, cdouble *resid, blas::integer *ncv,
     cdouble *V, blas::integer *ldv, blas::integer *iparam,
     blas::integer *ipntr, cdouble *workd,
     cdouble *workl, blas::integer *lworkl,
     double *rwork, blas::integer *info);

  void F77NAME(zneupd)
    (logical *rvec, const char *HowMny, logical *select,
     cdouble *d, cdouble *Z, blas::integer *ldz,
     cdouble *sigma, cdouble *workev,
     const char *bmat, blas::integer *n, const char *which, blas::integer *nev,
     double *tol, cdouble *resid, blas::integer *ncv,
     cdouble *V, blas::integer *ldv, blas::integer *iparam,
     blas::integer *ipntr, cdouble *workd,
     cdouble *workl, blas::integer *lworkl,
     double *rwork, blas::integer *info);

#ifdef __cplusplus
}
#endif
#endif // ARPACKF_H

