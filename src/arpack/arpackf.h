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

typedef integer logical;

#ifdef __cplusplus
extern "C"
{
#endif

  // debug "common" statement.

  extern struct { 
    integer logfil, ndigit, mgetv0;
    integer msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd;
    integer mnaupd, mnaup2, mnaitr, mneigt, mnapps, mngets, mneupd;
    integer mcaupd, mcaup2, mcaitr, mceigt, mcapps, mcgets, mceupd;
  } F77_FUNC(debug,DEBUG);


  // double precision symmetric routines.

  void F77_FUNC(dsaupd,DSAUPD)
    (integer *ido, const char *bmat, integer *n, const char *which,
     integer *nev, double *tol, double *resid,
     integer *ncv, double *V, integer *ldv,
     integer *iparam, integer *ipntr, double *workd,
     double *workl, integer *lworkl, integer *info);

  void F77_FUNC(dseupd,DSEUPD)
    (logical *rvec, const char *HowMny, logical *select,
     double *d, double *Z, integer *ldz,
     double *sigma, const char *bmat, integer *n,
     const char *which, integer *nev, double *tol,
     double *resid, integer *ncv, double *V,
     integer *ldv, integer *iparam, integer *ipntr,
     double *workd, double *workl,
     integer *lworkl, integer *info);

  // double precision nonsymmetric routines.

  void F77_FUNC(dnaupd,DNAUPD)
    (integer *ido, const char *bmat, integer *n, const char *which,
     integer *nev, double *tol, double *resid,
     integer *ncv, double *V, integer *ldv,
     integer *iparam, integer *ipntr, double *workd,
     double *workl, integer *lworkl, integer *info);

  void F77_FUNC(dneupd,DNEUPD)
    (logical *rvec, const char *HowMny, logical *select,
     double *dr, double *di, double *Z,
     integer *ldz, double *sigmar,
     double *sigmai, double *workev,
     const char *bmat, integer *n, const char *which,
     integer *nev, double *tol, double *resid,
     integer *ncv, double *V, integer *ldv,
     integer *iparam, integer *ipntr,
     double *workd, double *workl,
     integer *lworkl, integer *info);

  // double precision complex routines.

  void F77_FUNC(znaupd,ZNAUPD)
    (integer *ido,
     const char *bmat, integer *n, const char *which, integer *nev,
     double *tol, cdouble *resid, integer *ncv,
     cdouble *V, integer *ldv, integer *iparam,
     integer *ipntr, cdouble *workd,
     cdouble *workl, integer *lworkl,
     double *rwork, integer *info);

  void F77_FUNC(zneupd,ZNEUPD)
    (logical *rvec, const char *HowMny, logical *select,
     cdouble *d, cdouble *Z, integer *ldz,
     cdouble *sigma, cdouble *workev,
     const char *bmat, integer *n, const char *which, integer *nev,
     double *tol, cdouble *resid, integer *ncv,
     cdouble *V, integer *ldv, integer *iparam,
     integer *ipntr, cdouble *workd,
     cdouble *workl, integer *lworkl,
     double *rwork, integer *info);

#ifdef __cplusplus
}
#endif
#endif // ARPACKF_H

