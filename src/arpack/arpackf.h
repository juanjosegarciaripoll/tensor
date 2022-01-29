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

#include <memory>
#include <tensor/config.h>
#include <tensor/tensor_blas.h>

using namespace blas;

typedef blas::integer logical;

#ifdef __cplusplus
extern "C" {
#endif

// debug "common" statement.

extern struct {
  blas::integer logfil, ndigit, mgetv0;
  blas::integer msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd;
  blas::integer mnaupd, mnaup2, mnaitr, mneigt, mnapps, mngets, mneupd;
  blas::integer mcaupd, mcaup2, mcaitr, mceigt, mcapps, mcgets, mceupd;
} F77NAME(debug);

// double precision symmetric routines.

void F77NAME(dsaupd)(blas::integer *ido, const char *bmat, blas::integer *n,
                     const char *which, blas::integer *nev, double *tol,
                     double *resid, blas::integer *ncv, double *V,
                     blas::integer *ldv, blas::integer *iparam,
                     blas::integer *ipntr, double *workd, double *workl,
                     blas::integer *lworkl, blas::integer *info);

void F77NAME(dseupd)(logical *rvec, const char *HowMny, logical *select,
                     double *d, double *Z, blas::integer *ldz, double *sigma,
                     const char *bmat, blas::integer *n, const char *which,
                     blas::integer *nev, double *tol, double *resid,
                     blas::integer *ncv, double *V, blas::integer *ldv,
                     blas::integer *iparam, blas::integer *ipntr, double *workd,
                     double *workl, blas::integer *lworkl, blas::integer *info);

// double precision nonsymmetric routines.

void F77NAME(dnaupd)(blas::integer *ido, const char *bmat, blas::integer *n,
                     const char *which, blas::integer *nev, double *tol,
                     double *resid, blas::integer *ncv, double *V,
                     blas::integer *ldv, blas::integer *iparam,
                     blas::integer *ipntr, double *workd, double *workl,
                     blas::integer *lworkl, blas::integer *info);

void F77NAME(dneupd)(logical *rvec, const char *HowMny, logical *select,
                     double *dr, double *di, double *Z, blas::integer *ldz,
                     double *sigmar, double *sigmai, double *workev,
                     const char *bmat, blas::integer *n, const char *which,
                     blas::integer *nev, double *tol, double *resid,
                     blas::integer *ncv, double *V, blas::integer *ldv,
                     blas::integer *iparam, blas::integer *ipntr, double *workd,
                     double *workl, blas::integer *lworkl, blas::integer *info);

// double precision complex routines.

void F77NAME(znaupd)(blas::integer *ido, const char *bmat, blas::integer *n,
                     const char *which, blas::integer *nev, double *tol,
                     cdouble *resid, blas::integer *ncv, cdouble *V,
                     blas::integer *ldv, blas::integer *iparam,
                     blas::integer *ipntr, cdouble *workd, cdouble *workl,
                     blas::integer *lworkl, double *rwork, blas::integer *info);

void F77NAME(zneupd)(logical *rvec, const char *HowMny, logical *select,
                     cdouble *d, cdouble *Z, blas::integer *ldz, cdouble *sigma,
                     cdouble *workev, const char *bmat, blas::integer *n,
                     const char *which, blas::integer *nev, double *tol,
                     cdouble *resid, blas::integer *ncv, cdouble *V,
                     blas::integer *ldv, blas::integer *iparam,
                     blas::integer *ipntr, cdouble *workd, cdouble *workl,
                     blas::integer *lworkl, double *rwork, blas::integer *info);

#ifdef __cplusplus
}
#endif

namespace linalg {

//////////////////////////////////////////////////
// COMPLEX GENERAL PROBLEMS
//

static inline void gen_aupp(blas::integer &ido, char bmat, blas::integer n,
                            const char *which, blas::integer nev, double &tol,
                            tensor::cdouble resid[], blas::integer ncv,
                            tensor::cdouble V[], blas::integer ldv,
                            blas::integer iparam[], blas::integer ipntr[],
                            tensor::cdouble workd[], tensor::cdouble workl[],
                            blas::integer lworkl, double rwork[],
                            blas::integer &info) {
  F77NAME(znaupd)
  (&ido, &bmat, &n, which, &nev, &tol, reinterpret_cast<blas::cdouble *>(resid),
   &ncv, reinterpret_cast<blas::cdouble *>(V), &ldv, iparam, ipntr,
   reinterpret_cast<blas::cdouble *>(workd),
   reinterpret_cast<blas::cdouble *>(workl), &lworkl, rwork, &info);
}

static inline CTensor gen_eupp(
    CTensor *eigenvectors, tensor::cdouble sigma, tensor::cdouble workev[],
    char bmat, blas::integer n, const char *which, blas::integer nev,
    double tol, tensor::cdouble resid[], blas::integer ncv, tensor::cdouble V[],
    blas::integer ldv, blas::integer iparam[], blas::integer ipntr[],
    tensor::cdouble workd[], tensor::cdouble workl[], blas::integer lworkl,
    double rwork[], blas::integer &info) {
  auto iselect = std::make_unique<logical[]>(ncv);
  blas::integer rvec;
  char HowMny;
  blas::cdouble *Z;
  blas::integer ldz;
  CTensor eigenvalues = CTensor::empty(nev + 1);
  if (eigenvectors) {
    *eigenvectors = RTensor::empty(n, nev);
    Z = reinterpret_cast<blas::cdouble *>(eigenvectors->begin());
    HowMny = 'A';
    ldz = n;
    rvec = 1;
  } else {
    HowMny = 'P';
    Z = nullptr;
    ldz = 1;
    rvec = 0;
  }
  F77NAME(zneupd)
  (&rvec, &HowMny, iselect.get(),
   reinterpret_cast<blas::cdouble *>(eigenvalues.begin()), Z, &ldz,
   reinterpret_cast<blas::cdouble *>(&sigma),
   reinterpret_cast<blas::cdouble *>(workev), &bmat, &n, which, &nev, &tol,
   reinterpret_cast<blas::cdouble *>(resid), &ncv,
   reinterpret_cast<blas::cdouble *>(V), &ldv, &iparam[0], &ipntr[0],
   reinterpret_cast<blas::cdouble *>(workd),
   reinterpret_cast<blas::cdouble *>(workl), &lworkl, &rwork[0], &info);
  return eigenvalues(tensor::range(0, nev - 1));
}

//////////////////////////////////////////////////
// REAL SYMMETRIC PROBLEMS
//

static inline void gen_aupp(blas::integer &ido, char bmat, blas::integer n,
                            const char *which, blas::integer nev, double &tol,
                            double resid[], blas::integer ncv, double V[],
                            blas::integer ldv, blas::integer iparam[],
                            blas::integer ipntr[], double workd[],
                            double workl[], blas::integer lworkl, double[],
                            blas::integer &info) {
  F77NAME(dsaupd)
  (&ido, &bmat, &n, which, &nev, &tol, resid, &ncv, &V[0], &ldv, &iparam[0],
   &ipntr[0], &workd[0], &workl[0], &lworkl, &info);
}

static inline RTensor gen_eupp(RTensor *eigenvectors, double sigma, double[],
                               char bmat, blas::integer n, const char *which,
                               blas::integer nev, double tol, double resid[],
                               blas::integer ncv, double V[], blas::integer ldv,
                               blas::integer iparam[], blas::integer ipntr[],
                               double workd[], double workl[],
                               blas::integer lworkl, double[],
                               blas::integer &info) {
  auto iselect = std::make_unique<logical[]>(ncv);
  char HowMny;
  double *Z;
  blas::integer ldz;
  blas::integer rvec;
  if (eigenvectors) {
    *eigenvectors = RTensor::empty(n, nev);
    Z = eigenvectors->begin();
    HowMny = 'A';
    ldz = n;
    rvec = 1;
  } else {
    HowMny = 'P';
    Z = V;
    ldz = 1;
    rvec = 0;
  }
  RTensor eigenvalues = RTensor::empty(nev + 1);
  F77NAME(dseupd)
  (&rvec, &HowMny, iselect.get(), eigenvalues.begin(), Z, &ldz, &sigma, &bmat,
   &n, which, &nev, &tol, resid, &ncv, &V[0], &ldv, &iparam[0], &ipntr[0],
   &workd[0], &workl[0], &lworkl, &info);
  return eigenvalues(tensor::range(0, nev - 1));
}

}  // namespace linalg

#endif  // ARPACKF_H
