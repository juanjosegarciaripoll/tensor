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

#include <complex>
#include <memory>
#include <tensor/config.h>
#include <tensor/tensor_blas.h>
#include <tensor/io.h>
#include <tensor/vector.h>

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

using tensor::_;
using tensor::CTensor;
using tensor::range;
using tensor::RTensor;

template <bool symmetric>
class ArpackSymmetric {};

//////////////////////////////////////////////////
// COMPLEX GENERAL PROBLEMS
//

template <bool is_symmetric>
static inline void gen_aupp(blas::integer &ido, char bmat, blas::integer n,
                            const char *which, blas::integer nev, double &tol,
                            tensor::cdouble resid[], blas::integer ncv,
                            tensor::cdouble V[], blas::integer ldv,
                            blas::integer iparam[], blas::integer ipntr[],
                            tensor::cdouble workd[], tensor::cdouble workl[],
                            blas::integer lworkl, double rwork[],
                            blas::integer &info,
                            ArpackSymmetric<is_symmetric>) {
  F77NAME(znaupd)
  (&ido, &bmat, &n, which, &nev, &tol, reinterpret_cast<blas::cdouble *>(resid),
   &ncv, reinterpret_cast<blas::cdouble *>(V), &ldv, iparam, ipntr,
   reinterpret_cast<blas::cdouble *>(workd),
   reinterpret_cast<blas::cdouble *>(workl), &lworkl, rwork, &info);
}

template <bool is_symmetric>
static inline CTensor gen_eupp(
    CTensor *eigenvectors, tensor::cdouble sigma, tensor::cdouble workev[],
    char bmat, blas::integer n, const char *which, blas::integer nev,
    double tol, tensor::cdouble resid[], blas::integer ncv, tensor::cdouble V[],
    blas::integer ldv, blas::integer iparam[], blas::integer ipntr[],
    tensor::cdouble workd[], tensor::cdouble workl[], blas::integer lworkl,
    double rwork[], blas::integer &info, ArpackSymmetric<is_symmetric>) {
  auto iselect = tensor::SimpleVector<logical>::empty(static_cast<size_t>(ncv));
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
  (&rvec, &HowMny, iselect.begin(),
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
// REAL GENERAL PROBLEMS
//
static inline void gen_aupp(blas::integer &ido, char bmat, blas::integer n,
                            const char *which, blas::integer nev, double &tol,
                            double resid[], blas::integer ncv, double V[],
                            blas::integer ldv, blas::integer iparam[],
                            blas::integer ipntr[], double workd[],
                            double workl[], blas::integer lworkl,
                            double /*rwork*/[], blas::integer &info,
                            ArpackSymmetric<false>) {
  F77NAME(dnaupd)
  (&ido, &bmat, &n, which, &nev, &tol, resid, &ncv, &V[0], &ldv, &iparam[0],
   &ipntr[0], &workd[0], &workl[0], &lworkl, &info);
}

static inline CTensor gen_eupp(CTensor *eigenvectors, tensor::cdouble sigma,
                               double workv[], char bmat, blas::integer n,
                               const char *which, blas::integer nev, double tol,
                               double resid[], blas::integer ncv, double V[],
                               blas::integer ldv, blas::integer iparam[],
                               blas::integer ipntr[], double workd[],
                               double workl[], blas::integer lworkl,
                               double /*rwork*/[], blas::integer &info,
                               ArpackSymmetric<false>) {
  char HowMny;
  auto iselect = tensor::SimpleVector<logical>(static_cast<size_t>(ncv));
  RTensor Z;
  blas::integer returned = nev;
  blas::integer ldz;
  blas::integer rvec;
  double sigmar = real(sigma), sigmai = imag(sigma);
  if (eigenvectors) {
    Z = RTensor(tensor::Dimensions{n, ncv},
                tensor::Vector<double>(static_cast<size_t>(n * ncv), V));
    HowMny = 'A';
    ldz = n;
    rvec = 1;
  } else {
    HowMny = 'P';
    ldz = 1;
    rvec = 0;
  }
  auto dr = tensor::SimpleVector<double>(static_cast<size_t>(nev) + 1);
  auto di = tensor::SimpleVector<double>(static_cast<size_t>(nev) + 1);

  F77NAME(dneupd)
  (&rvec, &HowMny, iselect.begin(), dr.begin(), di.begin(), Z.begin(), &ldz,
   &sigmar, &sigmai, &workv[0], &bmat, &n, which, &returned, &tol, resid, &ncv,
   &V[0], &ldv, &iparam[0], &ipntr[0], &workd[0], &workl[0], &lworkl, &info);

  CTensor eigenvalues = CTensor::empty(returned);
  if (eigenvectors) {
    *eigenvectors = CTensor::empty(n, returned);
  }
  for (tensor::index i = 0; i < returned;) {
    if (di[i]) {
      if (i == nev - 1) {
        std::cerr << "Complex values found, exceeding number of desired "
                     "eigenvalues.\n";
        abort();
      }
      eigenvalues.at(i) = tensor::cdouble(dr[i], di[i]);
      eigenvalues.at(i + 1) = tensor::cdouble(dr[i + 1], di[i + 1]);
      if (eigenvectors) {
        eigenvectors->at(_, range(i)) =
            tensor::to_complex(Z(_, range(i)), Z(_, range(i + 1)));
        eigenvectors->at(_, range(i + 1)) =
            tensor::conj((*eigenvectors)(_, range(i)));
      }
      i += 2;
    } else {
      eigenvalues.at(i) = dr[i];
      if (eigenvectors) {
        eigenvectors->at(_, range(i)) = tensor::to_complex(Z(_, range(i)));
      }
      ++i;
    }
  }
  return eigenvalues;
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
                            blas::integer &info, ArpackSymmetric<true>) {
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
                               blas::integer &info, ArpackSymmetric<true>) {
  auto iselect = tensor::SimpleVector<logical>::empty(static_cast<size_t>(ncv));
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
  (&rvec, &HowMny, iselect.begin(), eigenvalues.begin(), Z, &ldz, &sigma, &bmat,
   &n, which, &nev, &tol, resid, &ncv, &V[0], &ldv, &iparam[0], &ipntr[0],
   &workd[0], &workl[0], &lworkl, &info);
  return eigenvalues(tensor::range(0, nev - 1));
}

}  // namespace linalg

#endif  // ARPACKF_H
