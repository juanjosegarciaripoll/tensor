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

   MODULE seupp.h.
   Interface to ARPACK subroutines dseupd and sseupd.

   ARPACK Authors
      Richard Lehoucq
      Danny Sorensen
      Chao Yang
      Dept. of Computational & Applied Mathematics
      Rice University
      Houston, Texas
*/

#ifndef SEUPP_H
#define SEUPP_H

#include <memory>
#include "arpackf.h"

inline void gen_eupp(bool rvec, char HowMny, double d[], double Z[],
                     blas::integer ldz, double sigma, double[], char bmat,
                     blas::integer n, const char* which, blas::integer nev,
                     double tol, double resid[], blas::integer ncv, double V[],
                     blas::integer ldv, blas::integer iparam[],
                     blas::integer ipntr[], double workd[], double workl[],
                     blas::integer lworkl, double[], blas::integer& info)

/*
  c++ version of ARPACK routine dseupd.
  This subroutine returns the converged approximations to eigenvalues
  of A*z = lambda*B*z and (optionally):

  (1) the corresponding approximate eigenvectors,
  (2) an orthonormal (Lanczos) basis for the associated approximate
      invariant subspace,

  There is negligible additional cost to obtain eigenvectors. An orthonormal
  (Lanczos) basis is always computed.  There is an additional storage cost
  of n*nev if both are requested (in this case a separate array Z must be
  supplied).
  These quantities are obtained from the Lanczos factorization computed
  by saupp for the linear operator OP prescribed by the MODE selection
  (see IPARAM[7] in saupp documentation). saupp must be called before
  this routine is called. These approximate eigenvalues and vectors are
  commonly called Ritz values and Ritz vectors respectively.  They are
  referred to as such in the comments that follow. The computed orthonormal
  basis for the invariant subspace corresponding to these Ritz values is
  referred to as a Lanczos basis.
  See documentation in the header of the subroutine dsaupp for a definition
  of OP as well as other terms and the relation of computed Ritz values
  and vectors of OP with respect to the given problem  A*z = lambda*B*z.
  The approximate eigenvalues of the original problem are returned in
  ascending algebraic order.  The user may elect to call this routine
  once for each desired Ritz vector and store it peripherally if desired.
  There is also the option of computing a selected set of these vectors
  with a single call.

  Parameters:

    rvec    (Input) Specifies whether Ritz vectors corresponding to the
            Ritz value approximations to the eigenproblem A*z = lambda*B*z
            are computed.
            rvec = false: Compute Ritz values only.
            rvec = true : Compute Ritz vectors.
    HowMny  (Input) Specifies how many Ritz vectors are wanted and the
            form of Z, the matrix of Ritz vectors. See remark 1 below.
            The only option already implemented is HowMny = 'A'.
    d       (Output) Array of dimension nev. On exit, d contains the Ritz
            value approximations to the eigenvalues of A*z = lambda*B*z.
            The values are returned in ascending order. If iparam[7] =
            3, 4, 5 then d represents the Ritz values of OP computed by
            dsaupp transformed to those of the original eigensystem A*z =
            lambda*B*z. If iparam[7] = 1,2 then the Ritz values of OP are
            the same as the those of A*z = lambda*B*z.
    Z       (Output) Array of dimension nev*n if HowMny = 'A'. On
            exit, Z contains the B-orthonormal Ritz vectors of the
            eigensystem A*z = lambda*B*z corresponding to the Ritz value
            approximations. If  rvec = false then Z is not referenced.
            NOTE: The array Z may be set equal to first nev columns of
            the Arnoldi/Lanczos basis array V computed by dsaupp.
    ldz     (Input) Dimension of the vectors contained in Z. This
            parameter MUST be set to n.
    sigma   (Input) If iparam[7] = 3,4,5 represents the shift. Not
            referenced if iparam[7] = 1 or 2.
    workl   (Input / Output) Array of length lworkl+1.
            workl[1:4*ncv] contains information obtained in saupp.
            They are not changed by seupp. workl[4*ncv+1:ncv*(ncv+8)]
            holds the untransformed Ritz values, the computed error
            estimates, and the associated eigenvector matrix of H.
            Note: ipntr[8:10] contains the pointer into workl for
            addresses of the above information computed by seupp.
    ipntr   (Input / Output) Array of length 12. Pointer to mark the
            starting locations in the workl array for matrices/vectors
            used by dsaupp and seupp.
            ipntr[8] : pointer to the RITZ values of the original system.
            ipntr[9] : pointer to the ncv corresponding error bounds.
            ipntr[10]: pointer to the ncv by ncv matrix of eigenvectors
                       of the tridiagonal matrix T. Only referenced by
                       seupp if rvec = true. See Remarks.
    info    (Output) Error flag.
            =  0 : Normal exit.
            = -1 : n must be positive.
            = -2 : nev must be positive.
            = -3 : ncv must satisfy nev < ncv <= n.
            = -5 : which must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.
            = -6 : bmat must be one of 'I' or 'G'.
            = -7 : Length of private work workl array is not sufficient.
            = -8 : Error return from trid. eigenvalue calculation;
                   Information error from LAPACK routine dsteqr.
            = -9 : Starting vector is zero.
            = -10: iparam[7] must be 1,2,3,4,5.
            = -11: iparam[7] = 1 and bmat = 'G' are incompatible.
            = -12: nev and which = 'BE' are incompatible.
            = -14: dsaupp did not find any eigenvalues to sufficient
                   accuracy.
            = -15: HowMny must be one of 'A' or 'S' if rvec = true.
            = -16: HowMny = 'S' not yet implemented.

  NOTE:     The following arguments

            bmat, n, which, nev, tol, resid, ncv, V, ldv, iparam,
            ipntr, workd, workl, lworkl, info

            must be passed directly to seupp following the last call
            to saupp.  These arguments MUST NOT BE MODIFIED between
            the the last call to saupp and the call to seupp.

  Remarks
    1. The converged Ritz values are always returned in increasing
       (algebraic) order.
    2. Currently only HowMny = 'A' is implemented. It is included at
       this stage for the user who wants to incorporate it.
*/

{
  blas::integer irvec = (int)rvec;
  auto iselect = std::make_unique<logical[]>(ncv);
  double* iZ = (Z == nullptr) ? &V[0] : Z;

  F77NAME(dseupd)
  (&irvec, &HowMny, iselect.get(), d, iZ, &ldz, &sigma, &bmat, &n, which, &nev,
   &tol, resid, &ncv, &V[0], &ldv, &iparam[0], &ipntr[0], &workd[0], &workl[0],
   &lworkl, &info);
}  // seupp (double).

#endif  // SEUPP_H
