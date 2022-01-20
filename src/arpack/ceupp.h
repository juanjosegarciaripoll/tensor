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

   MODULE ceupp.h.
   Interface to ARPACK subroutines zneupd and cneupd.

   ARPACK Authors
      Richard Lehoucq
      Danny Sorensen
      Chao Yang
      Dept. of Computational & Applied Mathematics
      Rice University
      Houston, Texas
*/

#ifndef CEUPP_H
#define CEUPP_H

#include <memory>
#include "arpackf.h"

inline void gen_eupp(bool rvec, char HowMny, tensor::cdouble d[],
                     tensor::cdouble Z[], blas::integer ldz,
                     tensor::cdouble sigma, tensor::cdouble workev[], char bmat,
                     blas::integer n, const char* which, blas::integer nev,
                     double tol, tensor::cdouble resid[], blas::integer ncv,
                     tensor::cdouble V[], blas::integer ldv,
                     blas::integer iparam[], blas::integer ipntr[],
                     tensor::cdouble workd[], tensor::cdouble workl[],
                     blas::integer lworkl, double rwork[], blas::integer& info)

/*
  c++ version of ARPACK routine zneupd.
  This subroutine returns the converged approximations to eigenvalues
  of A*z = lambda*B*z and (optionally):

  (1) the corresponding approximate eigenvectors,
  (2) an orthonormal basis for the associated approximate
      invariant subspace,

  There is negligible additional cost to obtain eigenvectors. An
  orthonormal basis is always computed.  There is an additional storage cost
  of n*nev if both are requested (in this case a separate array Z must be
  supplied).
  The approximate eigenvalues and eigenvectors of  A*z = lambda*B*z
  are derived from approximate eigenvalues and eigenvectors of
  of the linear operator OP prescribed by the MODE selection in the
  call to caupp. caupp must be called before this routine is called.
  These approximate eigenvalues and vectors are commonly called Ritz
  values and Ritz vectors respectively.  They are referred to as such
  in the comments that follow.  The computed orthonormal basis for the
  invariant subspace corresponding to these Ritz values is referred to
  as a Schur basis.
  See documentation in the header of the subroutine caupp for
  definition of OP as well as other terms and the relation of computed
  Ritz values and Ritz vectors of OP with respect to the given problem
  A*z = lambda*B*z.  For a brief description, see definitions of
  iparam[7], MODE and which in the documentation of caupp.

  Parameters:

    rvec    (Input) Specifies whether a basis for the invariant subspace
            corresponding to the converged Ritz value approximations for
            the eigenproblem A*z = lambda*B*z is computed.
            rvec = false: Compute Ritz values only.
            rvec = true : Compute the Ritz vectors or Schur vectors.
                          See Remarks below.
    HowMny  (Input) Specifies the form of the basis for the invariant
            subspace corresponding to the converged Ritz values that
            is to be computed.
            = 'A': Compute nev Ritz vectors;
            = 'P': Compute nev Schur vectors;
    d       (Output) Array of dimension nev+1. D contains the  Ritz
            approximations to the eigenvalues lambda for A*z = lambda*B*z.
    Z       (Output) Array of dimension nev*n. If rvec = TRUE. and
            HowMny = 'A', then Z contains approximate eigenvectors (Ritz
            vectors) corresponding to the NCONV=iparam[5] Ritz values for
            eigensystem A*z = lambda*B*z.
            If rvec = .FALSE. or HowMny = 'P', then Z is not referenced.
            NOTE: If if rvec = .TRUE. and a Schur basis is not required,
                  the array Z may be set equal to first nev+1 columns of
                  the Arnoldi basis array V computed by caupp.  In this
                  case the Arnoldi basis will be destroyed and overwritten
                  with the eigenvector basis.
    ldz     (Input) Dimension of the vectors contained in Z. This
            parameter MUST be set to n.
    sigma   (Input) If iparam[7] = 3, sigma represents the shift. Not
            referenced if iparam[7] = 1 or 2.
    workv   (Workspace) Array of dimension 2*ncv.
    V       (Input/Output) Array of dimension n*ncv+1.
            Upon Input: V contains the ncv vectors of the Arnoldi basis
                        for OP as constructed by caupp.
            Upon Output: If rvec = TRUE the first NCONV=iparam[5] columns
                        contain approximate Schur vectors that span the
                        desired invariant subspace.
            NOTE: If the array Z has been set equal to first nev+1 columns
                  of the array V and rvec = TRUE. and HowMny = 'A', then
                  the Arnoldi basis held by V has been overwritten by the
                  desired Ritz vectors.  If a separate array Z has been
                  passed then the first NCONV=iparam[5] columns of V will
                  contain approximate Schur vectors that span the desired
                  invariant subspace.
    workl   (Input / Output) Array of length lworkl+1.
            workl[1:ncv*ncv+3*ncv] contains information obtained in
            caupp. They are not changed by ceupp.
            workl[ncv*ncv+3*ncv+1:3*ncv*ncv+4*ncv] holds the untransformed
            Ritz values, the untransformed error estimates of the Ritz
            values, the upper triangular matrix for H, and the associated
            matrix representation of the invariant subspace for H.
    ipntr   (Input / Output) Array of length 14. Pointer to mark the
            starting locations in the workl array for matrices/vectors
            used by caupp and ceupp.
            ipntr[9]:  pointer to the ncv RITZ values of the original
                       system.
            ipntr[11]: pointer to the ncv corresponding error estimates.
            ipntr[12]: pointer to the ncv by ncv upper triangular
                       Schur matrix for H.
            ipntr[13]: pointer to the ncv by ncv matrix of eigenvectors
                       of the upper Hessenberg matrix H. Only referenced
                       by ceupp if rvec = TRUE. See Remark 2 below.
    info    (Output) Error flag.
            =  0 : Normal exit.
            =  1 : The Schur form computed by LAPACK routine csheqr
                   could not be reordered by LAPACK routine ztrsen.
                   Re-enter subroutine ceupp with iparam[5] = ncv and
                   increase the size of the array D to have
                   dimension at least dimension ncv and allocate at least
                   ncv columns for Z. NOTE: Not necessary if Z and V share
                   the same space. Please notify the authors if this error
                   occurs.
            = -1 : n must be positive.
            = -2 : nev must be positive.
            = -3 : ncv must satisfy nev+1 <= ncv <= n.
            = -5 : which must be one of 'LM','SM','LR','SR','LI','SI'.
            = -6 : bmat must be one of 'I' or 'G'.
            = -7 : Length of private work workl array is not sufficient.
            = -8 : Error return from LAPACK eigenvalue calculation.
                   This should never happened.
            = -9 : Error return from calculation of eigenvectors.
                   Informational error from LAPACK routine ztrevc.
            = -10: iparam[7] must be 1, 2 or 3.
            = -11: iparam[7] = 1 and bmat = 'G' are incompatible.
            = -12: HowMny = 'S' not yet implemented.
            = -13: HowMny must be one of 'A' or 'P' if rvec = TRUE.
            = -14: caupp did not find any eigenvalues to sufficient
                   accuracy.

  NOTE:     The following arguments

            bmat, n, which, nev, tol, resid, ncv, V, ldv, iparam,
            ipntr, workd, workl, lworkl, rwork, info

            must be passed directly to ceupp following the last call
            to caupp.  These arguments MUST NOT BE MODIFIED between
            the the last call to caupp and the call to ceupp.

  Remarks
    1. Currently only HowMny = 'A' and 'P' are implemented.
    2. Schur vectors are an orthogonal representation for the basis of
       Ritz vectors. Thus, their numerical properties are often superior.
       Let X' denote the transpose of X. If rvec = .TRUE. then the
       relationship A * V[:,1:iparam[5]] = V[:,1:iparam[5]] * T, and
       V[:,1:iparam[5]]' * V[:,1:iparam[5]] = I are approximately satisfied.
       Here T is the leading submatrix of order iparam[5] of the real
       upper quasi-triangular matrix stored workl[ipntr[12]].
*/

{
  auto iselect = std::make_unique<logical[]>(ncv);
  blas::cdouble* iZ = reinterpret_cast<blas::cdouble*>((Z == NULL) ? V : Z);
  blas::integer thervec = rvec;

  F77NAME(zneupd)
  (&thervec, &HowMny, iselect.get(), reinterpret_cast<blas::cdouble*>(d), iZ,
   &ldz, reinterpret_cast<blas::cdouble*>(&sigma),
   reinterpret_cast<blas::cdouble*>(workev), &bmat, &n, which, &nev, &tol,
   reinterpret_cast<blas::cdouble*>(resid), &ncv,
   reinterpret_cast<blas::cdouble*>(V), &ldv, &iparam[0], &ipntr[0],
   reinterpret_cast<blas::cdouble*>(workd),
   reinterpret_cast<blas::cdouble*>(workl), &lworkl, &rwork[0], &info);
}  // ceupp (cdouble).

#endif  // CEUPP_H
// Local variables:
// mode: c++
// fill-column: 80
// c-basic-offset: 4
