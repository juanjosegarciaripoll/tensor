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

#include <memory>
#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include <tensor/linalg.h>

namespace linalg {

using tensor::CTensor;
using tensor::RTensor;
using namespace lapack;

/**
     \overload const CTensor eig(const CTensor &A, CTensor *R, CTensor *L)
     Eigenvalue decomposition of a complex matrix.
     Given a square matrix A, we find a diagonal matrix D and a set of vectors R
     or L such that
     A R = R D
     or
     L A = D L

     The eigenvalue decomposition is computed using the ZGEEV routine
     from the LAPACK library. By default, only the diagonal elements of S are
     computed. However, also the U and V matrices can be computed if pointers to
     the associated variables are supplied.

     \ingroup Linalg
  */
const CTensor eig(const CTensor &A, CTensor *R, CTensor *L) {
  assert(A.rows() > 0);
  assert(A.rank() == 2);
  assert(A.rows() == A.columns());

  char jobvl[2] = "N";
  char jobvr[2] = "N";
  blas::integer lda, ldvl, ldvr, lwork, info;
  cdouble *vl, *vr, *w;
  CTensor aux(A);
  cdouble *a = tensor_pointer(aux);
  blas::integer n = A.rows();

  if (n != A.columns()) {
    std::cerr << "Routine eig() can only compute eigenvalues of square "
                 "matrices, and you\n"
              << "have passed a matrix that is " << A.rows() << " by "
              << A.columns();
    abort();
  }

  if (L) {
    (*L) = CTensor(n, n);
    vl = tensor_pointer(*L);
    jobvl[0] = 'V';
  } else {
    vl = NULL;
  }
  if (R) {
    (*R) = CTensor(n, n);
    vr = tensor_pointer(*R);
    jobvr[0] = 'V';
  } else {
    vr = NULL;
  }

  ldvl = ldvr = n;
  lda = n;
  CTensor output(n);
  w = tensor_pointer(output);
#ifdef TENSOR_USE_ACML
  zgeev(*jobvl, *jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, &info);
#else
  cdouble work0[1];
  lwork = -1;
  auto rwork = std::make_unique<double[]>(2 * n);
  F77NAME(zgeev)
  (jobvl, jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work0, &lwork,
   rwork.get(), &info);
  // On exit, work0 contains the optimal amount of work to be done
  lwork = static_cast<blas::integer>(lapack::real(work0[0]));

  auto work = std::make_unique<cdouble[]>(lwork);
  F77NAME(zgeev)
  (jobvl, jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work.get(), &lwork,
   rwork.get(), &info);
#endif
  return output;
}

}  // namespace linalg
