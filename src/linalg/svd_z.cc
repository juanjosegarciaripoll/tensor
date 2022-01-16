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

using namespace lapack;

/**Singular value decomposition of a complex matrix.

     The singular value decomposition of a matrix A, consists in finding two
     unitary matrices U and V, and diagonal one S with nonnegative elements, such
     that \f$A = U S V\f$. The svd() routine computes the diagonal elements of
     the matrix S and puts them in a 1D tensor, which is the output of the
     routine.  Optionally, the matrices U and V are also computed, and stored in
     the variables pointed to by U and VT.

     Unless otherwise specified, if the matrix A has \c MxN elements, then U is
     \c MxM, V is \c NxN and the vector S will have \c min(M,N) elements. However
     if flag \c economic is different from zero, then we get smaller matrices,
     U being \c MxR, V being \c RxN and S will have \c R=min(M,N) elements.
     
     \ingroup Linalg
  */
RTensor svd(CTensor A, CTensor *U, CTensor *VT, bool economic) {
  /*
    if (accurate_svd) {
      return block_svd(A, U, VT, economic);
    }
    */
  assert(A.rows() > 0);
  assert(A.columns() > 0);
  assert(A.rank() == 2);

  blas::integer m = A.rows();
  blas::integer n = A.columns();
  blas::integer k = std::min(m, n);
  blas::integer lwork, ldu, lda, ldv, info;
  RTensor output(k);
  cdouble *u, *v, *a = tensor_pointer(A);
  double *s = tensor_pointer(output);
  char jobv[1], jobu[1];

  if (U) {
    *U = CTensor(m, economic ? k : m);
    u = tensor_pointer(*U);
    jobu[0] = economic ? 'S' : 'A';
    ldu = m;
  } else {
    jobu[0] = 'N';
    u = NULL;
    ldu = 1;
  }
  if (VT) {
    (*VT) = CTensor(economic ? k : n, n);
    v = tensor_pointer(*VT);
    jobv[0] = economic ? 'S' : 'A';
    ldv = economic ? k : n;
  } else {
    jobv[0] = 'N';
    v = NULL;
    ldv = 1;
  }
  lda = m;
#ifdef TENSOR_USE_ACML
  zgesvd(*jobu, *jobv, m, n, a, m, s, u, ldu, v, ldv, &info);
#else
  lwork = -1;
  cdouble work0[1];
  double rwork0[1];
  F77NAME(zgesvd)
  (jobu, jobv, &m, &n, a, &m, s, u, &ldu, v, &ldv, work0, &lwork, rwork0,
   &info);
  // work[0] contains the optimal amount of space required
  lwork = static_cast<blas::integer>(lapack::real(work0[0]));
  auto work = std::make_unique<cdouble[]>(lwork);
  auto rwork = std::make_unique<double[]>(5 * k);
  F77NAME(zgesvd)
  (jobu, jobv, &m, &n, a, &m, s, u, &ldu, v, &ldv, work.get(), &lwork,
   rwork.get(), &info);
#endif
  return output;
}

}  // namespace linalg
