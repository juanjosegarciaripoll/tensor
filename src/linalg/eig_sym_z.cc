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

#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include <tensor/linalg.h>

namespace linalg {

using namespace lapack;

/**Eigenvalue decomposition of a complex matrix.
     Given a square matrix A, we find a diagonal matrix D and a set of vectors R
     or L such that
     A V = V D
     and
     adjoint(V) A = D adjoint(V)

     The matrix A must be Hermitian (adjoint(A)==A).

     By default, only the diagonal elements of D are computed. However, also the
     matrix V can be computed if a pointer to the associated variable is
     supplied.

     \ingroup Linalg
  */
RTensor eig_sym(const CTensor &A, CTensor *V) {
  assert(A.rows() > 0);
  assert(A.rank() == 2);
  assert(A.rows() == A.columns());

  //if (accurate_svd)
  //  return block_eig_sym(A, V);
  blas::integer n = A.rows();
  if (n != A.columns()) {
    std::cerr << "Routine eig() can only compute eigenvalues of square "
                 "matrices, and you\n"
              << "have passed a matrix that is " << A.rows() << " by "
              << A.columns();
    abort();
  }

  CTensor aux(A);
  cdouble *a = tensor_pointer(aux);
  blas::integer lda = n, info[1];
  char jobz[2] = {(V == 0) ? 'N' : 'V', 0};
  char uplo[2] = {'U', 0};
  RTensor output = RTensor::empty(n);
  double *w = tensor_pointer(output);
  RTensor rwork = RTensor::empty(3 * n);

#ifdef TENSOR_USE_ACML
  zheev(*jobz, *uplo, n, a, lda, w, info);
#else
  blas::integer lwork = -1;
  CTensor work = CTensor::empty(1);
  F77NAME(zheev)
  (jobz, uplo, &n, a, &lda, w, tensor_pointer(work), &lwork,
   tensor_pointer(rwork), info);
  lwork = (int)tensor::real(work[0]);

  work = CTensor::empty(lwork);
  F77NAME(zheev)
  (jobz, uplo, &n, a, &lda, w, tensor_pointer(work), &lwork,
   tensor_pointer(rwork), info);
#endif

  if (V) *V = aux;
  return output;
}

}  // namespace linalg
