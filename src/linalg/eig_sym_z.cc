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

using tensor::SimpleVector;

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
  tensor_assert(A.rows() > 0);
  tensor_assert(A.rank() == 2);
  tensor_assert(A.rows() == A.columns());

  //if (accurate_svd)
  //  return block_eig_sym(A, V);
  blas::integer n = blas::tensor_rows(A);
  if (n != A.columns()) {
    std::cerr << "Routine eig() can only compute eigenvalues of square "
                 "matrices, and you\n"
              << "have passed a matrix that is " << A.rows() << " by "
              << A.columns();
    abort();
  }

  CTensor aux(A);
  cdouble *a = tensor_pointer(aux);
  char jobz{(V == nullptr) ? 'N' : 'V'};
  char uplo{'U'};
  RTensor output = RTensor::empty(n);
  double *w = tensor_pointer(output);

  blas::integer info{};
#ifdef TENSOR_USE_ACML
  zheev(jobz, uplo, n, a, n, w, info);
#else
  blas::integer lwork = -1;
  SimpleVector<double> rwork(static_cast<size_t>(3 * n));
  {
    cdouble work0;
    F77NAME(zheev)
    (&jobz, &uplo, &n, a, &n, w, &work0, &lwork, rwork.begin(), &info);
    lwork = static_cast<blas::integer>(lapack::real(work0));
  }
  {
    SimpleVector<cdouble> work(tensor::safe_size_t(lwork));
    F77NAME(zheev)
    (&jobz, &uplo, &n, a, &n, w, work.begin(), &lwork, rwork.begin(), &info);
  }
#endif

  if (V) *V = aux;
  return output;
}

}  // namespace linalg
