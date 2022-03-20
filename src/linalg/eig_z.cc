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
#include <array>
#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include <tensor/linalg.h>

namespace linalg {

using tensor::CTensor;
using tensor::RTensor;
using tensor::SimpleVector;
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
CTensor eig(const CTensor &A, CTensor *R, CTensor *L) {
  tensor_assert(A.rows() > 0);
  tensor_assert(A.rank() == 2);
  tensor_assert(A.rows() == A.columns());

  CTensor aux(A);
  cdouble *a = tensor_pointer(aux);
  blas::integer n = blas::tensor_rows(A);

  if (n != A.columns()) {
    std::cerr << "Routine eig() can only compute eigenvalues of square "
                 "matrices, and you\n"
              << "have passed a matrix that is " << A.rows() << " by "
              << A.columns();
    abort();
  }

  char jobvl{'N'};
  cdouble *vl = nullptr;
  if (L) {
    (*L) = CTensor::empty(n, n);
    vl = tensor_pointer(*L);
    jobvl = 'V';
  }

  char jobvr{'N'};
  cdouble *vr = nullptr;
  if (R) {
    (*R) = CTensor::empty(n, n);
    vr = tensor_pointer(*R);
    jobvr = 'V';
  }

  blas::integer ldvl = n, ldvr = n;
  blas::integer lda = n;
  auto output = CTensor::empty(n);
  cdouble *w = tensor_pointer(output);
  blas::integer info{};
#ifdef TENSOR_USE_ACML
  zgeev(*jobvl, *jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, &info);
#else
  SimpleVector<double> rwork(2 * static_cast<size_t>(n));
  blas::integer lwork = -1;
  {
    std::array<cdouble, 1> work0{};
    F77NAME(zgeev)
    (&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, &work0[0], &lwork,
     rwork.begin(), &info);
    // On exit, work0 contains the optimal amount of work to be done
    lwork = static_cast<blas::integer>(lapack::real(work0[0]));
  }
  {
    SimpleVector<cdouble> work(static_cast<size_t>(lwork));
    F77NAME(zgeev)
    (&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work.begin(), &lwork,
     rwork.begin(), &info);
  }
#endif
  return output;
}

}  // namespace linalg
