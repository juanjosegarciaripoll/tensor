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
using tensor::SimpleVector;
using namespace lapack;
using tensor::Dimensions;

/**Eigenvalue decomposition of a real matrix.
     Given a square matrix A, we find a diagonal matrix D and a set of vectors R
     or L such that
     A R = R D
     or
     L A = D L

     The eigenvalue decomposition is computed using the DGEEV routine
     from the LAPACK library. By default, only the diagonal elements of S are
     computed. However, also the U and V matrices can be computed if pointers to
     the associated variables are supplied.

     \ingroup Linalg
  */
CTensor eig(const RTensor &A, CTensor *R, CTensor *L) {
  tensor_assert(A.rows() > 0);
  tensor_assert(A.rank() == 2);
  tensor_assert(A.rows() == A.columns());

  RTensor aux(A);
  double *a = tensor_pointer(aux);
  blas::integer n = blas::tensor_rows(A);

  if (n != A.columns()) {
    std::cerr << "Routine eig() can only compute eigenvalues of square "
                 "matrices, and you\n"
              << "have passed a matrix that is " << A.rows() << " by "
              << A.columns();
    abort();
  }

  char jobvl{'N'};
  std::unique_ptr<RTensor> realL;
  double *vl = nullptr;
  if (L) {
    realL = std::make_unique<RTensor>(Dimensions{n, n});
    vl = tensor_pointer(*realL);
    jobvl = 'V';
  }

  char jobvr{'N'};
  std::unique_ptr<RTensor> realR;
  double *vr = nullptr;
  if (R) {
    realR = std::make_unique<RTensor>(Dimensions{n, n});
    vr = tensor_pointer(*realR);
    jobvr = 'V';
  }

  blas::integer ldvl = n, ldvr = n, lda = n;
  auto real = RTensor::empty(n);
  auto imag = RTensor::empty(n);
  double *wr = tensor_pointer(real);
  double *wi = tensor_pointer(imag);

  blas::integer info{};
#ifdef TENSOR_USE_ACML
  dgeev(*jobvl, *jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &info);
#else
  blas::integer lwork = -1;
  {
    std::array<double, 1> work0{};
    F77NAME(dgeev)
    (&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &work0[0],
     &lwork, &info);
    lwork = static_cast<blas::integer>(work0[0]);
  }
  {
    SimpleVector<double> work(tensor::safe_size_t(lwork));
    F77NAME(dgeev)
    (&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work.begin(),
     &lwork, &info);
  }
#endif

  CTensor output(to_complex(real));
  if (L) *L = to_complex(*realL);
  if (R) *R = to_complex(*realR);
  for (blas::integer i = 0; i < n; i++) {
    if (imag[i] != 0) {
      // Complex eigenvalues and eigenvectors. The i-th elements have
      // the real part, the i+1-th, the imaginary.
      output.at(i) = tensor::to_complex(real[i], imag[i]);
      output.at(i + 1) = tensor::to_complex(real[i], -imag[i]);
      if (realL) {
        for (blas::integer j = 0; j < n; j++) {
          double re = (*realL)(j, i);
          double im = (*realL)(j, i + 1);
          (*L).at(j, i) = tensor::to_complex(re, im);
          (*L).at(j, i + 1) = tensor::to_complex(re, -im);
        }
      }
      if (realR) {
        for (blas::integer j = 0; j < n; j++) {
          double re = (*realR)(j, i);
          double im = (*realR)(j, i + 1);
          (*R).at(j, i) = tensor::to_complex(re, im);
          (*R).at(j, i + 1) = tensor::to_complex(re, -im);
        }
      }
      i++;
    }
  }
  return output;
}

}  // namespace linalg
