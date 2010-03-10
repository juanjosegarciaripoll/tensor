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
#include <tensor/clapack.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace lapack;

  /**Eigenvalue decomposition of a complex matrix.

     \ingroup Linalg
  */
  const CTensor
  eig(const CTensor &A, CTensor *R, CTensor *L)
  {
    assert(A.rows() > 0);
    assert(A.rank() == 2);
    assert(A.rows() == A.columns());

    char *jobvl, *jobvr;
    integer lda, ldvl, ldvr, lwork, info;
    cdouble *vl, *vr, *w;
    double *rwork;
    CTensor aux(A);
    cdouble *a = tensor_pointer(aux);
    integer n = A.rows();

    if ((size_t)n != A.columns()) {
      std::cerr << "Routine eig() can only compute eigenvalues of square matrices, and you\n"
                << "have passed a matrix that is " << A.rows() << " by " << A.columns();
      abort();
    }

    if (L) {
      (*L) = CTensor(n,n);
      vl = tensor_pointer(*L);
      jobvl = "V";
    } else {
      jobvl = "N";
      vl = NULL;
    }
    if (R) {
      (*R) = CTensor(n,n);
      vr = tensor_pointer(*R);
      jobvr = "V";
    } else {
      jobvr = "N";
      vr = NULL;
    }

    ldvl = ldvr = n;
    lda = n;
    lwork = -1;
    cdouble work0[1];
    w = NULL;
    rwork = new double[2*n];
    F77NAME(zgeev)(jobvl, jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work0, &lwork,
		   rwork, &info);
    lwork = lapack::real(work0[0]);

    cdouble *work = new cdouble[lwork];
    CTensor output(n);
    w = tensor_pointer(output);
    F77NAME(zgeev)(jobvl, jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork,
		   rwork, &info);
    delete[] rwork;
    delete[] work;
    return output;
  }

} // namespace linalg
