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
  const CTensor
  eig(const RTensor &A, CTensor *R, CTensor *L)
  {
    assert(A.rows() > 0);
    assert(A.rank() == 2);
    assert(A.rows() == A.columns());

    char jobvl[2] = "N";
    char jobvr[2] = "N";
    integer lda, ldvl, ldvr, lwork, info;
    double *vl, *vr, *wr, *wi;
    RTensor aux(A);
    double *a = tensor_pointer(aux);
    integer n = A.rows();
    RTensor *realL, *realR;

    if ((size_t)n != A.columns()) {
      std::cerr << "Routine eig() can only compute eigenvalues of square matrices, and you\n"
                << "have passed a matrix that is " << A.rows() << " by " << A.columns();
      abort();
    }

    if (L) {
      realL = new RTensor(n,n);
      vl = tensor_pointer(*realL);
      jobvl[0] = 'V';
    } else {
      realL = NULL;
      vl = NULL;
    }
    if (R) {
      realR = new RTensor(n, n);
      vr = tensor_pointer(*realR);
      jobvr[0] = 'V';
    } else {
      realR = NULL;
      vr = NULL;
    }
    ldvl = ldvr = n;
    lda = n;
    lwork = -1;
    double work0[1];
    wr = wi = NULL;
    F77NAME(dgeev)(jobvl, jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work0, &lwork,
		   &info);
    lwork = (int)work0[0];

    double *work = new double[lwork];
    RTensor real(n), imag(n);
    wr = tensor_pointer(real);
    wi = tensor_pointer(imag);
    F77NAME(dgeev)(jobvl, jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork,
		   &info);
    delete[] work;

    CTensor output(to_complex(real));
    if (L) *L = to_complex(*realL);
    if (R) *R = to_complex(*realR);
    for (size_t i = 0; i < (size_t)n; i++) {
      if (imag[i] != 0) {
        // Complex eigenvalues and eigenvectors. The i-th elements have
        // the real part, the i+1-th, the imaginary.
        output.at(i) = tensor::to_complex(real[i],imag[i]);
        output.at(i+1) = tensor::to_complex(real[i],-imag[i]);
        if (realL) {
          for (size_t j = 0; j < (size_t)n; j++) {
            double re = (*realL)(j,i);
            double im = (*realL)(j,i+1);
            (*L).at(j,i) = tensor::to_complex(re,im);
            (*L).at(j,i+1) = tensor::to_complex(re,-im);
          }
        }
        if (realR) {
          for (size_t j = 0; j < (size_t)n; j++) {
            double re = (*realR)(j,i);
            double im = (*realR)(j,i+1);
            (*R).at(j,i) = tensor::to_complex(re,im);
            (*R).at(j,i+1) = tensor::to_complex(re,-im);
          }
        }
        i++;
      }
    }
    delete realL;
    delete realR;
    return output;
  }

} // namespace linalg
