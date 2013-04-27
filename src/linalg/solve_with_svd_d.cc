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

#include <cfloat>
#include <tensor/tensor.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace tensor;

  /**Solution of a linear system of equations using Penrose's pseudoinvese.

     This function solves the system of equations A * X = B using the SVD
     of the matrix A = U * S * VT, through the formula X = V * (S^-1) * UT * B.
     When computing (S^-1), singular values below the tolerance are discarded.
     
     \ingroup Linalg
  */
  const RTensor
  solve_with_svd(const RTensor &A, const RTensor &B, double tol)
  {
    RTensor U, VT;
    RTensor s = svd(A, &U, &VT, SVD_ECONOMIC);
    if (tol <= 0) {
      tol = DBL_EPSILON;
    }
    for (tensor::index i = 0; i < s.size(); i++) {
      if (s[i] <= tol) {
	U = U(range(), range(0,i-1));
	s = s(range(0,i-1));
	VT = VT(range(0,i-1), range());
      }
    }
    RTensor X = foldc(U, 0, B, 0);
    scale_inplace(X, 0, 1.0/s);
    return foldc(VT, 0, X, 0);
  }

} // namespace linalg
