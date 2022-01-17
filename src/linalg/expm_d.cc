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

#include <algorithm>
#include <tensor/tensor.h>
#include <tensor/linalg.h>
#include <tensor/io.h>

namespace linalg {

#if defined(_MSC_VER) && (_MSC_VER < 1800)
static double log2(double n) { return log(n) / log((double)2.0); }
static double exp2(double n) { return exp(log((double)2.0) * n); }
#endif

/**Compute the exponential of a real matrix.
   This function computes the exponential of a matrix using a Pade
   approximation of any given order.

   This is potentially more accurate than using the eigenvalues of the matrix
   or a Taylor expansion of the exponential, and much faster, since it only
   involves products of matrices and solving one system of equations.

   The current algorithm has been adapted from Scientific Python, the version
   written by Travis Oliphant (2002). It is slightly more accurate than the
   same Matlab version when computed with the default order of 7.

   \ingroup Linalg
*/
RTensor expm(const RTensor &Aunorm, unsigned int order) {
  assert(Aunorm.rank() == 2);
  assert(Aunorm.columns() == Aunorm.rows());

  // Scale A until the norm is < 1/2
  double val = log2(matrix_norminf(Aunorm));
  int e = (int)floor(val);
  size_t j = std::max((int)0, (int)(e + 1));

  // Pade approximation for exp(A)
  double c = 1.0 / 2;
  RTensor A = Aunorm / exp2(j);
  RTensor N = RTensor::eye(A.rows()) + c * A;
  RTensor D = RTensor::eye(A.rows()) - c * A;
  RTensor X = A;

  for (size_t k = 2; k <= order; k++) {
    c = (c * (order - k + 1)) / (k * (2 * order - k + 1));
    X = mmult(A, X);
    RTensor cX = c * X;
    N = N + cX;
    if ((k & 1) == 0) {
      D = D + cX;
    } else {
      D = D - cX;
    }
  }
  X = solve(D, N);
  for (size_t k = 1; k <= j; k++) {
    X = mmult(X, X);
  }
  return X;
}

}  // namespace linalg
