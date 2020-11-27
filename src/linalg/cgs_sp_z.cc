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

#include <tensor/linalg.h>

namespace linalg {

/**Solve a real linear system of equations by the conjugate gradient method.

     Given a matrix A, and a right hand matrix B, we find the matrix X that
     satisfies
     A X = B
     using the iterative conjugate gradient method.
     \ingroup Linalg
  */
const CTensor cgs(const CSparse &A, const CTensor &b, const CTensor *x_start,
                  int maxiter, double tol) {
  return do_cgs(new tensor::MatrixMap<CSparse>(A), b, x_start, maxiter, tol);
}

}  // namespace linalg
