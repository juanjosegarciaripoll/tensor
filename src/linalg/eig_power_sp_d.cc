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
#include <tensor/sparse.h>

namespace linalg {

/**Right eigenvalue and eigenvector with the largest absolute
     value, computed using the power method. 'iter' is the maximum
     number of iterations of the algorithm. 'tol' is the maximum
     absolute error in the elements of the eigenvector.

     \ingroup Linalg
  */
double eig_power_right(const RSparse &O, RTensor *vector, size_t iter,
                       double tol) {
  tensor_assert(O.rows() == O.columns());
  return eig_power([&O](const RTensor &x) { return mmult(O, x); },
                   static_cast<size_t>(O.columns()), vector, iter, tol);
}

/**Left eigenvalue and eigenvector with the largest absolute
     value, computed using the power method. 'iter' is the maximum
     number of iterations of the algorithm. 'tol' is the maximum
     absolute error in the elements of the eigenvector.

     \ingroup Linalg
  */
double eig_power_left(const RSparse &O, RTensor *vector, size_t iter,
                      double tol) {
  tensor_assert(O.rows() == O.columns());
  auto OT = transpose(O);
  return eig_power([&OT](const RTensor &x) { return mmult(OT, x); },
                   static_cast<size_t>(O.columns()), vector, iter, tol);
}

}  // namespace linalg
