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
// MPS library
//
// (c) 2004 Juan Jose Garcia Ripoll
//
//----------------------------------------------------------------------
// ARPACK DRIVER FOR SYMMETRIC REAL EIGENVALUE PROBLEMS
//

#include <tensor/linalg.h>
#include <tensor/arpack.h>
#include <tensor/utils.h>
#include <tensor/sys/blas.h>

#undef COMPLEX

RTensor eigs_sym(const CTensor &A, RArpack::EigType t, size_t neig,
                 CTensor *eigenvectors, const CTensor::elt_t *initial_guess) {
  size_t n = A.columns();

  if ((A.rank() != 2) || (A.rows() != n)) {
    std::cerr << "In eigs(): Can only compute eigenvalues of square matrices.";
    myabort();
  }

  if (neig > n || neig == 0) {
    std::cerr << "In eigs(): Can only compute up to " << n << " eigenvalues\n"
              << "in a matrix that has " << n << " times " << n << " elements.";
    myabort();
  }

  if (n <= 4) {
    RTensor values = eig_sym(A, eigenvectors);
    UIVector ndx = RArpack::sort_values(values, t)(Range(0, neig - 1));
    if (eigenvectors) {
      *eigenvectors = (*eigenvectors)(Range(), Range(ndx));
    }
    return re_part(values(Range(ndx)));
  }

  RArpack data(2 * n, t, neig);

  if (initial_guess) data.set_start_vector((double *)initial_guess);

  while (data.update() < RArpack::Finished) {
    cdouble *x = (CTensor::elt_t *)data.get_x_vector();
    cdouble *y = (CTensor::elt_t *)data.get_y_vector();
    gemv('N', n, n, CTensor::elt_one(), A.pointer(), n, x, 1,
         CTensor::elt_zero(), y, 1);
  }
  if (data.get_status() == RArpack::Finished) {
    if (eigenvectors) {
      *eigenvectors = CTensor(n, neig);
    }
    return data.get_data(eigenvectors ? (double *)eigenvectors->pointer()
                                      : NULL);
  } else {
    std::cerr << data.error_message() << '\n';
    myabort();
  }
  return RTensor();
}

/// Local variables:
/// mode: c++
/// fill-column: 80
/// c-basic-offset: 4
