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
#include <tensor/arpack.h>

RTensor eigs_sym(const RSparse &A, RArpack::EigType t, size_t neig,
                 RTensor *eigenvectors, const RTensor::elt_t *initial_guess) {
  size_t n = A.columns();
#ifdef COMPLEX
  size_t ndouble = 2 * n;
#else
  size_t ndouble = n;
#endif

  if (n <= 10) {
    return eigs_sym(full(A), t, neig, eigenvectors, initial_guess);
  }

  if (A.rows() != n) {
    std::cerr << "In eigs(): Can only compute eigenvalues of square matrices.";
    myabort();
  }

  RArpack data(ndouble, t, neig);

  if (initial_guess) data.set_start_vector((const double *)initial_guess);

  size_t bytes = ndouble * sizeof(double);
  while (data.update() < RArpack::Finished) {
    RTensor x(n, (RTensor::elt_t *)data.get_x_vector());
    RTensor y = mmult(A, x);
    memcpy(data.get_y_vector(), (const double *)x.pointer(), bytes);
  }
  if (data.get_status() == RArpack::Finished) {
    if (eigenvectors) {
      *eigenvectors = RTensor(n, neig);
    }
    return data.get_data((double *)eigenvectors->pointer());
  } else {
    std::cerr << data.error_message() << '\n';
    myabort();
  }
  return RTensor();
}
