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
#include "gemv.cc"

namespace linalg {

using namespace tensor;

RTensor eigs(const RTensor &A, int eig_type, size_t neig, RTensor *eigenvectors,
             bool *converged) {
  EigType t = (EigType)eig_type;
  size_t n = A.columns();

  if ((A.rank() != 2) || (A.rows() != n)) {
    std::cerr << "In eigs(): Can only compute eigenvalues of square matrices.";
    abort();
  }

  if (neig > n || neig == 0) {
    std::cerr << "In eigs(): Can only compute up to " << n << " eigenvalues\n"
              << "in a matrix that has " << n << " times " << n << " elements.";
    abort();
  }

  if (n <= 4) {
    /* For small sizes, the ARPACK solver produces wrong results!
       * In any case, for these sizes it is more efficient to do the solving
       * using the full routine.
       */
    CTensor vectors;
    CTensor values = eig(A, NULL, eigenvectors ? &vectors : 0);
    Indices ndx = RArpack::sort_values(values, t);
    Indices ndx_out(neig);
    std::copy(ndx.begin(), ndx.begin() + neig, ndx_out.begin());
    if (eigenvectors) {
      *eigenvectors = tensor::real(vectors(range(), range(ndx_out)));
    }
    if (converged) {
      *converged = true;
    }
    return tensor::real(values(range(ndx_out)));
  }

  RArpack data(n, t, neig);

  if (eigenvectors && eigenvectors->size() >= A.columns())
    data.set_start_vector(eigenvectors->begin_const());

  while (data.update() < RArpack::Finished) {
    blas::gemv('N', n, n, 1.0, A.begin(), n, data.get_x_vector(), 1, 0.0,
               data.get_y_vector(), 1);
  }
  if (data.get_status() == RArpack::Finished) {
    if (converged) *converged = true;
    return data.get_data(eigenvectors);
  } else {
    std::cerr << "eigs: " << data.error_message() << '\n';
    if (converged) {
      *converged = false;
      return RTensor::zeros(igen << neig);
    } else {
      abort();
    }
  }
}

}  // namespace linalg
