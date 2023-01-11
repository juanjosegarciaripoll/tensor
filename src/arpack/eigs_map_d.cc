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

#include <tensor/arpack.h>
#include <tensor/linalg.h>

namespace linalg {

namespace arpack {

using namespace tensor;

CTensor eigs_gen_small(const RTensor &A, EigType eig_type, size_t neig,
                       CTensor *eigenvectors, bool *converged) {
  CTensor vectors;
  CTensor values = eig(A, nullptr, eigenvectors ? &vectors : nullptr);
  Indices ndx = RArpack::sort_values(values, eig_type);
  Indices ndx_out(static_cast<index_t>(neig));
  std::copy(ndx.begin(), ndx.begin() + neig, ndx_out.begin());
  if (eigenvectors) {
    *eigenvectors = tensor::real(vectors(_, range(ndx_out)));
  }
  if (converged) {
    *converged = true;
  }
  return tensor::real(values(range(ndx_out)));
}

CTensor eigs_gen(const InPlaceLinearMap<RTensor> &A, size_t n, EigType eig_type,
                 size_t neig, CTensor *eigenvectors, bool *converged) {
  if (neig > n || neig == 0) {
    std::cerr << "In eigs_gen(): Can only compute up to " << n
              << " eigenvalues\n"
              << "in a matrix that has " << n << " times " << n << " elements.";
    abort();
  }

  if (n <= 4) {
    /* For small sizes, the ARPACK solver produces wrong results!
       * In any case, for these sizes it is more efficient to do the solving
       * using the full routine.
       */
    return eigs_small(linalg::arpack::make_matrix(A, n), eig_type, neig,
                      eigenvectors, converged);
  }

  Arpack<double, false> data(n, eig_type, neig);

  if (eigenvectors && eigenvectors->size() >= n) {
    RTensor aux = real(*eigenvectors);
    data.set_start_vector(aux.cbegin());
  } else {
    data.set_random_start_vector();
  }

  while (data.update() < data.Finished) {
    A(data.get_x(), data.get_y());
  }

  if (data.get_status() == data.Finished) {
    if (converged) *converged = true;
    return data.get_data(eigenvectors);
  } else {
    std::cerr << "eigs: " << data.error_message() << '\n';
    if (converged) {
      *converged = false;
      return RTensor::zeros(static_cast<tensor::index>(neig));
    } else {
      abort();
    }
  }
}

}  // namespace arpack

}  // namespace linalg
