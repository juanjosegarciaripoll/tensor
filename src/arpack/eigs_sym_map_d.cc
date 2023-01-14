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

namespace linalg {

namespace arpack {

using namespace tensor;

RTensor make_matrix(const InPlaceLinearMap<RTensor> &A, size_t n) {
  auto l = static_cast<index_t>(n);
  auto M = RTensor::empty(l, l);
  auto v = RTensor::zeros(l);
  auto Av = RTensor::empty(l);
  for (index_t i = 0; i < l; i++) {
    v.at(i) = 1.0;
    A(v, Av);
    M.at(_, range(i)) = Av;
	v.at(i) = 0.0;
  }
  return M;
}

RTensor eigs_small(const RTensor &A, EigType eig_type, size_t neig,
                   RTensor *eigenvectors, bool *converged) {
  RTensor vectors;
  RTensor values = eig_sym(A, eigenvectors ? &vectors : nullptr);
  Indices ndx = RArpack::sort_values(values, eig_type);
  Indices ndx_out(static_cast<index_t>(neig));
  std::copy(ndx.begin(), ndx.begin() + neig, ndx_out.begin());
  if (eigenvectors) {
    *eigenvectors = vectors(_, range(ndx_out));
  }
  if (converged) {
    *converged = true;
  }
  return values(range(ndx_out));
}

RTensor eigs(const InPlaceLinearMap<RTensor> &A, size_t n, EigType eig_type,
             size_t neig, RTensor *eigenvectors, bool *converged) {
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
    return eigs_small(make_matrix(A, n), eig_type, neig, eigenvectors,
                      converged);
  }

  Arpack<double, true> data(n, eig_type, neig);

  if (eigenvectors && eigenvectors->size() >= n) {
    data.set_start_vector(eigenvectors->cbegin());
  } else {
    data.set_random_start_vector();
  }

  while (data.update() < RArpack::Finished) {
    A(data.get_x(), data.get_y());
  }

  if (data.get_status() == RArpack::Finished) {
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
