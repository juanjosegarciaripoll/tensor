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

#include "eigs_tools.h"

namespace linalg {

using namespace tensor;

CTensor eigs(const LinearMap<CTensor> &A, size_t n, EigType eig_type,
             size_t neig, CTensor *eigenvectors, bool *converged) {
  return eigs(
      [&](const CTensor &input, CTensor &output) -> void {
        CTensor aux = A(input);
        tensor_assert(aux.dimensions() == output.dimensions());
        std::copy(aux.begin(), aux.end(), output.begin());
      },
      n, eig_type, neig, eigenvectors, converged);
}

CTensor make_matrix(const InPlaceLinearMap<CTensor> &A, size_t n) {
  auto M = CTensor::empty(n, n);
  for (tensor::index i = 0, l = static_cast<tensor::index>(n); i < l; i++) {
    CTensor v = CTensor::zeros(l);
    CTensor Av = CTensor::empty(l);
    v.at(i) = 1.0;
    A(v, Av);
    M.at(_, range(i)) = Av;
  }
  return M;
}

CTensor eigs_small(const CTensor &A, EigType eig_type, size_t neig,
                   CTensor *eigenvectors, bool *converged) {
  CTensor vectors;
  CTensor values = eig(A, nullptr, eigenvectors ? &vectors : 0);
  Indices ndx = RArpack::sort_values(values, eig_type);
  Indices ndx_out(neig);
  std::copy(ndx.begin(), ndx.begin() + neig, ndx_out.begin());
  if (eigenvectors) {
    *eigenvectors = tensor::real(vectors(_, range(ndx_out)));
  }
  if (converged) {
    *converged = true;
  }
  return tensor::real(values(range(ndx_out)));
}

CTensor eigs(const InPlaceLinearMap<CTensor> &A, size_t n, EigType eig_type,
             size_t neig, CTensor *eigenvectors, bool *converged) {
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

  CArpack data(n, eig_type, neig);

  if (eigenvectors && eigenvectors->size() >= n)
    data.set_start_vector(eigenvectors->cbegin());

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
      return CTensor::zeros(static_cast<tensor::index>(neig));
    } else {
      abort();
    }
  }
}

}  // namespace linalg
