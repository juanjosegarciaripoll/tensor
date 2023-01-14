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

CTensor make_matrix(const InPlaceLinearMap<CTensor> &A, size_t n) {
  auto M = CTensor::empty(static_cast<index_t>(n), static_cast<index_t>(n));
  for (tensor::index i = 0, l = static_cast<tensor::index>(n); i < l; i++) {
    CTensor v = CTensor::zeros(l);
    CTensor Av = CTensor::empty(l);
    v.at(i) = 1.0;
    A(v, Av);
    M.at(_, range(i)) = Av;
  }
  return M;
}

// TODO: make 'neig' int in all eigs routines

RTensor eigs_small(const CTensor &A, EigType eig_type, size_t neig,
                   CTensor *eigenvectors, bool *converged) {
  CTensor vectors;
  auto values = eig_sym(A, eigenvectors ? &vectors : nullptr);
  Indices ndx = RArpack::sort_values(values, eig_type);
  Indices ndx_out(static_cast<index_t>(neig));
  std::copy(ndx.begin(), ndx.begin() + neig, ndx_out.begin());
  if (eigenvectors) {
    *eigenvectors = tensor::real(vectors(_, range(ndx_out)));
  }
  if (converged) {
    *converged = true;
  }
  return values(range(ndx_out));
}

CTensor eigs_gen_small(const CTensor &A, EigType eig_type, size_t neig,
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
  return values(range(ndx_out));
}

CTensor eigs(const InPlaceLinearMap<CTensor> &A, size_t dim, EigType eig_type,
             size_t neig, CTensor *eigenvectors, bool *converged) {
  if (neig > dim || neig == 0) {
    std::cerr << "In eigs(): Can only compute up to " << dim << " eigenvalues\n"
              << "in a matrix that has " << dim << " times " << dim
              << " elements.";
    abort();
  }

  if (dim <= linalg::arpack::min_arpack_size) {
    /* For small sizes, the ARPACK solver produces wrong results!
       * In any case, for these sizes it is more efficient to do the solving
       * using the full routine.
       */
    return eigs_small(make_matrix(A, dim), eig_type, neig, eigenvectors,
                      converged);
  }

  CArpack data(dim, eig_type, neig);

  if (eigenvectors && eigenvectors->size() >= dim)
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

CTensor eigs_gen(const InPlaceLinearMap<CTensor> &A, size_t dim,
                 EigType eig_type, size_t neig, CTensor *eigenvectors,
                 bool *converged) {
  return linalg::arpack::eigs(A, dim, eig_type, neig, eigenvectors, converged);
}

}  // namespace arpack

}  // namespace linalg
