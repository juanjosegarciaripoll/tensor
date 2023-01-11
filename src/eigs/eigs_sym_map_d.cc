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

#include <tensor/linalg/eigs.h>

namespace linalg {

using namespace tensor;

RTensor eigs(const LinearMap<RTensor> &A, size_t n, EigType eig_type,
             size_t neig, RTensor *eigenvectors, bool *converged) {
  return eigs(
      [&](const RTensor &input, RTensor &output) {
        RTensor aux = A(input);
        tensor_assert(aux.dimensions() == output.dimensions());
        std::copy(aux.begin(), aux.end(), output.begin());
      },
      n, eig_type, neig, eigenvectors, converged);
}

RTensor eigs(const InPlaceLinearMap<RTensor> &A, size_t n, EigType eig_type,
             size_t neig, RTensor *eigenvectors, bool *converged) {

  const EigsDriver driver = get_default_eigs_driver();
#ifdef TENSOR_USE_ARPACK
  if (driver == ArpackDriver) {
	return linalg::arpack::eigs(A, n, eig_type, neig, eigenvectors, converged);
  }
#endif
#ifdef TENSOR_USE_PRIMME
  return linalg::primme::eigs(A, n, eig_type, neig, eigenvectors, converged);
#endif
}


} // namespace linalg
