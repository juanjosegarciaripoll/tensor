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

using namespace tensor;

#if defined(TENSOR_USE_ARPACK) && defined(TENSOR_USE_PRIMME)
static EigsDriver default_eigs_driver = ArpackDriver;

EigsDriver get_default_eigs_driver() { return default_eigs_driver; }

void set_default_eigs_driver(EigsDriver driver) {
  tensor_assert((driver == ArpackDriver) || (driver == PrimmeDriver));
  default_eigs_driver = ArpackDriver;
}
#endif

CTensor eigs_gen(const LinearMap<RTensor> &A, size_t n, EigType eig_type,
                 size_t neig, CTensor *eigenvectors, bool *converged) {
  const InPlaceLinearMap<RTensor> map = [&](const RTensor &input,
                                            RTensor &output) {
    RTensor aux = A(input);
    tensor_assert(aux.dimensions() == output.dimensions());
    std::copy(aux.begin(), aux.end(), output.begin());
  };
  return eigs_gen(map, n, eig_type, neig, eigenvectors, converged);
}

CTensor eigs_gen(const InPlaceLinearMap<RTensor> &A, size_t n, EigType eig_type,
                 size_t neig, CTensor *eigenvectors, bool *converged) {
  EigsDriver driver = get_default_eigs_driver();
#ifdef TENSOR_USE_PRIMME
  if (driver == PrimmeDriver) {
    std::cerr << "Primme does not support non-symmetric real matrices.\n";
#ifdef TENSOR_USE_ARPACK
    std::cerr << "Using Arpack instead.\n";
    driver = ArpackDriver;
#else
    std::abort();
#endif
  }
#endif
#ifdef TENSOR_USE_ARPACK
  return linalg::arpack::eigs_gen(A, n, eig_type, neig, eigenvectors,
								  converged);
#endif
}

}  // namespace linalg
