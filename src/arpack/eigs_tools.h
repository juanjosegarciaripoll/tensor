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

RTensor eigs_small(const RTensor &A, EigType eig_type, size_t neig,
                   RTensor *eigenvectors, bool *converged);

CTensor eigs_gen_small(const RTensor &A, EigType eig_type, size_t neig,
                       CTensor *eigenvectors, bool *converged);

CTensor eigs_small(const CTensor &A, EigType eig_type, size_t neig,
                   CTensor *eigenvectors, bool *converged);

RTensor make_matrix(const InPlaceLinearMap<RTensor> &A, size_t n);

CTensor make_matrix(const InPlaceLinearMap<CTensor> &A, size_t n);

}  // namespace linalg
