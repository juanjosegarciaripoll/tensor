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

//----------------------------------------------------------------------
// ARPACK DRIVER FOR NONSYMMETRIC SPARSE EIGENVALUE PROBLEMS
//

#include <tensor/linalg.h>
#include <tensor/arpack_z.h>

#define COMPLEX

namespace linalg {

  CTensor
  eigs(const CSparse &A, int eig_type, size_t neig, CTensor *eigenvectors,
       const CTensor::elt_t *initial_guess)
  {
    EigType t = (EigType)eig_type;
    size_t n = A.columns();

    if (n <= 10) {
      return eigs(full(A), t, neig, eigenvectors, initial_guess);
    }

    if (A.rows() != n) {
      std::cerr << "In eigs(): Can only compute eigenvalues of square matrices.";
      abort();
    }

    CTensor output;
    {
      CArpack data(n, t, neig);

      if (initial_guess)
	data.set_start_vector(initial_guess);

      while (data.update() < CArpack::Finished) {
	data.set_y(mmult(A, data.get_x()));
      }
      if (data.get_status() != CArpack::Finished) {
	std::cerr << data.error_message() << '\n';
	abort();
      }
      output = data.get_data(eigenvectors);
    }
    return output;
  }

} // namespace linalg
