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
#include <tensor/arpack_d.h>
#include "gemv.cc"

namespace linalg {

  using namespace tensor;

  RTensor
  do_eigs(const Map<RTensor>  *A, size_t n, int eig_type, size_t neig,
          RTensor *eigenvectors, const double *initial_guess)
  {
    EigType t = (EigType)eig_type;

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
      CTensor M(n,n);
      for (int i = 0; i < n; i++) {
        RTensor v = RTensor::zeros(n);
        v.at(i) = 1.0;
        M.at(range(),range(i)) = (*A)(v);
      }
      CTensor vectors;
      CTensor values = eig(M, NULL, eigenvectors? &vectors : 0);
      Indices ndx = RArpack::sort_values(values, t);
      Indices ndx_out(neig);
      std::copy(ndx.begin(), ndx.begin() + neig, ndx_out.begin());
      if (eigenvectors) {
        *eigenvectors = tensor::real(vectors(range(), range(ndx_out)));
      }
      return tensor::real(values(range(ndx_out)));
    }

    RTensor output;
    {
      RArpack data(n, t, neig);

      if (initial_guess)
        data.set_start_vector(initial_guess);

      while (data.update() < RArpack::Finished) {
        data.set_y((*A)(data.get_x()));
      }
      if (data.get_status() != RArpack::Finished) {
        std::cerr << data.error_message() << '\n';
        abort();
      }
      output = data.get_data(eigenvectors);
    }
    return output;
  }

} // namespace linalg
