// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <mps/quantum.h>
#include "sparse_1d_hamiltonian.cc"

namespace mps {

  /**Create the Hamiltonian of a translationary invariant model. Given a nearest
     neighbor interaction \c H12, and a local term \c H1, this function
     constructs a sparse matrix representing the Hamiltonian
     \f[ H = \sum_{i=1}^{n} \left[H^{(12)}_{i,i+1} + H^{(1)}_i\right]\f]
     
     Only if the flag \c periodic is true, the term for the interaction between
     the last and first particle will be included, \f$H^{(12)}_{1N}\f$.

     If any of the matrices H12 or H1 is empty, it is not used.
     
     \ingroup QM
  */
  const RSparse sparse_1d_hamiltonian(const RSparse &H12, const RSparse &H1,
				      index N, bool periodic)
  {
    assert(N > 0);
    return do_pair_Hamiltonian<RSparse,RTensor>(H12, H1, N, periodic);
  }

} // namespace mps
