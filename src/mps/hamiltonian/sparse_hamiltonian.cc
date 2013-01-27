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

#include <mps/quantum.h>
#include <mps/hamiltonian.h>

namespace mps {

  const CSparse
  sparse_hamiltonian(const Hamiltonian &H, const double t)
  {
    index N = H.size();
    bool periodic = H.is_periodic();
    std::vector<CSparse> H12(N);
    std::vector<CSparse> H1(N);
    for (index k = 0; k < N; k++) {
      H1.at(k) = CSparse(H.local_term(k, t));
      if ((k < (N-1)) || periodic)
	H12.at(k) = CSparse(H.interaction(k, t));
    }
    return sparse_1d_hamiltonian(H12, H1, periodic);
  }

} // namespace mps
