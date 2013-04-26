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

#include <cassert>
#include <mps/mpo.h>

namespace mps {

  static const RTensor safe_real(const CTensor &R)
  {
    if (!all_equal(imag(R), 0.0)) {
      std::cerr << "In RMPO, tried to initialize the tensor from a complex Hamiltonian.\n";
      abort();
    }
    return real(R);
  }

  RMPO::RMPO(const Hamiltonian &H, double t) :
    parent(H.size())
  {
    for (index i = 0; i < size(); i++) {
      at(i) = safe_real(H.local_term(i, t));
    }
    for (index i = 0; i < size(); i++) {
      for (index j = 0; j < H.interaction_depth(i, t); j++) {
        add_interaction(*this, safe_real(H.interaction_left(i, t)),
                        safe_real(H.interaction_right(i, t)));
      }
    }
  }

} // namespace mps
