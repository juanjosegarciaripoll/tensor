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

  Hamiltonian::~Hamiltonian()
  {
  }

  index
  Hamiltonian::dimension(index k) const
  {
    return local_term(k,0).dimension(1);
  }

  const Indices
  Hamiltonian::dimensions() const
  {
    index l = size();
    Indices output(l);
    for (index i = 0; i < l; i++) {
      output.at(i) = dimension(i);
    }
    return output;
  }

  const CTensor
  Hamiltonian::interaction_left(index k, index ndx, double t) const
  {
    CTensor O1, O2;
    decompose_operator(interaction(k, t), &O1, &O2);
    return squeeze(O1(range(), range(), range(ndx)));
  }

  const CTensor
  Hamiltonian::interaction_right(index k, index ndx, double t) const
  {
    CTensor O1, O2;
    decompose_operator(interaction(k, t), &O1, &O2);
    return squeeze(O2(range(), range(), range(ndx)));
  }

  index
  Hamiltonian::interaction_depth(index k, double t) const
  {
    CTensor O1, O2;
    decompose_operator(interaction(k, t), &O1, &O2);
    return O1.dimension(3);
  }

} // namespace mps
