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
// TRANSLATIONALLY INVARIANT HAMILTONIAN
//

#include <mps/hamiltonian.h>

namespace mps {

  /** Create the ConstantHamiltonian, reserving space for the local
      terms and interactions.*/
  ConstantHamiltonian::ConstantHamiltonian(index N, bool periodic) :
    H12_(N), H12_left_(N), H12_right_(N), H1_(N), periodic_(periodic)
  {
  }

  const Hamiltonian *
  ConstantHamiltonian::duplicate() const
  {
    return new ConstantHamiltonian(*this);
  }

  index
  ConstantHamiltonian::size() const
  {
    return H12_.size();
  }

  bool
  ConstantHamiltonian::is_constant() const
  {
    return 1;
  }

  bool
  ConstantHamiltonian::is_periodic() const
  {
    return periodic_;
  }

  const CTensor
  ConstantHamiltonian::interaction(index k, double t) const
  {
    return H12_[k];
  }

  const CTensor
  ConstantHamiltonian::interaction_left(index k, index ndx, double t) const
  {
    return H12_left_[k][ndx];
  }

  const CTensor
  ConstantHamiltonian::interaction_right(index k, index ndx, double t) const
  {
    return H12_right_[k][ndx];
  }

  index
  ConstantHamiltonian::interaction_depth(index k, double t) const
  {
    return H12_left_[k].size();
  }

  const CTensor
  ConstantHamiltonian::local_term(index k, double t) const
  {
    return H1_[k];
  }

  /** Add a local term on the k-th site.*/
  void
  ConstantHamiltonian::set_local_term(index k, const CTensor &H1)
  {
    assert((k >= 0) && (k <= H1_.size()));
    H1_.at(k) = H1;
  }

  /** Add a nearest-neighbor interaction between sites 'k' and 'k+1'.*/
  void
  ConstantHamiltonian::set_interaction(index k, const CTensor &H1, const CTensor &H2)
  {
    assert((k >= 0) && (k <= H12_.size()));
    H12_left_[k].clear();
    H12_right_[k].clear();
    add_interaction(k, H1, H2);
  }

  /** Add a nearest-neighbor interaction between sites 'k' and 'k+1'.*/
  void
  ConstantHamiltonian::add_interaction(index k, const CTensor &H1, const CTensor &H2)
  {
    assert((k >= 0) && (k+1 < H12_.size()));
    H12_left_[k].push_back(H1);
    H12_right_[k].push_back(H2);
    H12_.at(k) = compute_interaction(k);
  }

  const CTensor
  ConstantHamiltonian::compute_interaction(index k) const
  {
    assert((k >= 0) && (k+1 < H12_.size()));
    CTensor H;
    for (index i = 0; i < H12_left_[k].size(); i++) {
      CTensor op = kron2(H12_left_[k][i], H12_right_[k][i]);
      if (i == 0)
        H = op;
      else
        H = H + op;
    }
    if (H.is_empty()) {
      index d = dimension(k) * dimension(k+1);
      return CTensor::zeros(d, d);
    } else {
      return H;
    }
  }

} // namespace mps
