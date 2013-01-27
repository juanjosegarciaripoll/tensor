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

  void
  ConstantHamiltonian::set_local_term(index k, const CTensor &H1)
  {
    assert((k >= 0) && (k <= H1_.size()));
    H1_.at(k) = H1;
  }

  void
  ConstantHamiltonian::set_interaction(index k, const CTensor &H12)
  {
    assert((k >= 0) && (k <= H12_.size()));
    H12_.at(k) = H12;
    split_interaction(H12, &H12_left_.at(k), &H12_right_.at(k));
  }

} // namespace mps
