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

#ifndef MPS_QUANTUM_H
#define MPS_QUANTUM_H

#include <tensor/tensor.h>
#include <tensor/sparse.h>

namespace mps {

  using namespace tensor;

  /*
   * Angular momentum operators.
   */

  void spin_operators(double s, CTensor *sx, CTensor *sy, CTensor *sz);

  double entropy(const RTensor &lambdas);

  extern const RTensor Pauli_id;
  extern const RTensor Pauli_x;
  extern const RTensor Pauli_z;
  extern const CTensor Pauli_y;

  /*
   * Fock space operators.
   */

  RSparse number_operator(int nmax);
  RSparse destruction_operator(int nmax);
  RSparse creation_operator(int nmax);
  RTensor coherent_state(double alpha, int nmax);
  CTensor coherent_state(cdouble alpha, int nmax);
}

#endif // MPS_QUANTUM_H
