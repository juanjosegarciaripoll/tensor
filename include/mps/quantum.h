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
  using tensor::index;

  /*
   * Angular momentum operators.
   */

  void spin_operators(double s, CTensor *sx, CTensor *sy, CTensor *sz);

  double entropy(const RTensor &lambdas);

  /** Two by two identity matrix. */
  extern const RTensor Pauli_id;
  /** \f$\sigma_x\f$ Pauli matrix. */
  extern const RTensor Pauli_x;
  /** \f$\sigma_z\f$ Pauli matrix. */
  extern const RTensor Pauli_z;
  /** \f$\sigma_y\f$ Pauli matrix. */
  extern const CTensor Pauli_y;

  /*
   * Fock space operators.
   */

  /** Fock number operator truncated to a maximum of 'nmax' bosons. */
  RSparse number_operator(int nmax);
  /** Fock destruction operator truncated to a maximum of 'nmax' bosons. */
  RSparse destruction_operator(int nmax);
  /** Fock creation operator truncated to a maximum of 'nmax' bosons. */
  RSparse creation_operator(int nmax);
  /** Real coherent state truncated to a maximum of 'nmax' bosons. */
  RTensor coherent_state(double alpha, int nmax);
  /** Complex coherent state truncated to a maximum of 'nmax' bosons. */
  CTensor coherent_state(cdouble alpha, int nmax);

  /* Create a sparse Hamiltonian with given local interaction and nearest
   * neighbor interaction. */
  const CSparse sparse_1d_hamiltonian(const CSparse &H12, const CSparse &Hlocal,
                                      index size, bool periodic = false);

  /* Create a sparse Hamiltonian with given local interaction and nearest
   * neighbor interaction. */
  const RSparse sparse_1d_hamiltonian(const RSparse &H12, const RSparse &Hlocal,
                                      index size, bool periodic = false);

  /* Create a sparse Hamiltonian with given local interaction and nearest
   * neighbor interaction. */
  const CSparse sparse_1d_hamiltonian(const std::vector<CSparse> &H12,
                                      const std::vector<CSparse> &Hlocal,
                                      bool periodic = false);

  /* Create a sparse Hamiltonian with given local interaction and nearest
   * neighbor interaction. */
  const RSparse sparse_1d_hamiltonian(const std::vector<RSparse> &H12,
                                      const std::vector<RSparse> &Hlocal,
                                      bool periodic = false);

  void decompose_operator(const RTensor &H12, RTensor *H1, RTensor *H2);
  void decompose_operator(const CTensor &H12, CTensor *H1, CTensor *H2);

}

#endif // MPS_QUANTUM_H
