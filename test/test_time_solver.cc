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

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <mps/mps.h>
#include <mps/time_evolve.h>
#include <mps/hamiltonian.h>
#include <mps/quantum.h>
#include <tensor/linalg.h>

namespace tensor_test {

  using namespace mps;
  using namespace linalg;
  using tensor::index;

  //////////////////////////////////////////////////////////////////////
  // EXACT SOLVERS
  //

  void
  split_Hamiltonian(Hamiltonian **ppHeven, Hamiltonian **ppHodd,
                    const Hamiltonian &H)
  {
    ConstantHamiltonian *pHeven = new ConstantHamiltonian(H.size());
    ConstantHamiltonian *pHodd = new ConstantHamiltonian(H.size());
    for (int i = 0; i < H.size(); i++) {
      ConstantHamiltonian &pHok = (i & 1)? (*pHodd) : (*pHeven);
      ConstantHamiltonian &pHno = (i & 1)? (*pHeven) : (*pHodd);
      if (i+1 < H.size()) {
        for (int j = 0; j < H.interaction_depth(i); j++) {
          pHok.add_interaction(i, H.interaction_left(i, j, 0.0),
                               H.interaction_right(i, j, 0.0));
          pHno.add_interaction(i, H.interaction_left(i, j, 0.0) * 0.0,
                               H.interaction_right(i, j, 0.0) * 0.0);
        }
      }
      pHok.set_local_term(i, H.local_term(i, 0.0));
      pHno.set_local_term(i, H.local_term(i, 0.0) * 0.0);
    }
    *ppHeven = pHeven;
    *ppHodd = pHodd;
  }

  CTensor
  apply_trotter2(const Hamiltonian &H, cdouble idt, const CTensor &psi)
  {
    Hamiltonian *pHeven, *pHodd;
    split_Hamiltonian(&pHeven, &pHodd, H);

    CTensor U1 = expm(full(sparse_hamiltonian(*pHeven)) * idt);
    CTensor U2 = expm(full(sparse_hamiltonian(*pHodd)) * idt);
    CTensor new_psi = mmult(U1, mmult(U2, psi));

    delete pHeven;
    delete pHodd;
    return new_psi;
  }

  CTensor
  apply_trotter3(const Hamiltonian &H, cdouble idt, const CTensor &psi)
  {
    Hamiltonian *pHeven, *pHodd;
    split_Hamiltonian(&pHeven, &pHodd, H);

    CTensor U1 = expm(full(sparse_hamiltonian(*pHeven)) * (idt/2.0));
    CTensor U2 = expm(full(sparse_hamiltonian(*pHodd)) * idt);
    CTensor new_psi = mmult(U1, mmult(U2, mmult(U1, psi)));

    delete pHeven;
    delete pHodd;
    return new_psi;
  }

  //////////////////////////////////////////////////////////////////////

  void evolve_identity(int size)
  {
    CMPS psi = ghz_state(size);
    // Id is a zero operator that causes the evolution operator to
    // be the identity
    TIHamiltonian H(size, RTensor::zeros(4,4), RTensor::zeros(2,2));
    {
      Trotter2Solver solver(H, 0.1);
      CMPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
    {
      Trotter3Solver solver(H, 0.1);
      CMPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
    {
      ForestRuthSolver solver(H, 0.1);
      CMPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
  }

  void evolve_global_phase(int size)
  {
    CMPS psi = ghz_state(size);
    // H is a multiple of the identity, causing the evolution
    // operator to be just a global phase
    TIHamiltonian H(size, RTensor::zeros(4,4), RTensor::eye(2,2));
    {
      Trotter2Solver solver(H, 0.1);
      CMPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    }
    {
      Trotter3Solver solver(H, 0.1);
      CMPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    }
    {
      ForestRuthSolver solver(H, 0.1);
      CMPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    }
  }

  void test_Hamiltonian_no_truncation(const Hamiltonian &H, double dt, const CMPS &psi, index Dmax = 0)
  {
    {
      CMPS aux = psi;
      Trotter2Solver solver(H, dt, false);
      double err = solver.one_step(&aux, Dmax);
      EXPECT_CEQ(err, 0.0);
      EXPECT_CEQ(norm2(aux), 1.0);
      CTensor aux2 = apply_trotter2(H, to_complex(0.0,-dt), mps_to_vector(psi));
      EXPECT_CEQ(mps_to_vector(aux), aux2);
    }
    {
      CMPS aux = psi;
      Trotter3Solver solver(H, dt, false);
      double err = solver.one_step(&aux, Dmax);
      EXPECT_CEQ(err, 0.0);
      EXPECT_CEQ(norm2(aux), 1.0);
      CTensor aux2 = apply_trotter3(H, to_complex(0.0,-dt), mps_to_vector(psi));
      EXPECT_CEQ(mps_to_vector(aux), aux2);
    }
  }

  void test_Hamiltonian_truncated(const Hamiltonian &H, double dt, const CMPS &psi, index Dmax = 0)
  {
    {
      CMPS truncated_psi_t = psi;
      Trotter2Solver solver(H, dt, true);
      double err = solver.one_step(&truncated_psi_t, Dmax);
      EXPECT_CEQ(norm2(truncated_psi_t), 1.0);
      CTensor psi_t = apply_trotter2(H, to_complex(0.0,-dt), mps_to_vector(psi));
      EXPECT_CEQ(mps_to_vector(truncated_psi_t), psi_t);
    }
    {
      CMPS truncated_psi_t = psi;
      Trotter3Solver solver(H, dt, true);
      double err = solver.one_step(&truncated_psi_t, Dmax);
      EXPECT_CEQ(norm2(truncated_psi_t), 1.0);
      CTensor psi_t = apply_trotter3(H, to_complex(0.0,-dt), mps_to_vector(psi));
      EXPECT_CEQ(mps_to_vector(truncated_psi_t), psi_t);
    }
  }

  void evolve_local_operator_sz(int size)
  {
    double dphi = 1.3 / size;
    ConstantHamiltonian H(size);
    for (int i = 0; i < size; i++) {
      H.set_local_term(i, mps::Pauli_z * (dphi * i));
      if (i > 0) H.add_interaction(i-1, CTensor::zeros(2,2), CTensor::zeros(2,2));
    }
    test_Hamiltonian_no_truncation(H, 0.1, ghz_state(size));
    test_Hamiltonian_no_truncation(H, 0.1, cluster_state(size));
    test_Hamiltonian_truncated(H, 0.1, ghz_state(size), 2);
    test_Hamiltonian_truncated(H, 0.1, cluster_state(size), 2);
  }

  void evolve_local_operator_sx(int size)
  {
    double dphi = 1.3 / size;
    ConstantHamiltonian H(size);
    for (int i = 0; i < size; i++) {
      H.set_local_term(i, mps::Pauli_x * (dphi * i));
      if (i > 0) H.add_interaction(i-1, CTensor::zeros(2,2), CTensor::zeros(2,2));
    }
    test_Hamiltonian_no_truncation(H, 0.1, ghz_state(size));
    test_Hamiltonian_no_truncation(H, 0.1, cluster_state(size));
    test_Hamiltonian_truncated(H, 0.1, ghz_state(size), 2);
    test_Hamiltonian_truncated(H, 0.1, cluster_state(size), 2);
  }

  void evolve_interaction_zz(int size)
  {
    double dphi = 1.3 / size;
    ConstantHamiltonian H(size);
    for (int i = 0; i < size; i++) {
      H.set_local_term(i, mps::Pauli_id * 0.0);
      if (i > 0) H.add_interaction(i-1, mps::Pauli_z, mps::Pauli_z);
    }
    test_Hamiltonian_no_truncation(H, 0.1, ghz_state(size));
    test_Hamiltonian_no_truncation(H, 0.1, cluster_state(size));
    test_Hamiltonian_truncated(H, 0.1, ghz_state(size), 2);
    test_Hamiltonian_truncated(H, 0.1, cluster_state(size), 3);
  }

  void evolve_interaction_xx(int size)
  {
    double dphi = 1.3 / size;
    ConstantHamiltonian H(size);
    for (int i = 0; i < size; i++) {
      H.set_local_term(i, mps::Pauli_id * 0.0);
      if (i > 0) H.add_interaction(i - 1, mps::Pauli_x, mps::Pauli_x);
    }
    test_Hamiltonian_no_truncation(H, 0.1, ghz_state(size));
    test_Hamiltonian_no_truncation(H, 0.1, cluster_state(size));
    test_Hamiltonian_truncated(H, 0.1, ghz_state(size), 4);
    test_Hamiltonian_truncated(H, 0.1, cluster_state(size), 4);
  }

  ////////////////////////////////////////////////////////////
  // EVOLVE WITH TROTTER METHODS
  //

  TEST(TimeSolver, Identity) {
    test_over_integers(2, 10, evolve_identity);
  }

  TEST(TimeSolver, GlobalPhase) {
    test_over_integers(2, 10, evolve_global_phase);
  }

  TEST(TimeSolver, LocalOperatorSz) {
    test_over_integers(2, 5, evolve_local_operator_sz);
  }

  TEST(TimeSolver, LocalOperatorSx) {
    test_over_integers(2, 5, evolve_local_operator_sx);
  }

  TEST(TimeSolver, NearestNeighborSzSz) {
    test_over_integers(2, 5, evolve_interaction_zz);
  }

  TEST(TimeSolver, NearestNeighborSxSx) {
    test_over_integers(2, 5, evolve_interaction_xx);
  }

} // namespace tensor_test
