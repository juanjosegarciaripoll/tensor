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
#include <mps/quantum.h>

namespace tensor_test {

  using namespace tensor;
  using namespace mps;
  using tensor::index;

  template<class MPS>
  void test_correlation_basic()
  {
    typename MPS::elt_t e0 = RTensor(igen << 2, rgen << 1.0 << 0.0);
    typename MPS::elt_t e1 = RTensor(igen << 2, rgen << 0.0 << 1.0);
    {
      // A product state with two vectors
      MPS psi = product_state(3, e0);
      psi.at(1) = reshape(e1, 1,2,1);
      psi.at(2) = reshape((e1 + e0)/sqrt(2), 1,2,1);

      EXPECT_DEATH(expected(psi, mps::Pauli_id, 3, mps::Pauli_id, 0), ".*");
      EXPECT_DEATH(expected(psi, mps::Pauli_id, -4, mps::Pauli_id, 0), ".*");
      EXPECT_DEATH(expected(psi, mps::Pauli_id, 0, mps::Pauli_id, 3), ".*");
      EXPECT_DEATH(expected(psi, mps::Pauli_id, 0, mps::Pauli_id, -4), ".*");

      EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, 1), -1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, 2), 0.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_x, 2), 1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_x, 2), -1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_z, 2), 0.0);

      EXPECT_CEQ(expected(psi, mps::Pauli_z, -3, mps::Pauli_z, 1), -1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, -3, mps::Pauli_z, 2), 0.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, -3, mps::Pauli_x, 2), 1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, -2, mps::Pauli_x, 2), -1.0);

      EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, -2), -1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, -1), 0.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_x, -1), 1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_x, -1), -1.0);

      // When we repeat the site, the operators are multiplied
      EXPECT_CEQ(expected(psi, mps::Pauli_x, 0, mps::Pauli_x, 0), 1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_z, 1), 1.0);
      EXPECT_CEQ(expected(psi, mps::Pauli_z, 2, mps::Pauli_z, 2), 1.0);
    }
  }

  template<class MPS>
  void test_correlation_order(int size)
  {
    /*
     * We create a random product state and verify that the
     * expectation value over the k-th site is the same as
     * that of the single-site operator on the associated state.
     */
    typedef typename MPS::elt_t Tensor;
    Tensor *states = new Tensor[size];
    
    for (index i = 0; i < size; i++) {
      states[i] = Tensor::random(2);
      states[i] = states[i] / norm2(states[i]);
    }

    MPS psi = product_state(size, states[0]);
    for (index i = 0; i < size; i++)
      psi.at(i) = reshape(states[i], 1,2,1);

    for (index i = 0; i < size; i++)
      for (index j = 0; j < i; j++)
	EXPECT_CEQ(expected(psi, mps::Pauli_z, j, mps::Pauli_z, i),
		   scprod(states[i], mmult(mps::Pauli_z, states[i])) *
		   scprod(states[j], mmult(mps::Pauli_z, states[j])));
  }

  ////////////////////////////////////////////////////////////
  // EXPECTATION VALUES OVER RMPS
  //

  TEST(MPSCorrelation, RMPSBasic) {
    test_correlation_basic<RMPS>();
  }

  TEST(MPSCorrelation, RMPSOrder) {
    test_over_integers(1, 10, test_correlation_order<RMPS>);
  }

  TEST(MPSCorrelation, GHZ) {
    for (index i = 1; i < 4; i++) {
      RMPS ghz = ghz_state(i);
      for (index j = 0; j < i; j++) {
	for (index k = 0; k < i; k++) {
	  EXPECT_DOUBLE_EQ(expected(ghz, mps::Pauli_z, j, mps::Pauli_z, k), 1.0);
	  EXPECT_DOUBLE_EQ(expected(ghz, mps::Pauli_x, j, mps::Pauli_z, k), 0.0);
	}
      }
    }
  }

  ////////////////////////////////////////////////////////////
  // EXPECTATION VALUES OVER CMPS
  //

  TEST(MPSCorrelation, CMPSBasic) {
    test_correlation_basic<CMPS>();
  }

  TEST(MPSCorrelation, CMPSOrder) {
    test_over_integers(1, 10, test_correlation_order<CMPS>);
  }



} // namespace tensor_test
