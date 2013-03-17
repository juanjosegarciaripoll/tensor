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
  void test_norm_basic()
  {
    typename MPS::elt_t e0 = RTensor(igen << 2, rgen << 1.0 << 0.0);
    typename MPS::elt_t e1 = RTensor(igen << 2, rgen << 0.0 << 1.0);
    {
      // A product state with two vectors
      MPS psi = product_state(2, e0);
      psi.at(1) = reshape(e1, 1,2,1);

      // The state is normalized
      EXPECT_CEQ(norm2(psi), 1.0);
      EXPECT_CEQ(scprod(psi, psi), 1.0);

      // A change of sign does not affect the norm but it does
      // change the scalar product
      MPS psi2 = psi;
      psi2.at(1) = -psi[1];
      EXPECT_CEQ(norm2(psi2), 1.0);
      EXPECT_CEQ(scprod(psi2, psi2), 1.0);
      EXPECT_CEQ(scprod(psi, psi2), -1.0);

      // Changing one state makes the states orthogonal
      psi2.at(0) = reshape(e1, 1,2,1);
      EXPECT_CEQ(scprod(psi, psi2), 0.0);
    }
  }

  template<class MPS>
  void test_norm_order(int size)
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

    EXPECT_CEQ(norm2(psi), 1.0);
    EXPECT_CEQ(scprod(psi, psi), 1.0);
  }

  ////////////////////////////////////////////////////////////
  // EXPECTATION VALUES OVER RMPS
  //

  TEST(MPSNorm, RMPSBasic) {
    test_norm_basic<RMPS>();
  }

  TEST(MPSNorm, RMPSOrder) {
    test_over_integers(1, 10, test_norm_order<RMPS>);
  }

  TEST(MPSNorm, GHZ) {
    for (index i = 1; i < 4; i++) {
      RMPS ghz = ghz_state(i);
      EXPECT_CEQ(norm2(ghz), 1.0);
      EXPECT_CEQ(scprod(ghz, ghz), 1.0);
    }
  }

  TEST(MPSNorm, Cluster) {
    for (index i = 1; i < 10; i++) {
      RMPS cluster = cluster_state(i);
      EXPECT_CEQ(norm2(cluster), 1.0);
      EXPECT_CEQ(scprod(cluster, cluster), 1.0);
    }
  }

  ////////////////////////////////////////////////////////////
  // EXPECTATION VALUES OVER CMPS
  //

  TEST(MPSNorm, CMPSBasic) {
    test_norm_basic<CMPS>();
  }

  TEST(MPSNorm, CMPSOrder) {
    test_over_integers(1, 10, test_norm_order<CMPS>);
  }

  TEST(MPSNorm, GlobalPhases) {
    // scprod() had a problem when computing the expectation value
    // over states that were affected by a global phase because it
    // took the real part or made conj() where it should not.
    CMPS psi = cluster_state(2);
    CMPS psa = psi;
    psi.at(0) = psi[0] * exp(to_complex(0,0.03));
    psi.at(1) = psi[1] * exp(to_complex(0,0.15));
    EXPECT_CEQ(norm2(psi), 1.0);
    cdouble s = scprod(psi, psa);
    EXPECT_CEQ(abs(s), 1.0);
    EXPECT_CEQ(s, exp(to_complex(0,-0.18)));
  }

} // namespace tensor_test
