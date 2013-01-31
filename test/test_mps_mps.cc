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
#include <mps/quantum.h>
#include <mps/mps.h>

namespace tensor_test {

  using namespace tensor;
  using namespace mps;
  using tensor::index;

  //////////////////////////////////////////////////////////////////////
  // CONSTRUCTING SMALL MPS
  //

  template<class MPS>
  void test_mps_constructor() {
    {
      MPS psi(/* size */ 1, /* physical dimension */ 2, /* bond dimension */ 3,
	      /* periodic */ false);
      EXPECT_EQ(psi.size(), 1);
      EXPECT_EQ(psi[0].dimensions(), igen << 1 << 2 << 1);
    }
    {
      MPS psi(/* size */ 1, /* physical dimension */ 2, /* bond dimension */ 3,
	      /* periodic */ true);
      EXPECT_EQ(psi.size(), 1);
      EXPECT_EQ(psi[0].dimensions(), igen << 3 << 2 << 3);
    }
    {
      MPS psi(/* size */ 2, /* physical dimension */ 2, /* bond dimension */ 3,
	      /* periodic */ false);
      EXPECT_EQ(psi.size(), 2);
      EXPECT_EQ(psi[0].dimensions(), igen << 1 << 2 << 3);
      EXPECT_EQ(psi[1].dimensions(), igen << 3 << 2 << 1);

    }
    {
      MPS psi(/* size */ 3, /* physical dimension */ 2, /* bond dimension */ 3,
	      /* periodic */ false);
      EXPECT_EQ(psi.size(), 3);
      EXPECT_EQ(psi[0].dimensions(), igen << 1 << 2 << 3);
      EXPECT_EQ(psi[1].dimensions(), igen << 3 << 2 << 3);
      EXPECT_EQ(psi[2].dimensions(), igen << 3 << 2 << 1);
    }
    {
      MPS psi(/* size */ 3, /* physical dimension */ 2, /* bond dimension */ 3,
	      /* periodic */ true);
      EXPECT_EQ(psi.size(), 3);
      EXPECT_EQ(psi[0].dimensions(), igen << 3 << 2 << 3);
      EXPECT_EQ(psi[1].dimensions(), igen << 3 << 2 << 3);
      EXPECT_EQ(psi[2].dimensions(), igen << 3 << 2 << 3);
    }
  }

  template<class MPS>
  void test_mps_random() {
    {
      MPS psi =
        MPS::random(/* size */ 1, /* physical dimension */ 2, /* bond dimension */ 3,
                    /* periodic */ false);
      EXPECT_EQ(psi.size(), 1);
      EXPECT_EQ(psi[0].dimensions(), igen << 1 << 2 << 1);
    }
    {
      MPS psi =
        MPS::random(/* size */ 1, /* physical dimension */ 2, /* bond dimension */ 3,
                    /* periodic */ true);
      EXPECT_EQ(psi.size(), 1);
      EXPECT_EQ(psi[0].dimensions(), igen << 3 << 2 << 3);
    }
    {
      MPS psi =
        MPS::random(/* size */ 2, /* physical dimension */ 2, /* bond dimension */ 3,
                    /* periodic */ false);
      EXPECT_EQ(psi.size(), 2);
      EXPECT_EQ(psi[0].dimensions(), igen << 1 << 2 << 3);
      EXPECT_EQ(psi[1].dimensions(), igen << 3 << 2 << 1);

    }
    {
      MPS psi =
        MPS::random(/* size */ 3, /* physical dimension */ 2, /* bond dimension */ 3,
                    /* periodic */ false);
      EXPECT_EQ(psi.size(), 3);
      EXPECT_EQ(psi[0].dimensions(), igen << 1 << 2 << 3);
      EXPECT_EQ(psi[1].dimensions(), igen << 3 << 2 << 3);
      EXPECT_EQ(psi[2].dimensions(), igen << 3 << 2 << 1);
    }
    {
      MPS psi =
        MPS::random(/* size */ 3, /* physical dimension */ 2, /* bond dimension */ 3,
                    /* periodic */ true);
      EXPECT_EQ(psi.size(), 3);
      EXPECT_EQ(psi[0].dimensions(), igen << 3 << 2 << 3);
      EXPECT_EQ(psi[1].dimensions(), igen << 3 << 2 << 3);
      EXPECT_EQ(psi[2].dimensions(), igen << 3 << 2 << 3);
    }
  }

  template<class MPS>
  void test_mps_product_state(int size) {
    typename MPS::elt_t psi = MPS::elt_t::random(3);
    MPS state = product_state(size, psi);
    EXPECT_EQ(state.size(), size);
    psi = reshape(psi, 1, psi.size(), 1);
    for (int i = 0; i < size; i++) {
      EXPECT_TRUE(all_equal(state[i], psi));
    }
  }

  void test_ghz_state(int size) {
    RMPS ghz = ghz_state(size);
    RTensor psi = mps_to_vector(ghz);
    double v = 1/sqrt((double)2.0);
    EXPECT_EQ(ghz.size(), size);
    EXPECT_EQ(psi.size(), 2 << (size-1));
    for (index i = 0; i < psi.size(); i++) {
      double psi_i = ((i == 0) || (i == psi.size()-1)) ? v : 0.0;
      EXPECT_DOUBLE_EQ(psi[i], psi_i);
    }
    EXPECT_DOUBLE_EQ(norm2(psi), 1.0);
  }

  const RMPS apply_cluster_state_stabilizer(RMPS state, int site) {
    int left = site - 1;
    if (left >= 0)
      state = apply_local_operator(state, mps::Pauli_z, left);
    state = apply_local_operator(state, mps::Pauli_x, site);
    int right = site + 1;
    if (right < state.size())
      state = apply_local_operator(state, mps::Pauli_z, site);
    return state;
  }

  void test_cluster_state(int size) {
    RMPS cluster = cluster_state(size);
    RTensor psi = mps_to_vector(cluster);
    EXPECT_EQ(cluster.size(), size);
    EXPECT_EQ(psi.size(), 2 << (size-1));
    EXPECT_DOUBLE_EQ(norm2(psi), 1.0);
    for (index i = 1; i < cluster.size(); i++) {
      RTensor psi2 = mps_to_vector(apply_cluster_state_stabilizer(cluster, i));
      if (!simeq(psi, psi2)) {
        std::cout << "psi2=" << psi2 << std::endl;
        std::cout << "psi=" << psi << std::endl;
        abort();
      }
      EXPECT_CEQ(psi, psi2);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RMPS, Constructor) {
    test_mps_constructor<RMPS>();
  }

  TEST(RMPS, Random) {
    test_mps_random<RMPS>();
  }

  TEST(RMPS, ProductState) {
    test_over_integers(1,4,test_mps_product_state<CMPS>);
  }

  TEST(RMPS, GHZState) {
    test_over_integers(1,10,test_ghz_state);
  }

  TEST(RMPS, ClusterState) {
    test_over_integers(4,10,test_cluster_state);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMPS, Constructor) {
    test_mps_constructor<CMPS>();
  }

  TEST(CMPS, Random) {
    test_mps_random<CMPS>();
  }

  TEST(CMPS, ProductState) {
    test_over_integers(1,10,test_mps_product_state<CMPS>);
  }

} // namespace linalg_test
