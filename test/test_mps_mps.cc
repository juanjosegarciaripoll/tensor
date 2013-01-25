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

namespace tensor_test {

  using namespace tensor;
  using namespace mps;

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

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RMPS, RMPSConstructor) {
    test_mps_constructor<RMPS>();
  }

  TEST(RMPS, RMPSRandom) {
    test_mps_random<RMPS>();
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMPS, CMPSConstructor) {
    test_mps_constructor<CMPS>();
  }

  TEST(CMPS, CMPSRandom) {
    test_mps_random<CMPS>();
  }

} // namespace linalg_test
