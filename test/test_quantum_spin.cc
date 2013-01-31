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

#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include "loops.h"
#include <tensor/gen.h>
#include <mps/quantum.h>

namespace tensor_test {

  using namespace tensor;

  //////////////////////////////////////////////////////////////////////
  // SPIN OPERATORS
  //

  TEST(RMPSTest, SpinSafetyTest) {
#ifndef NDEBUG
    CTensor s[3];
    ASSERT_DEATH(mps::spin_operators(0, s, s+1, s+2), ".*");
    ASSERT_DEATH(mps::spin_operators(-1, s, s+1, s+2), ".*");
    ASSERT_DEATH(mps::spin_operators(0.1, s, s+1, s+2), ".*");
#endif
  }

  TEST(RMPSTest, SpinCommutations) {
    for (double s = 0.5; s <= 1.2; s += 0.5) {
      CTensor S[3];
      mps::spin_operators(s, S, S+1, S+2);
      for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;
        EXPECT_TRUE(approx_eq(mmult(S[i], S[j])-mmult(S[j],S[i]),
                              to_complex(0.0,1.0) * S[k]));
      }
    }
  }

  TEST(RMPSTest, SpinHalf) {
    CTensor S[3];
    mps::spin_operators(0.5, S, S+1, S+2);
    EXPECT_TRUE(all_equal(to_complex(mps::Pauli_x/2.0), S[0]));
    EXPECT_TRUE(all_equal(mps::Pauli_y/2.0, S[1]));
    EXPECT_TRUE(all_equal(to_complex(mps::Pauli_z/2.0), S[2]));
  }

}
