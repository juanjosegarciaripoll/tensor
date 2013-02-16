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
#include <mps/iTEBD/itebd_expected_slow.hpp>
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <mps/itebd.h>
#include <mps/tools.h>

namespace tensor_test {

  using namespace mps;
  using namespace tensor;

  template<class t>
  const iTEBD<t> test_state(t *H12)
  {
    t A = RTensor(igen << 2 << 3 << 2,
                  rgen << -0.95912 << -0.29373 << -0.32075 << 0.82915
                  << -0.053633 << 0.29376 << -0.29375 << -0.070285
                  << 0.82919 << 0.42038 << 0.29377 << -1.2573);
    t lA = RTensor(igen << 2, rgen << 0.91919 << 0.39382);
    t B = RTensor(igen << 2 << 3 << 2,
                  rgen << -0.05363 << 0.29376 << 0.32074 << -0.82919
                  << -0.95913 << -0.29374 << 0.29375 << -1.2573
                  << -0.82915 << -0.42036 << -0.29373 << -0.070279);
    t lB = RTensor(igen << 2, rgen << 0.91923 << 0.39373);

    if (H12) {
      *H12 = RTensor(igen << 9 << 9,
		     rgen << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0
		     << 0 << 0 << 0 << 1 << 0 << 0 << 0 << 0 << 0
		     << 0 << 0 << -1 << 0 << 1 << 0 << 0 << 0 << 0
		     << 0 << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0
		     << 0 << 0 << 1 << 0 << 0 << 0 << 1 << 0 << 0
		     << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 1 << 0
		     << 0 << 0 << 0 << 0 << 1 << 0 << -1 << 0 << 0
		     << 0 << 0 << 0 << 0 << 0 << 1 << 0 << 0 << 0
		     << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 1);
    }

    return iTEBD<t>(A, lA, B, lB);
  }

  template<class t>
  void test_small_canonical_iTEBD()
  {
    t H12;
    iTEBD<t> psi = test_state(&H12);
    iTEBD<t> psic = psi.canonical_form();

    EXPECT_TRUE(simeq(expected12(psi, H12), expected12(psic, H12), 2e-6));
    EXPECT_TRUE(simeq(expected12(psi, H12), slow_expected12(psi, H12), 2e-6));
    EXPECT_TRUE(simeq(expected12(psic, H12), slow_expected12(psi, H12), 2e-6));

    /*
    std::cout << expected12(psi, H12) << std::endl
	      << expected12(psic, H12) << std::endl
	      << slow_expected12(psic, H12) << std::endl
	      << slow_expected12(psi, H12) << std::endl;
    */

    EXPECT_TRUE(simeq(energy(psi, H12), energy(psic, H12), 2e-6));
    EXPECT_TRUE(simeq(energy(psi, H12), slow_energy(psi, H12), 2e-6));
    EXPECT_TRUE(simeq(energy(psic, H12), slow_energy(psi, H12), 2e-6));
  }

  ////////////////////////////////////////////////////////////
  /// ITEBD WITH REAL TENSORS
  ///

  TEST(RiTEBDTest, ExpectedCanonicalForm) {
    test_small_canonical_iTEBD<RTensor>();
  }

  ////////////////////////////////////////////////////////////
  /// ITEBD WITH COMPLEX TENSORS
  ///

  TEST(CiTEBDTest, ExpectedCanonicalForm) {
    test_small_canonical_iTEBD<CTensor>();
  }

}
