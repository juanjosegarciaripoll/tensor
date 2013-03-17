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
#include <mps/mps_algorithms.h>

namespace tensor_test {

  using namespace mps;

  //
  // Canonical form of a state that does not require simplification.
  //
  template<class MPS>
  void test_canonical_form(int size)
  {
    MPS psi = cluster_state(size);
    {
      MPS aux = canonical_form(psi, -1);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
    {
      MPS aux = canonical_form(psi, +1);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
  }

  //
  // Normal form of a state that does not require simplification.
  //
  template<class MPS>
  void test_normal_form(int size)
  {
    MPS psi = cluster_state(size);
    {
      MPS aux = normal_form(psi, -1);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
    {
      MPS aux = normal_form(psi, +1);
      EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
      EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
    }
  }

  ////////////////////////////////////////////////////////////
  // SIMPLIFY RMPS
  //

  TEST(RMPSCanonical, SimpleStates) {
    test_over_integers(2, 10, test_canonical_form<RMPS>);
  }

  TEST(RMPSNormal, SimpleStates) {
    test_over_integers(2, 10, test_normal_form<RMPS>);
  }

  ////////////////////////////////////////////////////////////
  // SIMPLIFY CMPS
  //

  TEST(CMPSCanonical, SimpleStates) {
    test_over_integers(2, 10, test_canonical_form<CMPS>);
  }

  TEST(CMPSNormal, SimpleStates) {
    test_over_integers(2, 10, test_normal_form<CMPS>);
  }


} // namespace tensor_test
