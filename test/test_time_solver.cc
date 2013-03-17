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

namespace tensor_test {

  using namespace mps;

  //
  // Simplifying a state that does not require simplification.
  //
  template<class MPS>
  void evolve_identity(int size)
  {
    MPS psi = ghz_state(size);
    TIHamiltonian Id(size, RTensor::zeros(4,4), RTensor::eye(2));
    {
      Trotter2Solver solver(Id, 0.1);
      MPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    }
    {
      Trotter3Solver solver(Id, 0.1);
      MPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    }
    {
      ForestRuthSolver solver(Id, 0.1);
      MPS aux = psi;
      solver.one_step(&aux, 2);
      EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    }
  }

  ////////////////////////////////////////////////////////////
  // EVOLVE WITH TROTTER METHODS
  //

  TEST(TimeSolver, Identity) {
    test_over_integers(2, 10, evolve_identity<CMPS>);
  }

} // namespace tensor_test
