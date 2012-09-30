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

#include <cmath>
#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>
#include <tensor/linalg.h>

namespace tensor_test {

  using namespace tensor;

  //////////////////////////////////////////////////////////////////////
  // MATRIX EXPONENTIALS
  //

  template<typename elt_t>
  void test_expm_diag(int n) {
#ifndef NDEBUG
    if (n == 0) {
      ASSERT_DEATH(linalg::expm(Tensor<elt_t>(0)),".*");
      return;
    }
#endif
    Tensor<elt_t> d = Tensor<elt_t>::random(n);
    Tensor<elt_t> A = diag(d);
    Tensor<elt_t> simple = diag(exp(d));
    Tensor<elt_t> full = linalg::expm(A);
    EXPECT_TRUE(approx_eq(simple, full, 1e-12));
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RMatrixTest, ExpmDiagTest) {
    test_over_integers(0, 32, test_expm_diag<double>);
  }

  TEST(RMatrixTest, ExpmPauliTest) {
    RTensor sx(igen << 2 << 2, rgen << 0.0 << 1.0 << 1.0 << 0.0);
    RTensor sz(igen << 2 << 2, rgen << 1.0 << 0.0 << 0.0 << -1.0);
    RTensor id = RTensor::eye(2,2);
    for (int i = 0; i < 30; i++) {
      double theta = rand(M_PI);
      double phi = rand(M_PI);
      RTensor A = cos(theta) * sx + sin(theta) * sz;
      RTensor fA = phi * A;
      RTensor expfA = cosh(phi) * id + sinh(phi) * A;
      RTensor expmfA = linalg::expm(fA);
      EXPECT_TRUE(approx_eq(expmfA, expfA, 1e-13));
    }
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMatrixTest, ExpmDiagTest) {
    test_over_integers(0, 32, test_expm_diag<cdouble>);
  }

  TEST(CMatrixTest, ExpmPauliTest) {
    CTensor sx(igen << 2 << 2, cgen << 0.0 << 1.0 << 1.0 << 0.0);
    CTensor sz(igen << 2 << 2, cgen << 1.0 << 0.0 << 0.0 << -1.0);
    CTensor id = CTensor::eye(2,2);
    for (int i = 0; i < 30; i++) {
      double theta = rand(M_PI);
      double phi = rand(M_PI);
      CTensor A = cos(theta) * sx + sin(theta) * sz;
      CTensor fA = to_complex(0.0,phi) * A;
      CTensor expfA = cos(phi) * id + to_complex(0.0,sin(phi)) * A;
      CTensor expmfA = linalg::expm(fA);
      EXPECT_TRUE(approx_eq(expmfA, expfA, 1e-13));
    }
  }

} // namespace tensor_test
