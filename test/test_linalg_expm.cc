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

  static const RTensor sx(igen << 2 << 2, rgen << 0.0 << 1.0 << 1.0 << 0.0);
  static const RTensor sz(igen << 2 << 2, rgen << 1.0 << 0.0 << 0.0 << -1.0);
  static const RTensor id = RTensor::eye(2);

  //////////////////////////////////////////////////////////////////////
  // MATRIX EXPONENTIALS
  //

  template<typename elt_t>
  const Tensor<elt_t>
  expm_diag(int n, Tensor<elt_t> *pexponent)
  {
    /*
     * Create a random diagonal matrix, and exponentiate it
     * in a very precise way (the exponential of a diagonal is
     * a diagonal of exponentials)
     */
    Tensor<elt_t> d = Tensor<elt_t>::random(n);
    if (pexponent) {
      *pexponent = diag(d);
    }
    return diag(exp(d));
  }

  RTensor
  pauli_exponential(double theta, double phi, RTensor *pexponent)
  {
      RTensor A = cos(theta) * sx + sin(theta) * sz;
      if (pexponent) {
        *pexponent = phi * A;
      }
      return cosh(phi) * id + sinh(phi) * A;
  }

  CTensor
  pauli_exponential(double theta, double phi, CTensor *pexponent)
  {
      CTensor A = cos(theta) * sx + sin(theta) * sz;
      if (pexponent) {
        *pexponent = to_complex(0.0,phi) * A;
      }
      return cos(phi) * id + to_complex(0.0,sin(phi)) * A;
  }

  template<typename elt_t>
  void test_expm_diag(int n) {
    if (n == 0) {
#ifndef NDEBUG
      ASSERT_DEATH(linalg::expm(Tensor<elt_t>(0)),".*");
#endif
      return;
    }
    Tensor<elt_t> exponent, exponential = expm_diag(n, &exponent);
    EXPECT_TRUE(approx_eq(linalg::expm(exponent), exponential));
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RMatrixTest, ExpmDiagTest) {
    test_over_integers(0, 32, test_expm_diag<double>);
  }

  /*
   * We compute the exponential of a linear combination of Pauli
   * matrices, for which we have an exact formula.
   */
  TEST(RMatrixTest, ExpmPauliTest) {
    for (int i = 0; i < 30; i++) {
      double theta = rand(M_PI);
      double phi = rand(M_PI);
      RTensor fA, expfA = pauli_exponential(theta, phi, &fA);
      EXPECT_TRUE(approx_eq(linalg::expm(fA), expfA, 1e-13));
    }
  }

  /*
   * exp(A + B) = exp(A) exp(B) if A and B commute
   */
  TEST(RMatrixTest, ExpmKronecker) {
    for (int n = 1; n < 4; n++) {
      for (int i = 0; i < 30; i++) {
        double theta = rand(M_PI);
        double phi = rand(M_PI);
        RTensor fA, expfA = pauli_exponential(theta, phi, &fA);
        RTensor B, expB = expm_diag(n, &B);
        RTensor in = RTensor::eye(n,n);
        RTensor C = kron(fA, in) + kron(id, B);
        RTensor expC = kron(expfA, expB);
        EXPECT_TRUE(approx_eq(linalg::expm(C), expC, 1e-13));
      }
    }
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMatrixTest, ExpmDiagTest) {
    test_over_integers(0, 32, test_expm_diag<cdouble>);
  }

  /*
   * We compute the exponential of a linear combination of Pauli
   * matrices, for which we have an exact formula.
   */
  TEST(CMatrixTest, ExpmPauliTest) {
    for (int i = 0; i < 30; i++) {
      double theta = rand(M_PI);
      double phi = rand(M_PI);
      CTensor fA, expfA = pauli_exponential(theta, phi, &fA);
      EXPECT_TRUE(approx_eq(linalg::expm(fA), expfA, 1e-13));
    }
  }

  /*
   * exp(A + B) = exp(A) exp(B) if A and B commute
   */
  TEST(CMatrixTest, ExpmKronecker) {
    for (int n = 1; n < 4; n++) {
      for (int i = 0; i < 30; i++) {
        double theta = rand(M_PI);
        double phi = rand(M_PI);
        CTensor fA, expfA = pauli_exponential(theta, phi, &fA);
        CTensor B, expB = expm_diag(n, &B);
        CTensor in = CTensor::eye(n,n);
        CTensor C = kron(fA, in) + kron(CTensor(id), B);
        CTensor expC = kron(expfA, expB);
        EXPECT_TRUE(approx_eq(linalg::expm(C), expC, 1e-13));
      }
    }
  }

} // namespace tensor_test
