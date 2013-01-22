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
#include <tensor/tensor.h>
#include <tensor/arpack.h>

namespace tensor_test {

  using namespace tensor;

  //////////////////////////////////////////////////////////////////////
  // EIGENVALUE DECOMPOSITIONS
  //

  template<typename elt_t>
  void test_eye_eigs(int n) {
    if (n == 0) {
#ifndef NDEBUG
      ASSERT_DEATH(linalg::eig_sym(Tensor<elt_t>::eye(n,n)), ".*");
#endif
      return;
    }
    Tensor<elt_t> Inn = Tensor<elt_t>::eye(n,n);
    Tensor<elt_t> V1 = Tensor<elt_t>::ones(n);
    for (int neig = 1; i < 4; i++) {
      CTensor U, E = linalg::eigs(Inn, RArpack::Default, neig, &U);
      EXPECT_EQ(1.0, norm(U));
      EXPECT_EQ(2, U.rank());
      EXPECT_EQ(n, U.dimension(0));
      EXPECT_EQ(i, U.dimension(1));
      EXPECT_EQ(neig, E.size());
      EXPECT_EQ(CTensor::ones(neig), E);
    }
  }

  template<typename elt_t>
  void test_random_eig_sym(int n) {
    if (n == 0) {
#ifndef NDEBUG
      ASSERT_DEATH(linalg::eig_sym(Tensor<elt_t>::eye(n,n)), ".*");
#endif
      return;
    }
    for (int times = 10; times; --times) {
      Tensor<elt_t> R, A = Tensor<elt_t>::random(n,n);
      A = mmult(A, adjoint(A)) / norm0(A);
      RTensor s = linalg::eig_sym(A, &R);
      Tensor<elt_t> L = adjoint(R);
      RTensor dS = diag(s);
      EXPECT_TRUE(norm0(abs(s) - s) < 1e-13);
      EXPECT_TRUE(unitaryp(R, 1e-10));
      EXPECT_TRUE(approx_eq(mmult(A, R), mmult(R, dS), 1e-12));
      EXPECT_TRUE(approx_eq(mmult(L, A), mmult(dS, L), 1e-12));
      EXPECT_TRUE(approx_eq(A, mmult(R, mmult(dS, L)), 1e-12));
    }
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RMatrixTest, EyeEigsTest) {
    test_over_integers(0, 32, test_eye_eigs_sym<double>);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMatrixTest, EyeEigsTest) {
    test_over_integers(0, 32, test_eye_eigs_sym<cdouble>);
  }

} // namespace linalg_test
