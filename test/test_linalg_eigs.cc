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
  using namespace linalg;

  //////////////////////////////////////////////////////////////////////
  // EIGENVALUE DECOMPOSITIONS
  //

  template<typename elt_t>
  void test_eye_eigs(int n) {
    if (n == 0) {
#ifndef NDEBUG
      ASSERT_DEATH(eigs(Tensor<elt_t>::eye(n,n), 1, LargestMagnitude), ".*");
#endif
      return;
    }
    Tensor<elt_t> Inn = Tensor<elt_t>::eye(n,n);
    Tensor<elt_t> V1 = Tensor<elt_t>::ones(n);
    for (int neig = 1; neig < std::min(n,4); neig++) {
      Tensor<elt_t> U;
      for (int type = 0; type < LargestImaginary; type++) {
        CTensor E = eigs(Inn, LargestMagnitude, neig, &U);
        EXPECT_CEQ(sqrt((double)neig), norm2(U));
        EXPECT_EQ(2, U.rank());
        EXPECT_EQ(n, U.dimension(0));
        EXPECT_EQ(neig, U.dimension(1));
        EXPECT_EQ(neig, E.size());
        EXPECT_CEQ(CTensor::ones(igen << neig), E);
      }
    }
  }

  template<typename elt_t>
  void test_eigs_permuted_diagonal(int n) {
    RTensor p = random_permutation(n, n);
    RTensor d = diag(linspace((double)1.0, n, n), 0);

    Tensor<elt_t> e1 = RTensor::zeros(igen << n);
    e1.at(0) = 1.0;
    e1 = mmult(p, e1);

    Tensor<elt_t> en = RTensor::zeros(igen << n);
    en.at(n-1) = 1.0;
    en = mmult(p, en);

    Tensor<elt_t> A = mmult(adjoint(p), mmult(d, p));
    Tensor<elt_t> U;

    CTensor E = eigs(A, SmallestMagnitude, 1, &U);
    EXPECT_EQ(2, U.rank());
    EXPECT_EQ(n, U.dimension(0));
    EXPECT_EQ(1, U.dimension(1));
    EXPECT_EQ(1, E.size());
    EXPECT_CEQ(number_one<cdouble>(), E(0));
    EXPECT_CEQ(1.0, abs(fold(e1, 0, U, 0))(0));

    E = eigs(A, LargestMagnitude, 1, &U);
    EXPECT_EQ(2, U.rank());
    EXPECT_EQ(n, U.dimension(0));
    EXPECT_EQ(1, U.dimension(1));
    EXPECT_EQ(1, E.size());
    EXPECT_TRUE(simeq(to_complex((double)n), E(0), 15*EPSILON));
    EXPECT_CEQ(1.0, abs(fold(en, 0, U, 0))(0));
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RMatrixTest, EyeEigsTest) {
    test_over_integers(0, 32, test_eye_eigs<double>);
  }

  TEST(RMatrixTest, PermutedDiagonalEigsTest) {
    test_over_integers(1, 32, test_eigs_permuted_diagonal<double>);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMatrixTest, EyeEigsTest) {
    test_over_integers(0, 32, test_eye_eigs<cdouble>);
  }

  TEST(CMatrixTest, PermutedDiagonalEigsTest) {
    test_over_integers(1, 32, test_eigs_permuted_diagonal<cdouble>);
  }

} // namespace linalg_test
